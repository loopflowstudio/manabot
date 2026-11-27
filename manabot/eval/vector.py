"""Asynchronous vectorized match runner over one native `RecordedVectorEnv`.

`play_pairings` steps up to `slots` paired-deal legs concurrently on the v11
CSR `ObservationBatch` -> `Prepared` path. In each round, every active slot
needing the same checkpoint is gathered into one pre-collated `Prepared` input
for one awaited forward; heuristic/code seats act in the same round from
detached native observations. Finished or natively-failed legs free their slot
and backfill immediately via seeded `SlotLoad.fresh(..., seed=)`. Once all legs
of a match finish, the match is replayed, card-evidenced, retained to disk and
yielded as `(match, Result)`.

This module is Torch-free (`tests/test_package_boundaries.py`).
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field, replace
import inspect
import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any, Literal, Protocol
from uuid import uuid4

from etude.eval.records import Match
from etude.eval.result import Result
import numpy as np

from manabot.env.failures import (
    ActionCutoffCounts,
    ActionCutoffSample,
    RecordedVectorEnv,
)
from manabot.env.match import Match as EnvMatch
from manabot.env.observation import ObservationSpace
from manabot.eval.agents import Heuristic, Sampler
from manabot.eval.behaviour import Play
from manabot.eval.evaluation import EvaluationResult
from manabot.eval.match import (
    CellPlan,
    build_cell_plan,
    finalize_settled_cell,
)
from manabot.eval.pairing import (
    AdmittedPairing,
    admit_pairing,
    pairing_result,
)
from manabot.eval.results import EvaluationRow
from manabot.infra.config import MatchConfig
from manabot.infra.files import atomic_json
from manabot.infra.identity import canonical_sha256
from manabot.infra.provenance import RECORDED_ASSET_MANIFEST_HASH
from manabot.policy.admission import Spec, admit_native
from manabot.policy.prepare import Prepared, prepare
from manabot.policy.spans import ObservationSelection
from manabot.sim.agent import GamePlayer
from manabot.sim.evaluation_records import (
    bind_record,
    decision_annotation,
    set_execution_limits,
)
from manabot.sim.frames import build_command, build_viewer_frame
from manabot.sim.matchup import AuthoredMatchup, DeckMatchup
from manabot.sim.native_game import (
    arena_source_sha256,
    game_runtime_identities,
    resolve_game_match,
)
from manabot.sim.performance import EvalPerformance
from manabot.sim.registration import (
    EvalPrecision,
    PlayerRegistration,
    require_open_decklists,
)
from manabot.sim.supervision import failure_record, settle_record
import managym
from managym.experience import UNVERSIONED_CONTENT_HASH
from managym.recording import GameEnd


@dataclass(frozen=True)
class PairingSpec:
    """Arena parameters for one paired-deal match."""

    matchup: AuthoredMatchup | DeckMatchup
    deal: int
    matchup_mode: Literal["cross", "all"]
    game_seconds: float
    arena: str
    precision: EvalPrecision = "float32"
    max_commands: int = 10_000
    bootstrap_seed: int = 6510
    keep_game_logs: bool = False


class MatchSource(Protocol):
    """Abstract asynchronous match feed matching Etude's `Matches` contract."""

    async def take(self, limit: int) -> Sequence[tuple[Match, Sequence[object]]]: ...


class LocalMatches:
    """In-memory `MatchSource` over a fixed sequence of `(match, agents)` pairs."""

    def __init__(self, items: Sequence[tuple[Match, Sequence[object]]]) -> None:
        self._queue: deque[tuple[Match, Sequence[object]]] = deque(items)

    async def take(self, limit: int) -> list[tuple[Match, Sequence[object]]]:
        if limit < 1:
            raise ValueError("take limit must be positive")
        taken: list[tuple[Match, Sequence[object]]] = []
        while self._queue and len(taken) < limit:
            taken.append(self._queue.popleft())
        return taken


@dataclass(frozen=True)
class RawObservationEnv:
    """Minimal read-only adapter supplying `last_raw_obs` to code players."""

    last_raw_obs: managym.Observation


@dataclass
class PendingLeg:
    active_match: ActiveMatch
    leg: int
    game: dict[str, Any]
    registrations: list[PlayerRegistration]
    setup: managym.GameSetup
    seed: int
    evidence_path: Path


@dataclass(eq=False)
class ActiveMatch:
    match: Match
    spec: PairingSpec
    admitted: AdmittedPairing
    plan: CellPlan
    space: ObservationSpace
    temp_dir: TemporaryDirectory[str]
    out_dir: Path
    cards_dir: Path
    started: float
    finished_games: list[dict[str, Any]] = field(default_factory=list)
    closed: bool = False

    def cleanup(self) -> None:
        if not self.closed:
            self.closed = True
            self.temp_dir.cleanup()


@dataclass
class SlotState:
    slot: int
    leg: PendingLeg
    started: float
    revision: int = 0
    decision_counts: dict[str, int] = field(default_factory=dict)
    policy_counts: dict[str, ActionCutoffCounts] = field(default_factory=dict)
    policy_samples: list[ActionCutoffSample] = field(default_factory=list)


def admit_active_match(
    match: Match,
    agents: Sequence[object],
    spec: PairingSpec,
    *,
    arena_source: str,
    output_dir: Path | None,
) -> tuple[ActiveMatch, list[PendingLeg]]:
    """Admit players, resolve per-seat decision settings and construct leg work."""
    admitted = admit_pairing(
        match,
        agents,
        matchup=spec.matchup,
        deal=spec.deal,
        matchup_mode=spec.matchup_mode,
        game_seconds=spec.game_seconds,
        arena=spec.arena,
        precision=spec.precision,
        max_commands=spec.max_commands,
        bootstrap_seed=spec.bootstrap_seed,
        output_dir=output_dir,
    )
    player_spaces = {name: player.space for name, player in admitted.agents.items()}
    spaces = [space for space in player_spaces.values() if space is not None]
    if spaces and any(space.shapes != spaces[0].shapes for space in spaces):
        raise ValueError("players have different observation ABIs")
    space = spaces[0] if spaces else ObservationSpace()
    if not space.one_token:
        raise ValueError("vector evaluation requires policy input v11")

    journal_root = "/dev/shm" if Path("/dev/shm").is_dir() else None
    temp_dir = TemporaryDirectory(dir=journal_root)
    out_dir = Path(temp_dir.name) / "deal"
    cards_dir = Path(temp_dir.name) / "cards"
    try:
        plan = build_cell_plan(
            key=admitted.request.key,
            player_a=admitted.request.candidate,
            player_b=admitted.request.opponent,
            deal_seeds=admitted.request.deal_seeds,
            out_dir=out_dir,
            game_seconds=admitted.request.game_seconds,
            max_commands=admitted.request.max_commands,
            matchup_mode=admitted.request.matchup_mode,
            authored_matchup=admitted.request.authored_matchup,
        )
        active_match = ActiveMatch(
            match=match,
            spec=spec,
            admitted=admitted,
            plan=plan,
            space=space,
            temp_dir=temp_dir,
            out_dir=out_dir,
            cards_dir=cards_dir,
            started=time.perf_counter(),
        )
        legs: list[PendingLeg] = []
        for item in plan.work:
            game = item.game
            registrations = item.registrations
            match_config = MatchConfig.model_validate(game["match_hypers"])
            require_open_decklists(match_config.decklists, registrations)
            for registration in registrations:
                admitted.agents[registration.player_id].admit(match_config)
            resolved_match = resolve_game_match(
                game, match_config, registrations, player_spaces
            )
            game["observation_hypers"] = space.config.model_dump()
            set_execution_limits(
                game, registrations, spec.max_commands, spec.game_seconds
            )
            game["runtime_identities"] = game_runtime_identities(
                game["deal_seed"], resolved_match, space, arena_source
            )
            legs.append(
                PendingLeg(
                    active_match=active_match,
                    leg=int(game["leg"]),
                    game=game,
                    registrations=registrations,
                    setup=EnvMatch(resolved_match).to_rust(),
                    seed=int(game["deal_seed"]),
                    evidence_path=item.evidence_path,
                )
            )
        return active_match, legs
    except BaseException:
        temp_dir.cleanup()
        raise


def prepare_checkpoint_batch(
    obs_rows: Sequence[tuple[managym.ObservationBatch, int]],
    slots: Sequence[int],
    spec: Spec,
    *,
    blank_decision_memory: bool,
) -> Prepared[np.ndarray]:
    """Gather active slots' CSR rows and prepare one host `Prepared` batch."""
    coords = tuple(obs_rows[slot] for slot in slots)
    batch = ObservationSelection(spec.space, coords).batch
    if blank_decision_memory:
        batch = batch.blank_decision_memory()
    spans = admit_native(batch, spec, bucketed=False)
    model = spec.model
    return replace(
        prepare(
            spans,
            spec.tables,
            binding=spec.binding,
            memory_summaries=model.decision_memory_summaries
            if model.decision_memory != "none"
            else 0,
            memory_fold=model.decision_memory == "gru",
        ),
        programs=None,
    )


async def invoke_forward(
    forward: Callable[[list[object]], object],
    prepared: Prepared[np.ndarray],
) -> tuple[np.ndarray, float]:
    """Call an async or sync forward with one pre-collated `Prepared` input."""
    started = time.perf_counter()
    output = forward([prepared])
    if inspect.isawaitable(output):
        output = await output
    elapsed = time.perf_counter() - started
    results = list(output)  # type: ignore[arg-type]
    if len(results) != 1:
        raise ValueError(
            f"forward returned {len(results)} outputs for 1 pre-collated input"
        )
    return np.asarray(results[0], dtype=np.float32), elapsed


def sample_action(
    row_logits: np.ndarray,
    legal_row: np.ndarray,
    *,
    deterministic: bool,
    seed: int,
) -> int:
    slots = np.flatnonzero(legal_row)
    if deterministic:
        return int(slots[row_logits[slots].argmax()])
    weights = np.exp(row_logits[slots].astype(np.float64) - row_logits[slots].max())
    return int(np.random.default_rng(seed).choice(slots, p=weights / weights.sum()))


def choose_code_action(
    player: GamePlayer,
    raw: managym.Observation,
    *,
    seed: int,
    space: ObservationSpace,
) -> tuple[int, float]:
    started = time.perf_counter()
    if isinstance(player, Heuristic):
        count = len(raw.action_space.actions)
        if space.config.action_overflow == "first-legal":
            count = min(count, space.config.max_actions)
        obs = {"actions_valid": np.ones(count, dtype=np.float32)}
        action = int(player.factory(seed).act(RawObservationEnv(raw), obs))  # type: ignore[arg-type]
    else:
        raise TypeError(
            f"vector evaluation supports Sampler and Heuristic players, got {type(player).__name__}"
        )
    return action, time.perf_counter() - started


def record_slot_policy_cutoff(
    slot_state: SlotState,
    count: int,
    kind: str,
    capacity: int,
    vector: RecordedVectorEnv,
) -> None:
    if kind not in slot_state.policy_counts:
        slot_state.policy_counts[kind] = ActionCutoffCounts()
    counts = slot_state.policy_counts[kind]
    counts.decisions += 1
    retained = managym.policy_action_count(count, capacity, "first-legal")
    if retained == count:
        return
    counts.overflows += 1
    counts.dropped_actions += count - retained
    if len(slot_state.policy_samples) < 4:
        slot_state.policy_samples.append(
            ActionCutoffSample(
                original_count=count,
                retained_count=retained,
                dropped_count=count - retained,
                decision_kind=kind,
                game=vector.game_inputs(slot_state.slot).model_copy(deep=True),
            )
        )


def policy_cutoff_report(
    slot_state: SlotState, vector: RecordedVectorEnv
) -> dict[str, object]:
    return {
        "schema": "policy-action-cutoff-v1",
        "policy": "first-legal",
        "counts_by_kind": {
            kind: counts.model_dump()
            for kind, counts in slot_state.policy_counts.items()
        },
        "samples": [
            sample.model_dump(mode="json") for sample in slot_state.policy_samples
        ],
        "identities": vector.identities,
        "context": {},
        "seconds": time.monotonic() - slot_state.started,
    }


def settle_leg(
    slot_state: SlotState,
    record: managym.recording.GameRecord,
    vector: RecordedVectorEnv,
) -> dict[str, Any]:
    """Bind the slot's canonical record, settle audit fields and retain failures."""
    leg = slot_state.leg
    game = leg.game
    game["execution_seconds"] = time.monotonic() - slot_state.started
    if leg.active_match.space.config.action_overflow == "first-legal":
        game["policy_action_cutoff"] = policy_cutoff_report(slot_state, vector)
    game["worker_performance"] = EvalPerformance(
        str(game.get("match_id", "game")),
        bool(game.get("instrumentation", True)),
    ).snapshot()
    bind_record(game, record)
    keep = leg.active_match.spec.keep_game_logs
    settle_record(game, leg.evidence_path, keep_completed=keep)
    if game["failure"] is None and not keep:
        leg.evidence_path.unlink(missing_ok=True)
    return game


def finalize_match(active_match: ActiveMatch) -> Result:
    """Replay, verify, retain diagnostics and assemble the match's Etude `Result`."""
    admitted = active_match.admitted
    out = active_match.out_dir
    cards = active_match.cards_dir
    out.mkdir(parents=True, exist_ok=True)
    result = EvaluationResult(
        request=admitted.request,
        evaluation_id=uuid4().hex,
        card_evidence_dir=str(cards),
    )
    atomic_json(out / "evaluation.json", result.model_dump(mode="json"))
    try:
        try:
            rows, trace, replay = finalize_settled_cell(
                active_match.finished_games,
                plan=active_match.plan,
                key=admitted.request.key,
                player_a=admitted.request.candidate,
                player_b=admitted.request.opponent,
                deal_seeds=admitted.request.deal_seeds,
                out_dir=out,
                game_seconds=admitted.request.game_seconds,
                card_evidence_dir=cards,
                in_process_replay=True,
                reducers={"play": Play()},
            )
            for name, payload in (("rows", rows), ("trace", trace), ("replay", replay)):
                atomic_json(out / f"{name}.json", payload)
            result.rows = [EvaluationRow.model_validate(row) for row in rows]
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        finally:
            result.evaluation_seconds = time.perf_counter() - active_match.started
            try:
                result.validate_cohort()
            except ValueError as exc:
                result.error = f"{type(exc).__name__}: {exc}"
            atomic_json(out / "evaluation.json", result.model_dump(mode="json"))
        return pairing_result(
            result,
            out=out,
            cards=cards,
            matchup=active_match.spec.matchup,
            legs=admitted.legs,
            participants=admitted.participants,
        )
    finally:
        if out.is_dir():
            shutil.copytree(out, admitted.retained, dirs_exist_ok=True)
        active_match.cleanup()


def open_vector(
    num_envs: int, space: ObservationSpace
) -> tuple[RecordedVectorEnv, dict[str, np.ndarray]]:
    vector = RecordedVectorEnv(
        num_envs=num_envs,
        seed=0,
        skip_trivial=True,
        player_policies=("agent", "agent"),
        decision_memory=space.config.max_decision_records > 0,
        decision_memory_include_own=space.config.decision_memory_include_own,
        record_fraction=1.0,
        record_in_memory=True,
        context={"source_kind": "evaluation"},
    )
    state_buffers = {
        "rewards": np.zeros(num_envs, dtype=np.float64),
        "terminated": np.zeros(num_envs, dtype=np.uint8),
        "truncated": np.zeros(num_envs, dtype=np.uint8),
    }
    vector.set_observation_layout(
        state_buffers,
        space.config.action_overflow,
        layout=space.layout,
    )
    return vector, state_buffers


async def play_pairings(
    matches: MatchSource,
    spec_for: Callable[[Match], PairingSpec],
    *,
    slots: int = 64,
    output_dir: Path | None = None,
) -> AsyncIterator[tuple[Match, Result | Exception]]:
    """Step paired-deal legs across one `RecordedVectorEnv` and yield settled matches.

    Every active slot needing the same checkpoint in a round is collated into
    one `Prepared` batch for one awaited forward. Finished or natively-failed
    slots backfill immediately so no slot sits idle while legs are pending.
    Admission or per-match finalization errors yield `(match, exc)` without
    failing other held matches; policy forward exceptions escape after closing
    the native vector.
    """
    if slots < 1:
        raise ValueError("slots must be at least 1")

    arena_source = arena_source_sha256()
    pending_legs: deque[PendingLeg] = deque()
    deferred_matches: deque[tuple[ActiveMatch, list[PendingLeg]]] = deque()
    ready_errors: deque[tuple[Match, Exception]] = deque()
    active_matches: set[ActiveMatch] = set()
    current_space: ObservationSpace | None = None
    source_exhausted = False
    legs_per_match_hint = 4

    vector: RecordedVectorEnv | None = None
    state_buffers: dict[str, np.ndarray] | None = None
    slot_states: list[SlotState | None] = []

    async def fetch_legs(needed_legs: int, *, drain_if_empty: bool) -> None:
        nonlocal current_space, source_exhausted, legs_per_match_hint
        while not source_exhausted and len(pending_legs) < needed_legs:
            missing = needed_legs - len(pending_legs)
            request_count = max(1, math.ceil(missing / legs_per_match_hint))
            batch = await matches.take(request_count)
            if not batch:
                source_exhausted = True
                break
            for match, agents in batch:
                try:
                    spec = spec_for(match)
                    active_match, legs = admit_active_match(
                        match,
                        agents,
                        spec,
                        arena_source=arena_source,
                        output_dir=output_dir,
                    )
                except Exception as exc:
                    ready_errors.append((match, exc))
                    continue
                legs_per_match_hint = max(1, len(legs))
                active_matches.add(active_match)
                if current_space is None:
                    current_space = active_match.space
                    pending_legs.extend(legs)
                elif active_match.space.config == current_space.config:
                    pending_legs.extend(legs)
                else:
                    deferred_matches.append((active_match, legs))
            if not drain_if_empty:
                break
            if len(batch) < request_count and (pending_legs or deferred_matches):
                break

    def complete_leg(
        slot: int,
        slot_state: SlotState,
        record: managym.recording.GameRecord,
        completed_matches: list[ActiveMatch],
    ) -> None:
        assert vector is not None
        game = settle_leg(slot_state, record, vector)
        active_match = slot_state.leg.active_match
        active_match.finished_games.append(game)
        slot_states[slot] = None
        if len(active_match.finished_games) == active_match.admitted.legs:
            completed_matches.append(active_match)

    def settle_immediate_terminals(
        slots_to_check: Sequence[int],
        completed_matches: list[ActiveMatch],
    ) -> None:
        assert vector is not None and state_buffers is not None
        actors = vector.current_agent_indices()
        infos: list[dict[str, Any]] | None = None
        digests: list[str | None] | None = None
        for slot in slots_to_check:
            slot_state = slot_states[slot]
            if slot_state is None:
                continue
            if actors[slot] >= 0 and not state_buffers["terminated"][slot]:
                continue
            if infos is None:
                infos = vector.get_last_info()
                digests = vector.state_digests(
                    [i in slots_to_check for i in range(len(slot_states))]
                )
            info = infos[slot] if slot < len(infos) else {}
            winner = info.get("winner_index")
            slot_state.leg.game.update(
                winner=winner,
                terminated=True,
                termination_reason="draw" if winner is None else "terminal",
            )
            record = vector.slot_record(slot)
            record.execution_outcome = "last_command_succeeded"
            record.games[0].end = GameEnd(
                status="complete",
                winner=winner,
                digest=digests[slot] if digests is not None else None,
            )
            complete_leg(slot, slot_state, record, completed_matches)

    async def backfill_free_slots(completed_matches: list[ActiveMatch]) -> None:
        assert vector is not None
        while True:
            free = [i for i, st in enumerate(slot_states) if st is None]
            if not free:
                return
            if len(pending_legs) < len(free) and not source_exhausted:
                await fetch_legs(len(free), drain_if_empty=False)
            loads: list[managym.SlotLoad] = []
            loaded_slots: list[int] = []
            now = time.monotonic()
            for slot in free:
                if not pending_legs:
                    break
                leg = pending_legs.popleft()
                loads.append(managym.SlotLoad.fresh(slot, leg.setup, seed=leg.seed))
                slot_states[slot] = SlotState(
                    slot=slot,
                    leg=leg,
                    started=now,
                    decision_counts=dict.fromkeys(leg.game["seat_players"], 0),
                )
                loaded_slots.append(slot)
            if not loads:
                return
            vector.load_slots_into_buffers(loads)
            before = len(completed_matches)
            settle_immediate_terminals(loaded_slots, completed_matches)
            if len(completed_matches) == before and all(
                slot_states[slot] is not None for slot in loaded_slots
            ):
                return

    try:
        while True:
            while ready_errors:
                yield ready_errors.popleft()

            if vector is None:
                if not pending_legs and deferred_matches:
                    first_match, first_legs = deferred_matches.popleft()
                    current_space = first_match.space
                    pending_legs.extend(first_legs)
                    remaining: deque[tuple[ActiveMatch, list[PendingLeg]]] = deque()
                    while deferred_matches:
                        other_match, other_legs = deferred_matches.popleft()
                        if other_match.space.config == current_space.config:
                            pending_legs.extend(other_legs)
                        else:
                            remaining.append((other_match, other_legs))
                    deferred_matches = remaining

                if len(pending_legs) < slots and not source_exhausted:
                    await fetch_legs(slots, drain_if_empty=True)
                    while ready_errors:
                        yield ready_errors.popleft()

                if not pending_legs:
                    if deferred_matches:
                        continue
                    if source_exhausted:
                        break
                    continue

                assert current_space is not None
                num_envs = min(slots, len(pending_legs)) if source_exhausted else slots
                vector, state_buffers = open_vector(num_envs, current_space)
                slot_states = [None] * num_envs
                initial_setups: list[managym.GameSetup] = []
                initial_seeds: list[int] = []
                now = time.monotonic()
                fallback_setup: managym.GameSetup | None = None
                for slot in range(num_envs):
                    if pending_legs:
                        leg = pending_legs.popleft()
                        fallback_setup = leg.setup
                        initial_setups.append(leg.setup)
                        initial_seeds.append(leg.seed)
                        slot_states[slot] = SlotState(
                            slot=slot,
                            leg=leg,
                            started=now,
                            decision_counts=dict.fromkeys(leg.game["seat_players"], 0),
                        )
                    else:
                        assert fallback_setup is not None
                        initial_setups.append(fallback_setup)
                        initial_seeds.append(0)
                vector.reset_each_into_buffers(initial_setups, seeds=initial_seeds)
                completed_matches: list[ActiveMatch] = []
                settle_immediate_terminals(
                    [i for i, st in enumerate(slot_states) if st is not None],
                    completed_matches,
                )
                if any(st is None for st in slot_states):
                    await backfill_free_slots(completed_matches)
                for done_match in completed_matches:
                    active_matches.discard(done_match)
                    try:
                        yield done_match.match, finalize_match(done_match)
                    except Exception as exc:
                        yield done_match.match, exc
                while ready_errors:
                    yield ready_errors.popleft()
                continue

            assert state_buffers is not None and current_space is not None

            # Check per-game command caps and wall-clock timeouts before stepping.
            now = time.monotonic()
            limit_expired: list[tuple[int, str, str]] = []
            for slot, slot_state in enumerate(slot_states):
                if slot_state is None:
                    continue
                spec = slot_state.leg.active_match.spec
                if slot_state.revision >= spec.max_commands:
                    limit_expired.append(
                        (slot, "command_cap", "game exceeded its Command cap")
                    )
                elif now - slot_state.started >= spec.game_seconds:
                    limit_expired.append(
                        (slot, "timeout", "game wall-clock budget exhausted")
                    )

            if limit_expired:
                expired_set = {slot for slot, _, _ in limit_expired}
                digests = vector.state_digests(
                    [i in expired_set for i in range(len(slot_states))]
                )
                completed_matches = []
                for slot, reason, message in limit_expired:
                    slot_state = slot_states[slot]
                    assert slot_state is not None
                    failure_record(slot_state.leg.game, reason, message, None)
                    record = vector.slot_record(slot)
                    record.execution_outcome = "last_command_succeeded"
                    record.games[0].end = GameEnd(
                        status="interrupted",
                        reason=reason,
                        digest=digests[slot],
                    )
                    complete_leg(slot, slot_state, record, completed_matches)
                await backfill_free_slots(completed_matches)
                for done_match in completed_matches:
                    active_matches.discard(done_match)
                    try:
                        yield done_match.match, finalize_match(done_match)
                    except Exception as exc:
                        yield done_match.match, exc
                while ready_errors:
                    yield ready_errors.popleft()

            active_slots = [i for i, st in enumerate(slot_states) if st is not None]
            if not active_slots:
                vector.close()
                vector = None
                state_buffers = None
                slot_states = []
                current_space = None
                continue

            num_envs = len(slot_states)
            active_mask = [st is not None for st in slot_states]
            actors = vector.current_agent_indices()
            raw_observations = vector.decision_observations(active_mask)

            frames: dict[int, dict[str, Any]] = {}
            checkpoint_groups: dict[
                tuple[int, str, bool],
                tuple[
                    Callable[[list[object]], object],
                    Spec,
                    bool,
                    list[tuple[int, Sampler, int]],
                ],
            ] = {}
            code_slots: list[tuple[int, GamePlayer, int]] = []
            slot_meta: dict[int, tuple[int, str, int, int]] = {}

            for slot in active_slots:
                slot_state = slot_states[slot]
                assert slot_state is not None
                raw = raw_observations[slot]
                assert raw is not None
                actor = actors[slot]
                if actor != int(raw.agent.player_index):
                    raise RuntimeError(
                        f"vector actor {actor} differs from observation actor {raw.agent.player_index}"
                    )
                leg = slot_state.leg
                player_id = leg.game["seat_players"][actor]
                ordinal = slot_state.decision_counts[player_id]
                policy_seed = (leg.game["player_seeds"][player_id] + ordinal) % (2**63)
                slot_meta[slot] = (actor, player_id, ordinal, policy_seed)

                frame = build_viewer_frame(
                    raw,
                    match_id=str(leg.game["match_id"]),
                    revision=slot_state.revision,
                    content_hash=UNVERSIONED_CONTENT_HASH,
                    asset_manifest_hash=RECORDED_ASSET_MANIFEST_HASH,
                )
                if frame["projection"]["opponent"].get("hand"):
                    leg.game["integrity"]["private_exposures"] += 1
                    raise RuntimeError("viewer frame exposes a private hand")
                frames[slot] = frame

                agent = leg.active_match.admitted.agents[player_id]
                if isinstance(agent, Sampler):
                    group_key = (
                        id(agent.forward),
                        agent.spec.binding,
                        agent.blank_decision_memory,
                    )
                    if group_key not in checkpoint_groups:
                        checkpoint_groups[group_key] = (
                            agent.forward,
                            agent.spec,
                            agent.blank_decision_memory,
                            [],
                        )
                    checkpoint_groups[group_key][3].append((slot, agent, policy_seed))
                else:
                    code_slots.append((slot, agent, policy_seed))

            chosen_actions: dict[int, tuple[int, float]] = {}
            if checkpoint_groups:
                obs_rows = vector.observation_rows()
                group_list = list(checkpoint_groups.values())
                prepared_batches = [
                    prepare_checkpoint_batch(
                        obs_rows,
                        [slot for slot, _, _ in members],
                        spec,
                        blank_decision_memory=blank,
                    )
                    for _, spec, blank, members in group_list
                ]
                # A forward exception escapes the generator; finally closes the vector.
                forward_outputs = await asyncio.gather(
                    *(
                        invoke_forward(forward, prepared)
                        for (forward, _, _, _), prepared in zip(
                            group_list, prepared_batches, strict=True
                        )
                    )
                )
                for (_, _, _, members), prepared, (logits, elapsed) in zip(
                    group_list, prepared_batches, forward_outputs, strict=True
                ):
                    if (
                        logits.shape != prepared.legal.shape
                        or not np.isfinite(logits[prepared.legal]).all()
                    ):
                        raise ValueError("forward returned malformed legal logits")
                    per_row_latency = elapsed / len(members)
                    for row_idx, (slot, sampler, policy_seed) in enumerate(members):
                        action = sample_action(
                            logits[row_idx],
                            prepared.legal[row_idx],
                            deterministic=sampler.deterministic,
                            seed=policy_seed,
                        )
                        chosen_actions[slot] = (action, per_row_latency)

            for slot, agent, policy_seed in code_slots:
                raw = raw_observations[slot]
                assert raw is not None
                chosen_actions[slot] = choose_code_action(
                    agent, raw, seed=policy_seed, space=current_space
                )

            step_actions = [0] * num_envs
            annotations: list[dict[str, object] | None] = [None] * num_envs
            summaries = (
                vector.current_action_summaries()
                if current_space.config.action_overflow == "first-legal"
                else None
            )
            cutoff_capacity = current_space.config.max_actions

            for slot in active_slots:
                slot_state = slot_states[slot]
                assert slot_state is not None
                leg = slot_state.leg
                frame = frames[slot]
                action, elapsed = chosen_actions[slot]
                offers = {int(offer["id"]): offer for offer in frame["offers"]}
                if action not in offers:
                    leg.game["integrity"]["illegal_actions"] += 1
                    raise RuntimeError(f"illegal offer {action}")
                actor, player_id, ordinal, policy_seed = slot_meta[slot]
                command = build_command(frame, action)
                decision: dict[str, Any] = {
                    "actor": actor,
                    "action_space_kind": frame["action_space"],
                    "frame_sha256": canonical_sha256(frame),
                    "command": command,
                    "command_sha256": canonical_sha256(command),
                    "chosen_offer": offers[action],
                    "revision": slot_state.revision,
                    "player_id": player_id,
                    "player_seed": leg.game["player_seeds"][player_id],
                    "player_decision_ordinal": ordinal,
                    "policy_rng_seed": policy_seed,
                    "latency_seconds": elapsed,
                }
                step_actions[slot] = action
                annotations[slot] = decision_annotation(decision)
                if summaries is not None:
                    count, kind = summaries[slot]
                    record_slot_policy_cutoff(
                        slot_state, count, kind, cutoff_capacity, vector
                    )

            vector.step_isolated_into_buffers(
                step_actions,
                active_mask,
                recover_capacity=True,
                annotations=annotations,
            )

            infos = vector.get_last_info()
            terminated = state_buffers["terminated"]
            truncated = state_buffers["truncated"]
            completed_matches = []

            for slot in active_slots:
                slot_state = slot_states[slot]
                assert slot_state is not None
                leg = slot_state.leg
                _, player_id, _, _ = slot_meta[slot]
                info = infos[slot] if slot < len(infos) else {}

                if info.get("game_error"):
                    error_class = str(info.get("game_error_class", "AgentError"))
                    error_msg = str(info["game_error"])
                    if error_class == "ObservationCapacity":
                        leg.game["integrity"]["truncations"] += 1
                        failure_record(
                            leg.game,
                            "truncated",
                            "environment or observation truncated",
                            None,
                        )
                    else:
                        leg.game["error_class"] = error_class
                        if info.get("game_error_trace"):
                            leg.game["failure_trace"] = str(info["game_error_trace"])
                        failure_record(
                            leg.game,
                            "crash",
                            f"{error_class}: {error_msg}",
                            player_id,
                        )
                    record = vector.slot_record(slot)
                    record.execution_outcome = "failed"
                    record.error_class = error_class
                    record.error = error_msg
                    record.traceback = (
                        str(info["game_error_trace"])
                        if info.get("game_error_trace")
                        else None
                    )
                    complete_leg(slot, slot_state, record, completed_matches)
                    continue

                slot_state.revision += 1
                slot_state.decision_counts[player_id] += 1

                if truncated[slot]:
                    leg.game["integrity"]["truncations"] += 1
                    failure_record(
                        leg.game,
                        "truncated",
                        "environment or observation truncated",
                        None,
                    )
                    record = vector.slot_record(slot)
                    record.execution_outcome = "last_command_succeeded"
                    record.games[0].end = GameEnd(
                        status="interrupted",
                        reason="truncated",
                        digest=info.get("record_terminal_digest"),
                    )
                    complete_leg(slot, slot_state, record, completed_matches)
                    continue

                if terminated[slot]:
                    winner = info.get("winner_index")
                    leg.game.update(
                        winner=winner,
                        terminated=True,
                        termination_reason="draw" if winner is None else "terminal",
                    )
                    record = vector.slot_record(slot)
                    record.execution_outcome = "last_command_succeeded"
                    complete_leg(slot, slot_state, record, completed_matches)

            if any(st is None for st in slot_states):
                await backfill_free_slots(completed_matches)

            for done_match in completed_matches:
                active_matches.discard(done_match)
                try:
                    yield done_match.match, finalize_match(done_match)
                except Exception as exc:
                    yield done_match.match, exc

            while ready_errors:
                yield ready_errors.popleft()
    finally:
        if vector is not None:
            vector.close()
        for remaining_match in list(active_matches):
            remaining_match.cleanup()
        for deferred_match, _ in deferred_matches:
            deferred_match.cleanup()
