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
from collections.abc import AsyncGenerator, Callable, Sequence
from dataclasses import dataclass, field
import math
from pathlib import Path
import time
from time import monotonic
from typing import Any, Literal, Protocol
from uuid import uuid4

from etude.eval.agents import ForwardError
from etude.eval.records import CheckpointPlayer, Match
from etude.eval.result import Result
import numpy as np

from manabot.env.failures import (
    ActionCutoffCounts,
    ActionCutoffSample,
)
from manabot.env.match import Match as EnvMatch
from manabot.env.observation import ObservationSpace
from manabot.eval.agents import (
    CheckpointInputs,
    CheckpointReader,
    Heuristic,
    Sampler,
    Seat,
    checkpoint_reader,
    game_players,
)
from manabot.eval.behaviour import Play
from manabot.eval.evaluation import CellResult, retain_attempt
from manabot.eval.match import (
    CellPlan,
    build_cell_plan,
    finalize_settled_cell,
)
from manabot.eval.pairing import (
    AdmittedPairing,
    admit_pairing,
    pairing_attempt,
    pairing_result,
)
from manabot.infra.config import MatchConfig
from manabot.infra.identity import canonical_sha256
from manabot.infra.provenance import RECORDED_ASSET_MANIFEST_HASH
from manabot.policy.spans import BatchRow, ObservationSelection
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
    require_open_decklists,
)
from manabot.sim.supervision import failure_record, settle_record
import managym
from managym.experience import UNVERSIONED_CONTENT_HASH
from managym.recording import GameEnd, RecordedVectorEnv


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
    game: dict[str, Any]
    setup: managym.GameSetup
    seed: int


@dataclass(eq=False)
class ActiveMatch:
    match: Match
    spec: PairingSpec
    admitted: AdmittedPairing[Seat]
    plan: CellPlan
    space: ObservationSpace
    inputs: dict[str, CheckpointInputs]
    started: float
    finished_games: int = 0


@dataclass
class SlotState:
    slot: int
    leg: PendingLeg
    gameplay_started: float
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
    read: CheckpointReader,
) -> tuple[ActiveMatch, list[PendingLeg]]:
    """Admit players, resolve per-seat decision settings and construct leg work."""
    admitted = admit_pairing(
        match,
        game_players(match.players, agents),
        matchup=spec.matchup,
        deal=spec.deal,
        matchup_mode=spec.matchup_mode,
        game_seconds=spec.game_seconds,
        arena=spec.arena,
        precision=spec.precision,
        max_commands=spec.max_commands,
        bootstrap_seed=spec.bootstrap_seed,
        output_dir=output_dir,
        read=read,
    )
    inputs = {
        name: read(player.checkpoint)
        for name, player in zip(admitted.participants, match.players, strict=True)
        if isinstance(player, CheckpointPlayer)
    }
    player_spaces = {
        name: inputs[name].space if name in inputs else None for name in admitted.agents
    }
    spaces = [space for space in player_spaces.values() if space is not None]
    if spaces and any(space.shapes != spaces[0].shapes for space in spaces):
        raise ValueError("players have different observation ABIs")
    space = spaces[0] if spaces else ObservationSpace()
    if not space.one_token:
        raise ValueError("vector evaluation requires policy input v11")

    plan = build_cell_plan(
        key=admitted.request.key,
        player_a=admitted.request.candidate,
        player_b=admitted.request.opponent,
        deal_seeds=admitted.request.deal_seeds,
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
        inputs=inputs,
        started=time.perf_counter(),
    )
    legs: list[PendingLeg] = []
    for item in plan.work:
        game = item.game
        registrations = item.registrations
        match_config = MatchConfig.model_validate(game["match_hypers"])
        require_open_decklists(match_config.decklists, registrations)
        for registration in registrations:
            name = registration.player_id
            agent = admitted.agents[name]
            if isinstance(agent, Sampler):
                inputs[name].admit(
                    match_config, allow_deck_repetition=agent.allow_deck_repetition
                )
        resolved_match = resolve_game_match(
            game, match_config, registrations, player_spaces
        )
        game["observation_hypers"] = space.config.model_dump()
        set_execution_limits(game, registrations, spec.max_commands, spec.game_seconds)
        game["runtime_identities"] = game_runtime_identities(
            game["deal_seed"], resolved_match, space, arena_source
        )
        legs.append(
            PendingLeg(
                active_match=active_match,
                game=game,
                setup=EnvMatch(resolved_match).to_rust(),
                seed=int(game["deal_seed"]),
            )
        )
    return active_match, legs


def choose_code_action(
    player: Heuristic,
    raw: managym.Observation,
    *,
    seed: int,
    space: ObservationSpace,
) -> tuple[int, float]:
    started = time.perf_counter()
    count = len(raw.action_space.actions)
    if space.config.action_overflow == "first-legal":
        count = min(count, space.config.max_actions)
    obs = {"actions_valid": np.ones(count, dtype=np.float32)}
    action = int(player.factory(seed).act(RawObservationEnv(raw), obs))  # type: ignore[arg-type]
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
    slot_state: SlotState, vector: RecordedVectorEnv, seconds: float
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
        "seconds": seconds,
    }


def settle_leg(
    slot_state: SlotState,
    record: managym.recording.GameRecord,
    vector: RecordedVectorEnv,
    seconds: float,
) -> None:
    """Bind live slot metadata before the slot is reused."""
    leg = slot_state.leg
    game = leg.game
    game["execution_seconds"] = seconds
    if leg.active_match.space.config.action_overflow == "first-legal":
        game["policy_action_cutoff"] = policy_cutoff_report(slot_state, vector, seconds)
    game["worker_performance"] = EvalPerformance(
        str(game.get("match_id", "game")),
        bool(game.get("instrumentation", True)),
    ).snapshot()
    bind_record(game, record)


def finalize_match(active_match: ActiveMatch) -> Result:
    """Replay and retain the completed deal in the shared attempt format."""
    admitted = active_match.admitted
    request = admitted.request
    with pairing_attempt(admitted.retained) as (out, cards):
        out.mkdir(parents=True, exist_ok=True)

        def play() -> CellResult:
            for item in active_match.plan.work:
                settle_record(
                    item.game,
                    out / item.evidence_path,
                    keep_completed=active_match.spec.keep_game_logs,
                )
            return finalize_settled_cell(
                (item.game for item in active_match.plan.work),
                plan=active_match.plan,
                key=request.key,
                player_a=request.candidate,
                player_b=request.opponent,
                deal_seeds=request.deal_seeds,
                out_dir=out,
                game_seconds=request.game_seconds,
                card_evidence_dir=cards,
                in_process_replay=True,
                reducers={"play": Play()},
            )

        result = retain_attempt(
            request,
            play,
            out_dir=out,
            evaluation_id=uuid4().hex,
            started=active_match.started,
            card_evidence_dir=cards,
        )
        return pairing_result(
            result,
            out=out,
            cards=cards,
            matchup=active_match.spec.matchup,
            legs=admitted.legs,
            participants=admitted.participants,
        )


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
        decision_events=True,
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
) -> AsyncGenerator[tuple[Match, Result | Exception], None]:
    """Step paired-deal legs across one `RecordedVectorEnv` and yield settled matches.

    Every active slot needing the same checkpoint in a round is collated into
    one `Prepared` batch for one awaited forward. Finished or natively-failed
    slots backfill immediately so no slot sits idle while legs are pending.
    Admission, checkpoint forward and finalization errors yield `(match, exc)`
    for the affected matches while other slots continue.
    """
    if slots < 1:
        raise ValueError("slots must be at least 1")

    arena_source = arena_source_sha256()
    read = checkpoint_reader(2 * slots)
    pending_legs: deque[PendingLeg] = deque()
    deferred_matches: deque[tuple[ActiveMatch, list[PendingLeg]]] = deque()
    ready_errors: deque[tuple[Match, Exception]] = deque()
    active_matches: set[ActiveMatch] = set()
    current_space: ObservationSpace | None = None
    source_exhausted = False
    legs_per_match_hint = 4
    # Each live slot gets its share of a decision round. A new slot starts at
    # the current total, so replay, result delivery and admission cost nothing.
    gameplay_seconds = 0.0

    vector: RecordedVectorEnv | None = None
    state_buffers: dict[str, np.ndarray] | None = None
    slot_states: list[SlotState | None] = []

    async def fetch_legs(needed_legs: int, *, drain_if_empty: bool) -> None:
        nonlocal current_space, source_exhausted, legs_per_match_hint
        while not source_exhausted and len(pending_legs) < needed_legs:
            capacity = slots - max(len(deferred_matches), len(ready_errors))
            if capacity == 0:
                break
            missing = needed_legs - len(pending_legs)
            request_count = min(capacity, math.ceil(missing / legs_per_match_hint))
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
                        read=read,
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
        completed_matches: list[ActiveMatch],
        *,
        end: GameEnd | None = None,
        info: dict[str, Any] | None = None,
    ) -> None:
        assert vector is not None
        record = vector.take_slot_record(slot, end=end, info=info)
        settle_leg(
            slot_state, record, vector, gameplay_seconds - slot_state.gameplay_started
        )
        active_match = slot_state.leg.active_match
        active_match.finished_games += 1
        slot_states[slot] = None
        if active_match.finished_games == active_match.admitted.legs:
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
            complete_leg(
                slot,
                slot_state,
                completed_matches,
                end=GameEnd(
                    status="complete",
                    winner=winner,
                    digest=digests[slot] if digests is not None else None,
                ),
            )

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
            for slot in free:
                if not pending_legs:
                    break
                leg = pending_legs.popleft()
                loads.append(managym.SlotLoad.fresh(slot, leg.setup, seed=leg.seed))
                slot_states[slot] = SlotState(
                    slot=slot,
                    leg=leg,
                    gameplay_started=gameplay_seconds,
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
                if pending_legs:
                    current_space = pending_legs[0].active_match.space
                elif deferred_matches:
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
                            gameplay_started=gameplay_seconds,
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

            # Check per-game command caps and active gameplay time before stepping.
            limit_expired: list[tuple[int, str, str]] = []
            for slot, slot_state in enumerate(slot_states):
                if slot_state is None:
                    continue
                spec = slot_state.leg.active_match.spec
                if slot_state.revision >= spec.max_commands:
                    limit_expired.append(
                        (slot, "command_cap", "game exceeded its Command cap")
                    )
                elif (
                    gameplay_seconds - slot_state.gameplay_started >= spec.game_seconds
                ):
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
                    complete_leg(
                        slot,
                        slot_state,
                        completed_matches,
                        end=GameEnd(
                            status="interrupted", reason=reason, digest=digests[slot]
                        ),
                    )
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
            round_started = monotonic()
            active_mask = [st is not None for st in slot_states]
            actors = vector.current_agent_indices()
            raw_observations = vector.decision_observations(active_mask)

            frames: dict[int, dict[str, Any]] = {}
            checkpoint_groups: dict[str, list[tuple[int, Sampler, int]]] = {}
            code_slots: list[tuple[int, Heuristic, int]] = []
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
                    checkpoint_groups.setdefault(agent.checkpoint_key, []).append(
                        (slot, agent, policy_seed)
                    )
                else:
                    code_slots.append((slot, agent, policy_seed))

            chosen_actions: dict[int, tuple[int, float]] = {}
            if checkpoint_groups:
                obs_rows = vector.observation_rows()
                prepared_batches = []
                blank_batches: dict[
                    managym.ObservationBatch, managym.ObservationBatch
                ] = {}
                for members in checkpoint_groups.values():
                    first_slot, sampler, _ = members[0]
                    state = slot_states[first_slot]
                    assert state is not None
                    player_id = slot_meta[first_slot][1]
                    spec = state.leg.active_match.inputs[player_id].spec
                    coords: list[BatchRow] = []
                    for slot, seat, _ in members:
                        batch, row = obs_rows[slot]
                        if seat.blank_decision_memory:
                            if batch not in blank_batches:
                                blank_batches[batch] = batch.blank_decision_memory()
                            batch = blank_batches[batch]
                        coords.append((batch, row))
                    selection = ObservationSelection(spec.space, tuple(coords))
                    prepared_batches.append(sampler.prepare(selection, spec))
                forward_outputs = await asyncio.gather(
                    *(
                        members[0][1].logits(prepared)
                        for members, prepared in zip(
                            checkpoint_groups.values(), prepared_batches, strict=True
                        )
                    ),
                    return_exceptions=True,
                )
                checkpoint_errors: dict[str, ForwardError] = {}
                for output in forward_outputs:
                    if isinstance(output, ForwardError):
                        checkpoint_errors[output.checkpoint] = output
                    elif isinstance(output, BaseException):
                        raise output
                if checkpoint_errors:
                    failed_matches = {
                        active: checkpoint_errors[player.checkpoint.key]
                        for active in active_matches
                        for player in active.match.players
                        if isinstance(player, CheckpointPlayer)
                        and player.checkpoint.key in checkpoint_errors
                    }
                    pending_legs = deque(
                        leg
                        for leg in pending_legs
                        if leg.active_match not in failed_matches
                    )
                    deferred_matches = deque(
                        (active, legs)
                        for active, legs in deferred_matches
                        if active not in failed_matches
                    )
                    for slot, state in enumerate(slot_states):
                        if (
                            state is not None
                            and state.leg.active_match in failed_matches
                        ):
                            slot_states[slot] = None
                            active_mask[slot] = False
                    active_slots = [slot for slot in active_slots if active_mask[slot]]
                    for active, error in failed_matches.items():
                        active_matches.remove(active)
                        ready_errors.append((active.match, error))
                    if not active_slots:
                        continue

                for members, prepared, output in zip(
                    checkpoint_groups.values(),
                    prepared_batches,
                    forward_outputs,
                    strict=True,
                ):
                    if isinstance(output, BaseException):
                        continue
                    logits, elapsed = output
                    per_row_latency = elapsed / len(members)
                    for row_idx, (slot, sampler, policy_seed) in enumerate(members):
                        if not active_mask[slot]:
                            continue
                        action = sampler.sample(
                            logits[row_idx], prepared.legal[row_idx], policy_seed
                        )
                        chosen_actions[slot] = (action, per_row_latency)

            for slot, agent, policy_seed in code_slots:
                if not active_mask[slot]:
                    continue
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
            gameplay_seconds += (monotonic() - round_started) / len(active_slots)

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
                    complete_leg(slot, slot_state, completed_matches, info=info)
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
                    complete_leg(
                        slot,
                        slot_state,
                        completed_matches,
                        end=GameEnd(
                            status="interrupted",
                            reason="truncated",
                            digest=info.get("record_terminal_digest"),
                        ),
                    )
                    continue

                if terminated[slot]:
                    winner = info.get("winner_index")
                    leg.game.update(
                        winner=winner,
                        terminated=True,
                        termination_reason="draw" if winner is None else "terminal",
                    )
                    complete_leg(slot, slot_state, completed_matches)

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
