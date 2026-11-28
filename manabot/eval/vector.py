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
from collections import defaultdict, deque
from collections.abc import AsyncGenerator, Callable, Sequence
from dataclasses import dataclass, field
from itertools import islice
import math
from pathlib import Path
import time
from time import monotonic
from typing import Any, Protocol
from uuid import uuid4

from etude.eval.agents import ForwardError
from etude.eval.records import CheckpointPlayer, Match
from etude.eval.result import Result
import numpy as np

from manabot.env.failures import PolicyActionCutoffs
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
    prepare_batch,
)
from manabot.eval.behaviour import Play
from manabot.eval.evaluation import CellResult, retain_attempt
from manabot.eval.match import (
    CellPlan,
    build_cell_plan,
    finalize_settled_cell,
)
from manabot.eval.pairing import (
    PARTICIPANTS,
    AdmittedPairing,
    PairingSpec,
    admit_pairing,
    pairing_attempt,
    pairing_result,
)
from manabot.infra.config import MatchConfig
from manabot.policy.prepare import Prepared
from manabot.policy.spans import BatchRow, ObservationSelection
from manabot.sim.evaluation_records import (
    bind_record,
    decision_annotation,
    set_execution_limits,
)
from manabot.sim.native_game import (
    ArenaDecision,
    arena_decision,
    arena_source_sha256,
    chosen_offer,
    game_runtime_identities,
    resolve_game_match,
)
from manabot.sim.performance import EvalPerformance
from manabot.sim.registration import require_open_decklists
from manabot.sim.supervision import failure_record, settle_record
import managym
from managym.recording import GameEnd, RecordedVectorEnv


class MatchSource(Protocol):
    """Abstract asynchronous match feed matching Etude's `Matches` contract."""

    async def take(self, limit: int) -> Sequence[tuple[Match, Sequence[object]]]: ...


class LocalMatches:
    """In-memory `MatchSource` over a fixed sequence of `(match, agents)` pairs."""

    def __init__(self, items: Sequence[tuple[Match, Sequence[object]]]) -> None:
        self._matches = iter(items)

    async def take(self, limit: int) -> list[tuple[Match, Sequence[object]]]:
        if limit < 1:
            raise ValueError("take limit must be positive")
        return list(islice(self._matches, limit))


@dataclass(frozen=True)
class RawObservationEnv:
    """Minimal read-only adapter supplying `last_raw_obs` to code players."""

    last_raw_obs: managym.Observation


@dataclass
class PendingLeg:
    active_match: ActiveMatch
    game: dict[str, Any]
    setup: managym.GameSetup


@dataclass(eq=False)
class ActiveMatch:
    match: Match
    keep_game_logs: bool
    admitted: AdmittedPairing[Seat]
    plan: CellPlan
    space: ObservationSpace
    inputs: dict[str, CheckpointInputs]
    started: float
    finished_games: int = 0

    def forward_error(self, errors: dict[str, ForwardError]) -> ForwardError | None:
        for player in self.match.players:
            if isinstance(player, CheckpointPlayer) and player.checkpoint.key in errors:
                return errors[player.checkpoint.key]
        return None


@dataclass
class SlotState:
    slot: int
    leg: PendingLeg
    gameplay_started: float
    decision_counts: dict[str, int]
    cutoffs: PolicyActionCutoffs = field(default_factory=PolicyActionCutoffs)

    @property
    def revision(self) -> int:
        return sum(self.decision_counts.values())


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
        spec,
        output_dir=output_dir,
        read=read,
    )
    inputs = {
        name: read(player.checkpoint)
        for name, player in zip(PARTICIPANTS, match.players, strict=True)
        if isinstance(player, CheckpointPlayer)
    }
    player_spaces = {
        name: inputs[name].space if name in inputs else None for name in admitted.agents
    }
    # Pairing admission requires a checkpoint in the candidate seat.
    space = inputs["candidate"].space
    if any(checkpoint.space.shapes != space.shapes for checkpoint in inputs.values()):
        raise ValueError("players have different observation ABIs")
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
        keep_game_logs=spec.keep_game_logs,
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
        game["policy_action_cutoff"] = {
            **slot_state.cutoffs.report(),
            "identities": vector.identities,
            "context": {},
            "seconds": seconds,
        }
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
                    keep_completed=active_match.keep_game_logs,
                )
            return finalize_settled_cell(
                (item.game for item in active_match.plan.work),
                plan=active_match.plan,
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
        return pairing_result(result, out=out, cards=cards)


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


@dataclass(frozen=True)
class PlayedRound:
    players: dict[int, str]
    seconds_per_slot: float
    errors: dict[str, ForwardError]


async def step_round(
    vector: RecordedVectorEnv, states: list[SlotState | None], space: ObservationSpace
) -> PlayedRound:
    """Choose once per live slot, omitting matches whose checkpoint failed."""
    started = monotonic()
    active = [state is not None for state in states]
    actors = vector.current_agent_indices()
    observations = vector.decision_observations(active)
    turns: dict[int, ArenaDecision] = {}
    groups: dict[str, list[tuple[SlotState, Sampler]]] = defaultdict(list)
    code: list[tuple[SlotState, Heuristic, managym.Observation]] = []
    for state in states:
        if state is None:
            continue
        slot = state.slot
        raw = observations[slot]
        assert raw is not None
        if actors[slot] != int(raw.agent.player_index):
            raise RuntimeError(
                f"vector actor {actors[slot]} differs from observation actor {raw.agent.player_index}"
            )
        turn = arena_decision(
            state.leg.game, raw, state.revision, state.decision_counts
        )
        turns[slot] = turn
        seat = state.leg.active_match.admitted.agents[turn.player_id]
        if isinstance(seat, Sampler):
            groups[seat.checkpoint_key].append((state, seat))
        else:
            code.append((state, seat, raw))

    chosen: dict[int, tuple[int, float]] = {}
    errors: dict[str, ForwardError] = {}
    if groups:
        rows = vector.observation_rows()
        blank: dict[managym.ObservationBatch, managym.ObservationBatch] = {}

        def prepare_group(
            members: list[tuple[SlotState, Sampler]],
        ) -> Prepared[np.ndarray]:
            first, _ = members[0]
            spec = first.leg.active_match.inputs[turns[first.slot].player_id].spec
            coords: list[BatchRow] = []
            for state, seat in members:
                batch, index = rows[state.slot]
                if seat.blank_decision_memory:
                    if batch not in blank:
                        blank[batch] = batch.blank_decision_memory()
                    batch = blank[batch]
                coords.append((batch, index))
            return prepare_batch(ObservationSelection(spec.space, tuple(coords)), spec)

        async def sample_group(
            members: list[tuple[SlotState, Sampler]], prepared: Prepared[np.ndarray]
        ) -> None:
            _, sampler = members[0]
            try:
                logits, elapsed = await sampler.logits(prepared)
            except ForwardError as error:
                errors[error.checkpoint] = error
                return
            for row, (state, seat) in enumerate(members):
                action = seat.sample(
                    logits[row], prepared.legal[row], turns[state.slot].policy_seed
                )
                chosen[state.slot] = (action, elapsed / len(members))

        # Prepare all checkpoints before dispatch so their forwards can run together.
        batches = [(group, prepare_group(group)) for group in groups.values()]
        outputs = await asyncio.gather(
            *(sample_group(group, batch) for group, batch in batches),
            return_exceptions=True,
        )
        for output in outputs:
            if isinstance(output, BaseException):
                raise output

    if errors:
        for state in states:
            if state is not None and state.leg.active_match.forward_error(errors):
                active[state.slot] = False
    for state, seat, raw in code:
        if active[state.slot]:
            chosen[state.slot] = choose_code_action(
                seat, raw, seed=turns[state.slot].policy_seed, space=space
            )
    players = {slot: turn.player_id for slot, turn in turns.items() if active[slot]}
    if not players:
        return PlayedRound(players, 0.0, errors)

    summaries = (
        vector.current_action_summaries()
        if space.config.action_overflow == "first-legal"
        else None
    )
    actions = [0] * len(states)
    annotations: list[dict[str, object] | None] = [None] * len(states)
    for slot in players:
        state = states[slot]
        assert state is not None
        actions[slot], elapsed = chosen[slot]
        turn = turns[slot]
        offer = chosen_offer(state.leg.game, turn, actions[slot])
        # Replay derives the bound Command from the recorded offer and revision.
        annotations[slot] = decision_annotation(turn.evidence(offer, elapsed))
        if summaries is not None:
            count, kind = summaries[slot]
            state.cutoffs.record(
                count, kind, space.config.max_actions, lambda: vector.game_inputs(slot)
            )

    vector.step_isolated_into_buffers(
        actions, active, recover_capacity=True, annotations=annotations
    )
    return PlayedRound(players, (monotonic() - started) / len(players), errors)


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
    completed_matches: deque[ActiveMatch] = deque()
    current_space: ObservationSpace | None = None
    source_exhausted = False
    legs_per_match_hint = 4
    # Each live slot gets its share of a decision round. A new slot starts at
    # the current total, so replay, result delivery and admission cost nothing.
    gameplay_seconds = 0.0

    vector: RecordedVectorEnv | None = None
    state_buffers: dict[str, np.ndarray] | None = None
    slot_states: list[SlotState | None] = []

    async def fetch_legs(needed_legs: int, *, fill_layout: bool) -> None:
        nonlocal current_space, source_exhausted, legs_per_match_hint
        while not source_exhausted and len(pending_legs) < needed_legs:
            capacity = slots - max(len(deferred_matches), len(ready_errors))
            # One failed forward can settle live and deferred deals at once.
            if capacity <= 0:
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
                legs_per_match_hint = len(legs)
                active_matches.add(active_match)
                if current_space is None:
                    current_space = active_match.space
                if active_match.space.config == current_space.config:
                    pending_legs.extend(legs)
                else:
                    deferred_matches.append((active_match, legs))
            if not fill_layout:
                break
            if len(batch) < request_count and (pending_legs or deferred_matches):
                break

    def complete_leg(
        slot_state: SlotState,
        *,
        end: GameEnd | None = None,
        info: dict[str, Any] | None = None,
    ) -> None:
        assert vector is not None
        slot = slot_state.slot
        record = vector.take_slot_record(slot, end=end, info=info)
        settle_leg(
            slot_state, record, vector, gameplay_seconds - slot_state.gameplay_started
        )
        active_match = slot_state.leg.active_match
        active_match.finished_games += 1
        slot_states[slot] = None
        if active_match.finished_games == len(active_match.plan.work):
            active_matches.remove(active_match)
            completed_matches.append(active_match)

    def seat_legs(free: Sequence[int]) -> list[SlotState]:
        seated = []
        for slot in free:
            if not pending_legs:
                break
            leg = pending_legs.popleft()
            state = SlotState(
                slot,
                leg,
                gameplay_seconds,
                decision_counts=dict.fromkeys(leg.game["seat_players"], 0),
            )
            slot_states[slot] = state
            seated.append(state)
        return seated

    def settle_openings(seated: Sequence[SlotState]) -> bool:
        assert vector is not None and state_buffers is not None
        actors = vector.current_agent_indices()
        ended = [
            state
            for state in seated
            if actors[state.slot] < 0 or state_buffers["terminated"][state.slot]
        ]
        if not ended:
            return False
        mask = [False] * len(slot_states)
        for state in ended:
            mask[state.slot] = True
        digests = vector.state_digests(mask)
        infos = vector.get_last_info()
        for slot_state in ended:
            slot = slot_state.slot
            winner = infos[slot].get("winner_index")
            slot_state.leg.game.update(
                winner=winner,
                terminated=True,
                termination_reason="draw" if winner is None else "terminal",
            )
            complete_leg(
                slot_state,
                end=GameEnd(status="complete", winner=winner, digest=digests[slot]),
            )
        return True

    async def backfill_free_slots() -> None:
        assert vector is not None
        while free := [i for i, state in enumerate(slot_states) if state is None]:
            await fetch_legs(len(free), fill_layout=False)
            seated = seat_legs(free)
            if not seated:
                break
            vector.load_slots_into_buffers(
                [
                    managym.SlotLoad.fresh(
                        state.slot, state.leg.setup, seed=state.leg.game["deal_seed"]
                    )
                    for state in seated
                ]
            )
            if not settle_openings(seated):
                break

    try:
        while True:
            while completed_matches:
                done = completed_matches.popleft()
                try:
                    result = finalize_match(done)
                except Exception as exc:
                    yield done.match, exc
                else:
                    yield done.match, result
            while ready_errors:
                yield ready_errors.popleft()

            if vector is None:
                if pending_legs:
                    current_space = pending_legs[0].active_match.space
                elif deferred_matches:
                    current_space = deferred_matches[0][0].space
                    for _ in range(len(deferred_matches)):
                        match, legs = deferred_matches.popleft()
                        if match.space.config == current_space.config:
                            pending_legs.extend(legs)
                        else:
                            deferred_matches.append((match, legs))

                await fetch_legs(slots, fill_layout=True)
                if not pending_legs:
                    if source_exhausted and not (deferred_matches or ready_errors):
                        break
                    continue

                assert current_space is not None
                num_envs = min(slots, len(pending_legs)) if source_exhausted else slots
                vector, state_buffers = open_vector(num_envs, current_space)
                slot_states = [None] * num_envs
                seated = seat_legs(range(num_envs))
                # Native reset needs a setup for each slot, including idle ones.
                idle = num_envs - len(seated)
                vector.reset_each_into_buffers(
                    [state.leg.setup for state in seated]
                    + [seated[-1].leg.setup] * idle,
                    seeds=[state.leg.game["deal_seed"] for state in seated]
                    + [0] * idle,
                )
                settle_openings(seated)
                await backfill_free_slots()
                continue

            assert state_buffers is not None and current_space is not None

            # Check per-game command caps and active gameplay time before stepping.
            expired: dict[int, tuple[str, str]] = {}
            for slot, slot_state in enumerate(slot_states):
                if slot_state is None:
                    continue
                request = slot_state.leg.active_match.admitted.request
                if slot_state.revision >= request.max_commands:
                    expired[slot] = ("command_cap", "game exceeded its Command cap")
                elif (
                    gameplay_seconds - slot_state.gameplay_started
                    >= request.game_seconds
                ):
                    expired[slot] = ("timeout", "game wall-clock budget exhausted")

            if expired:
                digests = vector.state_digests(
                    [i in expired for i in range(len(slot_states))]
                )
                for slot, (reason, message) in expired.items():
                    slot_state = slot_states[slot]
                    assert slot_state is not None
                    failure_record(slot_state.leg.game, reason, message, None)
                    complete_leg(
                        slot_state,
                        end=GameEnd(
                            status="interrupted", reason=reason, digest=digests[slot]
                        ),
                    )
                await backfill_free_slots()
                continue

            if not any(slot_states):
                vector.close()
                vector = None
                state_buffers = None
                slot_states = []
                current_space = None
                continue

            round_result = await step_round(vector, slot_states, current_space)
            gameplay_seconds += round_result.seconds_per_slot
            if round_result.errors:
                failed = {
                    match: error
                    for match in active_matches
                    if (error := match.forward_error(round_result.errors)) is not None
                }
                pending_legs = deque(
                    leg for leg in pending_legs if leg.active_match not in failed
                )
                deferred_matches = deque(
                    (match, legs)
                    for match, legs in deferred_matches
                    if match not in failed
                )
                for slot, state in enumerate(slot_states):
                    if state is not None and state.leg.active_match in failed:
                        slot_states[slot] = None
                for match, error in failed.items():
                    active_matches.remove(match)
                    ready_errors.append((match.match, error))
                if not round_result.players:
                    continue

            infos = vector.get_last_info()
            terminated = state_buffers["terminated"]
            truncated = state_buffers["truncated"]

            for slot, player_id in round_result.players.items():
                slot_state = slot_states[slot]
                assert slot_state is not None
                leg = slot_state.leg
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
                    complete_leg(slot_state, info=info)
                    continue

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
                        slot_state,
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
                    complete_leg(slot_state)

            await backfill_free_slots()
    finally:
        if vector is not None:
            vector.close()
