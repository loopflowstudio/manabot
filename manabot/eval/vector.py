"""Asynchronous vectorized match runner over one native `RecordedVectorEnv`.

`play_pairings` steps up to `slots` paired-deal legs concurrently on the v11
CSR `ObservationBatch` -> `Prepared` path. In each round, every active slot
needing the same checkpoint is gathered into one pre-collated `Prepared` input
for one awaited forward. The native engine settles built-in opponents between
checkpoint choices, while registered code seats choose from detached native
observations. Finished or natively-failed legs free their slot for backfill via
seeded `SlotLoad.fresh(..., seed=)`. Replay workers verify and retain each
finished match while the runner continues stepping other matches.

This module is Torch-free (`tests/test_package_boundaries.py`).
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from collections.abc import AsyncGenerator, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import islice
import math
from pathlib import Path
import time
from time import monotonic
import traceback
from typing import Any, Literal, NamedTuple, Protocol
from uuid import uuid4

from etude.eval.agents import ForwardError
from etude.eval.records import CheckpointPlayer, Match
from etude.eval.result import Result
import numpy as np

from manabot.env.failures import PolicyActionCutoffs
from manabot.env.match import Match as EnvMatch
from manabot.env.observation import ObservationSpace, step_buffers
from manabot.eval.agents import (
    CheckpointInputs,
    CheckpointReader,
    CodeSeat,
    NativeHeuristic,
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
from manabot.sim.agent import GameChoice
from manabot.sim.evaluation_records import (
    DecisionIdentity,
    bind_record,
    set_execution_limits,
)
from manabot.sim.native_game import (
    game_runtime_identities,
    resolve_game_match,
)
from manabot.sim.performance import EvalPerformance
from manabot.sim.registration import require_open_decklists
from manabot.sim.supervision import failure_record, is_game_error, settle_record
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
class CodeDecisionEnv:
    """The code player's observation and, on demand, a detached search root."""

    last_raw_obs: managym.Observation
    vector: RecordedVectorEnv
    slot: int

    @property
    def _engine(self) -> managym.Env:
        mask = [i == self.slot for i in range(len(self.vector.games))]
        root = self.vector.snapshot_slots(mask)[self.slot]
        assert root is not None
        return root.restore()


@dataclass
class PendingLeg:
    active_match: ActiveMatch
    game: dict[str, Any]
    setup: managym.GameSetup
    player_policies: tuple[str, str]
    choosers: dict[str, Sampler | GameChoice]


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


class VectorLayout(NamedTuple):
    """The native layout and settings bound once for all slots."""

    shapes: dict[str, tuple[int, ...]]
    action_overflow: Literal["error", "first-legal"]
    decision_memory: bool
    include_own: bool


def vector_layout_key(space: ObservationSpace) -> VectorLayout:
    config = space.config
    return VectorLayout(
        space.shapes,
        config.action_overflow,
        config.max_decision_records > 0,
        config.decision_memory_include_own,
    )


def admit_active_match(
    match: Match,
    agents: Sequence[object],
    spec: PairingSpec,
    *,
    output_dir: Path | None,
    read: CheckpointReader,
) -> tuple[ActiveMatch, deque[PendingLeg]]:
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
    opponent = inputs.get("opponent")
    if opponent is not None and vector_layout_key(opponent.space) != vector_layout_key(
        space
    ):
        raise ValueError("players have different vector observation layouts")
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
    native_policies = {
        name: seat.player_policy if isinstance(seat, NativeHeuristic) else "agent"
        for name, seat in admitted.agents.items()
    }
    legs: deque[PendingLeg] = deque()
    for item in plan.work:
        game = item.game
        # Vector capture has no viewer projection to audit.
        del game["integrity"]["private_exposures"]
        registrations = item.registrations
        match_config = MatchConfig.model_validate(game["match_hypers"])
        require_open_decklists(match_config.decklists, registrations)
        choosers: dict[str, Sampler | GameChoice] = {}
        for registration in registrations:
            name = registration.player_id
            seat = admitted.agents[name]
            if isinstance(seat, Sampler):
                inputs[name].admit(
                    match_config, allow_deck_repetition=seat.allow_deck_repetition
                )
                choosers[name] = seat
            elif isinstance(seat, CodeSeat):
                choosers[name] = seat.open_game(game["player_seeds"][name])
        resolved_match = resolve_game_match(
            game, match_config, registrations, player_spaces
        )
        game["observation_hypers"] = space.config.model_dump()
        set_execution_limits(game, registrations, spec.max_commands, spec.game_seconds)
        game["runtime_identities"] = game_runtime_identities(resolved_match, space)
        first, second = game["seat_players"]
        legs.append(
            PendingLeg(
                active_match=active_match,
                game=game,
                setup=EnvMatch(resolved_match).to_rust(),
                player_policies=(native_policies[first], native_policies[second]),
                choosers=choosers,
            )
        )
    return active_match, legs


def choose_code_action(
    choose: GameChoice,
    env: CodeDecisionEnv,
    *,
    seed: int,
    space: ObservationSpace,
) -> tuple[int, float]:
    started = time.perf_counter()
    count = len(env.last_raw_obs.action_space.actions)
    if space.config.action_overflow == "first-legal":
        count = min(count, space.config.max_actions)
    obs = {"actions_valid": np.ones(count, dtype=np.float32)}
    action = int(choose(env, obs, seed))  # type: ignore[arg-type]
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
    layout = vector_layout_key(space)
    vector = RecordedVectorEnv(
        num_envs=num_envs,
        seed=0,
        skip_trivial=True,
        player_policies=("agent", "agent"),
        decision_memory=layout.decision_memory,
        decision_memory_include_own=layout.include_own,
        record_fraction=1.0,
        record_in_memory=True,
        decision_events=True,
        context={"source_kind": "evaluation"},
    )
    state_buffers = step_buffers(num_envs)
    vector.set_observation_layout(
        state_buffers,
        layout.action_overflow,
        layout=space.layout,
    )
    return vector, state_buffers


@dataclass(frozen=True)
class PlayedRound:
    players: dict[int, str]
    seconds_per_slot: float
    errors: dict[str, ForwardError]
    code_errors: dict[int, BaseException]


async def step_round(
    vector: RecordedVectorEnv, states: list[SlotState | None], space: ObservationSpace
) -> PlayedRound:
    """Choose once per live slot, omitting matches whose checkpoint failed."""
    started = monotonic()
    active = [state is not None for state in states]
    actors = vector.current_agent_indices()
    turns: dict[int, DecisionIdentity] = {}
    groups: dict[str, list[tuple[SlotState, Sampler]]] = defaultdict(list)
    code: dict[int, GameChoice] = {}
    for state in states:
        if state is None:
            continue
        slot = state.slot
        actor = actors[slot]
        assert actor in (0, 1)
        game = state.leg.game
        turn = DecisionIdentity.for_actor(
            game, actor, state.revision, state.decision_counts
        )
        turns[slot] = turn
        chooser = state.leg.choosers[turn.player_id]
        if isinstance(chooser, Sampler):
            groups[chooser.checkpoint_key].append((state, chooser))
        else:
            code[slot] = chooser

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
    code_errors: dict[int, BaseException] = {}
    code_active = [on and slot in code for slot, on in enumerate(active)]
    if any(code_active):
        observations = vector.decision_observations(code_active)
        for slot, raw in enumerate(observations):
            if not code_active[slot]:
                continue
            assert raw is not None and int(raw.agent.player_index) == actors[slot]
            try:
                action, elapsed = choose_code_action(
                    code[slot],
                    CodeDecisionEnv(raw, vector, slot),
                    seed=turns[slot].policy_seed,
                    space=space,
                )
            except BaseException as error:
                if not is_game_error(error):
                    raise
                code_errors[slot] = error
                active[slot] = False
                continue
            if not 0 <= action < len(raw.action_space.actions):
                raise RuntimeError(f"illegal offer {action}")
            chosen[slot] = action, elapsed
    players = {slot: turn.player_id for slot, turn in turns.items() if active[slot]}
    if not players:
        return PlayedRound(players, 0.0, errors, code_errors)

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
        annotations[slot] = {"extra": {"evaluation": turns[slot].evidence(elapsed)}}
        if summaries is not None:
            count, kind = summaries[slot]
            state.cutoffs.record(
                count, kind, space.config.max_actions, lambda: vector.game_inputs(slot)
            )

    vector.step_isolated_into_buffers(
        actions, active, recover_capacity=True, annotations=annotations
    )
    return PlayedRound(
        players, (monotonic() - started) / len(players), errors, code_errors
    )


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
    slots backfill while fewer than `slots` settled matches await replay.
    Admission, checkpoint forward and finalization errors yield `(match, exc)`
    for the affected matches while other slots continue.
    """
    if slots < 1:
        raise ValueError("slots must be at least 1")

    read = checkpoint_reader(2 * slots)
    # A held deal owns its unseated legs until its last slot finishes or fails.
    held: dict[ActiveMatch, deque[PendingLeg]] = {}
    ready_errors: deque[tuple[Match, Exception]] = deque()
    replay_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="eval-replay")
    loop = asyncio.get_running_loop()
    replays: dict[asyncio.Future[Result], Match] = {}
    current_space: ObservationSpace | None = None
    source_exhausted = False
    legs_per_match_hint = 4
    # Each live slot gets its share of a decision round. A new slot starts at
    # the current total, so replay, result delivery and admission cost nothing.
    gameplay_seconds = 0.0

    vector: RecordedVectorEnv | None = None
    state_buffers: dict[str, np.ndarray] | None = None
    slot_states: list[SlotState | None] = []

    async def take_legs(limit: int, space: ObservationSpace | None) -> list[PendingLeg]:
        """Fill a new vector from the oldest layout, or backfill an existing one."""
        nonlocal source_exhausted, legs_per_match_hint
        layout = vector_layout_key(space) if space is not None else None
        selected: list[PendingLeg] = []
        deferred = 0

        def select(match: ActiveMatch, queue: deque[PendingLeg]) -> None:
            nonlocal layout, deferred
            if not queue:
                return
            key = vector_layout_key(match.space)
            if layout is None:
                layout = key
            if key != layout:
                deferred += 1
                return
            while queue and len(selected) < limit:
                selected.append(queue.popleft())

        for active_match, queue in held.items():
            select(active_match, queue)
        while len(selected) < limit and not source_exhausted and not ready_errors:
            if deferred >= slots:
                break
            missing = limit - len(selected)
            request_count = min(
                slots - deferred, math.ceil(missing / legs_per_match_hint)
            )
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
                        output_dir=output_dir,
                        read=read,
                    )
                except Exception as exc:
                    ready_errors.append((match, exc))
                    continue
                legs_per_match_hint = len(legs)
                held[active_match] = legs
                select(active_match, legs)
            if space is not None or len(batch) < request_count:
                break
        return selected

    def complete_leg(
        slot_state: SlotState,
        *,
        end: GameEnd | None = None,
    ) -> None:
        assert vector is not None
        slot = slot_state.slot
        record = vector.take_slot_record(slot, end=end)
        settle_leg(
            slot_state, record, vector, gameplay_seconds - slot_state.gameplay_started
        )
        active_match = slot_state.leg.active_match
        active_match.finished_games += 1
        slot_states[slot] = None
        if active_match.finished_games == len(active_match.plan.work):
            del held[active_match]
            future = loop.run_in_executor(replay_pool, finalize_match, active_match)
            replays[future] = active_match.match

    def complete_failed_leg(slot_state: SlotState, info: dict[str, Any]) -> None:
        game = slot_state.leg.game
        error_class = str(info.get("game_error_class", "AgentError"))
        if error_class == "ObservationCapacity":
            game["integrity"]["truncations"] += 1
            failure_record(
                game, "truncated", "environment or observation truncated", None
            )
        else:
            game["error_class"] = error_class
            if info.get("game_error_trace"):
                game["failure_trace"] = str(info["game_error_trace"])
            seat = info.get("game_error_seat")
            failure_record(
                game,
                "crash",
                f"{error_class}: {info['game_error']}",
                game["seat_players"][seat] if seat is not None else None,
            )
        complete_leg(slot_state)

    def seat_legs(free: Sequence[int], legs: list[PendingLeg]) -> list[SlotState]:
        seated = []
        for slot, leg in zip(free, legs):
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
        infos = vector.get_last_info()
        mask = [False] * len(slot_states)
        for state in ended:
            mask[state.slot] = not infos[state.slot].get("game_error")
        digests = vector.state_digests(mask)
        for slot_state in ended:
            slot = slot_state.slot
            if infos[slot].get("game_error"):
                complete_failed_leg(slot_state, infos[slot])
                continue
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
        assert vector is not None and current_space is not None
        while len(replays) < slots and (
            free := [i for i, state in enumerate(slot_states) if state is None]
        ):
            seated = seat_legs(free, await take_legs(len(free), current_space))
            if not seated:
                break
            vector.load_slots_into_buffers(
                [
                    managym.SlotLoad.fresh(
                        state.slot,
                        state.leg.setup,
                        seed=state.leg.game["deal_seed"],
                        player_policies=state.leg.player_policies,
                    )
                    for state in seated
                ],
                isolated=True,
            )
            if not settle_openings(seated) or ready_errors:
                break

    try:
        while True:
            while ready_errors:
                yield ready_errors.popleft()
            for future in [future for future in replays if future.done()]:
                match = replays.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    yield match, exc
                else:
                    yield match, result

            if vector is None:
                legs = await take_legs(slots, None)
                if not legs:
                    if not ready_errors and source_exhausted:
                        if not replays:
                            break
                        await asyncio.wait(replays, return_when=asyncio.FIRST_COMPLETED)
                    continue

                current_space = legs[0].active_match.space
                num_envs = len(legs) if source_exhausted else slots
                vector, state_buffers = open_vector(num_envs, current_space)
                slot_states = [None] * num_envs
                seated = seat_legs(range(num_envs), legs)
                # Native reset needs a setup for each slot, including idle ones.
                idle = num_envs - len(seated)
                vector.reset_each_into_buffers(
                    [state.leg.setup for state in seated]
                    + [seated[-1].leg.setup] * idle,
                    seeds=[state.leg.game["deal_seed"] for state in seated]
                    + [0] * idle,
                    player_policies=[state.leg.player_policies for state in seated]
                    + [("agent", "agent")] * idle,
                    isolated=True,
                )
                settle_openings(seated)
                # Slot state now owns the legs. Release the opening deals on backfill.
                del legs, seated
                continue

            assert state_buffers is not None and current_space is not None
            # Deliver failures before asking Etude for replacements.
            await backfill_free_slots()
            if ready_errors:
                continue

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
                continue

            if not any(slot_states):
                if len(replays) >= slots:
                    await asyncio.wait(replays, return_when=asyncio.FIRST_COMPLETED)
                    continue
                vector.close()
                vector = None
                state_buffers = None
                slot_states = []
                current_space = None
                continue

            round_result = await step_round(vector, slot_states, current_space)
            gameplay_seconds += round_result.seconds_per_slot
            if round_result.errors:
                for match in list(held):
                    if (error := match.forward_error(round_result.errors)) is not None:
                        del held[match]
                        ready_errors.append((match.match, error))
                for slot, state in enumerate(slot_states):
                    if state is not None and state.leg.active_match not in held:
                        slot_states[slot] = None
            if round_result.code_errors:
                actors = vector.current_agent_indices()
                digests = vector.state_digests(
                    [i in round_result.code_errors for i in range(len(slot_states))]
                )
                for slot, error in round_result.code_errors.items():
                    slot_state = slot_states[slot]
                    assert slot_state is not None
                    game = slot_state.leg.game
                    game["error_class"] = type(error).__name__
                    game["failure_trace"] = "".join(traceback.format_exception(error))
                    failure_record(
                        game,
                        "crash",
                        f"{type(error).__name__}: {error}",
                        game["seat_players"][actors[slot]],
                    )
                    complete_leg(
                        slot_state,
                        end=GameEnd(
                            status="interrupted", reason="crash", digest=digests[slot]
                        ),
                    )
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
                    complete_failed_leg(slot_state, info)
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

    finally:
        try:
            if vector is not None:
                vector.close()
        finally:
            try:
                # Collect errors even if the consumer closes before reading them.
                if replays:
                    await asyncio.gather(*replays, return_exceptions=True)
            finally:
                replay_pool.shutdown()
