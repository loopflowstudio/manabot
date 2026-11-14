"""Scheduled frozen-policy capture with durable, private game evidence.

A runtime owns policy/native APIs; this module owns row selection, viewer history
and attempt accounting. Only completed games supply fitting rows. Receipts and
shards contain private seeds and replay commands and must stay private.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Literal, Protocol

from manabot.belief.sampling_data import (
    SamplerDataset,
    SamplerExample,
    SamplerGame,
    Split,
    UnsupportedSamplerRow,
    _read_game,
    public_sampler_input,
)
from manabot.belief.sampling_types import SamplerSchema, _digest
from manabot.belief.state import ViewerHistory
from managym.decision import Command, DecisionFrame, Observation, TransitionReceipt


@dataclass(frozen=True, slots=True)
class CaptureSlot:
    game_id: str
    seed: int
    assignment: int
    split: Split
    deck_ids: tuple[str, str]

    def __post_init__(self) -> None:
        if not self.game_id or self.assignment not in (0, 1):
            raise ValueError("capture slot requires an identity and seat assignment")
        if self.split not in ("train", "validation", "test"):
            raise ValueError("unknown capture split")


@dataclass(frozen=True, slots=True)
class CaptureTransition:
    receipt: TransitionReceipt
    terminated: bool
    truncated: bool


class CaptureRuntime(Protocol):
    """The behavior policy and privileged truth stay behind this boundary."""

    retained_actions: int

    def reset(self, slot: CaptureSlot) -> None: ...
    def observation(self, viewer: int) -> Observation: ...
    def hidden_hand_constraints_json(self, viewer: int) -> str: ...
    def hand_counts(self, player: int) -> Mapping[str, int]: ...
    def frame(self) -> DecisionFrame: ...
    def act(self) -> int: ...
    def step(self, command: Command) -> CaptureTransition: ...


class GameCaptureFailure(RuntimeError):
    """An isolated game failure; other scheduled slots may continue."""


@dataclass(frozen=True, slots=True)
class CaptureCoverage:
    decisions: int
    eligible_rows: tuple[int, int]
    unsupported_rows: tuple[int, int]
    selected_moments: int
    overflow_decisions: int
    dropped_actions: int


@dataclass(frozen=True, slots=True)
class CapturedGame:
    game: SamplerGame | None
    coverage: CaptureCoverage
    commands: tuple[Command, ...]
    failure: str | None = None


def retained_offer_count(
    frame: DecisionFrame,
    valid: Sequence[bool],
    *,
    overflow: Literal["error", "first-legal"],
) -> int:
    """Require exactly the configured legal prefix; never remap offer IDs."""
    legal = len(frame.offers)
    capacity = len(valid)
    if legal > capacity and overflow != "first-legal":
        raise ValueError("frozen policy cannot encode every legal offer")
    retained = min(legal, capacity)
    if retained == 0 or tuple(bool(x) for x in valid) != (
        (True,) * retained + (False,) * (capacity - retained)
    ):
        raise ValueError("frozen policy mask is not the configured legal prefix")
    return retained


def capture_game(
    runtime: CaptureRuntime,
    slot: CaptureSlot,
    schema: SamplerSchema,
    *,
    max_steps: int = 2000,
    moments: int | None = 8,
    omit_unsupported: bool = False,
    check: Callable[[], None] | None = None,
) -> CapturedGame:
    """Select moments without labels; advance both histories on every transition.

    The reservoir is uniform over exposed decision moments, including moments
    with omitted viewer rows. Each eligible row has inclusion probability
    min(1, moments / decisions). Private labels are read only after public input
    validation and reservoir admission. A failed game discards both viewers.
    Reset, malformed constraints and infrastructure errors propagate.
    """
    if max_steps < 1 or (moments is not None and moments < 1):
        raise ValueError("capture limits must be positive")
    if check is not None:
        check()
    runtime.reset(slot)
    histories = [
        ViewerHistory.from_observation(runtime.observation(v)) for v in range(2)
    ]
    # An independent RNG makes row selection invariant to policy and label RNGs.
    rng = random.Random(_digest({"slot": asdict(slot), "purpose": "moments"}))
    reservoir: list[tuple[SamplerExample, ...]] = []
    eligible = [0, 0]
    unsupported = [0, 0]
    commands: list[Command] = []
    overflows = dropped = decisions = 0
    failure: str | None = None
    try:
        for step in range(max_steps):
            if check is not None:
                check()
            decisions += 1
            index = (
                step if moments is None or step < moments else rng.randrange(step + 1)
            )
            selected = moments is None or index < moments
            inputs = []
            for viewer, history in enumerate(histories):
                try:
                    # CaptureRuntime supplies the same native projection method.
                    public, _ = public_sampler_input(runtime, history, schema)
                except UnsupportedSamplerRow:
                    if not omit_unsupported:
                        raise
                    unsupported[viewer] += 1
                    continue
                eligible[viewer] += 1
                if selected:
                    inputs.append((viewer, history, public))
            if selected:
                rows: list[SamplerExample] = []
                for viewer, history, public in inputs:
                    hand = runtime.hand_counts(1 - viewer)
                    if set(hand) - set(schema.card_names):
                        raise ValueError(
                            "private label is outside frozen sampler vocabulary"
                        )
                    target = tuple(hand.get(name, 0) for name in schema.card_names)
                    if sum(target) != public.hand_size or any(
                        type(count) is not int or count < low or count > high
                        for count, low, high in zip(
                            target, public.known_minima, public.pool_counts, strict=True
                        )
                    ):
                        raise ValueError(
                            "private label violates native hand constraints"
                        )
                    rows.append(
                        SamplerExample(
                            public,
                            target,
                            viewer,
                            history.current_revision,
                            history.current_viewer_state_hash,
                        )
                    )
                if index == len(reservoir):
                    reservoir.append(tuple(rows))
                else:
                    reservoir[index] = tuple(rows)
            frame = runtime.frame()
            action = runtime.act()
            if action < 0 or action >= len(frame.offers):
                raise ValueError("frozen policy selected an unavailable offer")
            # The runtime validates encoded support before returning an action.
            retained = runtime.retained_actions
            if len(frame.offers) > retained:
                overflows += 1
                dropped += len(frame.offers) - retained
            command = Command(
                f"sampler-{slot.game_id}-{step}",
                frame.revision,
                int(frame.offers[action]["id"]),
            )
            commands.append(command)  # Include a failed command in the private prefix.
            transition = runtime.step(command)
            histories = [
                history.advance(
                    transition.receipt, runtime.observation(viewer), acting=frame.actor
                )
                for viewer, history in enumerate(histories)
            ]
            if transition.truncated:
                raise GameCaptureFailure("belief self-play game truncated")
            if transition.terminated:
                break
        else:
            raise GameCaptureFailure("belief self-play exceeded the declared step cap")
    except GameCaptureFailure as error:
        failure = str(error)
    coverage = CaptureCoverage(
        decisions,
        (eligible[0], eligible[1]),
        (unsupported[0], unsupported[1]),
        len(reservoir),
        overflows,
        dropped,
    )
    rows = tuple(row for moment in reservoir for row in moment)
    game = (
        None
        if failure is not None or not rows
        else SamplerGame(slot.game_id, slot.seed, slot.assignment, slot.split, rows)
    )
    return CapturedGame(game, coverage, tuple(commands), failure)


def _write_private(path: Path, payload: object) -> None:
    """Atomically create immutable mode-0600 JSON, never replacing evidence."""
    descriptor, temporary = tempfile.mkstemp(prefix=".capture-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump(payload, output, sort_keys=True, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def collect_schedule(
    runtime: CaptureRuntime,
    slots: Sequence[CaptureSlot],
    schema: SamplerSchema,
    *,
    directory: Path,
    policy_identity: str,
    world_identity: str,
    max_steps: int = 2000,
    moments: int = 8,
    check: Callable[[], None] | None = None,
) -> None:
    """Persist each slot before reset and settle before starting the next one.

    Continuation never retries a started slot. Missing results become interrupted;
    untouched slots remain unattempted in schedule.json. Fatal exceptions leave a
    durable started receipt and propagate, stopping collection. Complete game
    shards precede terminal receipts; a crash between them stays interrupted.
    """
    if not slots or not policy_identity or not world_identity:
        raise ValueError("scheduled capture needs slots and producer/world identities")
    if len({s.game_id for s in slots}) != len(slots) or len(
        {(s.seed, s.assignment) for s in slots}
    ) != len(slots):
        raise ValueError("duplicate scheduled game identity")
    splits: dict[int, Split] = {}
    for slot in slots:
        if splits.setdefault(slot.seed, slot.split) != slot.split:
            raise ValueError("paired deal seeds cannot cross splits")
    directory.mkdir(parents=True, exist_ok=True)
    binding = {
        "schema": asdict(schema),
        "policy_identity": policy_identity,
        "world_identity": world_identity,
        "slots": [asdict(s) for s in slots],
        "max_steps": max_steps,
        "moments": moments,
    }
    identity = _digest(binding)
    schedule = directory / "schedule.json"
    if schedule.exists():
        if json.loads(schedule.read_text()) != {
            "identity": identity,
            "capture": json.loads(json.dumps(binding)),
        }:
            raise ValueError("saved capture schedule or bindings differ")
    else:
        _write_private(schedule, {"identity": identity, "capture": binding})
    for ordinal, slot in enumerate(slots):
        prefix = directory / f"game-{ordinal:06d}"
        started = prefix.with_suffix(".started.json")
        settled = prefix.with_suffix(".result.json")
        if settled.exists():
            receipt = json.loads(settled.read_text())
            if receipt["schedule_identity"] != identity:
                raise ValueError("game receipt belongs to another schedule")
            if receipt["status"] not in ("completed", "failed", "interrupted"):
                raise ValueError("unknown capture receipt status")
            if receipt["status"] != "interrupted":
                shard = prefix.with_suffix(".game.json").read_bytes()
                if _digest(json.loads(shard)) != receipt["shard_identity"]:
                    raise ValueError("settled game shard is corrupt")
            continue
        if started.exists():
            receipt = json.loads(started.read_text())
            if receipt != {
                "schedule_identity": identity,
                "slot": json.loads(json.dumps(asdict(slot))),
            }:
                raise ValueError("started game belongs to another schedule")
            _write_private(
                settled, {"schedule_identity": identity, "status": "interrupted"}
            )
            continue
        if check is not None:
            check()
        _write_private(started, {"schedule_identity": identity, "slot": asdict(slot)})
        result = capture_game(
            runtime,
            slot,
            schema,
            max_steps=max_steps,
            moments=moments,
            omit_unsupported=True,
            check=check,
        )
        shard = asdict(result)
        _write_private(prefix.with_suffix(".game.json"), shard)
        status: Literal["completed", "failed"] = (
            "failed" if result.failure else "completed"
        )
        _write_private(
            settled,
            {
                "schedule_identity": identity,
                "status": status,
                "shard_identity": _digest(shard),
            },
        )


def read_capture_dataset(directory: Path) -> SamplerDataset:
    """Assemble verified completed shards; missing split data refuses fitting.

    The schedule and all attempt receipts remain beside this dataset, including
    completed games with zero representable rows and failed/interrupted slots.
    """
    saved = json.loads((directory / "schedule.json").read_text())
    binding = saved["capture"]
    identity = saved["identity"]
    if _digest(binding) != identity:
        raise ValueError("capture schedule content digest mismatch")
    schema = SamplerSchema(
        **{**binding["schema"], "card_names": tuple(binding["schema"]["card_names"])}
    )
    games: list[SamplerGame] = []
    for ordinal, slot in enumerate(binding["slots"]):
        prefix = directory / f"game-{ordinal:06d}"
        result = prefix.with_suffix(".result.json")
        if not result.exists():
            continue
        receipt = json.loads(result.read_text())
        if receipt["schedule_identity"] != identity:
            raise ValueError("game receipt belongs to another schedule")
        if receipt["status"] not in ("completed", "failed", "interrupted"):
            raise ValueError("unknown capture receipt status")
        if receipt["status"] == "interrupted":
            continue
        shard = json.loads(prefix.with_suffix(".game.json").read_text())
        if _digest(shard) != receipt["shard_identity"]:
            raise ValueError("game shard content digest mismatch")
        if receipt["status"] == "completed" and shard["game"] is not None:
            game = _read_game(shard["game"])
            if (game.game_id, game.seed, game.assignment, game.split) != (
                slot["game_id"],
                slot["seed"],
                slot["assignment"],
                slot["split"],
            ):
                raise ValueError("game shard differs from scheduled slot")
            games.append(game)
    return SamplerDataset(
        schema, binding["policy_identity"], binding["world_identity"], tuple(games)
    )
