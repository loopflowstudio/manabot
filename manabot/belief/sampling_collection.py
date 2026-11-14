"""Scheduled frozen-policy capture with durable, private game evidence.

A runtime owns policy/native APIs; this module owns row selection, viewer history
and attempt accounting. Only completed games supply fitting rows. Receipts and
shards contain private seeds and replay commands and must stay private.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
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
    _validate_example,
    public_sampler_input,
    read_dataset,
    save_dataset,
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
    panel: Literal["same-field", "reserved"] | None = None

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
    omission_reasons: dict[str, tuple[int, int]]
    retained_rows: tuple[int, int]

    def __post_init__(self) -> None:
        counts = (
            self.decisions,
            self.selected_moments,
            self.overflow_decisions,
            self.dropped_actions,
            *self.eligible_rows,
            *self.unsupported_rows,
            *self.retained_rows,
            *(n for pair in self.omission_reasons.values() for n in pair),
        )
        if any(type(n) is not int or n < 0 for n in counts):
            raise ValueError("capture coverage counts must be nonnegative integers")
        if (
            self.selected_moments > self.decisions
            or self.overflow_decisions > self.decisions
        ):
            raise ValueError("capture moment counts exceed game decisions")
        for viewer in (0, 1):
            if (
                self.eligible_rows[viewer] + self.unsupported_rows[viewer]
                != self.decisions
            ):
                raise ValueError(
                    "capture coverage must account for every viewer decision"
                )
            if self.retained_rows[viewer] > min(
                self.eligible_rows[viewer], self.selected_moments
            ):
                raise ValueError("retained rows exceed selected eligible moments")
            if any(
                pair[viewer] > self.unsupported_rows[viewer]
                for pair in self.omission_reasons.values()
            ):
                raise ValueError("omission counts exceed unsupported viewer rows")


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
    on_fatal: Callable[[BaseException, tuple[Command, ...]], None] | None = None,
) -> CapturedGame:
    """Select moments without labels; advance both histories on every transition.

    The reservoir is uniform over exposed decision moments, including moments
    with omitted viewer rows. Each eligible row has inclusion probability
    min(1, moments / decisions). Private labels are read only after public input
    validation and reservoir admission. A failed game discards both viewers.
    Reset, malformed constraints and infrastructure errors propagate.
    """
    commands: list[Command] = []
    try:
        return _capture_game(
            runtime,
            slot,
            schema,
            max_steps=max_steps,
            moments=moments,
            omit_unsupported=omit_unsupported,
            check=check,
            commands=commands,
        )
    except BaseException as error:
        if on_fatal is not None:
            on_fatal(error, tuple(commands))
        raise


class CaptureSource(Protocol):
    """Viewer projection and private supervision, without policy or step ownership."""

    def hidden_hand_constraints_json(self, viewer: int) -> str: ...
    def hand_counts(self, player: int) -> Mapping[str, int]: ...


class GameAccumulator:
    """Passive, bounded both-viewer capture shared by frozen and training producers.

    Feed an exposed decision before acting and every resulting viewer transition
    before reset, including built-in continuation. This object never acts, steps,
    resets, publishes or touches a policy RNG. Reconstruction can replay into it
    without giving it a publication sink.
    """

    def __init__(
        self,
        slot: CaptureSlot,
        schema: SamplerSchema,
        observations: tuple[Observation, Observation],
        *,
        moments: int | None = 8,
        omit_unsupported: bool = False,
    ) -> None:
        if moments is not None and moments < 1:
            raise ValueError("capture limits must be positive")
        if tuple(o.viewer for o in observations) != (0, 1):
            raise ValueError("capture requires ordered observations for both viewers")
        self.slot = slot
        self.schema = schema
        self.histories = [ViewerHistory.from_observation(o) for o in observations]
        self.moments = moments
        self.omit_unsupported = omit_unsupported
        self.rng = random.Random(_digest({"slot": asdict(slot), "purpose": "moments"}))
        self.reservoir: list[tuple[SamplerExample, ...]] = []
        self.eligible = [0, 0]
        self.unsupported = [0, 0]
        self.omissions: dict[str, list[int]] = {}
        self.overflows = self.dropped = self.decisions = 0

    def capture_decision(self, source: CaptureSource) -> None:
        """Validate both public inputs before reading any admitted private label."""
        self.decisions += 1
        index = (
            self.decisions - 1
            if self.moments is None or self.decisions <= self.moments
            else self.rng.randrange(self.decisions)
        )
        selected = self.moments is None or index < self.moments
        inputs = []
        for viewer, history in enumerate(self.histories):
            try:
                # CaptureRuntime supplies the same native projection method.
                public, _ = public_sampler_input(source, history, self.schema)
            except UnsupportedSamplerRow as error:
                if not self.omit_unsupported:
                    raise
                self.unsupported[viewer] += 1
                for reason in error.reasons:
                    self.omissions.setdefault(reason, [0, 0])[viewer] += 1
                continue
            self.eligible[viewer] += 1
            if selected:
                inputs.append((viewer, history, public))
        if selected:
            rows: list[SamplerExample] = []
            for viewer, history, public in inputs:
                hand = source.hand_counts(1 - viewer)
                if set(hand) - set(self.schema.card_names):
                    raise ValueError(
                        "private label is outside frozen sampler vocabulary"
                    )
                target = tuple(hand.get(name, 0) for name in self.schema.card_names)
                row = SamplerExample(
                    public,
                    target,
                    viewer,
                    history.current_revision,
                    history.current_viewer_state_hash,
                )
                _validate_example(row, self.schema)
                rows.append(row)
            if index == len(self.reservoir):
                self.reservoir.append(tuple(rows))
            else:
                self.reservoir[index] = tuple(rows)

    def record_offer(self, legal: int, retained: int) -> None:
        if not 0 < retained <= legal:
            raise ValueError("capture offer support is invalid")
        if legal > retained:
            self.overflows += 1
            self.dropped += legal - retained

    def advance(
        self,
        receipt: TransitionReceipt,
        observations: tuple[Observation, Observation],
        *,
        acting: int,
    ) -> None:
        self.histories = [
            history.advance(receipt, observations[viewer], acting=acting)
            for viewer, history in enumerate(self.histories)
        ]

    def finish(
        self,
        commands: tuple[Command, ...],
        *,
        terminated: bool,
        failure: str | None = None,
    ) -> CapturedGame:
        if failure is None and not terminated:
            raise ValueError(
                "capture needs successful native termination before closure"
            )
        rows = tuple(row for moment in self.reservoir for row in moment)
        coverage = CaptureCoverage(
            self.decisions,
            (self.eligible[0], self.eligible[1]),
            (self.unsupported[0], self.unsupported[1]),
            len(self.reservoir),
            self.overflows,
            self.dropped,
            {name: (counts[0], counts[1]) for name, counts in self.omissions.items()},
            (sum(row.viewer == 0 for row in rows), sum(row.viewer == 1 for row in rows))
            if failure is None
            else (0, 0),
        )
        game = (
            None
            if failure is not None
            else SamplerGame(
                self.slot.game_id,
                self.slot.seed,
                self.slot.assignment,
                self.slot.split,
                rows,
            )
        )
        return CapturedGame(game, coverage, commands, failure)


def _capture_game(
    runtime: CaptureRuntime,
    slot: CaptureSlot,
    schema: SamplerSchema,
    *,
    max_steps: int,
    moments: int | None,
    omit_unsupported: bool,
    check: Callable[[], None] | None,
    commands: list[Command],
) -> CapturedGame:
    if max_steps < 1:
        raise ValueError("capture limits must be positive")
    if check is not None:
        check()
    runtime.reset(slot)
    accumulator = GameAccumulator(
        slot,
        schema,
        (runtime.observation(0), runtime.observation(1)),
        moments=moments,
        omit_unsupported=omit_unsupported,
    )
    failure: str | None = None
    terminated = False
    try:
        for step in range(max_steps):
            if check is not None:
                check()
            accumulator.capture_decision(runtime)
            frame = runtime.frame()
            action = runtime.act()
            if action < 0 or action >= len(frame.offers):
                raise ValueError("frozen policy selected an unavailable offer")
            accumulator.record_offer(len(frame.offers), runtime.retained_actions)
            command = Command(
                f"sampler-{slot.game_id}-{step}",
                frame.revision,
                int(frame.offers[action]["id"]),
            )
            commands.append(command)
            transition = runtime.step(command)
            accumulator.advance(
                transition.receipt,
                (runtime.observation(0), runtime.observation(1)),
                acting=frame.actor,
            )
            if transition.truncated:
                raise GameCaptureFailure("belief self-play game truncated")
            if transition.terminated:
                terminated = True
                break
        else:
            raise GameCaptureFailure("belief self-play exceeded the declared step cap")
    except GameCaptureFailure as error:
        failure = str(error)
    return accumulator.finish(tuple(commands), terminated=terminated, failure=failure)


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
    ordinals: Sequence[int] | None = None,
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
    if (directory / "closure.json").exists():
        raise ValueError("collection is closed")
    for stop in directory.glob("game-*.stop.json"):
        restored = stop.with_suffix(".restored.json")
        if not restored.exists() or json.loads(restored.read_text()).get(
            "stop_identity"
        ) != _digest(json.loads(stop.read_text())):
            raise ValueError("fatal capture prerequisite has not been restored")
    selected = tuple(range(len(slots))) if ordinals is None else tuple(ordinals)
    if tuple(sorted(set(selected))) != selected or any(
        ordinal < 0 or ordinal >= len(slots) for ordinal in selected
    ):
        raise ValueError("capture ordinals must be distinct scheduled slots in order")
    for ordinal in selected:
        slot = slots[ordinal]
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

        def retain_fatal(error: BaseException, commands: tuple[Command, ...]) -> None:
            _write_private(
                prefix.with_suffix(".stop.json"),
                {
                    "schedule_identity": identity,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "cause_type": None
                    if error.__cause__ is None
                    else type(error.__cause__).__name__,
                    "cause": None if error.__cause__ is None else str(error.__cause__),
                    "commands": [asdict(c) for c in commands],
                },
            )

        result = capture_game(
            runtime,
            slot,
            schema,
            max_steps=max_steps,
            moments=moments,
            omit_unsupported=True,
            check=check,
            on_fatal=retain_fatal,
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
    """Inspect verified completed shards; require three nonempty splits.

    The schedule and all attempt receipts remain beside this dataset, including
    completed games with zero representable rows and failed/interrupted slots.
    Missing results are skipped. This reader does not establish collection closure
    or exclude a live writer; the experiment must establish both before fitting.
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
            if game.examples:
                games.append(game)
    return SamplerDataset(
        schema, binding["policy_identity"], binding["world_identity"], tuple(games)
    )


def verify_closed_capture(directory: Path) -> str:
    closure = directory / "closure.json"
    payload = json.loads(closure.read_text())
    for name, identity in payload["sources"].items():
        if Path(name).name != name or _file_identity(directory / name) != identity:
            raise ValueError("closed collection evidence changed")
    current = {p.name for p in directory.glob("game-*.json")} | {"schedule.json"}
    if current != set(payload["sources"]):
        raise ValueError("closed collection gained or lost source evidence")
    return _file_identity(closure)


def read_closed_dataset(directory: Path) -> SamplerDataset:
    """Admit frozen bytes and their closed sources before fitting or reporting."""
    dataset = read_dataset(directory / "dataset.json")
    if dataset.binding is not None:
        return dataset
    binding = json.loads((directory / "dataset-binding.json").read_text())
    if (
        verify_closed_capture(directory / "main") != binding["closure_identity"]
        or _file_identity(directory / "dataset.json") != binding["dataset_sha256"]
    ):
        raise ValueError("frozen dataset binding changed")
    dataset = read_dataset(directory / "dataset.json")
    if dataset.identity != binding["dataset_identity"]:
        raise ValueError("frozen dataset identity changed")
    return dataset


def close_capture_collection(directory: Path, *, reason: str) -> None:
    """Seal finite schedule accounting after Etude has stopped its writer.

    This function owns only data closure; it never infers worker liveness or cost.
    Started or orphaned slots stay interrupted and fatal prefixes stay private.
    """
    if not reason.strip():
        raise ValueError("closure requires a stop reason")
    closure = directory / "closure.json"
    if closure.exists():
        verify_closed_capture(directory)
        return
    schedule = json.loads((directory / "schedule.json").read_text())
    if _digest(schedule["capture"]) != schedule["identity"]:
        raise ValueError("capture schedule digest mismatch")
    statuses: list[str] = []
    for ordinal, _ in enumerate(schedule["capture"]["slots"]):
        prefix = directory / f"game-{ordinal:06d}"
        result = prefix.with_suffix(".result.json")
        started = prefix.with_suffix(".started.json")
        if result.exists():
            receipt = json.loads(result.read_text())
            if receipt["schedule_identity"] != schedule["identity"] or receipt[
                "status"
            ] not in ("completed", "failed", "interrupted"):
                raise ValueError("invalid terminal game receipt")
            status = receipt["status"]
            if not started.exists():
                raise ValueError("terminal game has no started receipt")
            if status != "interrupted":
                shard = json.loads(prefix.with_suffix(".game.json").read_text())
                if _digest(shard) != receipt["shard_identity"]:
                    raise ValueError("terminal game shard digest mismatch")
                if (status == "failed") != (shard["failure"] is not None):
                    raise ValueError("terminal status disagrees with game failure")
        else:
            status = (
                "interrupted"
                if started.exists() or prefix.with_suffix(".game.json").exists()
                else "unattempted"
            )
        if started.exists() and json.loads(started.read_text()) != {
            "schedule_identity": schedule["identity"],
            "slot": schedule["capture"]["slots"][ordinal],
        }:
            raise ValueError("started game differs from schedule")
        statuses.append(status)
    sources = {
        path.name: _file_identity(path) for path in sorted(directory.glob("*.json"))
    }
    _write_private(
        closure,
        {
            "schedule_identity": schedule["identity"],
            "reason": reason,
            "statuses": statuses,
            "sources": sources,
        },
    )


def _file_identity(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def freeze_capture_dataset(directory: Path) -> SamplerDataset:
    """Freeze a closed finite collection without changing historical receipts."""
    closure_identity = verify_closed_capture(directory / "main")
    dataset = read_capture_dataset(directory / "main")
    path = directory / "dataset.json"
    if path.exists():
        if read_dataset(path) != dataset:
            raise ValueError("frozen dataset differs from closed collection")
    else:
        save_dataset(dataset, path)
    binding = {
        "closure_identity": closure_identity,
        "dataset_identity": dataset.identity,
        "dataset_sha256": _file_identity(path),
    }
    frozen = directory / "dataset-binding.json"
    if frozen.exists():
        if json.loads(frozen.read_text()) != binding:
            raise ValueError("frozen dataset binding changed")
    else:
        _write_private(frozen, binding)
    return dataset


def record_capture_repair(directory: Path, *, reason: str) -> None:
    """Retain a repaired prerequisite after Etude has stopped the old writer.

    This acknowledges a cause, not a retry: started slots remain consumed and
    `collect_schedule` may continue only untouched slots under the same schedule.
    The original fatal diagnostics and replay prefix are never replaced.
    """
    if not reason.strip():
        raise ValueError("restoration requires the repaired prerequisite")
    if (directory / "closure.json").exists():
        raise ValueError("collection is closed")
    for stop in sorted(directory.glob("game-*.stop.json")):
        restored = stop.with_suffix(".restored.json")
        identity = _digest(json.loads(stop.read_text()))
        if restored.exists():
            if json.loads(restored.read_text())["stop_identity"] != identity:
                raise ValueError(
                    "fatal evidence changed after prerequisite restoration"
                )
        else:
            _write_private(restored, {"stop_identity": identity, "reason": reason})
