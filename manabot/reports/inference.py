"""Read historical LLM evidence without importing a model or native runtime.

Receipt projections are read-only: the selection/Command files remain the
source of truth. Inventory digests bind copied evidence and corpus order; they
are not new inference results. Historical v1 timings include IPC and encoding.
"""

from __future__ import annotations

from collections.abc import Callable
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Count = Annotated[int, Field(ge=0, strict=True)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ObservedStatus = Literal["completed", "failed", "partial", "unknown"]


class _HistoricalRecord(BaseModel):
    # Only consumed fields are interpreted; unprojected historical fields remain
    # verbatim in evidence. This is not admission to the current executable ABI.
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)


class _Tokens(_HistoricalRecord):
    text: str
    token_ids: list[Count]


class _Generation(_HistoricalRecord):
    prompt: _Tokens | None = None
    response: _Tokens | None = None
    prompt_tokens: Count | None = None
    output_tokens: Count | None = None
    validation_error: str | None = None
    seconds: Seconds | None = None
    encode_seconds: Seconds | None = None
    generation_seconds: Seconds | None = None
    validation_seconds: Seconds | None = None

    @model_validator(mode="after")
    def counts_match(self) -> _Generation:
        for tokens, count in (
            (self.prompt, self.prompt_tokens),
            (self.response, self.output_tokens),
        ):
            if (
                tokens is not None
                and count is not None
                and len(tokens.token_ids) != count
            ):
                raise ValueError("token count does not match retained IDs")
        return self


class _Decision(_HistoricalRecord):
    text: str
    legal_references: list[str] = Field(min_length=1)
    representation_id: str
    source_identity: dict[str, str]


class _InferenceIdentity(_HistoricalRecord):
    model_id: str = Field(min_length=1)
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    tokenizer_id: str = Field(min_length=1)
    tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    device: Literal["cpu", "cuda"]
    precision: Literal["float32", "float16", "bfloat16"]
    context_tokens: Count
    max_new_tokens: Count
    deadline_seconds: Seconds
    fallback_seed: int
    decoding: Literal["greedy"]
    enable_thinking: bool | None = None


class _Selection(_HistoricalRecord):
    version: Literal["llm-decision-v1"]
    decision: _Decision
    config: _InferenceIdentity
    runtime: dict[str, JsonValue] | None = None
    attempts: list[_Generation]
    selected_reference: str | None
    origin: Literal["initial", "retry", "fallback"] | None
    failure: dict[str, JsonValue] | None
    seconds: Seconds

    @model_validator(mode="after")
    def legal_result(self) -> _Selection:
        if self.selected_reference is not None:
            if (
                self.selected_reference not in self.decision.legal_references
                or self.origin is None
                or self.failure is not None
            ):
                raise ValueError("inconsistent selection result")
        elif self.origin is not None:
            raise ValueError("selection origin without a reference")
        return self


class _Summary(_HistoricalRecord):
    completed: bool
    failure: str | None
    decisions: Count
    model_decisions: Count
    game_seconds: Seconds
    model_load_seconds: Seconds | None = None
    peak_gpu_bytes: Count | None = None
    replay_matches: bool
    decision_seconds_median: Seconds | None = None
    observations: dict[str, Seconds] = Field(default_factory=dict)
    mode: str | None = None
    winner: int | None = None
    opponent_latencies: list[Seconds] = Field(default_factory=list)
    reasoning_tokens: Count = 0
    answer_tokens: Count = 0
    output_tokens: Count = 0
    token_split_attempts: Count = 0
    generation_attempts: Count = 0
    llm_phases: dict[str, Seconds] = Field(default_factory=dict)


class _Command(_HistoricalRecord):
    command_id: str
    expected_revision: Count


class _TransitionReceipt(_HistoricalRecord):
    command_id: str
    before_revision: Count
    after_revision: Count


class _Transition(_HistoricalRecord):
    receipt: _TransitionReceipt


class _CacheFile(_HistoricalRecord):
    name: str
    bytes: Count
    sha256: Digest


class _Cache(_HistoricalRecord):
    acquisition_seconds: Seconds | None = None
    verification_seconds: Seconds | None = None
    seconds: Seconds
    files: list[_CacheFile]


class EvidenceFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str
    sha256: Digest
    bytes: Count


class EvidenceDependency(TypedDict):
    path: str
    sha256: str | None
    bytes: int | None
    status: Literal["available", "unavailable"]
    reason: str | None
    href: NotRequired[str | None]


class EvidenceReader:
    """Read digest-bound snapshots and expose every dependency, even absent ones.

    With pins, undeclared files are never consumed unless allow_unpinned is set
    for direct inputs. Those dependencies acquire read-time digests for copying.
    """

    def __init__(
        self,
        pins: dict[Path, EvidenceFile] | None = None,
        *,
        allow_unpinned: bool = False,
    ) -> None:
        self.pins = pins or {}
        self.allow_unpinned = pins is None or allow_unpinned
        self.files: dict[Path, EvidenceDependency] = {}
        self._data: dict[Path, bytes | None] = {}

    def read(self, path: Path) -> bytes | None:
        path = path.resolve()
        if path in self._data:
            return self._data[path]
        expected = self.pins.get(path)
        reason = None
        if expected is None and not self.allow_unpinned:
            reason = "dependency is not pinned in baseline"
        elif not path.is_file():
            reason = "dependency file is missing"
        data = path.read_bytes() if reason is None else None
        actual = (
            EvidenceFile(
                path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)
            )
            if data is not None
            else None
        )
        if (
            expected
            and actual
            and (expected.sha256, expected.bytes) != (actual.sha256, actual.bytes)
        ):
            raise ValueError(f"baseline dependency digest/size mismatch: {path}")
        file = actual or expected
        self.files[path] = {
            "sha256": file.sha256 if file else None,
            "bytes": file.bytes if file else None,
            "path": str(path),
            "status": "available" if data is not None else "unavailable",
            "reason": reason,
        }
        self._data[path] = data
        return data

    def json(self, path: Path) -> dict[str, Any] | None:
        data = self.read(path)
        if data is None:
            return None
        value = json.loads(data)
        if not isinstance(value, dict):
            raise ValueError(f"expected evidence object: {path}")
        return value


class BaselineAttempt(BaseModel):
    """One declared attempt, including cells that have no received evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    kind: Literal["llm", "evaluation", "retained-input"]
    cohort: str
    repeat: Count
    mode: Literal["ordinary", "diagnostic"] = "ordinary"
    status: Literal[
        "planned", "attempted", "completed", "failed", "unknown", "unattempted"
    ]
    reason: str
    root: str | None = None
    files: list[EvidenceFile] = Field(default_factory=list)
    # Interrupted collection retains only received pins. Unpinned raw files
    # remain on disk but cannot contribute measurements to a manifest report.
    inventory_complete: bool = True
    corpus: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def evidence_has_root(self) -> BaselineAttempt:
        if self.root is None and (self.files or self.corpus):
            raise ValueError("evidence references require an attempt root")
        if not self.inventory_complete and self.status == "completed":
            raise ValueError("completed attempt requires a complete inventory")
        return self

    def validate_inventory(self, actual: list[EvidenceFile]) -> None:
        """Check the projection's inventory; absent directories are not projected."""
        if not self.files:
            raise ValueError("baseline receipt requires a pinned file inventory")
        if not self.inventory_complete:
            pinned = {f.path for f in self.files}
            actual = [f for f in actual if f.path in pinned]
        if sorted(actual, key=lambda f: f.path) != sorted(
            self.files, key=lambda f: f.path
        ):
            raise ValueError(f"baseline digest/size/inventory mismatch: {self.id}")
        names = {f.path for f in actual}
        if self.kind == "llm" and self.corpus != sorted(
            name for name in names if name.endswith("/selection.json")
        ):
            raise ValueError("LLM corpus must retain all selections in ordinal order")
        if len(set(self.corpus)) != len(self.corpus) or not set(self.corpus) <= names:
            raise ValueError("invalid baseline corpus references")


class BaselineAllocation(BaseModel):
    """One shared allocation, including its external diagnostic evidence.

    File paths are relative to root (absolute paths are also supported for
    historical trace references). Identity survives temporarily absent receipts.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    identity: str = Field(min_length=1)
    root: str
    files: list[EvidenceFile]


class BaselineManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["inference-baseline-v1"]
    source: str
    protocol: EvidenceFile | None = None
    attempts: list[BaselineAttempt]
    allocations: list[BaselineAllocation] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_attempts(self) -> BaselineManifest:
        ids = [attempt.id for attempt in self.attempts]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate baseline attempt")
        allocation_ids = [allocation.identity for allocation in self.allocations]
        if len(allocation_ids) != len(set(allocation_ids)):
            raise ValueError("duplicate baseline allocation")
        return self


class AttemptObservation(_HistoricalRecord):
    version: Literal["inference-attempt-v1"]
    id: str
    protocol_sha256: Digest
    status: Literal["completed", "failed"]
    reason: str
    whole_attempt_seconds: Seconds
    collection_seconds: Seconds
    boundary: str
    supervision: dict[str, JsonValue]


@dataclass
class AttemptProjection:
    """Join a declaration to observations without changing its retained status.

    A missing source has no run, even when the declaration records completion.
    Nonterminal declarations may acquire observations as evidence is restored.
    """

    declaration: BaselineAttempt
    source: Path | None
    run: int | None = None
    observed_status: ObservedStatus = "unknown"
    execution: AttemptObservation | None = None

    def bind(self, run: int | None, observed_status: ObservedStatus) -> None:
        declared = self.declaration.status
        if {declared, observed_status} == {"completed", "failed"}:
            raise ValueError(
                f"attempt {self.declaration.id} source {self.source}: "
                f"declared {declared} contradicts observed {observed_status}"
            )
        self.run = run
        self.observed_status = observed_status

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            **self.declaration.model_dump(exclude={"files", "corpus", "status"}),
            "declared_status": self.declaration.status,
            "evidence_status": "available"
            if self.run is not None or self.execution is not None
            else "unavailable",
            "observed_status": self.observed_status,
            "run": self.run,
            "execution": self.execution.model_dump(mode="json")
            if self.execution
            else None,
        }


@dataclass(frozen=True)
class Measurement:
    value: float | int | None
    unit: str
    status: Literal["recorded", "derived", "unavailable", "not_applicable"]
    resource: str
    boundary: str
    coverage: str
    reason: str


@dataclass(frozen=True)
class Distribution:
    n: int
    population: int
    p50: float | None
    p95: float | None
    maximum: float | None
    method: str = (
        "sorted[floor((n-1)*q)]; seconds; retained observations, not random sampling"
    )


def _distribution(values: list[float], population: int) -> Distribution:
    values = sorted(values)
    return Distribution(
        len(values),
        population,
        values[math.floor((len(values) - 1) * 0.5)] if values else None,
        values[math.floor((len(values) - 1) * 0.95)] if values else None,
        values[-1] if values else None,
    )


def _selection_latencies(
    selections: list[_Selection], population: int
) -> dict[str, dict[str, JsonValue]]:
    """Use the same population rules for a game and a sequential session.

    Selection success, selection failure, and generation attempts are distinct;
    generation host failures include unfinished calls with no retained response.
    """
    successful = [s for s in selections if s.selected_reference is not None]
    failed = [s for s in selections if s.failure is not None]
    generations = [a for s in selections for a in s.attempts]
    return {
        "selection_latency": asdict(
            _distribution(
                [s.seconds for s in successful],
                population,
            )
        ),
        "failed_selection_latency": asdict(
            _distribution([s.seconds for s in failed], len(failed))
        ),
        "generation_latency": asdict(
            _distribution(
                [a.seconds for a in generations if a.seconds is not None],
                len(generations),
            )
        ),
        "generation_host_latency": asdict(
            _distribution(
                [
                    a.generation_seconds
                    for a in generations
                    if a.generation_seconds is not None and a.response is not None
                ],
                sum(a.response is not None for a in generations),
            )
        ),
        "failed_generation_host_latency": asdict(
            _distribution(
                [
                    a.generation_seconds
                    for a in generations
                    if a.generation_seconds is not None and a.response is None
                ],
                sum(a.response is None for a in generations),
            )
        ),
    }


def _object(path: Path) -> dict[str, JsonValue]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def pin_file(path: Path, *, relative_to: Path | None = None) -> EvidenceFile:
    """Bind a file's bytes, using an absolute path unless an inventory root is given."""
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return EvidenceFile(
        path=path.relative_to(relative_to).as_posix()
        if relative_to is not None
        else str(path.resolve()),
        sha256=digest,
        bytes=path.stat().st_size,
    )


def inventory(
    root: Path, *, on_file: Callable[[EvidenceFile], None] | None = None
) -> list[EvidenceFile]:
    """Bind raw receipts, optionally retaining each pin as hashing progresses."""
    paths = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and (
            p.suffix in {".json", ".txt", ".csv", ".npz", ".jsonl"}
            or p.name.endswith(".jsonl.gz")
        )
    ]
    if any(not p.resolve().is_relative_to(root.resolve()) for p in paths):
        raise ValueError("evidence symlink escapes receipt directory")
    files = []
    for p in sorted(paths):
        file = pin_file(p, relative_to=root)
        files.append(file)
        if on_file is not None:
            on_file(file)
    return files


def evidence_identity(files: list[EvidenceFile]) -> str:
    return hashlib.sha256(
        json.dumps([f.model_dump() for f in files], sort_keys=True).encode()
    ).hexdigest()


def read_manifest(path: Path) -> BaselineManifest:
    """Read declarations; the builder validates each available input's inventory."""
    manifest = BaselineManifest.model_validate_json(path.read_text())
    if manifest.protocol is not None:
        protocol = path.parent / manifest.protocol.path
        if manifest.protocol.sha256 != manifest.source:
            raise ValueError("protocol digest differs from manifest source")
        if protocol.exists():
            actual = pin_file(protocol)
            if (actual.bytes, actual.sha256) != (
                manifest.protocol.bytes,
                manifest.protocol.sha256,
            ):
                raise ValueError("frozen protocol digest/size mismatch")
    roots = [attempt.root for attempt in manifest.attempts if attempt.root is not None]
    if len(set((path.parent / root).resolve() for root in roots)) != len(roots):
        raise ValueError("duplicate baseline evidence root")
    return manifest


def project_llm(root: Path) -> dict[str, Any]:
    """Project one historical sequential game, retaining partial/failure coverage.

    Successful selections can precede a failed game. Their latency population is
    separate from failed selections and generation attempts (including repairs).
    A written Command without a transition is never counted as committed.
    """
    if not root.is_dir():
        raise ValueError(f"LLM receipt directory unavailable: {root}")
    if (root / "games.json").exists():
        return _project_llm_sequence(root)
    game = root / "game" if (root / "game").is_dir() else root
    files = inventory(root)
    summary_path = game / "summary.json"
    summary = (
        _Summary.model_validate(_object(summary_path))
        if summary_path.exists()
        else None
    )
    progress_path = game / "progress.json"
    progress = (
        _Summary.model_validate(_object(progress_path))
        if progress_path.exists()
        else None
    )
    observed = summary or progress
    config_path = game / "config.json"
    config = _object(config_path) if config_path.exists() else None
    runtime_path = game / "runtime.json"
    runtime = _object(runtime_path) if runtime_path.exists() else None
    selections = [
        (p, _Selection.model_validate(_object(p)))
        for p in sorted(game.glob("decision-*/selection.json"))
    ]
    if not files:
        raise ValueError("empty LLM receipt directory")
    if not observed and not selections and config is None:
        raise ValueError("unrecognized LLM receipt directory")
    successful = [s for _, s in selections if s.selected_reference is not None]
    failed = [s for _, s in selections if s.failure is not None]
    generations = [a for _, s in selections for a in s.attempts]
    commands = sorted(game.glob("decision-*/command.json"))
    committed = 0
    for command in commands:
        bound = _Command.model_validate(_object(command))
        transition = command.with_name("transition.json")
        if transition.exists():
            result = _Transition.model_validate(_object(transition)).receipt
            if (
                result.command_id != bound.command_id
                or result.before_revision != bound.expected_revision
                or result.after_revision <= result.before_revision
            ):
                raise ValueError("Command/transition binding mismatch")
            committed += 1
    if observed and (
        len(selections) > observed.model_decisions + (1 if summary is None else 0)
        or committed > observed.decisions + (1 if summary is None else 0)
    ):
        raise ValueError("retained counts exceed summary population")
    for path, selection in selections:
        capture = path.with_name("capture.json")
        if capture.exists() and selection.decision.source_identity.get(
            "capture_digest"
        ) != _object(capture).get("digest"):
            raise ValueError("selection/capture digest mismatch")
    identity = selections[0][1].config if selections else None
    if any(selection.config != identity for _, selection in selections):
        raise ValueError("model configuration changed within one game")
    cache_path = root / "cache.json"
    cache = _Cache.model_validate(_object(cache_path)) if cache_path.exists() else None
    metrics: dict[str, Measurement] = {}

    def measured(
        name: str,
        value: float | int | None,
        unit: str,
        resource: str,
        boundary: str,
        coverage: str,
        *,
        derived: bool = False,
        reason: str = "Not retained in historical receipt",
    ) -> None:
        status: Literal["recorded", "derived", "unavailable"] = "unavailable"
        if value is not None:
            status = "derived" if derived else "recorded"
            if not derived:
                reason = "Read from retained evidence"
        metrics[name] = Measurement(
            value, unit, status, resource, boundary, coverage, reason
        )

    measured(
        "cache_preparation",
        cache.seconds if cache else None,
        "s",
        "preparation host",
        "acquisition plus verification, not network-only",
        "one preparation receipt",
    )
    for name in ("acquisition_seconds", "verification_seconds"):
        measured(
            "cache_" + name.removesuffix("_seconds"),
            getattr(cache, name) if cache else None,
            "s",
            "preparation host",
            name.removesuffix("_seconds") + " only; nested within cache preparation",
            "one preparation receipt",
        )
    for name, field in (
        ("tokenization", "encode_seconds"),
        ("generation_host", "generation_seconds"),
        ("response_validation", "validation_seconds"),
    ):
        values = [
            getattr(a, field) for a in generations if getattr(a, field) is not None
        ]
        measured(
            name,
            sum(values) if len(values) == len(generations) and values else None,
            "s",
            "worker host",
            "summed generation-attempt intervals, including failed calls; nested within selection",
            f"{len(values)}/{len(generations)} attempts",
        )
    for name, value, unit in [
        ("verified_cache_files", len(cache.files) if cache else None, "files"),
        (
            "verified_cache_bytes",
            sum(f.bytes for f in cache.files) if cache else None,
            "bytes",
        ),
    ]:
        measured(
            name,
            value,
            unit,
            "artifact cache",
            "listed verified model files; not downloaded bytes or cache hits",
            "one cache receipt",
        )
    prompts = [a.prompt_tokens for a in generations if a.prompt_tokens is not None]
    outputs = [a.output_tokens for a in generations if a.output_tokens is not None]
    for name, value, count, boundary in [
        (
            "minimum_prompt_tokens",
            min(prompts) if prompts else None,
            len(prompts),
            "smallest retained encoded prompt, including repairs",
        ),
        (
            "maximum_prompt_tokens",
            max(prompts) if prompts else None,
            len(prompts),
            "largest retained encoded prompt; not a context-capacity proof",
        ),
        (
            "output_tokens",
            sum(outputs) if outputs and len(outputs) == len(generations) else None,
            len(outputs),
            "all retained generation outputs, including invalid answers",
        ),
    ]:
        measured(
            name,
            value,
            "tokens",
            "tokenizer",
            boundary,
            f"{count}/{len(generations)} generation attempts",
            derived=True,
            reason="Reduced retained token counts",
        )
    measured(
        "worker_readiness",
        observed.model_load_seconds if observed else None,
        "s",
        "session parent",
        "spawned session start-to-ready, including imports/IPC; not isolated weight load",
        "one session",
    )
    measured(
        "game_wall",
        summary.game_seconds if summary else None,
        "s",
        "game host",
        "reset through selection, native steps, writes and immediate clone replay; excludes session construction and outer collection",
        "one game, including failed work",
    )
    measured(
        "selection_sum",
        sum(s.seconds for _, s in selections) if selections else None,
        "s",
        "sequential selection host",
        "selection includes encoding, generation and IPC; overlaps game wall",
        f"{len(selections)} retained selections",
        derived=True,
        reason="Sum of retained selection intervals, including failures",
    )
    measured(
        "first_selection",
        selections[0][1].seconds if selections else None,
        "s",
        "selection host",
        "first retained selection; also included in distribution",
        "one observed request; first issued request unknown if receipts missing",
    )
    measured(
        "historical_median",
        observed.decision_seconds_median if observed else None,
        "s",
        "selection host",
        "original historical median, not recomputed floor-index p50",
        "historical summary population",
    )
    measured(
        "process_peak_allocated",
        observed.peak_gpu_bytes if observed else None,
        "bytes",
        "Torch allocator",
        "reported allocated peak; excludes non-Torch device allocations",
        "summary observation",
    )
    wall = summary.game_seconds if summary else None
    complete_commands = summary is not None and committed == summary.decisions
    complete_selections = (
        summary is not None and len(selections) == summary.model_decisions
    )
    valid_game = bool(
        summary
        and summary.completed
        and summary.failure is None
        and summary.replay_matches
        and complete_commands
        and complete_selections
    )
    game_outcome_known = bool(
        summary
        and (
            summary.failure is not None
            or (summary.completed and complete_commands and complete_selections)
        )
    )
    for name, count, available in [
        ("committed_commands_per_game_second", committed, complete_commands),
        ("model_decisions_per_game_second", len(successful), complete_selections),
        ("replay_verified_games_per_game_second", int(valid_game), game_outcome_known),
    ]:
        measured(
            name,
            count / wall if wall and available else None,
            "1/s",
            "game host",
            "game wall including failed work; excludes startup and outer collection",
            "one game attempt",
            derived=True,
            reason="Count divided by game wall; incomplete evidence cannot establish a successful game",
        )
    attempt_path = root / "attempt.json"
    attempt = (
        AttemptObservation.model_validate(_object(attempt_path))
        if attempt_path.exists()
        else None
    )
    measured(
        "whole_attempt",
        attempt.whole_attempt_seconds if attempt else None,
        "s",
        "parent",
        attempt.boundary
        if attempt
        else "entry through preparation, cleanup and final output",
        "one attempt",
    )
    for name in ("render_capture", "native_command", "clone_replay", "evidence_write"):
        measured(
            name,
            summary.observations.get(name) if summary else None,
            "s",
            "game host",
            "summed host intervals, nested within game wall; final summary write excluded",
            "one game",
        )
    for name, unit, resource, boundary in [
        ("prefill", "s", "device", "prefill only"),
        ("decode", "s", "device", "decode only"),
        (
            "queue_delay",
            "s",
            "server host",
            "receipt-to-dispatch before model readiness",
        ),
        (
            "native_work",
            "s",
            "native host",
            "environment work excluding inference and writes",
        ),
        ("process_peak_reserved", "bytes", "Torch allocator", "reserved process peak"),
        (
            "inclusive_cost",
            "USD",
            "allocation",
            "setup, idle, failures, cleanup, storage/network",
        ),
        ("games_per_dollar", "games/USD", "allocation", "inclusive settled billing"),
        ("warmup", "requests", "session", "separately recorded warmup"),
        (
            "cache_hits",
            "files",
            "artifact cache",
            "observed pre-preparation cache hits",
        ),
    ]:
        measured(name, None, unit, resource, boundary, "unavailable")
    device = _device_memory(root / "gpu-memory.csv")
    measured(
        "device_peak_used",
        device["peak_used_bytes"],
        "bytes",
        "whole device",
        "sampled device usage, including other processes; not an allocator peak",
        f"{device['samples']} samples",
    )
    measured(
        "device_total",
        device["total_bytes"],
        "bytes",
        "whole device",
        "reported device capacity",
        f"{device['samples']} samples",
    )
    # Identity uses retained game inputs, not directory spelling or redundant
    # tar/outer summary copies. Exact copied imports cannot add a population.
    game_files = [
        f.model_copy(
            update={"path": str(Path(f.path).relative_to(game.relative_to(root)))}
        )
        for f in files
        if (root / f.path).is_relative_to(game) and Path(f.path).name != "progress.json"
    ]
    # A game outcome does not certify retained Commands or replay coverage.
    observed_status: ObservedStatus = "unknown"
    if summary:
        if summary.failure is not None:
            observed_status = "failed"
        elif summary.completed:
            observed_status = "completed"
    return {
        "identity": evidence_identity(game_files),
        "matchup": _matchup(summary, config, valid_game),
        "label": root.name,
        "observed_status": observed_status,
        "failure": summary.failure if summary else None,
        "coverage": {
            "summary": summary is not None,
            "selection_files": len(selections),
            "selection_population": observed.model_decisions if observed else None,
            "command_files": len(commands),
            "committed_commands": committed,
            "command_population": observed.decisions if observed else None,
            "completed_games": int(summary.completed) if summary else None,
            "replay_verified_games": int(valid_game) if game_outcome_known else None,
            "initial": sum(s.origin == "initial" for s in successful),
            "repaired": sum(s.origin == "retry" for s in successful),
            "fallback": sum(s.origin == "fallback" for s in successful),
            "failed_selections": len(failed),
            "unknown_selections": len(selections) - len(successful) - len(failed),
            "generation_attempts": len(generations),
        },
        **_selection_latencies(
            [s for _, s in selections],
            observed.model_decisions if observed else len(selections),
        ),
        "metrics": {k: asdict(v) for k, v in metrics.items()},
        "device_memory": device,
        "config": config,
        "inference_config": identity.model_dump() if identity else None,
        "runtime": runtime,
        "corpus": [str(p.relative_to(root)) for p, _ in selections],
        "evidence": [f.model_dump() for f in files],
        "notes": [
            "Historical evidence only; no fresh inference or native replay executed.",
            "No separately recorded warmup; first selection remains in the population.",
            "Request, generation-attempt and game populations differ; no pooling across workload/hardware cells.",
            "Game remainder is unattributed, not environment time. Startup is outside game wall.",
            "Missing inclusive billing prevents games/dollar; historical cost estimates are not bills.",
        ],
    }


class _GameSequence(_HistoricalRecord):
    version: Literal["llm-game-sequence-v1"]
    mode: Literal["sequential", "concurrent"] = "sequential"
    games: list[Annotated[str, Field(pattern=r"^game-[0-9]{3}$")]] = Field(min_length=1)


def _matchup(
    summary: _Summary | None, config: dict[str, JsonValue] | None, valid: bool
) -> dict[str, Any] | None:
    """Explicit same-game host accounting; never infer simulation from remainder."""
    if summary is None or summary.mode != "qwen-v-checkpoint" or config is None:
        return None
    won = valid and summary.winner == config.get("model_seat")
    draw = valid and summary.winner not in (0, 1)
    return {
        "games": 1,
        "valid": int(valid),
        "wins": int(won),
        "draws": int(draw),
        "losses": int(valid and not won and not draw),
        "game_seconds": summary.game_seconds,
        "exclusive": summary.observations,
        "policy_samples": summary.opponent_latencies,
        "policy_latency": asdict(
            _distribution(summary.opponent_latencies, len(summary.opponent_latencies))
        ),
        "llm_phases": summary.llm_phases,
        "reasoning_tokens": summary.reasoning_tokens,
        "answer_tokens": summary.answer_tokens,
        "output_tokens": summary.output_tokens,
        "token_split_attempts": summary.token_split_attempts,
        "generation_attempts": summary.generation_attempts,
    }


def _combine_matchups(parts: list[dict[str, Any]]) -> dict[str, Any] | None:
    matchups = [p["matchup"] for p in parts if p.get("matchup") is not None]
    if not matchups:
        return None
    result = {
        key: sum(m[key] for m in matchups)
        for key in (
            "games",
            "valid",
            "wins",
            "draws",
            "losses",
            "game_seconds",
            "reasoning_tokens",
            "answer_tokens",
            "output_tokens",
            "token_split_attempts",
            "generation_attempts",
        )
    }
    for key in ("exclusive", "llm_phases"):
        names = set.intersection(*(set(m[key]) for m in matchups))
        # Missing a timer in any game is unavailable, not a zero contribution.
        result[key] = {name: sum(m[key][name] for m in matchups) for name in names}
    samples = [v for m in matchups for v in m["policy_samples"]]
    result["policy_samples"] = samples
    result["policy_latency"] = asdict(_distribution(samples, len(samples)))
    return result


def _project_llm_sequence(root: Path) -> dict[str, Any]:
    """One shared model, multiple games; keep overlap and game populations explicit."""
    sequence = _GameSequence.model_validate(_object(root / "games.json"))
    concurrent = sequence.mode == "concurrent"
    if len(set(sequence.games)) != len(sequence.games):
        raise ValueError("duplicate game in LLM sequence")
    available = [
        name
        for name in sequence.games
        if any(
            (root / name / path).exists()
            for path in ("config.json", "game/config.json")
        )
    ]
    parts = [project_llm(root / name) for name in available]
    runtimes = [p["runtime"] for p in parts if p["runtime"] is not None]
    if runtimes and any(runtime != runtimes[0] for runtime in runtimes):
        raise ValueError("LLM sequence changed runtime identity")
    selections = [
        (p, _Selection.model_validate(_object(p)))
        for name in sequence.games
        for p in sorted((root / name).rglob("selection.json"))
    ]
    identities = [s.config for _, s in selections]
    if identities and any(identity != identities[0] for identity in identities):
        raise ValueError("LLM sequence changed inference identity")
    complete = len(parts) == len(sequence.games)
    status: ObservedStatus = (
        "failed"
        if any(p["observed_status"] == "failed" for p in parts)
        else "completed"
        if complete and all(p["observed_status"] == "completed" for p in parts)
        else "partial"
    )
    coverage: dict[str, JsonValue] = {}
    fields = (
        "selection_files",
        "command_files",
        "committed_commands",
        "initial",
        "repaired",
        "fallback",
        "failed_selections",
        "unknown_selections",
        "generation_attempts",
    )
    for name in fields:
        coverage[name] = sum(p["coverage"][name] for p in parts)
    for name in (
        "selection_population",
        "command_population",
        "completed_games",
        "replay_verified_games",
    ):
        values = [p["coverage"][name] for p in parts]
        coverage[name] = (
            sum(values) if values and all(v is not None for v in values) else None
        )
    coverage.update(
        summary=complete and all(p["coverage"]["summary"] for p in parts),
        planned_games=len(sequence.games),
        observed_games=len(parts),
    )
    metrics: dict[str, dict[str, JsonValue]] = {}

    def measure(
        name: str, value: float | None, boundary: str, *, derived: bool = False
    ) -> None:
        metrics[name] = asdict(
            Measurement(
                value,
                "s",
                "unavailable"
                if value is None
                else "derived"
                if derived
                else "recorded",
                "sequence parent",
                boundary,
                "one shared model with concurrent games"
                if concurrent
                else "one sequential resident session",
                "Read from retained evidence" if value is not None else "Not retained",
            )
        )

    attempt_path = root / "attempt.json"
    attempt = (
        AttemptObservation.model_validate(_object(attempt_path))
        if attempt_path.exists()
        else None
    )
    measure(
        "whole_attempt",
        attempt.whole_attempt_seconds if attempt else None,
        attempt.boundary if attempt else "parent entry through cleanup",
    )
    cache_path = root / "cache.json"
    cache = _Cache.model_validate(_object(cache_path)) if cache_path.exists() else None
    measure(
        "cache_preparation",
        cache.seconds if cache else None,
        "acquisition plus verification once for the session",
    )
    for name in ("acquisition_seconds", "verification_seconds"):
        measure(
            "cache_" + name.removesuffix("_seconds"),
            getattr(cache, name) if cache else None,
            "nested within cache preparation",
        )
    readiness_path = root / "readiness.json"
    readiness = _object(readiness_path) if readiness_path.exists() else {}
    readiness_seconds = readiness.get("seconds")
    if readiness_seconds is not None and (
        not isinstance(readiness_seconds, (int, float))
        or not math.isfinite(readiness_seconds)
        or readiness_seconds < 0
    ):
        raise ValueError("invalid readiness seconds")
    measure(
        "worker_readiness",
        readiness_seconds,
        "spawn/runtime construction/ready IPC; charged once, outside all game windows",
    )
    if "policy_cpu_seconds" in readiness:
        seconds = readiness["policy_cpu_seconds"]
        if (
            not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds < 0
        ):
            raise ValueError("invalid policy readiness")
        measure(
            "policy_readiness",
            seconds,
            "checkpoint verification and CPU load, once outside games",
        )
    walls = [p["metrics"]["game_wall"]["value"] for p in parts]
    measure(
        "game_wall",
        sum(walls) if walls and all(v is not None for v in walls) else None,
        "sum of overlapping game windows, including failed work; not campaign wall or a throughput denominator"
        if concurrent
        else "sum of sequential attempted game windows, including failed work",
        derived=True,
    )
    measure(
        "inclusive_cost",
        None,
        "provider settlement unavailable; local wall does not establish billing",
    )
    metrics["inclusive_cost"]["unit"] = "USD"
    files = inventory(root)
    return {
        "matchup": _combine_matchups(parts),
        "identity": hashlib.sha256(
            json.dumps([p["identity"] for p in parts]).encode()
        ).hexdigest(),
        "label": root.name,
        "observed_status": status,
        "failure": next((p["failure"] for p in parts if p["failure"]), None),
        "coverage": coverage,
        **_selection_latencies([s for _, s in selections], len(selections)),
        "metrics": metrics,
        "device_memory": _device_memory(root / "gpu-memory.csv"),
        "config": {"games": [p["config"] for p in parts]},
        "inference_config": identities[0].model_dump() if identities else None,
        "runtime": parts[0]["runtime"] if parts else None,
        "corpus": [str(p.relative_to(root)) for p, _ in selections],
        "evidence": [f.model_dump() for f in files],
        "games": [
            {"label": name, "projection": part}
            for name, part in zip(available, parts, strict=True)
        ],
        "notes": [
            "Concurrent game processes share one model; readiness is charged once. Game windows overlap."
            if concurrent
            else "One sequential resident session; readiness is charged once.",
            "Per-game distributions remain separate below. Selection distributions describe this one declared workload/session.",
            "Missing planned games remain incomplete; no renderer is run by this report.",
        ],
    }


class _DeviceMemory(TypedDict):
    samples: int
    peak_used_bytes: int | None
    total_bytes: int | None
    device_names: list[str]
    window_start: str | None
    window_end: str | None
    device_uuid: str | None
    sample_interval_seconds: float | None
    note: str


def _device_memory(path: Path) -> _DeviceMemory:
    used: list[int] = []
    totals: list[int] = []
    stamps: list[str] = []
    names: set[str] = set()
    uuids: set[str] = set()
    sampling_path = path.with_name("device-sampling.json")
    sampling = _object(sampling_path) if sampling_path.exists() else {}
    interval = sampling.get("interval_seconds")
    if interval is not None and (
        not isinstance(interval, (int, float))
        or not math.isfinite(interval)
        or interval <= 0
    ):
        raise ValueError("invalid device sampling interval")
    if path.exists():
        with path.open() as stream:
            reader = csv.DictReader(stream, skipinitialspace=True)
            expected = {"timestamp", "name", "memory.total [MiB]", "memory.used [MiB]"}
            if not expected.issubset(reader.fieldnames or []):
                raise ValueError("malformed GPU memory CSV header")
            for row in reader:
                used.append(_mib_bytes(row["memory.used [MiB]"]))
                totals.append(_mib_bytes(row["memory.total [MiB]"]))
                stamps.append(row["timestamp"])
                names.add(row["name"])
                if row.get("uuid"):
                    uuids.add(row["uuid"])
        if (
            len(names) > 1
            or len(uuids) > 1
            or len(set(totals)) > 1
            or any(u > t for u, t in zip(used, totals))
        ):
            raise ValueError("ambiguous device memory population")
    return {
        "samples": len(used),
        "peak_used_bytes": max(used) if used else None,
        "total_bytes": totals[0] if totals else None,
        "device_names": sorted(names),
        "window_start": stamps[0] if stamps else None,
        "window_end": stamps[-1] if stamps else None,
        "device_uuid": next(iter(uuids)) if uuids else None,
        "sample_interval_seconds": interval,
        "note": "Sampled whole-device occupancy may include other processes. Missing UUID/interval observations remain unavailable.",
    }


def _mib_bytes(value: str) -> int:
    number = int(value.removesuffix(" MiB"))
    if number < 0:
        raise ValueError("negative device memory")
    return number * 1024 * 1024
