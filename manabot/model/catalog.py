"""Find trained policy checkpoints and check them against the running engine.

The catalog answers "which bots could play here?" without constructing them.
``inspect_checkpoint`` reads container metadata or a legacy ``.pt`` with
Torch's ``weights_only`` unpickler, and returns a ``CheckpointSummary``: exact bytes, world,
model shape, training step, producer and — when the checkpoint cannot play in
this runtime — a sentence naming the mismatch. ``discover_checkpoints`` walks
run directories, explicit files and Tournament files into summaries.

Admission belongs to the injected backend's policy loader; the catalog
uses its bound checkpoint reader so tensor and metadata refusals appear in the listing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Literal

from manabot.env.observation import ObservationSpace
from manabot.infra.config import (
    ModelConfig,
    ObservationConfig,
)
from manabot.infra.identity import file_sha256
from manabot.model.interface import model_interface
from manabot.policy.protocol import Backend
import managym

CheckpointSource = Literal["path", "runs", "tournament"]

# Run directories also hold datasets, optimizer state and recovery snapshots;
# none of those are policies, and very large files are never policies here.
MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class CheckpointSummary:
    path: Path
    sha256: str
    bytes: int
    label: str
    source: CheckpointSource
    producer: str
    # ``None`` when the checkpoint predates world binding.
    world: str | None
    model: str
    parameters: int
    # Trainer ``global_step`` (environment steps) when it records one.
    step: int | None
    # Each bound seat's deck exactly as the checkpoint recorded it.
    setups: tuple[managym.Deck, ...] = field(default=())
    # Why the running engine refuses this checkpoint; ``None`` means admissible.
    incompatibility: str | None = None

    run: str | None = None
    run_id: str | None = None
    stage: str | None = None
    weights: str | None = None
    updates: int | None = None
    transitions: int | None = None
    supports_search: bool = False
    policy_input_version: int | None = None
    search_reason: str | None = None

    @property
    def compatible(self) -> bool:
        return self.incompatibility is None


class NotAPolicyCheckpoint(ValueError):
    """The file is readable but holds no policy (optimizer, dataset, ...)."""


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


def inspect_checkpoint(
    path: Path,
    *,
    backend: Backend,
    label: str | None = None,
    source: CheckpointSource = "path",
    producer: str | None = None,
) -> CheckpointSummary:
    """Summarize one checkpoint file.

    Raises ``OSError`` when the file cannot be read and ``ValueError`` when it
    is not a policy checkpoint. Incompatibility with the running engine is not
    an error: it is reported on the summary so it can be shown and refused.
    """

    resolved = path.resolve()
    stat = resolved.stat()
    if stat.st_size > MAX_CHECKPOINT_BYTES:
        raise NotAPolicyCheckpoint(f"{resolved} is too large to be a policy")
    summary = _inspect_cached(str(resolved), stat.st_size, stat.st_mtime_ns, backend)
    if isinstance(summary, NotAPolicyCheckpoint):
        raise summary
    return replace(
        summary,
        label=label or summary.label,
        source=source,
        producer=producer or summary.producer,
    )


@lru_cache(maxsize=512)
def _inspect_cached(
    path: str, size: int, mtime_ns: int, backend: Backend
) -> CheckpointSummary | NotAPolicyCheckpoint:
    # Refusals are returned, not raised, so they are cached too: run
    # directories hold large optimizer and dataset files that every listing
    # and every game admission would otherwise read again in full.
    del mtime_ns  # cache key only: a rewritten file is re-read
    try:
        checkpoint = backend.metadata(path)
    except Exception as error:  # legacy unpickling and malformed container headers
        return NotAPolicyCheckpoint(f"{path} holds no policy: {error}")
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("hypers"), Mapping
    ):
        return NotAPolicyCheckpoint(f"{path} holds no policy")
    config = checkpoint["hypers"]
    binding = checkpoint.get("world_binding")
    world = binding.get("world") if isinstance(binding, Mapping) else None
    step = checkpoint.get("global_step")
    search_reason = checkpoint_search_incompatibility(checkpoint)
    return CheckpointSummary(
        path=Path(path),
        sha256=file_sha256(path),
        bytes=size,
        label=_default_label(Path(path)),
        source="path",
        producer=_producer(checkpoint),
        world=world if isinstance(world, str) else None,
        model=_model_shape(config.get("agent_hypers")),
        parameters=(checkpoint.get("architecture") or {})
        .get("parameters", {})
        .get("total", checkpoint.get("parameter_count", 0)),
        step=step if isinstance(step, int) and step > 0 else None,
        run_id=_text(checkpoint.get("bc"), "run_id"),
        stage=_text(checkpoint.get("bc"), "stage_id"),
        weights=_text(checkpoint.get("bc"), "weights"),
        setups=_setups(binding),
        incompatibility=refusal(lambda: backend.read_policy(path)),
        policy_input_version=config.get("observation_hypers", {}).get(
            "policy_input_version", 1
        ),
        supports_search=search_reason is None,
        search_reason=search_reason,
    )


def checkpoint_search_incompatibility(checkpoint: Mapping[str, object]) -> str | None:
    """Explain baseline search admission from the same declared capabilities."""
    hypers = checkpoint.get("hypers")
    agent = hypers.get("agent_hypers") if isinstance(hypers, Mapping) else None
    if not isinstance(agent, Mapping):
        return "Search requires a declared policy configuration."
    metadata = checkpoint.get("bc")
    semantic = metadata.get("value_semantic") if isinstance(metadata, Mapping) else None
    if agent.get("value_kind") != "categorical_wdl" and semantic != "signed_outcome":
        return (
            "Search requires a W/D/L value head or a scalar declared as signed_outcome."
        )
    if agent.get("recent_events"):
        return "Search requires evaluation from the current view alone, without policy history."
    if agent.get("belief_count_buckets"):
        return "Search requires evaluation without belief-model inputs."
    return None


def checkpoint_incompatibility(checkpoint: Mapping[str, Any]) -> str | None:
    """Explain metadata admission when only a checkpoint record is available."""

    def admit() -> None:
        observation = ObservationConfig.from_checkpoint(
            checkpoint["hypers"]["observation_hypers"]
        )
        model_interface(checkpoint, ObservationSpace(observation))

    return refusal(admit)


def refusal(admit: Callable[[], object]) -> str | None:
    """Run an admission and return its refusal as a sentence instead of raising."""
    try:
        admit()
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        RuntimeError,
        managym.AgentError,
    ) as error:
        return str(error)
    except BaseException as error:
        # Invalid retained native setups may raise a PyO3 panic.
        if isinstance(error, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            raise
        return "its deck setup or observation settings no longer build in this engine"
    return None


def _setups(binding: Any) -> tuple[managym.Deck, ...]:
    if not isinstance(binding, Mapping) or not isinstance(binding.get("setups"), list):
        return ()
    return tuple(
        managym.Deck(dict(setup.get("deck", {})), dict(setup.get("sideboard", {})))
        for setup in binding["setups"]
        if isinstance(setup, Mapping)
    )


def _producer(checkpoint: Mapping[str, Any]) -> str:
    extra = checkpoint.get("bc")
    if isinstance(extra, Mapping) and extra.get("run_id"):
        stage = f" {extra['stage_id']}" if extra.get("stage_id") else ""
        weights = f" ({extra['weights']})" if extra.get("weights") else ""
        return f"regime run {extra['run_id']}{stage}{weights}"
    if checkpoint["hypers"].get("train_hypers"):
        return "manabot train"
    return "unknown"


def _model_shape(agent_config: Any) -> str:
    try:
        spec = ModelConfig(**agent_config)
    except Exception:  # unknown fields from another code version
        return "unknown shape"
    if not spec.attention_on:
        return f"width {spec.hidden_dim}, no attention"
    layers = "layer" if spec.attention_layers == 1 else "layers"
    return (
        f"width {spec.hidden_dim}, {spec.attention_layers} attention {layers}"
        f" × {spec.num_attention_heads} heads"
    )


def _default_label(path: Path) -> str:
    # Run directories name the experiment; files name the step or stage.
    return f"{path.parent.name}/{path.stem}" if path.parent.name else path.stem


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UnavailableEntry:
    """A Tournament entrant whose exact bytes were not found locally."""

    label: str
    sha256: str
    producer: str
    reason: str


def discover_checkpoints(
    sources: Iterable[Path], *, backend: Backend, runs_dir: Path | None = None
) -> tuple[list[CheckpointSummary], list[UnavailableEntry]]:
    """Summarize every policy checkpoint reachable from ``sources``.

    A source may be a ``.pt`` file, a run directory (searched recursively) or
    a Tournament JSON file. Tournament entrant ``file`` names are resolved
    beside the Tournament file and in each directory source, and admitted
    only when the bytes match the entrant's pinned SHA-256. ``runs_dir``
    (``.runs/`` by convention) is searched like a directory source but labelled
    as local runs. Results are de-duplicated by SHA-256, first source wins.
    """

    found: dict[str, CheckpointSummary] = {}
    missing: list[UnavailableEntry] = []
    sources = [Path(source) for source in sources]
    directories = [source for source in sources if source.is_dir()]
    if runs_dir is not None and runs_dir.is_dir():
        directories.append(runs_dir)

    def add(summary: CheckpointSummary) -> None:
        previous = found.get(summary.sha256)
        if previous is None or len(str(summary.path)) < len(str(previous.path)):
            found[summary.sha256] = summary

    for source in sources:
        if source.suffix == ".json" and source.is_file():
            for entry in _tournament(source, directories, backend):
                if isinstance(entry, CheckpointSummary):
                    add(entry)
                else:
                    missing.append(entry)
        elif source.is_file():
            add(inspect_checkpoint(source, backend=backend))
        elif source.is_dir():
            for summary in _scan(source, "path", backend):
                add(summary)
    if runs_dir is not None and runs_dir.is_dir():
        for summary in _scan(runs_dir, "runs", backend):
            add(summary)
    return list(found.values()), missing


def _scan(
    directory: Path, source: CheckpointSource, backend: Backend
) -> list[CheckpointSummary]:
    summaries = []
    for path in sorted([*directory.rglob("*.pt"), *directory.rglob("*.safetensors")]):
        try:
            summaries.append(
                inspect_checkpoint(
                    path,
                    backend=backend,
                    label=str(path.relative_to(directory).with_suffix("")),
                    source=source,
                )
            )
        except (OSError, ValueError):
            continue
    return [_with_run_facts(summary, directory) for summary in summaries]


def _text(value: object, key: str) -> str | None:
    item = value.get(key) if isinstance(value, dict) else None
    return item if isinstance(item, str) else None


def _count(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


@lru_cache(maxsize=512)
def _run_record(path: str, size: int, mtime_ns: int) -> dict[str, Any]:
    del size, mtime_ns  # cache key only
    try:
        record = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def _with_run_facts(summary: CheckpointSummary, root: Path) -> CheckpointSummary:
    """Join portable run records by exact bytes, never by their recorded paths.

    Monitoring coordinates belong to their own checkpoint. Stage export counts
    apply only to completed stages; initialization is always update zero.
    """
    root = root.resolve()
    for parent in summary.path.parents:
        if not parent.is_relative_to(root):
            break
        path = parent / "run.json"
        try:
            stat = path.stat()
        except OSError:
            continue
        record = _run_record(str(path), stat.st_size, stat.st_mtime_ns)
        points = record.get("monitoring_checkpoints", [])
        stages = record.get("stages", [])
        for point in points if isinstance(points, list) else []:
            if not isinstance(point, dict):
                continue
            artifact = point.get("artifact")
            if isinstance(artifact, dict) and artifact.get("sha256") == summary.sha256:
                return replace(
                    summary,
                    run=parent.name,
                    run_id=_text(record, "id") or summary.run_id,
                    updates=_count(point.get("updates")),
                    transitions=_count(point.get("learner_transitions")),
                )
        for stage in stages if isinstance(stages, list) else []:
            if not isinstance(stage, dict) or not isinstance(
                stage.get("artifacts"), dict
            ):
                continue
            for weights, artifact in stage["artifacts"].items():
                if weights not in {"raw", "ema", "initial_raw"}:
                    continue
                if (
                    not isinstance(artifact, dict)
                    or artifact.get("sha256") != summary.sha256
                ):
                    continue
                completed = stage.get("status") == "completed"
                initial = weights == "initial_raw"
                return replace(
                    summary,
                    run=parent.name,
                    run_id=_text(record, "id") or summary.run_id,
                    stage=_text(stage, "id") or summary.stage,
                    weights=weights,
                    updates=0
                    if initial
                    else _count(stage.get("updates"))
                    if completed
                    else None,
                    transitions=0
                    if initial
                    else _count(stage.get("learner_transitions"))
                    if completed
                    else None,
                )
    return summary


def _tournament(
    path: Path, directories: list[Path], backend: Backend
) -> list[CheckpointSummary | UnavailableEntry]:
    """Read checkpoint entrants from a Tournament file.

    Only the entrant fields that identify exact bytes (``file``, ``sha256``)
    are read here; code anchors without a ``file`` are skipped.
    """

    try:
        entrants = json.loads(path.read_text())["entrants"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"{path} is not a Tournament file") from error
    results: list[CheckpointSummary | UnavailableEntry] = []
    for entrant in entrants:
        if not isinstance(entrant, Mapping) or not entrant.get("file"):
            continue
        label = str(entrant.get("label") or entrant.get("id"))
        sha256 = str(entrant.get("sha256", ""))
        producer = f"Tournament {path.stem}: {entrant.get('origin', 'unknown origin')}"
        for directory in [path.parent, *directories]:
            candidate = directory / str(entrant["file"])
            if not candidate.is_file():
                continue
            try:
                summary = inspect_checkpoint(
                    candidate,
                    backend=backend,
                    label=label,
                    source="tournament",
                    producer=producer,
                )
            except (OSError, ValueError):
                continue
            if summary.sha256 == sha256:
                results.append(summary)
                break
        else:
            results.append(
                UnavailableEntry(
                    label=label,
                    sha256=sha256,
                    producer=producer,
                    reason=f"{entrant['file']} with the pinned bytes was not found",
                )
            )
    return results
