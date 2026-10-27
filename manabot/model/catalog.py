"""Find trained policy checkpoints and check them against the running engine.

The catalog answers "which bots could play here?" without constructing them.
``inspect_checkpoint`` reads one ``.pt`` file with Torch's ``weights_only``
unpickler (tensors and plain containers only, so listing never executes code
from a checkpoint) and returns a ``CheckpointSummary``: exact bytes, world,
model shape, training step, producer and — when the checkpoint cannot play in
this runtime — a sentence naming the mismatch. ``discover_checkpoints`` walks
run directories, explicit files and Tournament files into summaries.

<<<<<<< HEAD
Admission is still ``manabot.sim.flat_mc.load_checkpoint_agent``; the catalog
=======
Admission is still ``manabot.sim.search.flat_mc.load_checkpoint``; the catalog
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
reuses its world binding rules (``manabot.model.world``) so a listed
"compatible" checkpoint and a loadable one cannot disagree about the world.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Literal

import torch

from manabot.env import ObservationSpace
<<<<<<< HEAD
from manabot.infra.hypers import AgentSpec, ObservationSpaceHypers
from manabot.model.world import WORLD, checkpoint_world
from managym import PlayerConfig
=======
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.model.world import WORLD, checkpoint_configs, checkpoint_world
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f

CheckpointSource = Literal["path", "runs", "tournament"]

# Run directories also hold datasets, optimizer state and recovery snapshots;
# none of those are policies, and very large files are never policies here.
MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class DeckSetup:
    """One seat's deck and sideboard exactly as the checkpoint was bound."""

    deck: dict[str, int]
    sideboard: dict[str, int]


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
    setups: tuple[DeckSetup, ...] = field(default=())
    # Why the running engine refuses this checkpoint; ``None`` means admissible.
    incompatibility: str | None = None

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
    summary = _inspect_cached(str(resolved), stat.st_size, stat.st_mtime_ns)
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
    path: str, size: int, mtime_ns: int
) -> CheckpointSummary | NotAPolicyCheckpoint:
    # Refusals are returned, not raised, so they are cached too: run
    # directories hold large optimizer and dataset files that every listing
    # and every game admission would otherwise read again in full.
    del size, mtime_ns  # cache key only: a rewritten file is re-read
    data = Path(path).read_bytes()
    try:
        checkpoint = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except Exception:  # torch raises several unpickling types
        return NotAPolicyCheckpoint(f"{path} is not a weights-only checkpoint")
    if (
        not isinstance(checkpoint, dict)
        or not isinstance(checkpoint.get("model_state_dict"), Mapping)
        or not isinstance(checkpoint.get("hypers"), Mapping)
    ):
        return NotAPolicyCheckpoint(f"{path} holds no policy")
<<<<<<< HEAD
    hypers = checkpoint["hypers"]
=======
    config = checkpoint["hypers"]
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    binding = checkpoint.get("world_binding")
    world = binding.get("world") if isinstance(binding, Mapping) else None
    step = checkpoint.get("global_step")
    return CheckpointSummary(
        path=Path(path),
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
        label=_default_label(Path(path)),
        source="path",
        producer=_producer(checkpoint),
        world=world if isinstance(world, str) else None,
<<<<<<< HEAD
        model=_model_shape(hypers.get("agent_hypers")),
=======
        model=_model_shape(config.get("agent_hypers")),
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
        parameters=sum(
            int(tensor.numel())
            for tensor in checkpoint["model_state_dict"].values()
            if isinstance(tensor, torch.Tensor)
        ),
        step=step if isinstance(step, int) and step > 0 else None,
        setups=_setups(binding),
        incompatibility=checkpoint_incompatibility(checkpoint),
    )


def checkpoint_incompatibility(checkpoint: Mapping[str, Any]) -> str | None:
    """Name the first way a checkpoint's world binding differs from this runtime.

    Mirrors ``validate_checkpoint_world`` but explains the difference instead
    of raising, field by field, in the order a researcher would fix it.
    """

    binding = checkpoint.get("world_binding")
    if not isinstance(binding, Mapping):
        return f"trained without a world binding; engine is {WORLD}"
    trained = binding.get("world")
    if trained != WORLD:
        return f"trained in {trained}, engine is {WORLD}"
    try:
<<<<<<< HEAD
        observation = ObservationSpaceHypers(
            **checkpoint["hypers"]["observation_hypers"]
        )
        configs = [
            PlayerConfig(str(index), setup["deck"], setup["sideboard"])
            for index, setup in enumerate(binding["setups"])
        ]
=======
        observation = ObservationConfig.from_checkpoint(
            checkpoint["hypers"]["observation_hypers"]
        )
        configs = checkpoint_configs(binding)
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
        expected = checkpoint_world(configs, ObservationSpace(observation))
    except BaseException as error:  # PyO3 panics on invalid native setups
        if isinstance(error, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            raise
        return "its deck setup or observation settings no longer build in this engine"
    for key, explanation in _BINDING_FIELDS:
        if binding.get(key) != expected.get(key):
            return explanation(binding.get(key), expected.get(key))
    return None


def _content_mismatch(trained: Any, engine: Any) -> str:
    def digest(manifest: Any) -> str:
        if isinstance(manifest, Mapping):
            return str(manifest.get("content_digest", "unknown"))[:12]
        return "unknown"

    return (
        f"card content differs: trained on content {digest(trained)}, "
        f"engine has {digest(engine)}"
    )


_BINDING_FIELDS = (
    (
        "policy_input_version",
        lambda trained, engine: (
            f"policy input version {trained}, engine expects {engine}"
        ),
    ),
    (
        "rules",
        lambda trained, engine: (
            f"rules decision/world-space versions {trained}, engine has {engine}"
        ),
    ),
    ("content_manifest", _content_mismatch),
    (
        "input_schema",
        lambda trained, engine: "observation tensor shapes differ from this engine",
    ),
    (
        "learning_schema_sha256",
        lambda trained, engine: (
            f"learning schema {str(trained)[:12]}, engine has {str(engine)[:12]}"
        ),
    ),
    (
        "setups",
        lambda trained, engine: "its recorded deck setup no longer resolves the same",
    ),
)


def _setups(binding: Any) -> tuple[DeckSetup, ...]:
    if not isinstance(binding, Mapping) or not isinstance(binding.get("setups"), list):
        return ()
    return tuple(
        DeckSetup(dict(setup.get("deck", {})), dict(setup.get("sideboard", {})))
        for setup in binding["setups"]
        if isinstance(setup, Mapping)
    )


def _producer(checkpoint: Mapping[str, Any]) -> str:
    extra = checkpoint.get("bc")
    if isinstance(extra, Mapping) and extra.get("run_id"):
        stage = f" {extra['stage_id']}" if extra.get("stage_id") else ""
        weights = f" ({extra['weights']})" if extra.get("weights") else ""
        return f"regime run {extra['run_id']}{stage}{weights}"
<<<<<<< HEAD
    if isinstance(extra, Mapping):
        return "supervised distillation"
=======
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    if checkpoint["hypers"].get("train_hypers"):
        return "manabot train"
    return "unknown"


<<<<<<< HEAD
def _model_shape(agent_hypers: Any) -> str:
    try:
        spec = AgentSpec(**agent_hypers)
=======
def _model_shape(agent_config: Any) -> str:
    try:
        spec = ModelConfig(**agent_config)
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
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
    sources: Iterable[Path], *, runs_dir: Path | None = None
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
        found.setdefault(summary.sha256, summary)

    for source in sources:
        if source.suffix == ".json" and source.is_file():
            for entry in _tournament(source, directories):
                if isinstance(entry, CheckpointSummary):
                    add(entry)
                else:
                    missing.append(entry)
        elif source.is_file():
            add(inspect_checkpoint(source))
        elif source.is_dir():
            for summary in _scan(source, "path"):
                add(summary)
    if runs_dir is not None and runs_dir.is_dir():
        for summary in _scan(runs_dir, "runs"):
            add(summary)
    return list(found.values()), missing


def _scan(directory: Path, source: CheckpointSource) -> list[CheckpointSummary]:
    summaries = []
    for path in sorted(directory.rglob("*.pt")):
        try:
            summaries.append(
                inspect_checkpoint(
                    path,
                    label=str(path.relative_to(directory).with_suffix("")),
                    source=source,
                )
            )
        except (OSError, ValueError):
            continue
    return summaries


def _tournament(
    path: Path, directories: list[Path]
) -> list[CheckpointSummary | UnavailableEntry]:
    """Read checkpoint entrants from a Tournament file.

<<<<<<< HEAD
    The schema owner is ``experiments.runners.head_to_head.Tournament``; only
    the entrant fields that identify exact bytes are read here, and code
    anchors without a ``file`` are skipped.
=======
    Only the entrant fields that identify exact bytes (``file``, ``sha256``)
    are read here; code anchors without a ``file`` are skipped.
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
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
                    candidate, label=label, source="tournament", producer=producer
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
