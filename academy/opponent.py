"""Trained opponents: the checkpoint catalog, chosen by exact SHA-256.

The catalog lists checkpoints the researcher trained: everything under the
manabot runs directory (``MANABOT_RUNS_DIR``, default ``./.runs``) plus the
files, run directories and Tournament files in ``ACADEMY_PLAY_CHECKPOINTS``
(``os.pathsep``-separated). manabot's ``manabot.model.catalog`` reads and
checks them; incompatible entries stay listed with the mismatch so the table
can say why they cannot play (``opponent_catalog``, ``catalog_opponent``).

Paths are server configuration; the browser only ever names a digest. The
catalog imports Torch, so it loads lazily and ordinary play never needs it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from manabot.model.catalog import CheckpointSummary, UnavailableEntry

OpponentSource = Literal["path", "runs", "tournament"]


@dataclass(frozen=True)
class Opponent:
    name: str
    producer: str
    sha256: str
    checkpoint: Path
    deterministic: bool
    source: OpponentSource
    world: str | None = None
    model: str | None = None
    step: int | None = None
    # Sorted ``{"deck", "sideboard"}`` setups the checkpoint was bound to.
    setups: list[dict] = field(default_factory=list)

    def identity(self) -> dict:
        """The table protocol's ``OpponentIdentity`` fields."""
        return {
            "name": self.name,
            "producer": self.producer,
            "sha256": self.sha256,
        }

    def record(self) -> dict:
        """Everything a trace needs to say exactly which bot played."""
        return {
            **self.identity(),
            "source": self.source,
            "world": self.world,
            "model": self.model,
            "step": self.step,
            "checkpoint": str(self.checkpoint),
            "deterministic": self.deterministic,
        }

    def verify(self) -> None:
        try:
            actual = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        except OSError as exc:
            raise ValueError(
                "The selected opponent checkpoint is unavailable."
            ) from exc
        if actual != self.sha256:
            raise ValueError(
                "The selected opponent checkpoint changed; restore the pinned artifact."
            )


# ---------------------------------------------------------------------------
# Checkpoint catalog
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CatalogEntry:
    """One selectable (or refused) trained opponent, as the table lists it."""

    sha256: str
    name: str
    producer: str
    source: OpponentSource
    world: str | None = None
    model: str | None = None
    parameters: int | None = None
    step: int | None = None
    setups: list[dict] = field(default_factory=list)
    available: bool = True
    reason: str | None = None

    @classmethod
    def from_summary(cls, summary: CheckpointSummary) -> CatalogEntry:
        return cls(
            sha256=summary.sha256,
            name=summary.label,
            producer=summary.producer,
            source=summary.source,
            world=summary.world,
            model=summary.model,
            parameters=summary.parameters,
            step=summary.step,
            setups=[asdict(setup) for setup in summary.setups],
            available=summary.compatible,
            reason=summary.incompatibility,
        )


def checkpoint_sources() -> list[Path]:
    value = os.environ.get("ACADEMY_PLAY_CHECKPOINTS", "")
    return [Path(part).expanduser() for part in value.split(os.pathsep) if part]


def runs_directory() -> Path:
    return Path(os.environ.get("MANABOT_RUNS_DIR") or Path.cwd() / ".runs")


def _discover() -> tuple[list[CheckpointSummary], list[UnavailableEntry]]:
    from manabot.model.catalog import discover_checkpoints

    return discover_checkpoints(checkpoint_sources(), runs_dir=runs_directory())


def opponent_catalog() -> dict:
    """Every trained opponent this server can name.

    ``catalog_error`` explains a catalog that could not be read at all (for
    example, a play-only install without Torch).
    """

    try:
        found, missing = _discover()
    except (ImportError, OSError, ValueError) as error:
        return {
            "opponents": [],
            "catalog_error": f"Checkpoints could not be listed: {error}",
        }
    entries = [CatalogEntry.from_summary(summary) for summary in found]
    entries.extend(
        CatalogEntry(
            sha256=entry.sha256,
            name=entry.label,
            producer=entry.producer,
            source="tournament",
            available=False,
            reason=entry.reason,
        )
        for entry in missing
    )
    return {"opponents": [asdict(entry) for entry in entries], "catalog_error": None}


def catalog_opponent(sha256: str) -> Opponent | None:
    """Resolve a catalog digest to a verified, admissible opponent.

    Returns ``None`` when no catalog checkpoint has these bytes. Raises
    ``ValueError`` naming the mismatch when the checkpoint is incompatible
    with the running engine or its bytes changed since listing.
    """

    try:
        found, _ = _discover()
    except ImportError as error:
        raise ValueError(
            "Trained checkpoints need the training dependencies on this server."
        ) from error
    summary = next((entry for entry in found if entry.sha256 == sha256), None)
    if summary is None:
        return None
    if summary.incompatibility is not None:
        raise ValueError(
            f"{summary.label} cannot play here: {summary.incompatibility}."
        )
    opponent = Opponent(
        name=summary.label,
        producer=summary.producer,
        sha256=summary.sha256,
        checkpoint=summary.path,
        # Trained policies play as sampled during training.
        deterministic=False,
        source=summary.source,
        world=summary.world,
        model=summary.model,
        step=summary.step,
        setups=[asdict(setup) for setup in summary.setups],
    )
    opponent.verify()
    return opponent
