"""Admit MTG-183's historical uniform control without another training run.

The protocol manifest names genuine update exports and the first window's
confirmed cleanup receipt. `publish_uniform` writes a clearly labelled Etude
projection of those records: its current publication time is not training time.
It never rents, starts learning, changes source checkpoints or fabricates rows.
"""

import argparse
from decimal import Decimal
from pathlib import Path
import shutil
from typing import Literal

from etude.run.listing import runs as list_runs
from etude.run.records import ArtifactRef
from etude.run.storage import Storage, storage_from_uri
from etude.train.read import read
from etude.train.records import Step
from etude.train.run import track
from pydantic import BaseModel, ConfigDict, Field, model_validator

from manabot.experiment.job import StagedExport
from manabot.infra.files import atomic_json
from manabot.infra.identity import canonical_sha256, file_sha256

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/protocols/curriculum-175-uniform.json"
EXPERIMENT = "curriculum_adapters"
UNIFORM_RUN = "mtg183-uniform-reused"


class SourceCoordinates(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    updates: int = Field(ge=0)
    learner_transitions: int = Field(ge=0)
    environment_decisions: int = Field(ge=0)
    games: int = Field(ge=0)
    optimizer_exposures: int = Field(ge=0)
    optimizer_updates: int = Field(ge=0)
    training_seconds: float = Field(ge=0)
    active_training_seconds: float = Field(ge=0)
    segment_updates: int = Field(ge=0)
    stage_id: str


class HistoricalCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    updates: int = Field(ge=0)
    coordinates: SourceCoordinates
    artifact: ArtifactRef


class UniformControl(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_: Literal["mtg175-reused-uniform-v1"] = Field(alias="schema")
    source_task: Literal["MTG-183"]
    source_job: str
    source_training_run: str
    source_commit: str
    source_outcome: Literal["paused"]
    seed: int
    policy_input_version: Literal[1]
    endpoint_updates: int = Field(gt=0)
    estimated_inclusive_dollars: Decimal = Field(ge=0, le=10)
    cleanup_confirmed_at: float = Field(gt=0)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoints: tuple[HistoricalCheckpoint, ...]

    @model_validator(mode="after")
    def genuine_sequence(self) -> "UniformControl":
        updates = [cp.updates for cp in self.checkpoints]
        if (
            not updates
            or updates != sorted(set(updates))
            or updates[0] != 0
            or updates[-1] != self.endpoint_updates
        ):
            raise ValueError(
                "uniform checkpoints must cover initialization through the retained endpoint"
            )
        for cp in self.checkpoints:
            if (
                cp.coordinates.updates != cp.updates
                or cp.artifact.sha256 is None
                or cp.artifact.size_bytes is None
            ):
                raise ValueError(
                    "uniform checkpoint coordinate or byte identity is missing"
                )
        return self

    @property
    def initial(self) -> StagedExport:
        """The existing R2 input used to initialize every new adaptive arm."""
        ref = self.checkpoints[0].artifact
        if not ref.uri.startswith("r2://manabot/jobs/"):
            raise ValueError(
                "uniform initialization must be retained in the training store"
            )
        assert ref.sha256 is not None and ref.size_bytes is not None
        return StagedExport(
            # Launcher Input.key is bucket-relative, not storage-prefix-relative.
            key=ref.uri.removeprefix("r2://manabot/"),
            sha256=ref.sha256,
            bytes=ref.size_bytes,
        )


def load_control() -> UniformControl:
    return UniformControl.model_validate_json(MANIFEST.read_bytes())


def materialize_uniform(directory: Path, checkpoint_cache: Path) -> Path:
    """Place verified original bytes where Etude's run-relative job inputs expect them.

    The original reference records and source manifest remain unchanged. Etude
    discovers these identical local copies when reading the imported run.
    Deliver them with `etude sync` before queuing evaluations.
    """
    control = load_control()
    for cp in control.checkpoints:
        source = checkpoint_cache / str(cp.artifact.sha256)
        if (
            source.stat().st_size != cp.artifact.size_bytes
            or file_sha256(source) != cp.artifact.sha256
        ):
            raise ValueError("cached uniform bytes differ from their retained export")
    copied: list[dict[str, str | int]] = []
    for cp in control.checkpoints:
        name = cp.artifact.uri.rpartition("/")[2]
        key = f"{UNIFORM_RUN}/checkpoints/{name}"
        target = directory / key
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if file_sha256(target) != cp.artifact.sha256:
                raise ValueError(
                    "uniform checkpoint destination already holds different bytes"
                )
        else:
            partial = target.with_name(target.name + ".partial")
            shutil.copyfile(checkpoint_cache / str(cp.artifact.sha256), partial)
            partial.replace(target)
        copied.append(
            {
                "updates": cp.updates,
                "source_uri": cp.artifact.uri,
                "key": key,
                "sha256": str(cp.artifact.sha256),
                "bytes": target.stat().st_size,
            }
        )
    receipt = directory / "mtg175-uniform-checkpoint-files.json"
    atomic_json(
        receipt, {"schema": "mtg175-uniform-checkpoint-files-v1", "files": copied}
    )
    return receipt


def publish_uniform(
    directory: Path,
    *,
    control: UniformControl | None = None,
    storages: tuple[Storage, ...] = (),
) -> Path:
    """Publish references to original exports once; partial imports resume by coordinate.

    Validation rejects a conflicting prior projection. The recorded source
    outcome remains `paused`; `finished` means only that this import is complete.
    """
    control = control or load_control()
    identity = canonical_sha256(control.model_dump(mode="json", by_alias=True))
    existing = set(list_runs(directory))
    known: dict[int, str | None] = {}
    if (EXPERIMENT, UNIFORM_RUN) in existing:
        data = read(directory, EXPERIMENT, UNIFORM_RUN)
        known = {int(cp.at["updates"]): cp.artifact.sha256 for cp in data.checkpoints}
        allowed = {cp.updates: cp.artifact.sha256 for cp in control.checkpoints}
        if any(allowed.get(update) != digest for update, digest in known.items()):
            raise ValueError("retained uniform projection differs from source exports")
    if len(known) != len(control.checkpoints):
        writer = track(
            experiment=EXPERIMENT,
            run=UNIFORM_RUN,
            kind="curriculum-175-u",
            seed=control.seed,
            clocks=(
                Step("updates", "original completed updates"),
                Step("learner_transitions", "original learner decisions"),
            ),
            config={
                "evidence_kind": "imported-uniform-control",
                "source": control.model_dump(mode="json", by_alias=True),
            },
            directory=directory,
            storages=storages,
            strict=True,
        )
        try:
            for cp in control.checkpoints:
                if cp.updates in known:
                    continue
                transitions = float(cp.coordinates.learner_transitions)
                metrics: dict[str, object] = {
                    "source/updates": cp.updates,
                    "source/training_seconds": cp.coordinates.training_seconds,
                    "source/training_outcome": control.source_outcome,
                    "source/imported": True,
                }
                if cp.updates == control.endpoint_updates:
                    metrics["source/estimated_inclusive_dollars"] = float(
                        control.estimated_inclusive_dollars
                    )
                writer.log(metrics, updates=cp.updates, learner_transitions=transitions)
                writer.checkpoint(
                    cp.artifact,
                    updates=cp.updates,
                    learner_transitions=transitions,
                    tags=("reused-uniform",),
                )
        except BaseException:
            writer.finish("failed")
            raise
        else:
            writer.finish("finished")
    target = directory / "mtg175-uniform-import.json"
    atomic_json(
        target,
        {
            "schema": "mtg175-uniform-import-v1",
            "control_sha256": identity,
            "source_job": control.source_job,
            "checkpoints": len(control.checkpoints),
            "estimated_historical_dollars": str(control.estimated_inclusive_dollars),
            "new_training_jobs": 0,
        },
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--storage")
    parser.add_argument("--checkpoint-cache", type=Path)
    args = parser.parse_args()
    if args.checkpoint_cache is not None:
        print(materialize_uniform(args.directory, args.checkpoint_cache))
    storages = (storage_from_uri(args.storage),) if args.storage else ()
    print(publish_uniform(args.directory, storages=storages))


if __name__ == "__main__":
    main()
