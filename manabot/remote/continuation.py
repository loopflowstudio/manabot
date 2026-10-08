"""Bind a completed job's final learner exports into a new bounded allocation.

Only the small final manifest crosses the controller. The worker retrieves exact
versioned bytes and TrainingRun admission checks learning semantics and counters.
"""

from pathlib import Path
import time
from typing import TYPE_CHECKING

from pydantic import Field

from manabot.training.models import LearningStateImport, TrainSelfPlay

from .job_client import job_manifest
from .jobs import Job, JobStatus
from .plan import Frozen, compile_plan
from .snapshots import JobManifest

if TYPE_CHECKING:
    from .cohort import CohortEntry


class Continuation(Frozen):
    job_id: str
    stage_id: str = "policy"
    iteration: int = Field(ge=1)


def bind_continuation(
    entry: "CohortEntry",
    status: JobStatus,
    cache: Path,
    *,
    manifest: JobManifest | None = None,
) -> Job:
    dependency = entry.continuation
    record = status.record
    if dependency is None or status.spec.job_id != dependency.job_id:
        raise ValueError("continuation predecessor differs")
    if (
        record is None
        or record.phase != "completed"
        or not record.artifacts_complete
        or record.updates != dependency.iteration
        or status.cleanup is None
    ):
        raise ValueError(
            "continuation requires completed, cleaned-up predecessor at exact iteration"
        )
    manifest = manifest or job_manifest(status.spec, record, cache)
    if (
        not manifest.complete
        or manifest.spec_sha256 != status.spec.identity
        or manifest.generation != record.generation
    ):
        raise ValueError("continuation manifest differs from final predecessor")
    names = {
        "source_run": "run/run.json",
        **{
            role: f"run/{dependency.stage_id}-{role}.pt"
            for role in ("raw", "ema", "optimizer")
        },
    }
    files = {
        item.relative_path: artifact
        for item, artifact in zip(
            manifest.bundle.files, manifest.artifacts, strict=True
        )
    }
    if not set(names.values()).issubset(files):
        raise ValueError("predecessor lacks raw/EMA/Adam/run exports")
    inputs = {role: files[name] for role, name in names.items()}
    regime = entry.plan.regime.model_copy(deep=True)
    stage = regime.stages[0]
    if not isinstance(stage, TrainSelfPlay) or stage.updates <= dependency.iteration:
        raise ValueError("continuation endpoint must exceed predecessor iteration")
    stage.learning_state = LearningStateImport.model_validate(
        {
            "source_stage": dependency.stage_id,
            **{
                role: {
                    "path": f"/opt/manabot/learning-inputs/{role}",
                    "sha256": artifact.sha256,
                    "bytes": artifact.bytes,
                }
                for role, artifact in inputs.items()
            },
        }
    )
    plan = compile_plan(
        regime.model_dump_json(), entry.plan.spec, entry.plan.source, entry.plan.seed
    )
    spec = entry.model_copy(update={"plan": plan}).job(time.time())
    return Job.model_validate(
        spec.model_copy(update={"learning_inputs": inputs}).model_dump()
    )
