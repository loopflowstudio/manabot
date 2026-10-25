"""Etude-owned single-stage training jobs, without the legacy experiment scheduler.

TrainingJob freezes an ordinary TrainingRegime and exact export milestones into
an SDK Job. Etude owns placement, deadlines and attempt identity; execute_regime
still owns learning, policy admission and scientific TrainingRun receipts. This
entry point does not claim portable optimizer/collector recovery from a policy.

A job may instead continue a parent TrainingRun's learner (LearnerContinuation):
the parent's run record and final raw/EMA/Adam exports arrive as digest-checked
Etude inputs, the worker binds them as the stage's LearningStateImport, and
training resumes at the parent's absolute update toward the same absolute
target. Game streams and RNGs restart; this is a new segment, not recovery.
"""

from datetime import datetime, timezone
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import time

from etude.launch.records import Input, Job
from pydantic import Field, model_validator

from manabot.experiment.numerics import check_numerics
from manabot.training.execution import execute_regime, validate_regime
from manabot.training.records import TrainingRun
from manabot.training.regime import (
    LearningStateImport,
    Strict,
    TrainingRegime,
    TrainSelfPlay,
)
from manabot.verify.store import VerifyStore

# Parent files arrive at ETUDE_INPUTS_DIR/<name>.
_PARENT_FILES = {
    "source_run": "parent-run.json",
    "raw": "parent-raw.pt",
    "ema": "parent-ema.pt",
    "optimizer": "parent-optimizer.pt",
}


class StagedExport(Strict):
    """One parent file held in the machine's Etude storage under `key`."""

    key: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=1)


class LearnerContinuation(Strict):
    """Continue a parent run's learner from its final exports.

    `from_update` is the parent's absolute completed update; the worker refuses
    a staged run record that ends anywhere else, or a seed other than the
    parent's. The exports must be the parent
    stage's own published raw/EMA/Adam artifacts (LearningStateImport admission
    compares them with the record).
    """

    from_update: int = Field(ge=1)
    source_stage: str
    source_run: StagedExport
    raw: StagedExport
    ema: StagedExport
    optimizer: StagedExport

    def files(self) -> dict[str, tuple[str, StagedExport]]:
        """Import role -> (input file name, staged export)."""
        return {
            role: (name, getattr(self, role)) for role, name in _PARENT_FILES.items()
        }

    def inputs(self) -> tuple[Input, ...]:
        return tuple(
            Input(
                name=name,
                key=export.key,
                sha256=export.sha256,
                size_bytes=export.bytes,
            )
            for name, export in self.files().values()
        )


class TrainingJob(Strict):
    """Resolved learning plus export coordinates; construction never launches.

    Milestones and the regime's `updates` are absolute; a continuation's
    milestones lie strictly between its `from_update` and the target.
    """

    regime: TrainingRegime
    milestones: tuple[int, ...] = ()
    finalization_seconds: float = Field(default=30, gt=0)
    # Run the CUDA numerics gate (manabot.experiment.numerics) before learning.
    validate_numerics: bool = Field(default=False, exclude_if=lambda value: not value)
    continuation: LearnerContinuation | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def admit(self) -> "TrainingJob":
        if len(self.regime.stages) != 1:
            raise ValueError("Etude training jobs require one self-play stage")
        stage = self.regime.stages[0]
        if not isinstance(stage, TrainSelfPlay) or any(
            value is not None
            for value in (
                stage.initial,
                stage.learning_state,
                stage.pretrained,
                stage.active_seconds,
            )
        ):
            raise ValueError(
                "Etude training jobs require step-target self-play;"
                " a learner continuation is bound from its inputs"
            )
        start = 0 if self.continuation is None else self.continuation.from_update
        if start >= stage.updates:
            raise ValueError("continuation must start before the update target")
        if tuple(sorted(set(self.milestones))) != self.milestones or any(
            n <= start or n >= stage.updates for n in self.milestones
        ):
            raise ValueError("milestones must be unique increasing interior updates")
        if self.continuation is not None:
            # Regime admission of the bound import (EMA, damped absolute-step
            # learning, one stage) happens here, before any job is frozen.
            self.bound_regime(Path("/inputs"))
        return self

    def bound_regime(self, inputs: Path) -> TrainingRegime:
        """The regime with the continuation's staged files bound as its import."""
        if self.continuation is None:
            return self.regime
        serialized = self.regime.model_dump(mode="json")
        serialized["stages"][0]["learning_state"] = LearningStateImport.model_validate(
            {
                "source_stage": self.continuation.source_stage,
                **{
                    role: {
                        "path": str(inputs / name),
                        "sha256": export.sha256,
                        "bytes": export.bytes,
                    }
                    for role, (name, export) in self.continuation.files().items()
                },
            }
        ).model_dump(mode="json")
        return TrainingRegime.model_validate(serialized)

    @property
    def updates(self) -> int:
        stage = self.regime.stages[0]
        assert isinstance(stage, TrainSelfPlay)
        return stage.updates

    def job(
        self,
        *,
        repo: Path,
        revision: str,
        max_hours: float,
        max_usd: Decimal | None = None,
    ) -> Job:
        """Freeze SDK input without submitting or granting any allocation.

        The selected machine must already provide the pinned native extension.
        Revision is explicit; no working-tree snapshot silently enters science.
        """
        spec = TrainingJob.model_validate(self.model_dump())
        if not revision.strip():
            raise ValueError("an explicit source revision is required")
        if not math.isfinite(max_hours) or max_hours * 3600 < (
            spec.regime.wall_seconds + spec.finalization_seconds
        ):
            raise ValueError("job lifetime must cover regime and finalization reserve")
        return Job(
            name=spec.regime.id,
            command="uv run --locked python -m manabot.experiment.job",
            config=spec.model_dump_json(),
            repo=str(repo.resolve()),
            revision=revision,
            until=f"updates={spec.updates}",
            max_hours=max_hours,
            max_usd=max_usd,
            max_attempts=1,
            inputs=() if spec.continuation is None else spec.continuation.inputs(),
        )


def run_worker() -> TrainingRun:
    """Consume SDK worker inputs, retaining failed runs without automatic retries."""
    environment = os.environ
    for key in ("ETUDE_RUN", "ETUDE_TRACK_DIR"):
        if not environment.get(key):
            raise ValueError(f"missing launcher identity: {key}")
    # A continuation is handed its parent's policy checkpoint, which cannot
    # restore collector/optimizer state; RunContext refuses a retry likewise.
    if environment.get("ETUDE_CONTINUES"):
        raise ValueError("policy artifacts cannot resume collector/optimizer state")
    spec = TrainingJob.model_validate_json(
        Path(environment["ETUDE_CONFIG"]).read_text()
    )
    if (
        environment.get("ETUDE_UNTIL_CLOCK") != "updates"
        or float(environment["ETUDE_UNTIL_VALUE"]) != spec.updates
    ):
        raise ValueError("launcher target differs from the frozen regime")
    deadline = datetime.fromisoformat(environment["ETUDE_DEADLINE"])
    if deadline.tzinfo is None:
        raise ValueError("launcher deadline must include timezone")
    remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
    if remaining < spec.regime.wall_seconds + spec.finalization_seconds:
        raise ValueError("remaining allocation cannot cover the frozen regime")
    regime = spec.regime
    if spec.continuation is not None:
        directory = environment.get("ETUDE_INPUTS_DIR")
        inputs = Path(directory or "")
        if not directory or not inputs.is_dir():
            raise ValueError("a learner continuation needs its staged inputs")
        parent = TrainingRun.model_validate_json(
            (inputs / _PARENT_FILES["source_run"]).read_bytes()
        )
        if parent.updates_through() != spec.continuation.from_update:
            raise ValueError("staged parent ends at a different update")
        if parent.seed != int(environment["ETUDE_SEED"]):
            raise ValueError("a learner continuation keeps its parent's seed")
        regime = spec.bound_regime(inputs)
    # Native world admission precedes any collector or optimizer work. Never
    # shrink the recipe to fit a late start or silently substitute a device.
    regime = validate_regime(regime)
    output = Path(environment["ETUDE_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "etude-result.json"
    if result_path.exists() or (output / "training").exists():
        raise ValueError("attempt output already contains training evidence")
    stop = Path(environment["ETUDE_STOP_FILE"])
    if stop.exists():
        raise ValueError("launcher requested stop before training admission")
    if spec.validate_numerics:
        # The gate spends at most five minutes of the attempt's own allowance.
        gate = check_numerics(output, min(time.time() + 300, deadline.timestamp()))
        if gate.status != "completed":
            raise RuntimeError(f"numerics gate did not pass: {gate.status}")
    with VerifyStore(output / "training.sqlite") as store:
        run = execute_regime(
            regime,
            int(environment["ETUDE_SEED"]),
            output / "training",
            store,
            checkpoint_updates=spec.milestones,
            stop_path=stop,
        )
    if run.status != "completed" or run.updates_through() != spec.updates:
        raise RuntimeError(
            f"training did not complete target: {run.status}: {run.error}"
        )
    artifact = run.selected_artifact
    if artifact is None:
        raise RuntimeError("completed training has no admitted policy")
    # Only completed targets become launcher successes. Private scientific
    # receipts and rejected attempts remain under output even on failure.
    result = {
        "clocks": {"updates": run.updates_through()},
        "checkpoint": str(
            Path(artifact["path"]).resolve().relative_to(output.resolve())
        ),
    }
    temporary = result_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, sort_keys=True) + "\n")
    temporary.replace(result_path)
    return run


if __name__ == "__main__":
    run_worker()
