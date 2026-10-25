"""Etude-owned single-stage training jobs, without the legacy experiment scheduler.

TrainingJob freezes an ordinary TrainingRegime and exact export milestones into
an SDK Job. Etude owns placement, deadlines and attempt identity; execute_regime
still owns learning, policy admission and scientific TrainingRun receipts. This
entry point does not claim portable optimizer/collector recovery from a policy.

Etude's stock machine supplies uv but no Rust, while maturin builds the native
engine during the locked sync. The job's setup installs the pinned toolchain and
syncs once; the command reuses that toolchain if uv must rebuild. Etude unpacks
the commit without `.git`, so the job also carries the commit's source bundle,
which the worker verifies and binds before anything reads source identity.
"""

from datetime import datetime, timezone
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import time

from etude.launch.records import Job
from pydantic import Field, model_validator

from manabot.experiment.numerics import check_numerics
from manabot.infra.source import SourceBundle, bind_etude_source, bundle_at
from manabot.training.execution import execute_regime, validate_regime
from manabot.training.records import TrainingRun
from manabot.training.regime import Strict, TrainingRegime, TrainSelfPlay
from manabot.verify.store import VerifyStore

RUST_TOOLCHAIN = "1.98.1"
_TOOLCHAIN = f'export PATH="$HOME/.cargo/bin:$PATH" RUSTUP_TOOLCHAIN={RUST_TOOLCHAIN}'


def _setup(extras: str) -> str:
    return (
        "set -eu\n"
        'test -x "$HOME/.cargo/bin/rustup" || curl --proto =https --tlsv1.2 -sSf '
        "https://sh.rustup.rs | sh -s -- -y --profile minimal "
        f"--default-toolchain {RUST_TOOLCHAIN}\n"
        f"{_TOOLCHAIN}\n"
        f"rustup toolchain install {RUST_TOOLCHAIN} --profile minimal\n"
        f"uv sync --locked --python 3.12{extras}\n"
    )


class TrainingJob(Strict):
    """Resolved learning plus export coordinates; construction never launches."""

    regime: TrainingRegime
    milestones: tuple[int, ...] = ()
    finalization_seconds: float = Field(default=30, gt=0)
    # Run the CUDA numerics gate (manabot.experiment.numerics) before learning.
    validate_numerics: bool = Field(default=False, exclude_if=lambda value: not value)
    # Frozen by `job` from the exact revision; absent only for local workers,
    # which read source identity from their git checkout.
    source_bundle: SourceBundle | None = Field(
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
            raise ValueError("Etude training jobs require fresh step-target self-play")
        if tuple(sorted(set(self.milestones))) != self.milestones or any(
            n <= 0 or n >= stage.updates for n in self.milestones
        ):
            raise ValueError("milestones must be unique increasing interior updates")
        return self

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
        if not revision.strip():
            raise ValueError("an explicit source revision is required")
        spec = TrainingJob.model_validate(
            {**self.model_dump(), "source_bundle": bundle_at(repo, revision)}
        )
        if not math.isfinite(max_hours) or max_hours * 3600 < (
            spec.regime.wall_seconds + spec.finalization_seconds
        ):
            raise ValueError("job lifetime must cover regime and finalization reserve")
        # The numerics gate runs the frozen pytest suite from the job's own venv.
        extras = " --extra dev" if spec.validate_numerics else ""
        return Job(
            name=spec.regime.id,
            setup=_setup(extras),
            command=f"{_TOOLCHAIN}\nuv run --locked{extras} python -m manabot.experiment.job",
            config=spec.model_dump_json(),
            repo=str(repo.resolve()),
            revision=revision,
            until=f"updates={spec.updates}",
            max_hours=max_hours,
            max_usd=max_usd,
            max_attempts=1,
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
    # Bind the unpacked archive before anything reads source identity.
    if spec.source_bundle is not None:
        bind_etude_source()
    deadline = datetime.fromisoformat(environment["ETUDE_DEADLINE"])
    if deadline.tzinfo is None:
        raise ValueError("launcher deadline must include timezone")
    remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
    if remaining < spec.regime.wall_seconds + spec.finalization_seconds:
        raise ValueError("remaining allocation cannot cover the frozen regime")
    # Native world admission precedes any collector or optimizer work. Never
    # shrink the recipe to fit a late start or silently substitute a device.
    regime = validate_regime(spec.regime)
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
