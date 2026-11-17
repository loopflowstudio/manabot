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

A job may start a fresh learner from a producer's published policy through
PretrainedInitialization. Its record and checkpoint arrive as digest-checked
inputs, bound to PretrainedPolicy; ordinary admission retains producer lineage
and cost, while Adam, EMA and schedule iteration start fresh.

A job may instead continue a parent TrainingRun's learner (LearnerContinuation):
the parent's run record and final raw/EMA/Adam exports arrive as digest-checked
Etude inputs, the worker binds them as the stage's LearningStateImport, and
training resumes at the parent's absolute update toward the same absolute
target. Game streams and RNGs restart; this is a new segment, not recovery.
A parent interrupted before its final exports can be continued only by
declaration (`from_monitoring`): from the raw weights of one of its monitoring
exports, with the EMA reset to them and a fresh Adam.

A segment ends before its wall budget rather than at it: the stage stops at an
update boundary once another update plus `stop_reserve_seconds` no longer fits,
exports raw/EMA/Adam and ends `paused`. That paused segment is a successful
attempt reporting the update it reached, and its successor continues normally.
A launcher's SIGTERM ends the attempt `interrupted`, recording the run as failed
at its last export.

A job's runs track under its `experiment`, the experiment file's stem
(`experiments/elemental.py` is `elemental`), so `etude eval` and `etude report`
find them. Etude preserves that experiment in the worker environment.
"""

from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time
from types import FrameType
from typing import Literal

from etude.jobs import TrainingJob as EtudeTrainingJob
from etude.launch.records import Input, Job
from etude.storage import SAFE_NAME, S3Storage, Storage
from pydantic import Field, model_validator

from manabot.experiment.numerics import check_numerics
from manabot.infra.source import SourceBundle, bind_etude_source, bundle_at
from manabot.policy.protocol import Backend
from manabot.training.execution import execute_regime, validate_regime
from manabot.training.records import TrainingRun
from manabot.training.regime import (
    LearningStateImport,
    PretrainedPolicy,
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
_INITIAL_WEIGHTS = "initial-weights.pt"


class StagedExport(Strict):
    """One immutable input held in the machine's Etude storage under `key`."""

    key: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=1)

    def input(self, name: str) -> Input:
        """The Etude input that delivers this export as `name`."""
        return Input(name=name, key=self.key, sha256=self.sha256, size_bytes=self.bytes)

    def delivered(self, path: Path) -> dict[str, str | int]:
        """The regime's file reference for this export once delivered at `path`."""
        return {"path": str(path), "sha256": self.sha256, "bytes": self.bytes}


def _inputs(files: dict[str, tuple[str, StagedExport]]) -> tuple[Input, ...]:
    return tuple(export.input(name) for name, export in files.values())


def _delivered(
    files: dict[str, tuple[str, StagedExport]], inputs: Path
) -> dict[str, dict[str, str | int]]:
    """Import role -> file reference under the worker's inputs directory."""
    return {
        role: export.delivered(inputs / name) for role, (name, export) in files.items()
    }


class LearnerContinuation(Strict):
    """Continue a parent run's learner from its final exports.

    `from_update` is the parent's absolute completed update; the worker refuses
    a staged run record that ends anywhere else, or a seed other than the
    parent's. The exports must be the parent
    stage's own published raw/EMA/Adam artifacts (LearningStateImport admission
    compares them with the record).

    With `from_monitoring`, `from_update` instead names one of an interrupted
    parent's monitoring exports and only `raw` (that export's bytes) is staged;
    admission resets the EMA to it and starts Adam afresh.
    """

    from_update: int = Field(ge=1)
    source_stage: str
    source_run: StagedExport
    raw: StagedExport
    ema: StagedExport | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    optimizer: StagedExport | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    from_monitoring: bool = Field(default=False, exclude_if=lambda value: not value)

    @model_validator(mode="after")
    def declared_roles(self) -> "LearnerContinuation":
        staged = (self.ema is not None, self.optimizer is not None)
        if staged != ((False, False) if self.from_monitoring else (True, True)):
            raise ValueError(
                "a continuation stages raw/EMA/Adam, or raw alone from a"
                " declared monitoring export"
            )
        return self

    def files(self) -> dict[str, tuple[str, StagedExport]]:
        """Import role -> (input file name, staged export), for staged roles."""
        return {
            role: (name, export)
            for role, name in _PARENT_FILES.items()
            if (export := getattr(self, role)) is not None
        }

    def inputs(self) -> tuple[Input, ...]:
        return _inputs(self.files())


class PretrainedInitialization(Strict):
    """Start a fresh learner from a completed producer's published policy.

    Both the immutable TrainingRun record and its selected raw/EMA checkpoint
    are staged. Normal PretrainedPolicy admission checks their provenance,
    model/world contract and producer cost after the worker binds local paths.
    """

    source_stage: str
    source_seed: int
    source_run: StagedExport
    checkpoint: StagedExport
    weights: Literal["raw", "ema"] = "raw"

    @classmethod
    def from_storage(
        cls,
        storage: Storage,
        record_key: str,
        *,
        source_stage: str,
        weights: Literal["raw", "ema"] = "raw",
    ) -> "PretrainedInitialization":
        """Read a completed producer and verify its published weight bytes.

        Exports live beside the worker's run.json, under their recorded file
        names. Returned input keys include an S3 storage prefix because Etude
        inputs address the bucket directly. Missing objects propagate storage
        errors; incomplete producers and mismatched exports raise ValueError.
        Model/world and cost admission still belongs to the worker.
        """
        payload = storage.get(record_key)
        producer = TrainingRun.model_validate_json(payload)
        if producer.status != "completed":
            raise ValueError("pretrained initialization requires a completed producer")
        stage = next((s for s in producer.stages if s.id == source_stage), None)
        if stage is None or stage.status != "completed":
            raise ValueError("pretrained initialization requires a completed stage")
        artifact = stage.artifacts.get(weights)
        if artifact is None:
            raise ValueError(f"producer published no {weights} weight export")
        key = str(Path(record_key).parent / Path(artifact["path"]).name)
        checkpoint = storage.get(key)
        if (hashlib.sha256(checkpoint).hexdigest(), len(checkpoint)) != (
            artifact["sha256"],
            artifact["bytes"],
        ):
            raise ValueError("stored weight export hash/size mismatch with producer")
        prefix = storage.prefix if isinstance(storage, S3Storage) else ""
        return cls(
            source_stage=source_stage,
            source_seed=producer.seed,
            source_run=StagedExport(
                key=prefix + record_key,
                sha256=hashlib.sha256(payload).hexdigest(),
                bytes=len(payload),
            ),
            checkpoint=StagedExport(
                key=prefix + key,
                sha256=artifact["sha256"],
                bytes=artifact["bytes"],
            ),
            weights=weights,
        )

    def files(self) -> dict[str, tuple[str, StagedExport]]:
        return {
            "source_run": ("producer-run.json", self.source_run),
            "checkpoint": ("producer-weights.pt", self.checkpoint),
        }

    def inputs(self) -> tuple[Input, ...]:
        return _inputs(self.files())


RUST_TOOLCHAIN = "1.98.1"
# Etude machine preparation shared by every job that runs repository code.
TOOLCHAIN_ENV = (
    f'export PATH="$HOME/.cargo/bin:$PATH" RUSTUP_TOOLCHAIN={RUST_TOOLCHAIN}'
)


def machine_setup(extras: str) -> str:
    """Setup script installing the pinned Rust toolchain and locked venv.

    `extras` is appended verbatim to `uv sync` (e.g. `" --extra dev"`). Etude runs
    setup once per attempt, and attempts sharing a machine share `$HOME/.cargo`, so
    a `flock` serializes the toolchain install. `uv sync` stays outside it: each
    attempt has its own source directory, and uv and cargo lock their own caches.
    Linux machines supply `flock`; macOS has none, so a local attempt installs
    unserialized.
    """
    return (
        "set -eu\n"
        "{\n"
        "if command -v flock >/dev/null; then flock 9; fi\n"
        'test -x "$HOME/.cargo/bin/rustup" || curl --proto =https --tlsv1.2 -sSf '
        "https://sh.rustup.rs | sh -s -- -y --profile minimal "
        f"--default-toolchain {RUST_TOOLCHAIN}\n"
        f"{TOOLCHAIN_ENV}\n"
        f"rustup toolchain install {RUST_TOOLCHAIN} --profile minimal\n"
        '} 9>"$HOME/.manabot-rust-setup.lock"\n'
        f"uv sync --locked --python 3.13{extras}\n"
    )


class TrainingJob(Strict):
    """Resolved learning plus export coordinates; construction never launches.

    Milestones and the regime's `updates` are absolute; a continuation's
    milestones lie strictly between its `from_update` and the target.
    """

    regime: TrainingRegime
    # The Etude experiment its runs track under.
    experiment: str = Field(pattern=SAFE_NAME.pattern)
    milestones: tuple[int, ...] = ()
    finalization_seconds: float = Field(default=30, gt=0)
    # Inside the watchdog: the final raw/EMA/Adam exports after a wall-bounded
    # stop (execute_regime's stop_reserve_seconds). Omitted: 120 s, or a tenth
    # of the watchdog when that is shorter.
    stop_reserve_seconds: float | None = Field(
        default=None, gt=0, exclude_if=lambda value: value is None
    )
    # Run the CUDA numerics gate (manabot.experiment.numerics) before learning.
    validate_numerics: bool = Field(default=False, exclude_if=lambda value: not value)
    training_exposure: bool = Field(default=False, exclude_if=lambda value: not value)
    training_game_logs: bool = Field(default=False, exclude_if=lambda value: not value)
    # Frozen by `job` from the exact revision; absent only for local workers,
    # which read source identity from their git checkout.
    source_bundle: SourceBundle | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    initial_checkpoint: StagedExport | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    continuation: LearnerContinuation | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    pretrained: PretrainedInitialization | None = Field(
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
                stage.initial_weights,
                stage.active_seconds,
            )
        ):
            raise ValueError(
                "Etude training jobs require step-target self-play;"
                " starting state is bound from its inputs"
            )
        starting_states = (self.initial_checkpoint, self.continuation, self.pretrained)
        if sum(value is not None for value in starting_states) > 1:
            raise ValueError(
                "choose one starting state: initial weights, continuation or pretrained"
            )
        start = 0 if self.continuation is None else self.continuation.from_update
        if start >= stage.updates:
            raise ValueError("continuation must start before the update target")
        if self.stop_reserve >= self._watchdog_seconds:
            raise ValueError("the stop reserve must lie inside the regime watchdog")
        if tuple(sorted(set(self.milestones))) != self.milestones or any(
            n <= start or n >= stage.updates for n in self.milestones
        ):
            raise ValueError("milestones must be unique increasing interior updates")
        if self.continuation is not None or self.pretrained is not None:
            # Validate the bound starting-state contract before freezing a job;
            # byte/provenance admission runs after input delivery on the worker.
            self.bound_regime(Path("/inputs"))
        return self

    def bound_regime(self, inputs: Path) -> TrainingRegime:
        """Bind staged inputs to the regime's single starting state."""
        if self.pretrained is not None:
            field = "pretrained"
            state = PretrainedPolicy.model_validate(
                {
                    "source_stage": self.pretrained.source_stage,
                    "weights": self.pretrained.weights,
                    **_delivered(self.pretrained.files(), inputs),
                }
            ).model_dump(mode="json")
        elif self.initial_checkpoint is not None:
            field = "initial_weights"
            state = self.initial_checkpoint.delivered(inputs / _INITIAL_WEIGHTS)
        elif self.continuation is not None:
            field = "learning_state"
            state = LearningStateImport.model_validate(
                {
                    "source_stage": self.continuation.source_stage,
                    **(
                        {"monitoring_update": self.continuation.from_update}
                        if self.continuation.from_monitoring
                        else {}
                    ),
                    **_delivered(self.continuation.files(), inputs),
                }
            ).model_dump(mode="json")
        else:
            return self.regime
        serialized = self.regime.model_dump(mode="json")
        serialized["stages"][0][field] = state
        return TrainingRegime.model_validate(serialized)

    @property
    def _watchdog_seconds(self) -> float:
        stage = self.regime.stages[0]
        return min(self.regime.wall_seconds, stage.execution.wall_seconds)

    @property
    def stop_reserve(self) -> float:
        if self.stop_reserve_seconds is not None:
            return self.stop_reserve_seconds
        return min(120.0, self._watchdog_seconds / 10)

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
        if spec.pretrained is not None:
            inputs = spec.pretrained.inputs()
        elif spec.continuation is not None:
            inputs = spec.continuation.inputs()
        elif spec.initial_checkpoint is not None:
            inputs = (spec.initial_checkpoint.input(_INITIAL_WEIGHTS),)
        else:
            inputs = ()
        job = EtudeTrainingJob(
            name=spec.regime.id,
            train=train,
            config=spec,
            setup=machine_setup(extras),
            extras=("dev",) if spec.validate_numerics else (),
            repo=str(repo.resolve()),
            revision=revision,
            until=f"updates={spec.updates}",
            max_hours=max_hours,
            max_usd=max_usd,
            max_attempts=1,
            inputs=inputs,
        ).plain()
        return job.model_copy(update={"command": f"{TOOLCHAIN_ENV}\n{job.command}"})


def train(config: TrainingJob) -> None:
    """Run the frozen learner through Etude's typed Python call entry point."""
    # Typed worker entry point selects the backend only when invoked.
    from manabot.backends import resolve

    signal.signal(signal.SIGTERM, _interrupt)
    _run_worker(config, backend=resolve("torch"))


def run_worker() -> TrainingRun:
    """Read a worker invocation from its command-line environment."""
    # Command-line entry point selects the backend only when invoked.
    from manabot.backends import resolve

    return _run_worker(
        TrainingJob.model_validate_json(Path(os.environ["ETUDE_CONFIG"]).read_text()),
        backend=resolve("torch"),
    )


def _run_worker(spec: TrainingJob, *, backend: Backend) -> TrainingRun:
    """Consume SDK worker inputs, retaining failed runs without automatic retries."""

    environment = os.environ
    for key in ("ETUDE_RUN", "ETUDE_TRACK_DIR"):
        if not environment.get(key):
            raise ValueError(f"missing launcher identity: {key}")
    # A continuation is handed its parent's policy checkpoint, which cannot
    # restore collector/optimizer state; RunContext refuses a retry likewise.
    if environment.get("ETUDE_CONTINUES"):
        raise ValueError("policy artifacts cannot resume collector/optimizer state")
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
    regime = spec.regime
    if any(
        value is not None
        for value in (spec.continuation, spec.initial_checkpoint, spec.pretrained)
    ):
        directory = environment.get("ETUDE_INPUTS_DIR")
        inputs = Path(directory or "")
        if not directory or not inputs.is_dir():
            raise ValueError(
                "training initialization/continuation needs its staged inputs"
            )
        if spec.continuation is not None:
            parent = TrainingRun.model_validate_json(
                (inputs / _PARENT_FILES["source_run"]).read_bytes()
            )
            # A monitoring continuation's parent trained past its export;
            # admission matches the export's receipt at from_update instead.
            if not spec.continuation.from_monitoring and (
                parent.updates_through() != spec.continuation.from_update
            ):
                raise ValueError("staged parent ends at a different update")
            if parent.seed != int(environment["ETUDE_SEED"]):
                raise ValueError("a learner continuation keeps its parent's seed")
        if spec.pretrained is not None:
            # The producer record owns its seed; the declared label is a copy.
            # Mismatched bytes are left to PretrainedPolicy admission, which
            # retains a failed run naming the digest fault.
            declared = spec.pretrained.source_run
            staged = (inputs / spec.pretrained.files()["source_run"][0]).read_bytes()
            if (hashlib.sha256(staged).hexdigest(), len(staged)) == (
                declared.sha256,
                declared.bytes,
            ):
                producer = TrainingRun.model_validate_json(staged)
                if producer.seed != spec.pretrained.source_seed:
                    raise ValueError(
                        "staged producer has a different seed than declared"
                    )
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
            experiment=spec.experiment,
            checkpoint_updates=spec.milestones,
            stop_path=stop,
            stop_reserve_seconds=spec.stop_reserve,
            training_exposure=spec.training_exposure,
            training_game_logs=spec.training_game_logs,
            backend=backend,
        )
    finished = run.status == "completed" and run.updates_through() == spec.updates
    # A wall-bounded stop exported the complete learner state short of target.
    paused = (
        run.status == "paused"
        and run.updates_through() < spec.updates
        and {"raw", "ema", "optimizer"} <= set(run.stages[0].artifacts)
    )
    if not (finished or paused):
        raise RuntimeError(
            f"training did not complete target: {run.status}: {run.error}"
        )
    artifact = run.selected_artifact
    if artifact is None:
        raise RuntimeError("completed training has no admitted policy")
    # Completed targets and wall-bounded segments become launcher successes,
    # reporting the update reached. Private scientific receipts and rejected
    # attempts remain under output even on failure.
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


def _interrupt(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt(f"signal {signum}")


if __name__ == "__main__":
    # Etude ends an attempt with SIGTERM, within 0.1 s of a cancellation and well
    # before training polls its stop file. Unhandled, it kills the process with
    # no end state, so the run reads as running for good. KeyboardInterrupt
    # because game-error isolation catches Exception.
    signal.signal(signal.SIGTERM, _interrupt)
    run_worker()
