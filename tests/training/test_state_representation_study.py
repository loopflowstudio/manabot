"""Scientific summaries treat seeds as units and refuse incomplete comparisons."""

from decimal import Decimal
import json
import math
from pathlib import Path
import sys
import time

from etude.launch.job_file import read_job_file
import pytest

from experiments import state_representation_gpu as gpu, state_representation_v7
from experiments.state_representation_v7 import (
    compare,
    evaluation_protocol,
    paired_interval,
    recipe,
)
from manabot.env.observation import ObservationConfig
from manabot.experiment.job import TrainingJob
from manabot.infra.identity import canonical_sha256
from manabot.training.records import MonitoringCheckpoint, StageRecord, TrainingRun
from manabot.training.regime import (
    ArtifactReference,
    DampedPolicyLearning,
    TrainingRegime,
    TrainSelfPlay,
)
from manabot.verify.store import VerifyStore


def test_paired_interval_uses_independent_seeds() -> None:
    # Four variable seed differences retain wide uncertainty, regardless of how
    # many correlated games produced each mean.
    estimate = paired_interval((0.1, 0.2, -0.1, 0.0))
    assert estimate.mean == pytest.approx(0.05)
    assert estimate.lower < 0 < estimate.upper
    assert estimate.upper - estimate.lower == pytest.approx(0.41085205135)
    with pytest.raises(ValueError, match="four"):
        paired_interval((0.1, 0.2, 0.0))
    with pytest.raises(ValueError, match="finite"):
        paired_interval((0.1, 0.2, 0.0, math.nan))
    with pytest.raises(ValueError, match="every"):
        compare([])


def test_science_and_smoke_have_distinct_endpoints() -> None:
    for arm in ("typed_pool", "typed_gru"):
        scientific = recipe(arm)
        smoke = recipe(arm, smoke=True)
        stage = scientific.stages[0]
        assert isinstance(stage, TrainSelfPlay)
        assert stage.learning_seats == "both" and stage.games_per_stream == 1
        assert stage.updates == 128 and smoke.stages[0].updates == 1
        assert stage.streams == 32
        assert scientific.observation.policy_input_version == 7
        assert scientific.observation.max_actions == 64
        assert smoke.observation.max_actions == ObservationConfig().max_actions
        assert scientific.observation.model_dump(exclude={"max_actions"}) == (
            smoke.observation.model_dump(exclude={"max_actions"})
        )
        assert scientific.id != smoke.id
    protocol = evaluation_protocol()
    assert protocol.legs == 4 and len(protocol.deal_seeds) == 32
    assert protocol.study_protocol_sha256 is not None
    assert protocol.purpose == "frozen-study-evaluation"


def test_scientific_recipes_retain_executed_identities() -> None:
    # TrainingRun receipts from source 92becff60, independently retained in the
    # artifact manifest. Fresh defaults must not redefine completed experiments.
    expected = {
        "typed_pool": "600f73d3be9038c4d90fb2e439704d0b6dc250783eaacbc630cd85a34ff79b4e",
        "typed_gru": "70aa7246bc3e99e77ccc378ff95d3da0e34489eacb0d7f1c18e0da857c142604",
    }
    for arm in state_representation_v7.ARMS:
        assert canonical_sha256(recipe(arm).model_dump(mode="json")) == expected[arm]


def test_declared_runs_are_visible_to_the_experiment_report() -> None:
    path = Path(state_representation_v7.__file__)
    runs = read_job_file(path, ["--revision", "HEAD"])
    jobs = [TrainingJob.model_validate_json(run.job.config or "") for run in runs]
    assert len(runs) == 8
    assert {job.experiment for job in jobs} == {path.stem}
    assert sorted(run.seed for run in runs) == sorted([73201, 73202, 73203, 73204] * 2)


def test_gpu_horizon_keeps_learning_and_separates_scoring() -> None:
    assert gpu.plateau([]).reached is False
    assert set(gpu.evaluation_protocol().deal_seeds).isdisjoint(
        gpu.evaluation_protocol(heldout=True).deal_seeds
    )
    for arm in gpu.ARMS:
        stage = gpu.recipe(arm).stages[0]
        assert isinstance(stage, TrainSelfPlay)
        assert stage.execution.device == "cuda"
        assert stage.updates * stage.streams == 65536
        assert stage.games_per_stream == 1
        assert stage.streams == 256
        assert isinstance(stage.learning, DampedPolicyLearning)
        assert stage.learning.iteration_schedule is not None
        assert stage.learning.iteration_schedule.steps == 256
        assert stage.learning.iteration_schedule.reference_steps == 2048
        assert stage.learning.rates(1)[0] == stage.learning.rates(256)[0] == 1e-4
    runs = read_job_file(Path(gpu.__file__), ["--revision", "HEAD"])
    assert len(runs) == 1
    run = runs[0]
    config = gpu.StudyJob.model_validate_json(run.job.config or "")
    assert config.source_bundle.commit
    assert run.job.experiment == gpu.EXPERIMENT
    assert run.job.max_attempts == 1
    assert run.job.max_usd == Decimal("0.50")
    assert run.job.max_hours == 0.5
    assert run.job.until == "updates=4"
    assert gpu.ADMISSION_SEED not in gpu.SEEDS
    assert len(gpu.SEEDS) == len(set(gpu.SEEDS)) == 4


def test_plateau_requires_small_changes_not_just_a_nonsignificant_gain() -> None:
    flat = gpu.PlateauInterval(
        "typed_pool", *gpu.POINTS[-2:], paired_interval((0.0, 0.0, 0.0, 0.0))
    )
    improving = gpu.PlateauInterval(
        "typed_pool", *gpu.POINTS[-2:], paired_interval((0.1, 0.1, 0.1, 0.1))
    )
    uncertain = gpu.PlateauInterval(
        "typed_pool", *gpu.POINTS[-2:], paired_interval((-0.1, 0.1, -0.1, 0.1))
    )
    assert flat.bounded
    assert not improving.bounded
    assert uncertain.estimate.lower < 0 < uncertain.estimate.upper
    assert not uncertain.bounded


@pytest.mark.parametrize("exit_code", (0, 7))
def test_gpu_admission_retains_child_exit(tmp_path: Path, exit_code: int) -> None:
    command = [sys.executable, "-c", f"raise SystemExit({exit_code})"]
    if exit_code:
        with pytest.raises(RuntimeError, match="child failed: exit 7"):
            gpu._run_child(command, tmp_path, tmp_path / "stop", time.time() + 10)
    else:
        gpu._run_child(command, tmp_path, tmp_path / "stop", time.time() + 10)
    receipt = json.loads((tmp_path / "processes.json").read_text())
    assert receipt["exit_codes"] == [exit_code]


@pytest.mark.parametrize("cancelled", (False, True))
def test_gpu_admission_reaps_child_on_stop(tmp_path: Path, cancelled: bool) -> None:
    stop = tmp_path / "stop"
    if cancelled:
        stop.touch()
    deadline = time.time() + 10 if cancelled else time.time() - 1
    command = [sys.executable, "-c", "import time; time.sleep(20)"]
    with pytest.raises(RuntimeError, match="cancelled or allocation expired"):
        gpu._run_child(command, tmp_path, stop, deadline)
    receipt = json.loads((tmp_path / "processes.json").read_text())
    (exit_code,) = receipt["exit_codes"]
    assert exit_code is not None and exit_code != 0


def test_gpu_admission_retains_failed_start(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        gpu._run_child(
            [str(tmp_path / "missing-worker")],
            tmp_path,
            tmp_path / "stop",
            time.time() + 10,
        )
    receipt = json.loads((tmp_path / "processes.json").read_text())
    assert receipt["exit_codes"] == []


def test_throughput_admission_projects_whole_cohort_cost() -> None:
    # Time includes BOTH arms. The full scientific workload remains 65,536
    # complete games per learner despite batching into fewer, larger updates.
    measured = gpu.ThroughputAdmission(
        concurrency=1, elapsed_seconds=320, updates_per_arm=4
    )
    expected_hours = (1.5 * 320 * 4 * 256 / 4 + 5400 + 900) / 3600
    assert measured.cohort_hours() == expected_hours
    assert measured.projected_usd(Decimal("0.25")) == pytest.approx(
        Decimal(str(expected_hours)) * Decimal("0.25")
    )
    with pytest.raises(ValueError, match="finite and positive"):
        measured.projected_usd(Decimal("NaN"))
    with pytest.raises(ValueError):
        gpu.ThroughputAdmission(
            concurrency=1, elapsed_seconds=math.inf, updates_per_arm=4
        )


def test_scientific_allocation_rejects_unaffordable_cohort(tmp_path: Path) -> None:
    allocation = gpu.ScientificAllocation(
        admission_sha256="a" * 64,
        admission=gpu.ThroughputAdmission(
            concurrency=1, elapsed_seconds=120, updates_per_arm=4
        ),
        hourly_usd=Decimal("0.60"),
        max_usd=Decimal("16.50"),
        hardware="fixture-gpu",
    )
    document = tmp_path / "allocation.json"
    document.write_text(allocation.model_dump_json())
    (run,) = read_job_file(
        Path(gpu.__file__), ["--revision", "HEAD", "--allocation", str(document)]
    )
    spec = gpu.StudyJob.model_validate_json(run.job.config or "")
    assert spec.allocation == allocation
    assert run.job.until == "updates=256"
    assert run.job.max_hours == allocation.work_hours
    assert run.job.max_usd == Decimal("16.50")
    for change in ({"max_usd": "0.10"}, {"max_usd": "18"}):
        with pytest.raises(ValueError):
            gpu.ScientificAllocation.model_validate(
                {**allocation.model_dump(), **change}
            )


def _curve_run() -> TrainingRun:
    regime = gpu.recipe("typed_pool")
    artifact: ArtifactReference = {
        "path": "/fixture.pt",
        "sha256": "a" * 64,
        "bytes": 1,
    }
    return TrainingRun(
        id="synthetic-curve",
        regime=regime,
        regime_digest=canonical_sha256(regime.model_dump(mode="json")),
        seed=gpu.SEEDS[0],
        seed_streams={},
        identities={},
        status="completed",
        stages=[
            StageRecord(
                id=regime.stages[0].id,
                status="completed",
                updates=gpu.UPDATES,
                cumulative_seconds=100,
                games=gpu.UPDATES * gpu.STREAMS,
                artifacts={"initial_raw": artifact, "raw": artifact},
            )
        ],
        monitoring_checkpoints=[
            MonitoringCheckpoint(
                stage_id=regime.stages[0].id,
                ordinal=index,
                updates=update,
                training_seconds=update,
                games=update * gpu.STREAMS,
                artifact=artifact,
            )
            for index, update in enumerate(gpu.POINTS[1:-1])
        ],
    )


@pytest.mark.parametrize("admission", (False, True))
@pytest.mark.parametrize("completed", (False, True))
def test_gpu_training_preserves_admission_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, admission: bool, completed: bool
) -> None:
    seed = gpu.ADMISSION_SEED if admission else gpu.SEEDS[0]
    stop = tmp_path / "stop"
    out = tmp_path / "learner"

    def execute(
        regime: TrainingRegime,
        actual_seed: int,
        directory: Path,
        store: VerifyStore,
        *,
        experiment: str,
        checkpoint_updates: tuple[int, ...],
        stop_path: Path,
        stop_reserve_seconds: int,
    ) -> TrainingRun:
        assert actual_seed == seed
        assert directory == out / "training"
        assert stop_path == stop and stop_reserve_seconds == 120
        stage = regime.stages[0]
        assert stage.updates == (4 if admission else 256)
        assert regime.wall_seconds == (600 if admission else 66000)
        assert stage.execution.wall_seconds == regime.wall_seconds
        assert checkpoint_updates == (() if admission else gpu.POINTS[1:-1])
        assert experiment == (
            f"{gpu.EXPERIMENT}_admission" if admission else gpu.EXPERIMENT
        )
        run = _curve_run()
        run.stages[0].updates = stage.updates if completed else stage.updates - 1
        run.status = "completed" if completed else "paused"
        return run

    def set_memory_fraction(fraction: float) -> None:
        assert fraction == 0.80

    monkeypatch.setattr(gpu, "execute_regime", execute)
    monkeypatch.setattr(gpu.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        gpu.torch.cuda, "set_per_process_memory_fraction", set_memory_fraction
    )
    monkeypatch.setattr(gpu.torch.cuda, "get_device_name", lambda index: "fixture-gpu")
    monkeypatch.setattr(gpu.torch.cuda, "max_memory_reserved", lambda: 1024)
    if completed:
        gpu.train_arm("typed_pool", seed, out, stop, admission=admission)
    else:
        error = RuntimeError if admission else ValueError
        with pytest.raises(error, match="incomplete admission|completed training"):
            gpu.train_arm("typed_pool", seed, out, stop, admission=admission)
    assert json.loads((out / "device.json").read_text())["device"] == "fixture-gpu"
    wrong_seed = gpu.SEEDS[0] if admission else gpu.ADMISSION_SEED
    with pytest.raises(ValueError, match="undeclared training seed"):
        gpu.train_arm("typed_pool", wrong_seed, out, stop, admission=admission)


def test_gpu_curve_requires_every_export_and_keeps_exposure() -> None:
    run = _curve_run()
    points = gpu.checkpoints(run)
    assert tuple(point.coordinates.updates for point in points) == gpu.POINTS
    assert tuple(point.coordinates.games for point in points) == tuple(
        update * gpu.STREAMS for update in gpu.POINTS
    )
    run.monitoring_checkpoints.pop()
    with pytest.raises(ValueError, match="missing, duplicate or undeclared"):
        gpu.checkpoints(run)


@pytest.mark.parametrize("failure", ("duplicate", "export", "paused"))
def test_gpu_curve_refuses_failed_or_duplicate_endpoints(failure: str) -> None:
    run = _curve_run()
    if failure == "duplicate":
        run.monitoring_checkpoints.append(run.monitoring_checkpoints[0])
    elif failure == "export":
        run.monitoring_checkpoints[0].error = "export failed"
    else:
        run.status = "paused"
    with pytest.raises(ValueError):
        gpu.checkpoints(run)


def test_gpu_heldout_requires_all_seeds() -> None:
    with pytest.raises(ValueError, match="every held-out"):
        gpu.compare_heldout([])
