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
from manabot.training.regime import DampedPolicyLearning, TrainSelfPlay


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
        assert stage.updates == 2048 and stage.games_per_stream == 1
        assert isinstance(stage.learning, DampedPolicyLearning)
        assert stage.learning.iteration_schedule is None
        assert stage.learning.rates(1)[0] == stage.learning.rates(2048)[0] == 1e-4
    runs = read_job_file(Path(gpu.__file__), ["--revision", "HEAD"])
    assert len(runs) == 1
    run = runs[0]
    config = gpu.StudyJob.model_validate_json(run.job.config or "")
    assert config.source_bundle.commit
    assert run.job.experiment == gpu.EXPERIMENT
    assert run.job.max_attempts == 1
    assert run.job.max_usd == Decimal("17.20")
    assert len(gpu.SEEDS) == len(set(gpu.SEEDS)) == 4


def test_plateau_requires_small_changes_not_just_a_nonsignificant_gain() -> None:
    flat = gpu.PlateauInterval(
        "typed_pool", 1792, 2048, paired_interval((0.0, 0.0, 0.0, 0.0))
    )
    improving = gpu.PlateauInterval(
        "typed_pool", 1792, 2048, paired_interval((0.1, 0.1, 0.1, 0.1))
    )
    uncertain = gpu.PlateauInterval(
        "typed_pool", 1792, 2048, paired_interval((-0.1, 0.1, -0.1, 0.1))
    )
    assert flat.bounded
    assert not improving.bounded
    assert uncertain.estimate.lower < 0 < uncertain.estimate.upper
    assert not uncertain.bounded


def test_gpu_cohort_reaps_siblings_after_failure(tmp_path: Path) -> None:
    commands = [
        [sys.executable, "-c", "raise SystemExit(7)"],
        [sys.executable, "-c", "import time; time.sleep(20)"],
    ]
    with pytest.raises(RuntimeError, match="child failed"):
        gpu._batch(commands, tmp_path, tmp_path / "stop", time.time() + 10)
    receipt = json.loads((tmp_path / "processes.json").read_text())
    assert receipt["exit_codes"][0] == 7
    assert all(code is not None for code in receipt["exit_codes"])


def test_throughput_admission_projects_whole_cohort_cost() -> None:
    # This time includes BOTH arms. Pairing affects measured time, not how many
    # training updates the full cohort needs. Never divide by concurrency twice.
    serial = gpu.ThroughputAdmission(
        concurrency=1, elapsed_seconds=320, updates_per_arm=8
    )
    paired = gpu.ThroughputAdmission(
        concurrency=2, elapsed_seconds=320, updates_per_arm=8
    )
    expected_hours = (1.5 * 320 * 4 * 2048 / 8 + 5400 + 900) / 3600
    assert serial.cohort_hours() == paired.cohort_hours() == expected_hours
    assert paired.projected_usd(Decimal("0.25")) > Decimal("25")
    with pytest.raises(ValueError, match="finite and positive"):
        paired.projected_usd(Decimal("NaN"))
    with pytest.raises(ValueError):
        gpu.ThroughputAdmission(
            concurrency=1, elapsed_seconds=math.inf, updates_per_arm=8
        )


def test_throughput_admission_is_bounded_and_excludes_scientific_seeds() -> None:
    (run,) = read_job_file(Path(gpu.__file__), ["--revision", "HEAD", "--admission"])
    assert gpu.StudyJob.model_validate_json(run.job.config or "").mode == "admission"
    assert run.job.max_usd == Decimal("0.50")
    assert run.job.max_hours == 1
    assert run.job.until == "updates=8"
    assert gpu.ADMISSION_SEED not in gpu.SEEDS
