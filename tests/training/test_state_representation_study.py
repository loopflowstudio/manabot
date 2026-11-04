"""Scientific summaries treat seeds as units and refuse incomplete comparisons."""

import math
from pathlib import Path

from etude.launch.job_file import read_job_file
import pytest

from experiments import state_representation_v7
from experiments.state_representation_v7 import (
    compare,
    evaluation_protocol,
    paired_interval,
    recipe,
)
from manabot.experiment.job import TrainingJob
from manabot.training.regime import TrainSelfPlay


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
        assert scientific.observation == smoke.observation
        assert scientific.id != smoke.id
    protocol = evaluation_protocol()
    assert protocol.legs == 4 and len(protocol.deal_seeds) == 32
    assert protocol.study_protocol_sha256 is not None
    assert protocol.purpose == "frozen-study-evaluation"


def test_declared_runs_are_visible_to_the_experiment_report() -> None:
    path = Path(state_representation_v7.__file__)
    runs = read_job_file(path, ["--revision", "HEAD"])
    jobs = [TrainingJob.model_validate_json(run.job.config or "") for run in runs]
    assert len(runs) == 8
    assert {job.experiment for job in jobs} == {path.stem}
    assert sorted(run.seed for run in runs) == sorted([73201, 73202, 73203, 73204] * 2)
