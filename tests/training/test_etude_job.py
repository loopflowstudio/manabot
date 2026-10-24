"""Direct SDK worker proof: actual learner, admitted bytes and failure admission."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
import subprocess

from etude.train.read import read_all
import pytest

from experiments.ataraxos_move import ataraxos_move_scalar
from manabot.experiment.config import Baseline, Case, Experiment
from manabot.experiment.execution import PlannedRun
from manabot.experiment.job import TrainingJob, run_worker
from manabot.model.checkpoint import load_checkpoint
from manabot.training.execution import execute_regime
from manabot.training.records import TrainingRun
from manabot.training.regime import TrainSelfPlay
from manabot.verify.store import VerifyStore

ROOT = Path(__file__).resolve().parents[2]


def specification() -> TrainingJob:
    regime = ataraxos_move_scalar()
    regime.stages = regime.stages[:1]
    regime.agent.hidden_dim = 8
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.updates = 2
    stage.transitions = 4
    return TrainingJob(regime=regime, milestones=(1,))


def worker_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TrainingJob:
    spec = specification()
    config = tmp_path / "config.json"
    config.write_text(spec.model_dump_json())
    for key in ("ETUDE_CHECKPOINT", "ETUDE_FROM_VALUE"):
        monkeypatch.delenv(key, raising=False)
    for key, value in {
        "ETUDE_RUN": "sdk-worker-fixture",
        "ETUDE_TRACK_DIR": str(tmp_path / "tracking"),
        "ETUDE_CONFIG": str(config),
        "ETUDE_OUTPUT_DIR": str(tmp_path / "output"),
        "ETUDE_STOP_FILE": str(tmp_path / "stop"),
        "ETUDE_SEED": "12250",
        "ETUDE_UNTIL_CLOCK": "updates",
        "ETUDE_UNTIL_VALUE": "2",
        "ETUDE_DEADLINE": (
            datetime.now(timezone.utc)
            + timedelta(seconds=spec.regime.wall_seconds + 90)
        ).isoformat(),
    }.items():
        monkeypatch.setenv(key, value)
    return spec


def test_job_freezes_recipe_without_submission(tmp_path: Path) -> None:
    spec = specification()
    job = spec.job(repo=tmp_path, revision="fixture-commit", max_hours=1)
    assert job.max_attempts == 1
    assert job.until == "updates=2"
    assert job.config is not None
    assert TrainingJob.model_validate_json(job.config) == spec
    assert job.runs(11, 12)[1].seed == 12
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError, match="lifetime"):
        spec.job(repo=tmp_path, revision="fixture-commit", max_hours=0.0001)
    with pytest.raises(ValueError, match="milestones"):
        TrainingJob(regime=spec.regime, milestones=(2,))


def test_worker_runs_real_learner_and_reports_admitted_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker_environment(tmp_path, monkeypatch)
    subprocess.run(
        ["uv", "run", "--locked", "python", "-m", "manabot.experiment.job"],
        cwd=ROOT,
        check=True,
        timeout=30,
        capture_output=True,
        text=True,
    )
    run = TrainingRun.model_validate_json(
        (tmp_path / "output/training/run.json").read_text()
    )
    assert run.status == "completed"
    result = json.loads((tmp_path / "output/etude-result.json").read_text())
    assert result["clocks"] == {"updates": 2}
    load_checkpoint(str(tmp_path / "output" / result["checkpoint"]))
    tracked = read_all(tmp_path / "tracking")
    assert len(tracked) == 1
    assert tracked[0].name == "sdk-worker-fixture"
    assert [c.at["updates"] for c in tracked[0].checkpoints] == [0, 1, 2]
    with pytest.raises(ValueError, match="already contains"):
        run_worker()


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("ETUDE_CHECKPOINT", "policy.pt", "cannot resume"),
        ("ETUDE_UNTIL_VALUE", "3", "target differs"),
        ("ETUDE_DEADLINE", "2000-01-01T00:00:00+00:00", "remaining allocation"),
        ("ETUDE_RUN", "", "missing launcher identity"),
    ],
)
def test_worker_rejects_before_learning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: str,
    message: str,
) -> None:
    worker_environment(tmp_path, monkeypatch)
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match=message):
        run_worker()
    assert not (tmp_path / "output").exists()


def test_executor_cancellation_retains_failed_receipt(tmp_path: Path) -> None:
    stop = tmp_path / "stop"
    stop.touch()
    with VerifyStore(tmp_path / "store.sqlite") as store:
        with pytest.raises(RuntimeError, match="cancelled by launcher"):
            execute_regime(
                specification().regime, 12250, tmp_path / "run", store, stop_path=stop
            )
    run = TrainingRun.model_validate_json((tmp_path / "run/run.json").read_text())
    assert run.status == "failed"
    assert run.error is not None and "cancelled by launcher" in run.error
    assert run.selected_artifact is None
    assert (tmp_path / "run/run.json").exists()


def test_experiment_plans_become_etude_runs_with_their_seeds(tmp_path: Path) -> None:
    regime = specification().regime
    experiment = Experiment(
        "plans",
        Baseline.capture("fixture", regime),
        cases=(Case("control"),),
        jobs=(
            PlannedRun(case="plans-control", seed=11, max_hours=1, milestones=(1,)),
            PlannedRun(
                case="plans-control",
                seed=12,
                max_hours=2,
                max_usd=Decimal(3),
                validate_numerics=True,
            ),
        ),
    )
    runs = experiment.runs(repo=tmp_path, revision="fixture-commit")
    assert [run.seed for run in runs] == [11, 12]
    assert [run.params for run in runs] == [{"case": "plans-control"}] * 2
    assert [run.job.max_hours for run in runs] == [1, 2]
    assert runs[1].job.max_usd == Decimal(3)
    first, second = (
        TrainingJob.model_validate_json(run.job.config or "") for run in runs
    )
    assert first.milestones == (1,) and not first.validate_numerics
    assert second.validate_numerics
    assert first.regime.stages == experiment.resolve().cases[0].regime.stages
