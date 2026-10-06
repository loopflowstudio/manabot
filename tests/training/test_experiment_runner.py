"""Shared Experiment execution, placement and editable notebook acceptance."""

from dataclasses import replace
import json
from pathlib import Path
import socket

from nbclient import NotebookClient
import nbformat
import pytest

from experiments.runners import depth_screen, history_input
from experiments.runners.experiment_demo import declaration as demo_declaration
from experiments.runners.experiment_depth import declaration as depth_declaration
from experiments.runners.experiment_history import declaration as history_declaration
from experiments.runners.experiment_screen import admit_screen
from experiments.runners.run_history_input import history_plan
from manabot.training.checkpoint_queue import MonitoringBudget
from manabot.training.comparison_notebook import write_comparison_notebook
from manabot.training.experiment_execution import (
    ExperimentSchedule,
    Hardware,
    HardwareInventory,
)
from manabot.training.experiment_runner import run_experiment
from manabot.training.monitor_evaluation import MonitorProtocol
from manabot.verify.store import VerifyStore
from tests.training.test_history_input import calibration


def hardware() -> HardwareInventory:
    return HardwareInventory(
        resources=(Hardware(name="fixture", host=socket.gethostname(), cpu_threads=2),)
    )


def test_screen_declarations_preserve_frozen_recipes_and_cohorts() -> None:
    for spec, factory in (
        (history_input, history_declaration),
        (depth_screen, depth_declaration),
    ):
        plan = history_plan(calibration(spec=spec), spec=spec)
        schedule = ExperimentSchedule(
            seeds=spec.SEEDS,
            order=spec.ORDER,
            scientific_deal_seeds=spec.DEALS,
            hardware="fixture",
            wall_seconds=spec.TRAINING_SECONDS,
            process_seconds=spec.TRAINING_SECONDS,
            monitoring=MonitoringBudget(seconds=600, attempt_seconds=60),
        )
        experiment = factory(plan, schedule)
        admit_screen(plan, experiment)
        assert experiment.resolve().digests == plan.protocol.regime_digests
        with pytest.raises(ValueError, match="seed/order"):
            admit_screen(
                plan,
                replace(experiment, schedule=schedule.model_copy(update={"order": ()})),
            )


def test_configured_hardware_and_budget_admission(tmp_path: Path) -> None:
    experiment = demo_declaration()
    with pytest.raises(ValueError, match="not configured"):
        run_experiment(experiment, HardwareInventory(resources=()), tmp_path / "absent")
    assert not (tmp_path / "absent").exists()
    with pytest.raises(ValueError, match="not this host"):
        Hardware(
            name="foreign", host="unconfigured.remote.invalid", cpu_threads=2
        ).admit()
    with pytest.raises(ValueError, match="disjoint"):
        ExperimentSchedule(
            seeds=(1,),
            hardware="fixture",
            wall_seconds=30,
            process_seconds=30,
            monitoring=MonitoringBudget(
                seconds=10,
                attempt_seconds=5,
                protocol=MonitorProtocol(deal_seeds=(900001,)),
            ),
            scientific_deal_seeds=(900001,),
        )


def test_tiny_comparison_evaluates_live_and_notebook_refresh_preserves_edits(
    tmp_path: Path,
) -> None:
    experiment = demo_declaration()
    out = tmp_path / "comparison"
    record = run_experiment(experiment, hardware(), out)
    assert record.status == "completed", (
        record.model_dump_json(),
        list(out.glob("*.log")),
    )
    assert len(record.attempts) == 4
    assert all(a.run_id for a in record.attempts)
    evaluation_attempts = [
        json.loads(p.read_text())
        for p in (out / "monitoring").glob("attempt-*/attempt.json")
    ]
    assert len(evaluation_attempts) == 4
    assert min(a["started_unix"] for a in evaluation_attempts) < max(
        a.finished_unix for a in record.attempts
    )
    assert record.process_seconds == pytest.approx(
        sum(a.process_seconds for a in record.attempts) + record.evaluator_seconds
    )
    with VerifyStore(out / "experiment.sqlite") as store:
        assert store.experiment_run(record.id) == record
    notebook_path = Path(record.notebook)
    notebook = nbformat.read(notebook_path, as_version=4)
    notebook.cells.append(nbformat.v4.new_code_cell("personal_note = 'keep this edit'"))
    nbformat.write(notebook, notebook_path)
    before = notebook_path.read_bytes()
    write_comparison_notebook(out, notebook_path)
    assert notebook_path.read_bytes() == before
    NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(out)}},
    ).execute()
    assert (
        sum(
            "image/png" in output.get("data", {})
            for cell in notebook.cells
            for output in cell.get("outputs", [])
        )
        == 0
    )
    assert "<svg" in (out / "comparison.html").read_text()
    assert not any(
        output.output_type == "error"
        for cell in notebook.cells
        for output in cell.get("outputs", [])
    )
    # Persist the executed artifact; a later report refresh must preserve outputs too.
    nbformat.write(notebook, notebook_path)
    before = notebook_path.read_bytes()
    resumed = run_experiment(experiment, hardware(), out, resume=True)
    assert (
        len(resumed.attempts) == 4
        and resumed.evaluator_seconds == record.evaluator_seconds
    )
    assert notebook_path.read_bytes() == before


def test_failed_learner_restart_retains_attempts_without_retraining(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from manabot.training import experiment_runner as runner
    from manabot.training.execution import atomic_json
    from manabot.training.models import TrainingRegime, TrainingRun

    launched: list[str] = []

    class Finished:
        pid = 99999999
        returncode = 1

        def poll(self) -> int:
            return self.returncode

        def wait(self) -> int:
            return self.returncode

    def launch(arguments: list[str], **kwargs: object) -> Finished:
        payload = json.loads(Path(arguments[-1]).read_text())
        directory = Path(payload["out"])
        directory.mkdir(parents=True)
        run = TrainingRun(
            id=f"failed-{len(launched)}",
            regime=TrainingRegime.model_validate(payload["regime"]),
            regime_digest="fixture",
            seed=payload["seed"],
            seed_streams={},
            identities={},
            status="failed",
            error="fixture learner failure",
        )
        atomic_json(directory / "run.json", run.model_dump(mode="json"))
        launched.append(run.id)
        return Finished()

    def stop(process: object) -> None:
        pass

    runtime = runner._runtime()
    monkeypatch.setattr(runner, "_runtime", lambda: runtime)
    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    monkeypatch.setattr(runner, "_stop", stop)
    experiment = demo_declaration()
    out = tmp_path / "failed"
    record = run_experiment(experiment, hardware(), out)
    assert record.status == "incomplete" and len(launched) == 4
    assert all(
        a.status == "failed" and "fixture learner failure" in a.error
        for a in record.attempts
    )
    before = record.process_seconds
    restarted = run_experiment(experiment, hardware(), out, resume=True)
    assert restarted.status == "incomplete" and len(launched) == 4
    assert restarted.process_seconds == before
    with pytest.raises(ValueError, match="intent"):
        assert experiment.schedule is not None
        run_experiment(
            replace(
                experiment,
                schedule=experiment.schedule.model_copy(update={"seeds": (119,)}),
            ),
            hardware(),
            out,
            resume=True,
        )


def test_budget_exhaustion_retains_pending_work_without_launch(tmp_path: Path) -> None:
    experiment = demo_declaration()
    assert experiment.schedule is not None
    schedule = experiment.schedule.model_copy(update={"process_seconds": 651})
    out = tmp_path / "no-room"
    with pytest.raises(TimeoutError, match="monitoring reserve"):
        run_experiment(replace(experiment, schedule=schedule), hardware(), out)
    payload = json.loads((out / "experiment.json").read_text())
    assert payload["status"] == "incomplete"
    assert all(a["status"] == "pending" for a in payload["attempts"])
    assert payload["process_seconds"] == 0
    assert not list(out.glob("training-*.log"))
