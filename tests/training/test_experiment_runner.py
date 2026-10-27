"""Shared Experiment execution, placement and editable notebook acceptance."""

from dataclasses import replace
import json
from pathlib import Path
import socket

from nbclient import NotebookClient
import nbformat
import pytest

<<<<<<< HEAD
from experiments.runners import depth_screen, history_input
from experiments.runners.experiment_demo import declaration as demo_declaration
from experiments.runners.experiment_depth import declaration as depth_declaration
from experiments.runners.experiment_history import declaration as history_declaration
from experiments.runners.experiment_screen import admit_screen
from experiments.runners.run_history_input import history_plan
from manabot.training.checkpoint_queue import Attempt, MonitoringBudget
from manabot.training.comparison_notebook import write_comparison_notebook
from manabot.training.experiment_execution import (
=======
from manabot.eval.monitor import MonitorProtocol, MonitorResult
from manabot.experiment.config import Baseline, Case, Experiment, ModelOverride
from manabot.experiment.evaluation_queue import Attempt, MonitoringBudget
from manabot.experiment.execution import (
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    ExperimentRun,
    ExperimentSchedule,
    Hardware,
    HardwareInventory,
)
<<<<<<< HEAD
from manabot.training.experiment_runner import run_experiment
from manabot.training.monitor_evaluation import MonitorProtocol, MonitorResult
from manabot.verify.store import VerifyStore
from tests.training.test_history_input import calibration
=======
from manabot.experiment.runner import run_experiment
from manabot.infra.config import ModelConfig
from manabot.recipes.ataraxos import ataraxos
from manabot.reports.notebook import write_comparison_notebook
from manabot.verify.store import VerifyStore
import managym
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
from tests.training.test_monitor_evaluation import failure_summary


@pytest.fixture(autouse=True)
def isolated_account_leases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixtures must not acquire a real experiment's host/evaluator leases."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "test-account")


<<<<<<< HEAD
=======
def demo_declaration() -> Experiment:
    """Four short training runs and four milestones: a workflow fixture."""
    base = ataraxos().regime()
    base.world = managym.WORLD_VERSION
    base.id = f"fixture-{base.world}"
    base.stages = base.stages[:1]
    # Work stays update-bounded; wall allowances include interpreter startup.
    base.wall_seconds = 60
    for stage in base.stages:
        stage.execution.wall_seconds = 30
        stage.streams = 2
        stage.transitions = 8
    base.agent.hidden_dim = 8
    base.agent.num_attention_heads = 2
    # Four real monitoring cohorts include process startup and exact replay.
    cohort_seconds = MonitoringBudget.model_fields["attempt_seconds"].default
    monitoring_seconds = 4 * cohort_seconds
    return Experiment(
        name="fixture",
        baseline=Baseline.capture(f"fixture-{base.world}", base),
        cases=(
            Case("small"),
            Case("wide", (ModelOverride(ModelConfig(hidden_dim=16)),)),
        ),
        schedule=ExperimentSchedule(
            seeds=(117, 118),
            hardware="fixture",
            wall_seconds=monitoring_seconds + 300,
            process_seconds=monitoring_seconds + 300,
            monitoring=MonitoringBudget(
                seconds=monitoring_seconds,
                protocol=MonitorProtocol(
                    deal_seeds=(1910101000,), bootstrap_replicates=20
                ),
            ),
            checkpoint_updates=1000,
            disk_reserve_bytes=0,
        ),
    )


>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
def hardware() -> HardwareInventory:
    return HardwareInventory(
        resources=(Hardware(name="fixture", host=socket.gethostname(), cpu_threads=2),)
    )


def _experiment_failures(record: ExperimentRun, monitoring: Path) -> str:
    """Name each failed attempt's reason; full records exceed pytest's limit."""
    lines = [f"experiment {record.status}: {record.error}"]
    lines += [
        f"training attempt {a.ordinal} {a.status}: {a.error}"
        for a in record.attempts
        if a.status != "completed"
    ]
    for path in sorted(monitoring.glob("attempt-*/attempt.json")):
        attempt = Attempt.model_validate_json(path.read_text())
        if attempt.status == "completed":
            continue
        lines.append(
            f"monitoring {path.parent.name} {attempt.status} after "
            f"{attempt.charged_seconds:.1f}s of {attempt.reserved_seconds:.0f}s: "
            f"{attempt.error}"
        )
        result = path.parent / "evaluation/monitor.json"
        if result.exists():
            summary = failure_summary(
                MonitorResult.model_validate_json(result.read_text())
            )
            lines += [f"  {line}" for line in summary.splitlines()]
    return "\n".join(lines)


<<<<<<< HEAD
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


=======
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
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
    if record.status != "completed":
        pytest.fail(_experiment_failures(record, out / "monitoring"), pytrace=False)
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
<<<<<<< HEAD
    from manabot.training import experiment_runner as runner
    from manabot.training.execution import atomic_json
    from manabot.training.models import TrainingRegime, TrainingRun
=======
    from manabot.experiment import runner as runner
    from manabot.infra.files import atomic_json
    from manabot.training.records import TrainingRun
    from manabot.training.regime import TrainingRegime
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f

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
    schedule = experiment.schedule.model_copy(
        update={"process_seconds": experiment.schedule.monitoring.seconds + 1}
    )
    out = tmp_path / "no-room"
    with pytest.raises(TimeoutError, match="monitoring reserve"):
        run_experiment(replace(experiment, schedule=schedule), hardware(), out)
    payload = json.loads((out / "experiment.json").read_text())
    assert payload["status"] == "incomplete"
    assert all(a["status"] == "pending" for a in payload["attempts"])
    assert payload["process_seconds"] == 0
    assert not list(out.glob("training-*.log"))


def test_live_diagnostics_and_requested_exports_remain_separate(tmp_path: Path) -> None:
<<<<<<< HEAD
    from manabot.training.experiments import Baseline, Experiment
    from manabot.training.models import TrainingRun
=======
    from manabot.experiment.config import Baseline, Experiment
    from manabot.training.records import TrainingRun
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    from tests.training.test_recovery import recipe

    value = recipe()
    value.recovery_max_microsteps = None
    value.stages[0].updates = 2
    # Five real four-game cohorts include subprocess startup and exact replay.
    # This is a routing check, not an evaluator performance threshold, so the
    # default per-game and per-cohort budgets serve only as hang guards.
    cohort_seconds = MonitoringBudget.model_fields["attempt_seconds"].default
    schedule = ExperimentSchedule(
        seeds=(11840,),
        hardware="fixture",
        wall_seconds=3600,
        process_seconds=3600,
        checkpoint_updates=(1,),
        monitoring=MonitoringBudget(
            seconds=5 * cohort_seconds,
            include_initial=True,
            protocol=MonitorProtocol(deal_seeds=(1911182000,)),
        ),
        diagnostic_protocol=MonitorProtocol(
            deal_seeds=(1911182001,),
            opponent="random",
        ),
        diagnostic_updates=(0, 2),
        scientific_deal_seeds=(1911182002,),
    )
    value.wall_seconds = 60
    value.stages[0].execution.wall_seconds = 60
    record = run_experiment(
        Experiment("diagnostic", Baseline.capture("fixture", value), schedule=schedule),
        hardware(),
        tmp_path / "live",
    )
    if record.status != "completed":
        pytest.fail(
            _experiment_failures(record, tmp_path / "live/monitoring"), pytrace=False
        )
    results = [
        MonitorResult.model_validate_json(p.read_text())
        for p in (tmp_path / "live/monitoring").rglob("monitor.json")
    ]
    assert {(r.coordinates.updates, r.protocol.opponent) for r in results} == {
        (0, "scripted_greedy"),
        (1, "scripted_greedy"),
        (2, "scripted_greedy"),
        (0, "random"),
        (2, "random"),
    }
    assert len(results) == 5
    assert all(r.status == "completed" and len(r.rows) == 4 for r in results)
    saved = TrainingRun.model_validate_json(
        (Path(record.attempts[0].path) / "run.json").read_text()
    )
    assert [c.updates for c in saved.monitoring_checkpoints] == [1]
