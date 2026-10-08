"""Allocation authority, frozen identities and the real CLI path, without rentals."""

from datetime import datetime
from pathlib import Path
import time

import pytest
from typer.testing import CliRunner

from manabot.cli import app
from manabot.remote import cli, job_client, supervisor
from manabot.remote.job_store import worker_policy
from manabot.remote.jobs import Job, JobRecord, JobStatus
from manabot.remote.plan import (
    AccessScope,
    DeploymentPlan,
    JobSpec,
    compile_plan,
    digest,
)
from manabot.remote.provider import Pod
from manabot.remote.transport import job_startup
from manabot.training.checkpoint_queue import MonitoringBudget
from manabot.training.experiment_execution import PlannedRun
from manabot.training.experiments import Baseline, Experiment
from manabot.training.models import TrainingRegime, TrainSelfPlay
from tests.remote.job_fixtures import FileStore
from tests.remote.test_compile import ROOT, SOURCE
from tests.remote.test_jobs import admitted, published
from tests.remote.test_lifecycle import Clock, Provider
from tests.training.test_checkpoint_queue import run_fixture


def job_spec(hours: float = 0.5) -> JobSpec:
    template = JobSpec.model_validate_json(
        (ROOT / "ops/jobs/runpod-small.json").read_text()
    )
    return JobSpec(machine=template.machine, lifetime_hours=hours, spending_limit=500)


def recipe() -> TrainingRegime:
    value = TrainingRegime.model_validate_json(
        (ROOT / "ops/examples/step-target.json").read_text()
    )
    value.stages = value.stages[:1]
    value.schedule_clock = "iteration_fraction"
    return value


def test_fractional_and_month_plans_preserve_step_targets() -> None:
    raw = recipe().model_dump_json()
    plans = [
        compile_plan(raw, job_spec(hours), SOURCE, 197) for hours in (0.5, 4, 24 * 30)
    ]
    assert all(plan.input_json == raw for plan in plans)
    assert plans[0].regime == plans[1].regime == plans[2].regime
    assert [plan.spec.lifetime_seconds for plan in plans] == [1800, 14400, 2592000]
    assert plans[0].projected_dollars * 1440 == pytest.approx(
        plans[2].projected_dollars
    )
    assert DeploymentPlan.model_validate_json(plans[2].model_dump_json()) == plans[2]
    job_spec(0.5).admit(0)
    with pytest.raises(ValueError, match="renewable.*complete-state CUDA"):
        job_spec(720).admit(0)


def test_deadline_reserves_cost_extension_and_renewal() -> None:
    spec = job_spec()
    allocation = spec.admit(1000)
    assert allocation.deadline == 2800
    assert spec.admit(1000.75).deadline == 2800
    assert allocation.renewal_cutoff(1001) == allocation.deadline
    assert (
        allocation.pause_at + spec.checkpoint_seconds == allocation.checkpoint_deadline
    )
    assert (
        allocation.checkpoint_deadline + spec.upload_seconds + spec.cleanup_seconds
        == allocation.deadline
    )
    with pytest.raises(ValueError, match="expired"):
        allocation.renewal_cutoff(2800)
    with pytest.raises(ValueError, match="extension unavailable"):
        allocation.extend(1)
    assert allocation.deadline == 2800
    with pytest.raises(ValueError, match="reserves"):
        JobSpec.model_validate(spec.model_dump() | {"lifetime_hours": 0.1})
    with pytest.raises(ValueError, match="spending limit"):
        JobSpec.model_validate(spec.model_dump() | {"spending_limit": 0.01})
    with pytest.raises(ValueError):
        JobSpec.model_validate(spec.model_dump() | {"credential_hours": 720})
    with pytest.raises(ValueError):
        JobSpec.model_validate(spec.model_dump() | {"lifetime_hours": float("inf")})
    policy = worker_policy("s3://bucket/jobs/one", allocation.deadline)
    statements = policy["Statement"]
    assert isinstance(statements, list)
    cutoff = statements[-1]["Condition"]["DateGreaterThanEquals"]["aws:CurrentTime"]
    assert datetime.fromisoformat(cutoff).timestamp() == allocation.deadline
    job = Job(
        job_id="one",
        plan=compile_plan(recipe().model_dump_json(), spec, SOURCE, 1),
        created_at=1000,
        deadline=2800,
    )
    assert job.work_deadline == allocation.checkpoint_deadline
    assert "export MANABOT_DEADLINE=2800" in job_startup(job)


def test_frozen_four_hour_deployment_keeps_exact_identity() -> None:
    raw = (
        (Path(__file__).parent / "fixtures/legacy-capacity-deployment.json")
        .read_text()
        .strip()
    )
    old = DeploymentPlan.model_validate_json(raw)
    assert old.model_dump_json() == raw
    assert (
        digest(raw.encode())
        == "3d1ccbe077df74010c9c76b7738b546be7d73d1a5c62b82829b8d96d87912fc1"
    )
    assert (
        old.input_sha256
        == "e6415e4d638f3da19c9b5c9312d7e43e7e1d12040986377ca7a42fb5b839dea2"
    )
    assert old.allocation_at(1000) is None and old.schema_version == 1
    stage = old.regime.stages[0]
    assert isinstance(stage, TrainSelfPlay) and stage.active_seconds == 14400
    with pytest.raises(ValueError, match="step-target"):
        compile_plan(old.input_json, job_spec(4), SOURCE, 1)


def test_experiment_binds_seeded_cases_without_changing_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bindings = tuple(
        PlannedRun(
            case="steps",
            seed=seed,
            spec=job_spec(hours),
            monitoring=MonitoringBudget(seconds=100, attempt_seconds=10),
        )
        for seed, hours in ((1, 0.5), (2, 4))
    )
    experiment = Experiment(
        name="steps", baseline=Baseline.capture("baseline", recipe()), jobs=bindings
    )
    plans = experiment.compile_jobs(SOURCE)
    assert [plan.seed for plan in plans] == [1, 2]
    assert plans[0].regime == plans[1].regime
    assert plans[0].input_json == experiment.resolve().cases[0].configuration
    store = FileStore(tmp_path / "experiment.sqlite")
    monkeypatch.setattr(job_client, "S3JobStore", lambda prefix: store)
    spec = experiment.prepare_job(0, SOURCE, "experiment-launch")
    assert spec.plan == plans[0]
    assert spec.monitoring == bindings[0].monitoring
    assert spec.checkpoint_seconds == bindings[0].checkpoint_seconds
    assert spec.experiment_json is not None


def test_cli_compile_and_submit_use_real_job_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Clock()
    store = FileStore(tmp_path / "control.sqlite")
    monkeypatch.setattr(job_client, "time", clock)
    monkeypatch.setattr(job_client, "S3JobStore", lambda prefix: store)
    monkeypatch.setattr(cli, "current_source", lambda root: SOURCE)
    monkeypatch.setattr(job_client, "current_source", lambda root: SOURCE)
    monkeypatch.setattr(job_client, "verify_public_source", lambda source: None)
    startup: list[str] = []

    class AcceptingProvider(Provider):
        def create(self, payload: dict[str, object]) -> Pod:
            pod = super().create(payload)
            startup.append(str(payload["dockerStartCmd"]))
            if self.created == 2:
                raw = store.read("spec.json")
                assert raw is not None
                spec = Job.model_validate_json(raw.data)
                record = JobRecord(
                    spec_sha256=spec.identity,
                    pod_id=pod.id,
                    phase="running",
                    accepted_at=clock.time(),
                    heartbeat_at=clock.time(),
                )
                store.create("runtime/record.json", record.model_dump_json().encode())
            return pod

    provider = AcceptingProvider(clock)
    monkeypatch.setattr(job_client, "RunPod", lambda: provider)
    original_submit = job_client.submit_job

    def submit(spec: Job) -> JobStatus:
        return original_submit(
            spec, credentials=lambda value: {"AWS_SESSION_TOKEN": "fake-delegated"}
        )

    monkeypatch.setattr(cli, "submit_job", submit)
    regime_path, launch_path, plan_path = (
        tmp_path / name for name in ("regime.json", "launch.json", "plan.json")
    )
    regime_path.write_text(recipe().model_dump_json())
    declaration = job_spec().model_copy(
        update={"access": AccessScope(destination="s3://bucket/private/jobs")}
    )
    launch_path.write_text(declaration.model_dump_json())
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "deploy",
            "compile",
            "--regime",
            str(regime_path),
            "--spec",
            str(launch_path),
            "--out",
            str(plan_path),
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    result = runner.invoke(
        app,
        ["deploy", "submit", "--plan", str(plan_path), "--job-id", "bounded-launch"],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    raw = store.read("spec.json")
    assert raw is not None
    spec = Job.model_validate_json(raw.data)
    assert spec.deadline == 2800 and spec.destination == declaration.access.destination
    assert provider.created == 2
    assert "MANABOT_DEADLINE=2800" in startup[-1]
    assert "issuer" not in startup[-1]
    # Reconnect retains the original allocation and never creates another rental.
    result = runner.invoke(
        app,
        ["deploy", "submit", "--plan", str(plan_path), "--job-id", "bounded-launch"],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert provider.created == 2


def test_long_job_fails_before_writing_intent(tmp_path: Path) -> None:
    store = FileStore(tmp_path / "control.sqlite")
    plan = compile_plan(recipe().model_dump_json(), job_spec(720), SOURCE, 1)
    with pytest.raises(ValueError, match="renewable"):
        job_client.prepare_job(plan, "month", store=store)
    assert store.read("spec.json") is None


def test_supervisor_retains_paused_learner_and_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    store = FileStore(tmp_path / "control.sqlite")
    plan = compile_plan(recipe().model_dump_json(), job_spec(), SOURCE, 1)
    spec = job_client.prepare_job(plan, "pause", store=store)
    resource = admitted(spec, store)
    retained = run_fixture(tmp_path / "fixture.json")
    retained.status = retained.stages[0].status = "paused"
    monkeypatch.setattr(supervisor, "_training_run", lambda root: retained)
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "run").mkdir()
    (root / "run/run.json").write_text(retained.model_dump_json())
    result = supervisor.supervise(
        spec,
        store,
        root,
        resource.pod.id,
        command=["uv", "run", "--no-sync", "python", "-c", "pass"],
        publish=published,
    )
    assert result.phase == "paused" and result.terminal and result.artifacts_complete
    assert spec.allocation is not None and spec.allocation.pause_at > time.time()
    assert (
        root.parent / "allocation.json"
    ).read_text() == spec.allocation.model_dump_json()


def test_current_plan_has_one_specification() -> None:
    plan = compile_plan(recipe().model_dump_json(), job_spec(), SOURCE, 1)
    payload = plan.model_dump(mode="json")
    assert payload["schema_version"] == 2
    assert payload["spec"] == job_spec().model_dump(mode="json")
    assert "mix" not in payload and "launch" not in payload
    assert DeploymentPlan.model_validate_json(plan.model_dump_json()) == plan


def test_schema1_job_retains_identity_and_deadline() -> None:
    raw = (Path(__file__).parent / "fixtures/legacy-v1-job.json").read_text().strip()
    job = Job.model_validate_json(raw)
    assert job.model_dump_json() == raw
    assert (
        job.identity
        == "83304a98885f4825c47ca5c84f2116d3ffd7ba6295f0310b6098b9420d697234"
    )
    assert job.deadline == 26200.25
    assert job.allocation is None
    assert job.work_deadline == job.deadline - 660 - 120
    assert Job.model_validate(job.model_dump()).identity == job.identity


def test_schema1_plan_cannot_silently_rewrite_frozen_content() -> None:
    path = Path(__file__).parent / "fixtures/legacy-capacity-deployment.json"
    plan = DeploymentPlan.model_validate_json(path.read_text())
    stage = plan.regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.updates += 1
    with pytest.raises(ValueError, match="immutable"):
        plan.model_dump_json()


def test_historical_reserves_require_explicit_new_authoring() -> None:
    path = Path(__file__).parent / "fixtures/legacy-capacity-deployment.json"
    old = DeploymentPlan.model_validate_json(path.read_text())
    with pytest.raises(ValueError, match="checkpoint_seconds"):
        compile_plan(recipe().model_dump_json(), old.spec, SOURCE, 1)


def test_experiment_cohort_preserves_planned_order_and_receipts() -> None:
    regime = recipe()
    spec = job_spec().model_copy(update={"spending_limit": 2})
    experiment = Experiment(
        name="ordered",
        baseline=Baseline.capture("baseline", regime),
        jobs=tuple(
            PlannedRun(
                case="ordered",
                seed=seed,
                spec=spec,
                monitoring=MonitoringBudget(seconds=120, attempt_seconds=30),
            )
            for seed in (199, 197)
        ),
    )
    cohort = experiment.cohort(
        SOURCE,
        "ordered",
        deadline=100000,
        spending_limit=5,
        prior_dollars=0.75,
        controller_dollars=0.25,
    )
    assert [entry.plan.seed for entry in cohort.entries] == [199, 197]
    assert [entry.job_id for entry in cohort.entries] == ["ordered-0", "ordered-1"]
    assert cohort.reserved_dollars == 5
    assert all(entry.experiment_json is not None for entry in cohort.entries)
    with pytest.raises(ValueError, match="inclusive spending"):
        experiment.cohort(
            SOURCE,
            "ordered",
            deadline=100000,
            spending_limit=4.99,
            prior_dollars=0.75,
            controller_dollars=0.25,
        )
