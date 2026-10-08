"""Fault injection through the shared job client: no rental, learner or telemetry."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import pytest
from typer.testing import CliRunner

from manabot.cli import app
from manabot.infra.artifacts import StoredArtifact
from manabot.remote import cohort_cli, job_client
from manabot.remote.cohort import Cohort, CohortEntry, CohortSupervisor, cancel_cohort
from manabot.remote.cohort_service import owner_lock
from manabot.remote.jobs import Job, JobPhase, JobRecord, JobStatus, Resource
from manabot.remote.plan import JobSpec, compile_plan
from manabot.remote.provider import Pod, ProviderError, RunPod
from tests.remote.job_fixtures import FileStore
from tests.remote.test_compile import ROOT, SOURCE
from tests.remote.test_lifecycle import Clock, Provider


class CohortProvider(Provider):
    def get(self, pod_id: str) -> Pod | None:
        return next((pod for pod in self.pods if pod.id == pod_id), None)


class Harness:
    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        self.store = FileStore(root / "cohort.sqlite")
        self.clock = Clock()
        self.provider = CohortProvider(self.clock)
        monkeypatch.setattr(job_client, "current_source", lambda root: SOURCE)
        monkeypatch.setattr(job_client, "verify_public_source", lambda source: None)
        monkeypatch.setattr(job_client, "time", self.clock)
        monkeypatch.setattr("manabot.remote.cohort.time", self.clock)
        plan = compile_plan(
            (ROOT / "ops/examples/step-target.json").read_text(),
            JobSpec.model_validate_json(
                (ROOT / "ops/jobs/runpod-small.json").read_text()
            ),
            SOURCE,
            197,
        )
        self.cohort = Cohort(
            cohort_id="test-cohort",
            deadline=100000,
            spending_limit=12,
            prior_dollars=1.25,
            controller_dollars=0.25,
            entries=tuple(CohortEntry(job_id=f"test-{i}", plan=plan) for i in range(2)),
        )
        self.verified: list[str] = []

    def jobs(self, spec: Job) -> FileStore:
        return FileStore(self.root / f"{spec.job_id}.sqlite")

    def persist(self, spec: Job) -> Job:
        value = job_client.persist_job(spec, store=self.jobs(spec))
        # Guardian behavior has its own lifecycle tests. Retain its proof so this
        # fixture can put faults exactly around the training provider request.
        self.jobs(spec).create("guardian-proof.json", b"{}")
        return value

    def observe(self, spec: Job) -> JobStatus:
        return job_client.reconcile_job(
            spec, store=self.jobs(spec), provider=cast(RunPod, self.provider)
        )

    def submit(self, spec: Job) -> JobStatus:
        return job_client.submit_job(
            spec,
            store=self.jobs(spec),
            provider=cast(RunPod, self.provider),
            credentials=lambda spec: {},
        )

    def cancel(self, spec: Job) -> None:
        job_client.cancel_job(spec, store=self.jobs(spec))

    def verify(self, status: JobStatus, cache: Path) -> str:
        assert status.record and status.record.manifest
        self.verified.append(status.spec.job_id)
        return status.record.manifest.sha256

    def restart(self, owner: str = "host-and-directory") -> CohortSupervisor:
        return CohortSupervisor(
            self.cohort,
            owner,
            self.root / "cache",
            store=self.store,
            persist=self.persist,
            observe=self.observe,
            submit=self.submit,
            cancel=self.cancel,
            verify=self.verify,
        )

    def finish(
        self, spec: Job, phase: JobPhase = "completed", *, complete: bool = True
    ) -> None:
        store = self.jobs(spec)
        resource = store.read("training.json")
        assert resource is not None

        pod = Resource.model_validate_json(resource.data).pod
        record = JobRecord(
            spec_sha256=spec.identity,
            pod_id=pod.id,
            phase=phase,
            accepted_at=self.clock.now,
            heartbeat_at=self.clock.now,
            generation=1,
            manifest=StoredArtifact(
                uri=f"{spec.prefix}/runtime/manifest", sha256="f" * 64, bytes=1
            ),
            artifacts_complete=complete,
        )
        store.create("runtime/record.json", record.model_dump_json().encode())
        self.provider.delete(pod.id)

    def launch(self) -> Job:
        state = self.restart().tick()
        spec = state.attempts[-1].spec
        # A fake pod does not run a remote supervisor; acceptance times out while
        # the durable create response remains available for a later observer.
        with pytest.raises(TimeoutError, match="acceptance"):
            self.restart().tick()
        return spec


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Harness:
    return Harness(tmp_path, monkeypatch)


def test_restart_retains_deadlines_and_advances_two_jobs(harness: Harness) -> None:
    first = harness.launch()
    harness.finish(first)
    state = harness.restart().tick()
    assert len(state.attempts) == 2
    assert state.attempts[0].spec == first
    second = state.attempts[1].spec
    with pytest.raises(TimeoutError):
        harness.restart().tick()
    harness.finish(second)
    state = harness.restart().tick()
    assert state.phase == "completed"
    charged = state.charged_dollars(harness.cohort)
    assert harness.restart().tick().charged_dollars(harness.cohort) == charged
    assert harness.provider.created == 2
    assert harness.verified == [first.job_id, second.job_id]


def test_lost_create_response_reconciles_without_resubmission(harness: Harness) -> None:
    state = harness.restart().tick()
    spec = state.attempts[0].spec
    harness.provider.failure = "ambiguous"
    with pytest.raises(ProviderError):
        harness.restart().tick()
    harness.provider.failure = ""
    # Reconciliation recovers the provider identity before attempting submission.
    with pytest.raises(TimeoutError):
        harness.restart().tick()
    assert harness.provider.created == 1
    assert harness.restart().state.attempts[0].spec == spec
    harness.finish(spec)
    assert len(harness.restart().tick().attempts) == 2


@pytest.mark.parametrize("phase", ["failed", "paused", "cancelled", "deadline"])
def test_unsuccessful_attempt_never_replaced(harness: Harness, phase: JobPhase) -> None:
    spec = harness.launch()
    harness.finish(spec, phase)
    for _ in range(2):
        state = harness.restart().tick()
        assert state.phase == phase
        assert len(state.attempts) == harness.provider.created == 1


def test_missing_final_evidence_halts(harness: Harness) -> None:
    spec = harness.launch()
    harness.finish(spec, complete=False)
    state = harness.restart().tick()
    assert state.phase == "missing" and len(state.attempts) == 1


def test_overrun_is_charged_once_and_blocks_remaining_jobs(harness: Harness) -> None:
    spec = harness.launch()
    harness.finish(spec)
    harness.clock.now += 100000  # Late confirmed absence conservatively charges time.
    harness.cohort = harness.cohort.model_copy(update={"deadline": 1000000})
    # Keep the same frozen deadline from first admission; changing budget/intent
    # after a failure is prohibited even if the new file looks plausible.
    with pytest.raises(ValueError, match="different immutable"):
        harness.restart()
    harness.cohort = harness.cohort.model_copy(update={"deadline": 100000})
    harness.clock.now = 90000
    state = harness.restart().tick()
    assert state.phase == "budget-exhausted"
    assert state.charged_dollars(harness.cohort) > harness.cohort.spending_limit
    assert harness.restart().tick().charged_dollars(
        harness.cohort
    ) == state.charged_dollars(harness.cohort)
    assert harness.provider.created == 1


def test_cancel_cli_survives_restart_before_submission(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness.restart().tick()
    monkeypatch.setattr(cohort_cli, "load_cohort", lambda *args: harness.cohort)
    monkeypatch.setattr(
        cohort_cli,
        "cancel_cohort",
        lambda cohort: cancel_cohort(cohort, harness.store, cancel=harness.cancel),
    )
    result = CliRunner().invoke(
        app, ["deploy", "cohort", "cancel", "--cohort-id", "test-cohort"]
    )
    assert result.exit_code == 0, result.output
    assert harness.restart().tick().phase == "cancelled"
    assert harness.provider.created == 0


def test_network_outage_and_missing_claim_are_not_permission_to_rent(
    harness: Harness,
) -> None:
    spec = harness.launch()
    harness.provider.failure = "inventory"

    # Existing resource receipts use provider.get, so inject the failed read.
    def unavailable(pod_id: str) -> None:
        raise ProviderError("offline")

    original = harness.provider.get
    harness.provider.get = unavailable  # type: ignore[method-assign]
    assert harness.restart().tick().phase == "uncertain"
    harness.provider.get = original  # type: ignore[method-assign]
    harness.provider.failure = ""
    harness.finish(spec)
    assert len(harness.restart().tick().attempts) == 2
    assert harness.provider.created == 1


def test_second_owner_and_concurrent_local_driver_rejected(harness: Harness) -> None:
    harness.restart()
    with pytest.raises(ValueError, match="another owner"):
        harness.restart("another-host")
    path = harness.root / "service"
    with owner_lock(path):

        def competing() -> None:
            with owner_lock(path):
                pytest.fail("second owner acquired")

        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(RuntimeError, match="already running"):
                pool.submit(competing).result()
