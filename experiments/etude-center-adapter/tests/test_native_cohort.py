"""Native wire fixtures are synthetic; only the historical capacity input is real."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pytest
from etude_client.api import Client
from etude_client.errors import EtudeError
from etude_client.models import (
    JobSnapshot,
    Principal,
    QueueObservation,
    ReportObservation,
    sha256,
)
from etude_fantasia.fantasia import FantasiaImport
from etude_fantasia.live_fantasia import (
    NativeCohort,
    NativeJob,
    NativeSelection,
    NativeTranslation,
    cohort_entry,
    refresh_native,
)
from etude_fantasia.reader import Reader
from etude_fantasia.values import Object, key, obj, seq
from test_native_fantasia import ReadOnlySource, native_fixture

from etude.admission import Queue


def entry(identifier: str, target: int = 26000) -> Object:
    return {
        "job_id": identifier,
        "plan": {
            "schema_version": 2,
            "source": {"commit": "a" * 40},
            "spec": {"lifetime_hours": 11.95, "spending_limit": 6.2},
            "regime": {"stages": [{"operation": "train_self_play", "updates": target}]},
        },
    }


def cohort_fixture(
    source: ReadOnlySource,
    identifier: str = "cohort",
    entries: list[Object] | None = None,
    heartbeat: int = 1791490000,
) -> NativeCohort:
    entries = entries or [entry("original"), entry("second")]
    prefix = "s3://fixture/cohorts/" + identifier
    spec = source.put_fixture(
        prefix + "/spec.json",
        "spec",
        {
            "schema_version": 1,
            "cohort_id": identifier,
            "destination": "s3://fixture/cohorts",
            "entries": list(entries),
        },
    )
    state = source.put_fixture(
        prefix + "/state.json",
        str(heartbeat),
        {
            "cohort_sha256": spec.sha256,
            "heartbeat_at": heartbeat,
            "phase": "running",
            "owner": "existing-fantasia-owner",
            "attempts": [],
        },
    )
    report = source.put_fixture(
        prefix + "/export/report.json",
        str(heartbeat),
        {
            "cohort_sha256": spec.sha256,
            "observed_at": heartbeat - 10,
            "supervisor_heartbeat_at": heartbeat - 30,
            "jobs": [
                {
                    "job_id": entries[0]["job_id"],
                    "heartbeat_at": heartbeat - 60,
                    "phase": "running",
                    "updates": 0,
                    "evaluations": 0,
                    "dashboards": ["private/unfetched.json"],
                }
            ],
        },
    )
    return NativeCohort(spec=spec, state=state, report=report)


def test_cohort_order_insertion_and_source_dated_report(service: tuple[str, str]) -> None:
    source = ReadOnlySource()
    cohort = cohort_fixture(source)
    state = obj(json.loads(source.objects[cohort.state.key, cohort.state.version]))
    continuation = entry("continued", 63000)
    continuation["continuation"] = {"job_id": "original", "iteration": 26000}
    state["insertion"] = {"after_job": "original", "entries": [continuation]}
    cohort = cohort.model_copy(
        update={
            "state": source.put_fixture(cohort.state.key, "inserted", state),
        }
    )
    selection = NativeSelection(cohort=cohort)
    with Client(*service) as client:
        experiment = client.register("cohort", namespace="fantasia")
        refresh_native(client, source, experiment.experiment_id, selection)
        detail = client.get(experiment.experiment_id)
        observed = next(
            f.observation
            for f in detail.current_facts
            if isinstance(f.observation, QueueObservation)
        )
        assert [j.source_key.source_id for j in observed.jobs] == [
            "original",
            "continued",
            "second",
        ]
        assert [j.target_step for j in observed.jobs] == [26000, 63000, 26000]
        assert observed.jobs[1].continuation is not None
        assert observed.jobs[1].continuation.parent_job == key("job", "original")
        assert observed.jobs[1].continuation.origin_step == 26000
        report = next(
            f.observation
            for f in detail.current_facts
            if isinstance(f.observation, ReportObservation) and f.observation.generated_at
        )
        assert report.generated_at == datetime.fromtimestamp(1791489990, UTC)
        assert report.evidence_watermark == datetime.fromtimestamp(1791489940, UTC)
        assert not any(isinstance(f.observation, JobSnapshot) for f in detail.facts)
        assert all(j.fact is None for j in detail.jobs)  # Report defaults aren't measured zeroes.
        assert len(source.reads) == 3  # No dashboard links or provider reads.
        assert len(detail.evidence) == 3 and detail.evidence_state == "partial"
        refresh_native(client, source, experiment.experiment_id, selection)
        replayed = client.get(experiment.experiment_id)
        assert replayed.facts == detail.facts and replayed.evidence == detail.evidence
        assert not client.resources()


@pytest.mark.parametrize(
    "corruption",
    [
        "state_hash",
        "report_hash",
        "duplicate",
        "boundary",
        "foreign_job",
        "foreign_selected_job",
        "future_clock",
        "wrong_experiment",
        "missing_report",
        "forbidden_report",
        "timeout_report",
    ],
)
def test_cohort_invalid_or_unavailable_cut_is_atomic(
    service: tuple[str, str],
    corruption: str,
) -> None:
    source = ReadOnlySource()
    cohort = cohort_fixture(source)
    assert cohort.report is not None
    ref = cohort.state if corruption in {"state_hash", "duplicate", "boundary"} else cohort.report
    value = obj(json.loads(source.objects[ref.key, ref.version]))
    if corruption.endswith("_hash"):
        value["cohort_sha256"] = "b" * 64
    elif corruption in {"duplicate", "boundary"}:
        value["insertion"] = {
            "after_job": "absent" if corruption == "boundary" else "original",
            "entries": [entry("original")],
        }
    elif corruption == "foreign_job":
        obj(seq(value["jobs"])[0])["job_id"] = "another-cohort-job"
    elif corruption == "future_clock":
        value["supervisor_heartbeat_at"] = 1791500000
    elif corruption.endswith("_report"):
        source.error, source.failed_key = corruption.split("_")[0], ref.key
    replaced = source.put_fixture(ref.key, ref.version, value)
    cohort = cohort.model_copy(update={"state" if ref == cohort.state else "report": replaced})
    selection = (
        native_fixture(source).model_copy(update={"cohort": cohort})
        if corruption == "foreign_selected_job"
        else NativeSelection(cohort=cohort)
    )
    with Client(*service) as client:
        experiment = client.register(
            "different" if corruption == "wrong_experiment" else "cohort",
            namespace="fantasia",
        )
        with pytest.raises(EtudeError):
            refresh_native(client, source, experiment.experiment_id, selection)
        detail = client.get(experiment.experiment_id)
        assert not detail.facts and not detail.evidence
        assert detail.collectors[0].error is not None


@pytest.mark.parametrize("missing", ["supervisor", "job", "error"])
def test_report_watermark_requires_all_source_clocks(missing: str) -> None:
    source = ReadOnlySource()
    cohort = cohort_fixture(source)
    assert cohort.report is not None
    value = obj(json.loads(source.objects[cohort.report.key, cohort.report.version]))
    if missing == "supervisor":
        value["supervisor_heartbeat_at"] = 0
    elif missing == "job":
        obj(seq(value["jobs"])[0])["heartbeat_at"] = None
    else:
        obj(seq(value["jobs"])[0])["error"] = "source unavailable"
    report_ref = source.put_fixture(cohort.report.key, "undated", value)
    reader = NativeTranslation(Reader(source))
    queue = reader.cohort(cohort.model_copy(update={"report": None}))
    report = reader.report(report_ref, cohort.spec, queue)
    assert report.generated_at is not None and report.evidence_watermark is None


def test_targets_do_not_flatten_stage_or_active_time_coordinates() -> None:
    value = entry("original")
    plan = obj(value["plan"])
    regime = obj(plan["regime"])
    stage = obj(seq(regime["stages"])[0])
    stage["active_seconds"] = 60
    assert cohort_entry(value).target_step is None
    plan["schema_version"] = 1
    plan["mix"] = {"wall_seconds": 7200, "dollar_cap": 3}
    assert cohort_entry(value).lease_hours == 2
    assert cohort_entry(value).cap_usd == 3
    del stage["active_seconds"]
    regime["stages"] = [stage, stage]
    assert cohort_entry(value).target_step is None


@pytest.mark.parametrize("field", ["target", "lease", "cap", "revision", "origin"])
def test_native_refresh_cannot_rewrite_frozen_definitions(
    service: tuple[str, str],
    field: str,
) -> None:
    source = ReadOnlySource()
    cohort = cohort_fixture(source)
    with Client(*service) as client:
        experiment = client.register("cohort", namespace="fantasia")
        refresh_native(client, source, experiment.experiment_id, NativeSelection(cohort=cohort))
        before = client.get(experiment.experiment_id)
        changed = entry("original")
        plan = obj(changed["plan"])
        if field == "target":
            obj(seq(obj(plan["regime"])["stages"])[0])["updates"] = 27000
        elif field in {"lease", "cap"}:
            obj(plan["spec"])["lifetime_hours" if field == "lease" else "spending_limit"] = 1
        elif field == "revision":
            obj(plan["source"])["commit"] = "b" * 40
        else:
            changed["continuation"] = {"job_id": "parent", "iteration": 20000}
        altered = cohort_fixture(source, entries=[changed, entry("second")], heartbeat=1791490100)
        with pytest.raises(EtudeError):
            refresh_native(
                client, source, experiment.experiment_id, NativeSelection(cohort=altered)
            )
        after = client.get(experiment.experiment_id)
        assert after.facts == before.facts and after.evidence == before.evidence
        assert after.jobs == before.jobs and after.collectors[0].error == "invalid"


def test_new_cohort_heartbeat_does_not_freshen_old_report(service: tuple[str, str]) -> None:
    source = ReadOnlySource()
    cohort = cohort_fixture(source)
    with Client(*service) as client:
        experiment = client.register("cohort", namespace="fantasia")
        refresh_native(client, source, experiment.experiment_id, NativeSelection(cohort=cohort))
        before = client.get(experiment.experiment_id)
        advanced = cohort_fixture(source, heartbeat=1791490100)
        # Keep the old report selection and bytes, not the newer fixture export.
        assert cohort.report is not None
        advanced = advanced.model_copy(update={"report": cohort.report})
        refresh_native(client, source, experiment.experiment_id, NativeSelection(cohort=advanced))
        refresh_native(client, source, experiment.experiment_id, NativeSelection(cohort=cohort))
        after = client.get(experiment.experiment_id)
        reports_before = [f for f in before.facts if isinstance(f.observation, ReportObservation)]
        reports_after = [f for f in after.facts if isinstance(f.observation, ReportObservation)]
        assert reports_before == reports_after
        queue_fact = next(
            f for f in after.current_facts if isinstance(f.observation, QueueObservation)
        )
        assert queue_fact.source_time == datetime.fromtimestamp(1791490100, UTC)
        assert not after.conflicts


@contextmanager
def service_process(queue: Queue, token: str) -> Iterator[tuple[Client, int]]:
    """A fresh interpreter for each start; only PostgreSQL and the token survive."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "etude.app:create_app",
                "--factory",
                "--fd",
                str(listener.fileno()),
                "--log-level",
                "critical",
                "--no-access-log",
            ],
            env=os.environ | {"ETUDE_DATABASE_URL": queue.dsn},
            pass_fds=(listener.fileno(),),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ) as process:
            try:
                with Client(f"http://127.0.0.1:{listener.getsockname()[1]}", token) as client:
                    for _ in range(100):
                        assert process.poll() is None, "service failed to start"
                        try:
                            client.list()
                            break
                        except EtudeError:
                            time.sleep(0.05)
                    else:
                        pytest.fail("service did not become ready")
                    yield client, process.pid
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def test_historical_native_refresh_survives_actual_service_restart(queue: Queue) -> None:
    imported = FantasiaImport(
        Path(__file__).resolve().parents[2] / "fantasia", "capacity-recovery-v3"
    )
    token = queue.issue_token(
        Principal(organization_id=queue.organization_id, name="test", scope="write")
    )
    source = ReadOnlySource()
    with service_process(queue, token) as (client, first_pid):
        receipt = client.register_record(imported.registration())
        for batch in imported.batches(receipt.experiment_id):
            client.ingest(batch)
        historical = client.get(receipt.experiment_id)
        original = next(
            f.observation
            for f in historical.current_facts
            if isinstance(f.observation, QueueObservation)
        )
        entries: list[Object] = []
        for job in original.jobs:
            value = entry(job.source_key.source_id, job.target_step or 0)
            plan = obj(value["plan"])
            plan["source"] = {"commit": job.source_revision}
            plan["spec"] = {
                "lifetime_hours": float(job.lease_hours or 0),
                "spending_limit": float(job.cap_usd or 0),
            }
            if job.continuation is not None:
                assert job.continuation.parent_job is not None
                value["continuation"] = {
                    "job_id": job.continuation.parent_job.source_id,
                    "iteration": job.continuation.origin_step,
                }
            entries.append(value)
        # Synthetic newer native exports reuse real historical identities. They are
        # explicitly NOT claims about the current Fantasia jobs.
        cohort = cohort_fixture(source, original.source_key.source_id, entries)
        identifier = original.jobs[1].source_key.source_id
        prefix = "s3://fixture/jobs/" + identifier
        spec = source.put_fixture(
            prefix + "/spec.json",
            "spec",
            {
                "job_id": identifier,
                "destination": "s3://fixture/jobs",
                "publish_seconds": 30,
                "plan": entries[1]["plan"],
            },
        )
        record = source.put_fixture(
            prefix + "/runtime/record.json",
            "advanced",
            {
                "spec_sha256": spec.sha256,
                "phase": "running",
                "generation": 100,
                "heartbeat_at": 1791490000,
                "updates": 25000,
                "evaluations_completed": 3,
            },
        )
        selection = NativeSelection(cohort=cohort, jobs=(NativeJob(spec=spec, record=record),))
        refresh_native(client, source, receipt.experiment_id, selection)
        before = client.get(receipt.experiment_id)
        assert not before.conflicts and len(client.list().items) == 1
        native_queue = next(
            f.observation
            for f in before.current_facts
            if isinstance(f.observation, QueueObservation)
        )
        assert native_queue == original  # All eight definitions, scientific scope and ancestry.
        assert [j.job_id for j in before.jobs] == [j.job_id for j in historical.jobs]
        assert {e.evidence_id for e in historical.evidence} <= {
            e.evidence_id for e in before.evidence
        }
        live = next(j.fact for j in before.jobs if j.source_key.source_id == identifier)
        assert live is not None and isinstance(live.observation, JobSnapshot)
        assert live.observation.progress.absolute_step == 25000
        assert live.observation.evaluation.completed_cohorts == 3
        assert live.observation.evaluation.completed_games is None
    # The first process has exited before another starts against the same DB.
    with service_process(queue, token) as (client, second_pid):
        assert second_pid != first_pid
        assert client.register_record(imported.registration()) == receipt
        for batch in imported.batches(receipt.experiment_id):
            assert client.ingest(batch).inserted == 0
        refresh_native(client, source, receipt.experiment_id, selection)
        after = client.get(receipt.experiment_id)
        assert len(client.list().items) == 1 and not after.conflicts
        assert after.facts == before.facts and after.evidence == before.evidence
        assert after.jobs == before.jobs
        old_ages = {age.observation_id: age for age in before.report_ages}
        for age in after.report_ages:
            previous = old_ages[age.observation_id]
            for field in ("generated_age_seconds", "watermark_age_seconds"):
                now_age, old_age = getattr(age, field), getattr(previous, field)
                assert now_age is None if old_age is None else now_age >= old_age
        for evidence in after.evidence:
            assert (
                sha256(client.request("GET", f"/evidence/{evidence.evidence_id}"))
                == evidence.sha256
            )
        assert not client.resources()
