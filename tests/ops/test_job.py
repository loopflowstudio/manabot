"""
test_job.py
Tests for job launch/list/log/cancel/resume orchestration with provider doubles.
"""

from __future__ import annotations

from pathlib import Path

from ops.job import JobManager, JobRecord, JobStore
from tests.ops.conftest import FakeProvider, make_job_spec, make_runtime_spec


def test_job_launch_persists_record_and_tags(tmp_path: Path):
    provider = FakeProvider()
    store = JobStore(tmp_path / "jobs.json")
    manager = JobManager(provider, make_job_spec(), make_runtime_spec(), store=store)

    record, machine = manager.launch("simple")

    assert record.job_id.startswith("job-")
    assert machine.id == "i-fake-1"
    assert store.get(record.job_id) == record
    created_tags = provider.create_calls[0]["tags"]
    assert created_tags["manabot:role"] == "job"
    assert created_tags["manabot:job-id"] == record.job_id
    assert created_tags["manabot:wandb-run-id"] == record.wandb_run_id


def test_job_resume_reuses_existing_wandb_run_id(tmp_path: Path):
    provider = FakeProvider()
    store = JobStore(tmp_path / "jobs.json")
    manager = JobManager(provider, make_job_spec(), make_runtime_spec(), store=store)

    seeded = JobRecord(
        job_id="job-abc123",
        wandb_run_id="run-xyz",
        config_name="simple",
        region="us-east-1",
    )
    store.upsert(seeded)

    record, _machine = manager.resume("job-abc123")

    assert record == seeded
    created_tags = provider.create_calls[0]["tags"]
    assert created_tags["manabot:job-id"] == "job-abc123"
    assert created_tags["manabot:wandb-run-id"] == "run-xyz"


def test_job_list_logs_and_cancel(tmp_path: Path):
    provider = FakeProvider()
    store = JobStore(tmp_path / "jobs.json")
    manager = JobManager(provider, make_job_spec(), make_runtime_spec(), store=store)

    record, machine = manager.launch("simple")

    jobs = manager.list_jobs()
    assert jobs
    assert jobs[0].id == machine.id

    lines = manager.logs(record.job_id)
    assert lines == [f"{record.job_id}/ log line"]

    cancelled = manager.cancel(record.job_id)
    assert cancelled == 1
    assert provider.terminated == [machine.id]
