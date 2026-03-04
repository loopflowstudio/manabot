"""
test_job.py
Tests for job launch/list/log/cancel/resume orchestration with provider doubles.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ops.job import JobManager, JobRecord, JobStore
from ops.provider import CommandResult, Machine, MachineSpec, RuntimeSpec


@dataclass
class FakeProvider:
    user: str = "tester"

    def __post_init__(self):
        self.machines: dict[str, Machine] = {}
        self.create_calls: list[dict] = []
        self.terminated: list[str] = []

    def create(self, spec, tags, *, user_data=None, iam_instance_profile=None):
        machine = Machine(
            id=f"i-job-{len(self.create_calls) + 1}",
            public_ip="5.6.7.8",
            status="running",
            tags=tags,
        )
        self.create_calls.append(
            {
                "spec": spec,
                "tags": tags,
                "user_data": user_data,
                "iam_instance_profile": iam_instance_profile,
            }
        )
        self.machines[machine.id] = machine
        return machine

    def start(self, machine):
        return machine

    def wait_until_ready(self, machine, timeout=300):
        return machine

    def wait_for_ssm(self, machine, timeout=300):
        return None

    def stop(self, machine):
        return None

    def terminate(self, machine):
        self.terminated.append(machine.id)

    def list(self, tags):
        def matches(machine):
            return all(machine.tags.get(key) == value for key, value in tags.items())

        return [machine for machine in self.machines.values() if matches(machine)]

    def run_command(self, machine, command, timeout=3600):
        if "test -f" in command:
            return CommandResult("verify", "Success", "READY\n", "")
        return CommandResult("start", "Success", "started", "")

    def logs(self, group, stream_prefix, limit=200):
        return [f"{stream_prefix} log line"]


def _job_spec() -> MachineSpec:
    return MachineSpec(
        instance_type="g5.xlarge",
        region="us-east-1",
        disk_gb=50,
        ami="ami-test",
        spot=True,
        max_spot_price="1.50",
    )


def _runtime_spec() -> RuntimeSpec:
    return RuntimeSpec(
        image="ghcr.io/loopflowstudio/manabot-gpu:latest",
        fallback_build=True,
        log_group_prefix="/manabot",
    )


def test_job_launch_persists_record_and_tags(tmp_path: Path):
    provider = FakeProvider()
    store = JobStore(tmp_path / "jobs.json")
    manager = JobManager(provider, _job_spec(), _runtime_spec(), store=store)

    record, machine = manager.launch("simple")

    assert record.job_id.startswith("job-")
    assert machine.id == "i-job-1"
    assert store.get(record.job_id) == record
    created_tags = provider.create_calls[0]["tags"]
    assert created_tags["manabot:role"] == "job"
    assert created_tags["manabot:job-id"] == record.job_id
    assert created_tags["manabot:wandb-run-id"] == record.wandb_run_id


def test_job_resume_reuses_existing_wandb_run_id(tmp_path: Path):
    provider = FakeProvider()
    store = JobStore(tmp_path / "jobs.json")
    manager = JobManager(provider, _job_spec(), _runtime_spec(), store=store)

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
    manager = JobManager(provider, _job_spec(), _runtime_spec(), store=store)

    record, machine = manager.launch("simple")

    jobs = manager.list_jobs()
    assert jobs
    assert jobs[0].id == machine.id

    lines = manager.logs(record.job_id)
    assert lines == [f"{record.job_id}/ log line"]

    cancelled = manager.cancel(record.job_id)
    assert cancelled == 1
    assert provider.terminated == [machine.id]
