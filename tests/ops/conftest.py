"""
conftest.py
Shared test doubles and spec factories for ops tests.
"""

from __future__ import annotations

from dataclasses import dataclass

from ops.provider import CommandResult, Machine, MachineSpec, RuntimeSpec


@dataclass
class FakeProvider:
    user: str = "tester"

    def __post_init__(self):
        self.machines: dict[str, Machine] = {}
        self.create_calls: list[dict] = []
        self.started: list[str] = []
        self.stopped: list[str] = []
        self.terminated: list[str] = []

    def create(self, spec, tags, *, user_data=None, iam_instance_profile=None):
        machine = Machine(
            id=f"i-fake-{len(self.create_calls) + 1}",
            public_ip="1.2.3.4",
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
        self.started.append(machine.id)
        updated = Machine(
            id=machine.id,
            public_ip=machine.public_ip or "1.2.3.4",
            status="running",
            tags=machine.tags,
        )
        self.machines[machine.id] = updated
        return updated

    def wait_until_ready(self, machine, timeout=300):
        return machine

    def wait_for_ssm(self, machine, timeout=300):
        return None

    def stop(self, machine):
        self.stopped.append(machine.id)
        self.machines[machine.id] = Machine(
            id=machine.id,
            public_ip=machine.public_ip,
            status="stopped",
            tags=machine.tags,
        )

    def terminate(self, machine):
        self.terminated.append(machine.id)
        self.machines[machine.id] = Machine(
            id=machine.id,
            public_ip=None,
            status="terminated",
            tags=machine.tags,
        )

    def list(self, tags):
        def matches(m):
            return all(m.tags.get(key) == value for key, value in tags.items())

        return [m for m in self.machines.values() if matches(m)]

    def run_command(self, machine, command, timeout=3600):
        if "test -f" in command:
            return CommandResult("verify", "Success", "READY\n", "")
        return CommandResult("cmd", "Success", "", "")

    def logs(self, group, stream_prefix, limit=200):
        return [f"{stream_prefix} log line"]


def make_sandbox_spec() -> MachineSpec:
    return MachineSpec(
        instance_type="g5.xlarge",
        region="us-east-1",
        disk_gb=100,
        ami="ami-test",
        spot=False,
    )


def make_job_spec() -> MachineSpec:
    return MachineSpec(
        instance_type="g5.xlarge",
        region="us-east-1",
        disk_gb=50,
        ami="ami-test",
        spot=True,
        max_spot_price="1.50",
    )


def make_runtime_spec() -> RuntimeSpec:
    return RuntimeSpec(
        image="ghcr.io/loopflowstudio/manabot-gpu:latest",
        fallback_build=True,
        log_group_prefix="/manabot",
    )
