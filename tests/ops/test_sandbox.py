"""
test_sandbox.py
Tests for sandbox lifecycle orchestration with a provider double.
"""

from __future__ import annotations

from dataclasses import dataclass

from ops.provider import CommandResult, Machine, MachineSpec, RuntimeSpec
from ops.sandbox import SandboxManager


@dataclass
class FakeProvider:
    user: str = "tester"

    def __post_init__(self):
        self.machines: dict[str, Machine] = {}
        self.created = 0
        self.started = 0
        self.stopped = 0
        self.terminated = 0

    def create(self, spec, tags, *, user_data=None, iam_instance_profile=None):
        self.created += 1
        machine = Machine(
            id=f"i-created-{self.created}",
            public_ip="1.2.3.4",
            status="running",
            tags=tags,
        )
        self.machines[machine.id] = machine
        return machine

    def start(self, machine):
        self.started += 1
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
        self.stopped += 1
        self.machines[machine.id] = Machine(
            id=machine.id,
            public_ip=machine.public_ip,
            status="stopped",
            tags=machine.tags,
        )

    def terminate(self, machine):
        self.terminated += 1
        self.machines[machine.id] = Machine(
            id=machine.id,
            public_ip=None,
            status="terminated",
            tags=machine.tags,
        )

    def list(self, tags):
        def matches(machine):
            return all(machine.tags.get(key) == value for key, value in tags.items())

        return [machine for machine in self.machines.values() if matches(machine)]

    def run_command(self, machine, command, timeout=3600):
        return CommandResult(
            command_id="cmd-1",
            status="Success",
            stdout="READY\n",
            stderr="",
        )

    def logs(self, group, stream_prefix, limit=200):
        return []


def _sandbox_spec() -> MachineSpec:
    return MachineSpec(
        instance_type="g5.xlarge",
        region="us-east-1",
        disk_gb=100,
        ami="ami-test",
        spot=False,
    )


def _runtime_spec() -> RuntimeSpec:
    return RuntimeSpec(
        image="ghcr.io/loopflowstudio/manabot-gpu:latest",
        fallback_build=True,
    )


def test_sandbox_up_creates_when_missing():
    provider = FakeProvider()
    manager = SandboxManager(provider, _sandbox_spec(), _runtime_spec(), no_ssh=True)

    machine = manager.up()

    assert machine.id == "i-created-1"
    assert provider.created == 1
    assert provider.started == 0


def test_sandbox_up_starts_existing_stopped_machine():
    provider = FakeProvider()
    existing = Machine(
        id="i-existing",
        public_ip="1.2.3.4",
        status="stopped",
        tags={
            "manabot:user": "tester",
            "manabot:role": "sandbox",
            "manabot:managed": "true",
            "manabot:region": "us-east-1",
        },
    )
    provider.machines[existing.id] = existing

    manager = SandboxManager(provider, _sandbox_spec(), _runtime_spec(), no_ssh=True)
    machine = manager.up()

    assert machine.id == "i-existing"
    assert provider.started == 1
    assert provider.created == 0


def test_sandbox_stop_and_terminate_are_noops_when_missing():
    provider = FakeProvider()
    manager = SandboxManager(provider, _sandbox_spec(), _runtime_spec(), no_ssh=True)

    manager.stop()
    manager.terminate()

    assert provider.stopped == 0
    assert provider.terminated == 0


def test_sandbox_stop_and_terminate_operate_on_existing_machine():
    provider = FakeProvider()
    existing = Machine(
        id="i-existing",
        public_ip="1.2.3.4",
        status="running",
        tags={
            "manabot:user": "tester",
            "manabot:role": "sandbox",
            "manabot:managed": "true",
            "manabot:region": "us-east-1",
        },
    )
    provider.machines[existing.id] = existing
    manager = SandboxManager(provider, _sandbox_spec(), _runtime_spec(), no_ssh=True)

    manager.stop()
    manager.terminate()

    assert provider.stopped == 1
    assert provider.terminated == 1
