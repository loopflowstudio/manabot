"""
test_sandbox.py
Tests for sandbox lifecycle orchestration with a provider double.
"""

from __future__ import annotations

from ops.provider import Machine
from ops.sandbox import SandboxManager
from tests.ops.conftest import FakeProvider, make_runtime_spec, make_sandbox_spec


def test_sandbox_up_creates_when_missing():
    provider = FakeProvider()
    manager = SandboxManager(provider, make_sandbox_spec(), make_runtime_spec(), no_ssh=True)

    machine = manager.up()

    assert machine.id == "i-fake-1"
    assert len(provider.create_calls) == 1
    assert provider.started == []


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

    manager = SandboxManager(provider, make_sandbox_spec(), make_runtime_spec(), no_ssh=True)
    machine = manager.up()

    assert machine.id == "i-existing"
    assert len(provider.started) == 1
    assert len(provider.create_calls) == 0


def test_sandbox_stop_and_terminate_are_noops_when_missing():
    provider = FakeProvider()
    manager = SandboxManager(provider, make_sandbox_spec(), make_runtime_spec(), no_ssh=True)

    manager.stop()
    manager.terminate()

    assert provider.stopped == []
    assert provider.terminated == []


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
    manager = SandboxManager(provider, make_sandbox_spec(), make_runtime_spec(), no_ssh=True)

    manager.stop()
    manager.terminate()

    assert len(provider.stopped) == 1
    assert len(provider.terminated) == 1
