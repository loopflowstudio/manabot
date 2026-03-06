"""
test_sandbox.py
Tests for sandbox lifecycle orchestration with a provider double.
"""

from __future__ import annotations

import pytest

from ops.provider import Machine
from ops.sandbox import SandboxManager, build_parser, detect_local_repo_ref
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


def test_sandbox_start_running_machine_opens_ssh_with_explicit_key(monkeypatch):
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

    captured: dict[str, object] = {}

    def fake_run(command, check):
        captured["command"] = command
        captured["check"] = check

    monkeypatch.setattr("ops.sandbox.subprocess.run", fake_run)

    manager = SandboxManager(
        provider,
        make_sandbox_spec(),
        make_runtime_spec(),
        ssh_key_path="/tmp/test-key",
        ssh_command="uname -a",
        update_ssh_config=False,
    )

    manager.start()

    assert captured["check"] is False
    command = captured["command"]
    assert command[0] == "ssh"
    assert "-i" in command
    assert "/tmp/test-key" in command
    assert "IdentitiesOnly=yes" in command
    assert "ubuntu@1.2.3.4" in command
    assert command[-1] == "uname -a"


def test_sandbox_start_updates_ssh_config_alias(monkeypatch, tmp_path):
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

    captured: dict[str, object] = {}
    config_path = tmp_path / "ssh_config"

    def fake_run(command, check):
        captured["command"] = command
        captured["check"] = check

    monkeypatch.setattr("ops.sandbox.subprocess.run", fake_run)
    monkeypatch.setattr(SandboxManager, "_ssh_config_path", lambda self: config_path)

    manager = SandboxManager(
        provider,
        make_sandbox_spec(),
        make_runtime_spec(),
        ssh_host_alias="manabox",
    )

    manager.start()

    assert captured["command"] == ["ssh", "manabox"]
    rendered = config_path.read_text(encoding="utf-8")
    assert "Host manabox" in rendered
    assert "HostName 1.2.3.4" in rendered
    assert "User ubuntu" in rendered


def test_sandbox_start_raises_without_public_ip_when_ssh_enabled():
    provider = FakeProvider()
    existing = Machine(
        id="i-existing",
        public_ip=None,
        status="running",
        tags={
            "manabot:user": "tester",
            "manabot:role": "sandbox",
            "manabot:managed": "true",
            "manabot:region": "us-east-1",
        },
    )
    provider.machines[existing.id] = existing

    manager = SandboxManager(provider, make_sandbox_spec(), make_runtime_spec())

    with pytest.raises(RuntimeError, match="has no public IP"):
        manager.start()


def test_parser_reads_ssh_key_path_from_environment(monkeypatch):
    monkeypatch.setenv("MANABOT_SSH_KEY_PATH", "/tmp/from-env")

    parser = build_parser()
    args = parser.parse_args(["--status"])

    assert args.ssh_key_path == "/tmp/from-env"


def test_detect_local_repo_ref_prefers_environment(monkeypatch):
    monkeypatch.setenv("MANABOT_REPO_REF", "feature/test")

    assert detect_local_repo_ref() == "feature/test"
