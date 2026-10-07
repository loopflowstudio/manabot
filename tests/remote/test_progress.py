"""Live transport preserves producer receipts and stops on corrupt committed bytes."""

from pathlib import Path
import subprocess
import time
from typing import Callable

import pytest

from manabot.remote.plan import digest
from manabot.remote.progress import LiveExports
from manabot.remote.transport import Transport
from tests.training.test_checkpoint_queue import run_fixture


class ExportTransport(Transport):
    def __init__(self, raw: bytes, policy: bytes) -> None:
        self.raw, self.policy = raw, policy
        self.downloads = 0

    def shell(self, script: str, *, observe: Callable[[], None] | None = None) -> bytes:
        return self.raw

    def get(self, remote: str, local: Path) -> None:
        self.downloads += 1
        local.write_bytes(self.policy)


def test_live_exports_relocate_without_rewriting_source(tmp_path: Path) -> None:
    run = run_fixture(tmp_path / "original.json")
    reference = {
        "path": "/workspace/evidence/run/policy-0.pt",
        "sha256": digest(b"policy"),
        "bytes": 6,
    }
    run.stages[0].artifacts["raw"] = reference
    raw = run.model_dump_json(indent=2).encode()
    remote = ExportTransport(raw, b"policy")
    live = LiveExports(tmp_path / "live")
    snapshot = live.retrieve(remote)
    assert snapshot is not None and snapshot.read_bytes() == raw
    local = live.resolve(reference)
    assert (
        local["sha256"] == reference["sha256"]
        and Path(local["path"]).read_bytes() == b"policy"
    )
    assert (
        run.stages[0].artifacts["raw"]["path"] == "/workspace/evidence/run/policy-0.pt"
    )
    assert live.retrieve(remote) == snapshot and remote.downloads == 1


@pytest.mark.parametrize("bad_path", [False, True])
def test_live_export_corruption_and_escaping_paths_fail(
    tmp_path: Path, bad_path: bool
) -> None:
    run = run_fixture(tmp_path / "original.json")
    run.stages[0].artifacts["raw"] = {
        "path": "/workspace/evidence/run/../../key"
        if bad_path
        else "/workspace/evidence/run/policy.pt",
        "sha256": digest(b"policy"),
        "bytes": 6,
    }
    remote = ExportTransport(run.model_dump_json().encode(), b"broken")
    live = LiveExports(tmp_path / "live")
    with pytest.raises(ValueError):
        live.retrieve(remote)
    assert live.latest is None and not live.references
    assert remote.downloads == (0 if bad_path else 1)


class Process:
    returncode: int | None = None
    calls: int = 0
    killed: bool = False

    def communicate(
        self, payload: bytes | None = None, timeout: float | None = None
    ) -> tuple[bytes, bytes]:
        self.calls += 1
        if self.calls == 1:
            raise subprocess.TimeoutExpired("ssh", timeout or 1)
        self.returncode = 0
        return b"finished", b""

    def poll(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


@pytest.mark.parametrize("observer_error", [False, True])
def test_observer_runs_during_command_and_failure_reaps_ssh(
    monkeypatch: pytest.MonkeyPatch, observer_error: bool
) -> None:
    process = Process()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    transport = Transport.__new__(Transport)
    transport.options, transport.port, transport.target = [], 22, "root@127.0.0.1"
    transport.deadline = time.time() + 60
    deadline = transport.deadline
    observed: list[int] = []

    def observe() -> None:
        observed.append(process.calls)
        if observer_error:
            raise ValueError("observer failed")

    if observer_error:
        with pytest.raises(ValueError, match="observer failed"):
            transport.shell("true", observe=observe)
        assert process.killed
    else:
        assert transport.shell("true", observe=observe) == b"finished"
        assert observed == [1, 2]
    assert process.calls >= 2 and transport.deadline == deadline
