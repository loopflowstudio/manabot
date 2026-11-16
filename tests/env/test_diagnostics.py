"""Injected time and blocking gates diagnose stalls without resource pressure."""

import json
from pathlib import Path
import threading

import managym
from managym.diagnostics import Diagnostics
from managym.recording import GameRecord, ScalarCapture, read_record


def test_stall_retains_phase_stacks_and_cached_cursor(tmp_path: Path) -> None:
    env = managym.Env(seed=7)
    capture = ScalarCapture(seed=7)
    capture.reset(
        env,
        [managym.PlayerConfig(name=str(i), decklist={"Mountain": 12}) for i in (0, 1)],
    )
    now = [0.0]
    with Diagnostics(
        tmp_path, captures=[capture], stall_seconds=10, clock=lambda: now[0]
    ) as diagnostics:
        with diagnostics.operation("collection"):
            now[0] = 11
            diagnostics.check()
            assert diagnostics.level == "detailed"
        marker = json.loads((tmp_path / "operation.json").read_text())
        assert marker["operation"] == "collection"
        snapshot = read_record(tmp_path / "scalar.snapshot.json")
        assert snapshot.games[0].actions == []
        assert "test_stall_retains" in (tmp_path / "stacks.txt").read_text()
        # The current native call was never retroactively entered or retried.
        assert not (tmp_path / "scalar.jsonl").exists()
    assert not any(t.name == "managym-watchdog" for t in threading.enumerate())


def test_injected_slow_rate_escalates(tmp_path: Path) -> None:
    now = [0.0]
    with Diagnostics(tmp_path, clock=lambda: now[0]) as diagnostics:
        now[0] = 1
        diagnostics.progress(100)
        now[0] = 11
        diagnostics.progress(100)
        assert diagnostics.level == "detailed"
        assert (
            "collection rate"
            in json.loads((tmp_path / "escalations.jsonl").read_text())["reason"]
        )


def test_off_does_not_create_output(tmp_path: Path) -> None:
    path = tmp_path / "off"
    with Diagnostics(path, level="off") as diagnostics:
        with diagnostics.operation("collection"):
            diagnostics.progress(10)
            diagnostics.escalate("operator")
    assert not path.exists()


def test_blocked_operation_is_diagnosed_without_a_logging_flag(tmp_path: Path) -> None:
    retained = threading.Event()

    class Cursor(ScalarCapture):
        def snapshot(self) -> "GameRecord":
            snapshot = super().snapshot()
            retained.set()
            return snapshot

    env = managym.Env(seed=7)
    capture = Cursor(seed=7)
    capture.reset(
        env,
        [managym.PlayerConfig(name=str(i), decklist={"Mountain": 12}) for i in (0, 1)],
    )
    with Diagnostics(tmp_path, captures=[capture], stall_seconds=0.05) as diagnostics:
        with diagnostics.operation("collection"):
            assert retained.wait(2), "watchdog did not retain the blocked operation"
    assert (tmp_path / "scalar.snapshot.json").exists()
    assert (
        "collection"
        == json.loads((tmp_path / "escalations.jsonl").read_text())["operation"]
    )


def test_detailed_attempt_journals_its_first_call(tmp_path: Path) -> None:
    env = managym.Env(seed=7)
    capture = ScalarCapture(seed=7)
    with Diagnostics(tmp_path, captures=[capture], level="detailed"):
        capture.reset(
            env,
            [
                managym.PlayerConfig(name=str(i), decklist={"Mountain": 12})
                for i in (0, 1)
            ],
        )
        capture.step(env, 0)
    record = read_record(tmp_path / "scalar.jsonl")
    assert record.boundary is not None and record.boundary.phase == "ready"
    assert record.games[0].committed == 1
    assert record.games[0].actions == [0]


def test_phase_progress_excludes_time_between_phases(tmp_path: Path) -> None:
    now = [0.0]
    with Diagnostics(tmp_path, clock=lambda: now[0]) as diagnostics:
        with diagnostics.operation("gather") as operation:
            now[0] = 1
            operation.progress(100)
        # Learning time between collections must not look like collection slowdown.
        now[0] = 100
        with diagnostics.operation("gather") as operation:
            now[0] = 101
            operation.progress(100)
        assert diagnostics.level == "standard"
        with diagnostics.operation("gather") as operation:
            now[0] = 111
            operation.progress(100)
        assert diagnostics.level == "detailed"


def test_per_step_marking_and_escalation_history(tmp_path: Path) -> None:
    with Diagnostics(tmp_path) as diagnostics:
        with diagnostics.operation("predict", per_step=True):
            assert not (tmp_path / "operation.json").exists()
        diagnostics.escalate("operator request")
        with diagnostics.operation("predict", per_step=True):
            assert (
                json.loads((tmp_path / "operation.json").read_text())["state"]
                == "pending"
            )
        diagnostics.escalate("second incident")
        reasons = [
            json.loads(line)["reason"]
            for line in (tmp_path / "escalations.jsonl").read_text().splitlines()
        ]
        assert reasons == ["operator request", "second incident"]


def test_signal_dumps_stacks_while_scalar_lock_is_held(tmp_path: Path) -> None:
    import os
    import signal
    import subprocess
    import sys
    import time

    # PyDLL retains the interpreter lock across this controlled blocking call,
    # matching scalar native calls without inducing any resource pressure.
    source = """
import ctypes
from pathlib import Path
import sys
from managym.diagnostics import Diagnostics
path = Path(sys.argv[1])
with Diagnostics(path, stall_seconds=600) as diagnostics:
    with diagnostics.operation("scalar_step"):
        (path / "ready").touch()
        ctypes.PyDLL(None).sleep(30)
"""
    process = subprocess.Popen([sys.executable, "-c", source, str(tmp_path)])
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "ready").exists():
            assert process.poll() is None, "signal fixture exited before entering call"
            assert time.monotonic() < deadline, "signal fixture did not start"
            time.sleep(0.01)
        os.kill(process.pid, signal.SIGUSR1)
        while not (tmp_path / "stacks.txt").read_text():
            assert time.monotonic() < deadline, "signal did not dump blocked stacks"
            time.sleep(0.01)
        assert 'File "<string>"' in (tmp_path / "stacks.txt").read_text()
        assert (
            json.loads((tmp_path / "operation.json").read_text())["operation"]
            == "scalar_step"
        )
    finally:
        process.terminate()
        process.wait(timeout=5)
