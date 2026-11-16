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
        snapshot = read_record(tmp_path / "0.snapshot.jsonl")
        assert snapshot.games[0].actions == []
        assert "test_stall_retains" in (tmp_path / "stacks.txt").read_text()
        # The current native call was never retroactively entered or retried.
        assert not (tmp_path / "0.jsonl").exists()
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
            in json.loads((tmp_path / "stall.json").read_text())["reason"]
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
    assert (tmp_path / "0.snapshot.jsonl").exists()
    assert (
        "collection" == json.loads((tmp_path / "stall.json").read_text())["operation"]
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
    record = read_record(tmp_path / "0.jsonl")
    assert record.boundary is not None and record.boundary.phase == "ready"
    assert record.games[0].committed == 1
    assert record.games[0].actions == [0]
