"""A bounded child call stops its whole process tree at the deadline."""

import json
import multiprocessing as mp
from multiprocessing.connection import Connection
from pathlib import Path
import time

import psutil
import pytest

from manabot._process import bounded_call


def _wait() -> None:
    # A sleeping child is a lifecycle fixture, not artificial resource pressure.
    time.sleep(30)


def _orphan(sender: Connection, path: Path) -> None:
    child = mp.get_context("spawn").Process(target=_wait)
    child.start()
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps({"pid": child.pid}))
    temporary.replace(path)
    time.sleep(30)


def test_timeout_stops_owned_descendants_and_retains_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress = tmp_path / "child.json"
    real_time = time.time
    deadline = real_time() + 30
    # Advance the parent deadline only after the child's durable readiness
    # witness. Slow interpreter startup is not the behavior under test.
    monkeypatch.setattr(
        time, "time", lambda: deadline if progress.exists() else real_time()
    )
    outcome = bounded_call(
        _orphan,
        (progress,),
        deadline_unix=deadline,
        process_tree_bytes=2 * 2**30,
    )
    assert outcome.error and "deadline" in outcome.error
    assert outcome.cleanup_complete
    pid = json.loads(progress.read_text())["pid"]
    assert (
        not psutil.pid_exists(pid)
        or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    )
