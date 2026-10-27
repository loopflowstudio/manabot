"""The training memory guard survives children exiting mid-sample."""

import subprocess
import time
from types import SimpleNamespace

import psutil
import pytest

from manabot.training.execution import process_tree_rss


class _Child:
    def __init__(self, rss: int | None = None, error: Exception | None = None) -> None:
        self.rss, self.error = rss, error

    def memory_info(self) -> SimpleNamespace:
        if self.error is not None:
            raise self.error
        return SimpleNamespace(rss=self.rss)


class _Parent:
    def __init__(self, children: list[_Child]) -> None:
        self._children = children

    def memory_info(self) -> SimpleNamespace:
        return SimpleNamespace(rss=100)

    def children(self, recursive: bool) -> list[_Child]:
        assert recursive
        return self._children


def test_exited_and_zombie_children_hold_no_memory() -> None:
    parent = _Parent(
        [
            _Child(rss=10),
            _Child(error=psutil.ZombieProcess(53116)),
            _Child(error=psutil.NoSuchProcess(53117)),
            _Child(rss=5),
        ]
    )
    assert process_tree_rss(parent) == 115  # type: ignore[arg-type]


def test_permission_failures_still_surface() -> None:
    parent = _Parent([_Child(error=psutil.AccessDenied(53118))])
    with pytest.raises(psutil.AccessDenied):
        process_tree_rss(parent)  # type: ignore[arg-type]


<<<<<<< HEAD
=======
def test_measures_the_real_process_tree() -> None:
    assert process_tree_rss(psutil.Process()) > 0


>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
def test_real_unreaped_child_does_not_abort_sampling() -> None:
    # An exited child that has not been waited on is a zombie; on macOS reading
    # its memory raises ZombieProcess (seed 12262's Mini abort, 2026-10-16).
    child = subprocess.Popen(["true"])
    try:
        deadline = time.monotonic() + 5
        while psutil.Process(child.pid).status() != psutil.STATUS_ZOMBIE:
            assert time.monotonic() < deadline, "child never became a zombie"
            time.sleep(0.01)
        assert process_tree_rss(psutil.Process()) > 0
    finally:
        child.wait()
