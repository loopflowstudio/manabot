"""The training memory guard survives children exiting mid-sample."""

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


def test_measures_the_real_process_tree() -> None:
    assert process_tree_rss(psutil.Process()) > 0
