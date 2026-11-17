"""Standalone native child lifetime and bounded pool environment."""

from multiprocessing.connection import Connection
import os

import pytest

from manabot.sim.child import ChildProcess
from manabot.sim.threads import POOL_VARIABLES, child_environment


def _report_environment(connection: Connection, names: tuple[str, ...]) -> None:
    connection.send({name: os.environ.get(name) for name in names})
    connection.close()


def test_child_starts_with_single_threaded_pools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = ChildProcess(
        "tests.sim.test_child_surface:_report_environment", (POOL_VARIABLES,)
    )
    try:
        assert child.connection.recv() == dict.fromkeys(POOL_VARIABLES, "1")
        child.join(60)
    finally:
        child.close()
    assert child.exitcode == 0
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("HOME", "/h")
    assert child_environment() == {
        **os.environ,
        **dict.fromkeys(POOL_VARIABLES, "1"),
    }
    assert child_environment()["HOME"] == "/h"


def test_child_with_a_missing_target_exits_with_failure() -> None:
    missing = ChildProcess("manabot.sim.child_absent:main", ())
    missing.join(60)
    assert missing.exitcode not in (None, 0)
    missing.close()
