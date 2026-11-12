"""Tests marked ``experiment_data(path, ...)`` fetch those manifest paths first.

Retained experiment data lives in S3 (see experiments/README.md). A test that
reads it fetches the paths it names; where the store cannot be read (no
credentials or network) it is skipped with the command that would fetch them.
Bytes that fail verification fail the test.

Every test gets its own ``ETUDE_HOME``, so Etude's kept statistics databases
and launcher state never reach the developer's home directory.

No test runs for five minutes. When one reaches the limit, the run prints
every thread's stack and exits with status 1. Collection has the same limit.
"""

import faulthandler
import os
from pathlib import Path

import pytest

from manabot.data import ROOT, DataUnavailable, fetch_data

_fetched: set[tuple[str, ...]] = set()

LIMIT_SECONDS = 300
_limit_output = pytest.StashKey[int]()


def _start_limit(config: pytest.Config) -> None:
    """Restart the five-minute clock; the process exits when it runs out.

    faulthandler's watchdog is a C thread, so it also ends a test stuck in
    native code or holding the interpreter lock. Its report goes to the real
    stderr saved before output capture begins.
    """
    faulthandler.dump_traceback_later(
        LIMIT_SECONDS, exit=True, file=config.stash[_limit_output]
    )


@pytest.fixture(autouse=True)
def _etude_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ETUDE_HOME", str(tmp_path / "etude-home"))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "experiment_data(*paths): fetch retained experiment data before the test",
    )
    config.stash[_limit_output] = os.dup(2)
    _start_limit(config)


def pytest_unconfigure(config: pytest.Config) -> None:
    faulthandler.cancel_dump_traceback_later()
    os.close(config.stash[_limit_output])


def pytest_runtest_setup(item: pytest.Item) -> None:
    _start_limit(item.config)
    paths = tuple(
        path for marker in item.iter_markers("experiment_data") for path in marker.args
    )
    if not paths or paths in _fetched:
        return
    try:
        fetch_data([str(ROOT / path) for path in paths])
    except DataUnavailable as error:
        pytest.skip(
            f"needs experiment data ({error}); fetch it with "
            f"`uv run --extra artifacts python -m manabot.data fetch {' '.join(paths)}`"
        )
    _fetched.add(paths)
