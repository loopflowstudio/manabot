"""Tests marked ``experiment_data(path, ...)`` fetch those manifest paths first.

Retained experiment data lives in S3 (see experiments/README.md). A test that
reads it fetches the paths it names; where the store cannot be read (no
credentials or network) it is skipped with the command that would fetch them.
Bytes that fail verification fail the test.

Every test gets its own ``ETUDE_HOME``, so Etude's kept statistics databases
and launcher state never reach the developer's home directory.
"""

from pathlib import Path

import pytest

from manabot.data import ROOT, DataUnavailable, fetch_data

_fetched: set[tuple[str, ...]] = set()


@pytest.fixture(autouse=True)
def _etude_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ETUDE_HOME", str(tmp_path / "etude-home"))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "experiment_data(*paths): fetch retained experiment data before the test",
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
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
