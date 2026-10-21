"""Tests marked ``experiment_data(path, ...)`` fetch those manifest paths first.

Retained experiment data lives in S3 (see experiments/README.md). A test that
reads it fetches the paths it names; where the store cannot be read (no
credentials or network) it is skipped with the command that would fetch them.
Bytes that fail verification fail the test.
"""

import pytest

from manabot.data import ROOT, DataUnavailable, fetch_data

_fetched: set[tuple[str, ...]] = set()


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
