"""Experiment data stays out of git; the manifest is its only record there."""

from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_no_experiment_data_is_tracked_by_git() -> None:
    if not (ROOT / ".git").exists():
        pytest.skip(
            "not a git checkout; source archives export-ignore experiments/data"
        )
    tracked = subprocess.run(
        ["git", "ls-files", "--", "experiments/data"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    assert not tracked, (
        f"{len(tracked)} files under experiments/data are tracked by git "
        f"(first: {tracked[0]}). Store them and record them in the manifest with "
        "`uv run --extra artifacts python -m manabot.data publish <path>`, then "
        "`git rm --cached` them."
    )
