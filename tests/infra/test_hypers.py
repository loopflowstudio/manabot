"""
test_hypers.py
Tests for hyperparameter defaults that depend on process environment.
"""

import os
import shutil

from manabot.infra import Hypers


def test_runs_dir_env_is_used_when_current_directory_was_removed(monkeypatch, tmp_path):
    original_cwd = os.getcwd()
    removed_cwd = tmp_path / "removed"
    removed_cwd.mkdir()
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MANABOT_RUNS_DIR", str(runs_dir))

    try:
        os.chdir(removed_cwd)
        shutil.rmtree(removed_cwd)

        hypers = Hypers()
        assert hypers.experiment.runs_dir == runs_dir
    finally:
        os.chdir(original_cwd)
