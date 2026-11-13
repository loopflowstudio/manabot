"""Frozen attack execution refuses before creating a new cohort."""

from pathlib import Path

import pytest

from manabot.experiment.attack_execution import execute_attack_plan


def test_attack_execution_names_its_last_runnable_revision(tmp_path: Path) -> None:
    out = tmp_path / "attack"
    with pytest.raises(RuntimeError, match="f09f5c2919b8b1a5cf6d72c4f4f4721dac68f653"):
        execute_attack_plan(None, out)
    assert not out.exists()
