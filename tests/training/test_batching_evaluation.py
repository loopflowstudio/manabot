"""Matched-experience evaluation cannot silently substitute another export."""

from pathlib import Path
import shutil

import pytest
import torch

from experiments.runners.batching_evaluation import select_checkpoints
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.training.execution import execute_regime
from manabot.training.models import TrainingRun
from manabot.verify.store import VerifyStore
from tests.training.test_recovery import recipe


@pytest.fixture
def retained_run(tmp_path: Path) -> Path:
    torch.set_num_threads(1)
    source = tmp_path / "original"
    with VerifyStore(tmp_path / "training.sqlite") as store:
        run = execute_regime(recipe(), 13590, source, store, checkpoint_updates=(1, 2))
    assert run.status == "completed"
    relocated = tmp_path / "retained"
    shutil.move(str(source), relocated)
    return relocated


def test_selects_initial_milestone_and_endpoint_after_relocation(
    retained_run: Path,
) -> None:
    selected = select_checkpoints(retained_run, (0, 64, 128, 192))
    assert [point.coordinates.updates for point in selected.points] == [0, 1, 2, 3]
    assert [point.coordinates.learner_transitions for point in selected.points] == [
        0,
        64,
        128,
        192,
    ]
    for point in selected.points:
        assert Path(point.artifact["path"]).parent == retained_run
        agent, _ = load_checkpoint_agent(point.artifact["path"])
        assert all(bool(torch.isfinite(p).all()) for p in agent.parameters())
    # Selection does not rewrite the original producer paths or its run receipt.
    original = TrainingRun.model_validate_json((retained_run / "run.json").read_text())
    assert Path(original.stages[0].artifacts["raw"]["path"]).parent.name == "original"


def test_rejects_missing_experience_or_corrupt_export(retained_run: Path) -> None:
    with pytest.raises(ValueError, match="65 transitions"):
        select_checkpoints(retained_run, (0, 65, 192))
    selected = select_checkpoints(retained_run, (0, 192))
    Path(selected.points[-1].artifact["path"]).write_bytes(b"corrupt checkpoint")
    with pytest.raises(ValueError, match="checkpoint bytes differ"):
        select_checkpoints(retained_run, (0, 192))


def test_rejects_incomplete_trajectory(retained_run: Path) -> None:
    path = retained_run / "run.json"
    run = TrainingRun.model_validate_json(path.read_text())
    run.status = "interrupted"
    path.write_text(run.model_dump_json())
    with pytest.raises(ValueError, match="completed fresh single"):
        select_checkpoints(retained_run, (0, 64))
