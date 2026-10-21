"""The supplemental watcher never treats initialization as learned weights."""

import pytest

from experiments.runners.baseline_checkpoint_watch import ready
from experiments.runners.corrected_baseline import recipe
from manabot.training.models import MonitoringCheckpoint, TrainingRun


def test_requires_exact_completed_export() -> None:
    run = TrainingRun.model_construct(
        seed=12261, regime=recipe(10000, 60000), monitoring_checkpoints=[]
    )
    assert not ready(run)
    checkpoint = MonitoringCheckpoint(
        stage_id="self-play", updates=5000, training_seconds=20000, ordinal=0
    )
    run.monitoring_checkpoints = [checkpoint]
    assert not ready(run)
    checkpoint.artifact = {"path": "/unused", "bytes": 1, "sha256": "a" * 64}
    assert ready(run)
    checkpoint.error = "failed export"
    assert not ready(run)
    run.seed = 12262
    with pytest.raises(ValueError):
        ready(run)
