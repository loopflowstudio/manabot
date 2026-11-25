"""A published policy supplies weights and input readings, while a fresh loop
owns Adam and counters."""

import json
from pathlib import Path

import pytest
import torch

from manabot.backends.torch.checkpoint import load_checkpoint
from manabot.policy.checkpoint import read_safetensors, write_safetensors
from manabot.policy.readings import Reading, fresh_readings, legacy_readings
from manabot.recipes.builders import gray_ogre_self_play
from manabot.training.loops.self_play import CLOCKS, create, self_play
from tests.fixtures.training_records import recording


def publish(directory: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """One update of the Gray Ogre loop; the path of its published policy."""
    monkeypatch.setenv("ETUDE_OUT", str(directory / "parent"))
    with recording(
        experiment="test",
        run="pretrained",
        clocks=CLOCKS,
        directory=directory / "track",
    ) as run:
        return self_play(gray_ogre_self_play(horizon=1), run, updates=1)


@pytest.fixture
def published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return publish(tmp_path, monkeypatch)


def test_pretrained_weights_start_with_fresh_optimizer_and_counts(
    published: Path,
) -> None:
    settings = gray_ogre_self_play(horizon=1)
    learner, _ = create(settings.model_copy(update={"pretrained": published}), seed=198)
    saved = load_checkpoint(published)[0].state_dict()
    for name, value in learner.agent.state_dict().items():
        torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
    assert not learner.optimizer.state
    changed = settings.model_copy(
        update={
            "model": settings.model.model_copy(update={"hidden_dim": 32}),
            "pretrained": published,
        }
    )
    with pytest.raises(ValueError, match="differ from requested arm"):
        create(changed, seed=198)


def test_warm_start_keeps_the_readings_of_a_policy_that_recorded_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A policy from before readings were recorded: built by the old rule, its
    # record stripped of the readings.
    with monkeypatch.context() as old_rule:
        old_rule.setattr(
            "manabot.backends.torch.policy.fresh_readings", legacy_readings
        )
        published_before = publish(tmp_path, old_rule)
    arrays, metadata = read_safetensors(published_before)
    del metadata["input_readings"]
    parent = tmp_path / "before-readings.safetensors"
    write_safetensors(parent, arrays, metadata)

    settings = gray_ogre_self_play(horizon=1)
    learner, _ = create(settings.model_copy(update={"pretrained": parent}), seed=198)
    space = learner.agent.observation_space
    widths = {
        key: len(value) for key, value in json.loads(metadata["input_columns"]).items()
    }
    legacy = {
        key: value[: widths[key]] for key, value in legacy_readings(space).items()
    }
    assert fresh_readings(space)["agent_permanents"] != legacy["agent_permanents"]
    assert Reading.RANK not in learner.agent.readings["agent_permanents"]
    assert learner.agent.readings == legacy
