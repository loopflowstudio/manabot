"""A checkpoint is read with the readings it recorded, never with today's rule."""

from pathlib import Path

import numpy as np
import pytest

from manabot.backends.torch.checkpoint import load_checkpoint, save_checkpoint
from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.policy.admission import binding
from manabot.policy.readings import (
    Reading,
    fresh_readings,
    kind_reading,
    legacy_readings,
    published_columns,
)
import managym
from tests.model.checkpoint_fixture import read_checkpoint, write_checkpoint


def _batch(space: ObservationSpace) -> dict[str, np.ndarray]:
    env = managym.Env(seed=7, skip_trivial=True)
    raw, _ = env.reset(
        [
            managym.PlayerConfig("A", {"Mountain": 20, "Gray Ogre": 20}),
            managym.PlayerConfig("B", {"Forest": 20, "Gray Ogre": 20}),
        ]
    )
    for _ in range(12):
        raw, *_ = env.step(0)
    buffers = space.encoder.allocate()
    env.encode_observation_into(raw, buffers)
    return {key: value[None] for key, value in buffers.items()}


def _timestamp(space: ObservationSpace) -> int:
    names = [c.name for c in published_columns(space)["agent_permanents"]]
    return names.index("timestamp")


@pytest.fixture
def saved(tmp_path: Path) -> tuple[ActorCritic, ObservationSpace, Path]:
    space = ObservationSpace()
    policy = ActorCritic(space, ModelConfig(hidden_dim=8, world="world")).eval()
    path = tmp_path / "policy.safetensors"
    save_checkpoint(policy, space, path, player_configs=None)
    return policy, space, path


def test_saved_model_reloads_with_its_recorded_readings(
    saved: tuple[ActorCritic, ObservationSpace, Path],
) -> None:
    policy, space, path = saved
    record = read_checkpoint(path)
    timestamp = _timestamp(space)
    assert record["input_readings"]["agent_permanents"][timestamp] == "rank"
    assert record["input_readings"] == {
        key: list(values) for key, values in fresh_readings(space).items()
    }
    loaded, _ = load_checkpoint(str(path))
    assert loaded.readings == policy.readings
    batch = _batch(space)
    for before, after in zip(
        policy.evaluate(policy.prepare(batch)),
        loaded.evaluate(loaded.prepare(batch)),
        strict=True,
    ):
        np.testing.assert_array_equal(before, after)


def test_checkpoint_without_readings_reads_as_before(
    saved: tuple[ActorCritic, ObservationSpace, Path],
) -> None:
    _, space, path = saved
    record = read_checkpoint(path)
    del record["input_readings"]
    write_checkpoint(record, path)
    loaded, _ = load_checkpoint(str(path))
    assert loaded.readings == legacy_readings(space)
    assert loaded.readings["agent_permanents"][_timestamp(space)] is Reading.AS_IS


def test_unknown_recorded_reading_is_refused(
    saved: tuple[ActorCritic, ObservationSpace, Path],
) -> None:
    _, space, path = saved
    record = read_checkpoint(path)
    record["input_readings"]["agent_permanents"][_timestamp(space)] = "squared"
    write_checkpoint(record, path)
    with pytest.raises(ValueError, match="agent_permanents.timestamp.*'squared'"):
        load_checkpoint(str(path))


def test_models_differing_in_one_reading_never_share_a_batch() -> None:
    space = ObservationSpace()
    model = ModelConfig()
    readings = fresh_readings(space)
    timestamp = _timestamp(space)
    permanents = list(readings["agent_permanents"])
    permanents[timestamp] = Reading.AS_IS
    other = {
        **readings,
        "agent_permanents": tuple(permanents),
        "opponent_permanents": tuple(permanents),
    }
    assert binding(model, space, readings) != binding(model, space, other)


def test_a_kind_without_a_rule_has_no_reading() -> None:
    with pytest.raises(ValueError, match="no reading for column kind 'magnitude'"):
        kind_reading("magnitude", None)
