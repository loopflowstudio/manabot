"""A checkpoint is read with the readings it recorded, never with today's rule."""

from pathlib import Path

import numpy as np
import pytest
import torch

from manabot.backends.torch.checkpoint import load_checkpoint, save_checkpoint
from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import (
    FACT_COUNTER,
    FACT_DIM,
    FACT_DURATION_INDEX,
    FACT_EFFECT_PROGRAM,
    FACT_KIND_INDEX,
    FACT_TRANSFORMS,
    FACT_VALID_INDEX,
    ObservationSpace,
    encode_observation,
)
from manabot.infra.config import ModelConfig
from manabot.policy.admission import binding
from manabot.policy.readings import (
    OWN_PREPARATION,
    Reading,
    fresh_readings,
    kind_reading,
    legacy_readings,
    published_columns,
    read_numeric,
    signed_log,
)
import managym
from tests.model.checkpoint_fixture import read_checkpoint, write_checkpoint
from tests.model.prepared_routes import fact_rows


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
    return {key: value[None] for key, value in encode_observation(space, raw).items()}


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


def test_checkpoint_without_readings_reads_as_before(tmp_path: Path) -> None:
    space = ObservationSpace()
    policy = ActorCritic(
        space, ModelConfig(hidden_dim=8, world="world"), readings=legacy_readings(space)
    )
    path = tmp_path / "policy.safetensors"
    save_checkpoint(policy, space, path, player_configs=None)
    record = read_checkpoint(path)
    del record["input_readings"]
    write_checkpoint(record, path)
    loaded, _ = load_checkpoint(str(path))
    assert loaded.readings == legacy_readings(space)
    assert loaded.readings["agent_permanents"][_timestamp(space)] is Reading.AS_IS


def test_legacy_readings_ignore_native_kind_corrections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    space = ObservationSpace()
    original = legacy_readings(space)
    observation_layout = managym.observation_layout

    def corrected_layout(
        **options: bool,
    ) -> dict[str, "managym.ObservationArrayLayout"]:
        layout = observation_layout(**options)
        layout["relations"]["columns"]["parent"]["kind"] = "reference"
        return layout

    monkeypatch.setattr(managym, "observation_layout", corrected_layout)
    parent = next(
        i
        for i, column in enumerate(published_columns(space)["relations"])
        if column.name == "parent"
    )
    assert fresh_readings(space)["relations"][parent] is Reading.GATHER
    assert legacy_readings(space) == original


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


def test_a_float_buffer_without_a_reading_stops_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(OWN_PREPARATION, "decklists")
    with pytest.raises(ValueError, match="no reading or own preparation.*decklists"):
        ActorCritic(ObservationSpace(), ModelConfig(hidden_dim=8))


def test_fact_payloads_read_by_their_fact_kind() -> None:
    values = np.array([[0.3, 0.0], [7.0, 0.0], [4.0, 81.0]], dtype=np.float32)
    kinds = np.array([FACT_COUNTER, FACT_TRANSFORMS, FACT_EFFECT_PROGRAM])
    fresh = read_numeric(values, (Reading.FACT_VALUE,) * 2, kinds)
    expected = np.array([[0.3, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    expected[1, 0] = signed_log(np.array([7.0]))[0]
    np.testing.assert_array_equal(fresh, expected)
    raw = read_numeric(values, (Reading.FACT_VALUE_RAW,) * 2, kinds)
    np.testing.assert_array_equal(raw, values)


@pytest.mark.parametrize("duration", [3, 4])
def test_durations_three_and_four_reach_a_fresh_model_only(duration: int) -> None:
    space = ObservationSpace()
    facts = torch.zeros(1, 2, FACT_DIM)
    facts[0, :, FACT_KIND_INDEX] = FACT_COUNTER
    facts[0, :, FACT_VALID_INDEX] = 1
    facts[0, 1, FACT_DURATION_INDEX] = duration
    objects = torch.zeros(1, 2, 8)
    with torch.no_grad():
        fresh = fact_rows(facts, objects, readings=fresh_readings(space)).columns
        legacy = fact_rows(facts, objects).columns
    assert not torch.equal(fresh[0, 0], fresh[0, 1])
    torch.testing.assert_close(legacy[0, 0], legacy[0, 1], rtol=0, atol=0)
