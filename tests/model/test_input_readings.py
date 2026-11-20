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
)
from manabot.infra.config import ModelConfig
from manabot.policy.admission import binding
from manabot.policy.readings import (
    OWN_PREPARATION,
    ColumnKind,
    Reading,
    Sized,
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
        key: [str(value) for value in values]
        for key, values in fresh_readings(space).items()
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
    with pytest.raises(ValueError, match="no reading for column x of kind 'magnitude'"):
        kind_reading(ColumnKind("x", "magnitude", None))
    with pytest.raises(ValueError, match="size None"):
        kind_reading(ColumnKind("x", "category", None))


def _column(space: ObservationSpace, key: str, name: str) -> int:
    return [c.name for c in published_columns(space)[key]].index(name)


def test_saved_model_records_sized_readings(
    saved: tuple[ActorCritic, ObservationSpace, Path],
) -> None:
    _, space, path = saved
    recorded = read_checkpoint(path)["input_readings"]
    frame_kind = managym.observation_layout()["relations"]["columns"]["frame_kind"]
    assert recorded["relations"][frame_kind["index"]] == f"one_hot:{frame_kind['size']}"
    assert recorded["relations"][_column(space, "relations", "frame")] == "gather"
    assert recorded["actions"][_column(space, "actions", "mode")] == "one_hot:15"
    assert (
        recorded["agent_permanents"][_column(space, "agent_permanents", "power")]
        == "steps:8"
    )


@pytest.mark.parametrize(
    "value",
    ["one_hot", "one_hot:0", "as_is:3", "as_is:", "steps:x", "steps:+8", "steps:٨", 8],
)
def test_a_malformed_sized_reading_is_refused(
    saved: tuple[ActorCritic, ObservationSpace, Path], value: object
) -> None:
    _, space, path = saved
    record = read_checkpoint(path)
    record["input_readings"]["agent_permanents"][_timestamp(space)] = value
    write_checkpoint(record, path)
    with pytest.raises(ValueError, match="agent_permanents.timestamp"):
        load_checkpoint(str(path))


def test_a_category_size_is_kept_when_the_category_grows(tmp_path: Path) -> None:
    """A checkpoint keeps its recorded width; a code above it lights only the
    last flag, and every code it trained on reads as before."""
    space = ObservationSpace()
    readings = fresh_readings(space)
    frame_kind = _column(space, "relations", "frame_kind")
    relations = list(readings["relations"])
    relations[frame_kind] = Sized(Reading.ONE_HOT, 12)
    readings = {**readings, "relations": tuple(relations)}
    policy = ActorCritic(
        space, ModelConfig(hidden_dim=8, world="world"), readings=readings
    )
    path = tmp_path / "policy.safetensors"
    save_checkpoint(policy, space, path, player_configs=None)
    loaded, _ = load_checkpoint(str(path))
    assert loaded.readings["relations"][frame_kind] == Sized(Reading.ONE_HOT, 12)
    codes = np.array([[0], [3], [12], [14]], dtype=np.float32)
    flags = read_numeric(codes, (Sized(Reading.ONE_HOT, 12),))
    assert flags.shape == (4, 13)
    np.testing.assert_array_equal(flags.sum(1), [0, 1, 1, 1])
    assert flags[1, 2] == flags[2, 11] == flags[3, 12] == 1


def test_subunit_codes_leave_the_whole_one_hot_empty() -> None:
    rows = np.array([[0.25, 0.5], [0.25, -1]], dtype=np.float32)
    actual = read_numeric(rows, (Reading.AS_IS, Sized(Reading.ONE_HOT, 2)))
    np.testing.assert_array_equal(actual, [[0.25, 0, 0, 0]] * 2)


def test_one_hot_and_steps_read_as_documented() -> None:
    """`steps:8` is the stored tenths, then flags for at least 1..8; a 2/2 and
    a 3/3 differ by a whole flag. `one_hot` gives 0 no flag."""
    power = np.array([[0.2], [0.3], [1.2], [-0.1]], dtype=np.float32)
    steps = read_numeric(power, (Sized(Reading.STEPS, 8),))
    np.testing.assert_array_equal(
        steps,
        np.array(
            [
                [0.2, 1, 1, 0, 0, 0, 0, 0, 0],
                [0.3, 1, 1, 1, 0, 0, 0, 0, 0],
                [1.2, 1, 1, 1, 1, 1, 1, 1, 1],
                [-0.1, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        ),
    )
    assert np.linalg.norm(steps[1] - steps[0]) >= 1


def test_a_checkpoint_that_recorded_log_for_categories_loads_unchanged(
    tmp_path: Path,
) -> None:
    """Models written when categories and selectors read `log` keep it."""
    space = ObservationSpace()
    kinds = published_columns(space)
    readings = {
        key: tuple(
            Reading.LOG
            if column.kind in ("category", "selector") and isinstance(value, Sized)
            else (Reading.AS_IS if isinstance(value, Sized) else value)
            for column, value in zip(kinds[key], values, strict=True)
        )
        for key, values in fresh_readings(space).items()
    }
    policy = ActorCritic(
        space, ModelConfig(hidden_dim=8, world="world"), readings=readings
    ).eval()
    path = tmp_path / "policy.safetensors"
    save_checkpoint(policy, space, path, player_configs=None)
    loaded, _ = load_checkpoint(str(path))
    assert loaded.readings == readings
    batch = _batch(space)
    for before, after in zip(
        policy.evaluate(policy.prepare(batch)),
        loaded.evaluate(loaded.prepare(batch)),
        strict=True,
    ):
        np.testing.assert_array_equal(before, after)


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
