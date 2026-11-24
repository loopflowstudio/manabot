"""Native observation arrays by column name, and the Python mirror of them.

Outside consumers read columns from `managym.observation_layout()`; manabot's
column constants and space shapes are held to that layout here.
"""

import json
import subprocess
import sys
from typing import Literal

import numpy as np
import pytest

from manabot.env import observation
from manabot.env.decision_memory import MEMORY_WIDTHS
from manabot.env.legacy_input import LEGACY_PERMANENT_DIM
from manabot.env.observation import (
    ObservationSpace,
    allocate_vector_buffers,
    encode_observation,
)
from manabot.env.shared_effects import SHARED_EFFECT_SHAPES
from manabot.infra.config import ObservationConfig
from manabot.verify.representation import kellan_upgrade_root
from manabot.verify.text_observation import priority_root
import managym


@pytest.fixture(scope="module")
def board() -> tuple[managym.Env, managym.Observation]:
    env = priority_root(("Lightning Bolt",), ("Mountain", "Shivan Dragon"))
    return env, env.observation_for_player(env.current_agent_index())


def test_column_constants_are_read_from_the_native_layout() -> None:
    layout = managym.observation_layout()
    assert observation.ACTION_TYPE_DIM == len(observation.ActionEnum)
    assert (
        observation.PERMANENT_DEFINITION_INDEX
        == layout["agent_permanents"]["columns"]["definition"]["index"]
    )
    assert observation.FACT_DIM == len(layout["facts"]["columns"])
    assert observation.RELATION_ROWS == layout["relations"]["shape"][0]
    assert LEGACY_PERMANENT_DIM == len(
        managym.observation_layout(legacy=True)["agent_permanents"]["columns"]
    )
    for key, shape in SHARED_EFFECT_SHAPES.items():
        assert tuple(layout[key]["shape"]) == shape, key
    memory = managym.observation_layout(decision_memory=True)
    assert MEMORY_WIDTHS.keys() == {k for k in memory if k.startswith("decision_")}


@pytest.mark.parametrize("version", [9, 10, 11])
def test_axis_extents_use_the_configured_layout(version: int) -> None:
    layout = managym.ObservationLayout(
        policy_input_version=version, max_stack_objects=32 if version == 11 else 0
    )
    obs = layout.allocate(3)
    # Reverse the row and column strides and broadcast the batch stride.
    deck = obs["decklists"][:, :, ::-1, ::-1]
    deck[0, 1, 9, 0] = np.nan
    obs["decklists"] = np.broadcast_to(deck[:1], deck.shape)
    obs["action_focus"].fill(-1)
    obs["actions_valid"][1, 2] = 1
    obs["action_focus"][:, 2, 0] = 11
    obs["action_focus"][:, 3, 0] = 31  # beyond every mask prefix
    obs["actions"][2, 4, 0] = 1  # extends action rows, never eligible focus
    extents = layout.axis_extents(obs)
    assert extents is not None and ("stack" in extents) == (version == 11)
    np.testing.assert_array_equal(
        extents["cards"], np.array([0, 10, 0], np.int64), strict=True
    )
    np.testing.assert_array_equal(extents["actions"], [0, 3, 5])
    np.testing.assert_array_equal(extents["decklists"], [10, 10, 10])
    empty = layout.axis_extents(layout.allocate(0))
    assert empty is not None and all(value.shape == (0,) for value in empty.values())


@pytest.mark.parametrize("version", [9, 10, 11])
def test_axis_buffers_locate_each_measured_row_dimension(version: int) -> None:
    layout = managym.ObservationLayout(policy_input_version=version)
    buffers = layout.axis_buffers
    rows = layout.allocate(1)
    extents = layout.axis_extents(rows)
    assert extents is not None and extents.keys() == buffers.keys()
    shapes = layout.shapes
    # Plant only the final row of each companion, including seat-major buffers.
    for axis, companions in buffers.items():
        for key, dim in companions:
            row = rows[key]
            capacity = shapes[key][dim]
            last = [0] * row.ndim
            last[dim + 1] = capacity - 1
            row[tuple(last)] = 1
            if key == "action_focus":
                rows["actions_valid"][0, -1] = 1
            measured = layout.axis_extents(rows)
            assert measured is not None and measured[axis][0] == capacity, key
            row[tuple(last)] = 0
            rows["actions_valid"][0, -1] = 0


def test_axis_extents_require_matching_batched_arrays() -> None:
    layout = managym.ObservationLayout(
        max_decision_records=2,
        max_decision_references=2,
        max_decision_reference_details=2,
        max_decision_members=2,
    )
    obs = layout.allocate(1)
    del obs["decision_records"]
    assert layout.axis_extents(obs) is None
    assert layout.axis_extents(obs, omit_decision_memory=True) is not None
    obs["extra"] = obs["facts"]
    assert layout.axis_extents(obs, omit_decision_memory=True) is None
    del obs["extra"]
    obs["decision_members"] = obs["decision_members"][:, :1]
    assert layout.axis_extents(obs, omit_decision_memory=True) is None
    assert layout.axis_extents(layout.allocate()) is None


def test_fact_kinds_are_named_native_row_kinds() -> None:
    facts = managym.observation_layout()["facts"]
    names = facts["row_kind_names"]
    assert set(names.values()) == set(facts["row_kinds"])
    assert observation.FACT_COUNTER == names["counter"] == 1
    assert observation.FACT_CONTINUATION_PROGRAM == names["continuation_program"]
    # The model's kind embedding is sized by the largest kind.
    assert observation.FACT_KINDS == max(names.values()) == 14
    assert observation.FACT_EFFECT not in names.values(), "reserved: shared-effect rows"


def test_encoder_names_warn_and_name_their_removal(
    board: tuple[managym.Env, managym.Observation],
) -> None:
    _, obs = board
    space = ObservationSpace()
    with pytest.warns(DeprecationWarning, match=r"ObservationSpace\.encode .*0\.4\.0"):
        encoded = space.encode(obs)
    expected = encode_observation(space, obs)
    assert encoded.keys() == expected.keys()
    for key, value in expected.items():
        np.testing.assert_array_equal(encoded[key], value, err_msg=key)
    with pytest.warns(DeprecationWarning, match=r"encoder is deprecated .*0\.4\.0"):
        assert space.encoder is space
    with pytest.warns(DeprecationWarning, match=r"ObservationEncoder .*0\.4\.0"):
        layout = observation.ObservationEncoder(ObservationConfig(max_actions=7))
    assert layout.config.max_actions == layout.allocate()["actions"].shape[0] == 7


@pytest.mark.parametrize(
    "history,input_version,memory",
    [
        (0, 9, False),
        (1, 9, False),
        (1, 10, False),
        (0, 11, False),
        (1, 11, False),
        (1, 11, True),
    ],
    ids=["raw", "v9", "v10", "v11-raw", "v11", "v11-memory"],
)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_space_shapes_are_the_native_layout_at_its_capacities(
    board: tuple[managym.Env, managym.Observation],
    history: Literal[0, 1],
    input_version: Literal[9, 10, 11],
    memory: bool,
) -> None:
    _, obs = board
    config = ObservationConfig(
        policy_input_version=input_version,
        policy_history_version=history,
        max_cards_per_player=64,
        max_permanents_per_player=44,
        max_events=7,
        **(
            dict(
                max_decision_records=8,
                max_decision_references=16,
                max_decision_reference_details=32,
                max_decision_members=16,
            )
            if memory
            else {}
        ),
    )
    space = ObservationSpace(config)
    layout = managym.observation_layout(
        policy_history=history == 1,
        turn_state=input_version >= 10,
        legacy=input_version < 11,
        decision_memory=memory,
    )
    encoded = encode_observation(space, obs)
    assert layout.keys() == encoded.keys() == space.shapes.keys()
    for key, array in encoded.items():
        metadata = layout[key]
        assert len(metadata["axes"]) == array.ndim
        assert np.dtype(metadata["dtype"]) == array.dtype == space.dtypes[key]
        for extent, actual in zip(metadata["shape"], array.shape, strict=True):
            assert extent is None or extent == actual, key
        columns = metadata["columns"]
        if columns:
            assert metadata["axes"][-1] == "feature"
            assert sorted(column["index"] for column in columns.values()) == list(
                range(array.shape[-1])
            )
        else:
            assert "element" in metadata and "feature" not in metadata["axes"]
    columns = layout["agent_permanents"]["columns"]
    assert columns["definition"]["kind"] == "id"
    assert columns["definition"]["encoding"] == "CardDefId + 1; 0 means absent"
    valid = encoded["agent_permanents_valid"].astype(bool)
    rows = encoded["agent_permanents"][valid]
    assert len(rows) == len(obs.agent_permanents) > 0
    for row, permanent in zip(rows, obs.agent_permanents, strict=True):
        assert row[columns["definition"]["index"]] == permanent.registry_key + 1
        assert row[columns["power"]["index"]] == pytest.approx(
            permanent.characteristics.power / 10
        )
        assert (
            row[columns["flying"]["index"]] == permanent.characteristics.keywords.flying
        )
        assert row[columns["valid"]["index"]] == 1


def test_named_layout_describes_tagged_rows_and_history() -> None:
    raw = managym.observation_layout()
    history = managym.observation_layout(policy_history=True)
    assert raw["agent_cards"]["columns"]["valid"]["index"] == 55
    assert raw["facts"]["shape"] == [None, 12]
    assert set(raw["facts"]["row_kinds"]) == set(range(1, 15)) - {6}
    assert raw["effect_details"]["shape"] == [4096, 3]
    assert set(raw["relations"]["row_kinds"]) == set(range(1, 11))
    assert raw["name_candidates"]["columns"]["group_last"]["index"] == 3
    assert raw["events"]["columns"]["source_id"]["kind"] == "id"
    assert history["events"]["columns"]["source_context"]["kind"] == "reference"
    assert history.keys() == raw.keys()
    assert raw["agent_permanents"]["columns"]["valid"]["index"] == 50
    assert raw["stack"]["columns"]["source"]["kind"] == "reference"
    legacy = managym.observation_layout(legacy=True)
    assert legacy["agent_permanents"]["columns"]["valid"]["index"] == 37
    assert raw.keys() - legacy.keys() == {"stack", "stack_valid", "known_library"}
    assert not raw["action_focus"]["columns"]
    assert raw["action_focus"]["element"]["kind"] == "reference"


def test_named_layout_names_v10_turn_state() -> None:
    v9 = managym.observation_layout(policy_history=True, turn_state=False, legacy=True)
    v10 = managym.observation_layout(policy_history=True, legacy=True)
    assert v10.keys() - v9.keys() == {"player_turn_state", "card_turn_state"}
    player = v10["player_turn_state"]
    assert player["shape"] == [2, 8]
    assert list(player["columns"]) == [
        "life_lost",
        "spells_cast",
        *(
            f"nonartifact_mana_{color}"
            for color in ("white", "blue", "black", "red", "green", "colorless")
        ),
    ]
    assert player["columns"]["life_lost"]["encoding"] == "Life lost this turn / 20"
    card = v10["card_turn_state"]
    assert card["shape"] == [2, None] and not card["columns"]
    assert card["element"]["kind"] == "flag"


def test_named_layout_reads_effect_programs_by_name() -> None:
    env = kellan_upgrade_root()
    encoded = encode_observation(
        ObservationSpace(), env.observation_for_player(env.current_agent_index())
    )
    details = managym.observation_layout()["effect_details"]
    kind = details["columns"]["kind"]["index"]
    assert (
        details["row_kind_names"]["program_integer"]
        in encoded["effect_details"][:, kind]
    )


def test_named_layout_discovery_is_torch_free() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('layout discovery imported Torch')
sys.meta_path.insert(0, NoTorch())
import managym
assert managym.observation_layout()['agent_permanents']['columns']['definition']['index'] == 24
assert 'torch' not in sys.modules
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "old,new,args",
    [
        ("learning_catalog", "catalog_tokens", ()),
        ("learning_token_value", "catalog_token_value", ("Ally",)),
        ("learning_record_names", "record_vocabulary", ()),
    ],
)
def test_deprecated_catalog_names_warn_and_delegate(
    old: str, new: str, args: tuple[str, ...]
) -> None:
    with pytest.warns(DeprecationWarning, match=rf"0\.3\.0.*managym\.{new}"):
        assert getattr(managym, old)(*args) == getattr(managym, new)(*args)


def test_world_version_warns_and_names_its_removal() -> None:
    assert "WORLD_VERSION" not in vars(managym)
    with pytest.warns(DeprecationWarning, match=r"WORLD_VERSION.*0\.4\.0"):
        assert managym.WORLD_VERSION == "w17"
    assert set(managym.world_identity()) == {"card_digest"}


def test_managym_layout_rejects_missing_and_extra_memory_before_writing(
    board: tuple[managym.Env, managym.Observation],
) -> None:
    env, obs = board
    plain = managym.ObservationLayout(max_events=7)
    memory = managym.ObservationLayout(
        max_events=7,
        max_decision_records=8,
        max_decision_references=16,
        max_decision_reference_details=32,
        max_decision_members=16,
    )
    for layout, other in ((plain, memory), (memory, plain)):
        scalar = other.allocate()
        batch = other.allocate(2)
        for arrays in (scalar, batch):
            for array in arrays.values():
                array.fill(17)
        with pytest.raises(ValueError, match="decision_"):
            managym.encode_observation_into(obs, layout, scalar)
        with pytest.raises(ValueError, match="decision_"):
            managym.encode_observations_into([env], layout, batch, 2)
        assert all(
            (array == 17).all()
            for arrays in (scalar, batch)
            for array in arrays.values()
        )


def test_public_history_round_trips_for_both_viewers() -> None:
    env = priority_root(("Mountain",), ())
    env.step(0)  # Play the only land. Its arrival has a public object reference.
    layout = managym.ObservationLayout(policy_history=True)
    source_definition = observation.column("events", "source_definition")
    for viewer in (0, 1):
        obs = env.observation_for_player(viewer)
        live, restored = layout.allocate(), layout.allocate()
        managym.encode_observation_into(obs, layout, live)
        text = obs.toJSON()
        managym.encode_observation_into(
            managym.Observation.from_json(text), layout, restored
        )
        assert live["events_valid"].sum() == 1
        for field in ("events", "events_valid"):
            np.testing.assert_array_equal(live[field], restored[field])
        historical = json.loads(text)
        semantic = json.loads(env.semantic_observation_json(viewer))
        assert "policy_history" not in semantic["viewer_state"]
        semantic["viewer_state"]["policy_history"] = historical["policy_history"]
        with pytest.raises(ValueError, match="noncanonical viewer observation"):
            managym.validate_observation_json(json.dumps(semantic))
        oversized = {**historical, "policy_history": historical["policy_history"] * 33}
        with pytest.raises(ValueError, match="public policy history exceeds"):
            managym.Observation.from_json(json.dumps(oversized))
        historical["policy_history"][0]["source"]["Card"]["definition"] = 2**32 - 1
        managym.encode_observation_into(
            managym.Observation.from_json(json.dumps(historical)), layout, restored
        )
        assert restored["events"][0, source_definition] == 2**32
        del historical["policy_history"]
        managym.encode_observation_into(
            managym.Observation.from_json(json.dumps(historical)), layout, restored
        )
        assert not restored["events_valid"].any()


def test_dense_vector_and_batch_encoders_emit_deprecation_warning() -> None:
    space = ObservationSpace()
    vector = managym.VectorEnv(1)
    with pytest.warns(
        DeprecationWarning, match=r"set_observation_layout.*0\.4\.0"
    ):
        vector.set_buffers(allocate_vector_buffers(space, 1), layout=space.layout)

    env = managym.Env(seed=7)
    env.reset(
        [managym.PlayerConfig(name, {"Mountain": 8}) for name in ("a", "b")]
    )
    with pytest.warns(
        DeprecationWarning, match=r"encode_observation_batch.*0\.4\.0"
    ):
        managym.encode_observations_into([env], space.layout, space.allocate(1), 1)


def test_layoutless_vector_registration_warns_and_preserves_memory_configuration() -> (
    None
):
    buffers = allocate_vector_buffers(ObservationSpace(), 1)
    plain = managym.VectorEnv(1)
    with pytest.warns(DeprecationWarning, match="0.4.0"):
        plain.set_buffers(buffers)
    with (
        pytest.warns(DeprecationWarning, match="0.4.0"),
        pytest.raises(ValueError, match="unexpected buffer"),
    ):
        plain.set_buffers(dict(buffers, unknown=buffers["rewards"]))
    memory = managym.VectorEnv(1, decision_memory=True)
    with (
        pytest.warns(DeprecationWarning, match="0.4.0"),
        pytest.raises(ValueError, match="decision memory"),
    ):
        memory.set_buffers(buffers)


def test_effect_detail_kinds_publish_the_catalog_token_the_encoder_writes() -> None:
    details = managym.observation_layout()["effect_details"]
    names = details["row_kind_names"]
    assert set(names.values()) == details["row_kinds"].keys()
    tokens = details["program_token_kinds"]
    assert tokens.keys() == {
        value for name, value in names.items() if name.startswith("program_")
    }
    assert tokens[names["program_integer"]] == 4
    assert tokens[names["program_boolean"]] == 5
