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
from manabot.env.observation import ObservationSpace
from manabot.env.shared_effects import SHARED_EFFECT_SHAPES
from manabot.infra.config import ObservationConfig
from manabot.verify.representation import kellan_upgrade_root
from manabot.verify.text_observation import priority_root
import managym
from tests.fixtures.observation_rows import observation_arrays


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
    for key, shape in SHARED_EFFECT_SHAPES.items():
        assert tuple(layout[key]["shape"]) == shape, key
    memory = managym.observation_layout(decision_memory=True)
    assert MEMORY_WIDTHS.keys() == {k for k in memory if k.startswith("decision_")}


def test_fact_kinds_are_named_native_row_kinds() -> None:
    facts = managym.observation_layout()["facts"]
    names = facts["row_kind_names"]
    assert set(names.values()) == set(facts["row_kinds"])
    assert observation.FACT_COUNTER == names["counter"] == 1
    assert observation.FACT_CONTINUATION_PROGRAM == names["continuation_program"]
    # The model's kind embedding is sized by the largest kind.
    assert observation.FACT_TRIGGERING_AMOUNT == names["triggering_amount"] == 16
    assert observation.FACT_KINDS == max(names.values())
    assert observation.FACT_EFFECT not in names.values(), "reserved: shared-effect rows"


@pytest.mark.parametrize(
    "history,memory",
    [
        (0, False),
        (1, False),
        (1, True),
    ],
    ids=["v11-raw", "v11", "v11-memory"],
)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_space_shapes_are_the_native_layout_at_its_capacities(
    board: tuple[managym.Env, managym.Observation],
    history: Literal[0, 1],
    memory: bool,
) -> None:
    _, obs = board
    config = ObservationConfig(
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
        decision_memory=memory,
    )
    encoded = observation_arrays(space, obs)
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
    assert set(raw["facts"]["row_kinds"]) == set(range(1, 17)) - {6}
    assert raw["effect_details"]["shape"] == [4096, 3]
    assert set(raw["relations"]["row_kinds"]) == set(range(1, 12))
    assert raw["name_candidates"]["columns"]["group_last"]["index"] == 3
    assert raw["events"]["columns"]["source_id"]["kind"] == "id"
    assert history["events"]["columns"]["source_context"]["kind"] == "reference"
    assert history.keys() == raw.keys()
    assert raw["agent_permanents"]["columns"]["valid"]["index"] == 50
    assert raw["stack"]["columns"]["source"]["kind"] == "reference"
    assert not raw["action_focus"]["columns"]
    assert raw["action_focus"]["element"]["kind"] == "reference"


def test_named_layout_reads_effect_programs_by_name() -> None:
    env = kellan_upgrade_root()
    encoded = observation_arrays(
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
    with pytest.warns(DeprecationWarning, match=r"WORLD_VERSION.*0\.3\.0"):
        assert managym.WORLD_VERSION == "w17"
    assert set(managym.world_identity()) == {"card_digest"}


def test_public_history_round_trips_for_both_viewers() -> None:
    env = priority_root(("Mountain",), ())
    env.step(0)  # Play the only land. Its arrival has a public object reference.
    space = ObservationSpace(ObservationConfig(policy_history_version=1))
    source_definition = observation.column("events", "source_definition")
    for viewer in (0, 1):
        obs = env.observation_for_player(viewer)
        live = observation_arrays(space, obs)
        text = obs.toJSON()
        restored = observation_arrays(space, managym.Observation.from_json(text))
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
        historical["policy_history"][0]["source"]["Card"]["definition"] = 2**31 - 2
        restored = observation_arrays(
            space, managym.Observation.from_json(json.dumps(historical))
        )
        assert restored["events"][0, source_definition] == 2**31 - 1
        # Batch words are int32: a definition id past that range is rejected.
        historical["policy_history"][0]["source"]["Card"]["definition"] = 2**32 - 1
        with pytest.raises(ValueError, match="cannot represent"):
            observation_arrays(
                space, managym.Observation.from_json(json.dumps(historical))
            )
        historical["policy_history"][0]["source"]["Card"]["definition"] = 0
        del historical["policy_history"]
        restored = observation_arrays(
            space, managym.Observation.from_json(json.dumps(historical))
        )
        assert not restored["events_valid"].any()


def test_batch_encoder_accepts_observations_and_envs() -> None:
    space = ObservationSpace()
    env = managym.Env(seed=7)
    obs, _ = env.reset(
        [managym.PlayerConfig(name, {"Mountain": 8}) for name in ("a", "b")]
    )
    # Standalone Observation items encode directly into an ObservationBatch.
    batch = managym.encode_observation_batch([obs, env], space.layout)
    assert batch.batch_size == 2
    assert batch.action_counts[0] == batch.action_counts[1]


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
