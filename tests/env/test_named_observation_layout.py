"""Native observation arrays by column name, and the Python mirror of them.

Outside consumers read columns from `managym.observation_layout()`; manabot's
column constants and space shapes are held to that layout here.
"""

import subprocess
import sys
from typing import Literal

import numpy as np
import pytest

from manabot.env import observation
from manabot.env.decision_memory import MEMORY_WIDTHS
from manabot.env.legacy_input import LEGACY_PERMANENT_DIM
from manabot.env.observation import ObservationSpace
from manabot.env.shared_effects import SHARED_EFFECT_SHAPES
from manabot.infra.config import ObservationConfig
from manabot.verify.representation import kellan_upgrade_root
from manabot.verify.text_observation import priority_root
import managym


@pytest.fixture(scope="module")
def board() -> tuple[managym.Env, managym.Observation]:
    env = priority_root(("Lightning Bolt",), ("Mountain", "Shivan Dragon"))
    return env, env.observation_for_player(env.current_agent_index())


# Every column constant manabot reads, as (constant, array, native column).
# A moved or renamed native column fails here instead of being read silently
# from the wrong place by the model.
_MIRRORED_COLUMNS = [
    # The action family one-hot fills the columns before `declared`.
    ("ACTION_TYPE_DIM", "actions", "declared"),
    ("ACTION_DECLARED_INDEX", "actions", "declared"),
    ("ACTION_MODE_INDEX", "actions", "mode"),
    ("ACTION_ABILITY_INDEX", "actions", "ability"),
    ("ACTION_CONTINUATION_INDEX", "actions", "continuation"),
    ("PLAYER_RESTRICTED_INDEX", "agent_player", "restricted_lesson_white"),
    ("PLAYER_NO_MAXIMUM_HAND_SIZE_INDEX", "agent_player", "no_maximum_hand_size"),
    ("CARD_PERMISSION_INDEX", "agent_cards", "permission"),
    ("PERMANENT_DEFINITION_INDEX", "agent_permanents", "definition"),
    ("PERMANENT_ONCE_EACH_TURN_INDEX", "agent_permanents", "once_each_turn_used"),
    ("PERMANENT_ATTACHED_INDEX", "agent_permanents", "attached_to"),
    ("PERMANENT_BLOCKING_INDEX", "agent_permanents", "blocking"),
    ("PERMANENT_TIMESTAMP_INDEX", "agent_permanents", "timestamp"),
    ("PERMANENT_MANA_VALUE_INDEX", "agent_permanents", "mana_value"),
    ("PERMANENT_TYPES_INDEX", "agent_permanents", "is_land"),
    ("STACK_IS_MINE_INDEX", "stack", "is_mine"),
    ("STACK_DEFINITION_INDEX", "stack", "definition"),
    ("STACK_SOURCE_INDEX", "stack", "source"),
    ("RELATION_VALID_INDEX", "relations", "valid"),
    ("RELATION_ELIGIBILITY_INDEX", "relations", "legal_at_resolution_start"),
    ("RELATION_X_INDEX", "relations", "x"),
    ("RELATION_THIS_TURN_INDEX", "relations", "this_turn"),
    ("FACT_OBJECT_INDEX", "facts", "object"),
    ("FACT_KIND_INDEX", "facts", "kind"),
    ("FACT_ID_INDEX", "facts", "id_high"),
    ("FACT_DETAIL_INDEX", "facts", "detail_high"),
    ("FACT_VALUE_INDEX", "facts", "value"),
    ("FACT_DURATION_INDEX", "facts", "duration"),
    ("FACT_ORDER_INDEX", "facts", "timestamp"),
    ("FACT_SOURCE_INDEX", "facts", "source_definition"),
    ("FACT_VALID_INDEX", "facts", "valid"),
    ("TURN_STATE_LIFE_LOST_INDEX", "player_turn_state", "life_lost"),
    ("TURN_STATE_SPELLS_CAST_INDEX", "player_turn_state", "spells_cast"),
    (
        "TURN_STATE_NONARTIFACT_MANA_INDEX",
        "player_turn_state",
        "nonartifact_mana_white",
    ),
]
# Widths, as (constant, array).
_MIRRORED_WIDTHS = [
    ("NAME_DIM", "name_candidates"),
    ("PLAYER_DIM", "agent_player"),
    ("CARD_DIM", "agent_cards"),
    ("PERMANENT_DIM", "agent_permanents"),
    ("STACK_DIM", "stack"),
    ("RELATION_DIM", "relations"),
    ("FACT_DIM", "facts"),
    ("DECKLIST_DIM", "decklists"),
    ("TURN_STATE_DIM", "player_turn_state"),
]


@pytest.mark.parametrize("constant,array,column", _MIRRORED_COLUMNS)
def test_python_column_constants_name_the_native_columns(
    constant: str, array: str, column: str
) -> None:
    layout = managym.observation_layout()
    assert getattr(observation, constant) == layout[array]["columns"][column]["index"]


@pytest.mark.parametrize("constant,array", _MIRRORED_WIDTHS)
def test_python_widths_are_the_native_widths(constant: str, array: str) -> None:
    layout = managym.observation_layout()[array]
    width = len(layout["columns"])
    assert getattr(observation, constant) == width == (layout["shape"][-1] or width)


def test_every_column_constant_is_mirrored() -> None:
    listed = {name for name, *_ in (*_MIRRORED_COLUMNS, *_MIRRORED_WIDTHS)}
    # Imported into the module; held by test_legacy_and_shared_widths_are_native.
    listed.add("LEGACY_PERMANENT_DIM")
    named = {name for name in vars(observation) if name.endswith(("_INDEX", "_DIM"))}
    assert named == listed


def test_python_fact_kinds_are_native_row_kinds() -> None:
    kinds = set(managym.observation_layout()["facts"]["row_kinds"])
    assert observation.FACT_KINDS == max(kinds)
    assert observation.FACT_EFFECT not in kinds, "reserved: shared-effect rows"
    for name in dir(observation):
        if name.startswith("FACT_") and name not in (
            *(constant for constant, *_ in _MIRRORED_COLUMNS),
            "FACT_DIM",
            "FACT_KINDS",
            "FACT_EFFECT",
            "FACT_TIMESTAMP_SCALE",
        ):
            assert getattr(observation, name) in kinds, name


def test_legacy_and_shared_widths_are_native() -> None:
    legacy = managym.observation_layout(legacy=True)
    assert len(legacy["agent_permanents"]["columns"]) == LEGACY_PERMANENT_DIM
    raw = managym.observation_layout()
    assert raw["relations"]["shape"][0] == observation.RELATION_ROWS
    for key, shape in SHARED_EFFECT_SHAPES.items():
        assert tuple(raw[key]["shape"]) == shape, key
    memory = managym.observation_layout(decision_memory=True)
    for key, width in MEMORY_WIDTHS.items():
        assert memory[key]["shape"][-1] == width, key


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
    encoded = space.encode(obs)
    assert layout.keys() == encoded.keys() == space.shapes.keys()
    for key, array in encoded.items():
        metadata = layout[key]
        assert len(metadata["axes"]) == array.ndim
        assert np.dtype(metadata["dtype"]) == array.dtype == space.encoder.dtypes[key]
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
    encoded = ObservationSpace().encode(
        env.observation_for_player(env.current_agent_index())
    )
    details = managym.observation_layout()["effect_details"]
    kind = details["columns"]["kind"]["index"]
    assert (encoded["effect_details"][:, kind] >= 11).any()


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
