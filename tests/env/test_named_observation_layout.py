"""Outside consumers can read native observation arrays by column name."""

import subprocess
import sys
from typing import Literal

import numpy as np
import pytest

from manabot.env.observation import ObservationSpace
from manabot.infra.config import ObservationConfig
from manabot.verify.representation import kellan_upgrade_root
from manabot.verify.text_observation import priority_root
import managym


@pytest.fixture(scope="module")
def board() -> tuple[managym.Env, managym.Observation]:
    env = priority_root(("Lightning Bolt",), ("Mountain", "Shivan Dragon"))
    return env, env.observation_for_player(env.current_agent_index())


@pytest.mark.parametrize(
    "history,input_version", [(0, 9), (1, 9), (1, 10)], ids=["raw", "v9", "v10"]
)
def test_named_layout_matches_native_buffers_and_python_values(
    board: tuple[managym.Env, managym.Observation],
    history: Literal[0, 1],
    input_version: Literal[9, 10],
) -> None:
    env, obs = board
    config = ObservationConfig(
        policy_input_version=input_version,
        policy_history_version=history,
        max_cards_per_player=64,
        max_permanents_per_player=44,
        max_events=7,
    )
    space = ObservationSpace(config)
    layout = managym.observation_layout(
        policy_history=history == 1, turn_state=input_version == 10
    )
    native = space.encoder.allocate()
    env.encode_observation_into(obs, native)
    reference = space.encode(obs)
    assert layout.keys() == native.keys() == reference.keys() == space.shapes.keys()
    for key, array in native.items():
        metadata = layout[key]
        assert len(metadata["axes"]) == array.ndim
        assert len(metadata["shape"]) == array.ndim
        assert np.dtype(metadata["dtype"]) == array.dtype
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
        np.testing.assert_array_equal(array, reference[key], err_msg=key)
    if history == 0 and input_version == 9:
        scalar = env.encode_observation(obs)
        assert scalar.keys() == layout.keys()
        for key, array in scalar.items():
            columns = layout[key]["columns"]
            if columns:
                assert len(columns) == array.shape[-1]
    columns = layout["agent_permanents"]["columns"]
    assert columns["definition"]["index"] == 24
    assert columns["definition"]["kind"] == "id"
    assert columns["definition"]["encoding"] == "CardDefId + 1; 0 means absent"
    valid = native["agent_permanents_valid"].astype(bool)
    rows = native["agent_permanents"][valid]
    assert len(rows) == len(obs.agent_permanents) > 0
    for row, permanent in zip(rows, obs.agent_permanents, strict=True):
        assert row[columns["definition"]["index"]] == permanent.registry_key + 1
        assert row[columns["power"]["index"]] == pytest.approx(permanent.power / 10)
        assert row[columns["flying"]["index"]] == permanent.keywords.flying
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
    assert not raw["action_focus"]["columns"]
    assert raw["action_focus"]["element"]["kind"] == "reference"


def test_named_layout_names_v10_turn_state() -> None:
    v9 = managym.observation_layout(policy_history=True)
    v10 = managym.observation_layout(policy_history=True, turn_state=True)
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
    encoded = env.encode_observation(
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
