"""Outside consumers can interpret native arrays by name without changing saved inputs."""

import subprocess
import sys
from typing import Literal

import numpy as np
import pytest

from manabot import managym
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ObservationConfig
from manabot.verify.representation import kellan_upgrade_root
from manabot.verify.text_observation import priority_root


@pytest.fixture(scope="module")
def board() -> tuple[managym.Env, managym.Observation]:
    env = priority_root(("Lightning Bolt",), ("Mountain", "Shivan Dragon"))
    return env, env.observation_for_player(env.current_agent_index())


@pytest.mark.parametrize("version", range(1, 10))
@pytest.mark.parametrize("history", [0, 1])
def test_named_layout_matches_native_buffers_and_python_values(
    board: tuple[managym.Env, managym.Observation],
    version: int,
    history: Literal[0, 1],
) -> None:
    env, obs = board
    config = ObservationConfig(
        policy_input_version=version,
        policy_history_version=history,
        max_cards_per_player=64,
        max_permanents_per_player=44,
        max_events=7,
    )
    space = ObservationSpace(config)
    layout = managym.observation_layout(version, history_version=history)
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
    if history == 0:
        scalar = env.encode_observation(obs, version)
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


def test_named_layout_distinguishes_tagged_rows_and_history() -> None:
    v4 = managym.observation_layout(4)
    v5 = managym.observation_layout(5)
    v7 = managym.observation_layout(7)
    v9 = managym.observation_layout(9, history_version=1)
    assert "permission" not in v4["agent_cards"]["columns"]
    assert v4["agent_cards"]["columns"]["valid"]["index"] == 40
    assert v5["agent_cards"]["columns"]["valid"]["index"] == 55
    assert v5["facts"]["shape"] == [128, 12]
    assert v7["facts"]["shape"] == [None, 12]
    assert 6 in v5["facts"]["row_kinds"] and 6 not in v7["facts"]["row_kinds"]
    assert 14 not in v7["facts"]["row_kinds"] and 14 in v9["facts"]["row_kinds"]
    assert v7["effect_details"]["shape"] == [512, 3]
    assert v9["effect_details"]["shape"] == [4096, 3]
    assert 10 not in v7["relations"]["row_kinds"] and 10 in v9["relations"]["row_kinds"]
    assert v9["name_candidates"]["columns"]["group_last"]["index"] == 3
    assert v5["events"]["columns"]["source_id"]["kind"] == "id"
    assert v9["events"]["columns"]["source_context"]["kind"] == "reference"
    assert not v9["action_focus"]["columns"]
    assert v9["action_focus"]["element"]["kind"] == "reference"


@pytest.mark.parametrize("version", [-1, 0, 10, 256])
def test_named_layout_rejects_unsupported_input(version: int) -> None:
    with pytest.raises(ValueError, match="encoding version"):
        managym.observation_layout(version)


@pytest.mark.parametrize("history", [-1, 2, 256])
def test_named_layout_rejects_unsupported_history(history: int) -> None:
    with pytest.raises(ValueError, match="history version"):
        managym.observation_layout(9, history_version=history)


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
from manabot import managym
assert managym.recommended_encoding_version() == 9
assert managym.observation_layout(9)['agent_permanents']['columns']['definition']['index'] == 24
assert 'torch' not in sys.modules
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_recommended_input_admits_payload_and_keeps_defaults(
    board: tuple[managym.Env, managym.Observation],
) -> None:
    version = managym.recommended_encoding_version()
    assert version == 9
    assert ObservationConfig().policy_input_version == 6
    assert ObservationConfig.from_checkpoint({}).policy_input_version == 1
    env, obs = board
    implicit = env.encode_observation(obs)
    explicit = env.encode_observation(obs, 6)
    assert implicit.keys() == explicit.keys()
    for key in implicit:
        np.testing.assert_array_equal(implicit[key], explicit[key])
    env = kellan_upgrade_root()
    obs = env.observation_for_player(env.current_agent_index())
    for old in range(1, 8):
        with pytest.raises(ValueError, match=rf"v{old}.*select v8.*recommended.*v9"):
            env.encode_observation(obs, old)
    accepted = env.encode_observation(obs, version)
    assert (accepted["effect_details"][:, 0] >= 11).any()
    assert "name_candidates" in accepted


def test_latest_encoding_version_bounds_layouts() -> None:
    latest = managym.latest_encoding_version()
    assert managym.recommended_encoding_version() <= latest
    assert managym.observation_layout(latest)
    with pytest.raises(ValueError, match="encoding version"):
        managym.observation_layout(latest + 1)


def test_deprecated_engine_names_warn_and_delegate(
    board: tuple[managym.Env, managym.Observation],
) -> None:
    env, obs = board
    with pytest.warns(DeprecationWarning, match="encoding_version"):
        legacy = env.encode_observation(obs, policy_input_version=5)
    current = env.encode_observation(obs, encoding_version=5)
    assert legacy.keys() == current.keys()
    for key in current:
        np.testing.assert_array_equal(legacy[key], current[key])
    with pytest.raises(TypeError, match="deprecated alias"):
        env.encode_observation(obs, 5, policy_input_version=5)

    aliases = {
        "learning_catalog": managym.catalog_tokens,
        "learning_token_value": managym.catalog_token_value,
        "learning_record_names": managym.record_vocabulary,
    }
    for old, new in aliases.items():
        args = ("Ally",) if old == "learning_token_value" else ()
        with pytest.warns(DeprecationWarning, match=new.__name__):
            assert getattr(managym, old)(*args) == new(*args)
