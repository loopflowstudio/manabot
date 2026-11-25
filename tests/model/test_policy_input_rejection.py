"""Checkpoints and replays of another policy input are rejected, never loaded.

This revision encodes, binds and models policy inputs v9 (the fresh default)
and v10. A saved configuration naming another version, or omitting it (the
input-v1 contract), names the revision that can run it.
"""

from pathlib import Path
from typing import Any

from pydantic import ValidationError
import pytest

from manabot.backends.torch.checkpoint import (
    BACKEND as TORCH_BACKEND,
    load_checkpoint,
    save_checkpoint,
)
from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
from manabot.infra.config import (
    LAST_MULTI_INPUT_REVISION,
    POLICY_INPUT_VERSION,
    MatchConfig,
    ModelConfig,
    ObservationConfig,
    UnsupportedPolicyInput,
)
from manabot.model.catalog import inspect_checkpoint
from manabot.sim.replay import replay_environment
from manabot.sim.table_bots import CheckpointVillain
from tests.model.checkpoint_fixture import read_checkpoint, write_checkpoint

SETUP = "ur-lessons-vs-gw-allies"


def _match() -> Match:
    return Match(MatchConfig.authored(SETUP, "ur_lessons", "gw_allies"))


def _saved(path: Path, version: int | None) -> dict[str, Any]:
    """A current checkpoint whose saved observation names `version` (None: omits it)."""
    space = ObservationSpace()
    agent = ActorCritic(
        space, ModelConfig(hidden_dim=8, num_attention_heads=2, world=SETUP)
    )
    save_checkpoint(agent, space, path, player_configs=_match().to_rust())
    payload = read_checkpoint(path, map_location="cpu", weights_only=False)
    saved = payload["hypers"]["observation_hypers"]
    assert saved["policy_input_version"] == POLICY_INPUT_VERSION
    if version is None:
        del saved["policy_input_version"]
    else:
        saved["policy_input_version"] = version
    write_checkpoint(payload, path)
    return payload


def _assert_names_the_recording_revision(message: str, version: int) -> None:
    assert f"written for policy input v{version}" in message
    assert "run from the revision that recorded it" in message
    assert LAST_MULTI_INPUT_REVISION == "ac601834d"
    assert (
        f"{LAST_MULTI_INPUT_REVISION} is the last revision supporting v1-v8" in message
    )


def test_v11_is_the_fresh_default_and_every_input_is_serialized() -> None:
    config = ObservationConfig()
    assert config.policy_input_version == POLICY_INPUT_VERSION == 11
    dumped = config.model_dump()
    assert dumped["policy_input_version"] == 11 and dumped["max_facts"] == 512
    assert ObservationConfig.from_checkpoint(dumped) == config
    for version in (9, 10):
        # The deprecated v9/v10 rows still load, and warn.
        with pytest.warns(DeprecationWarning, match="no earlier than 0.3.0") as caught:
            legacy = ObservationConfig(policy_input_version=version)
        assert caught[0].filename == __file__  # The caller, not pydantic.
        assert legacy.model_dump()["policy_input_version"] == version
        with pytest.warns(DeprecationWarning):
            assert ObservationConfig.from_checkpoint(legacy.model_dump()) == legacy


@pytest.mark.parametrize("version", [1, 5, 6, 7, 8, 12])
def test_saved_configuration_of_another_input_is_rejected(version: int) -> None:
    payload = {**ObservationConfig().model_dump(), "policy_input_version": version}
    with pytest.raises(UnsupportedPolicyInput) as raised:
        ObservationConfig.from_checkpoint(payload)
    assert raised.value.version == version
    _assert_names_the_recording_revision(str(raised.value), version)
    # Direct construction (a regime file) reports the same reason.
    with pytest.raises(ValidationError, match="run from the revision that recorded it"):
        ObservationConfig(**payload)


def test_saved_configuration_without_a_version_is_input_v1() -> None:
    payload = ObservationConfig().model_dump()
    del payload["policy_input_version"]
    with pytest.raises(UnsupportedPolicyInput) as raised:
        ObservationConfig.from_checkpoint(payload)
    _assert_names_the_recording_revision(str(raised.value), 1)
    with pytest.raises(UnsupportedPolicyInput):
        ObservationConfig.from_checkpoint({})


@pytest.mark.parametrize("version", [7, None])
def test_checkpoint_of_another_input_is_rejected_before_loading(
    tmp_path: Path, version: int | None
) -> None:
    path = tmp_path / "policy.pt"
    _saved(path, version)
    recorded = 1 if version is None else version
    with pytest.raises(UnsupportedPolicyInput) as raised:
        load_checkpoint(str(path))
    _assert_names_the_recording_revision(str(raised.value), recorded)

    # Academy lists the checkpoint with the same reason and refuses to seat it.
    summary = inspect_checkpoint(path, backend=TORCH_BACKEND)
    assert not summary.compatible and summary.incompatibility is not None
    _assert_names_the_recording_revision(summary.incompatibility, recorded)
    with pytest.raises(UnsupportedPolicyInput):
        CheckpointVillain(str(path), player_configs=_match().to_rust())


def test_replay_of_another_input_is_rejected() -> None:
    game = {
        "seat_players": ["a", "b"],
        "deal_seed": 1,
        "observation_hypers": {
            **ObservationConfig().model_dump(),
            "policy_input_version": 6,
        },
    }
    with pytest.raises(UnsupportedPolicyInput) as raised:
        replay_environment(game)
    _assert_names_the_recording_revision(str(raised.value), 6)
    del game["observation_hypers"]
    with pytest.raises(UnsupportedPolicyInput):
        replay_environment(game)
