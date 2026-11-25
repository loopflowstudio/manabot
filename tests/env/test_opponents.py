"""Supported opponent policies and rejection at public API boundaries."""

import pytest

from manabot.env.failures import RecordedVectorEnv
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.env.opponents import RandomPolicy, build_opponent_policy
from manabot.env.vector_env import VectorEnv
from manabot.infra.config import RewardConfig
import managym
from managym.recording import GameInputs, GameRecord
from managym.replay import replay_failure_inputs


def test_random_policy_picks_only_valid_actions() -> None:
    obs, _ = managym.Env(seed=3).reset(Match().to_rust())
    policy = RandomPolicy()
    for _ in range(100):
        assert 0 <= policy(obs) < len(obs.action_space.actions)


def test_build_opponent_policy_none() -> None:
    assert build_opponent_policy("none") is None


@pytest.mark.parametrize("name", ["passive", "unknown"])
@pytest.mark.parametrize("api", ["factory", "native", "captured", "torch"])
def test_unsupported_opponent_policy(name: str, api: str) -> None:
    with pytest.raises(ValueError, match=f"[Uu]nsupported .*policy:.*{name}"):
        if api == "factory":
            build_opponent_policy(name)
        elif api == "native":
            managym.VectorEnv(num_envs=1, player_policies=("agent", name))
        elif api == "captured":
            RecordedVectorEnv(num_envs=1, player_policies=("agent", name))
        else:
            VectorEnv(
                1,
                Match(),
                ObservationSpace(),
                Reward(RewardConfig()),
                "cpu",
                player_policies=("agent", name),
            )


@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_retained_passive_evidence_is_readable_but_cannot_replay() -> None:
    evidence = GameRecord(
        mode="vector",
        operation="reset",
        games=[GameInputs(seed=42, players=[])],
        opponent_policy="passive",
        identities={},
        context={},
        seconds=0,
    )
    retained = evidence.model_dump_json(by_alias=True)
    restored = GameRecord.model_validate_json(retained)
    assert restored.opponent_policy == "passive"
    with pytest.raises(ValueError, match="[Uu]nsupported player policy:.*passive"):
        replay_failure_inputs(restored)
    assert restored.model_dump_json(by_alias=True) == retained
