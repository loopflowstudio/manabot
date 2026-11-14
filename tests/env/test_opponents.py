"""Supported opponent policies and rejection at public API boundaries."""

import numpy as np
import pytest

from manabot.env.failures import CapturedVector, FailureEvidence, GameInputs
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.env.opponents import RandomPolicy, build_opponent_policy
from manabot.env.replay import replay_failure_inputs
from manabot.env.vector_env import VectorEnv
from manabot.infra.config import RewardConfig
import managym


def test_random_policy_picks_only_valid_actions() -> None:
    obs = {"actions_valid": np.array([1, 0, 1, 0], dtype=np.float32)}
    policy = RandomPolicy()
    chosen = {policy(obs) for _ in range(100)}
    assert chosen.issubset({0, 2})
    assert chosen


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
            CapturedVector(num_envs=1, player_policies=("agent", name))
        else:
            VectorEnv(
                1,
                Match(),
                ObservationSpace(),
                Reward(RewardConfig()),
                "cpu",
                player_policies=("agent", name),
            )


def test_retained_passive_evidence_is_readable_but_cannot_replay() -> None:
    evidence = FailureEvidence(
        mode="vector",
        operation="reset",
        games=[GameInputs(seed=42, players=[])],
        opponent_policy="passive",
        identities={},
        context={},
        seconds=0,
    )
    retained = evidence.model_dump_json(by_alias=True)
    restored = FailureEvidence.model_validate_json(retained)
    assert restored.opponent_policy == "passive"
    with pytest.raises(ValueError, match="unsupported opponent_policy: passive"):
        replay_failure_inputs(restored)
    assert restored.model_dump_json(by_alias=True) == retained
