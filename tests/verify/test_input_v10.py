"""Policy input v10: the turn state v9 cannot hold.

V10 is v9 plus `player_turn_state` and `card_turn_state`. The native and
Python encoders write them identically; a v9 encoder refuses an observation
that needs them, naming v10 and the state, before writing any buffer.
"""

from typing import Iterator

import numpy as np
import pytest
import torch

from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match, Reward
from manabot.env.observation import (
    TURN_STATE_DIM,
    TURN_STATE_LIFE_LOST_INDEX,
    TURN_STATE_SPELLS_CAST_INDEX,
    ObservationEncoder,
    ObservationSpace,
)
from manabot.env.vector_env import VectorEnv
from manabot.infra.config import (
    ModelConfig,
    ObservationConfig,
    RewardConfig,
    UnsupportedPolicyInput,
)
from manabot.verify.text_observation import priority_root
import managym

V9 = ObservationConfig()
V10 = ObservationConfig(policy_input_version=10)


def _lost_life(obs: managym.Observation, amount: int) -> managym.Observation:
    """`obs` with the opponent having lost `amount` life this turn."""
    opponent = obs.opponent
    history = opponent.turn_history
    history.life_lost = amount
    opponent.turn_history = history
    obs.opponent = opponent
    return obs


def _native(env: managym.Env, obs: managym.Observation, config: ObservationConfig):
    out = ObservationEncoder(config).allocate()
    env.encode_observation_into(obs, out)
    return out


def test_v9_refuses_life_lost_only_once_a_visible_card_reads_it() -> None:
    env = priority_root((), ("Mountain",))
    obs = _lost_life(env.scenario_refresh(), 3)
    assert managym.input_v10_state(obs) is None
    ObservationEncoder(V9).encode(obs)

    env = priority_root((), ("Bloodchief Ascension",))
    obs = _lost_life(env.scenario_refresh(), 3)
    assert managym.input_v10_state(obs) == "life lost this turn"
    with pytest.raises(UnsupportedPolicyInput, match="life lost this turn") as raised:
        ObservationEncoder(V9).encode(obs)
    assert raised.value.version == 10 and raised.value.state == "life lost this turn"
    with pytest.raises(ValueError, match="policy input v10 required"):
        _native(env, obs, V9)


def test_native_and_python_v10_encoders_agree() -> None:
    env = priority_root(("Brought Back",), ("Bloodchief Ascension",))
    obs = _lost_life(env.scenario_refresh(), 3)
    # The encoders copy the card flag; game behavior is covered natively.
    cards = obs.agent_cards
    cards[0].from_battlefield_this_turn = True
    obs.agent_cards = cards
    agent = obs.agent
    agent.spells_cast = 2
    obs.agent = agent
    python = ObservationEncoder(V10).encode(obs)
    native = _native(env, obs, V10)
    assert set(python) == set(native)
    for key, value in python.items():
        np.testing.assert_allclose(native[key], value, atol=1e-6, err_msg=key)
    assert python["player_turn_state"].shape == (2, TURN_STATE_DIM)
    assert python["player_turn_state"][1, TURN_STATE_LIFE_LOST_INDEX] == pytest.approx(
        3 / 20
    )
    assert python["player_turn_state"][0, TURN_STATE_SPELLS_CAST_INDEX] == (
        pytest.approx(2 / 10)
    )
    assert python["card_turn_state"][0, 0] == 1.0
    assert python["card_turn_state"].sum() == 1.0


@pytest.fixture
def v10_observation() -> Iterator[tuple[ObservationSpace, dict[str, torch.Tensor]]]:
    space = ObservationSpace(V10)
    env = VectorEnv(
        num_envs=2,
        match=Match(),
        observation_space=space,
        reward=Reward(RewardConfig(trivial=True)),
        device="cpu",
        player_policies=("agent", "random"),
    )
    obs, _ = env.reset()
    yield space, obs
    env.close()


def test_v10_vector_buffers_reach_the_model(
    v10_observation: tuple[ObservationSpace, dict[str, torch.Tensor]],
) -> None:
    space, obs = v10_observation
    assert obs["player_turn_state"].shape == (2, 2, TURN_STATE_DIM)
    assert obs["card_turn_state"].shape == (2, 2, V10.max_cards_per_player)
    torch.manual_seed(0)
    agent = ActorCritic(space, ModelConfig(hidden_dim=4, num_attention_heads=2))
    with torch.no_grad():
        _, _, _, value = agent.get_action_and_value(obs)
    assert value.shape == (2,)
