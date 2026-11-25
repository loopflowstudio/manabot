"""Policy input v10: the turn state v9 cannot hold.

V10 is v9 plus `player_turn_state` and `card_turn_state`, written by the
native encoder. A v9 encoder refuses an observation that needs them, naming
v10 and the state, before writing any buffer.
"""

from contextlib import closing
from typing import Iterator

import numpy as np
import pytest
import torch

from manabot.backends.torch.policy import ActorCritic
from manabot.env.failures import RecordedVectorEnv
from manabot.env.match import Match
from manabot.env.observation import (
    TURN_STATE_LIFE_LOST_INDEX,
    TURN_STATE_NONARTIFACT_MANA_INDEX,
    TURN_STATE_SPELLS_CAST_INDEX,
    ObservationSpace,
    allocate_vector_buffers,
    encode_observation,
)
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.verify.text_observation import priority_root
import managym
from tests.model.prepared_routes import device_input


def _config(version: int) -> ObservationConfig:
    """A deprecated v9/v10 configuration, which warns when built."""
    with pytest.warns(DeprecationWarning, match=f"policy input v{version}"):
        return ObservationConfig(policy_input_version=version)


def _lost_life(obs: managym.Observation, amount: int) -> managym.Observation:
    """`obs` with the opponent having lost `amount` life this turn."""
    opponent = obs.opponent
    history = opponent.turn_history
    history.life_lost = amount
    opponent.turn_history = history
    obs.opponent = opponent
    return obs


def test_v9_refuses_life_lost_only_once_a_visible_card_reads_it() -> None:
    space = ObservationSpace(_config(9))
    env = priority_root((), ("Mountain",))
    obs = _lost_life(env.scenario_refresh(), 3)
    assert managym.input_v10_state(obs) is None
    encode_observation(space, obs)

    env = priority_root((), ("Bloodchief Ascension",))
    obs = _lost_life(env.scenario_refresh(), 3)
    assert managym.input_v10_state(obs) == "life lost this turn"
    out = space.allocate()
    for value in out.values():
        value.fill(-7)
    with pytest.raises(
        ValueError, match="policy input v10 required: .*life lost this turn"
    ):
        managym.encode_observation_into(obs, space.layout, out)
    assert all(np.all(value == -7) for value in out.values())


def test_v10_turn_state_rows_hold_life_lost_spells_cast_and_left_battlefield() -> None:
    config = _config(10)
    env = priority_root(("Brought Back",), ("Bloodchief Ascension",))
    obs = _lost_life(env.scenario_refresh(), 3)
    # The encoder copies the card flag; game behavior is covered natively.
    cards = obs.agent_cards
    cards[0].from_battlefield_this_turn = True
    obs.agent_cards = cards
    agent = obs.agent
    agent.spells_cast = 2
    obs.agent = agent
    encoded = encode_observation(ObservationSpace(config), obs)
    players = encoded["player_turn_state"]
    assert players.shape == (2, 8)
    # Agent row first: no life lost, two spells cast.
    assert players[0, TURN_STATE_LIFE_LOST_INDEX] == 0.0
    assert players[0, TURN_STATE_SPELLS_CAST_INDEX] == pytest.approx(2 / 10)
    assert players[1, TURN_STATE_LIFE_LOST_INDEX] == pytest.approx(3 / 20)
    assert players[1, TURN_STATE_SPELLS_CAST_INDEX] == 0.0
    assert not players[:, TURN_STATE_NONARTIFACT_MANA_INDEX:].any()
    cards_state = encoded["card_turn_state"]
    assert cards_state.shape == (2, config.max_cards_per_player)
    assert cards_state[0, 0] == 1.0
    assert cards_state.sum() == 1.0


@pytest.fixture
def v10_observation() -> Iterator[
    tuple[ObservationSpace, ObservationConfig, dict[str, np.ndarray]]
]:
    config = _config(10)
    space = ObservationSpace(config)
    buffers = allocate_vector_buffers(space, 2)
    with closing(
        RecordedVectorEnv(num_envs=2, player_policies=("agent", "random"))
    ) as vector:
        vector.set_buffers(buffers, layout=space.layout)
        vector.reset_all_into_buffers(Match().to_rust())
        yield space, config, {key: buffers[key] for key in space.shapes}


def test_v10_vector_buffers_reach_the_model(
    v10_observation: tuple[ObservationSpace, ObservationConfig, dict[str, np.ndarray]],
) -> None:
    space, config, obs = v10_observation
    assert obs["player_turn_state"].shape == (2, 2, 8)
    assert obs["card_turn_state"].shape == (2, 2, config.max_cards_per_player)
    torch.manual_seed(0)
    agent = ActorCritic(space, ModelConfig(hidden_dim=4, num_attention_heads=2))
    with torch.no_grad():
        _, _, _, value = agent.get_action_and_value(device_input(agent, obs))
    assert value.shape == (2,)
