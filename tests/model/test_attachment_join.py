"""Policy input v5's permanent links are joins between object embeddings."""

import numpy as np
import torch

from manabot.env.observation import (
    PERMANENT_ATTACHED_INDEX,
    ObservationConfig,
    ObservationSpace,
)
from manabot.infra.config import ModelConfig
from manabot.model.actor_critic import ActorCritic
import managym


def _attached_observation() -> tuple[ObservationSpace, dict[str, np.ndarray], int]:
    env = managym.Env(seed=3, skip_trivial=False)
    env.reset(
        [
            managym.PlayerConfig("a", {"Mountain": 10, "Gray Ogre": 20}),
            managym.PlayerConfig("b", {"Forest": 10, "Llanowar Elves": 20}),
        ]
    )
    env.scenario_force_battlefield(0, "Gray Ogre")
    env.scenario_force_battlefield(0, "Gray Ogre")
    obs = env.scenario_refresh()
    first, second = obs.agent_permanents
    second.attached_to = first.id
    obs.agent_permanents = [first, second]
    space = ObservationSpace(ObservationConfig(policy_input_version=5))
    host = 2 + 2 * space.encoder.cards_per_player
    return space, space.encode(obs), host


def _batch(encoded: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value)[None] for key, value in encoded.items()}


def test_attachment_link_is_a_join_not_a_numeric_feature() -> None:
    space, encoded, host = _attached_observation()
    assert encoded["agent_permanents"][1, PERMANENT_ATTACHED_INDEX] == host + 1
    torch.manual_seed(5)
    agent = ActorCritic(space, ModelConfig(hidden_dim=16, num_attention_heads=2))
    assert len(agent.link_to_target) == len(agent.link_from_target) == 2
    detached = dict(encoded)
    detached["agent_permanents"] = encoded["agent_permanents"].copy()
    detached["agent_permanents"][1, PERMANENT_ATTACHED_INDEX] = 0
    with torch.no_grad():
        # Zero-initialized joins leave the initial function unchanged.
        same = [
            agent._gather_object_embeddings(_batch(e))[0] for e in (encoded, detached)
        ]
        torch.testing.assert_close(same[0], same[1])
        for layer in (*agent.link_to_target, *agent.link_from_target):
            layer.weight.normal_()
        linked, unlinked = (
            agent._gather_object_embeddings(_batch(e))[0] for e in (encoded, detached)
        )
    changed = (linked - unlinked).abs().sum(-1)[0]
    assert changed[host] > 0, "the host reads its attachment"
    assert changed[host + 1] > 0, "the attachment reads its host"
    assert changed[host + 2 :].sum() == 0 and changed[:host].sum() == 0


def test_permanent_link_joins_exist_only_for_v5() -> None:
    torch.manual_seed(5)
    v4 = ActorCritic(
        ObservationSpace(ObservationConfig(policy_input_version=4)),
        ModelConfig(hidden_dim=16, num_attention_heads=2),
    )
    assert not any("link_" in name for name in v4.state_dict())
