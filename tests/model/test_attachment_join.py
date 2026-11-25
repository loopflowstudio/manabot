"""Policy input v5's permanent links are joins between object embeddings."""

import pytest
import torch

from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import (
    PERMANENT_ATTACHED_INDEX,
    PERMANENT_LINK_INDEXES,
    ObservationConfig,
    ObservationSpace,
)
from manabot.infra.config import ModelConfig
from manabot.verify.choice_inputs import encode, reorder_objects
import managym
from tests.fixtures.observation_rows import observation_arrays
from tests.model.prepared_routes import device_input


def _attached_observation() -> tuple[ObservationSpace, managym.Observation, int]:
    env = managym.Env(seed=3, skip_trivial=False)
    env.reset(
        [
            managym.PlayerConfig("a", {"Mountain": 10, "Gray Ogre": 20}),
            managym.PlayerConfig("b", {"Forest": 10, "Llanowar Elves": 20}),
        ]
    )
    env.step(0)
    env.step(0)
    env.scenario_force_battlefield(0, "Gray Ogre")
    env.scenario_force_battlefield(0, "Gray Ogre")
    obs = env.scenario_refresh()
    first, second = obs.agent_permanents
    second.attached_to = first.id
    obs.agent_permanents = [first, second]
    space = ObservationSpace(ObservationConfig())
    host = 2 + 2 * space.config.max_cards_per_player
    return space, obs, host


def test_attachment_link_is_a_join_not_a_numeric_feature() -> None:
    space, raw, host = _attached_observation()
    encoded = observation_arrays(space, raw)
    assert encoded["agent_permanents"][1, PERMANENT_ATTACHED_INDEX] == host + 1
    first, second = raw.agent_permanents
    second.attached_to = None
    raw.agent_permanents = [first, second]
    detached = encode(raw, space)
    assert detached.project("agent_permanents")[0, 1, PERMANENT_ATTACHED_INDEX] == 0
    second.attached_to = first.id
    raw.agent_permanents = [first, second]
    attached = encode(raw, space)
    torch.manual_seed(5)
    agent = ActorCritic(space, ModelConfig(hidden_dim=16, num_attention_heads=2))
    assert len(agent.link_to_target) == len(agent.link_from_target) == 2
    with torch.no_grad():
        # Zero-initialized joins leave the initial function unchanged.
        same = [
            agent._gather_object_embeddings(device_input(agent, e)).objects
            for e in (attached, detached)
        ]
        torch.testing.assert_close(same[0], same[1])
        for layer in (*agent.link_to_target, *agent.link_from_target):
            layer.weight.normal_()
        linked, unlinked = (
            agent._gather_object_embeddings(device_input(agent, e)).objects
            for e in (attached, detached)
        )
    changed = (linked - unlinked).abs().sum(-1)[0]
    assert changed[host] > 0, "the host reads its attachment"
    assert changed[host + 1] > 0, "the attachment reads its host"
    assert changed[host + 2 :].sum() == 0 and changed[:host].sum() == 0


@pytest.mark.parametrize("column", PERMANENT_LINK_INDEXES)
def test_object_permutation_preserves_learned_permanent_links(column: int) -> None:
    space, raw, host = _attached_observation()
    first, second = raw.agent_permanents
    second.attached_to = None
    if column == PERMANENT_ATTACHED_INDEX:
        second.attached_to = first.id
    else:
        second.blocking = first.id
    raw.agent_permanents = [first, second]
    torch.manual_seed(5)
    agent = ActorCritic(space, ModelConfig(hidden_dim=16, num_attention_heads=2))
    agent.eval()
    with torch.no_grad():
        # Initialization zeros these joins, so an untrained model hides bad remaps.
        for layer in (*agent.link_to_target, *agent.link_from_target):
            layer.weight.normal_()
        original = encode(raw, space).batch.to_bytes()
        expected_logits, expected_value = agent(device_input(agent, raw))
        with reorder_objects(raw):
            permuted = encode(raw, space)
            assert permuted.project("agent_permanents")[0, 0, column] == host + 2
            logits, value = agent(device_input(agent, permuted.batch))
        torch.testing.assert_close(logits.softmax(-1), expected_logits.softmax(-1))
        torch.testing.assert_close(value, expected_value)
        assert encode(raw, space).batch.to_bytes() == original
