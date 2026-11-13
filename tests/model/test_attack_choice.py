"""Attack/decline must remain learnable through the native policy input path."""

from pathlib import Path

import numpy as np
import pytest
import torch

from manabot.backends.torch.checkpoint import load_checkpoint
from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match
from manabot.env.observation import (
    ACTION_DECLARED_INDEX,
    ACTION_TYPE_DIM,
    ObservationSpace,
)
from manabot.infra.config import ModelConfig
from manabot.model.admission import admit_observation
from manabot.model.world import WORLD, checkpoint_world
from manabot.policy.prepare import reference_distribution
from manabot.verify.action_cases import (
    DecisionCase,
    decision_case,
    identity_removed,
    reorder_offers,
)


@pytest.fixture
def attack_case() -> DecisionCase:
    case = decision_case("attack")
    obs = case.observation
    torch.testing.assert_close(obs["action_focus"][0, 0], obs["action_focus"][0, 1])
    torch.testing.assert_close(
        obs["actions"][0, :2, ACTION_DECLARED_INDEX], torch.tensor([1.0, 0.0])
    )
    torch.testing.assert_close(
        obs["actions"][0, :2, :ACTION_TYPE_DIM].sum(-1), torch.ones(2)
    )
    return case


@pytest.mark.parametrize("preferred", [0, 1])
def test_native_attack_choice_can_learn_either_preference(
    attack_case: DecisionCase,
    preferred: int,
) -> None:
    attack_observation = attack_case.observation
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng():
            torch.manual_seed(17)
            agent = ActorCritic(
                ObservationSpace(),
                ModelConfig(hidden_dim=8, num_attention_heads=2, attention_on=False),
            )
        # Erasing the declaration reproduces the old, weight-independent tie.
        logits, _ = agent(admit_observation(agent, identity_removed(attack_case)))
        torch.testing.assert_close(logits[0, 0], logits[0, 1], rtol=0, atol=0)

        optimizer = torch.optim.Adam(agent.parameters(), lr=0.02)
        for _ in range(40):
            logits, _ = agent(admit_observation(agent, attack_observation))
            loss = -logits.log_softmax(-1)[0, preferred]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            logits, _ = agent(admit_observation(agent, attack_observation))
            assert float(logits.softmax(-1)[0, preferred]) > 0.95
            swapped, _ = agent(
                admit_observation(
                    agent, reorder_offers(attack_observation, torch.tensor([1, 0]))
                )
            )
            torch.testing.assert_close(swapped[:, :2], logits[:, [1, 0]])
        for kind in ("uniform", "action_type_uniform"):
            prior = reference_distribution(
                {k: v.numpy() for k, v in attack_observation.items()}, kind
            )
            np.testing.assert_allclose(prior[0, :2], [0.5, 0.5])
            assert not prior[0, 2:].any()
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("old_world,width", [("w4", 17), ("w5", 18), ("w6", 20)])
def test_old_checkpoint_is_rejected_before_loading_weights(
    tmp_path: Path, old_world: str, width: int
) -> None:
    space = ObservationSpace()
    binding = checkpoint_world(Match().to_rust(), space)
    assert binding["world"] == WORLD
    binding["world"] = old_world
    binding["input_schema"]["actions"]["shape"][-1] = width
    path = tmp_path / "w4.pt"
    torch.save(
        {
            "world_binding": binding,
            "hypers": {
                "agent_hypers": ModelConfig().model_dump(),
                "observation_hypers": space.encoder.config.model_dump(),
            },
            "model_state_dict": {},
        },
        path,
    )
    with pytest.raises(ValueError, match=f"compatible {WORLD} world binding"):
        load_checkpoint(str(path))
