"""Training contracts, credit boundaries, and experiment controls."""

from pathlib import Path

import numpy as np
import pytest
import torch

from manabot.backends.torch.rl import selected_rows
from manabot.env.observation import ObservationSpace
from manabot.policy.prepare import reference_distribution
from manabot.policy.spans import ObservationSelection, observation_spans
from manabot.training.learning import DampedPolicyLearning
from manabot.verify.text_observation import priority_root
import managym

ROOT = Path(__file__).resolve().parents[2]


def test_reference_has_full_support_and_is_permutation_invariant():
    obs = {
        "actions_valid": torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
        "actions": torch.tensor(
            [
                [
                    [1.0, *([0.0] * 15), 1.0, 1.0],
                    [1.0, *([0.0] * 15), 0.0, 1.0],
                    [0.0, 1.0, *([0.0] * 14), 0.0, 1.0],
                    [0.0] * 18,
                ]
            ]
        ),
    }
    obs = {key: value.numpy() for key, value in obs.items()}
    probs = reference_distribution(obs, "action_type_uniform")
    np.testing.assert_allclose(probs, [[0.25, 0.25, 0.5, 0.0]], rtol=1e-6)
    permutation = np.array([2, 0, 3, 1])
    moved = reference_distribution(
        {k: v[:, permutation] for k, v in obs.items()}, "action_type_uniform"
    )
    np.testing.assert_allclose(moved, probs[:, permutation], rtol=1e-6)
    assert selected_rows(torch.zeros(4), 0.25, 0.01).numel() == 0


@pytest.mark.parametrize("kind", ["uniform", "action_type_uniform"])
def test_reference_mass_on_native_offers_in_selected_order(kind: str) -> None:
    space = ObservationSpace()
    games = [
        priority_root(hand, ("Mountain",))
        for hand in (("Mountain", "Mountain", "Lightning Bolt"), ("Mountain",))
    ]
    raw = [game.observation_for_player(game.current_agent_index()) for game in games]
    batch = managym.encode_observation_batch(raw, space.layout)
    expected = np.zeros((len(raw), space.config.max_actions), np.float32)
    for row, obs in enumerate(raw):
        types = [int(offer.action_type) for offer in obs.action_space.actions]
        for slot, tag in enumerate(types):
            expected[row, slot] = 1 / (
                len(types) if kind == "uniform" else len(set(types)) * types.count(tag)
            )
    np.testing.assert_allclose(reference_distribution(batch, kind), expected)
    bucketed = observation_spans(batch, space, bucketed=True, objects=True)
    np.testing.assert_allclose(reference_distribution(bucketed, kind), expected)
    selected = ObservationSelection(space, ((batch, 1), (batch, 0), (batch, 1)))
    np.testing.assert_allclose(
        reference_distribution(selected, kind), expected[[1, 0, 1]]
    )


def test_legacy_learning_discriminator_is_rejected() -> None:
    with pytest.raises(ValueError):
        DampedPolicyLearning.model_validate({"gradient": "ataraxos_move"})
    rule = DampedPolicyLearning(gradient="damped_policy")
    assert rule.model_dump()["gradient"] == "damped_policy"
