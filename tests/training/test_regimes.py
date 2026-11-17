"""Training contracts, credit boundaries, and experiment controls."""

from pathlib import Path

import numpy as np
import pytest
import torch

from manabot.backends.torch.rl import selected_rows
from manabot.policy.prepare import reference_distribution
from manabot.training.learning import DampedPolicyLearning

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


def test_legacy_learning_discriminator_is_rejected() -> None:
    with pytest.raises(ValueError):
        DampedPolicyLearning.model_validate({"gradient": "ataraxos_move"})
    rule = DampedPolicyLearning(gradient="damped_policy")
    assert rule.model_dump()["gradient"] == "damped_policy"
