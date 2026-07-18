from __future__ import annotations

import numpy as np
import pytest
import torch

from manabot.arena.guidance import PolicyPriorRandomLeafEvaluator
from manabot.sim.mcts import LeafEvaluation


class _PolicyOnlyAgent(torch.nn.Module):
    def __init__(self, logits: list[float], value: float):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.logits = torch.tensor(logits, dtype=torch.float32)
        self.value = value

    def forward(
        self, observation: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = next(iter(observation.values())).shape[0]
        logits = self.logits.unsqueeze(0).repeat(batch_size, 1) + self.anchor * 0
        values = torch.full((batch_size,), self.value) + self.anchor * 0
        return logits, values


class _UnusedObservationSpace:
    def encode(self, observation: object) -> dict[str, np.ndarray]:
        raise AssertionError(
            f"mapping observation should not be encoded: {observation}"
        )


class _ThreeActionEnv:
    @staticmethod
    def action_count() -> int:
        return 3


class _FixedRandomLeaf:
    def __init__(self, value: float):
        self.value = value

    def evaluate(self, *args: object, **kwargs: object) -> LeafEvaluation:
        return LeafEvaluation(
            priors=np.array([1.0]), root_value=self.value, cap_hit=True
        )


def test_policy_prior_evaluator_discards_checkpoint_value() -> None:
    observation = {"features": np.array([1.0], dtype=np.float32)}
    low_value = PolicyPriorRandomLeafEvaluator(
        _PolicyOnlyAgent([0.0, 1.0, -1.0], -100.0), _UnusedObservationSpace()
    )
    high_value = PolicyPriorRandomLeafEvaluator(
        _PolicyOnlyAgent([0.0, 1.0, -1.0], 100.0), _UnusedObservationSpace()
    )
    low_value.random_leaf = _FixedRandomLeaf(0.375)
    high_value.random_leaf = _FixedRandomLeaf(0.375)

    low = low_value.evaluate(
        _ThreeActionEnv(),
        observation,
        root_player=0,
        node_player=0,
        seed=7,
        max_steps=10,
    )
    high = high_value.evaluate(
        _ThreeActionEnv(),
        observation,
        root_player=0,
        node_player=1,
        seed=99,
        max_steps=10,
    )

    np.testing.assert_array_equal(low.priors, high.priors)
    assert low.root_value == high.root_value == 0.375
    assert low.cap_hit and high.cap_hit


def test_policy_prior_evaluator_normalizes_only_legal_logits() -> None:
    evaluator = PolicyPriorRandomLeafEvaluator(
        _PolicyOnlyAgent([0.0, 1.0, 50.0], 0.0), _UnusedObservationSpace()
    )
    priors = evaluator.root_priors(
        {"features": np.array([1.0], dtype=np.float32)}, action_count=2
    )

    np.testing.assert_allclose(priors, np.array([0.26894143, 0.7310586]))
    np.testing.assert_array_equal(evaluator.last_root_priors, priors)


def test_policy_prior_evaluator_rejects_nonfinite_legal_logits() -> None:
    evaluator = PolicyPriorRandomLeafEvaluator(
        _PolicyOnlyAgent([0.0, float("nan")], 0.0), _UnusedObservationSpace()
    )
    with pytest.raises(RuntimeError, match="nonfinite"):
        evaluator.root_priors(
            {"features": np.array([1.0], dtype=np.float32)}, action_count=2
        )
