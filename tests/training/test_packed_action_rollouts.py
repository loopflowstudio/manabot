"""Complete behavior evidence survives compact storage and minibatch replay."""

import numpy as np
import pytest
import torch

from manabot.model.policy_distribution import (
    NumericalError,
    clipped_surrogate,
    policy_logs,
    reverse_kl,
)
from manabot.training.learning.damped_policy import damped_policy_loss
from manabot.training.rollout.actions import PackedActions


def evidence() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(243)
    valid = np.zeros((4, 928), dtype=np.float32)
    for row, count in enumerate((928, 21, 4, 4)):
        valid[row, rng.choice(928, count, replace=False)] = 1
    observation = {
        "actions_valid": valid,
        "actions": rng.normal(size=(4, 928, 18)).astype(np.float32),
        "action_focus": rng.integers(-1, 5, size=(4, 928, 3)).astype(np.float32),
        "name_candidates": np.zeros((4, 928, 2), dtype=np.float32),
    }
    observation["name_candidates"][0, :, 0] = np.arange(1, 929)
    for key, value in observation.items():
        if key != "actions_valid":
            value[valid == 0] = -1 if key == "action_focus" else 0
    logits = torch.from_numpy(rng.normal(size=valid.shape).astype(np.float32))
    # Retain underflowed legal probabilities without replacing finite logs.
    logits[0, 927] = -150
    logs = policy_logs(logits, torch.from_numpy(valid).bool()).numpy()
    actions = valid.argmax(-1).astype(np.int64)
    return observation, logs, np.exp(logs), actions


def test_packed_actions_preserve_sparse_coordinates_distributions_and_copy_ownership() -> (
    None
):
    obs, logs, probabilities, actions = evidence()
    packed = PackedActions.pack(obs, probabilities, logs)
    selected = logs[np.arange(4), actions]
    packed.check(actions, selected)
    order = np.array([2, 0, 3, 1])
    restored = packed.unpack(order)
    for key, value in obs.items():
        np.testing.assert_array_equal(restored.observation[key], value[order])
    np.testing.assert_array_equal(restored.log_probabilities, logs[order])
    np.testing.assert_array_equal(restored.probabilities, probabilities[order])
    assert np.isfinite(packed.log_probabilities).all()
    assert (packed.probabilities == 0).any()
    assert (
        packed.nbytes
        < sum(v.nbytes for v in obs.values()) + logs.nbytes + probabilities.nbytes
    )
    obs["actions"].fill(0)
    np.testing.assert_array_equal(
        packed.unpack(order).observation["actions"], restored.observation["actions"]
    )
    joined = PackedActions.concatenate(
        [
            PackedActions.pack(
                {k: v[i : i + 1] for k, v in restored.observation.items()},
                restored.probabilities[i : i + 1],
                restored.log_probabilities[i : i + 1],
            )
            for i in range(4)
        ]
    )
    np.testing.assert_array_equal(
        joined.slots,
        np.concatenate(
            [packed.slots[packed.offsets[i] : packed.offsets[i + 1]] for i in order]
        ),
    )


@pytest.mark.parametrize("rule", ["ppo", "damped"])
def test_packed_action_learning_loss_and_gradients_match_dense_evidence(
    rule: str,
) -> None:
    obs, logs, probabilities, actions = evidence()
    packed = PackedActions.pack(obs, probabilities, logs)
    indexes = np.array([3, 0])
    restored = packed.unpack(indexes)
    gradients = []
    losses = []
    for old, valid in (
        (logs[indexes], obs["actions_valid"][indexes]),
        (restored.log_probabilities, restored.observation["actions_valid"]),
    ):
        current = torch.tensor(np.where(valid, old, 0) + 0.01, requires_grad=True)
        mask = torch.tensor(valid).bool()
        behavior = torch.tensor(old)
        selected = torch.tensor(actions[indexes])
        advantages = torch.tensor([-0.2, 0.7])
        reference = mask.float() / mask.sum(-1, keepdim=True)
        if rule == "damped":
            loss = damped_policy_loss(
                current,
                selected,
                advantages,
                behavior,
                reference,
                mask,
                clip=0.2,
                collection_kl=0.1,
                tau=0.05,
            ).mean()
        else:
            policy = policy_logs(current, mask)
            ratio = (policy - behavior).gather(-1, selected[:, None]).squeeze(-1)
            loss = (
                clipped_surrogate(ratio, advantages, 0.2)
                + 0.1 * reverse_kl(policy, behavior, mask)
                + 0.05 * reverse_kl(policy, reference.masked_fill(~mask, 1).log(), mask)
            ).mean()
        loss.backward()
        assert current.grad is not None
        losses.append(loss.detach())
        gradients.append(current.grad)
    torch.testing.assert_close(losses[0], losses[1], atol=0, rtol=0)
    torch.testing.assert_close(gradients[0], gradients[1], atol=0, rtol=0)


def test_packed_action_checks_reject_corruption_in_original_rows() -> None:
    obs, logs, probabilities, actions = evidence()
    packed = PackedActions.pack(obs, probabilities, logs)
    packed.probabilities[-1] = np.nan
    with pytest.raises(NumericalError, match="nonfinite_probabilities") as caught:
        packed.check(actions, logs[np.arange(4), actions])
    assert caught.value.tensors["minibatch_indexes"].tolist() == [0, 1, 2, 3]
    probabilities[1, np.flatnonzero(obs["actions_valid"][1] == 0)[0]] = 0.1
    with pytest.raises(ValueError, match="illegal_probability_mass"):
        PackedActions.pack(obs, probabilities, logs)
