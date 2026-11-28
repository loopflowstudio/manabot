"""Complete behavior evidence survives compact storage and minibatch replay."""

import numpy as np
import pytest
import torch

import manabot.backends.torch.rl as rl
from manabot.backends.torch.rl import (
    NumericalError,
    clipped_surrogate,
    policy_logs,
    reverse_kl,
)
import manabot.backends.torch.rollout as torch_rollout
from manabot.env.match import Match
from manabot.env.observation import ObservationSpace, encode_policy_batch
from manabot.policy.spans import ObservationSelection
from manabot.training.rollout.actions import PackedActions
from manabot.training.updates.objective import damped_policy_loss
import managym


def decisions(count: int = 4) -> list[managym.Observation]:
    """Real decisions with at least two legal offers, from one seeded game."""
    env = managym.Env(seed=243, skip_trivial=True)
    env.reset(Match().to_rust())
    found = []
    while len(found) < count:
        obs = env.observation_for_player(env.current_agent_index())
        if len(obs.action_space.actions) >= 2:
            found.append(obs)
        env.step(len(obs.action_space.actions) - 1)
    return found


def evidence() -> tuple[ObservationSelection, np.ndarray, np.ndarray]:
    space = ObservationSpace()
    batch = encode_policy_batch(space, decisions())
    observation = ObservationSelection(
        space, tuple((batch, row) for row in range(batch.batch_size))
    )
    valid = observation.project("actions_valid")
    rng = np.random.default_rng(243)
    logits = torch.from_numpy(rng.normal(size=valid.shape).astype(np.float32))
    # Retain an underflowed legal probability without replacing finite logs.
    logits[0, int(observation.action_counts[0]) - 1] = -150
    logs = policy_logs(logits, torch.from_numpy(valid).bool()).numpy()
    actions = (observation.action_counts - 1).astype(np.int64)
    return observation, logs, actions


def test_packed_actions_preserve_coordinates_distributions_and_copy_ownership() -> None:
    obs, logs, actions = evidence()
    packed = PackedActions.pack(obs, logs, validator=torch_rollout)
    selected = logs[np.arange(4), actions]
    packed.check(actions, selected, validator=torch_rollout)
    order = np.array([2, 0, 3, 1])
    restored = packed.unpack_logs(order)
    np.testing.assert_array_equal(restored, logs[order])
    assert packed.selection.select(order).coordinates == obs.select(order).coordinates
    assert np.isfinite(packed.log_probabilities).all()
    assert (np.exp(packed.log_probabilities) == 0).any()
    assert packed.nbytes < logs.nbytes
    logs.fill(0)
    np.testing.assert_array_equal(packed.unpack_logs(order), restored)
    joined = PackedActions.concatenate(
        [
            PackedActions.pack(
                obs.select(order[i : i + 1]),
                restored[i : i + 1],
                validator=torch_rollout,
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
    obs, logs, actions = evidence()
    packed = PackedActions.pack(obs, logs, validator=torch_rollout)
    indexes = np.array([3, 0])
    restored = packed.behavior(indexes)
    gradients = []
    losses = []
    for old, valid in (
        (logs[indexes], obs.project("actions_valid")[indexes]),
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
                rl,
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
    obs, logs, actions = evidence()
    packed = PackedActions.pack(obs, logs, validator=torch_rollout)
    packed.log_probabilities[-1] = np.nan
    with pytest.raises(NumericalError, match="nonfinite_behavior_logs") as caught:
        packed.check(actions, logs[np.arange(4), actions], validator=torch_rollout)
    assert caught.value.tensors["minibatch_indexes"].tolist() == [0, 1, 2, 3]
    logs[1, int(obs.action_counts[1])] = -1
    with pytest.raises(ValueError, match="illegal_log_mass"):
        PackedActions.pack(obs, logs, validator=torch_rollout)


def test_rows_of_a_packed_block_reassemble_it_exactly() -> None:
    obs, logs, _ = evidence()
    packed = PackedActions.pack(obs, logs, validator=torch_rollout)
    whole = PackedActions.concatenate([packed.row(index) for index in range(4)])
    np.testing.assert_array_equal(whole.offsets, packed.offsets)
    np.testing.assert_array_equal(whole.slots, packed.slots)
    np.testing.assert_array_equal(whole.log_probabilities, packed.log_probabilities)
    assert whole.selection.coordinates == packed.selection.coordinates


def test_selected_types_read_the_chosen_action_of_every_row() -> None:
    obs, logs, actions = evidence()
    packed = PackedActions.pack(obs, logs, validator=torch_rollout)
    expected = obs.project("actions")[np.arange(4), actions, :6].argmax(-1)
    np.testing.assert_array_equal(
        packed.selected_types(actions.reshape(2, 2), 6), expected.reshape(2, 2)
    )


def test_selected_types_reject_an_action_outside_packed_support() -> None:
    obs, logs, actions = evidence()
    packed = PackedActions.pack(obs, logs, validator=torch_rollout)
    illegal = actions.copy()
    illegal[2] = obs.action_counts[2]
    with pytest.raises(ValueError, match="absent from packed support"):
        packed.selected_types(illegal, 6)
