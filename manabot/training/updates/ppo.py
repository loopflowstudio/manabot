"""The PPO update rule: the control the damped rule is measured against.

`update_iteration` is PPO (Schulman et al. 2017) over complete-game GAE:
shuffled minibatches, advantage filtering and normalization, a scalar value
target and optional reference and collection KL penalties. Like the damped
rule it is written once over a backend module (`manabot.backends.torch`):
array math through `backend.rl`, `backend.learner` and `backend.arrays`, row
bookkeeping in NumPy.
"""

import math
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

from manabot.env.observation import ACTION_TYPE_DIM
from manabot.policy.prepare import reference_distribution
from manabot.training.regime import Learning
from manabot.training.rollout.collection import RolloutBatch
from manabot.training.updates.health import (
    OptimizerHealth,
    batch_health,
    failure_tensors,
    minibatch_tensors,
)
from manabot.training.updates.selection import UpdateDiagnostics, selection_diagnostics

if TYPE_CHECKING:
    # The torch backend's types are the reference signatures.
    from torch import Tensor

    from manabot.backends.torch.learner import Learner
    from manabot.backends.torch.policy import ActorCritic


def update_iteration(
    backend: ModuleType,
    learner: "Learner",
    batch: RolloutBatch,
    learning: Learning,
    progress: float,
    rng: np.random.Generator,
    *,
    iteration: int = 1,
    bootstrap_agent: "ActorCritic | None" = None,
) -> UpdateDiagnostics:
    """Optimize one fresh collector batch on the shared learner and Adam owner.

    `backend` is the module whose arrays `learner` holds
    (`manabot.backends.torch`).
    """
    rl, to_device = backend.rl, backend.arrays.from_numpy
    dev = learner.device
    track = learner.profiler.track
    with track("Batch transfer, targets and behavior checks"):
        batch.action_data.check(batch.actions, batch.logprobs)
        admitted = backend.rollout.AdmittedRollout(learner.agent, batch)
        values = to_device(batch.values, dev)
        rewards = to_device(batch.rewards, dev)
        ends = to_device(batch.dones, dev)
        with backend.learner.no_grad():
            with learner.forward_precision():
                next_value = (bootstrap_agent or learner.agent).get_value(
                    admitted.next_observation(dev)
                )
            next_value = next_value.float()
            advantages, _ = rl.transition_gae(
                rewards,
                values,
                ends,
                next_value,
                learning.gamma,
                learning.policy_lambda,
            )
            _, returns = rl.transition_gae(
                rewards, values, ends, next_value, learning.gamma, learning.value_lambda
            )
        if batch.assignments is not None:
            batch.target_residuals = backend.arrays.to_numpy(returns - values)
        advantages, returns = advantages.flatten(), returns.flatten()
        actions = to_device(batch.actions, dev).flatten()
        old_logs = to_device(batch.logprobs, dev).flatten()
        rl.require_finite("nonfinite_advantages", advantages=advantages)
        rl.require_finite("nonfinite_value_targets", targets=returns)
    with track("Selection and health diagnostics"):
        selected = rl.selected_rows(
            advantages,
            learning.retained_fraction,
            learning.min_advantage,
            kind=learning.filter_kind,
        )
        # Row bookkeeping on the host: retained indexes in selection order,
        # the actor's mask and the rows each epoch trains on.
        retained = backend.arrays.to_numpy(selected)
        actor_rows = np.zeros(len(advantages), dtype=bool)
        actor_rows[retained] = True
        mask = to_device(actor_rows, dev)
        residuals = returns - values.flatten()
        training_rows = (
            retained
            if learning.filter_scope == "actor_critic"
            else np.arange(len(advantages))
        )
        diagnostics: UpdateDiagnostics = {
            "rows": len(advantages),
            "retained": len(retained),
            "optimizer_exposures": 0,
            "actor_exposures": 0,
            "critic_exposures": 0,
            "advantage_abs_mean": float(advantages.abs().mean()),
            "bootstrapped_tail_fraction": float(
                np.mean(~batch.dones[-1], dtype=np.float32)
            ),
            "learning_rate": learning.learning_rate.at(progress),
            "tau": learning.tau.at(progress),
            "schedule_progress": progress,
            "schedule_clock": "elapsed-training-budget-fraction",
            "gamma": learning.gamma,
            "policy_lambda": learning.policy_lambda,
            "value_lambda": learning.value_lambda,
            "reference": learning.reference,
            "collection_kl_coefficient": learning.collection_kl,
            "retained_fraction": len(retained) / len(advantages),
            "return_mean": float(returns.mean()),
            "value_residual_abs_mean": float(residuals.abs().mean()),
        }
        types = batch.action_data.selected_types(batch.actions, ACTION_TYPE_DIM)
        chosen_types = types.reshape(-1)
        diagnostics["action_types"] = np.bincount(
            chosen_types, minlength=ACTION_TYPE_DIM
        ).tolist()
        diagnostics["selected_action_types"] = np.bincount(
            chosen_types[retained], minlength=ACTION_TYPE_DIM
        ).tolist()
        advantage_rows = backend.arrays.to_numpy(advantages)
        diagnostics["selection_groups"] = selection_diagnostics(
            advantage_rows.reshape(types.shape),
            backend.arrays.to_numpy(residuals).reshape(types.shape),
            actor_rows.reshape(types.shape),
            batch.dones,
            types,
        )
        health = batch_health(
            batch.action_data,
            advantage_rows,
            actor_rows,
            actor_only=learning.filter_scope == "actor",
        )
        diagnostics["numerical"] = health
    observer = OptimizerHealth(health, iteration, learner.profiler, backend=backend)
    if not len(training_rows):
        diagnostics["skipped"] = "empty advantage filter"
        health["skipped_steps"] = learning.epochs * learning.minibatches
        return diagnostics
    if len(retained):
        advantages = rl.standardized_advantages(advantages, selected)
    learner.set_learning_rate(diagnostics["learning_rate"])
    size = max(1, math.ceil(len(training_rows) / learning.minibatches))
    with track("Minibatches"):
        admitted.admit_rows(training_rows, size)
        for _ in range(learning.epochs):
            order = training_rows[rng.permutation(len(training_rows))]
            for start in range(0, len(order), size):
                rows_host = order[start : start + size]
                indices = to_device(rows_host, dev)
                with track("Forward and loss"):
                    obs, rows = admitted.rows(rows_host, dev)
                    valid = rl.legal_mask(admitted.legal(obs))
                    behavior_logs = to_device(rows.log_probabilities, dev)
                    reference = to_device(
                        reference_distribution(rows.observation, learning.reference),
                        dev,
                    )
                    with learner.forward_precision():
                        logits, value = learner.agent(obs)
                    logits, value = logits.float(), value.float()
                    logs = rl.policy_logs(logits, valid)
                    log_ratio = (
                        rl.action_logs(logs, actions[indices]) - old_logs[indices]
                    )
                    actor_mask = mask[indices]
                    policy = rl.selected_mean(
                        rl.clipped_surrogate(
                            log_ratio, advantages[indices], learning.clip
                        ),
                        actor_mask,
                    )
                    kl_ref = rl.selected_mean(
                        rl.reverse_kl(logs, rl.reference_logs(reference, valid), valid),
                        actor_mask,
                    )
                    kl_behavior = rl.selected_mean(
                        rl.reverse_kl(logs, behavior_logs, valid),
                        actor_mask,
                    )
                    value_loss = (
                        0.5 * rl.squared_value_loss(value, returns[indices]).mean()
                    )
                    loss = (
                        policy
                        + learning.value_weight * rl.float64_operand(value_loss)
                        + diagnostics["tau"] * kl_ref
                        + learning.collection_kl * kl_behavior
                    )

                    def _failure() -> "dict[str, Tensor]":
                        return minibatch_tensors(
                            logits=rl.constant(logits),
                            value_logits=rl.constant(value),
                            valid=valid,
                            actions=actions[indices],
                            behavior_logs=behavior_logs,
                            reference=reference,
                            advantages=advantages[indices],
                            value_targets=returns[indices],
                            actor_selected=actor_mask,
                            indexes=indices,
                            obs=admitted.evidence(obs),
                        )

                    with failure_tensors(_failure, rl.NumericalError):
                        observer.check(loss)
                with failure_tensors(_failure, rl.NumericalError):
                    norm = learner.step(loss, learning.max_grad_norm, recorder=observer)
                with track("Minibatch diagnostics"):
                    health["clip_fraction"] = rl.clip_fraction(log_ratio, learning.clip)
                    diagnostics["optimizer_exposures"] += len(rows_host)
                    diagnostics["actor_exposures"] += int(actor_rows[rows_host].sum())
                    diagnostics["critic_exposures"] += len(rows_host)
                    diagnostics.update(
                        loss=float(loss.detach()),
                        policy_loss=float(policy.detach()),
                        value_loss=float(value_loss.detach()),
                        entropy=float(rl.entropy(logs, valid).mean().detach()),
                        gradient_norm=float(norm),
                        reference_kl=float(kl_ref.detach()),
                        collection_kl=float(kl_behavior.detach()),
                    )
    return diagnostics
