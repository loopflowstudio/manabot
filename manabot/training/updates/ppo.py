"""The PPO update rule: the control the damped rule is measured against.

`update_iteration` is PPO (Schulman et al. 2017) over complete-game GAE:
shuffled minibatches, advantage filtering and normalization, a scalar value
target and optional reference and collection KL penalties.
"""

import math

import numpy as np
import torch

from manabot.env.observation import ACTION_TYPE_DIM
from manabot.model.actor_critic import ActorCritic
from manabot.model.policy_distribution import legal_mask, policy_logs
from manabot.rl.losses import (
    clipped_surrogate,
    entropy,
    reverse_kl,
    squared_value_loss,
)
from manabot.rl.numerics import float64_operand, require
from manabot.rl.returns import standardized_advantages, transition_gae
from manabot.rl.selection import selected_mean, selected_rows
from manabot.rl.step import gradient_step
from manabot.training.regime import Learning
from manabot.training.rollout.collection import RolloutBatch
from manabot.training.updates.health import (
    OptimizerHealth,
    batch_health,
    clip_fraction,
    failure_tensors,
    minibatch_tensors,
)
from manabot.training.updates.learner import AdmittedRollout, Learner
from manabot.training.updates.references import reference_distribution
from manabot.training.updates.selection import UpdateDiagnostics, selection_diagnostics


def update_iteration(
    learner: Learner,
    batch: RolloutBatch,
    learning: Learning,
    progress: float,
    rng: np.random.Generator,
    *,
    iteration: int = 1,
    bootstrap_agent: ActorCritic | None = None,
) -> UpdateDiagnostics:
    """Optimize one fresh collector batch on the shared learner and Adam owner."""
    dev = learner.device
    track = learner.profiler.track
    with track("Batch transfer, targets and behavior checks"):
        batch.action_data.check(batch.actions, batch.logprobs)
        admitted = AdmittedRollout(learner.agent, batch)
        values = torch.as_tensor(batch.values, device=dev)
        rewards = torch.as_tensor(batch.rewards, device=dev)
        ends = torch.as_tensor(batch.dones, device=dev)
        with torch.no_grad():
            with learner.forward_precision():
                next_value = (bootstrap_agent or learner.agent).get_value(
                    admitted.next_observation(dev)
                )
            next_value = next_value.float()
            advantages, _ = transition_gae(
                rewards,
                values,
                ends,
                next_value,
                learning.gamma,
                learning.policy_lambda,
            )
            _, returns = transition_gae(
                rewards, values, ends, next_value, learning.gamma, learning.value_lambda
            )
        if batch.assignments is not None:
            batch.target_residuals = (returns - values).detach().cpu().numpy()
        advantages, returns = advantages.flatten(), returns.flatten()
        actions = torch.as_tensor(batch.actions, device=dev).flatten()
        old_logs = torch.as_tensor(batch.logprobs, device=dev).flatten()
        require(
            torch.isfinite(advantages), "nonfinite_advantages", advantages=advantages
        )
        require(torch.isfinite(returns), "nonfinite_value_targets", targets=returns)
    with track("Selection and health diagnostics"):
        selected = selected_rows(
            advantages,
            learning.retained_fraction,
            learning.min_advantage,
            kind=learning.filter_kind,
        )
        mask = torch.zeros_like(advantages, dtype=torch.bool)
        mask[selected] = True
        residuals = returns - values.flatten()
        training_rows = (
            selected
            if learning.filter_scope == "actor_critic"
            else torch.arange(len(advantages), device=dev)
        )
        diagnostics: UpdateDiagnostics = {
            "rows": len(advantages),
            "retained": len(selected),
            "optimizer_exposures": 0,
            "actor_exposures": 0,
            "critic_exposures": 0,
            "advantage_abs_mean": float(advantages.abs().mean()),
            "bootstrapped_tail_fraction": float((~ends[-1]).float().mean()),
            "learning_rate": learning.learning_rate.at(progress),
            "tau": learning.tau.at(progress),
            "schedule_progress": progress,
            "schedule_clock": "elapsed-training-budget-fraction",
            "gamma": learning.gamma,
            "policy_lambda": learning.policy_lambda,
            "value_lambda": learning.value_lambda,
            "reference": learning.reference,
            "collection_kl_coefficient": learning.collection_kl,
            "retained_fraction": len(selected) / len(advantages),
            "return_mean": float(returns.mean()),
            "value_residual_abs_mean": float(residuals.abs().mean()),
        }
        chosen_types = torch.as_tensor(
            batch.action_data.selected_types(batch.actions, ACTION_TYPE_DIM), device=dev
        ).flatten()
        diagnostics["action_types"] = torch.bincount(
            chosen_types, minlength=ACTION_TYPE_DIM
        ).tolist()
        diagnostics["selected_action_types"] = torch.bincount(
            chosen_types[selected], minlength=ACTION_TYPE_DIM
        ).tolist()
        diagnostics["selection_groups"] = selection_diagnostics(
            advantages.reshape_as(ends),
            residuals.reshape_as(ends),
            mask.reshape_as(ends),
            ends,
            chosen_types.reshape_as(ends),
        )
        health = batch_health(
            batch.action_data,
            advantages,
            mask,
            actor_only=learning.filter_scope == "actor",
        )
        diagnostics["numerical"] = health
    observer = OptimizerHealth(health, iteration, learner.profiler)
    if not len(training_rows):
        diagnostics["skipped"] = "empty advantage filter"
        health["skipped_steps"] = learning.epochs * learning.minibatches
        return diagnostics
    if len(selected):
        advantages = standardized_advantages(advantages, selected)
    for group in learner.optimizer.param_groups:
        group["lr"] = diagnostics["learning_rate"]
    size = max(1, math.ceil(len(training_rows) / learning.minibatches))
    with track("Minibatches"):
        admitted.admit_rows(training_rows.cpu().numpy(), size)
        for _ in range(learning.epochs):
            order = training_rows[
                torch.as_tensor(rng.permutation(len(training_rows)), device=dev)
            ]
            for indices in order.split(size):
                with track("Forward and loss"):
                    obs, rows = admitted.rows(indices.cpu().numpy(), dev)
                    valid = legal_mask(obs["actions_valid"])
                    behavior_logs = torch.as_tensor(rows.log_probabilities, device=dev)
                    reference = reference_distribution(obs, learning.reference)
                    with learner.forward_precision():
                        logits, value = learner.agent(obs)
                    logits, value = logits.float(), value.float()
                    logs = policy_logs(logits, valid)
                    log_ratio = (
                        logs.gather(-1, actions[indices, None]).squeeze(-1)
                        - old_logs[indices]
                    )
                    actor_mask = mask[indices]
                    policy = selected_mean(
                        clipped_surrogate(
                            log_ratio, advantages[indices], learning.clip
                        ),
                        actor_mask,
                    )
                    kl_ref = selected_mean(
                        reverse_kl(
                            logs,
                            reference.masked_fill(~valid, 1).log(),
                            valid,
                        ),
                        actor_mask,
                    )
                    kl_behavior = selected_mean(
                        reverse_kl(logs, behavior_logs, valid),
                        actor_mask,
                    )
                    value_loss = (
                        0.5 * squared_value_loss(value, returns[indices]).mean()
                    )
                    loss = (
                        policy
                        + learning.value_weight * float64_operand(value_loss)
                        + diagnostics["tau"] * kl_ref
                        + learning.collection_kl * kl_behavior
                    )

                    def _failure() -> dict[str, torch.Tensor]:
                        return minibatch_tensors(
                            logits=logits,
                            value_logits=value,
                            valid=valid,
                            actions=actions[indices],
                            behavior_logs=behavior_logs,
                            reference=reference,
                            advantages=advantages[indices],
                            value_targets=returns[indices],
                            actor_selected=actor_mask,
                            indexes=indices,
                            obs=obs,
                        )

                    with failure_tensors(_failure):
                        observer.check(loss)
                with failure_tensors(_failure):
                    norm = gradient_step(
                        loss,
                        learner.agent,
                        learner.optimizer,
                        learning.max_grad_norm,
                        recorder=observer,
                    )
                with track("Minibatch diagnostics"):
                    health["clip_fraction"] = clip_fraction(log_ratio, learning.clip)
                    diagnostics["optimizer_exposures"] += len(indices)
                    diagnostics["actor_exposures"] += int(actor_mask.sum())
                    diagnostics["critic_exposures"] += len(indices)
                    diagnostics.update(
                        loss=float(loss.detach()),
                        policy_loss=float(policy.detach()),
                        value_loss=float(value_loss.detach()),
                        entropy=float(entropy(logs, valid).mean().detach()),
                        gradient_norm=float(norm),
                        reference_kl=float(kl_ref.detach()),
                        collection_kl=float(kl_behavior.detach()),
                    )
    return diagnostics
