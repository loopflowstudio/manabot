"""Explicit self-play estimators, regularization references and sample selection."""

import math

import numpy as np
import torch

from manabot.model.agent import Agent
from manabot.model.policy_distribution import (
    NumericalError,
    check_behavior,
    clipped_surrogate,
    legal_mask,
    policy_logs,
    require,
    reverse_kl,
)
from manabot.sim.net_opponent import NetOpponentTrainer, RolloutBatch, transition_gae
from manabot.training.ataraxos import update_move_iteration
from manabot.training.health import OptimizerHealth, batch_health
from manabot.training.models import AtaraxosMoveLearning, Learning
from manabot.training.references import reference_distribution
from manabot.training.selection import (
    UpdateDiagnostics,
    selected_mean,
    selected_rows,
    selection_diagnostics,
)


@torch.no_grad()
def update_ema(
    averaged: torch.nn.Module, learner: torch.nn.Module, rate: float
) -> None:
    """Advance once per collection/update iteration, including filtered skips.

    Initialize with a deep copy of the learner. Parameters are averaged;
    buffers describe the current model and are copied exactly. The executor
    explicitly selects raw or averaged behavior; this helper changes neither.
    """
    if not 0 <= rate < 1:
        raise ValueError("EMA rate must be in [0, 1)")
    for name, dest in averaged.named_parameters():
        dest.lerp_(learner.get_parameter(name), 1 - rate)
    for name, dest in averaged.named_buffers():
        dest.copy_(learner.get_buffer(name))


def update_iteration(
    trainer: NetOpponentTrainer,
    batch: RolloutBatch,
    learning: Learning | AtaraxosMoveLearning,
    progress: float,
    rng: np.random.Generator,
    *,
    iteration: int = 1,
    bootstrap_agent: Agent | None = None,
) -> UpdateDiagnostics:
    """Optimize one fresh collector batch on the existing trainer and Adam owner."""
    if isinstance(learning, AtaraxosMoveLearning):
        return update_move_iteration(
            trainer, batch, learning, iteration, bootstrap_agent=bootstrap_agent
        )

    dev = trainer.experiment.device
    obs = trainer._obs_to_tensors(batch.obs, dev)
    values = torch.as_tensor(batch.values, device=dev)
    rewards = torch.as_tensor(batch.rewards, device=dev)
    ends = torch.as_tensor(batch.dones, device=dev)
    with torch.no_grad():
        next_value = (bootstrap_agent or trainer.agent).get_value(
            trainer._obs_to_tensors(batch.next_obs, dev)
        )
        advantages, _ = transition_gae(
            rewards, values, ends, next_value, learning.gamma, learning.policy_lambda
        )
        _, returns = transition_gae(
            rewards, values, ends, next_value, learning.gamma, learning.value_lambda
        )
    obs = {k: v.flatten(0, 1) for k, v in obs.items()}
    advantages, returns = advantages.flatten(), returns.flatten()
    actions = torch.as_tensor(batch.actions, device=dev).flatten()
    old_logs = torch.as_tensor(batch.logprobs, device=dev).flatten()
    behavior = torch.as_tensor(batch.probabilities, device=dev).flatten(0, 1)
    behavior_logs = torch.as_tensor(batch.log_probabilities, device=dev).flatten(0, 1)
    valid = legal_mask(obs["actions_valid"])
    check_behavior(behavior_logs, behavior, actions, old_logs, valid)
    require(torch.isfinite(advantages), "nonfinite_advantages", advantages=advantages)
    require(torch.isfinite(returns), "nonfinite_value_targets", targets=returns)
    selected = selected_rows(
        advantages,
        learning.retained_fraction,
        learning.min_advantage,
        kind=learning.filter_kind,
    )
    mask = torch.zeros_like(advantages, dtype=torch.bool)
    mask[selected] = True
    training_rows = (
        selected
        if learning.filter_scope == "actor_critic"
        else torch.arange(len(advantages), device=dev)
    )
    diagnostics: UpdateDiagnostics = {
        "rows": len(advantages),
        "retained": len(selected),
        "optimizer_exposures": 0,
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
        "value_residual_abs_mean": float((returns - values.flatten()).abs().mean()),
    }
    chosen_types = obs["actions"][
        torch.arange(len(actions), device=dev), actions, :-1
    ].argmax(-1)
    diagnostics["action_types"] = torch.bincount(
        chosen_types, minlength=obs["actions"].shape[-1] - 1
    ).tolist()
    diagnostics["selected_action_types"] = torch.bincount(
        chosen_types[selected], minlength=obs["actions"].shape[-1] - 1
    ).tolist()
    diagnostics["selection_groups"] = selection_diagnostics(
        advantages.reshape_as(ends),
        (returns - values.flatten()).reshape_as(ends),
        mask.reshape_as(ends),
        ends,
        chosen_types.reshape_as(ends),
    )
    health = batch_health(
        behavior_logs,
        behavior,
        valid,
        advantages,
        mask,
        actor_only=learning.filter_scope == "actor",
    )
    diagnostics["numerical"] = health
    observer = OptimizerHealth(trainer.agent, trainer.optimizer, health, iteration)
    diagnostics["actor_exposures"] = 0
    diagnostics["critic_exposures"] = 0
    if not len(training_rows):
        diagnostics["skipped"] = "empty advantage filter"
        health["skipped_steps"] = learning.epochs * learning.minibatches
        return diagnostics
    if len(selected):
        advantages = (advantages - advantages[selected].mean()) / advantages[
            selected
        ].std(unbiased=False).clamp_min(1e-8)
    for group in trainer.optimizer.param_groups:
        group["lr"] = diagnostics["learning_rate"]
    reference = reference_distribution(obs, learning.reference)
    size = max(1, math.ceil(len(training_rows) / learning.minibatches))
    for _ in range(learning.epochs):
        order = training_rows[
            torch.as_tensor(rng.permutation(len(training_rows)), device=dev)
        ]
        for indices in order.split(size):
            logits, value = trainer.agent({k: v[indices] for k, v in obs.items()})
            logs = policy_logs(logits, valid[indices])
            log_ratio = (
                logs.gather(-1, actions[indices, None]).squeeze(-1) - old_logs[indices]
            )
            actor_mask = mask[indices]
            policy = selected_mean(
                clipped_surrogate(log_ratio, advantages[indices], learning.clip),
                actor_mask,
            )
            kl_ref = selected_mean(
                reverse_kl(
                    logs,
                    reference[indices].masked_fill(~valid[indices], 1).log(),
                    valid[indices],
                ),
                actor_mask,
            )
            kl_behavior = selected_mean(
                reverse_kl(logs, behavior_logs[indices], valid[indices]), actor_mask
            )
            value_loss = 0.5 * (value.flatten() - returns[indices]).square().mean()
            loss = (
                policy
                + learning.value_weight * value_loss
                + diagnostics["tau"] * kl_ref
                + learning.collection_kl * kl_behavior
            )
            try:
                norm = observer.step(loss, learning.max_grad_norm)
            except NumericalError as error:
                error.tensors.update(
                    {
                        "logits": logits.detach(),
                        "value_logits": value.detach(),
                        "valid": valid[indices],
                        "actions": actions[indices],
                        "behavior_logs": behavior_logs[indices],
                        "reference": reference[indices],
                        "advantages": advantages[indices],
                        "value_targets": returns[indices],
                        "actor_selected": actor_mask,
                        "minibatch_indexes": indices,
                        **{
                            f"minibatch_observation/{key}": value[indices]
                            for key, value in obs.items()
                        },
                    }
                )
                raise
            health["clip_fraction"] = float(
                ((log_ratio.detach().double().exp() - 1).abs() > learning.clip)
                .float()
                .mean()
            )
            diagnostics["optimizer_exposures"] += len(indices)
            diagnostics["actor_exposures"] += int(actor_mask.sum())
            diagnostics["critic_exposures"] += len(indices)
            diagnostics.update(
                loss=float(loss.detach()),
                policy_loss=float(policy.detach()),
                value_loss=float(value_loss.detach()),
                entropy=float(
                    -(logs.exp() * logs.masked_fill(~valid[indices], 0))
                    .sum(-1)
                    .mean()
                    .detach()
                ),
                gradient_norm=float(norm),
                reference_kl=float(kl_ref.detach()),
                collection_kl=float(kl_behavior.detach()),
            )
    return diagnostics
