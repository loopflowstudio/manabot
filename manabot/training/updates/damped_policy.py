"""The damped policy rule: what recipes select, with scalar or categorical targets.

The damped update is a clipped likelihood-ratio surrogate with two reverse
KL penalties, not a novel replacement for that surrogate. Targets, behavior
likelihoods and the magnet are constants. Only current policy/critic logits
receive gradients. MTG uses same-viewer transitions and an action-type magnet;
see docs/learning.md#fidelity-to-the-sources for the fidelity boundary and unresolved source details.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from manabot.backends.torch.learner import Learner, gradient_step
from manabot.backends.torch.policy import ActorCritic
from manabot.backends.torch.rl import (
    categorical_lambda_returns,
    categorical_value_loss,
    clipped_surrogate,
    entropy,
    float64_operand,
    require,
    reverse_kl,
    selected_mean,
    selection_mask,
    squared_value_loss,
    transition_gae,
)
from manabot.env.observation import ACTION_TYPE_DIM
from manabot.model.policy_distribution import (
    check_logs,
    check_probabilities,
    legal_mask,
    policy_logs,
)
from manabot.training.regime import DampedPolicyLearning
from manabot.training.rollout.collection import RolloutBatch
from manabot.training.updates.health import (
    OptimizerHealth,
    batch_health,
    clip_fraction,
    failure_tensors,
    minibatch_tensors,
)
from manabot.training.updates.learner import AdmittedRollout
from manabot.training.updates.references import reference_distribution
from manabot.training.updates.selection import (
    UpdateDiagnostics,
    selection_diagnostics,
    value_scale,
)


def damped_policy_loss(
    logits: Tensor,
    actions: Tensor,
    advantages: Tensor,
    behavior_logs: Tensor,
    reference: Tensor,
    valid: Tensor,
    *,
    clip: float,
    collection_kl: float,
    tau: float,
) -> Tensor:
    """The damped policy objective, equation (6), per row [B] over offers [B,A].

    Sources and departures: docs/learning.md#fidelity-to-the-sources.

    Ratios use the actual action likelihood at collection, even after earlier
    minibatches change the learner. The reverse KLs sum all legal actions.
    This is per-action reuse correction, not trajectory importance weighting.
    MPS combines these terms on the host to retain float64 surrogate range.
    """
    valid = valid.bool()
    if logits.ndim != 2 or logits.shape != valid.shape:
        raise ValueError("policy logits require [B,A] legal support")
    if actions.shape != logits.shape[:1] or advantages.shape != actions.shape:
        raise ValueError("actions and advantages require [B]")
    if not 0 < clip < 1 or collection_kl < 0 or tau < 0:
        raise ValueError("invalid damping coefficients")
    check_logs(behavior_logs, valid)
    check_probabilities(reference, valid, positive=True)
    require(
        torch.isfinite(advantages),
        "nonfinite_advantages",
        advantages=advantages,
        valid=valid,
    )
    if ((actions < 0) | (actions >= logits.shape[-1])).any():
        raise ValueError("action index outside legal support")
    if not valid.gather(-1, actions[:, None]).all():
        raise ValueError("sampled action is not legal")
    logs = policy_logs(logits, valid)
    reference_logs = reference.detach().masked_fill(~valid, 1).log()
    ratio_logs = (
        (logs - behavior_logs.detach()).gather(-1, actions[:, None]).squeeze(-1)
    )
    surrogate = clipped_surrogate(ratio_logs, advantages, clip)
    return (
        surrogate
        + collection_kl * float64_operand(reverse_kl(logs, behavior_logs, valid))
        + tau * float64_operand(reverse_kl(logs, reference_logs, valid))
    )


@dataclass(frozen=True)
class DampedTargets:
    advantages: Tensor
    values: Tensor
    selected: Tensor


@torch.no_grad()
def _targets(
    learner: Learner,
    admitted: AdmittedRollout,
    learning: DampedPolicyLearning,
    bootstrap_agent: ActorCritic | None = None,
) -> DampedTargets:
    device = learner.device
    batch = admitted.batch
    rewards = torch.as_tensor(batch.rewards, device=device)
    ends = torch.as_tensor(batch.dones, device=device)
    values = torch.as_tensor(batch.values, device=device)
    require(torch.isfinite(values), "nonfinite_collection_values", values=values)
    require(torch.isfinite(rewards), "nonfinite_rewards", rewards=rewards)
    next_obs = admitted.next_observation(device)
    with learner.forward_precision():
        _, next_raw = (bootstrap_agent or learner.agent).forward_distribution(next_obs)
    next_raw = next_raw.float()
    require(torch.isfinite(next_raw), "nonfinite_bootstrap", values=next_raw)
    if learner.agent.config.value_kind == "categorical_wdl":
        if batch.outcome_probabilities is None:
            raise ValueError(
                "categorical update requires collection outcome probabilities"
            )
        probabilities = torch.as_tensor(batch.outcome_probabilities, device=device)
        if not torch.allclose(
            values, probabilities[..., 2] - probabilities[..., 0], atol=1e-5
        ):
            raise ValueError(
                "saved value differs from collection outcome probabilities"
            )
        bootstrap = next_raw.softmax(-1)
        value_targets = categorical_lambda_returns(
            probabilities, rewards, ends, bootstrap, learning.value_lambda
        )
        next_value = bootstrap[:, 2] - bootstrap[:, 0]
    else:
        next_value = next_raw.squeeze(-1)
        _, value_targets = transition_gae(
            rewards, values, ends, next_value, 1.0, learning.value_lambda
        )
    advantages, _ = transition_gae(
        rewards, values, ends, next_value, 1.0, learning.policy_lambda
    )
    require(torch.isfinite(advantages), "nonfinite_advantages", advantages=advantages)
    require(
        torch.isfinite(value_targets), "nonfinite_value_targets", targets=value_targets
    )
    # Inclusive quantile keeps all ties, unlike a top-k filter with fixed count.
    selected = selection_mask(
        advantages,
        learning.filter_kind,
        1 - learning.advantage_quantile,
        learning.min_advantage,
    )
    return DampedTargets(advantages, value_targets, selected)


def update_damped_iteration(
    learner: Learner,
    batch: RolloutBatch,
    learning: DampedPolicyLearning,
    iteration: int,
    *,
    bootstrap_agent: ActorCritic | None = None,
) -> UpdateDiagnostics:
    """One fresh epoch grouped by native stream count, with frozen targets.

    EMA belongs to the calling stage and advances once even on an empty
    filter. No batch shuffling, advantage normalization, replay or hidden truth.
    Scalar value is an explicit MSE ablation; categorical uses equation (5).

    Timed phases end where the code already reads device values (the behavior
    checks, target and diagnostic floats, the objective and gradient checks).
    """
    device = learner.device
    track = learner.profiler.track
    with track("Batch transfer and behavior checks"):
        batch.action_data.check(batch.actions, batch.logprobs)
        admitted = AdmittedRollout(learner.agent, batch)
        actions = torch.as_tensor(batch.actions, device=device)
    with track("Targets"):
        targets = _targets(learner, admitted, learning, bootstrap_agent)
        advantage_abs_mean = float(targets.advantages.abs().mean())
    rate, tau = learning.rates(iteration)
    diagnostics: UpdateDiagnostics = {
        "gradient": learning.gradient,
        "value_kind": learner.agent.config.value_kind,
        "rows": actions.numel(),
        "retained": int(targets.selected.sum()),
        "optimizer_exposures": 0,
        "actor_exposures": 0,
        "critic_exposures": 0,
        "iteration": iteration,
        "learning_rate": rate,
        "tau": tau,
        "schedule_clock": "collection-update-iteration",
        "reference": learning.reference,
        "advantage_abs_mean": advantage_abs_mean,
        "bootstrapped_tail_fraction": float(
            (~torch.as_tensor(batch.dones[-1])).float().mean()
        ),
    }
    with track("Selection and health diagnostics"):
        health = batch_health(
            batch.action_data,
            targets.advantages,
            targets.selected,
            actor_only=learning.filter_scope == "actor",
        )
        diagnostics["numerical"] = health
        observer = OptimizerHealth(health, iteration, learner.profiler)
        expected_targets = (
            targets.values[..., 2] - targets.values[..., 0]
            if learner.agent.config.value_kind == "categorical_wdl"
            else targets.values
        )
        predictions = torch.as_tensor(batch.values, device=device)
        if batch.assignments is not None:
            batch.target_residuals = (
                (expected_targets - predictions).detach().cpu().numpy()
            )
        chosen_types = torch.as_tensor(
            batch.action_data.selected_types(batch.actions, ACTION_TYPE_DIM),
            device=device,
        )
        diagnostics["selection_groups"] = selection_diagnostics(
            targets.advantages,
            expected_targets - predictions,
            targets.selected,
            torch.as_tensor(batch.dones, device=device),
            chosen_types,
        )
        diagnostics["value_scale"] = value_scale(
            predictions,
            expected_targets,
            targets.advantages,
            kind=learning.filter_kind,
            quantile=learning.advantage_quantile,
            minimum=learning.min_advantage,
        )
    for group in learner.optimizer.param_groups:
        group["lr"] = rate
    # Compute credit on the original seat trajectories before regrouping rows.
    # Rectangular rollouts retain one timestep per minibatch. Complete-game
    # rollouts are packed into one column; use the same native stream count
    # rather than accidentally taking one optimizer step per recorded decision.
    width = learner.minibatch_rows
    actions = actions.flatten(0, 1)
    targets = DampedTargets(
        targets.advantages.flatten(0, 1),
        targets.values.flatten(0, 1),
        targets.selected.flatten(0, 1),
    )
    if not targets.selected.any() and learning.filter_scope == "actor_critic":
        diagnostics["skipped"] = "empty advantage filter"
        health["skipped_steps"] = (actions.numel() + width - 1) // width
        return diagnostics
    with track("Minibatches"):
        trained = (
            targets.selected
            if learning.filter_scope == "actor_critic"
            else torch.ones_like(targets.selected)
        )
        admitted.admit_rows(trained.nonzero().flatten().cpu().numpy(), width)
        for start in range(0, actions.numel(), width):
            actor_selected = targets.selected[start : start + width]
            selected = (
                actor_selected
                if learning.filter_scope == "actor_critic"
                else torch.ones_like(actor_selected)
            )
            if not selected.any():
                health["skipped_steps"] += 1
                continue
            with track("Forward and loss"):
                indexes = torch.arange(start, start + len(selected), device=device)[
                    selected
                ]
                obs, rows = admitted.rows(indexes.cpu().numpy(), device)
                valid = legal_mask(obs["actions_valid"])
                behavior_logs = torch.as_tensor(rows.log_probabilities, device=device)
                with learner.forward_precision():
                    logits, value_logits = learner.agent.forward_distribution(obs)
                logits, value_logits = logits.float(), value_logits.float()
                reference = reference_distribution(obs, learning.reference)
                policy_loss = damped_policy_loss(
                    logits,
                    actions[indexes],
                    targets.advantages[indexes],
                    behavior_logs,
                    reference,
                    valid,
                    clip=learning.clip,
                    collection_kl=learning.collection_kl,
                    tau=tau,
                )
                actor_mask = actor_selected[selected]
                policy_loss = selected_mean(policy_loss, actor_mask)
                if learner.agent.config.value_kind == "categorical_wdl":
                    value_loss = categorical_value_loss(
                        value_logits, targets.values[indexes]
                    ).mean()
                else:
                    # Representation ablation, not the paper's outcome objective.
                    value_loss = squared_value_loss(
                        value_logits, targets.values[indexes]
                    ).mean()
                loss = policy_loss + float64_operand(value_loss)

                def _failure() -> dict[str, Tensor]:
                    return minibatch_tensors(
                        logits=logits,
                        value_logits=value_logits,
                        valid=valid,
                        actions=actions[indexes],
                        behavior_logs=behavior_logs,
                        reference=reference,
                        advantages=targets.advantages[indexes],
                        value_targets=targets.values[indexes],
                        actor_selected=actor_mask,
                        indexes=indexes,
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
                diagnostics["optimizer_exposures"] += int(selected.sum())
                diagnostics["actor_exposures"] += int(actor_mask.sum())
                diagnostics["critic_exposures"] += int(selected.sum())
                with torch.no_grad():
                    logs = policy_logs(logits, valid)
                    diagnostics["entropy"] = float(entropy(logs, valid).mean())
                    diagnostics["collection_kl"] = float(
                        reverse_kl(logs, behavior_logs, valid).mean()
                    )
                    diagnostics["reference_kl"] = float(
                        reverse_kl(
                            logs,
                            reference.masked_fill(~valid, 1).log(),
                            valid,
                        ).mean()
                    )
                    ratio_logs = (
                        (logs - behavior_logs)
                        .gather(-1, actions[indexes, None])
                        .squeeze(-1)
                    )
                    health["clip_fraction"] = clip_fraction(ratio_logs, learning.clip)
                diagnostics.update(
                    loss=float(loss.detach()),
                    value_loss=float(value_loss.detach()),
                    policy_loss=float(policy_loss.detach()),
                    gradient_norm=float(norm),
                )
    return diagnostics
