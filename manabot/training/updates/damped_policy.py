"""The damped policy rule: what recipes select, with scalar or categorical targets.

The damped update is a clipped likelihood-ratio surrogate with two reverse
KL penalties, not a novel replacement for that surrogate. Targets, behavior
likelihoods and the magnet are constants. Only current policy/critic logits
receive gradients. MTG uses same-viewer transitions and an action-type magnet;
see docs/learning.md#fidelity-to-the-sources for the fidelity boundary and unresolved source details.

The update is written once over a backend module (`manabot.backends.torch`):
array math goes through `backend.rl`, `backend.learner` and `backend.arrays`,
and row bookkeeping (which rows train, minibatch indexes) is NumPy.
The objective itself is `backend.rl.damped_policy_loss`.
"""

from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

from manabot.env.observation import ACTION_TYPE_DIM
from manabot.policy.prepare import reference_distribution
from manabot.training.regime import DampedPolicyLearning
from manabot.training.rollout.collection import RolloutBatch
from manabot.training.updates.health import (
    OptimizerHealth,
    batch_health,
    failure_tensors,
    minibatch_tensors,
)
from manabot.training.updates.learner import AdmittedRollout
from manabot.training.updates.selection import (
    UpdateDiagnostics,
    selection_diagnostics,
    value_scale,
)

if TYPE_CHECKING:
    # The torch backend's types are the reference signatures.
    from torch import Tensor

    from manabot.backends.torch.learner import Learner
    from manabot.backends.torch.policy import ActorCritic


@dataclass(frozen=True)
class DampedTargets:
    advantages: "Tensor"
    values: "Tensor"
    selected: "Tensor"


def _targets(
    backend: ModuleType,
    learner: "Learner",
    admitted: AdmittedRollout,
    learning: DampedPolicyLearning,
    bootstrap_agent: "ActorCritic | None" = None,
) -> DampedTargets:
    rl, device = backend.rl, learner.device
    batch = admitted.batch
    with backend.learner.no_grad():
        rewards = backend.arrays.from_numpy(batch.rewards, device)
        ends = backend.arrays.from_numpy(batch.dones, device)
        values = backend.arrays.from_numpy(batch.values, device)
        rl.require_finite("nonfinite_collection_values", values=values)
        rl.require_finite("nonfinite_rewards", rewards=rewards)
        next_obs = admitted.next_observation(device)
        with learner.forward_precision():
            policy = bootstrap_agent or learner.agent
            _, next_raw = policy.forward_distribution(next_obs)
        next_raw = next_raw.float()
        rl.require_finite("nonfinite_bootstrap", values=next_raw)
        if learner.agent.config.value_kind == "categorical_wdl":
            if batch.outcome_probabilities is None:
                raise ValueError(
                    "categorical update requires collection outcome probabilities"
                )
            outcomes = batch.outcome_probabilities
            if not np.allclose(
                batch.values, outcomes[..., 2] - outcomes[..., 0], atol=1e-5
            ):
                raise ValueError(
                    "saved value differs from collection outcome probabilities"
                )
            probabilities = backend.arrays.from_numpy(outcomes, device)
            bootstrap = next_raw.softmax(-1)
            value_targets = rl.categorical_lambda_returns(
                probabilities, rewards, ends, bootstrap, learning.value_lambda
            )
            next_value = rl.expected_outcome(bootstrap)
        else:
            next_value = next_raw.squeeze(-1)
            _, value_targets = rl.transition_gae(
                rewards, values, ends, next_value, 1.0, learning.value_lambda
            )
        advantages, _ = rl.transition_gae(
            rewards, values, ends, next_value, 1.0, learning.policy_lambda
        )
        rl.require_finite("nonfinite_advantages", advantages=advantages)
        rl.require_finite("nonfinite_value_targets", targets=value_targets)
        # Inclusive quantile keeps all ties, unlike a top-k filter with fixed count.
        selected = rl.selection_mask(
            advantages,
            learning.filter_kind,
            1 - learning.advantage_quantile,
            learning.min_advantage,
        )
    return DampedTargets(advantages, value_targets, selected)


def update_damped_iteration(
    backend: ModuleType,
    learner: "Learner",
    batch: RolloutBatch,
    learning: DampedPolicyLearning,
    iteration: int,
    *,
    bootstrap_agent: "ActorCritic | None" = None,
) -> UpdateDiagnostics:
    """One fresh epoch grouped by native stream count, with frozen targets.

    `backend` is the module whose arrays `learner` holds
    (`manabot.backends.torch`). EMA belongs to the calling stage and advances
    once even on an empty filter. No batch shuffling, advantage normalization,
    replay or hidden truth. Scalar value is an explicit MSE ablation;
    categorical uses equation (5).

    Timed phases end where the code already reads device values (the behavior
    checks, target and diagnostic floats, the objective and gradient checks).
    """
    rl, to_device = backend.rl, backend.arrays.from_numpy
    device = learner.device
    categorical = learner.agent.config.value_kind == "categorical_wdl"
    track = learner.profiler.track
    with track("Batch transfer and behavior checks"):
        batch.action_data.check(batch.actions, batch.logprobs)
        admitted = AdmittedRollout(learner.agent, batch)
        actions = to_device(batch.actions, device)
    with track("Targets"):
        targets = _targets(backend, learner, admitted, learning, bootstrap_agent)
        advantage_abs_mean = float(targets.advantages.abs().mean())
    # Which rows the actor trains on, [T,E] on the host.
    actor_rows = backend.arrays.to_numpy(targets.selected)
    rate, tau = learning.rates(iteration)
    diagnostics: UpdateDiagnostics = {
        "gradient": learning.gradient,
        "value_kind": learner.agent.config.value_kind,
        "rows": batch.actions.size,
        "retained": int(actor_rows.sum()),
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
            np.mean(~batch.dones[-1], dtype=np.float32)
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
            rl.expected_outcome(targets.values)
            if categorical
            else targets.values
        )
        predictions = to_device(batch.values, device)
        residuals = expected_targets - predictions
        if batch.assignments is not None:
            batch.target_residuals = backend.arrays.to_numpy(residuals)
        chosen_types = to_device(
            batch.action_data.selected_types(batch.actions, ACTION_TYPE_DIM), device
        )
        diagnostics["selection_groups"] = selection_diagnostics(
            targets.advantages,
            residuals,
            targets.selected,
            to_device(batch.dones, device),
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
    learner.set_learning_rate(rate)
    # Compute credit on the original seat trajectories before regrouping rows.
    # Rectangular rollouts retain one timestep per minibatch. Complete-game
    # rollouts are packed into one column; use the same native stream count
    # rather than accidentally taking one optimizer step per recorded decision.
    width = learner.minibatch_rows
    actions = actions.flatten(0, 1)
    advantages = targets.advantages.flatten(0, 1)
    value_targets = targets.values.flatten(0, 1)
    actor_rows = actor_rows.reshape(-1)
    actor_critic = learning.filter_scope == "actor_critic"
    if not actor_rows.any() and actor_critic:
        diagnostics["skipped"] = "empty advantage filter"
        health["skipped_steps"] = (len(actor_rows) + width - 1) // width
        return diagnostics
    # Rows each minibatch trains: the actor's under actor_critic scope, else all.
    trained = actor_rows if actor_critic else np.ones_like(actor_rows)
    with track("Minibatches"):
        admitted.admit_rows(np.flatnonzero(trained), width)
        for start in range(0, len(actor_rows), width):
            window = trained[start : start + width]
            if not window.any():
                health["skipped_steps"] += 1
                continue
            with track("Forward and loss"):
                rows_host = np.arange(start, start + len(window))[window]
                indexes = to_device(rows_host, device)
                obs, rows = admitted.rows(rows_host, device)
                valid = rl.legal_mask(obs["actions_valid"])
                behavior_logs = to_device(rows.log_probabilities, device)
                with learner.forward_precision():
                    logits, value_logits = learner.agent.forward_distribution(obs)
                logits, value_logits = logits.float(), value_logits.float()
                reference = to_device(
                    reference_distribution(rows.observation, learning.reference),
                    device,
                )
                policy_loss = rl.damped_policy_loss(
                    logits,
                    actions[indexes],
                    advantages[indexes],
                    behavior_logs,
                    reference,
                    valid,
                    clip=learning.clip,
                    collection_kl=learning.collection_kl,
                    tau=tau,
                )
                actor_selected = actor_rows[rows_host]
                actor_mask = to_device(actor_selected, device)
                policy_loss = rl.selected_mean(policy_loss, actor_mask)
                if categorical:
                    value_loss = rl.categorical_value_loss(
                        value_logits, value_targets[indexes]
                    ).mean()
                else:
                    # Representation ablation, not the paper's outcome objective.
                    value_loss = rl.squared_value_loss(
                        value_logits, value_targets[indexes]
                    ).mean()
                loss = policy_loss + rl.float64_operand(value_loss)

                def _failure() -> "dict[str, Tensor]":
                    return minibatch_tensors(
                        logits=logits,
                        value_logits=value_logits,
                        valid=valid,
                        actions=actions[indexes],
                        behavior_logs=behavior_logs,
                        reference=reference,
                        advantages=advantages[indexes],
                        value_targets=value_targets[indexes],
                        actor_selected=actor_mask,
                        indexes=indexes,
                        obs=obs,
                    )

                with failure_tensors(_failure):
                    observer.check(loss)
            with failure_tensors(_failure):
                norm = backend.learner.gradient_step(
                    loss,
                    learner.agent,
                    learner.optimizer,
                    learning.max_grad_norm,
                    recorder=observer,
                )
            with track("Minibatch diagnostics"):
                diagnostics["optimizer_exposures"] += len(rows_host)
                diagnostics["actor_exposures"] += int(actor_selected.sum())
                diagnostics["critic_exposures"] += len(rows_host)
                with backend.learner.no_grad():
                    logs = rl.policy_logs(logits, valid)
                    diagnostics["entropy"] = float(rl.entropy(logs, valid).mean())
                    diagnostics["collection_kl"] = float(
                        rl.reverse_kl(logs, behavior_logs, valid).mean()
                    )
                    diagnostics["reference_kl"] = float(
                        rl.reverse_kl(
                            logs, rl.reference_logs(reference, valid), valid
                        ).mean()
                    )
                    ratio_logs = rl.action_logs(logs - behavior_logs, actions[indexes])
                    health["clip_fraction"] = rl.clip_fraction(
                        ratio_logs, learning.clip
                    )
                diagnostics.update(
                    loss=float(loss.detach()),
                    value_loss=float(value_loss.detach()),
                    policy_loss=float(policy_loss.detach()),
                    gradient_norm=float(norm),
                )
    return diagnostics
