"""The PPO update rule: the control the damped rule is measured against.

`update_iteration` is PPO (Schulman et al. 2017) over complete-game GAE:
shuffled minibatches, advantage filtering and normalization, a scalar value
target and optional reference and collection KL penalties. Like the damped
rule it is written once over a backend module: array math through
`backend.rl`, `backend.learner` and `backend.arrays`, row bookkeeping in
NumPy. It needs a `PPOBackend` (`manabot.training.updates.backend`), which
only `manabot.backends.torch` provides.
"""

import math

import numpy as np

from manabot.env.observation import ACTION_TYPE_DIM
from manabot.policy.prepare import reference_distribution
from manabot.training.learning import Learning
from manabot.training.rollout.collection import RolloutBatch
from manabot.training.updates.backend import (
    Array,
    Observations,
    PPOBackend,
    PPOLearner,
    PPOPolicy,
    ppo_rl,
)
from manabot.training.updates.decision_weights import (
    actor_term,
    batch_weights,
    critic_term,
    logical_weights,
)
from manabot.training.updates.health import (
    OptimizerHealth,
    batch_health,
    failure_tensors,
    minibatch_tensors,
)
from manabot.training.updates.selection import UpdateDiagnostics, selection_diagnostics
from manabot.training.updates.targets import BatchTargets, compute_batch_targets


def update_iteration(
    backend: PPOBackend,
    learner: PPOLearner,
    batch: RolloutBatch,
    learning: Learning,
    progress: float,
    rng: np.random.Generator,
    *,
    iteration: int = 1,
    bootstrap_agent: PPOPolicy | None = None,
    targets: BatchTargets | None = None,
) -> UpdateDiagnostics:
    """Optimize one fresh collector batch on the shared learner and Adam owner.

    `backend` is the module whose arrays `learner` holds.
    """
    rl, to_device = ppo_rl(backend.rl), backend.arrays.from_numpy
    dev = learner.device
    track = learner.profiler.track
    with track("Batch transfer, targets and behavior checks"):
        batch.action_data.check(
            batch.actions,
            batch.logprobs,
            validator=backend.rollout,
        )
        admitted = backend.rollout.AdmittedRollout(learner.agent, batch)
        if targets is None:
            targets = compute_batch_targets(
                backend, learner, admitted, learning, bootstrap_agent
            )
        else:
            targets.check(batch, learning)
        residuals = backend.arrays.to_numpy(targets.residuals)
        if batch.assignments is not None:
            batch.target_residuals = residuals
        advantages, returns = (
            targets.advantages.flatten(),
            targets.value_targets.flatten(),
        )
        actions = to_device(batch.actions, dev).flatten()
        old_logs = to_device(batch.logprobs, dev).flatten()
    with track("Selection and health diagnostics"):
        retained = targets.retained
        mask = targets.selected.flatten()
        actor_rows = backend.arrays.to_numpy(mask)
        training_rows = (
            retained
            if learning.filter_scope == "actor_critic"
            else np.flatnonzero(targets.positive_weight)
        )
        diagnostics: UpdateDiagnostics = {
            "rows": len(advantages),
            "retained": len(retained),
            "source_selection": {
                s.source: s.diagnostics() for s in targets.source_selection
            },
            "optimizer_exposures": 0,
            "actor_exposures": 0,
            "critic_exposures": 0,
            "advantage_abs_mean": float(advantages.abs().mean()),
            "bootstrapped_tail_fraction": batch.bootstrapped_tail_fraction(),
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
            "value_residual_abs_mean": float(targets.residuals.abs().mean()),
        }
        types = batch.action_data.selected_types(batch.actions, ACTION_TYPE_DIM)
        chosen_types = types.reshape(-1)
        diagnostics["selected_action_types"] = np.bincount(
            chosen_types[retained], minlength=ACTION_TYPE_DIM
        ).tolist()
        advantage_rows = backend.arrays.to_numpy(advantages)
        diagnostics["selection_groups"] = selection_diagnostics(
            advantage_rows.reshape(types.shape),
            residuals,
            actor_rows.reshape(types.shape),
            batch.dones,
            types,
            batch.segment_ends(),
        )
        health = batch_health(
            batch.action_data,
            advantage_rows,
            actor_rows,
            actor_only=learning.filter_scope == "actor",
        )
        health["critic_rows"] = len(training_rows)
        diagnostics["numerical"] = health
    observer = OptimizerHealth(health, iteration, learner.profiler, backend=backend)
    if not len(training_rows):
        diagnostics["skipped"] = targets.empty_filter_reason
        health["skipped_steps"] = learning.epochs * learning.minibatches
        return diagnostics
    if len(retained):
        advantages = rl.standardized_advantages(advantages, to_device(retained, dev))
    learner.set_learning_rate(diagnostics["learning_rate"])
    size = max(1, math.ceil(len(training_rows) / learning.minibatches))
    normalized = batch_weights(
        batch, training_rows, actor_rows, learning.root_actor_weight
    )
    with track("Minibatches"):
        admitted.admit_rows(training_rows, min(size, learner.execution_rows or size))
        for _ in range(learning.epochs):
            order = training_rows[rng.permutation(len(training_rows))]
            for start in range(0, len(order), size):
                logical_rows = order[start : start + size]
                execution_rows = learner.execution_rows or size
                split = len(logical_rows) > execution_rows
                actor_count = int(actor_rows[logical_rows].sum())
                weights = logical_weights(
                    normalized, logical_rows, actor_count, backend.arrays, dev
                )
                totals = dict(
                    loss=0.0,
                    policy_loss=0.0,
                    value_loss=0.0,
                    entropy=0.0,
                    reference_kl=0.0,
                    collection_kl=0.0,
                    clip_fraction=0.0,
                )
                with admitted.minibatches(
                    logical_rows, execution_rows, dev
                ) as minibatches:
                    for offset in range(0, len(logical_rows), execution_rows):
                        with track("Preparation wait and transfer"):
                            rows_host, obs, rows = next(minibatches)
                        indices = to_device(rows_host, dev)
                        with track("Forward and loss"):
                            valid = rl.legal_mask(admitted.legal(obs))
                            behavior_logs = to_device(rows.log_probabilities, dev)
                            reference = to_device(
                                reference_distribution(
                                    rows.observation, learning.reference
                                ),
                                dev,
                            )
                            with learner.forward_precision():
                                logits, value = learner.agent(obs)
                            logits, value = logits.float(), value.float()
                            logs = rl.policy_logs(logits, valid)
                            log_ratio = (
                                rl.action_logs(logs, actions[indices])
                                - old_logs[indices]
                            )
                            actor_mask = mask[indices]
                            policy_weight = int(actor_rows[rows_host].sum()) / max(
                                1, actor_count
                            )
                            value_weight = len(rows_host) / len(logical_rows)
                            coefficients = (
                                None
                                if weights is None
                                else weights[offset : offset + execution_rows]
                            )
                            policy = actor_term(
                                rl,
                                rl.clipped_surrogate(
                                    log_ratio, advantages[indices], learning.clip
                                ),
                                actor_mask,
                                policy_weight,
                                coefficients,
                            )
                            kl_ref = actor_term(
                                rl,
                                rl.reverse_kl(
                                    logs, rl.reference_logs(reference, valid), valid
                                ),
                                actor_mask,
                                policy_weight,
                                coefficients,
                            )
                            kl_behavior = actor_term(
                                rl,
                                rl.reverse_kl(logs, behavior_logs, valid),
                                actor_mask,
                                policy_weight,
                                coefficients,
                            )
                            value_loss = 0.5 * critic_term(
                                rl,
                                rl.squared_value_loss(value, returns[indices]),
                                value_weight,
                                coefficients,
                            )
                            loss = (
                                policy
                                + learning.value_weight * rl.float64_operand(value_loss)
                                + diagnostics["tau"] * kl_ref
                                + learning.collection_kl * kl_behavior
                            )

                            def _failure(
                                logits: Array = logits,
                                value: Array = value,
                                obs: Observations = obs,
                            ) -> dict[str, Array]:
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
                            if split:
                                learner.accumulate(loss, first=offset == 0)
                            if offset + execution_rows >= len(logical_rows):
                                norm = learner.step(
                                    loss,
                                    learning.max_grad_norm,
                                    recorder=observer,
                                    accumulated=split,
                                )
                        with track("Minibatch diagnostics"):
                            totals["clip_fraction"] += (
                                float(rl.clip_fraction(log_ratio, learning.clip))
                                * value_weight
                            )
                            totals["loss"] += float(loss.detach())
                            totals["policy_loss"] += float(policy.detach())
                            totals["value_loss"] += float(value_loss.detach())
                            totals["entropy"] += (
                                float(rl.entropy(logs, valid).mean().detach())
                                * value_weight
                            )
                            totals["reference_kl"] += float(kl_ref.detach())
                            totals["collection_kl"] += float(kl_behavior.detach())
                        del (
                            obs,
                            rows,
                            logits,
                            value,
                            logs,
                            loss,
                            policy,
                            value_loss,
                            kl_ref,
                            kl_behavior,
                            log_ratio,
                            _failure,
                        )
                health["clip_fraction"] = totals.pop("clip_fraction")
                diagnostics.update(totals, gradient_norm=float(norm))
                diagnostics["optimizer_exposures"] += len(logical_rows)
                diagnostics["actor_exposures"] += actor_count
                diagnostics["critic_exposures"] += len(logical_rows)
    return diagnostics
