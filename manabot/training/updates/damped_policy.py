"""The damped policy rule: what recipes select, with scalar or categorical targets.

The damped update is a clipped likelihood-ratio surrogate with two reverse
KL penalties, not a novel replacement for that surrogate. Targets, behavior
likelihoods and the magnet are constants. Only current policy/critic logits
receive gradients. MTG uses same-viewer transitions and an action-type magnet;
see docs/learning.md#fidelity-to-the-sources for the fidelity boundary and unresolved source details.

The update is written once over a backend module (`manabot.backends.torch`,
`manabot.backends.mlx`; the surface it calls is
`manabot.training.updates.backend.Backend`): array math goes through
`backend.rl`, `backend.learner` and `backend.arrays`, observations are
admitted by `backend.rollout`, and row bookkeeping and diagnostics (which rows train,
minibatch indexes, selection groups, value scale) are NumPy. The minibatch
loss is `minibatch_objective`, equation (6) in `backend.rl.damped_policy_loss`
plus the critic's loss; the learner differentiates it as a closure over the
policy forward (`Learner.loss`), which is how a functional framework takes a
gradient.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from manabot.env.observation import ACTION_TYPE_DIM
from manabot.policy.prepare import reference_distribution
from manabot.training.regime import DampedPolicyLearning
from manabot.training.rollout.collection import RolloutBatch
from manabot.training.updates.backend import (
    RL,
    AdmittedRollout,
    Array,
    Backend,
    Learner,
    Policy,
    categorical_rl,
)
from manabot.training.updates.health import (
    OptimizerHealth,
    batch_health,
    failure_tensors,
    minibatch_tensors,
)
from manabot.training.updates.selection import (
    UpdateDiagnostics,
    selection_diagnostics,
    value_scale,
)


@dataclass(frozen=True)
class DampedTargets:
    advantages: Array
    values: Array
    selected: Array


def minibatch_objective(
    rl: RL,
    logits: Array,
    value_logits: Array,
    valid: Array,
    *,
    actions: Array,
    advantages: Array,
    value_targets: Array,
    behavior_logs: Array,
    reference: Array,
    actor_mask: Array,
    categorical: bool,
    clip: float,
    collection_kl: float,
    tau: float,
) -> tuple[Array, Array, Array]:
    """The damped rule's loss on one minibatch: (loss, policy loss, value loss).

    `rl` is a backend's `rl` module and every array is that backend's:
    float32 `logits` [B,A] and critic output ([B,1], or [B,3] outcome logits
    when `categorical`) from the current forward, the legal mask [B,A], and
    the constants `actions` [B], `advantages` [B], `value_targets` ([B] or
    [B,3]), `behavior_logs` [B,A], `reference` probabilities [B,A] and the
    actor-selected rows [B]. The policy term averages equation (6) over the
    actor's rows; the critic term averages every row. Gradients reach only
    `logits` and `value_logits`.
    """
    policy_loss = rl.selected_mean(
        rl.damped_policy_loss(
            logits,
            actions,
            advantages,
            behavior_logs,
            reference,
            valid,
            clip=clip,
            collection_kl=collection_kl,
            tau=tau,
        ),
        actor_mask,
    )
    if categorical:
        value_loss = (
            categorical_rl(rl).categorical_value_loss(value_logits, value_targets)
        ).mean()
    else:
        # Representation ablation, not the paper's outcome objective.
        value_loss = rl.squared_value_loss(value_logits, value_targets).mean()
    return policy_loss + rl.float64_operand(value_loss), policy_loss, value_loss


def _targets(
    backend: Backend,
    learner: Learner,
    admitted: AdmittedRollout,
    learning: DampedPolicyLearning,
    bootstrap_agent: Policy | None = None,
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
        next_raw = backend.arrays.float32(next_raw)
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
            wdl = categorical_rl(rl)
            bootstrap = wdl.outcome_probabilities(next_raw)
            value_targets = wdl.categorical_lambda_returns(
                probabilities, rewards, ends, bootstrap, learning.value_lambda
            )
            next_value = wdl.expected_outcome(bootstrap)
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
    backend: Backend,
    learner: Learner,
    batch: RolloutBatch,
    learning: DampedPolicyLearning,
    iteration: int,
    *,
    bootstrap_agent: Policy | None = None,
) -> UpdateDiagnostics:
    """One fresh epoch grouped by native stream count, with frozen targets.

    `backend` is the module whose arrays `learner` holds. EMA belongs to the calling stage and advances
    once even on an empty filter. No batch shuffling, advantage normalization,
    replay or hidden truth. Scalar value is an explicit MSE ablation;
    categorical uses equation (5).

    Timed phases end where the code already reads device values (the behavior
    checks, target and diagnostic floats, the objective and gradient checks).
    """
    rl, to_device = backend.rl, backend.arrays.from_numpy
    to_host, float32 = backend.arrays.to_numpy, backend.arrays.float32
    device = learner.device
    categorical = learner.agent.config.value_kind == "categorical_wdl"
    track = learner.profiler.track
    with track("Batch transfer and behavior checks"):
        batch.action_data.check(batch.actions, batch.logprobs)
        admitted = backend.rollout.AdmittedRollout(learner.agent, batch)
        actions = to_device(batch.actions, device)
    with track("Targets"):
        targets = _targets(backend, learner, admitted, learning, bootstrap_agent)
        advantage_abs_mean = float(targets.advantages.abs().mean())
    # Which rows the actor trains on, [T,E] on the host.
    actor_rows = to_host(targets.selected)
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
        advantage_rows = to_host(targets.advantages)
        health = batch_health(
            batch.action_data,
            advantage_rows,
            actor_rows,
            actor_only=learning.filter_scope == "actor",
        )
        diagnostics["numerical"] = health
        observer = OptimizerHealth(health, iteration, learner.profiler, backend=backend)
        expected_targets = (
            categorical_rl(rl).expected_outcome(targets.values)
            if categorical
            else targets.values
        )
        expected_rows = to_host(expected_targets)
        # In the backend's arithmetic: these residuals are credit evidence.
        residuals = to_host(expected_targets - to_device(batch.values, device))
        if batch.assignments is not None:
            batch.target_residuals = residuals
        diagnostics["selection_groups"] = selection_diagnostics(
            advantage_rows,
            residuals,
            actor_rows,
            batch.dones,
            batch.action_data.selected_types(batch.actions, ACTION_TYPE_DIM),
        )
        diagnostics["value_scale"] = value_scale(
            batch.values,
            expected_rows,
            advantage_rows,
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
                minibatch_actions = actions[indexes]
                obs, rows = admitted.rows(rows_host, device)
                valid = rl.legal_mask(admitted.legal(obs))
                behavior_logs = to_device(rows.log_probabilities, device)
                reference = to_device(
                    reference_distribution(rows.observation, learning.reference),
                    device,
                )
                actor_selected = actor_rows[rows_host]
                actor_mask = to_device(actor_selected, device)

                def _objective() -> tuple[Array, ...]:
                    with learner.forward_precision():
                        logits, value_logits = learner.agent.forward_distribution(obs)
                    logits, value_logits = float32(logits), float32(value_logits)
                    return (
                        *minibatch_objective(
                            rl,
                            logits,
                            value_logits,
                            valid,
                            actions=minibatch_actions,
                            advantages=advantages[indexes],
                            value_targets=value_targets[indexes],
                            behavior_logs=behavior_logs,
                            reference=reference,
                            actor_mask=actor_mask,
                            categorical=categorical,
                            clip=learning.clip,
                            collection_kl=learning.collection_kl,
                            tau=tau,
                        ),
                        logits,
                        value_logits,
                    )

                loss, policy_loss, value_loss, logits, value_logits = learner.loss(
                    _objective
                )

                def _failure() -> dict[str, Any]:
                    return minibatch_tensors(
                        logits=rl.constant(logits),
                        value_logits=rl.constant(value_logits),
                        valid=valid,
                        actions=minibatch_actions,
                        behavior_logs=behavior_logs,
                        reference=reference,
                        advantages=advantages[indexes],
                        value_targets=value_targets[indexes],
                        actor_selected=actor_mask,
                        indexes=indexes,
                        obs=admitted.evidence(obs),
                    )

                with failure_tensors(_failure, rl.NumericalError):
                    observer.check(loss)
            with failure_tensors(_failure, rl.NumericalError):
                norm = learner.step(loss, learning.max_grad_norm, recorder=observer)
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
                    ratio_logs = rl.action_logs(logs - behavior_logs, minibatch_actions)
                    health["clip_fraction"] = rl.clip_fraction(
                        ratio_logs, learning.clip
                    )
                diagnostics.update(
                    loss=float(rl.constant(loss)),
                    value_loss=float(rl.constant(value_loss)),
                    policy_loss=float(rl.constant(policy_loss)),
                    gradient_norm=float(norm),
                )
    return diagnostics
