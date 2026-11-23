"""The damped policy rule: what recipes select, with scalar or categorical targets.

The damped update is a clipped likelihood-ratio surrogate with two reverse
KL penalties, not a novel replacement for that surrogate. Targets, behavior
likelihoods and the magnet are constants. Only current policy/critic logits
receive gradients. MTG uses same-viewer transitions and an action-type magnet;
see docs/learning.md#fidelity-to-the-sources for the fidelity boundary and unresolved source details.

The update is written once over a backend module (`manabot.backends.torch`,
`manabot.backends.mlx`; the surface it calls is
`manabot.training.updates.backend.UpdateBackend`): array math goes through
`backend.rl`, `backend.learner` and `backend.arrays`, observations are
admitted by `backend.rollout`, and row bookkeeping and diagnostics (which rows train,
minibatch indexes, selection groups, value scale) are NumPy. The minibatch
loss is `minibatch_objective`, equation (6) in `updates.objective.damped_policy_loss`
plus the critic's loss; the learner differentiates it as a closure over the
policy forward (`Learner.loss`), which is how a functional framework takes a
gradient.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from manabot.env.observation import ACTION_TYPE_DIM
from manabot.policy.prepare import reference_distribution
from manabot.training.learning import DampedPolicyLearning
from manabot.training.rollout.collection import RolloutBatch
from manabot.training.updates.backend import (
    AdmittedRollout,
    Array,
    Learner,
    Observations,
    Policy,
    UpdateBackend,
    categorical_rl,
)
from manabot.training.updates.bootstrap import bootstrap_predictions
from manabot.training.updates.health import (
    OptimizerHealth,
    batch_health,
    failure_tensors,
    minibatch_tensors,
)
from manabot.training.updates.objective import minibatch_objective
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


def compute_targets(
    backend: UpdateBackend,
    learner: Learner,
    admitted: AdmittedRollout,
    learning: DampedPolicyLearning,
    bootstrap_agent: Policy | None = None,
) -> DampedTargets:
    """Build detached learning targets and selection from an admitted rollout."""
    rl, device = backend.rl, learner.device
    batch = admitted.batch
    with backend.learner.no_grad():
        rewards = backend.arrays.from_numpy(batch.rewards, device)
        ends = backend.arrays.from_numpy(batch.dones, device)
        values = backend.arrays.from_numpy(batch.values, device)
        rl.require_finite("nonfinite_collection_values", values=values)
        rl.require_finite("nonfinite_rewards", rewards=rewards)
        next_raw = bootstrap_predictions(
            backend, learner, admitted, bootstrap_agent or learner.agent
        )
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
    backend: UpdateBackend,
    learner: Learner,
    batch: RolloutBatch,
    learning: DampedPolicyLearning,
    iteration: int,
    *,
    bootstrap_agent: Policy | None = None,
    learner_transitions: int | None = None,
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
        batch.action_data.check(
            batch.actions,
            batch.logprobs,
            validator=backend.rollout,
        )
        admitted = backend.rollout.AdmittedRollout(learner.agent, batch)
        actions = to_device(batch.actions, device)
    with track("Targets"):
        targets = compute_targets(backend, learner, admitted, learning, bootstrap_agent)
        advantage_abs_mean = float(targets.advantages.abs().mean())
    # Which rows the actor trains on, [T,E] on the host.
    actor_rows = to_host(targets.selected)
    rate, tau = learning.rates(iteration, learner_transitions=learner_transitions)
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
        "reference_forward_kl_coefficient": learning.reference_forward_kl,
        "schedule_clock": (
            "learner_transitions"
            if learning.decision_schedule is not None
            else "collection-update-iteration"
        ),
        "reference": learning.reference,
        "advantage_abs_mean": advantage_abs_mean,
        "bootstrapped_tail_fraction": float(
            np.mean(~batch.dones[-1], dtype=np.float32)
        ),
    }
    if learning.decision_schedule is not None:
        assert learner_transitions is not None
        diagnostics["schedule_progress"] = min(
            1.0, learner_transitions / learning.decision_schedule.transitions
        )
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
        admitted.admit_rows(
            np.flatnonzero(trained), min(width, learner.execution_rows or width)
        )
        for start in range(0, len(actor_rows), width):
            window = trained[start : start + width]
            if not window.any():
                health["skipped_steps"] += 1
                continue
            logical_rows = np.arange(start, start + len(window))[window]
            execution_rows = learner.execution_rows or len(logical_rows)
            split = len(logical_rows) > execution_rows
            actor_count = int(actor_rows[logical_rows].sum())
            totals = dict(
                loss=0.0,
                policy_loss=0.0,
                value_loss=0.0,
                entropy=0.0,
                collection_kl=0.0,
                reference_kl=0.0,
                reference_forward_kl=0.0,
                clip_fraction=0.0,
            )
            for offset in range(0, len(logical_rows), execution_rows):
                rows_host = logical_rows[offset : offset + execution_rows]
                with track("Forward and loss"):
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
                    # Actor and critic have different denominators under actor-only filtering.
                    policy_weight = int(actor_selected.sum()) / max(1, actor_count)
                    value_weight = len(rows_host) / len(logical_rows)

                    def _objective(obs: Observations = obs) -> tuple[Array, ...]:
                        with learner.forward_precision():
                            logits, value_logits = learner.agent.forward_distribution(
                                obs
                            )
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
                                reference_forward_kl=learning.reference_forward_kl,
                                policy_weight=policy_weight,
                                value_weight=value_weight,
                            ),
                            logits,
                            value_logits,
                        )

                    loss, policy_loss, value_loss, logits, value_logits = learner.loss(
                        _objective
                    )

                    def _failure(
                        logits: Array = logits,
                        value_logits: Array = value_logits,
                        obs: Observations = obs,
                    ) -> dict[str, Any]:
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
                    if split:
                        learner.accumulate(loss, first=offset == 0)
                    # Clip and Adam stay at the logical boundary, after all contributions.
                    if offset + execution_rows >= len(logical_rows):
                        norm = learner.step(
                            loss,
                            learning.max_grad_norm,
                            recorder=observer,
                            accumulated=split,
                        )
                with track("Minibatch diagnostics"), backend.learner.no_grad():
                    logs = rl.policy_logs(logits, valid)
                    totals["entropy"] += (
                        float(rl.entropy(logs, valid).mean()) * value_weight
                    )
                    totals["collection_kl"] += (
                        float(rl.reverse_kl(logs, behavior_logs, valid).mean())
                        * value_weight
                    )
                    magnet_logs = rl.reference_logs(reference, valid)
                    totals["reference_kl"] += (
                        float(rl.reverse_kl(logs, magnet_logs, valid).mean())
                        * value_weight
                    )
                    totals["reference_forward_kl"] += (
                        float(
                            rl.selected_mean(
                                rl.forward_kl(logs, magnet_logs, valid), actor_mask
                            )
                        )
                        * policy_weight
                    )
                    totals["clip_fraction"] += (
                        float(
                            rl.clip_fraction(
                                rl.action_logs(logs - behavior_logs, minibatch_actions),
                                learning.clip,
                            )
                        )
                        * value_weight
                    )
                    totals["loss"] += float(rl.constant(loss))
                    totals["value_loss"] += float(rl.constant(value_loss))
                    totals["policy_loss"] += float(rl.constant(policy_loss))
                del (
                    obs,
                    rows,
                    logits,
                    value_logits,
                    loss,
                    policy_loss,
                    value_loss,
                    logs,
                    _objective,
                    _failure,
                )
            health["clip_fraction"] = totals.pop("clip_fraction")
            diagnostics.update(totals, gradient_norm=float(norm))
            diagnostics["optimizer_exposures"] += len(logical_rows)
            diagnostics["actor_exposures"] += actor_count
            diagnostics["critic_exposures"] += len(logical_rows)
    return diagnostics
