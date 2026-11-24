"""Frozen on-policy targets shared by the PPO and damped updates.

Credit follows same-viewer trajectories. Independent continuations of one
decision supply a leave-one-out actor baseline (Kool et al., 2019) and a shared critic
target at that decision. Selection and scoring use these detached targets
before any optimizer step.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from manabot.training.learning import DampedPolicyLearning, Learning
from manabot.training.rollout.collection import RolloutBatch, TrajectorySpan
from manabot.training.rollout.signals import BatchSignals
from manabot.training.updates.backend import (
    AdmittedRollout,
    Array,
    Learner,
    Policy,
    UpdateBackend,
)
from manabot.training.updates.bootstrap import bootstrap_predictions
from manabot.training.updates.selection import SourceSelection, select_sources


@dataclass(frozen=True)
class BatchTargets:
    """Detached targets and selection for one collected batch and learning rule."""

    batch: RolloutBatch
    learning: Learning | DampedPolicyLearning
    advantages: Array
    value_targets: Array
    expected_targets: Array
    residuals: Array
    positive_weight: np.ndarray
    selected: Array
    retained: np.ndarray
    source_selection: tuple[SourceSelection, ...]
    signals: BatchSignals

    @property
    def empty_filter_reason(self) -> str:
        """Discovery skips EMA too. An ordinary empty actor filter still advances it."""
        return (
            "empty advantage filter"
            if self.positive_weight.any()
            else "zero sample weight"
        )

    def check(
        self, batch: RolloutBatch, learning: Learning | DampedPolicyLearning
    ) -> None:
        if self.batch is not batch or self.learning != learning:
            raise ValueError(
                "precomputed targets require the same rollout batch and learning rule"
            )


def grouped_roots(batch: RolloutBatch) -> list[list[TrajectorySpan]]:
    ids = batch.branch_group_id
    if ids is None or not (ids >= 0).any():
        return []
    if (
        batch.trajectories is None
        or batch.is_root_decision is None
        or batch.snapshot_id is None
    ):
        raise ValueError(
            "branch groups require explicit trajectories, root markers and snapshot IDs"
        )
    groups: dict[int, list[TrajectorySpan]] = defaultdict(list)
    for span in batch.trajectories:
        group = int(ids[span.start, 0])
        if group >= 0:
            roots = groups[group]
            if batch.is_root_decision[span.start, 0]:
                roots.append(span)
    for group, roots in groups.items():
        if not roots:
            raise ValueError(f"branch group {group} has no root")
        identities = {(int(batch.snapshot_id[s.start, 0]), s.seat) for s in roots}
        if len(identities) != 1 or batch.snapshot_id[roots[0].start, 0] < 0:
            raise ValueError(
                f"branch group {group} roots must share a snapshot ID and seat"
            )
    return [roots for roots in groups.values() if len(roots) > 1]


def replace_root_targets(
    groups: list[list[TrajectorySpan]],
    rewards: np.ndarray,
    bootstrap: np.ndarray,
    gamma: float,
    advantages: np.ndarray,
    targets: np.ndarray,
) -> None:
    """Write RLOO advantages and sibling-mean targets into root rows in place."""
    categorical = targets.ndim == 3
    for roots in groups:
        outcomes = np.empty((len(roots), 3 if categorical else 1), np.float64)
        for outcome, span in zip(outcomes, roots, strict=True):
            if categorical:
                if span.bootstrap_index is None:
                    outcome.fill(0)
                    outcome[int(rewards[span.stop - 1, 0]) + 1] = 1
                else:
                    outcome[:] = bootstrap[span.bootstrap_index]
            else:
                value = (
                    0.0
                    if span.bootstrap_index is None
                    else float(bootstrap[span.bootstrap_index])
                )
                path = rewards[span.start : span.stop, 0]
                if gamma == 1:
                    value += path.sum(dtype=np.float64)
                else:
                    for reward in path[::-1]:
                        value = float(reward) + gamma * value
                outcome[0] = value
        rows = [s.start for s in roots]
        returns = outcomes[:, 2] - outcomes[:, 0] if categorical else outcomes[:, 0]
        advantages[rows, 0] = returns - (returns.sum() - returns) / (len(rows) - 1)
        targets[rows, 0] = outcomes.mean(axis=0)


def batch_signals(
    batch: RolloutBatch, residuals: np.ndarray, action_values: np.ndarray | None
) -> BatchSignals:
    """Score collection distributions; optional action values align with packed legal offers."""
    residuals = residuals.ravel()
    starts = batch.action_data.offsets[:-1]
    logs = batch.action_data.log_probabilities
    policy_entropy = -np.add.reduceat(np.exp(logs) * logs, starts)
    probabilities = batch.outcome_probabilities
    entropy = None
    if probabilities is not None:
        logs_wdl = np.zeros_like(probabilities)
        np.log(probabilities, out=logs_wdl, where=probabilities > 0)
        entropy = -(probabilities * logs_wdl).sum(axis=-1).ravel()
    spread = None
    if action_values is not None:
        if action_values.shape != logs.shape or not np.isfinite(action_values).all():
            raise ValueError(
                "action_values must be finite and aligned with packed legal offers"
            )
        action_values = action_values.astype(np.float64, copy=False)
        spread = np.maximum.reduceat(action_values, starts) - np.minimum.reduceat(
            action_values, starts
        )
    absolute, positive = np.abs(residuals), np.maximum(residuals, 0)
    for array in (absolute, positive, policy_entropy, entropy, spread):
        if array is not None:
            array.setflags(write=False)
    return BatchSignals(absolute, positive, policy_entropy, entropy, spread)


def compute_batch_targets(
    backend: UpdateBackend,
    learner: Learner,
    admitted: AdmittedRollout,
    learning: DampedPolicyLearning | Learning,
    bootstrap_agent: Policy | None = None,
    *,
    action_values: np.ndarray | None = None,
) -> BatchTargets:
    """Freeze actor/critic targets and scoring signals before learning on this rollout."""
    rl, device = backend.rl, learner.device
    to_device, to_host = backend.arrays.from_numpy, backend.arrays.to_numpy
    batch = admitted.batch
    if isinstance(learning, Learning):
        gamma, fraction = learning.gamma, learning.retained_fraction
    else:
        gamma, fraction = 1.0, 1 - learning.advantage_quantile
    cuts = batch.bootstrap_indexes()
    with backend.learner.no_grad():
        rewards = to_device(batch.rewards, device)
        ends = to_device(batch.dones, device)
        values = to_device(batch.values, device)
        rl.require_finite("nonfinite_collection_values", values=values)
        rl.require_finite("nonfinite_rewards", rewards=rewards)
        raw = bootstrap_predictions(
            backend, learner, admitted, bootstrap_agent or learner.agent
        )
        rl.require_finite("nonfinite_bootstrap", values=raw)
        categorical = (
            isinstance(learning, DampedPolicyLearning)
            and learner.agent.config.value_kind == "categorical_wdl"
        )
        if raw.shape[-1] != (3 if categorical else 1):
            raise ValueError("bootstrap critic width differs from the learning rule")
        if categorical:
            outcomes = batch.outcome_probabilities
            if outcomes is None:
                raise ValueError(
                    "categorical update requires collection outcome probabilities"
                )
            if not np.allclose(
                batch.values, outcomes[..., 2] - outcomes[..., 0], atol=1e-5
            ):
                raise ValueError(
                    "saved value differs from collection outcome probabilities"
                )
            bootstrap = rl.outcome_probabilities(raw)
            value_targets = rl.categorical_lambda_returns(
                to_device(outcomes, device),
                rewards,
                ends,
                bootstrap,
                learning.value_lambda,
                cuts,
            )
            next_value = rl.expected_outcome(bootstrap)
        else:
            bootstrap = next_value = raw.squeeze(-1)
            _, value_targets = rl.transition_gae(
                rewards, values, ends, next_value, gamma, learning.value_lambda, cuts
            )
        advantages, _ = rl.transition_gae(
            rewards, values, ends, next_value, gamma, learning.policy_lambda, cuts
        )
        groups = grouped_roots(batch)
        if groups:
            host_advantages = to_host(advantages).copy()
            host_targets = to_host(value_targets).copy()
            replace_root_targets(
                groups,
                batch.rewards,
                to_host(bootstrap),
                gamma,
                host_advantages,
                host_targets,
            )
            advantages = to_device(host_advantages, device)
            value_targets = to_device(host_targets, device)
        rl.require_finite("nonfinite_advantages", advantages=advantages)
        rl.require_finite("nonfinite_value_targets", targets=value_targets)
        expected = rl.expected_outcome(value_targets) if categorical else value_targets
        residuals = expected - values
        positive = (
            np.ones(batch.actions.shape, bool)
            if batch.sample_weight is None
            else batch.sample_weight > 0
        )
        eligible = positive
        if learning.root_actor_weight == 0 and batch.is_root_decision is not None:
            eligible = positive & ~batch.is_root_decision
        retained, sources = select_sources(
            backend,
            device,
            advantages,
            eligible,
            batch.episode_source,
            learning.filter_kind,
            fraction,
            learning.min_advantage,
        )
        selected = np.zeros(batch.actions.shape, bool)
        selected.flat[retained] = True
        return BatchTargets(
            batch=batch,
            learning=learning.model_copy(deep=True),
            advantages=advantages,
            value_targets=value_targets,
            expected_targets=expected,
            residuals=residuals,
            positive_weight=positive,
            selected=to_device(selected, device),
            retained=retained,
            source_selection=sources,
            signals=batch_signals(batch, to_host(residuals), action_values),
        )
