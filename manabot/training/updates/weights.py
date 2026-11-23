"""Per-head coefficients for one logical minibatch, before execution chunking."""

import numpy as np

from manabot.training.rollout.collection import RolloutBatch
from manabot.training.updates.backend import RL, Array


def logical_weights(
    batch: RolloutBatch,
    rows: np.ndarray,
    selected: np.ndarray,
    root_actor_weight: float,
) -> np.ndarray | None:
    """Normalized [logical rows, actor/critic] weights, or ordinary count reductions.

    All rows already passed the critic's filter. Actors with empty support
    contribute a differentiable zero. Normalize on the host before transfer.
    """
    if batch.sample_weight is None and root_actor_weight == 1:
        return None
    critic = (
        np.ones(len(rows))
        if batch.sample_weight is None
        else batch.sample_weight.ravel()[rows].astype(np.float64)
    )
    actor = critic * selected[rows]
    if batch.is_root_decision is not None and root_actor_weight != 1:
        roots = batch.is_root_decision.ravel()[rows]
        if actor[roots].any() and actor[~roots].any():
            # Only the ratio matters. Scale down to keep finite config values finite.
            if root_actor_weight > 1:
                actor[~roots] /= root_actor_weight
            else:
                actor[roots] *= root_actor_weight
    actor_total = actor.sum()
    if actor_total:
        actor /= actor_total
    return np.column_stack((actor, critic / critic.sum())).astype(np.float32)


def actor_term(
    rl: RL, values: Array, mask: Array, share: float, weights: Array | None
) -> Array:
    """This execution chunk's contribution to the logical actor mean."""
    if weights is None:
        return rl.selected_mean(values, mask) * share
    return rl.weighted_sum(values, weights[:, 0])


def critic_term(rl: RL, values: Array, share: float, weights: Array | None) -> Array:
    """This execution chunk's contribution to the logical critic mean."""
    if weights is None:
        return values.mean() * share
    return rl.weighted_sum(values, weights[:, 1])
