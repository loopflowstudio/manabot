"""Per-decision actor and critic loss weights across logical minibatches and execution chunks."""

import numpy as np

from manabot.training.rollout.collection import RolloutBatch
from manabot.training.updates.backend import RL, Array, Arrays, Device


def batch_weights(
    batch: RolloutBatch,
    training_rows: np.ndarray,
    actor_rows: np.ndarray,
    root_actor_weight: float,
) -> np.ndarray | None:
    """Scale each head's participating weights to mean one across the update.

    Actor selection has already excluded zero-weight roots. A common scale
    cancels in the mean, so even large root multipliers need no large product.
    """
    if batch.sample_weight is None and root_actor_weight == 1:
        return None
    critic = (
        np.ones(batch.actions.size)
        if batch.sample_weight is None
        else batch.sample_weight.ravel().astype(np.float64)
    )
    actor = np.where(actor_rows, critic, 0)
    if actor_rows.any():
        if batch.is_root_decision is not None and root_actor_weight != 1:
            roots = batch.is_root_decision.ravel()
            if roots[actor_rows].any() and not roots[actor_rows].all():
                if root_actor_weight > 1:
                    actor[~roots] /= root_actor_weight
                else:
                    actor[roots] *= root_actor_weight
        actor /= actor[actor_rows].mean()
    critic /= critic[training_rows].mean()
    return np.column_stack((actor, critic))


def logical_weights(
    normalized: np.ndarray | None,
    rows: np.ndarray,
    actor_count: int,
    arrays: Arrays,
    device: Device,
) -> Array | None:
    """Transfer the two coefficients per trained row once, before execution splits."""
    if normalized is None:
        return None
    coefficients = normalized[rows] / (max(actor_count, 1), len(rows))
    return arrays.from_numpy(coefficients.astype(np.float32), device)


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
