"""Returns and advantage selection over host NumPy rows.

The recursions here run on the host in every backend: a complete-game rollout
is one packed column of tens of thousands of rows whose trajectories advance
together (`reverse_waves`), which is launch latency on a device, not work. A
backend whose framework has no better form wraps these (`manabot.backends.mlx.rl`);
the update diagnostics in `manabot.training.updates.selection` read them
directly. The torch backend keeps the same algorithms over host tensors
(`manabot.backends.torch.rl`), and these match it bit for bit, `quantile`
included.

Rows are [T,E]: T same-viewer transitions of E packed columns. An end flag
belongs to its own row's reward and cuts both bootstrap and trace.
"""

from collections.abc import Iterator
import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray


def terminal_distances(ends: NDArray[np.bool_]) -> NDArray[np.int64]:
    """[T,E] distances to the next end (terminal=0, censored=-1), reset at each end."""
    count = len(ends)
    steps = np.arange(count)[:, None]
    # The nearest end at or after each row, by a running minimum toward the start.
    following = np.minimum.accumulate(np.where(ends, steps, count)[::-1], axis=0)[::-1]
    return np.where(following < count, following - steps, -1)


def reverse_waves(
    ends: NDArray[np.bool_],
) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
    """Yield (time, column) rows of [T,E] flags, each after the row it reads.

    A trajectory ends at an end flag or a column's last row. Wave k holds the
    row k from the end of every trajectory with more than k rows.
    """
    last = ends.copy()
    last[-1] = True
    waves = terminal_distances(last).ravel()
    steps, columns = np.unravel_index(np.argsort(waves, kind="stable"), ends.shape)
    bounds = np.cumsum(np.bincount(waves))[:-1]
    return zip(np.split(steps, bounds), np.split(columns, bounds), strict=True)


def transition_gae(
    rewards: NDArray[np.floating],
    values: NDArray[np.floating],
    ends: NDArray[np.bool_],
    next_value: NDArray[np.floating],
    gamma: float,
    lam: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """GAE advantages and returns [T,E] for transition-end flags; terminal
    rewards credit their own episode. `next_value` [E] bootstraps the tail."""
    ends = ends.astype(bool)
    live = (~ends).astype(rewards.dtype)
    following = np.concatenate([values[1:], next_value.reshape(1, -1)])
    deltas = rewards + gamma * live * following - values
    carried = gamma * lam * live
    # The extra zero row is the empty tail past each column's last row.
    advantages = np.zeros((len(rewards) + 1, rewards.shape[1]), rewards.dtype)
    for steps, columns in reverse_waves(ends):
        advantages[steps, columns] = (
            deltas[steps, columns]
            + carried[steps, columns] * advantages[steps + 1, columns]
        )
    return advantages[:-1], advantages[:-1] + values


def quantile(values: NDArray[np.float32], q: float) -> float:
    """The linearly interpolated `q` quantile of float32 `values`, bit for bit
    as `torch.quantile` computes it on CPU.

    Torch holds `q` in float32, takes the rank in float64 and interpolates in
    float32 with one fused multiply-add from the nearer end: `low + w * span`
    below one half, `high + (w - 1) * span` above. The float64 products here
    are exact, so their single rounding is that fused operation.
    """
    ordered = np.sort(values.astype(np.float32, copy=False), axis=None)
    if not len(ordered):
        raise ValueError("quantile of no rows")
    rank = np.float64(np.float32(q)) * (len(ordered) - 1)
    below = int(np.floor(rank))
    weight = np.float32(rank - below)
    low, high = ordered[below], ordered[min(below + 1, len(ordered) - 1)]
    span = np.float64(high - low)
    if weight < 0.5:
        return float(np.float32(np.float64(low) + np.float64(weight) * span))
    return float(
        np.float32(np.float64(high) + np.float64(weight - np.float32(1)) * span)
    )


def magnitude_quantile(advantages: NDArray[np.float32], q: float) -> float:
    """The inclusive `q` quantile of |advantages| over every row."""
    return quantile(np.abs(advantages), q)


def selection_mask(
    advantages: NDArray[np.float32],
    kind: Literal["top_count", "quantile"],
    fraction: float,
    minimum: float,
) -> NDArray[np.bool_]:
    """Boolean mask shaped like `advantages` keeping `fraction` of rows by
    magnitude, |A| >= minimum. A quantile filter keeps every tie at its
    inclusive threshold; a top-count filter keeps a fixed share, stable ties."""
    magnitude = np.abs(advantages)
    if kind == "quantile":
        return magnitude >= max(quantile(magnitude, 1 - fraction), minimum)
    flat = magnitude.ravel()
    count = max(1, math.ceil(len(flat) * fraction))
    kept = np.argsort(-flat, kind="stable")[:count]
    mask = np.zeros(len(flat), bool)
    mask[kept[flat[kept] >= minimum]] = True
    return mask.reshape(advantages.shape)
