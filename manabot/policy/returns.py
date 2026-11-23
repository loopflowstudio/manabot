"""Returns and advantage selection over host NumPy rows.

The recursions here run on the host in every backend: a complete-game rollout
is one packed column of tens of thousands of rows whose trajectories advance
together (`reverse_waves`), which is launch latency on a device, not work. A
backend's `transition_gae` wraps them (`manabot.backends.torch.rl`,
`manabot.backends.mlx.rl`), and the update diagnostics in
`manabot.training.updates.selection` read them directly. Torch's categorical
targets walk `reverse_waves` over host tensors, and its device-side advantage
selection (`torch.quantile`) matches `quantile` bit for bit.

Rows are [T,E]: T same-viewer transitions of E packed columns. An end flag
belongs to its own row's reward. A terminal ends both bootstrap and trace.
A horizon cut ends the trace with a frozen value from that viewer.
"""

from collections.abc import Iterator
import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray


def terminal_distances(
    ends: NDArray[np.bool_], segment_ends: NDArray[np.bool_] | None = None
) -> NDArray[np.int64]:
    """[T,E] distances to the next end (terminal=0, censored=-1), reset at each end."""
    count = len(ends)
    steps = np.arange(count)[:, None]
    # The nearest end at or after each row, by a running minimum toward the start.
    boundaries = ends if segment_ends is None else ends | segment_ends
    following = np.minimum.accumulate(np.where(boundaries, steps, count)[::-1], axis=0)[
        ::-1
    ]
    observed = following < count
    if segment_ends is not None:
        observed &= np.take_along_axis(ends, following.clip(max=count - 1), axis=0)
    return np.where(observed, following - steps, -1)


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


def bootstrap_cuts(
    ends: NDArray[np.bool_], indexes: NDArray[np.intp], count: int
) -> NDArray[np.bool_]:
    """Validate explicit cut coordinates and return their [T,E] mask."""
    if indexes.shape != ends.shape or not np.issubdtype(
        indexes.dtype, np.signedinteger
    ):
        raise ValueError("bootstrap indexes must be signed integers shaped like ends")
    cuts = indexes >= 0
    if np.any(indexes < -1) or np.any(indexes >= count):
        raise ValueError("bootstrap index outside frozen observations")
    if (ends & cuts).any() or not (ends[-1] | cuts[-1]).all():
        raise ValueError("each trajectory must finish at a terminal or a bootstrap cut")
    return cuts


def transition_gae(
    rewards: NDArray[np.floating],
    values: NDArray[np.floating],
    ends: NDArray[np.bool_],
    next_value: NDArray[np.floating],
    gamma: float,
    lam: float,
    bootstrap_indexes: NDArray[np.intp] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """GAE (Schulman et al. 2016): advantages and lambda returns, both [T,E].

    `rewards`, `values` and `ends` are [T,E]; terminal rewards credit their own
    episode. `next_value` [E] is read only where a column's last row is not an
    end. With explicit cuts, nonnegative entries of `bootstrap_indexes` [T,E]
    address the corresponding frozen viewer values in `next_value` [M].
    """
    if (
        rewards.ndim != 2
        or values.shape != rewards.shape
        or ends.shape != rewards.shape
        or not len(rewards)
        or (bootstrap_indexes is not None and next_value.ndim != 1)
        or (bootstrap_indexes is None and next_value.size != rewards.shape[1])
    ):
        raise ValueError(
            "transition_gae requires rewards, values and ends [T,E] and next_value [E]; "
            f"got {rewards.shape}, {values.shape}, {ends.shape} and {next_value.shape}"
        )
    ends = ends.astype(bool)
    live = (~ends).astype(rewards.dtype)
    if bootstrap_indexes is None:
        boundaries = ends
        following = np.concatenate([values[1:], next_value.reshape(1, -1)])
    else:
        cuts = bootstrap_cuts(ends, bootstrap_indexes, len(next_value))
        boundaries = ends | cuts
        following = np.concatenate([values[1:], np.zeros_like(values[:1])])
        following[cuts] = next_value[bootstrap_indexes[cuts]]
    deltas = rewards + gamma * live * following - values
    carried = gamma * lam * (~boundaries).astype(rewards.dtype)
    # The extra zero row is the empty tail past each column's last row.
    advantages = np.zeros((len(rewards) + 1, rewards.shape[1]), rewards.dtype)
    for steps, columns in reverse_waves(boundaries):
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
