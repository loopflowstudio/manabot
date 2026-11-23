"""Host observations and identity-preserving permutations for choice inspection."""

from collections.abc import Sequence

import numpy as np

from manabot.env.observation import (
    ObservationBuffers,
    ObservationSpace,
    encode_observation,
)
from manabot.policy.buckets import object_segments
import managym


def encode(
    raw: managym.Observation, space: ObservationSpace | None = None
) -> dict[str, np.ndarray]:
    if space is None:
        space = ObservationSpace()
    return {key: value[None] for key, value in encode_observation(space, raw).items()}


def reorder_offers(
    obs: ObservationBuffers,
    order: Sequence[int] | np.ndarray,
    layout: managym.ObservationLayout,
) -> dict[str, np.ndarray]:
    """Move each legal action with its mask, focus and naming candidates."""
    result = {key: value.copy() for key, value in obs.items()}
    for key, dimension in layout.axis_buffers["actions"]:
        offers = np.moveaxis(result[key], dimension + 1, 1)
        offers[:, : len(order)] = offers[:, order]
    return result


def reorder_objects(
    obs: ObservationBuffers, layout: managym.ObservationLayout
) -> dict[str, np.ndarray]:
    """Reverse each seat's card and permanent slots. Players and stack keep their order."""
    result = {key: value.copy() for key, value in obs.items()}
    capacities = {
        axis: obs[f"agent_{axis}"].shape[1] for axis in ("cards", "permanents")
    }
    capacities["stack"] = obs["stack"].shape[1] if "stack" in obs else 0
    # One-based lookup also maps the zero sentinel to itself. Stack targets
    # keep their coordinates even when a permanent points to them.
    references = np.arange(
        3 + 2 * (capacities["cards"] + capacities["permanents"]) + capacities["stack"]
    )
    for axis, start, rows in object_segments(capacities):
        if axis == "stack":
            break
        segment = slice(start + 1, start + rows + 1)
        references[segment] = references[segment][::-1]
    for axis, buffers in layout.axis_buffers.items():
        if axis in ("cards", "permanents"):
            for key, dimension in buffers:
                result[key][:] = np.flip(obs[key], axis=dimension + 1)

    for key, fields in layout.object_references.items():
        for column, base in fields:
            values = result[key] if column is None else result[key][..., column]
            offset = 1 - base
            values[:] = references[values.astype(np.intp) + offset] - offset
    return result
