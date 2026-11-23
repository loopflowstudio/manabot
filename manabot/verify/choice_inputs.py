"""Host observations and identity-preserving permutations for choice inspection."""

from collections.abc import Sequence

import numpy as np

from manabot.env.observation import (
    ObservationBuffers,
    ObservationSpace,
    encode_observation,
)
from manabot.policy.buckets import (
    AXIS_BUFFERS,
    SEAT_MAJOR,
    object_reference_columns,
    object_segments,
)
import managym


def encode(
    raw: managym.Observation, space: ObservationSpace | None = None
) -> dict[str, np.ndarray]:
    if space is None:
        space = ObservationSpace()
    return {key: value[None] for key, value in encode_observation(space, raw).items()}


def reorder_offers(
    obs: ObservationBuffers, order: Sequence[int] | np.ndarray
) -> dict[str, np.ndarray]:
    """Move each legal action with its mask, focus and naming candidates."""
    result = {key: value.copy() for key, value in obs.items()}
    for key in AXIS_BUFFERS["actions"]:
        result[key][:, : len(order)] = obs[key][:, order]
    return result


def reorder_objects(obs: ObservationBuffers) -> dict[str, np.ndarray]:
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
    for axis in ("cards", "permanents"):
        for key in AXIS_BUFFERS[axis]:
            if key in SEAT_MAJOR:
                if key in obs:
                    result[key][:] = obs[key][:, :, ::-1]
            else:
                result[key][:] = obs[key][:, ::-1]

    focus = result["action_focus"]
    focus[:] = references[focus.astype(np.intp) + 1] - 1
    for key, columns, base in object_reference_columns(result):
        for col in columns:
            values = result[key][..., col]
            values[:] = references[values.astype(np.intp) + 1 - base] - (1 - base)
    return result
