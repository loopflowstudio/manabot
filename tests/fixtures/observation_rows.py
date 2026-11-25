"""Padded per-field views of one encoded observation, for inspecting the encoder.

The arrays are read from the native batch through `ObservationSelection.project`,
so a test reads exactly the columns a policy is prepared from.
"""

from collections.abc import Sequence

import numpy as np

from manabot.env.observation import ObservationSpace, encode_policy_batch
from manabot.policy.spans import ObservationSelection
import managym


def observation_arrays(
    space: ObservationSpace, obs: managym.Observation
) -> dict[str, np.ndarray]:
    """Every field of `obs` in `space`, without a batch axis."""
    batch = encode_policy_batch(space, [obs])
    selection = ObservationSelection(space, ((batch, 0),))
    return {key: selection.project(key)[0] for key in selection.fields}


def vector_arrays(
    space: ObservationSpace,
    rows: Sequence[tuple[managym.ObservationBatch, int]],
) -> dict[str, np.ndarray]:
    """Every field of a vector's current rows (`observation_rows()`), batch axis first."""
    selection = ObservationSelection(space, tuple(rows))
    return {key: selection.project(key) for key in selection.fields}
