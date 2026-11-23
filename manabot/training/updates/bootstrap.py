"""Ordered critic predictions with bounded bootstrap preparation lifetimes.

Both learning rules use this path; only small predictions survive each chunk.
The caller owns the no-gradient scope and the learner owns forward precision.
"""

import numpy as np

from manabot.training.updates.backend import (
    AdmittedRollout,
    Array,
    Learner,
    Policy,
    UpdateBackend,
)


def bootstrap_predictions(
    backend: UpdateBackend, learner: Learner, admitted: AdmittedRollout, policy: Policy
) -> Array:
    """Preserve trajectory order; only raw critic predictions survive each chunk."""
    if not len(admitted.batch.next_done):
        width = 3 if policy.config.value_kind == "categorical_wdl" else 1
        return backend.arrays.from_numpy(
            np.empty((0, width), np.float32), learner.device
        )
    outputs: list[np.ndarray] = []
    size = learner.bootstrap_rows or len(admitted.batch.next_done)
    for obs in admitted.next_observations(learner.device, size):
        with learner.forward_precision():
            _, values = policy.forward_distribution(obs)
        outputs.append(backend.arrays.to_numpy(backend.arrays.float32(values)))
        del obs, values
    return backend.arrays.from_numpy(np.concatenate(outputs), learner.device)
