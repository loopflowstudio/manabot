"""Padded array views of recorded rollouts for matrix-level assertions."""

import numpy as np

from manabot.policy.spans import ObservationSelection
from manabot.training.rollout.actions import ActionRows
from manabot.training.rollout.collection import RolloutBatch


def projected_fields(rows: ObservationSelection) -> dict[str, np.ndarray]:
    return {key: rows.project(key) for key in rows.fields}


def trajectory_field(batch: RolloutBatch, key: str) -> np.ndarray:
    value = batch.observations(np.arange(batch.actions.size)).observation.project(key)
    return value.reshape(*batch.actions.shape, *value.shape[1:])


def trajectory_rows(batch: RolloutBatch) -> ActionRows:
    rows = batch.observations(np.arange(batch.actions.size))
    return ActionRows(
        {
            key: value.reshape(*batch.actions.shape, *value.shape[1:])
            for key, value in projected_fields(rows.observation).items()
        },
        rows.log_probabilities.reshape(*batch.actions.shape, -1),
    )
