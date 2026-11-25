"""Mutable array fixtures for matrix-level assertions and deliberately invalid inputs."""

import numpy as np

from manabot.env.observation import ObservationBuffers
from manabot.policy.spans import ObservationSelection
from manabot.training.rollout.actions import ActionRows
from manabot.training.rollout.collection import RolloutBatch


def projected_fields(
    rows: ObservationSelection | ObservationBuffers,
) -> dict[str, np.ndarray]:
    if isinstance(rows, ObservationSelection):
        return {key: rows.project(key) for key in rows.fields}
    return dict(rows)


def trajectory_field(batch: RolloutBatch, key: str) -> np.ndarray:
    rows = batch.observations(np.arange(batch.actions.size)).observation
    value = rows.project(key) if isinstance(rows, ObservationSelection) else rows[key]
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
