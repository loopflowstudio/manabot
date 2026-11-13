"""Rollout admission for the updates: a collected batch's observations as the
MLX policy reads them.

The updates reach `AdmittedRollout` as `backend.rollout.AdmittedRollout`, with
the methods of the torch backend's. The MLX policy reads `Prepared` batches,
and uses the same host admission checks as torch before preparing indexes.
Admission is recorded per row, so later epochs do not repeat the checks.
"""

from dataclasses import fields
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray

from manabot.backends.mlx.policy import MLXActorCritic
from manabot.policy.prepare import Prepared

if TYPE_CHECKING:
    from manabot.training.rollout.actions import ActionRows
    from manabot.training.rollout.collection import RolloutBatch


class AdmittedRollout:
    """A rollout batch whose observations one MLX policy reads. The batch must
    not change afterwards."""

    def __init__(self, agent: MLXActorCritic, batch: "RolloutBatch") -> None:
        self.agent = agent
        self.batch = batch
        self._admitted = np.zeros(batch.actions.size, dtype=bool)
        self._next = agent.prepare(batch.next_obs)

    def admit_rows(self, indexes: NDArray[np.int64], chunk_rows: int) -> None:
        """Validate learner rows in bounded chunks, once across all epochs."""
        pending = np.unique(indexes[~self._admitted[indexes]])
        for start in range(0, len(pending), chunk_rows):
            chunk = pending[start : start + chunk_rows]
            self.agent.check_observation(self.batch.observations(chunk).observation)
            self._admitted[chunk] = True

    def next_observation(self, device: object) -> Prepared:
        """Bootstrap observations [trajectory, ...]."""
        return self._next

    def rows(
        self, indexes: NDArray[np.int64], device: object
    ) -> "tuple[Prepared, ActionRows]":
        """Prepared observations of flat rollout `indexes`, with their behavior
        distributions."""
        if not self._admitted[indexes].all():
            raise ValueError("rollout rows were not admitted")
        rows = self.batch.observations(indexes)
        return self.agent.prepare_admitted(rows.observation), rows

    def legal(self, obs: Prepared) -> mx.array:
        """The legal-action mask [B,A] of rows this rollout returned."""
        return mx.array(obs.legal)

    def evidence(self, obs: Prepared) -> dict[str, np.ndarray]:
        """The arrays a rejected step retains for rows this rollout returned."""
        return {
            item.name: value
            for item in fields(obs)
            if isinstance(value := getattr(obs, item.name), np.ndarray)
        }
