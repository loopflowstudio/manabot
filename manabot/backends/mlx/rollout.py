"""Rollout admission for the updates: a collected batch's observations as the
MLX policy reads them.

The updates reach `AdmittedRollout` as `backend.rollout.AdmittedRollout`, with
the methods of the torch backend's. The MLX policy reads `Prepared` batches,
so "admitting" a row is preparing it on the host; nothing is kept between
minibatches.
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

    def admit_rows(self, indexes: NDArray[np.int64], chunk_rows: int) -> None:
        """Nothing to check ahead of time: `rows` prepares what it returns."""

    def next_observation(self, device: object) -> Prepared:
        """Bootstrap observations [trajectory, ...]."""
        return self.agent.prepare(self.batch.next_obs)

    def rows(
        self, indexes: NDArray[np.int64], device: object
    ) -> "tuple[Prepared, ActionRows]":
        """Prepared observations of flat rollout `indexes`, with their behavior
        distributions."""
        rows = self.batch.observations(indexes)
        return self.agent.prepare(rows.observation), rows

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
