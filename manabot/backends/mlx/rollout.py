"""Rollout admission for the updates: a collected batch's observations as the
MLX policy reads them.

The updates reach `AdmittedRollout` as `backend.rollout.AdmittedRollout`, with
the methods of the torch backend's. The MLX policy reads `Prepared` batches,
and uses the same host admission checks as torch before preparing indexes.
Admission is recorded per row, so later epochs do not repeat the checks.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import fields
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray

from manabot.backends.mlx.policy import MLXActorCritic
from manabot.backends.mlx.rl import NumericalError, check_behavior, legal_mask, require
from manabot.policy.prepare import Prepared

if TYPE_CHECKING:
    from manabot.training.rollout.actions import ActionRows, PackedActions
    from manabot.training.rollout.collection import RolloutBatch


class AdmittedRollout:
    """A rollout batch whose observations one MLX policy reads. The batch must
    not change afterwards."""

    def __init__(self, agent: MLXActorCritic, batch: "RolloutBatch") -> None:
        self.agent = agent
        self.batch = batch
        self._admitted = np.zeros(batch.actions.size, dtype=bool)
        self._next = agent.prepare(batch.next_obs) if len(batch.next_done) else None

    def admit_rows(self, indexes: NDArray[np.int64], chunk_rows: int) -> None:
        """Validate learner rows in bounded chunks, once across all epochs."""
        pending = np.unique(indexes[~self._admitted[indexes]])
        for start in range(0, len(pending), chunk_rows):
            chunk = pending[start : start + chunk_rows]
            self.agent.check_observation(self.batch.observations(chunk).observation)
            self._admitted[chunk] = True

    def next_observation(self, device: object) -> Prepared | None:
        """Bootstrap observations [trajectory, ...]."""
        return self._next

    def next_observations(self, device: object, size: int) -> Iterator[Prepared]:
        # Explicit limits are refused by the MLX learner before this path.
        if self._next is not None:
            yield self._next

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


def packing_mask(
    mask: NDArray[np.generic], logs: NDArray[np.generic]
) -> NDArray[np.bool_]:
    """Validate legal support and padding before the host drops padding rows."""
    valid = np.array(legal_mask(mx.array(mask)))
    if valid.ndim != 2 or logs.shape != valid.shape:
        raise ValueError("packed actions require matching [B,A] distributions")
    if np.any(~np.isneginf(logs[~valid])):
        require(
            mx.array(np.isneginf(logs) | valid),
            "illegal_log_mass",
            logs=mx.array(logs),
            valid=mx.array(valid),
        )
    counts = valid.sum(-1)
    if np.any(counts == 0):
        require(
            mx.array(counts != 0),
            "empty_legal_support",
            valid=mx.array(valid),
        )
    return valid


def check_packed(
    packed: PackedActions,
    actions: NDArray[np.generic],
    selected_logs: NDArray[np.generic],
) -> None:
    """Admit all behavior evidence in bounded CPU chunks with original row ids."""
    if actions.size != len(packed.offsets) - 1 or selected_logs.shape != actions.shape:
        raise NumericalError(
            "behavior_shape",
            {
                "actions": mx.array(actions),
                "selected_logs": mx.array(selected_logs),
                "offsets": mx.array(packed.offsets),
            },
        )
    for start in range(0, len(packed.offsets) - 1, 32):
        indexes = np.arange(start, min(start + 32, len(packed.offsets) - 1))
        rows = packed.unpack(indexes)
        try:
            check_behavior(
                mx.array(rows.log_probabilities),
                mx.array(actions.reshape(-1)[indexes]),
                mx.array(selected_logs.reshape(-1)[indexes]),
                mx.array(rows.observation["actions_valid"]).astype(mx.bool_),
            )
        except NumericalError as error:
            # Preserve original rollout coordinates in incident evidence.
            error.tensors["minibatch_indexes"] = mx.array(indexes)
            raise
