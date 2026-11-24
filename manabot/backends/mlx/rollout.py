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

from manabot.backends.mlx.rl import NumericalError, check_behavior, legal_mask, require
from manabot.policy.prepare import Prepared
from manabot.training.rollout.admission import AdmittedRows

if TYPE_CHECKING:
    from manabot.training.rollout.actions import ActionRows, PackedActions


def observation_arrays(obs: Prepared[np.ndarray]) -> dict[str, np.ndarray]:
    """Top-level host arrays retained as numerical incident evidence."""
    return {
        item.name: value
        for item in fields(obs)
        if isinstance(value := getattr(obs, item.name), np.ndarray)
    }


class AdmittedRollout(AdmittedRows):
    """A rollout batch whose observations one MLX policy reads. The batch must
    not change afterwards."""

    def next_observation(self, device: object) -> Prepared | None:
        """Inference-prepared bootstrap observations [trajectory, ...]."""
        return next(
            self.next_observations(device, max(1, len(self.batch.next_done))), None
        )

    def next_observations(self, device: object, size: int) -> Iterator[Prepared]:
        """Admit and prepare one bootstrap chunk at a time."""
        return self.bootstrap_inputs(size)

    def rows(
        self, indexes: NDArray[np.int64], device: object
    ) -> "tuple[Prepared, ActionRows]":
        """Prepared observations of flat rollout `indexes`, with their behavior
        distributions."""
        return self.prepare_rows(indexes)

    def legal(self, obs: Prepared) -> mx.array:
        """The legal-action mask [B,A] of rows this rollout returned."""
        return mx.array(obs.legal)

    evidence = staticmethod(observation_arrays)


def packing_mask(
    mask: NDArray[np.generic], logs: NDArray[np.generic]
) -> NDArray[np.bool_]:
    """Validate legal support and padding before the host drops padding rows."""
    valid = mask if mask.dtype == np.bool_ else np.array(legal_mask(mx.array(mask)))
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
        rows = packed.behavior(indexes)
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
