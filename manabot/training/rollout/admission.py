"""Admit rollout rows once and prepare bounded host batches for either learner."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

from manabot.env.memory_store import select_rows
from manabot.policy.buckets import row_needs

if TYPE_CHECKING:
    from manabot.env.observation import ObservationBuffers, ObservationSpace
    from manabot.policy.prepare import Prepared
    from manabot.training.rollout.actions import ActionRows
    from manabot.training.rollout.collection import RolloutBatch


class RowPolicy(Protocol):
    bucketed_shapes: bool

    @property
    def observation_space(self) -> ObservationSpace: ...
    def check_observation(self, obs: ObservationBuffers) -> None: ...
    def prepare(
        self, obs: ObservationBuffers, *, inference: bool = False
    ) -> Prepared[np.ndarray]: ...
    def prepare_admitted(
        self, obs: ObservationBuffers, *, needs: np.ndarray | None = None
    ) -> Prepared[np.ndarray]: ...


class AdmittedRows:
    """Only rows validated for this policy may be prepared for a learner.

    The rollout stays immutable across epochs. Bucket needs come from collection
    when available, or are measured on admission.
    """

    def __init__(self, agent: RowPolicy, batch: RolloutBatch) -> None:
        self.agent = agent
        self.batch = batch
        self._admitted = np.zeros(batch.actions.size, dtype=bool)
        self._needs: np.ndarray | None = None

    def admit_rows(self, indexes: NDArray[np.int64], chunk_rows: int) -> None:
        pending = np.unique(indexes[~self._admitted[indexes]])
        for start in range(0, len(pending), chunk_rows):
            chunk = pending[start : start + chunk_rows]
            obs = self.batch.observations(chunk).observation
            self.agent.check_observation(obs)
            self._admitted[chunk] = True
            if not self.agent.bucketed_shapes:
                continue
            needs = self.batch.recorded.gather_needs(chunk)
            if needs is None:
                needs = row_needs(obs, self.agent.observation_space)
            if needs is None:
                continue
            if self._needs is None:
                # As in RecordedObservations, an unmeasured row has negative needs.
                self._needs = np.full(
                    (len(self._admitted), needs.shape[1]), -1, np.int64
                )
            self._needs[chunk] = needs

    def prepare_rows(
        self, indexes: NDArray[np.int64]
    ) -> tuple[Prepared[np.ndarray], ActionRows]:
        if not self._admitted[indexes].all():
            raise ValueError("rollout rows were not admitted")
        rows = self.batch.observations(indexes)
        needs = self._needs[indexes] if self._needs is not None else None
        if needs is not None and (needs < 0).any():
            needs = None
        return self.agent.prepare_admitted(rows.observation, needs=needs), rows

    def bootstrap_inputs(self, size: int) -> Iterator[Prepared[np.ndarray]]:
        if size < 1:
            raise ValueError("bootstrap rows must be positive")
        count = len(self.batch.next_done)
        for start in range(0, count, size):
            indexes = np.arange(start, min(start + size, count))
            yield self.agent.prepare(
                select_rows(self.batch.next_obs, indexes), inference=True
            )
