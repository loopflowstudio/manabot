"""Admit rollout rows once and prepare bounded host batches for either learner."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

from manabot.env.memory_store import PrefixObservation, select_rows
from manabot.policy.spans import ObservationSelection

if TYPE_CHECKING:
    from manabot.env.observation import ObservationBuffers
    from manabot.policy.prepare import Prepared
    from manabot.training.rollout.actions import ActionRows
    from manabot.training.rollout.collection import RolloutBatch


class RowPolicy(Protocol):
    def check_observation(
        self, obs: ObservationBuffers | ObservationSelection
    ) -> np.ndarray | None: ...
    def prepare(
        self, obs: ObservationBuffers | ObservationSelection, *, inference: bool = False
    ) -> Prepared[np.ndarray]: ...
    def prepare_admitted(
        self,
        obs: ObservationBuffers | ObservationSelection,
        *,
        needs: np.ndarray | None = None,
    ) -> Prepared[np.ndarray]: ...


class AdmittedRows:
    """Only rows validated for this policy may be prepared for a learner.

    Rows may change before admission and must remain immutable across epochs.
    Collection extents provide a floor for action rows dropped during packing.
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
            needs = self.agent.check_observation(obs)
            self._admitted[chunk] = True
            if needs is None:
                continue
            recorded = self.batch.recorded.gather_needs(chunk)
            if recorded is not None:
                needs = np.maximum(needs, recorded)
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
        prepared = self.agent.prepare_admitted(rows.observation, needs=needs)
        # Behavior/reference calculations need ordinary action fields, not a
        # second raw memory owner beside Prepared (including in the prefetch).
        if isinstance(rows.observation, PrefixObservation):
            raw = rows.observation
            rows.observation = PrefixObservation(
                raw, raw.memory.prefixes, raw.memory.capacities
            )
        return prepared, rows

    @contextmanager
    def prefetch_minibatches(
        self, order: NDArray[np.int64], size: int
    ) -> Iterator[Iterator[tuple[NDArray[np.int64], Prepared[np.ndarray], ActionRows]]]:
        """Prepare learner rows one batch ahead, independent of autograd mode.

        Only immutable host input is read on the worker. Transfers and all
        model/optimizer work stay on the caller's device stream. At most two
        host minibatches are live; exiting joins the worker even on failure.
        """
        if size < 1:
            raise ValueError("minibatch size must be positive")
        with ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ppo-prepare"
        ) as worker:

            def batches() -> Iterator[
                tuple[NDArray[np.int64], Prepared[np.ndarray], ActionRows]
            ]:
                if not len(order):
                    return
                pending = worker.submit(self.prepare_rows, order[:size])
                for start in range(0, len(order), size):
                    stop = start + size
                    ready = pending
                    if stop < len(order):
                        pending = worker.submit(
                            self.prepare_rows, order[stop : stop + size]
                        )
                    prepared, rows = ready.result()
                    yield order[start:stop], prepared, rows
                    del ready, prepared, rows

            with closing(batches()) as prepared_batches:
                yield prepared_batches

    def bootstrap_inputs(self, size: int) -> Iterator[Prepared[np.ndarray]]:
        if size < 1:
            raise ValueError("bootstrap rows must be positive")
        count = len(self.batch.next_done)
        for start in range(0, count, size):
            indexes = slice(start, start + size)
            yield self.agent.prepare(
                select_rows(self.batch.next_obs, indexes), inference=True
            )
