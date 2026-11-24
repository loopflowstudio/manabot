"""Numerical collection contract; native streams and stored rows stay on the host.

A session owns the private sampling RNG across frozen policy scopes. Its state
is opaque recovery data, serialized by the framework.
"""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from manabot.env.memory_store import select_rows
from manabot.env.observation import ObservationBuffers
from manabot.infra.profiler import Profiler
from manabot.policy.prepare import Prepared
from manabot.policy.protocol import BackendPolicy
from manabot.policy.spans import ObservationSelection
from manabot.training.rollout.actions import ActionValidator


@dataclass(frozen=True)
class SampledRows:
    actions: np.ndarray
    logprobs: np.ndarray
    values: np.ndarray
    log_probabilities: np.ndarray
    outcomes: np.ndarray | None
    needs: np.ndarray | None


def prepared_chunks[A](
    observation: ObservationBuffers,
    execution_rows: int | None,
    prepare: Callable[[ObservationBuffers], Prepared[A]],
    profiler: Profiler,
) -> Iterator[Prepared[A]]:
    """Prepare borrowed rows in order. The caller samples once across all chunks."""
    count = (
        len(observation.coordinates)
        if isinstance(observation, ObservationSelection)
        else len(observation["actions_valid"])
    )
    width = count if execution_rows is None else execution_rows
    if width < 1:
        raise ValueError("execution rows must be positive")
    for start in range(0, count, width):
        with profiler.track("Observation preparation"):
            rows = (
                observation
                if width >= count
                else select_rows(observation, slice(start, start + width))
            )
            prepared = prepare(rows)
        yield prepared
        # Release the generator's reference before preparing the next chunk.
        del prepared


class CollectionScope(Protocol):
    def sample(
        self,
        observation: ObservationBuffers,
        profiler: Profiler,
        *,
        execution_rows: int | None = None,
    ) -> SampledRows: ...
    def opponent_actions(
        self, observation: ObservationBuffers, *, execution_rows: int | None = None
    ) -> np.ndarray: ...


class CollectionSession(ActionValidator, Protocol):
    @property
    def device_name(self) -> str: ...
    def scope(
        self, agent: BackendPolicy
    ) -> AbstractContextManager[CollectionScope]: ...
    def snapshot_rng(self) -> object: ...
    def restore_rng(self, state: object) -> None: ...
