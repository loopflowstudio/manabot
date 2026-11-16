"""Numerical collection contract; native streams and stored rows stay on the host.

A session owns the private sampling RNG across frozen policy scopes. Its state
is opaque recovery data, retained in CollectorSnapshot at its existing path.
"""

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from manabot.env.observation import ObservationBuffers
from manabot.infra.profiler import Profiler
from manabot.policy.protocol import BackendPolicy
from manabot.training.rollout.actions import ActionValidator


@dataclass(frozen=True)
class SampledRows:
    actions: np.ndarray
    logprobs: np.ndarray
    values: np.ndarray
    log_probabilities: np.ndarray
    outcomes: np.ndarray | None
    needs: np.ndarray | None


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
