"""Belief sampling capabilities independent of training/run orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from manabot.policy.protocol import BackendPolicy

if TYPE_CHECKING:
    from manabot.belief.encoding import BeliefEncodingSchema
    from manabot.belief.sampling_types import SamplerSchema
    from manabot.sim.search.sampling import HandDistribution, LearnedDistribution


class SamplerCodec(Protocol):
    def decode_legacy_sampler(self, data: bytes) -> dict[str, Any]: ...
    def sampler_parameters(
        self, model: LearnedDistribution
    ) -> Mapping[str, np.ndarray]: ...
    def restore_sampler(
        self,
        schema: SamplerSchema,
        hidden_size: int,
        history_dropout: float,
        parameters: Mapping[str, np.ndarray],
    ) -> LearnedDistribution: ...


@runtime_checkable
class BeliefSampling(SamplerCodec, Protocol):
    def PhysicalDistribution(self) -> HandDistribution: ...
    def freeze_sampler(self, model: LearnedDistribution) -> None: ...


def belief_sampling(backend: object) -> BeliefSampling:
    """Resolve the optional hand sampler only for search modes which draw hands."""
    sampling = getattr(backend, "belief", None)
    if not isinstance(sampling, BeliefSampling):
        raise ValueError("this search mode requires a backend with belief sampling")
    return sampling


@runtime_checkable
class BeliefPolicy(BackendPolicy, Protocol):
    def admit_belief(self, belief_schema: BeliefEncodingSchema) -> None: ...
