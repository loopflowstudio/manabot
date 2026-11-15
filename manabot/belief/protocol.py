"""Belief sampling capabilities independent of training/run orchestration."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Protocol

from manabot.policy.protocol import Backend

if TYPE_CHECKING:
    from manabot.belief.sampling_types import SamplerSchema
    from manabot.sim.search.sampling import HandDistribution, LearnedDistribution


class SamplerCodec(Protocol):
    def decode_sampler(self, data: bytes) -> Any: ...
    def restore_sampler(
        self,
        schema: SamplerSchema,
        hidden_size: int,
        history_dropout: float,
        state: Any,
    ) -> LearnedDistribution: ...


class BeliefSampling(SamplerCodec, Protocol):
    def PhysicalDistribution(self) -> HandDistribution: ...
    def freeze_sampler(self, model: LearnedDistribution) -> None: ...


class SamplerBackend(Backend, Protocol):
    @property
    def belief(self) -> BeliefSampling: ...
    def seeded_inference(
        self, seed: int, *, cpu_only: bool = False
    ) -> AbstractContextManager[object]: ...
