"""Direct hidden-hand sampling for frozen-policy search.

PreparedHands binds viewer history to native constraints. SearchHandSampler is
an injectable distribution boundary; the caller validates each returned batch
before native materialization. Neither production implementation enumerates
possible worlds. The physical baseline ignores behavioral history deliberately.
"""

import json
from typing import Callable, Literal, Protocol

from manabot.belief.sampling_data import hand_sampling_pool, public_sampler_input
from manabot.belief.sampling_types import SamplerInput, SamplerSchema, sampling_digest
from manabot.belief.state import ViewerHistory
from manabot.sim.search.sampling_receipt import (
    PHYSICAL_SAMPLER_IDENTITY,
    HandBatch,
    PreparedHands,
)
import managym
from managym.decision import Observation


class SearchHandSampler(Protocol):
    """Injected implementations cannot change the declared distribution identity."""

    @property
    def kind(self) -> Literal["compatible_prior", "learned"]: ...

    @property
    def identity(self) -> str: ...

    def prepare(self, engine: managym.Env, history: ViewerHistory) -> PreparedHands: ...

    def sample(
        self,
        prepared: PreparedHands,
        *,
        count: int,
        seed: int,
        check: Callable[[], None],
    ) -> HandBatch: ...


class HandDistribution(Protocol):
    def sample_hands(
        self,
        prepared: PreparedHands,
        *,
        count: int,
        seed: int,
        check: Callable[[], None],
    ) -> HandBatch: ...


class LearnedDistribution(HandDistribution, Protocol):
    schema: SamplerSchema


def _batch(
    sampler: SearchHandSampler,
    distribution: HandDistribution,
    prepared: PreparedHands,
    count: int,
    seed: int,
    check: Callable[[], None],
) -> HandBatch:
    if prepared.distribution_identity != sampler.identity or count < 1:
        raise ValueError("sampler preparation identity or count differs")
    return distribution.sample_hands(prepared, count=count, seed=seed, check=check)


class PhysicalHandSampler:
    """Exact physical-deal measure after reserving known hand and library copies."""

    kind: Literal["compatible_prior"] = "compatible_prior"
    identity = PHYSICAL_SAMPLER_IDENTITY

    def __init__(self, distribution: HandDistribution) -> None:
        self.distribution = distribution

    def prepare(self, engine: managym.Env, history: ViewerHistory) -> PreparedHands:
        observation = Observation.from_json(
            engine.semantic_observation_json(history.viewer)
        )
        if (
            observation.revision != history.current_revision
            or observation.viewer_state_hash != history.current_viewer_state_hash
            or observation.schema_version != history.schema_version
        ):
            raise ValueError("sampling history does not match current observation")
        encoded = engine.hidden_hand_constraints_json(history.viewer)
        constraints = json.loads(encoded)
        pool = hand_sampling_pool(
            constraints["pool"], constraints.get("known_library", {})
        )
        names = tuple(sorted(pool))
        inputs = SamplerInput(
            sampling_digest({"method": self.identity, "names": names}),
            tuple(pool[name] for name in names),
            tuple(constraints["known_hand"].get(name, 0) for name in names),
            constraints["hand_size"],
            (),
        )
        return PreparedHands(self.identity, history.identity, names, inputs, encoded)

    def sample(
        self,
        prepared: PreparedHands,
        *,
        count: int,
        seed: int,
        check: Callable[[], None],
    ) -> HandBatch:
        return _batch(self, self.distribution, prepared, count, seed, check)


class LearnedHandSampler:
    """An already admitted learned artifact, with unchanged public feature encoding."""

    kind: Literal["learned"] = "learned"

    def __init__(self, model: LearnedDistribution, identity: str) -> None:
        self.model = model
        self.identity = identity

    def prepare(self, engine: managym.Env, history: ViewerHistory) -> PreparedHands:
        inputs, constraints = public_sampler_input(engine, history, self.model.schema)
        return PreparedHands(
            self.identity,
            history.identity,
            self.model.schema.card_names,
            inputs,
            constraints,
        )

    def sample(
        self,
        prepared: PreparedHands,
        *,
        count: int,
        seed: int,
        check: Callable[[], None],
    ) -> HandBatch:
        return _batch(self, self.model, prepared, count, seed, check)
