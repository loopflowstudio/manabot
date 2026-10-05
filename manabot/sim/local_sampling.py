"""Direct hidden-hand sampling for frozen-policy search.

PreparedHands binds viewer history to native constraints. SearchHandSampler is
an injectable distribution boundary; the caller validates each returned batch
before native materialization. Neither production implementation enumerates
possible worlds. The physical baseline ignores behavioral history deliberately.
"""

from dataclasses import dataclass
import json
import math
from typing import Callable, Literal, Protocol

import torch

from manabot.belief.sampling import (
    AutoregressiveBeliefSampler,
    SamplerInput,
    physical_deal_log_prob,
    sample_physical_deal,
)
from manabot.belief.sampling_data import _digest, public_sampler_input
from manabot.belief.state import ViewerHistory
import managym
from managym.decision import Observation


@dataclass(frozen=True)
class PreparedHands:
    distribution_identity: str
    history_identity: str
    card_names: tuple[str, ...]
    inputs: SamplerInput
    constraints_json: str


@dataclass(frozen=True)
class HandSample:
    counts: tuple[tuple[str, int], ...]
    log_probability: float


@dataclass(frozen=True)
class HandBatch:
    prepared: PreparedHands
    seed: int
    hands: tuple[HandSample, ...]


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


def _batch(
    sampler: SearchHandSampler,
    prepared: PreparedHands,
    count: int,
    seed: int,
    check: Callable[[], None],
    model: AutoregressiveBeliefSampler | None = None,
) -> HandBatch:
    if prepared.distribution_identity != sampler.identity or count < 1:
        raise ValueError("sampler preparation identity or count differs")
    generator = torch.Generator().manual_seed(seed)
    samples: list[HandSample] = []
    with torch.inference_mode():
        for _ in range(count):
            check()
            rows = [prepared.inputs]
            hand = (
                sample_physical_deal(rows, generator=generator)
                if model is None
                else model.sample(rows, generator=generator)
            )
            probability = (
                physical_deal_log_prob(rows, hand)
                if model is None
                else model.log_prob(rows, hand)
            )
            samples.append(
                HandSample(
                    tuple(
                        (name, int(n))
                        for name, n in zip(
                            prepared.card_names, hand[0].tolist(), strict=True
                        )
                        if n
                    ),
                    float(probability[0]),
                )
            )
            check()
    return HandBatch(prepared, seed, tuple(samples))


class PhysicalHandSampler:
    """Exact physical-deal measure after removing publicly known hand minima."""

    kind: Literal["compatible_prior"] = "compatible_prior"
    identity = "physical-compatible-hand/v1"

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
        names = tuple(sorted(constraints["pool"]))
        inputs = SamplerInput(
            _digest({"method": self.identity, "names": names}),
            tuple(constraints["pool"][name] for name in names),
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
        return _batch(self, prepared, count, seed, check)


class LearnedHandSampler:
    """An already admitted learned artifact, with unchanged public feature encoding."""

    kind: Literal["learned"] = "learned"

    def __init__(self, model: AutoregressiveBeliefSampler, identity: str) -> None:
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
        return _batch(self, prepared, count, seed, check, self.model)


def validate_hand(hand: HandSample, constraints_json: str) -> None:
    """Validate persisted or injected samples before any native branch is opened."""
    source = json.loads(constraints_json)
    counts = dict(hand.counts)
    if (
        len(counts) != len(hand.counts)
        or sum(counts.values()) != source["hand_size"]
        or any(
            type(n) is not int or n <= 0 or n > source["pool"].get(name, 0)
            for name, n in hand.counts
        )
        or any(counts.get(name, 0) < n for name, n in source["known_hand"].items())
        or not math.isfinite(hand.log_probability)
        or hand.log_probability > 1e-8
    ):
        raise ValueError("sampled hand violates public constraints")
