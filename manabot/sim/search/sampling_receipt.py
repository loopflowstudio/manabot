"""Retained hidden-hand samples and their public-constraint validation."""

from dataclasses import dataclass
import json
import math

from manabot.belief.sampling_types import SamplerInput

PHYSICAL_SAMPLER_IDENTITY = "physical-compatible-hand/v1"


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
