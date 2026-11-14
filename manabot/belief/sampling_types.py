"""Public hand constraints and vocabulary identities, independent of model execution."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math


@dataclass(frozen=True, slots=True)
class SamplerSchema:
    """Ordered vocabulary and history feature semantics bound by identity."""

    vocabulary_identity: str
    card_names: tuple[str, ...]
    history_size: int
    max_count: int = 64

    def __post_init__(self) -> None:
        if not self.vocabulary_identity or not self.card_names:
            raise ValueError("sampler requires an identified vocabulary")
        if any(not name for name in self.card_names) or len(
            set(self.card_names)
        ) != len(self.card_names):
            raise ValueError("sampler card names must be nonempty and unique")
        if self.history_size < 0 or self.max_count < 1:
            raise ValueError("invalid sampler dimensions")

    @property
    def identity(self) -> str:
        """Bind ordered card coordinates, feature semantics and count capacity."""
        payload = {"schema": "manabot.autoregressive-hand-counts/v1", **asdict(self)}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class SamplerInput:
    """Public constraints; history feature encoding is part of vocabulary identity."""

    vocabulary_identity: str
    pool_counts: tuple[int, ...]
    known_minima: tuple[int, ...]
    hand_size: int
    history_features: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.vocabulary_identity or len(self.pool_counts) != len(
            self.known_minima
        ):
            raise ValueError("invalid sampler input vocabulary or dimensions")
        if type(self.hand_size) is not int or any(
            type(value) is not int for value in (*self.pool_counts, *self.known_minima)
        ):
            raise ValueError("hand constraints must be integer counts")
        if any(
            low < 0 or high < low
            for high, low in zip(self.pool_counts, self.known_minima, strict=True)
        ):
            raise ValueError("known hand minima must lie inside the pool")
        if not sum(self.known_minima) <= self.hand_size <= sum(self.pool_counts):
            raise ValueError("hand size has no compatible deal")
        if any(not math.isfinite(value) for value in self.history_features):
            raise ValueError("history features must be finite")

    def validate(self, schema: SamplerSchema) -> None:
        """Reject coordinate/feature mismatches before inference or persistence."""
        if (
            self.vocabulary_identity != schema.vocabulary_identity
            or len(self.pool_counts) != len(schema.card_names)
            or len(self.history_features) != schema.history_size
        ):
            raise ValueError("input does not match sampler schema")
        if max(self.pool_counts) > schema.max_count:
            raise ValueError("public pool exceeds sampler count capacity")


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
