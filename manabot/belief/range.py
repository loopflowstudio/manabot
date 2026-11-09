"""Normalized manabot beliefs over canonical managym possible worlds.

``BeliefState`` owns probability and nothing else.  Its ordered rows, hand
semantics, compatible-deal weights, identity, and materializer all come from
one :class:`managym.possible_worlds.PossibleWorldSpace`.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Mapping

from managym.possible_worlds import PossibleWorldSpace
import numpy as np
from numpy.typing import NDArray


class BeliefError(ValueError):
    """A belief update violated canonical support or normalization."""


def _logsumexp(values: NDArray[np.float64]) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise BeliefError("belief has no finite mass")
    maximum = float(np.max(finite))
    return maximum + math.log(float(np.exp(finite - maximum).sum()))


def _row_key(hand: Iterable[tuple[str, int]]) -> tuple[tuple[str, int], ...]:
    return tuple((str(name), int(count)) for name, count in hand if int(count) > 0)


@dataclass(frozen=True, slots=True)
class BeliefState:
    """A normalized probability vector bound to one canonical world space."""

    space: PossibleWorldSpace
    model_id: str
    log_probabilities: NDArray[np.float64]

    def __post_init__(self) -> None:
        values = np.asarray(self.log_probabilities, dtype=np.float64).copy()
        if values.shape != (self.space.support_size,):
            raise BeliefError("belief must have one probability per canonical world")
        if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
            raise BeliefError("belief contains a non-finite positive mass")
        normalizer = _logsumexp(values)
        if not math.isclose(normalizer, 0.0, abs_tol=1e-10):
            raise BeliefError(f"belief is not normalized: log_z={normalizer}")
        values.setflags(write=False)
        object.__setattr__(self, "log_probabilities", values)

    @classmethod
    def compatible_prior(
        cls, space: PossibleWorldSpace, *, model_id: str = "compatible-deal-prior/v1"
    ) -> "BeliefState":
        weights = np.asarray([world.weight for world in space.worlds], dtype=np.float64)
        if np.any(weights <= 0.0):
            raise BeliefError("canonical compatible-deal weights must be positive")
        log_probabilities = np.log(weights)
        log_probabilities -= _logsumexp(log_probabilities)
        return cls(space, model_id, log_probabilities)

    @classmethod
    def from_probabilities(
        cls,
        space: PossibleWorldSpace,
        model_id: str,
        probabilities: Iterable[float],
    ) -> "BeliefState":
        values = np.asarray(tuple(probabilities), dtype=np.float64)
        if values.shape != (space.support_size,):
            raise BeliefError("probabilities must match canonical support")
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise BeliefError("probabilities must be finite and non-negative")
        total = float(values.sum())
        if total <= 0.0:
            raise BeliefError("belief has zero total mass")
        normalized = values / total
        logs = np.full(values.shape, -math.inf, dtype=np.float64)
        positive = normalized > 0.0
        logs[positive] = np.log(normalized[positive])
        return cls(space, model_id, logs)

    @property
    def support_size(self) -> int:
        """Number of canonical rows, including exact zero-mass hypotheses."""

        return self.space.support_size

    @property
    def positive_support_size(self) -> int:
        return int(np.count_nonzero(np.isfinite(self.log_probabilities)))

    @property
    def probabilities(self) -> NDArray[np.float64]:
        return np.exp(self.log_probabilities)

    @property
    def normalization_error(self) -> float:
        return abs(float(self.probabilities.sum()) - 1.0)

    @property
    def effective_range_size(self) -> float:
        probabilities = self.probabilities
        return 1.0 / float(np.square(probabilities).sum())

    @property
    def entropy(self) -> float:
        probabilities = self.probabilities
        positive = probabilities[probabilities > 0.0]
        return -float(np.sum(positive * np.log(positive)))

    @property
    def allocated_bytes(self) -> int:
        return self.probability_bytes + self.space.allocated_bytes

    @property
    def probability_bytes(self) -> int:
        return int(self.log_probabilities.nbytes)

    @property
    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.space.identity.encode("ascii"))
        digest.update(b"\0")
        digest.update(self.model_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.asarray(self.log_probabilities, dtype="<f8").tobytes())
        return digest.hexdigest()

    def probability_at(self, world_index: int) -> float:
        self.space.world(world_index)
        return float(self.probabilities[world_index])

    def index_for_hand(self, hand: Mapping[str, int]) -> int | None:
        target = _row_key(sorted(hand.items()))
        for world in self.space.worlds:
            if world.hand == target:
                return world.index
        return None

    def probability_of_hand(self, hand: Mapping[str, int]) -> float:
        index = self.index_for_hand(hand)
        return 0.0 if index is None else self.probability_at(index)

    def rank(self, world_index: int) -> int | None:
        probability = self.probability_at(world_index)
        if probability <= 0.0:
            return None
        return 1 + int(np.count_nonzero(self.probabilities > probability))

    def inclusion_probabilities(self) -> dict[str, float]:
        probabilities = self.probabilities
        names = [name for name, _ in self.space.pool]
        return {
            name: float(
                sum(
                    probabilities[world.index]
                    for world in self.space.worlds
                    if world.count(name) > 0
                )
            )
            for name in names
        }

    def sample_indexes(self, count: int, *, seed: int) -> list[int]:
        if count < 1:
            raise BeliefError("sample count must be positive")
        rng = np.random.default_rng(seed)
        return [
            int(index)
            for index in rng.choice(
                self.space.support_size, size=count, p=self.probabilities
            )
        ]

    def condition_likelihood(
        self,
        likelihoods: Iterable[float],
        legal_action_counts: Iterable[int],
        matching_action_counts: Iterable[int],
        *,
        epsilon: float,
        model_id: str | None = None,
    ) -> "BeliefState":
        """Condition on one provider-owned semantic commitment identity."""

        if not 0.0 <= epsilon < 1.0:
            raise BeliefError("epsilon must be in [0, 1)")
        likelihood = np.asarray(tuple(likelihoods), dtype=np.float64)
        legal_counts = np.asarray(tuple(legal_action_counts), dtype=np.int64)
        matching_counts = np.asarray(tuple(matching_action_counts), dtype=np.int64)
        expected = (self.space.support_size,)
        if (
            likelihood.shape != expected
            or legal_counts.shape != expected
            or matching_counts.shape != expected
        ):
            raise BeliefError("likelihood rows must match the canonical space")
        if np.any(~np.isfinite(likelihood)) or np.any(
            (likelihood < 0.0) | (likelihood > 1.0)
        ):
            raise BeliefError("action likelihoods must be finite probabilities")
        if np.any(legal_counts < 0) or np.any(matching_counts < 0):
            raise BeliefError("action counts cannot be negative")
        if np.any(matching_counts > legal_counts):
            raise BeliefError("matching actions cannot exceed legal actions")
        mixed = np.zeros_like(likelihood)
        legal = (legal_counts > 0) & (matching_counts > 0)
        mixed[legal] = (1.0 - epsilon) * likelihood[legal] + epsilon * (
            matching_counts[legal] / legal_counts[legal]
        )
        updated = np.full(expected, -math.inf, dtype=np.float64)
        positive = mixed > 0.0
        updated[positive] = self.log_probabilities[positive] + np.log(mixed[positive])
        updated -= _logsumexp(updated)
        return BeliefState(self.space, model_id or self.model_id, updated)

    def transport(
        self,
        next_space: PossibleWorldSpace,
        *,
        known_exits: Iterable[str] = (),
        known_returns: Iterable[str] = (),
        hidden_draws: int = 0,
        foretold_exits: Iterable[str] = (),
        hidden_foretells: int = 0,
    ) -> "BeliefState":
        """Apply canonical public-zone/chance facts, then join the next space.

        The foretold slot is hidden but public as a count. ``foretold_exits``
        are revealed cards leaving the slot (a cast from exile): they leave the
        pool, not the hand. ``hidden_foretells`` move one unrevealed hand card
        each into the slot; the card stays in the pool. Which card was
        foretold is not modelled yet (MTG-215): each hand row loses one card
        uniformly by count, rows the next space excludes are conditioned
        away, and later hidden draws treat slot cards like library cards.
        """

        if hidden_draws < 0 or hidden_foretells < 0:
            raise BeliefError("hidden draw and foretell counts cannot be negative")
        pool = dict(self.space.pool)
        hand_size = self.space.hand_size
        foretold = self.space.foretold
        rows = {
            world.hand: float(self.log_probabilities[world.index])
            for world in self.space.worlds
            if math.isfinite(self.log_probabilities[world.index])
        }

        for card in known_exits:
            if pool.get(card, 0) <= 0:
                raise BeliefError(f"known hand exit for absent card {card!r}")
            next_rows: dict[tuple[tuple[str, int], ...], float] = {}
            for key, mass in rows.items():
                hand = dict(key)
                count = hand.get(card, 0)
                if count <= 0:
                    continue
                if count == 1:
                    del hand[card]
                else:
                    hand[card] = count - 1
                target = _row_key(sorted(hand.items()))
                next_rows[target] = float(
                    np.logaddexp(next_rows.get(target, -math.inf), mass)
                )
            rows = next_rows
            pool[card] -= 1
            if pool[card] == 0:
                del pool[card]
            hand_size -= 1

        for card in known_returns:
            next_rows = {}
            for key, mass in rows.items():
                hand = dict(key)
                hand[card] = hand.get(card, 0) + 1
                target = _row_key(sorted(hand.items()))
                next_rows[target] = float(
                    np.logaddexp(next_rows.get(target, -math.inf), mass)
                )
            rows = next_rows
            pool[card] = pool.get(card, 0) + 1
            hand_size += 1

        for card in foretold_exits:
            if pool.get(card, 0) <= 0 or foretold <= 0:
                raise BeliefError(f"foretold exit for absent card {card!r}")
            pool[card] -= 1
            if pool[card] == 0:
                del pool[card]
            foretold -= 1
            # The revealed copy was never in hand, so hands holding every
            # remaining copy are no longer possible.
            remaining = pool.get(card, 0)
            rows = {
                key: mass
                for key, mass in rows.items()
                if dict(key).get(card, 0) <= remaining
            }

        for _ in range(hidden_foretells):
            if hand_size <= 0:
                raise BeliefError("cannot foretell from an empty hidden hand")
            next_rows = {}
            for key, mass in rows.items():
                for card, count in key:
                    hand = dict(key)
                    if count == 1:
                        del hand[card]
                    else:
                        hand[card] = count - 1
                    target = _row_key(sorted(hand.items()))
                    contribution = mass + math.log(count / hand_size)
                    next_rows[target] = float(
                        np.logaddexp(next_rows.get(target, -math.inf), contribution)
                    )
            rows = next_rows
            hand_size -= 1
            foretold += 1

        for _ in range(hidden_draws):
            library_size = sum(pool.values()) - hand_size
            if library_size <= 0:
                raise BeliefError("cannot draw from an empty hidden library")
            next_rows = {}
            for key, mass in rows.items():
                hand = dict(key)
                for card, total in pool.items():
                    available = total - hand.get(card, 0)
                    if available <= 0:
                        continue
                    drawn = dict(hand)
                    drawn[card] = drawn.get(card, 0) + 1
                    target = _row_key(sorted(drawn.items()))
                    contribution = mass + math.log(available / library_size)
                    next_rows[target] = float(
                        np.logaddexp(next_rows.get(target, -math.inf), contribution)
                    )
            rows = next_rows
            hand_size += 1

        if tuple(sorted(pool.items())) != next_space.pool:
            raise BeliefError("transport facts do not match the next canonical pool")
        if hand_size != next_space.hand_size:
            raise BeliefError("transport facts do not match the next public hand size")
        if foretold != next_space.foretold:
            raise BeliefError("transport facts do not match the next foretold count")
        return self._join_rows(
            next_space, rows, condition_on_support=hidden_foretells > 0
        )

    def _join_rows(
        self,
        next_space: PossibleWorldSpace,
        rows: Mapping[tuple[tuple[str, int], ...], float],
        *,
        condition_on_support: bool = False,
    ) -> "BeliefState":
        if (
            next_space.viewer != self.space.viewer
            or next_space.opponent != self.space.opponent
        ):
            raise BeliefError("belief transport crossed a viewer boundary")
        target = np.full(next_space.support_size, -math.inf, dtype=np.float64)
        unmatched = set(rows)
        for world in next_space.worlds:
            if world.hand in rows:
                target[world.index] = rows[world.hand]
                unmatched.discard(world.hand)
        if unmatched and not condition_on_support:
            raise BeliefError("probability transport left canonical support")
        target -= _logsumexp(target)
        return BeliefState(next_space, self.model_id, target)


__all__ = ["BeliefError", "BeliefState"]
