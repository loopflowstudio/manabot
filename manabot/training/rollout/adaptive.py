"""Deterministic fresh-game allocation over explicitly supplied ordered matchups.

AdaptiveConfig declares cells, not world admission. AdaptiveSampler owns coverage,
conditional propensities and an eight-generation diagnostic window. Its private
state is retained by the collector's existing recovery snapshot. Value error is
an unnormalized target residual, a PLR-inspired proxy rather than learning gain.
"""

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
import math
import random
from statistics import median
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import TypedDict

from manabot.infra.config import MatchConfig


class AdaptiveConfig(BaseModel):
    """Ordered allocation cells; runtime admission remains the checkpoint owner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    signal: Literal["U", "M", "E", "ME"]
    # Each entry is learner-role first; the sampler separately chooses engine seat.
    cells: tuple[MatchConfig, ...] = Field(min_length=1)
    seed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def unique_cells(self) -> "AdaptiveConfig":
        identities = [
            (
                c.content_pack,
                tuple(sorted(c.hero_deck.items())),
                tuple(sorted(c.hero_sideboard.items())),
                tuple(sorted(c.villain_deck.items())),
                tuple(sorted(c.villain_sideboard.items())),
            )
            for c in self.cells
        ]
        if len(set(identities)) != len(identities):
            raise ValueError("adaptive cells must be distinct ordered setups")
        return self


@dataclass(frozen=True)
class Assignment:
    """A committed start with propensity conditional on pre-draw queue state."""

    cell: int
    seat: int
    generation: int
    component: Literal["initial", "coverage", "uniform", "age", "priority"]
    probability: float
    start: int


@dataclass(frozen=True)
class GameResult:
    cell: int
    seat: int
    game: tuple[int, int]  # stream, ordinal: both seat records share one identity
    outcome: float


@dataclass(frozen=True)
class ErrorChunk:
    cell: int
    total: float
    count: int
    games: frozenset[tuple[int, int]]


@dataclass(frozen=True)
class ScoreGeneration:
    generation: int
    results: tuple[GameResult, ...]
    errors: tuple[ErrorChunk, ...]


@dataclass
class SamplerState:
    generation: int = 0
    starts: int = 0
    queue: list[int] = field(default_factory=list)
    last_visit: list[int] = field(default_factory=list)
    next_seats: list[int] = field(default_factory=list)
    window: list[ScoreGeneration] = field(default_factory=list)
    matchup_scores: list[float | None] = field(default_factory=list)
    error_scores: list[float] = field(default_factory=list)
    probabilities: list[float] = field(default_factory=list)
    rng: tuple[int, tuple[int, ...], float | None] | None = None


class AllocationRow(TypedDict):
    stream: int
    game: int
    cell: int
    seat: int
    generation: int
    component: str
    probability: float
    learner_deck: str
    opponent_deck: str


class CompletedRow(TypedDict):
    stream: int
    game: int
    cell: int
    seat: int
    outcome: float


class ErrorRow(TypedDict):
    cell: int
    total: float
    count: int
    games: list[list[int]]


class ExposureRow(TypedDict):
    cell: int
    seat: int
    transitions: int
    residual_total: float
    action_surprise_total: float
    legal_offer_total: int
    forced_actions: int


class AdaptiveDiagnostics(TypedDict):
    signal: str
    generation: int
    starts: int
    allocations: list[AllocationRow]
    completed: list[CompletedRow]
    errors: list[ErrorRow]
    exposures: list[ExposureRow]
    matchup_scores: list[float | None]
    error_scores: list[float]
    priority_probabilities: list[float]


def _ranks(scores: list[float]) -> list[float]:
    """Descending inverse ranks, with tied ranks averaged before inversion."""
    counts = Counter(scores)
    inverse_ranks: dict[float, float] = {}
    greater = 0
    for score in sorted(counts, reverse=True):
        equal = counts[score]
        inverse_ranks[score] = 1 / (greater + (equal + 1) / 2)
        greater += equal
    weights = [inverse_ranks[score] for score in scores]
    total = sum(weights)
    return [weight / total for weight in weights]


def _cap(weights: list[float]) -> list[float]:
    # Tiny admitted test catalogues cannot satisfy a 5% cap; their feasible cap
    # is 1/N. The production 100-cell recipe uses exactly 5%.
    cap = max(0.05, 1 / len(weights))
    result = [0.0] * len(weights)
    remaining = set(range(len(weights)))
    mass = 1.0
    while remaining:
        total = sum(weights[i] for i in remaining)
        clipped = {i for i in remaining if mass * weights[i] / total > cap}
        if not clipped:
            for i in remaining:
                result[i] = mass * weights[i] / total
            break
        for i in sorted(clipped):
            result[i] = cap
            mass -= cap
        remaining -= clipped
    return result


class AdaptiveSampler:
    def __init__(self, config: AdaptiveConfig) -> None:
        self.config = config.model_copy(deep=True)
        self._rng = random.Random(config.seed)
        n = len(config.cells)
        queue = list(range(2 * n))
        self._rng.shuffle(queue)
        self.state = SamplerState(
            queue=queue,
            last_visit=[-1] * (2 * n),
            next_seats=[self._rng.randrange(2) for _ in range(n)],
            matchup_scores=[None] * n,
            error_scores=[0.0] * n,
            probabilities=[1 / n] * n,
        )

    def draw(self) -> Assignment:
        s = self.state
        n = len(self.config.cells)
        if s.starts < 2 * n:
            component = "initial"
        elif (s.starts - 2 * n) % 5 == 0:
            component = "coverage"
        else:
            component = "uniform" if self.config.signal == "U" else "signal"
        if component in ("initial", "coverage"):
            if not s.queue:
                s.queue = list(range(2 * n))
                self._rng.shuffle(s.queue)
            stratum = s.queue.pop()
            probability = 1.0
        elif component == "uniform":
            stratum = self._rng.randrange(2 * n)
            probability = 1 / (2 * n)
        else:
            # Conditional on a non-protected slot: 1/4 age and 3/4 priority.
            # Together with every fifth protected slot this is 20/20/60.
            oldest = min(s.last_visit)
            aged = [i for i, visit in enumerate(s.last_visit) if visit == oldest]
            if self._rng.random() < 0.25:
                stratum = self._rng.choice(aged)
                component = "age"
            else:
                cell = self._rng.choices(range(n), weights=s.probabilities)[0]
                stratum = cell * 2 + s.next_seats[cell]
                component = "priority"
            cell, seat = divmod(stratum, 2)
            # Marginal over the random component, conditional on queues/state.
            probability = (0.25 / len(aged) if stratum in aged else 0.0) + (
                0.75 * s.probabilities[cell] if seat == s.next_seats[cell] else 0.0
            )
        cell, seat = divmod(stratum, 2)
        assignment = Assignment(
            cell, seat, s.generation, component, probability, s.starts
        )
        s.last_visit[stratum] = s.starts
        if component == "priority":
            s.next_seats[cell] = 1 - seat
        s.starts += 1
        return assignment

    def commit(self, results: list[GameResult], errors: list[ErrorChunk]) -> None:
        """Freeze diagnostics only after a successful collection/update boundary.

        Chunks use all pre-filter learner rows. Completed game IDs support
        shrinkage once, even if their trajectory contributed several chunks.
        """
        n = len(self.config.cells)
        for result in results:
            if not (
                0 <= result.cell < n
                and result.seat in (0, 1)
                and result.outcome in (0.0, 0.5, 1.0)
            ):
                raise ValueError("invalid adaptive game result")
        for chunk in errors:
            if not (
                0 <= chunk.cell < n
                and chunk.count > 0
                and math.isfinite(chunk.total)
                and chunk.total >= 0
            ):
                raise ValueError("invalid adaptive error chunk")
        s = self.state
        window = (
            s.window
            + [ScoreGeneration(s.generation + 1, tuple(results), tuple(errors))]
        )[-8:]
        completed: set[tuple[int, int]] = set()
        chunks = [chunk for generation in window for chunk in generation.errors]
        seat_outcomes: list[list[list[float]]] = [[[], []] for _ in range(n)]
        cell_chunks: list[list[ErrorChunk]] = [[] for _ in range(n)]
        for generation in window:
            for result in generation.results:
                if result.game in completed:
                    raise ValueError("duplicate completed adaptive game")
                completed.add(result.game)
                seat_outcomes[result.cell][result.seat].append(result.outcome)
        for chunk in chunks:
            cell_chunks[chunk.cell].append(chunk)
        pooled = median([c.total / c.count for c in chunks]) if chunks else 0.0
        matchup: list[float | None] = []
        error: list[float] = []
        for seats, selected in zip(seat_outcomes, cell_chunks, strict=True):
            if min(map(len, seats)) >= 20:
                p = sum((1 + sum(values)) / (2 + len(values)) for values in seats) / 2
                score = max(0.0, 1 - 6 * abs(p - 1 / 3), 1 - 6 * abs(p - 2 / 3))
                # Inverse ranks amplify even one ulp at the theoretical zeros
                # (0, 1/2, 1). Normalize arithmetic residue before tying ranks.
                matchup.append(0.0 if score <= 8 * math.ulp(1.0) else score)
            else:
                matchup.append(None)
            supported = {game for c in selected for game in c.games} & completed
            weight = len(supported) / (len(supported) + 20)
            raw = median([c.total / c.count for c in selected]) if selected else pooled
            error.append(weight * raw + (1 - weight) * pooled)
        # Unknown outcomes have no learned M score. Preserve their uniform
        # baseline share; rank known cells only within their collective mass.
        # Ranking unknowns at the maximum would reward uncertainty as a signal.
        m = [1 / n] * n
        if self.config.signal in ("M", "ME"):
            known = [i for i, score in enumerate(matchup) if score is not None]
            if known:
                ranks = _ranks([score for score in matchup if score is not None])
                for cell, weight in zip(known, ranks, strict=True):
                    m[cell] = weight * len(known) / n
        match self.config.signal:
            case "U":
                priority = [1 / n] * n
            case "M":
                priority = m
            case "E":
                priority = _ranks(error)
            case "ME":
                priority = [(a + b) / 2 for a, b in zip(m, _ranks(error), strict=True)]
        s.window, s.matchup_scores, s.error_scores = window, matchup, error
        s.probabilities = _cap(priority)
        s.generation += 1

    def snapshot(self) -> SamplerState:
        state = deepcopy(self.state)
        state.rng = self._rng.getstate()
        return state

    def restore(self, state: SamplerState) -> None:
        if len(state.probabilities) != len(self.config.cells):
            raise ValueError("sampler catalogue differs")
        if state.rng is None:
            raise ValueError("sampler RNG state missing")
        self._rng.setstate(state.rng)
        self.state = deepcopy(state)
