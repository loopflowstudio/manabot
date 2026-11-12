"""Curriculum scores, allocation propensities and deterministic sampler recovery."""

from copy import deepcopy
from typing import Literal

import pytest

from manabot.infra.config import MatchConfig
from manabot.training.rollout.adaptive import (
    AdaptiveConfig,
    AdaptiveSampler,
    ErrorChunk,
    GameResult,
)


def config(
    signal: Literal["U", "M", "E", "ME"] = "E", count: int = 4
) -> AdaptiveConfig:
    return AdaptiveConfig(
        signal=signal,
        seed=123,
        cells=tuple(
            MatchConfig(hero_deck={"Mountain": 8 + i}, villain_deck={"Forest": 8})
            for i in range(count)
        ),
    )


@pytest.mark.parametrize("signal", ["U", "M", "E", "ME"])
def test_deterministic_coverage_and_restored_probabilities(
    signal: Literal["U", "M", "E", "ME"],
) -> None:
    first = AdaptiveSampler(config(signal))
    initial = [first.draw() for _ in range(8)]
    assert {(a.cell, a.seat) for a in initial} == {
        (c, s) for c in range(4) for s in (0, 1)
    }
    assert all(a.probability == 1 for a in initial)
    results = [
        GameResult(cell, seat, (cell * 2 + seat, game), float(game % 3 == 0))
        for cell in range(4)
        for seat in (0, 1)
        for game in range(20)
    ]
    errors = [
        ErrorChunk(cell, float(cell + 1), 1, frozenset({(cell * 2, 0)}))
        for cell in range(4)
    ]
    first.commit(results, errors)
    restored = AdaptiveSampler(config(signal))
    restored.restore(first.snapshot())
    left = [first.draw() for _ in range(1000)]
    assert left == [restored.draw() for _ in range(1000)]
    assert all(0 < a.probability <= 1 for a in left)
    protected = [a for a in left if a.component == "coverage"]
    for offset in range(0, len(protected) - 7, 8):
        assert len({(a.cell, a.seat) for a in protected[offset : offset + 8]}) == 8


def test_rank_cap_signal_and_invalid_score_atomicity() -> None:
    sampler = AdaptiveSampler(config("E", 100))
    results = [GameResult(i, 0, (i, 0), 1.0) for i in range(100)]
    errors = [ErrorChunk(i, float(i * 100), 1, frozenset({(i, 0)})) for i in range(100)]
    sampler.commit(results, errors)
    assert sum(sampler.state.probabilities) == pytest.approx(1)
    assert max(sampler.state.probabilities) <= 0.05
    assert sampler.state.probabilities[-1] > sampler.state.probabilities[0]
    state = sampler.snapshot()
    with pytest.raises(ValueError, match="invalid adaptive error"):
        sampler.commit([], [ErrorChunk(0, float("nan"), 1, frozenset())])
    assert sampler.snapshot() == state
    for _ in range(8):
        sampler.commit([], [])
    assert not any(g.results or g.errors for g in sampler.state.window)
    assert sampler.state.error_scores == [0.0] * 100


def test_conditional_probability_includes_random_component() -> None:
    sampler = AdaptiveSampler(config())
    for _ in range(9):
        sampler.draw()  # initialization plus first protected coverage slot
    before = deepcopy(sampler.state)
    assignment = sampler.draw()
    aged = [
        i
        for i, visit in enumerate(before.last_visit)
        if visit == min(before.last_visit)
    ]
    stratum = assignment.cell * 2 + assignment.seat
    expected = (0.25 / len(aged) if stratum in aged else 0.0) + (
        0.75 * before.probabilities[assignment.cell]
        if before.next_seats[assignment.cell] == assignment.seat
        else 0.0
    )
    assert assignment.probability == pytest.approx(expected)


def test_matchup_prior_seat_threshold_and_error_allocation_differ_from_uniform() -> (
    None
):
    adaptive, uniform = (
        AdaptiveSampler(config("ME", 100)),
        AdaptiveSampler(config("U", 100)),
    )
    results = [
        GameResult(cell, seat, (cell * 2 + seat, game), float(game < wins))
        for cell, wins in ((0, 6), (1, 10))
        for seat in (0, 1)
        for game in range(20)
    ]
    errors = [
        ErrorChunk(cell, float(cell + 1), 1, frozenset({(cell * 2, 0)}))
        for cell in (0, 1)
    ]
    adaptive.commit(results, errors)
    uniform.commit(results, errors)
    assert adaptive.state.matchup_scores[0] is not None
    assert adaptive.state.matchup_scores[1] is not None
    assert adaptive.state.matchup_scores[0] > adaptive.state.matchup_scores[1]
    assert adaptive.state.matchup_scores[1] == pytest.approx(0)
    assert adaptive.state.matchup_scores[2] is None  # missing outcomes use the prior
    assert adaptive.state.error_scores == uniform.state.error_scores
    assert adaptive.state.matchup_scores == uniform.state.matchup_scores
    assert [adaptive.draw() for _ in range(200)] == [uniform.draw() for _ in range(200)]
    left, right = (
        [adaptive.draw() for _ in range(100)],
        [uniform.draw() for _ in range(100)],
    )
    assert [(a.cell, a.seat) for a in left] != [(a.cell, a.seat) for a in right]


@pytest.mark.parametrize("signal", ["M", "ME"])
def test_unknown_matchups_keep_neutral_mass_between_peak_and_midpoint(
    signal: Literal["M", "ME"],
) -> None:
    sampler = AdaptiveSampler(config(signal, 100))
    # Beta(1,1): (7+1)/(22+2) = 1/3, (11+1)/(22+2) = 1/2.
    results = [
        GameResult(cell, seat, (cell * 2 + seat, game), float(game < wins))
        for cell, wins in ((0, 7), (1, 11))
        for seat in (0, 1)
        for game in range(22)
    ]
    sampler.commit(results, [])  # E has no signal, so ME mixes with uniform.
    probabilities = sampler.state.probabilities
    assert sampler.state.matchup_scores[:2] == pytest.approx([1.0, 0.0])
    assert sampler.state.matchup_scores[2] is None
    assert probabilities[0] > probabilities[2] > probabilities[1]
    assert probabilities[2:] == pytest.approx([0.01] * 98)
    assert probabilities[0] + probabilities[1] == pytest.approx(0.02)
    assert sum(probabilities) == pytest.approx(1.0)
    assert max(probabilities) < 0.05  # The cap does not conceal the distinction.


@pytest.mark.parametrize("signal", ["M", "ME"])
def test_theoretical_zero_scores_have_equal_priority_at_catalogue_scale(
    signal: Literal["M", "ME"],
) -> None:
    sampler = AdaptiveSampler(config(signal, 100))
    # A single draw plus 20 wins/losses per seat gives smoothed p=1/2,
    # p=1/24 or p=23/24. All lie at/outside the triangle's zero boundaries.
    results = [
        GameResult(cell, seat, (cell * 2 + seat, game), outcome)
        for cell in range(100)
        for seat in (0, 1)
        for game, outcome in enumerate(
            [0.5] * 22 if cell % 3 == 0 else [float(cell % 3 == 1)] * 22
        )
    ]
    sampler.commit(results, [])
    assert sampler.state.matchup_scores == [0.0] * 100
    assert sampler.state.probabilities == pytest.approx([0.01] * 100)
    assert len(set(sampler.state.probabilities)) == 1
