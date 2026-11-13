"""MTG-183: one long two-seat Elemental seed, rated against its own checkpoints.

Jack Heart's request (2026-10-23): train a single seed on the ten-deck
Elemental world past the point where its score against the heuristic stops
rising, and find whether newer checkpoints still beat older ones there. The
protocol is experiments/protocols/elemental-183.md; the recipe is MTG-181's
(experiments/regimes/elemental-long-183-v1.json differs only in its ID).

Windows w1 and w2 ran under the retired RunPod cohort controller (frozen
commit 8e7df92e4); later Etude segments continued raw/EMA/Adam exports.
Staging and launch entry points now refuse with the last runnable revision.
The frozen recipe, rated checkpoint cadence, heuristic grid and cross-checkpoint
pairing graph remain readable here; no historical run is translated or resumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Literal

from experiments.elemental_grid import DECKS, Game, schedule
from manabot.infra.identity import canonical_sha256
from manabot.training.regime import (
    TrainingRegime,
    TrainSelfPlay,
    refuse_frozen_execution,
)

ROOT = Path(__file__).resolve().parents[1]
REGIME = ROOT / "experiments/regimes/elemental-long-183-v1.json"
PROTOCOLS = ROOT / "experiments/protocols"

SEED = 18301
# The regime's one self-play stage; also the Etude experiment its runs track under.
STAGE = "self-play"
# Inclusive authorization: every window, segment and failure.
CAP_DOLLARS = 50.0
# Raw exports every 50 absolute updates in every window; all are retained.
CADENCE = 50


# ---------------------------------------------------------------------------
# Etude segments


def stage_parent(*args: object, **kwargs: object) -> None:
    """Retired frozen execution entry point; retained data and evaluation still read."""
    refuse_frozen_execution()


def stage_monitoring(*args: object, **kwargs: object) -> None:
    """Retired frozen execution entry point; retained data and evaluation still read."""
    refuse_frozen_execution()


def recipe(
    updates: int,
    *,
    hours: float,
    finalization_seconds: float = 600,
    setup_seconds: float = 1800,
) -> TrainingRegime:
    """The frozen MTG-181 recipe trained to an absolute `updates` target in `hours`.

    Only the endpoint and the watchdog change. `setup_seconds` leaves room for
    machine setup and the numerics gate inside the allocation.
    """
    regime = TrainingRegime.model_validate_json(REGIME.read_bytes())
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.updates = updates
    # Damped-policy rates follow the absolute iteration, not updates/endpoint,
    # so moving the endpoint leaves the learning schedule unchanged.
    regime.wall_seconds = hours * 3600 - setup_seconds - finalization_seconds
    stage.execution.wall_seconds = regime.wall_seconds
    return TrainingRegime.model_validate(regime.model_dump(mode="json"))


def segment_job(*args: object, **kwargs: object) -> None:
    """Retired frozen execution entry point; retained data and evaluation still read."""
    refuse_frozen_execution()


# ---------------------------------------------------------------------------
# Rated checkpoints and the heuristic grid

# Full 50-game cells anchor the heuristic curve at these updates and the
# endpoint; every other rated checkpoint plays the declared 10-game subset.
HEURISTIC_FULL = (0, 100, 200, 400, 800, 1600, 3200, 6400)
HEURISTIC_SUBSET_GAMES = 10


def rated_updates(latest: int, endpoint: int | None = None) -> tuple[int, ...]:
    """Every 50 to 400, every 200 to 2,000, every 400 after, then the endpoint.

    `latest` bounds the cadence points known so far; the endpoint (the seed's
    final update) joins once the last segment has ended.
    """
    bound = latest if endpoint is None else endpoint
    found = [
        u
        for u in range(0, bound + 1, CADENCE)
        if u <= 400 or (u <= 2000 and u % 200 == 0) or u % 400 == 0
    ]
    if endpoint is not None and endpoint not in found:
        found.append(endpoint)
    return tuple(found)


def heuristic_games(
    updates: int, endpoint: int | None
) -> tuple[Literal["full", "subset"], list[Game]]:
    """The grid one rated checkpoint plays: `full` or the paired `subset`."""
    if updates in HEURISTIC_FULL or updates == endpoint:
        return "full", schedule()
    return "subset", [g for g in schedule() if g.index < HEURISTIC_SUBSET_GAMES]


# ---------------------------------------------------------------------------
# Cross-checkpoint games

H2H_SEED = 18_390_001
H2H_GAMES_PER_CELL = 10
H2H_DEAL_BASE = 1_839_000_000
# Fixed earlier references every later rated checkpoint plays.
REFERENCES = (200, 2000)


@dataclass(frozen=True)
class H2HGame:
    newer_deck: str
    older_deck: str
    index: int
    deal: int
    newer_seat: Literal[0, 1]


def h2h_schedule() -> list[H2HGame]:
    """1,000 games: newer deck × older deck × 10, seat drawn per game.

    Every pair plays these same deals, so pairs are paired by deal.
    """
    rng = random.Random(H2H_SEED)
    games = []
    for x, newer in enumerate(DECKS):
        for y, older in enumerate(DECKS):
            for index in range(H2H_GAMES_PER_CELL):
                seat: Literal[0, 1] = 0 if rng.random() < 0.5 else 1
                deal = H2H_DEAL_BASE + (x * len(DECKS) + y) * 1_000 + index
                games.append(H2HGame(newer, older, index, deal, seat))
    return games


def h2h_schedule_sha256() -> str:
    return canonical_sha256([vars(game) for game in h2h_schedule()])


H2H_SCHEDULE_SHA256 = "27f22397b3462d2856784726d953b2a4c05a44be6b595085d04cf664cc0a7f60"

Role = Literal["consecutive", "reference", "endpoint", "held-out", "halving"]


@dataclass(frozen=True)
class Pair:
    newer: int
    older: int
    role: Role


def pair_design(rated: tuple[int, ...], endpoint: int | None) -> list[Pair]:
    """The connected pairing graph over rated checkpoints, newer first.

    Fit pairs: consecutive rated checkpoints, every later checkpoint against
    each fixed reference, and the endpoint against every earlier checkpoint.
    Held-out pairs (rated[i] vs rated[i+2] for even i, when not already a fit
    pair) never enter the rating fit; they test its predictions.
    """
    pairs: dict[tuple[int, int], Role] = {}
    for older, newer in zip(rated, rated[1:]):
        pairs[(newer, older)] = "consecutive"
    for reference in REFERENCES:
        if reference in rated:
            for newer in rated:
                if newer > reference:
                    pairs.setdefault((newer, reference), "reference")
    if endpoint is not None:
        for older in rated:
            if older < endpoint:
                pairs.setdefault((endpoint, older), "endpoint")
    for i in range(0, len(rated) - 2, 2):
        pairs.setdefault((rated[i + 2], rated[i]), "held-out")
    return [Pair(newer, older, role) for (newer, older), role in pairs.items()]


# ---------------------------------------------------------------------------
# CLI


def main(*args: object, **kwargs: object) -> None:
    """Retired frozen execution entry point; retained data and evaluation still read."""
    refuse_frozen_execution()


if __name__ == "__main__":
    main()
