"""MTG-183: one long two-seat Elemental seed, rated against its own checkpoints.

Jack Heart's request (2026-10-23): train a single seed on the ten-deck
Elemental world past the point where its score against the heuristic stops
rising, and find whether newer checkpoints still beat older ones there. The
protocol is experiments/protocols/elemental-183.md; the recipe is MTG-181's
(experiments/regimes/elemental-long-183-v1.json differs only in its ID).

One RunPod job is limited to twelve hours, so the seed trains as one cohort of
chained learner-state continuations (`manabot deploy cohort`): each window
binds the previous window's final raw/EMA/Adam exports at its last complete
update. This module freezes those windows, the export cadence and the
evaluation design:

- `rated_updates` — the checkpoints that are evaluated and rated;
- `heuristic_games` — the 10×10 heuristic grid (full or declared subset) for one;
- `h2h_schedule` / `pair_design` — the cross-checkpoint games and pairing graph.

`python -m experiments.elemental_long specs` writes the job specs;
`python -m experiments.elemental_long cohort --plan-dir DIR` assembles the
cohort intent from plans compiled at the frozen commit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Literal

from experiments.elemental_grid import DECKS, Game, schedule
from manabot.arena.models import canonical_sha256
from manabot.remote.cohort import Cohort, CohortEntry
from manabot.remote.continuation import Continuation
from manabot.remote.plan import DeploymentPlan, JobSpec, Machine

ROOT = Path(__file__).resolve().parents[1]
REGIME = ROOT / "experiments/regimes/elemental-long-183-v1.json"
PROTOCOLS = ROOT / "experiments/protocols"

SEED = 18301
COHORT = "mtg183-elemental-long"
# Inclusive authorization: every window, guardian proof and failure.
CAP_DOLLARS = 50.0
# Raw exports every 50 absolute updates in every window; all are retained.
CADENCE = 50

L4 = Machine(
    gpu_types=("NVIDIA L4",),
    vcpus=6,
    memory_gb=48,
    image="runpod/pytorch@sha256:cb154fcca15d1d6ce858cfa672b76505e30861ef981d28ec94bd44168767d853",
    hourly_ceiling=0.60,
)
# The guardian proof's allowance inside each window's spending limit.
PROBE_DOLLARS = 0.10


def _window(hours: float) -> JobSpec:
    projected = hours * (L4.hourly_ceiling + L4.storage_hourly_allowance)
    return JobSpec(
        machine=L4,
        lifetime_hours=hours,
        spending_limit=round(projected + PROBE_DOLLARS, 2),
        setup_seconds=900,
        checkpoint_seconds=120,
        upload_seconds=300,
        cleanup_seconds=120,
    )


# Six windows just under the twelve-hour credential limit, then one shorter
# window that fits the remaining cap: 6 × $7.48 + $3.82 = $48.70 reserved.
WINDOWS: tuple[tuple[str, JobSpec], ...] = (
    *((f"mtg183-elemental-long-w{i}", _window(11.9)) for i in range(1, 7)),
    ("mtg183-elemental-long-w7", _window(6.0)),
)
JOBS = tuple(job for job, _ in WINDOWS)
# Cohort scheduling ends six days after admission; windows need about 3.3 days.
COHORT_SECONDS = 6 * 24 * 3600


def spec_path(job: str) -> Path:
    hours = dict(WINDOWS)[job].lifetime_hours
    return PROTOCOLS / f"elemental-183-job-{hours:g}h.json"


def cohort(plans: dict[str, DeploymentPlan], admitted_at: float) -> Cohort:
    """The immutable cohort: window 1 trains from initialization, each later
    window continues its predecessor's last complete update."""
    entries = []
    previous: str | None = None
    for job, spec in WINDOWS:
        plan = plans[job]
        if plan.spec != spec or plan.seed != SEED:
            raise ValueError(f"{job} plan differs from its frozen window")
        entries.append(
            CohortEntry(
                job_id=job,
                plan=plan,
                checkpoint_updates=CADENCE,
                validate_numerics=True,
                continuation=None
                if previous is None
                else Continuation(
                    job_id=previous, stage_id="self-play", iteration="last-complete"
                ),
            )
        )
        previous = job
    return Cohort(
        cohort_id=COHORT,
        entries=tuple(entries),
        deadline=admitted_at + COHORT_SECONDS,
        spending_limit=CAP_DOLLARS,
        prior_dollars=0.0,
        controller_dollars=0.0,
    )


# ---------------------------------------------------------------------------
# Rated checkpoints and the heuristic grid

# Full 50-game cells anchor the heuristic curve at these updates and the
# endpoint; every other rated checkpoint plays the declared 10-game subset.
HEURISTIC_FULL = (0, 100, 200, 400, 800, 1600, 3200, 6400)
HEURISTIC_SUBSET_GAMES = 10


def rated_updates(latest: int, endpoint: int | None = None) -> tuple[int, ...]:
    """Every 50 to 400, every 200 to 2,000, every 400 after, then the endpoint.

    `latest` bounds the cadence points known so far; the endpoint (the last
    window's final update) joins once the cohort is settled.
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


def heuristic_games(updates: int, endpoint: int | None) -> tuple[str, list[Game]]:
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

Role = Literal["consecutive", "reference", "endpoint", "held-out"]


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("specs", help="write the frozen job specs")
    build = sub.add_parser("cohort", help="assemble the cohort from compiled plans")
    build.add_argument("--plan-dir", type=Path, required=True)
    build.add_argument("--admitted-at", type=float, required=True)
    build.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "specs":
        for job, spec in WINDOWS:
            spec_path(job).write_text(spec.model_dump_json(indent=2) + "\n")
        return
    plans = {
        job: DeploymentPlan.model_validate_json(
            (args.plan_dir / f"{job}.json").read_text()
        )
        for job in JOBS
    }
    value = cohort(plans, args.admitted_at)
    args.out.write_text(value.model_dump_json(indent=2) + "\n")
    print(
        f"{value.cohort_id}: reserved ${value.reserved_dollars:.2f} of ${CAP_DOLLARS}"
    )


if __name__ == "__main__":
    main()
