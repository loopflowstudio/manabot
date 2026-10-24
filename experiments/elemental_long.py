"""MTG-183: one long two-seat Elemental seed, rated against its own checkpoints.

Jack Heart's request (2026-10-23): train a single seed on the ten-deck
Elemental world past the point where its score against the heuristic stops
rising, and find whether newer checkpoints still beat older ones there. The
protocol is experiments/protocols/elemental-183.md; the recipe is MTG-181's
(experiments/regimes/elemental-long-183-v1.json differs only in its ID).

Windows w1 and w2 ran under the retired RunPod cohort controller (frozen
commit 8e7df92e4). The seed continues as Etude segments (Jack Heart,
2026-10-24): each binds its parent's final raw/EMA/Adam exports through
`manabot.experiment.job.LearnerContinuation` and trains to an absolute update
target. This module freezes those segments, the export cadence and the
evaluation design:

- `stage_parent` / `segment_job` — a parent's exports as Etude inputs, and the
  segment that continues them;
- `rated_updates` — the checkpoints that are evaluated and rated;
- `heuristic_games` — the 10×10 heuristic grid (full or declared subset) for one;
- `h2h_schedule` / `pair_design` — the cross-checkpoint games and pairing graph.

    uv run python -m experiments.elemental_long stage --parent DIR \
        --storage s3://BUCKET --prefix mtg183/w2 --out continuation.json
    uv run python -m experiments.elemental_long segment \
        --continuation continuation.json --updates N --hours H --max-usd D \
        --revision COMMIT --out job.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal
import hashlib
from pathlib import Path
import random
from typing import Literal

from etude.launch.records import Job
from etude.run.storage import Storage, storage_from_uri

from experiments.elemental_grid import DECKS, Game, schedule
from manabot.arena.models import canonical_sha256, file_sha256
from manabot.experiment.job import LearnerContinuation, StagedExport, TrainingJob
from manabot.training.records import TrainingRun
from manabot.training.regime import TrainingRegime, TrainSelfPlay

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

# A parent export directory holds the stage's files under these names.
_PARENT_FILES = {
    "source_run": "run.json",
    "raw": f"{STAGE}-raw.pt",
    "ema": f"{STAGE}-ema.pt",
    "optimizer": f"{STAGE}-optimizer.pt",
}


def stage_parent(parent: Path, storage: Storage, prefix: str) -> LearnerContinuation:
    """Upload a parent's final exports as Etude inputs, checked against its record.

    The raw/EMA/Adam files must be the bytes the parent's run record published;
    the continuation starts at the record's absolute update. An object already
    under its key is kept only when its digest matches.
    """
    run = TrainingRun.model_validate_json((parent / "run.json").read_bytes())
    if run.seed != SEED or [s.id for s in run.stages] != [STAGE]:
        raise ValueError("parent is not an MTG-183 self-play window")
    if run.status not in {"completed", "paused"}:
        raise ValueError(f"parent did not finish its window: {run.status}")
    staged = {}
    for role, name in _PARENT_FILES.items():
        path = parent / name
        sha256, size = file_sha256(path), path.stat().st_size
        if role != "source_run":
            published = run.stages[0].artifacts.get(role)
            if published is None or (published["sha256"], published["bytes"]) != (
                sha256,
                size,
            ):
                raise ValueError(f"{name} differs from the parent's published {role}")
        folder = f"{prefix.strip('/')}/"
        key = folder + name
        if key in storage.keys(folder):
            if hashlib.sha256(storage.get(key)).hexdigest() != sha256:
                raise ValueError(f"{key} already holds other bytes")
        else:
            storage.put_file(key, path)
        staged[role] = StagedExport(key=key, sha256=sha256, bytes=size)
    return LearnerContinuation(
        from_update=run.updates_through(), source_stage=STAGE, **staged
    )


def segment_job(
    continuation: LearnerContinuation,
    updates: int,
    *,
    hours: float,
    finalization_seconds: float = 600,
    setup_seconds: float = 1800,
) -> TrainingJob:
    """The frozen recipe continued to an absolute `updates` target.

    Only the endpoint and the watchdog change; the learning recipe, streams and
    seed are the parent's. `setup_seconds` leaves room for machine setup and the
    numerics gate inside the allocation.
    """
    regime = TrainingRegime.model_validate_json(REGIME.read_bytes())
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.updates = updates
    # Damped-policy rates follow the absolute iteration, not updates/endpoint,
    # so moving the endpoint leaves the parent's learning schedule unchanged.
    regime.wall_seconds = hours * 3600 - setup_seconds - finalization_seconds
    stage.execution.wall_seconds = regime.wall_seconds
    first = (continuation.from_update // CADENCE + 1) * CADENCE
    return TrainingJob(
        regime=TrainingRegime.model_validate(regime.model_dump(mode="json")),
        milestones=tuple(range(first, updates, CADENCE)),
        finalization_seconds=finalization_seconds,
        validate_numerics=True,
        continuation=continuation,
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
    stage = sub.add_parser("stage", help="upload a parent's exports as Etude inputs")
    stage.add_argument("--parent", type=Path, required=True)
    stage.add_argument("--storage", required=True)
    stage.add_argument("--prefix", required=True)
    stage.add_argument("--out", type=Path, required=True)
    segment = sub.add_parser("segment", help="write a continuation's Etude job")
    segment.add_argument("--continuation", type=Path, required=True)
    segment.add_argument("--updates", type=int, required=True)
    segment.add_argument("--hours", type=float, required=True)
    segment.add_argument("--max-usd", type=Decimal, required=True)
    segment.add_argument("--revision", required=True)
    segment.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "stage":
        continuation = stage_parent(
            args.parent, storage_from_uri(args.storage), args.prefix
        )
        args.out.write_text(continuation.model_dump_json(indent=2) + "\n")
        print(f"continues from update {continuation.from_update}")
        return
    spec = segment_job(
        LearnerContinuation.model_validate_json(args.continuation.read_bytes()),
        args.updates,
        hours=args.hours,
    )
    job: Job = spec.job(
        repo=ROOT,
        revision=args.revision,
        max_hours=args.hours,
        max_usd=args.max_usd,
    )
    args.out.write_text(job.model_dump_json(indent=2) + "\n")


if __name__ == "__main__":
    main()
