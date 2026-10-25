"""MTG-186: one two-seat seed trained only on UR vs GW, compared with MTG-181.

Jack Heart's question (2026-10-25): does training on the one matchup beat
MTG-181's training on all ten Elemental decks, on that matchup and across the
other 99 grid cells? The protocol is experiments/protocols/mtg186.md; the
recipe is experiments/regimes/elemental-ur-gw-186-v1.json, MTG-181's frozen
recipe with both ordered UR/GW pairs in place of the uniform ten-deck roster.

`train` builds the single Etude training run (one L4, update target 616,
exports at 0/200/400/600 and the endpoint). `identity` writes the pinned
identity record the protocol commits before allocation. Evaluation plays each
export through MTG-181's frozen 10×10 grid (`experiments.elemental_grid`) on
local CPU against the current heuristic. `follow` plays each wanted export
once it appears in Etude storage (R2), then MTG-181's matched-update UR/GW cells.

    uv run python -m experiments.mtg186 identity
    uv run python -m experiments.mtg186 quote --revision <commit>
    doppler run -p manabot -c prd -- doppler run -p etude -c prd -- uv run python -m experiments.mtg186 launch --revision <commit>
    doppler run -p manabot -c prd -- doppler run -p etude -c prd -- uv run --extra artifacts python -m experiments.mtg186 follow --run <run>

Launching and following need Etude's RunPod key and the R2 keys.
"""

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import subprocess
import time

from etude.launch.client import queue_cohort, quote, status
from etude.launch.records import Run
from etude.run.listing import runs as stored_runs
from etude.run.storage import storage_from_uri
from etude.train.read import read
from pydantic import BaseModel

from experiments.elemental_grid import (
    DECKS,
    GAMES_PER_CELL,
    GRID_SEED,
    PACK,
    SCHEDULE_SHA256,
    Summary,
    evaluate,
    schedule_sha256,
    summarize,
)
from manabot.env import Match, ObservationSpace
from manabot.experiment.config import Baseline, Case, Experiment
from manabot.experiment.execution import PlannedRun
from manabot.infra.identity import canonical_json, file_sha256
from manabot.infra.provenance import runtime_fingerprints
from manabot.model.world import checkpoint_world
from manabot.training.regime import TrainingRegime
import managym

ROOT = Path(__file__).resolve().parents[1]
REGIME = ROOT / "experiments/regimes/elemental-ur-gw-186-v1.json"
IDENTITY = ROOT / "experiments/protocols/mtg186-identity.json"
STORAGE = "r2://manabot/jobs"
FOLLOW = ROOT / ".runs/mtg186/follow"
RESULTS = ROOT / "experiments/data/mtg186/grids"

SEED = 18601
COHORT = "mtg186"
RENT = "l4-etude"
# MTG-181's longest seed reached 616 updates; exports match its 200-update cadence.
MILESTONES = (200, 400, 600)
# Inclusive authorization for the Task: training, setup, failures and evaluation.
CAP_USD = Decimal(10)
# Etude cohorts are single-use. Earlier cohorts' runs failed before learning
# (protocol amendments); the next cohort gets the cap that remains.
LAUNCHES = {"mtg186": Decimal("0.148"), "mtg186-r2": Decimal("0.05")}
LAUNCH = "mtg186-r3"
# 8.25 hours covers setup, the 7.5-hour training ceiling and finalization. At the
# on-demand's $0.60/hour ceiling plus storage this is at most about $5.10.
MAX_HOURS = 8.25
JOB_USD = Decimal("5.50")

PINNED = (
    "experiments/elemental_grid.py",
    "experiments/mtg186.py",
    "experiments/regimes/elemental-ur-gw-186-v1.json",
    "manabot/eval/match.py",
    "manabot/experiment/job.py",
    "manabot/sim/heuristic.py",
    "manabot/managym/_managym.cpython-312-darwin.so",
    "uv.lock",
)


def regime() -> TrainingRegime:
    return TrainingRegime.model_validate_json(REGIME.read_text())


def experiment() -> Experiment:
    return Experiment(
        COHORT,
        Baseline.capture("elemental-ur-gw-186-v1", regime()),
        cases=(Case("ur-gw", label="UR vs GW only"),),
        jobs=(
            PlannedRun(
                case=f"{COHORT}-ur-gw",
                seed=SEED,
                max_hours=MAX_HOURS,
                max_usd=JOB_USD,
                milestones=MILESTONES,
                validate_numerics=True,
            ),
        ),
    )


def train(revision: str) -> tuple[Run, ...]:
    """The one training run, frozen at an explicit committed revision."""
    return experiment().runs(repo=ROOT, revision=revision)


def identity() -> dict[str, object]:
    """World, roster, schemas, runtime and pinned file digests at this checkout."""
    trained = regime()
    match = Match(trained.match)
    roster = {
        key: managym.authored_deck_setup(PACK, key)
        for key in managym.authored_deck_keys(PACK)
    }
    if schedule_sha256() != SCHEDULE_SHA256:
        raise ValueError("grid schedule differs from MTG-181's frozen digest")
    return {
        "seed": SEED,
        "cohort": COHORT,
        "rent": RENT,
        "cap_usd": str(CAP_USD),
        "launch": LAUNCH,
        "job_usd": str(JOB_USD),
        "max_hours": MAX_HOURS,
        "milestones": list(MILESTONES),
        "regime_digest": Baseline.capture("elemental-ur-gw-186-v1", trained).digest,
        "decks": {
            key: {"deck": dict(setup.decklist), "lesson_pool": dict(setup.sideboard)}
            for key, setup in roster.items()
        },
        "trained_pairs": [
            ["elemental_ur", "elemental_gw"],
            ["elemental_gw", "elemental_ur"],
        ],
        "grid": {
            "decks": list(DECKS),
            "games_per_cell": GAMES_PER_CELL,
            "seed": GRID_SEED,
            "schedule_sha256": SCHEDULE_SHA256,
            "opponent": "heuristic",
        },
        "runtime_fingerprints": runtime_fingerprints(match_config=trained.match),
        "world_binding": checkpoint_world(
            match.to_rust(), ObservationSpace(trained.observation)
        ),
        "sha256": {path: file_sha256(ROOT / path) for path in PINNED},
        "note": (
            "Darwin native bytes bind local evaluation; the pod builds its own "
            "Linux extension from the same source and records it in its TrainingRun."
        ),
    }


# ---------------------------------------------------------------------------
# Evaluation: MTG-186 exports on the full grid, MTG-181 at matched updates on UR/GW

TRAINING_CELLS = frozenset(
    {("elemental_ur", "elemental_gw"), ("elemental_gw", "elemental_ur")}
)
MTG181_CACHE = ROOT.parent / "manabot.mtg-181-trial-two-seat/.runs/mtg181/follow/cache"
WANTED = (0, *MILESTONES)


class Point(BaseModel):
    """One evaluated checkpoint; `cells` is None for the full grid."""

    trial: str
    seed: int
    job: str
    updates: int
    sha256: str
    cells: list[tuple[str, str]] | None = None


class Result(BaseModel):
    point: Point
    schedule_sha256: str
    seconds: float
    summary: Summary


# MTG-181's exports at updates 200/400/600 (all that exist), from its summary.json.
MTG181_MATCHED = (
    Point(
        trial="mtg181",
        seed=18101,
        job="mtg181-elemental-two-seat-18101-r2",
        updates=200,
        sha256="39f244d88396f1a313e46f4a1ca5f81063b9984db7a0b0a5587069ca3f761ab6",
    ),
    Point(
        trial="mtg181",
        seed=18101,
        job="mtg181-elemental-two-seat-18101-r2",
        updates=400,
        sha256="ee45113ff9e752ac6b14a8649559c7c5e6ad250edfbeb744b1b818471e05a8e9",
    ),
    Point(
        trial="mtg181",
        seed=18101,
        job="mtg181-elemental-two-seat-18101-r2",
        updates=600,
        sha256="a452cd255e1729f45af9e400756ea18b1b4d3aeed3581af137629069f6f0ccdb",
    ),
    Point(
        trial="mtg181",
        seed=18102,
        job="mtg181-elemental-two-seat-18102",
        updates=200,
        sha256="8c401238ff9620fc1e2112c825d1f4de1652cd03bebeabdcd597ff891eb41b81",
    ),
    Point(
        trial="mtg181",
        seed=18102,
        job="mtg181-elemental-two-seat-18102",
        updates=400,
        sha256="0a5535a1675b448a58e805e5dc1773da1e76d185c376d1066bffc6d76c87d11f",
    ),
    Point(
        trial="mtg181",
        seed=18103,
        job="mtg181-elemental-two-seat-18103",
        updates=200,
        sha256="3fb4c1c12be7ded1e344ac8f6e9bfc8a2994b578991eabbe61503b322462e4c9",
    ),
    Point(
        trial="mtg181",
        seed=18103,
        job="mtg181-elemental-two-seat-18103",
        updates=400,
        sha256="6b5766243b07dbf6cc49741d1b040881f7d178b66cd9a46883bec006cc7680be",
    ),
)


def _name(point: Point) -> str:
    scope = "grid" if point.cells is None else "urgw"
    return f"{point.trial}-{point.seed}-{point.updates:06d}-{point.sha256[:12]}-{scope}"


def _play(point: Point, checkpoint: Path) -> None:
    """Play one point once; its result file is the durable record."""
    out = RESULTS / f"{_name(point)}.json"
    if out.exists():
        return
    started = time.time()
    cells = None if point.cells is None else frozenset(point.cells)
    rows = evaluate(
        checkpoint,
        point.sha256,
        point.seed,
        FOLLOW / "groups" / _name(point),
        cells=cells,
    )
    result = Result(
        point=point,
        schedule_sha256=SCHEDULE_SHA256,
        seconds=time.time() - started,
        summary=summarize(rows),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.model_dump_json(indent=2) + "\n")
    overall = result.summary.overall
    print(
        f"{_name(point)}: {overall.score} [{overall.low}, {overall.high}]", flush=True
    )


def _cached(sha256: str, body: bytes) -> Path:
    if hashlib.sha256(body).hexdigest() != sha256:
        raise ValueError("checkpoint bytes differ from their recorded digest")
    path = FOLLOW / "cache" / sha256
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial")
        partial.write_bytes(body)
        partial.replace(path)
    return path


def _mtg186_points(run: str) -> tuple[list[tuple[Point, Path]], bool]:
    """Stored exports at wanted updates, plus the endpoint once the run is done."""
    storage = storage_from_uri(STORAGE)
    found = [(e, r) for e, r in stored_runs(storage) if r == run or e == run]
    records = [c for e, r in found for c in read(storage, e, r).checkpoints]
    done = status(run).done
    last = max((c.at.get("updates", -1) for c in records), default=-1)
    points = []
    for record in records:
        updates = int(record.at.get("updates", -1))
        if updates not in WANTED and not (done and updates == last):
            continue
        artifact = record.artifact
        if artifact.sha256 is None:
            continue
        key = artifact.uri.removeprefix(storage.uri(""))
        path = FOLLOW / "cache" / artifact.sha256
        if not path.exists():
            path = _cached(artifact.sha256, storage.get(key))
        point = Point(
            trial="mtg186", seed=SEED, job=run, updates=updates, sha256=artifact.sha256
        )
        points.append((point, path))
    return sorted(points, key=lambda p: p[0].updates), done


def follow(run: str, every: float = 600) -> None:
    """Evaluate oldest update first until the run is done and nothing is left.

    It never rents, cancels or changes training. MTG-181's matched-update cells
    play while MTG-186 has no new export waiting.
    """
    matched = [
        p.model_copy(update={"cells": sorted(TRAINING_CELLS)}) for p in MTG181_MATCHED
    ]
    while True:
        points, done = _mtg186_points(run)
        pending = [p for p in points if not (RESULTS / f"{_name(p[0])}.json").exists()]
        if pending:
            _play(*pending[0])
            continue
        waiting = [p for p in matched if not (RESULTS / f"{_name(p)}.json").exists()]
        if waiting:
            point = waiting[0]
            path = MTG181_CACHE / point.job / point.sha256
            _play(point, _cached(point.sha256, path.read_bytes()))
            continue
        if done:
            return
        time.sleep(every)


def _clean_revision(revision: str) -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    recorded = json.loads(
        subprocess.run(
            ["git", "show", f"{commit}:{IDENTITY.relative_to(ROOT)}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    if recorded != json.loads(canonical_json(identity())):
        raise ValueError("this checkout differs from the revision's identity record")
    return commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("identity", "quote", "launch", "follow"))
    parser.add_argument("--revision")
    parser.add_argument("--run", help="the Etude run to follow")
    args = parser.parse_args()
    if args.command == "identity":
        IDENTITY.write_text(json.dumps(identity(), indent=2, sort_keys=True) + "\n")
        return
    if args.command == "follow":
        if not args.run:
            parser.error("follow needs --run")
        follow(args.run)
        return
    if not args.revision:
        parser.error("quote and launch need --revision")
    runs = list(train(_clean_revision(args.revision)))
    remaining = CAP_USD - sum(LAUNCHES.values())
    priced = quote(runs, rent=RENT, cohort=LAUNCH, max_usd=remaining)
    print(priced.model_dump_json(indent=2))
    if priced.max_usd > JOB_USD:
        raise SystemExit(f"quote {priced.max_usd} exceeds the job's ${JOB_USD}")
    if args.command == "launch":
        print(queue_cohort(LAUNCH, runs, rent=RENT, max_usd=remaining))


if __name__ == "__main__":
    main()
