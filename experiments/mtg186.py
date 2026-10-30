"""MTG-186: one two-seat seed trained only on UR vs GW, compared with MTG-181.

Jack Heart's question (2026-10-25): does training on the one matchup beat
MTG-181's training on all ten Elemental decks, on that matchup and across the
other 99 grid cells? The protocol is experiments/protocols/mtg186.md; the
recipe is experiments/regimes/elemental-ur-gw-186-v1.json, MTG-181's frozen
recipe with both ordered UR/GW pairs in place of the uniform ten-deck roster.

`train` builds the single Etude training run (one L4, update target 616,
exports at 0/200/400/600 and the endpoint). `identity` writes the pinned
identity record the protocol commits before allocation. Evaluation is
`experiments.mtg186_eval`, on rented CPU through Etude.

    uv run python -m experiments.mtg186 identity
    uv run python -m experiments.mtg186 quote --revision <commit>
    doppler run -p manabot -c prd -- doppler run -p etude -c prd -- uv run python -m experiments.mtg186 launch --revision <commit>

Launching needs Etude's RunPod key and the R2 keys.
"""

import argparse
from decimal import Decimal
import json
from pathlib import Path
import subprocess

from etude.launch.client import queue_cohort, quote
from etude.launch.records import Run

from experiments.elemental_grid import (
    DECKS,
    GAMES_PER_CELL,
    GRID_SEED,
    PACK,
    SCHEDULE_SHA256,
    schedule_sha256,
)
from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
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
    parser.add_argument("command", choices=("identity", "quote", "launch"))
    parser.add_argument("--revision")
    args = parser.parse_args()
    if args.command == "identity":
        IDENTITY.write_text(json.dumps(identity(), indent=2, sort_keys=True) + "\n")
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
