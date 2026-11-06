"""Project the frozen GPU study into Etude's offline report, without running games.

The original monitors predate this study's Etude eval integration, and its
tracker declarations omit treatment groups. This read-side adapter verifies
admitted monitors, preserves their statistics, and supplies arm/seed identity
only in the report. Source receipts and declarations are never rewritten.

Run from the repository root:
uv run --extra reports python -m scripts.report_representation_gpu
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from etude.page import Cell, Coverage, Environment, Page, Player, render, write
from etude.report import Curve, Flame, Matches, Report, Score
from etude.run.records import Clock
from etude.train.read import RunData, read_all
from etude.train.report import Job, Mark, Training, training

from experiments.state_representation_gpu import (
    ASSIGNMENTS,
    ENDPOINTS,
    EXPERIMENT,
    Endpoint,
    endpoint_attempts,
    evaluation_protocol,
    recipe_digests,
    verified_result,
)
from experiments.state_representation_v7 import Arm, ArmEvaluation
from manabot.eval.monitor import MonitorResult
from manabot.training.panels import ML, PERF

DEFAULT_SOURCE = Path(
    "experiments/data/state-representation-gpu-5da5f4/recovery-20261105"
)
DEFAULT_OUTPUT = Path(".runs/representation/gpu-report/state_representation_gpu.html")
ARM_NAMES: dict[Arm, str] = {"typed_pool": "Pool", "typed_gru": "GRU"}
CURVE = "Learning curve · deals 85001–85032"
HELDOUT = "Held-out final · deals 86001–86064"


@dataclass(frozen=True)
class EvaluatedEndpoint:
    arm: Arm
    endpoint: Endpoint
    result: MonitorResult


def read_endpoints(root: Path) -> tuple[EvaluatedEndpoint, ...]:
    """Require the entire frozen cohort, one hash-verified result per endpoint."""
    found: list[EvaluatedEndpoint] = []
    recipes = recipe_digests()
    for arm, seed in ASSIGNMENTS:
        for endpoint in ENDPOINTS:
            results = [
                result
                for path in endpoint_attempts(root / "ledger", arm, seed, endpoint)
                if (result := verified_result(path)) is not None
            ]
            if len(results) != 1:
                raise ValueError(
                    f"{arm}/{seed}/{endpoint.name}: expected one verified result"
                )
            result = results[0]
            if (
                result.training_seed != seed
                or result.coordinates.updates != endpoint.update
            ):
                raise ValueError(
                    "evaluation seed or checkpoint does not match its ledger"
                )
            ArmEvaluation(arm=arm, result=result).score_for(
                evaluation_protocol(heldout=endpoint.heldout), recipes[arm]
            )
            found.append(EvaluatedEndpoint(arm, endpoint, result))
    return tuple(found)


def grouped_training(root: Path, endpoints: tuple[EvaluatedEndpoint, ...]) -> Training:
    """Group real telemetry by treatment; missing telemetry never becomes zeros.

    Evaluations retain checkpoint coordinates for all eight learners. Those
    coordinates supply marks even for learners whose metric blocks were lost.
    Elapsed checkpoint time is the monitor's measured training_seconds, not a
    reconstructed timestamp. Perf excludes checkpoint wall-time/pace panels;
    logged metric series retain their own actual tracker wall clock.
    """
    identities = {e.result.run_id: (e.arm, e.result.training_seed) for e in endpoints}
    if len(identities) != len(ASSIGNMENTS) or set(identities.values()) != set(
        ASSIGNMENTS
    ):
        raise ValueError("each arm/seed must identify one distinct learner")
    runs: list[RunData] = []
    for run in read_all(root, EXPERIMENT):
        if run.name not in identities or run.file is None:
            raise ValueError(f"tracker outside admitted cohort: {run.name}")
        arm, seed = identities[run.name]
        declaration = run.file.declaration
        if declaration.seed != seed:
            raise ValueError("tracker seed differs from the admitted evaluation")
        # In-memory presentation only; config bytes, identity and source stay intact.
        declaration = declaration.model_copy(
            update={
                "job": arm,
                "job_name": ARM_NAMES[arm],
                "clocks": tuple(
                    sorted(declaration.clocks, key=lambda c: c.name != "updates")
                ),
            }
        )
        runs.append(
            run.model_copy(
                update={
                    "file": run.file.model_copy(update={"declaration": declaration})
                }
            )
        )
    result = training(runs, prefixes=[p.prefix for p in PERF if isinstance(p, Flame)])
    jobs = tuple(
        Job(
            key=arm,
            name=name,
            label=f"{name} · 4 seeds",
            runs=tuple(
                run_id
                for run_id, (a, seed) in sorted(
                    identities.items(), key=lambda item: item[1][1]
                )
                if a == arm
            ),
        )
        for arm, name in ARM_NAMES.items()
    )
    marks = tuple(
        Mark(
            run=e.result.run_id,
            ordinal=e.endpoint.update,
            at={
                "updates": e.endpoint.update,
                "learner_transitions": e.result.coordinates.learner_transitions,
            },
            align=e.endpoint.update,
            elapsed=e.result.coordinates.training_seconds,
            pace={},
        )
        for e in endpoints
        if not e.endpoint.heldout
    )
    return result.model_copy(
        update={
            "jobs": jobs,
            "checkpoints": marks,
            "clocks": (
                Clock(
                    name="updates",
                    kind="step",
                    meaning="training updates (256 games each)",
                ),
                Clock(
                    name="learner_transitions",
                    kind="step",
                    meaning="learner transitions",
                ),
            ),
        }
    )


def build_report(root: Path) -> Page:
    endpoints = read_endpoints(root)
    trained = grouped_training(root, endpoints)
    players = [
        Player(
            key="heuristic", name="Fixed heuristic", run=None, ordinal=None, config="{}"
        )
    ]
    subjects: dict[tuple[str, int], int] = {}
    cells: list[Cell] = []
    coverage: list[Coverage] = []
    latest: list[float] = []
    for entry in endpoints:
        result, endpoint = entry.result, entry.endpoint
        key = (result.run_id, endpoint.update)
        if key not in subjects:
            subjects[key] = len(players)
            players.append(
                Player(
                    key=result.artifact["sha256"],
                    name=f"{ARM_NAMES[entry.arm]} seed {result.training_seed} · update {endpoint.update}",
                    run=result.run_id,
                    ordinal=endpoint.update,
                    config="{}",
                    at={
                        "updates": endpoint.update,
                        "learner_transitions": result.coordinates.learner_transitions,
                    },
                    line=f"{ARM_NAMES[entry.arm]} seed {result.training_seed}",
                )
            )
        elif players[subjects[key]].key != result.artifact["sha256"]:
            raise ValueError(
                "curve and held-out evaluation name different final checkpoints"
            )
        score = result.score
        assert score is not None  # read_endpoints admits only complete scored results.
        cells.append(
            Cell(
                selection=int(endpoint.heldout),
                commit=None,
                environment=0,
                score=0,
                subject=subjects[key],
                opponent=0,
                fixed=False,
                split={},
                value=score.mean,
                interval=(score.lower, score.upper),
                rows=len(result.rows),
                matches=len(result.rows),
            )
        )
        coverage.append(
            Coverage(
                run=result.run_id,
                ordinal=endpoint.update,
                tournament=HELDOUT if endpoint.heldout else CURVE,
                played=len(result.rows),
                failed=0,
                unplayed=0,
            )
        )
        if result.finished_unix is not None:
            latest.append(result.finished_unix)
    missing = [
        f"{ARM_NAMES[e.arm]} {e.result.training_seed}"
        for e in endpoints
        if e.endpoint.update == 0
        and e.result.run_id not in {r.name for r in trained.runs}
    ]
    score = Score(table="retained_monitor", value="score", reference=0.5)
    return Page(
        experiment=EXPERIMENT,
        file="experiments/state_representation_gpu.py",
        source=str(root),
        report=Report(
            clock="updates",
            score=score,
            evals=(
                Curve(
                    title="Learning curve · all four seeds per arm",
                    matches=Matches(tournament=CURVE),
                ),
                Curve(
                    title="Held-out final · all four seeds per arm",
                    matches=Matches(tournament=HELDOUT),
                ),
            ),
            perf=tuple(p for p in PERF if p.kind not in {"elapsed", "pace"}),
            ml=ML,
        ),
        origin="scripts/report_representation_gpu.py · projection of verified retained monitors",
        notes=(
            "72 curve endpoints + 8 held-out endpoints; 11,264 replay-validated games. No new games were run.",
            "Pool and GRU each contain seeds 74211–74214. Align on updates: 256 games per update; 65,536 per learner.",
            "Bands show the range across training seeds, not a confidence interval on the arm difference. Individual endpoint intervals retain the deal-block bootstrap.",
            "Held-out GRU minus Pool: −1.95 percentage points; paired-seed 95% interval [−5.88, +1.98]. No winner or qualified plateau.",
            f"All 8 learners have evaluations; detailed training metrics survive for only {len(trained.runs)}. Missing logs: {', '.join(missing)}. Perf/ML means use available logs only (Pool 3, GRU 2).",
            "This is an offline report projection, not new Etude evaluation execution. Frozen monitor receipts and tracker declarations are unchanged. Failed attempts remain in the study ledger and scientific report; curves show admitted endpoints only.",
        ),
        wall_meaning="Tracker metrics: seconds since tracker start. Evaluation checkpoint wall axis: saved training_seconds, excluding setup.",
        training=trained,
        scores=(score,),
        selections=(Matches(tournament=CURVE), Matches(tournament=HELDOUT)),
        players=tuple(players),
        environments=(Environment(name="w17 · UR Lessons / GW Allies", config={}),),
        commits=(),
        cells=tuple(cells),
        coverage=tuple(coverage),
        failures=(),
        latest_result_ms=int(max(latest) * 1000) if latest else None,
        absent=(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(args.source)
    print(write(render(report), args.out))


if __name__ == "__main__":
    main()
