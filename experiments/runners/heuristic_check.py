"""MTG-152 exploratory check: a retained checkpoint against greedy and heuristic.

Run on the Mini (one process per checkpoint × anchor; each uses one core):

uv run python -m experiments.runners.heuristic_check run \
    --outputs <retained training outputs> --point endpoint --anchor heuristic \
    --out <attempt dir> --deals 25

uv run python -m experiments.runners.heuristic_check report <attempt dir>...

`run` evaluates through the ordinary monitor evaluator with mirrors, so each
deal plays eight legs covering LvA, AvL, LvL and AvA from both seats. Deal
seeds come from MTG-152's own namespace, disjoint from monitoring and MTG-137
families. This is exploratory, not a frozen study: no prediction, no
promotion. `report` prints per-cell scores (bot's deck first, LvA first) with
whole-deal bootstrap intervals, and paired differences between attempts that
share deals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Literal

import numpy as np

from manabot.arena.match import selected_match
from manabot.arena.models import canonical_sha256, file_sha256
from manabot.training.models import ArtifactReference, TrainingCoordinates, TrainingRun
from manabot.training.monitor_evaluation import (
    Checkpoint,
    MonitorProtocol,
    evaluate_checkpoint,
    stage_checkpoint,
)

DEAL_START = 1_520_152_000
DECK_LETTER = {"ur_lessons": "L", "gw_allies": "A"}
CELLS = ("LvA", "AvL", "LvL", "AvA")


def _relocate(artifact: ArtifactReference, outputs: Path) -> ArtifactReference:
    """Resolve the retained export by name, size and digest."""
    matches = [
        path
        for path in outputs.rglob(Path(artifact["path"]).name)
        if path.is_file()
        and path.stat().st_size == artifact["bytes"]
        and file_sha256(path) == artifact["sha256"]
    ]
    if not matches:
        raise ValueError(f"retained checkpoint missing: {artifact['path']}")
    return {**artifact, "path": str(sorted(matches)[0].resolve())}


def _checkpoint(
    run: TrainingRun, outputs: Path, point: Literal["initial", "endpoint"]
) -> Checkpoint:
    if point == "initial":
        return Checkpoint(
            artifact=_relocate(run.stages[0].artifacts["initial_raw"], outputs),
            coordinates=TrainingCoordinates(
                stage_id="initialization", updates=0, training_seconds=0
            ),
        )
    endpoint = stage_checkpoint(run, run.stages[0].id)
    if endpoint is None:
        raise ValueError("run has no endpoint checkpoint")
    endpoint.artifact = _relocate(endpoint.artifact, outputs)
    return endpoint


def _admit(run: TrainingRun, note: Path) -> TrainingRun:
    """Admit an MTG-137 run whose match differs only by an explicit pack.

    MTG-137 sources name `content_pack` on cross-deck setups; this branch's
    selected match leaves it None with identical decks and sideboards. Only
    that field is normalized, both anchors see the same run, and the original
    identity is written beside the attempt.
    """
    recorded = run.identities["matchup_sha256"]
    selected = selected_match()
    match = run.regime.match.model_copy(update={"content_pack": None})
    if recorded == canonical_sha256(selected.model_dump()):
        return run
    if match.model_dump() != selected.model_dump():
        raise ValueError("producer match differs beyond its content pack")
    admitted = run.model_copy(deep=True)
    admitted.regime.match = match
    admitted.identities["matchup_sha256"] = canonical_sha256(selected.model_dump())
    note.write_text(
        json.dumps(
            {
                "recorded_matchup_sha256": recorded,
                "admitted_matchup_sha256": admitted.identities["matchup_sha256"],
                "normalized_field": "regime.match.content_pack",
                "recorded_content_pack": run.regime.match.content_pack,
            },
            indent=2,
        )
    )
    return admitted


def run(args: argparse.Namespace) -> None:
    outputs = Path(args.outputs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    producer = _admit(
        TrainingRun.model_validate_json((outputs / "training/run.json").read_text()),
        out.parent / f"{out.name}.admission.json",
    )
    checkpoint = _checkpoint(producer, outputs, args.point)
    # MTG-137 checkpoints were produced by an older engine source whose world,
    # ABIs and content manifest match this branch; the diagnostic purpose keeps
    # those checks and records the producer's engine source instead of matching it.
    protocol = MonitorProtocol(
        purpose="supplemental-matchup-diagnostic",
        include_mirrors=True,
        deal_seeds=tuple(range(DEAL_START, DEAL_START + args.deals)),
        opponent=args.anchor,
        bootstrap_seed=12290,
        bootstrap_replicates=10000,
    )
    result = evaluate_checkpoint(
        producer,
        checkpoint.artifact,
        checkpoint.coordinates,
        out,
        protocol=protocol,
        concurrent_activity=args.activity,
    )
    print(result.status, result.score, result.error, result.evaluation_seconds)


def _cells(monitor: dict) -> dict[str, dict[int, list[float]]]:
    """Cell -> deal seed -> candidate scores (two legs per deal and cell)."""
    out: dict[str, dict[int, list[float]]] = {cell: {} for cell in CELLS}
    for row in monitor["rows"]:
        seat = row["player_a_seat"]
        decks = row["seat_decks"]
        cell = f"{DECK_LETTER[decks[seat]]}v{DECK_LETTER[decks[1 - seat]]}"
        if row["score_a"] is not None:
            out[cell].setdefault(row["deal_seed"], []).append(row["score_a"])
    return out


def _interval(
    per_deal: np.ndarray, rng: np.random.Generator
) -> tuple[float, float, float]:
    draws = per_deal[rng.integers(0, len(per_deal), (10000, len(per_deal)))].mean(
        axis=1
    )
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return float(per_deal.mean()), float(lower), float(upper)


def report(args: argparse.Namespace) -> None:
    attempts = {
        Path(path).name: json.loads((Path(path) / "monitor.json").read_text())
        for path in args.attempts
    }
    rng = np.random.default_rng(12290)
    cells = {name: _cells(monitor) for name, monitor in attempts.items()}
    print("attempt".ljust(28), *(c.ljust(20) for c in CELLS), "games status")
    for name, monitor in attempts.items():
        parts = []
        for cell in CELLS:
            deals = cells[name][cell]
            if not deals:
                parts.append("absent".ljust(20))
                continue
            mean, lower, upper = _interval(
                np.array([np.mean(v) for v in deals.values()]), rng
            )
            parts.append(f"{mean:.2f} [{lower:.2f},{upper:.2f}]".ljust(20))
        print(name.ljust(28), *parts, len(monitor["rows"]), monitor["status"])
    names = list(attempts)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            parts = []
            for cell in CELLS:
                shared = sorted(set(cells[first][cell]) & set(cells[second][cell]))
                if not shared:
                    parts.append("no shared deals".ljust(20))
                    continue
                diff = np.array(
                    [
                        np.mean(cells[second][cell][s]) - np.mean(cells[first][cell][s])
                        for s in shared
                    ]
                )
                mean, lower, upper = _interval(diff, rng)
                parts.append(f"{mean:+.2f} [{lower:+.2f},{upper:+.2f}]".ljust(20))
            print(f"{second} - {first}"[:28].ljust(28), *parts)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--outputs", required=True)
    run_parser.add_argument("--point", choices=("initial", "endpoint"), required=True)
    run_parser.add_argument(
        "--anchor", choices=("scripted_greedy", "heuristic"), required=True
    )
    run_parser.add_argument("--out", required=True)
    run_parser.add_argument("--deals", type=int, default=25)
    run_parser.add_argument("--activity", default="MTG-152 quick check on idle Mini")
    report_parser = commands.add_parser("report")
    report_parser.add_argument("attempts", nargs="+")
    args = parser.parse_args(argv)
    {"run": run, "report": report}[args.command](args)


if __name__ == "__main__":
    main(sys.argv[1:])
