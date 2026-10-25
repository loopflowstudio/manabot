"""MTG-186 report: the UR-vs-GW specialist against MTG-181's ten-deck seeds.

Reads the follower's grid results (`experiments/data/mtg186/grids/`), MTG-181's
endpoint grids against the same heuristic (`experiments/data/mtg181/new-heuristic/`)
and the training run's TrainingRun receipt, and writes `report.md` and
`summary.json` beside the grids. Every number in the report comes from those
files; nothing is selected.

    uv run python -m experiments.mtg186_report --run-json <TrainingRun run.json> \
        --usd <inclusive dollars> --mtg181 experiments/data/mtg181/new-heuristic \
        [--mtg181-groups <MTG-181 new-heuristic group directory>]

`report.md` is written by hand from the generated `tables.md` and `summary.json`.
"""

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from pydantic import BaseModel

from experiments.elemental_grid import (
    DECKS,
    GameRow,
    Score,
    Summary,
    grid_table,
    wilson,
)
from experiments.mtg186 import RESULTS, ROOT, Result
from manabot.training.records import TrainingRun

OUT = RESULTS.parent
CELLS = (("elemental_ur", "elemental_gw"), ("elemental_gw", "elemental_ur"))


class MTG181Endpoint(BaseModel):
    seed: int
    job: str
    updates: int
    sha256: str


class MTG181Result(BaseModel):
    endpoint: MTG181Endpoint
    schedule_sha256: str
    summary: Summary


class Training(BaseModel):
    """Counters of the one training stage, both clocks kept separate."""

    status: str
    updates: int
    learner_decisions: int
    environment_decisions: int | None
    games: int
    collection_hours: float
    learning_hours: float
    learner_decisions_per_active_hour: float
    error: str | None


class Row(BaseModel):
    """One checkpoint's view: training-cell scores, play/draw and overall."""

    trial: str
    seed: int
    updates: int
    scope: str
    ur_vs_gw: Score
    gw_vs_ur: Score
    ur_vs_gw_play: Score
    ur_vs_gw_draw: Score
    gw_vs_ur_play: Score
    gw_vs_ur_draw: Score
    overall: Score | None


class Report(BaseModel):
    training: Training | None
    inclusive_usd: float | None
    rows: list[Row]


def _rows_by_seat(groups: Path, x: str, y: str, seat: int) -> Score:
    """Per-cell play/draw from retained group rows; summaries keep only grid totals."""
    target = groups / f"{x}--{y}--seat{seat}" / "games.json"
    if not target.exists():
        return wilson([])
    return wilson([GameRow.model_validate(r) for r in json.loads(target.read_text())])


def _row(
    trial: str,
    seed: int,
    updates: int,
    scope: str,
    summary: Summary,
    groups: Path | None,
) -> Row:
    seats = {
        f"{name}_{label}": (
            _rows_by_seat(groups, x, y, seat) if groups is not None else wilson([])
        )
        for name, (x, y) in zip(("ur_vs_gw", "gw_vs_ur"), CELLS, strict=True)
        for label, seat in (("play", 0), ("draw", 1))
    }
    return Row(
        trial=trial,
        seed=seed,
        updates=updates,
        scope=scope,
        ur_vs_gw=summary.cells[CELLS[0][0]][CELLS[0][1]],
        gw_vs_ur=summary.cells[CELLS[1][0]][CELLS[1][1]],
        overall=summary.overall if scope == "grid" else None,
        **seats,
    )


def training(path: Path) -> Training:
    run = TrainingRun.model_validate_json(path.read_text())
    [stage] = run.stages
    active = stage.collection_seconds + stage.learning_seconds
    return Training(
        status=run.status,
        updates=stage.updates,
        learner_decisions=stage.learner_transitions,
        environment_decisions=stage.environment_decisions,
        games=stage.games,
        collection_hours=stage.collection_seconds / 3600,
        learning_hours=stage.learning_seconds / 3600,
        learner_decisions_per_active_hour=stage.learner_transitions * 3600 / active
        if active
        else 0.0,
        error=run.error,
    )


def _fmt(score: Score | None) -> str:
    if score is None or score.score is None:
        return "—"
    invalid = f", {score.invalid} invalid" if score.invalid else ""
    return f"{score.score:.2f} [{score.low:.2f}, {score.high:.2f}]{invalid}"


def _table(rows: Sequence[Row]) -> str:
    lines = [
        "| Trial | Seed | Update | UR vs h-GW | (play / draw) | GW vs h-UR | (play / draw) | Overall |",
        "|---|---:|---:|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (r.trial != "mtg186", r.updates, r.seed)):
        lines.append(
            f"| {r.trial} | {r.seed} | {r.updates} | {_fmt(r.ur_vs_gw)} | "
            f"{_fmt(r.ur_vs_gw_play)} / {_fmt(r.ur_vs_gw_draw)} | {_fmt(r.gw_vs_ur)} | "
            f"{_fmt(r.gw_vs_ur_play)} / {_fmt(r.gw_vs_ur_draw)} | {_fmt(r.overall)} |"
        )
    return "\n".join(lines)


def _margins(summaries: dict[str, Summary]) -> str:
    names = list(summaries)
    head = (
        "| Deck | "
        + " | ".join(f"{n}: model plays | {n}: heuristic plays" for n in names)
        + " |"
    )
    rule = "|---|" + "---:|---:|" * len(names)
    lines = [head, rule]
    for deck in DECKS:
        cells = []
        for name in names:
            s = summaries[name]
            for score in (s.model_decks[deck], s.heuristic_decks[deck]):
                cells.append("—" if score.score is None else f"{score.score:.3f}")
        lines.append(
            f"| {deck.removeprefix('elemental_').upper()} | " + " | ".join(cells) + " |"
        )
    return "\n".join(lines)


def build(
    run_json: Path | None, usd: float | None, mtg181: Path, mtg181_groups: Path | None
) -> tuple[Report, dict[str, Summary]]:
    """Rows for every evaluated checkpoint, and the full grids by label."""
    follow_groups = ROOT / ".runs/mtg186/follow/groups"
    rows: list[Row] = []
    grids: dict[str, Summary] = {}
    for path in sorted(RESULTS.glob("*.json")):
        result = Result.model_validate_json(path.read_text())
        point = result.point
        scope = "grid" if point.cells is None else "urgw"
        groups = follow_groups / path.stem
        rows.append(
            _row(
                point.trial,
                point.seed,
                point.updates,
                scope,
                result.summary,
                groups if groups.exists() else None,
            )
        )
        if point.trial == "mtg186" and scope == "grid":
            grids[f"MTG-186 u{point.updates}"] = result.summary
    for path in sorted(mtg181.glob("*.json")):
        other = MTG181Result.model_validate_json(path.read_text())
        end = other.endpoint
        groups = None if mtg181_groups is None else mtg181_groups / path.stem / "groups"
        rows.append(
            _row(
                "mtg181",
                end.seed,
                end.updates,
                "grid",
                other.summary,
                groups if groups is not None and groups.exists() else None,
            )
        )
        grids[f"MTG-181 {end.seed} u{end.updates}"] = other.summary
    report = Report(
        training=training(run_json) if run_json is not None else None,
        inclusive_usd=usd,
        rows=rows,
    )
    return report, grids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-json", type=Path)
    parser.add_argument("--usd", type=float)
    parser.add_argument("--mtg181", type=Path, required=True)
    parser.add_argument("--mtg181-groups", type=Path)
    args = parser.parse_args()
    report, grids = build(args.run_json, args.usd, args.mtg181, args.mtg181_groups)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(report.model_dump_json(indent=2) + "\n")
    parts = [
        "# MTG-186 results (generated tables)",
        "",
        "## UR/GW cells",
        "",
        _table(report.rows),
    ]
    if grids:
        parts += ["", "## Deck margins", "", _margins(grids), "", "## Grids", ""]
        parts += [grid_table(s, name) for name, s in grids.items()]
    (OUT / "tables.md").write_text("\n".join(parts) + "\n")
    print(_table(report.rows))


if __name__ == "__main__":
    main()
