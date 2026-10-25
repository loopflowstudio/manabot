"""MTG-186 report: the UR-vs-GW specialist head-to-head against MTG-181 seed 18101.

Reads each collected pairing (`experiments/data/mtg186/pairings/`, see
`experiments.mtg186_eval`) and the training run's TrainingRun receipt, and writes
`tables.md` and `summary.json` beside the pairings. Every score is MTG-186's
(win 1, draw ½, loss 0) over valid games with a 95% Wilson interval; invalid
games are counted, never replaced. Nothing is selected.

    uv run python -m experiments.mtg186_report --run-json <TrainingRun run.json> --usd <inclusive dollars>

`report.md` is written by hand from the generated `tables.md` and `summary.json`.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from experiments.elemental_grid import Score, wilson
from experiments.mtg186_eval import GW, ORDER, RESULTS, UR, PairResult
from manabot.training.records import TrainingRun

OUT = RESULTS.parent


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
    """One pairing: MTG-186's scores on UR/GW (and its splits) and on the rest."""

    pairing: str
    mtg186_updates: int
    mtg181_updates: int
    urgw: Score
    mtg186_on_ur: Score
    mtg186_on_gw: Score
    mtg186_on_play: Score
    mtg186_on_draw: Score
    # The four combinations: MTG-186's deck and whether it is on the play.
    ur_play: Score
    ur_draw: Score
    gw_play: Score
    gw_draw: Score
    rest: Score
    rest_on_play: Score
    rest_on_draw: Score
    play_seconds: float


class Report(BaseModel):
    training: Training | None
    inclusive_usd: float | None
    rows: list[Row]


def row(result: PairResult) -> Row:
    def score(urgw: bool, deck: str | None = None, seat: int | None = None) -> Score:
        """MTG-186's score on UR/GW (or the rest), optionally on one deck or seat."""
        return wilson(
            [
                r
                for r in result.rows
                if ((r.model_deck, r.heuristic_deck) in ((UR, GW), (GW, UR))) == urgw
                and deck in (None, r.model_deck)
                and seat in (None, r.model_seat)
            ]
        )

    return Row(
        pairing=result.pairing.name,
        mtg186_updates=result.pairing.mtg186.updates,
        mtg181_updates=result.pairing.mtg181.updates,
        urgw=score(True),
        mtg186_on_ur=score(True, deck=UR),
        mtg186_on_gw=score(True, deck=GW),
        mtg186_on_play=score(True, seat=0),
        mtg186_on_draw=score(True, seat=1),
        ur_play=score(True, UR, 0),
        ur_draw=score(True, UR, 1),
        gw_play=score(True, GW, 0),
        gw_draw=score(True, GW, 1),
        rest=score(False),
        rest_on_play=score(False, seat=0),
        rest_on_draw=score(False, seat=1),
        play_seconds=result.seconds,
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


def _fmt(score: Score) -> str:
    if score.score is None:
        return f"— ({score.invalid} invalid)"
    invalid = f", {score.invalid} invalid" if score.invalid else ""
    return f"{score.score:.3f} [{score.low:.3f}, {score.high:.3f}] n={score.valid}{invalid}"


def _table(rows: Sequence[Row], columns: Sequence[tuple[str, str]]) -> str:
    lines = [
        "| Pairing | " + " | ".join(title for title, _ in columns) + " |",
        "|---|" + "---|" * len(columns),
    ]
    for r in rows:
        values = (_fmt(getattr(r, field)) for _, field in columns)
        lines.append(
            f"| MTG-186 u{r.mtg186_updates} vs 18101 u{r.mtg181_updates} | "
            + " | ".join(values)
            + " |"
        )
    return "\n".join(lines)


def build(run_json: Path | None, usd: float | None) -> Report:
    found = {
        p.stem: PairResult.model_validate_json(p.read_text())
        for p in RESULTS.glob("*.json")
    }
    return Report(
        training=training(run_json) if run_json is not None else None,
        inclusive_usd=usd,
        rows=[row(found[name]) for name in ORDER if name in found],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-json", type=Path)
    parser.add_argument("--usd", type=float)
    args = parser.parse_args()
    report = build(args.run_json, args.usd)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(report.model_dump_json(indent=2) + "\n")
    tables = [
        "# MTG-186 results (generated tables)",
        "",
        "Scores are MTG-186's against MTG-181 seed 18101, valid games only, 95% Wilson.",
        "",
        "## UR/GW (1,000 games) and the rest (1,000 games)",
        "",
        _table(
            report.rows,
            (
                ("UR/GW", "urgw"),
                ("MTG-186 on UR", "mtg186_on_ur"),
                ("MTG-186 on GW", "mtg186_on_gw"),
                ("Rest, pooled", "rest"),
            ),
        ),
        "",
        "## UR/GW by play/draw",
        "",
        _table(
            report.rows,
            (
                ("On the play", "mtg186_on_play"),
                ("On the draw", "mtg186_on_draw"),
                ("UR, play", "ur_play"),
                ("UR, draw", "ur_draw"),
                ("GW, play", "gw_play"),
                ("GW, draw", "gw_draw"),
            ),
        ),
        "",
        "## Rest by play/draw",
        "",
        _table(
            report.rows,
            (("On the play", "rest_on_play"), ("On the draw", "rest_on_draw")),
        ),
    ]
    (OUT / "tables.md").write_text("\n".join(tables) + "\n")
    print("\n".join(tables))


if __name__ == "__main__":
    main()
