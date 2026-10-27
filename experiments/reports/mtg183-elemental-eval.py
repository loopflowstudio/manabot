"""MTG-183's evaluation page, drawn by Etude: experiment `mtg183-elemental-eval`.

Etude names an experiment by its file's stem, so this file is named for the
experiment its eval runs record. It only reports; the follower
(`experiments.elemental_long_eval`) and the rented batches
(`experiments.elemental_long_rent`) play it. Seats declare each checkpoint's
update and its line, so the evaluation-only page draws curves over updates
(`experiments.elemental_long_db.place` declares them on older records).

    uv run --extra evaluation python -m experiments.elemental_long_db build
    uv run etude report experiments/reports/mtg183-elemental-eval.py \\
        --source ~/.manabot/mtg183-eval-db/runs --out ~/.manabot/mtg183-eval-db/reports --open
"""

from etude.report import Anchors, Crosstable, Curve, Earlier, Report, Score


def report() -> Report:
    return Report(
        # `games` holds one row per game from player 0's seat: the rated (or
        # newer) checkpoint's score, 1 for a win and 0.5 for a draw.
        score=Score(table="games", value="score", reference=0.5),
        clock="updates",
        evals=(
            Curve(title="Against heuristic v2", opponent="heuristic"),
            Curve(
                title="Against the checkpoint 400 updates earlier",
                opponent=Earlier(clock="updates", by=400),
            ),
            Crosstable(),
            Anchors(),
        ),
    )
