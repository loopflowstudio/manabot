"""MTG-183's evaluation page, drawn by Etude: experiment `mtg183-elemental-eval`.

Etude names an experiment by its file's stem, so this file is named for the
experiment its eval runs record. It only reports; the follower
(`experiments.elemental_long_eval`) and the rented batches
(`experiments.elemental_long_rent`) play it. Seats declare each checkpoint's
update and its line, so the evaluation-only page draws curves over updates
(`experiments.elemental_long_db.normalize` declares them on older records).

The Opponent control selects the heuristic. The former 400-updates-earlier
and half-updates curves cannot be selected under Etude 0.3.1: these historical
matches carry neither seating names nor tags identifying those comparisons.

    uv run --extra evaluation python -m experiments.elemental_long_db build
    ETUDE_HOME=~/.manabot/mtg183-eval-db/etude uv run --extra evaluation \\
        etude report experiments/reports/mtg183-elemental-eval.py \\
        --source ~/.manabot/mtg183-eval-db/store --out ~/.manabot/mtg183-eval-db/reports --open
"""

from etude.report import Anchors, Crosstable, Curve, Report, Score


def report() -> Report:
    return Report(
        # `games` holds one row per game from player 0's seat: the rated (or
        # newer) checkpoint's score, 1 for a win and 0.5 for a draw.
        score=Score(table="games", value="score", reference=0.5),
        clock="updates",
        evals=(
            Curve(title="Against heuristic v2"),
            Crosstable(),
            Anchors(),
        ),
    )
