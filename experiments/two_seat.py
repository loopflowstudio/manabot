"""Two-seat Allies/Lessons on the scale ladder: does learning from both seats pay (MTG-165)?

Jack Heart's ladder experiment (PR #333, raised to $50): the scale-ladder
recipe with both players' decisions retained (`experiments.batched_baseline.
scale_ladder_both`), scored against the heuristic on the two-deck benchmark
(`manabot.eval.benchmark`: 25 deals × 8 deck/seat legs) and directly against
one-seat checkpoints at equal updates (`manabot.eval.duel`). The write-up is
experiments/two-seat-ladder.md.

This file is the experiment `two_seat` for Etude; it defines only `report()`.
The ladder ran under the retired cohort scheduler from a frozen regime
(`experiments/data/two-seat-ladder/frozen-regime.json`), and its benchmark is
bound to each checkpoint's training record, so neither its training nor its
evaluation is yet a `runs()` or `eval(state)` here.

    uv run etude report experiments/two_seat.py --source r2://manabot/jobs --open
"""

from etude.report import Curve, HeadToHead, Matrix, Ranking, Report, Score

from manabot.training.panels import ML, PERF

# The one-seat scale-ladder job the two-seat treatment is compared with.
BASELINE = "scale-ladder-allies-lessons-v1"


def report() -> Report:
    """Score against the heuristic by update, by matchup and seat, and head to head.

    Two decks make four matchups (LvA, AvL, LvL, AvA), read best as small
    multiples. `plays` holds a row per leg and player.
    """
    return Report(
        score=Score(table="plays", value="score", reference=0.5),
        clock="updates",
        baseline=BASELINE,
        evals=(
            Curve(title="Score against the heuristic"),
            HeadToHead(title="Both seats against one seat, direct games"),
            Curve(title="By matchup", split="matchup"),
            Matrix(
                title="Model deck × heuristic deck",
                rows="opponent_deck",
                columns="deck",
            ),
            Ranking(title="On the play or the draw", by="on_play"),
        ),
        perf=PERF,
        ml=ML,
    )
