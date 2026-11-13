"""Report helpers name jobs, ask one question per chart and draw planned schedules."""

from pathlib import Path

from etude.report import Curve, Matches, Report, Score
import pytest

from experiments.tla_frontier import report
from manabot.training.regime import (
    DampedPolicyLearning,
    Learning,
    TrainingRegime,
    TrainSelfPlay,
)
from manabot.training.report import (
    against_earlier_self,
    against_fixed,
    head_to_head,
    job_labels,
    learning_schedule,
)

REGIME = (
    Path(__file__).resolve().parents[2]
    / "experiments/regimes/tla-frontier-c5fba4-w64.json"
)


def _regime(*, width: int = 64, layers: int | None = None) -> TrainingRegime:
    regime = TrainingRegime.model_validate_json(REGIME.read_bytes())
    agent = regime.agent.model_copy(
        update={
            "hidden_dim": width,
            "attention_layers": layers or regime.agent.attention_layers,
        }
    )
    return regime.model_copy(update={"agent": agent})


def _learning(regime: TrainingRegime) -> DampedPolicyLearning:
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    assert isinstance(stage.learning, DampedPolicyLearning)
    return stage.learning


def test_labels_name_width_and_only_a_differing_depth() -> None:
    small = _regime()
    large = _regime(width=160, layers=8)

    assert job_labels({"w64": ("Small", small), "w160l8": ("Large", large)}) == {
        "w64": "Small (width 64)",
        "w160l8": "Large (width 160, 8 layers)",
    }
    assert job_labels({"a": ("Wide", _regime(width=128))}) == {"a": "Wide (width 128)"}


def test_comparisons_draw_one_chart_per_question_across_jobs() -> None:
    earlier = against_earlier_self("Vs itself", tags=("half-w64", "half-w160"))
    fixed = against_fixed("Vs heuristic", tags=("heur-w64", "heur-w160"))
    pair = head_to_head(
        "Large vs Small", ("large u25 vs small u50", "large u50 vs small u100")
    )

    assert earlier.matches == (Matches(tag="half-w64"), Matches(tag="half-w160"))
    assert fixed.matches == (Matches(tag="heur-w64"), Matches(tag="heur-w160"))
    assert pair.matches == (
        Matches(seating="large u25 vs small u50"),
        Matches(seating="large u50 vs small u100"),
    )
    # They compose into an ordinary Etude report.
    Report(score=Score(table="games", value="score"), evals=(earlier, fixed, pair))


def test_head_to_head_refuses_an_empty_selection() -> None:
    with pytest.raises(ValueError, match="at least one seating"):
        head_to_head("Large vs Small", ())


def test_frontier_selects_only_recorded_head_to_head_seatings() -> None:
    hours, updates = report().evals[2:]
    assert isinstance(hours, Curve)
    assert isinstance(updates, Curve)
    assert hours.matches == tuple(
        Matches(seating=name)
        for name in (
            "w160l8 u25 vs w64 u50 at equal hours",
            "w160l8 u50 vs w64 u100 at equal hours",
            "w160l8 u75 vs w64 u175 at equal hours",
            "w160l8 u100 vs w64 u225 at equal hours",
            "w160l8 u150 vs w64 u350 at equal hours",
            "w160l8 u200 vs w64 u475 at equal hours",
            "w160l8 u300 vs w64 u700 at equal hours",
            "w160l8 u400 vs w64 u875 at equal hours",
            "w160l8 u600 vs w64 u1200 at equal hours",
            "w160l8 u800 vs w64 u1525 at equal hours",
            "w160l8 final u925 vs w64 u1700 at equal hours",
        )
    )
    assert updates.matches == tuple(
        Matches(seating=f"w160l8 u{update} vs w64 u{update}")
        for update in (25, 50, 75, 100, 150, 200, 300, 400, 600, 800)
    )


def test_schedule_draws_the_learners_own_rates_to_the_horizon() -> None:
    regime = _regime()
    learning = _learning(regime)
    assert learning.iteration_schedule is not None
    horizon = learning.iteration_schedule.steps

    rate, damping = learning_schedule({"w64": regime})

    points = rate.jobs["w64"]
    assert (points[0].at, points[-1].at) == (1, horizon)
    for point in (points[0], points[len(points) // 2], points[-1]):
        assert point.value == learning.rates(int(point.at))[0]
    assert damping.jobs["w64"][-1].value == learning.rates(horizon)[1]
    assert rate.clock == damping.clock == "updates"


def test_schedule_refuses_a_regime_without_damped_learning() -> None:
    regime = _regime()
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    ppo = regime.model_copy(
        update={"stages": [stage.model_copy(update={"learning": Learning()})]}
    )
    with pytest.raises(ValueError, match="damped"):
        learning_schedule({"ppo": ppo})
