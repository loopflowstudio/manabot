"""Report helpers name jobs, ask one question per chart."""

from pathlib import Path
import subprocess
import sys

from etude.report import Curve, Matches, Report, Score, Series
from etude.train.read import read
import pytest

from experiments.tla_frontier import report
from manabot.infra.config import ModelConfig
from manabot.recipes.builders import gray_ogre_self_play
from manabot.training.loops.self_play import CLOCKS, self_play
from manabot.training.report import (
    against_earlier_self,
    against_fixed,
    behaviour_curves,
    head_to_head,
    job_labels,
)
from tests.fixtures.training_records import recording


def _model(*, width: int = 64, layers: int = 2) -> ModelConfig:
    return ModelConfig(hidden_dim=width, attention_layers=layers)


def test_labels_name_width_and_only_a_differing_depth() -> None:
    small = _model()
    large = _model(width=160, layers=8)

    assert job_labels({"w64": ("Small", small), "w160l8": ("Large", large)}) == {
        "w64": "Small (width 64)",
        "w160l8": "Large (width 160, 8 layers)",
    }
    assert job_labels({"a": ("Wide", _model(width=128))}) == {"a": "Wide (width 128)"}


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
    behaviour = behaviour_curves(matches=Matches(tag="heur-w64"))
    assert len(behaviour) == 11
    assert all(
        c.matches == Matches(tag="heur-w64")
        and c.score is not None
        and c.score.table == "behaviour"
        and c.score.unit == "games"
        for c in behaviour
    )
    assert {c.title for c in behaviour if c.score and c.score.range is None} == {
        "Turns per game"
    }
    assert all(
        c.score and c.score.range == (0.0, 1.0)
        for c in behaviour
        if c.title != "Turns per game"
    )
    Report(
        score=Score(table="games", value="score"),
        evals=(earlier, fixed, *behaviour, pair),
    )


def test_head_to_head_refuses_an_empty_selection() -> None:
    with pytest.raises(ValueError, match="at least one seating"):
        head_to_head("Large vs Small", ())


def test_frontier_selects_only_recorded_head_to_head_seatings() -> None:
    hours, updates = report().evals[2:]
    assert isinstance(hours, Curve)
    assert isinstance(updates, Curve)
    assert hours.matches == tuple(
        selection
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
        for selection in (Matches(seating=name), Matches(tag=name.replace(" ", "-")))
    )
    assert updates.matches == tuple(
        selection
        for update in (25, 50, 75, 100, 150, 200, 300, 400, 600, 800)
        for selection in (
            Matches(seating=f"w160l8 u{update} vs w64 u{update}"),
            Matches(tag=f"w160l8-u{update}-vs-w64-u{update}"),
        )
    )


def test_frontier_opens_ml_with_logged_learning_rates() -> None:
    rate, damping = report().ml[:2]
    assert isinstance(rate, Series) and rate.keys == ("learning_rate",)
    assert isinstance(damping, Series) and damping.keys == ("tau",)


def test_local_report_draws_sparse_combat_probe_readings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    track = tmp_path / "track"
    monkeypatch.setenv("ETUDE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ETUDE_OUT", str(tmp_path / "run"))
    with recording(
        experiment="local", run="probe", seed=197, clocks=CLOCKS, directory=track
    ) as run:
        self_play(gray_ogre_self_play(horizon=10), run, updates=10)
    recorded = read(track, "local", "probe")
    for key in ("combat/attack/gap", "time/combat_probe"):
        assert list(recorded.points(key, "updates").x) == [5, 10]

    # Render in the CLI's process so Etude can reload the recipe's neighbours.
    root = Path(__file__).resolve().parents[2]
    out = tmp_path / "report"
    subprocess.run(
        [
            str(Path(sys.executable).with_name("etude")),
            "report",
            str(root / "manabot/recipes/local.py"),
            "--source",
            str(track),
            "--out",
            str(out),
        ],
        cwd=root,
        check=True,
    )
    (page,) = out.rglob("*.html")
    html = page.read_text()
    for label in (
        "Attack and block log-odds",
        "Attack gap shift under board edits",
        "Block gap shift under board edits",
        "time/combat_probe",
    ):
        assert label in html
