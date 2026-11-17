"""Selection, recovery accounting and real Etude logs feed one offline report."""

import json
from pathlib import Path
import subprocess

from etude.records import Clock, Continuation
from etude.train.read import read_all
from etude.train.records import Time
import pytest

from manabot.recipes.local import LocalConfig, train
from manabot.reports.time_cohort import CohortRun, TimeCohort, write_cohort_report
from manabot.reports.time_sources import load_time_sources
from tests.fixtures.training_records import recording

ROOT = Path(__file__).resolve().parents[2]
JS = ROOT / "manabot/reports/time_cohort.js"


def _js(expression: str, payload: object) -> object:
    result = subprocess.run(
        [
            "node",
            "-e",
            "const T=require(process.argv[1]); const d=JSON.parse(require('fs').readFileSync(0,'utf8')); console.log(JSON.stringify("
            + expression
            + "));",
            str(JS),
        ],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def test_etude_written_logs_are_selectable_without_invented_phases(
    tmp_path: Path,
) -> None:

    for seed, seconds in [(1, 10.0), (2, 100.0)]:
        run = recording(
            experiment="real-training",
            run=f"seed-{seed}",
            job="recipe",
            seed=seed,
            kind="training",
            directory=tmp_path,
            clocks=[
                Time("training_seconds", "active training seconds"),
                Clock(
                    name="learner_transitions",
                    kind="step",
                    meaning="retained learner rows",
                ),
                Clock(name="updates", kind="step", meaning="learning iterations"),
            ],
        )
        run.log(
            {"loss": 0.5, "run/kind": "smoke", "run/recipe": "other"},
            training_seconds=seconds,
            learner_transitions=seed * 100,
            updates=seed,
        )
        run.finish()
    cohort = load_time_sources([str(tmp_path)])
    assert len(cohort.runs) == 2
    assert sorted(r.active_seconds for r in cohort.runs) == [10, 100]
    assert sorted(r.learner_transitions for r in cohort.runs) == [100, 200]
    assert all(r.kind == "training" and r.recipe == "recipe" for r in cohort.runs)
    assert all(
        r.intervals and r.intervals[0].basis.startswith("Logged") for r in cohort.runs
    )
    assert cohort.sources and not cohort.warnings
    report = write_cohort_report(cohort, tmp_path / "report.html")
    assert "Save selection" in report.read_text()
    data = json.loads((tmp_path / "report-training-time.json").read_text())
    assert _js("T.largest(d.runs, 'active_seconds', 1, false)", data) == [
        cohort.runs[1].id
    ]


def test_missing_source_and_malformed_records_remain_visible(tmp_path: Path) -> None:
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad/run.json").write_text("not JSON")
    cohort = load_time_sources([str(tmp_path), str(tmp_path / "missing")])
    assert not cohort.runs
    assert any("Unreadable record" in w for w in cohort.warnings)
    assert any("Source incomplete" in w for w in cohort.warnings)


def test_etude_resume_keeps_sunk_time_but_not_duplicate_work(tmp_path: Path) -> None:

    clocks = [
        Time("training_seconds", "active seconds"),
        Clock(name="learner_transitions", kind="step", meaning="retained rows"),
    ]
    first = recording(
        experiment="resume", run="same", directory=tmp_path, clocks=clocks
    )
    first.log({"loss": 1.0}, training_seconds=8, learner_transitions=80)
    first.log({"loss": 1.0}, training_seconds=10, learner_transitions=100)
    first.finish()
    # The second attempt restored the save at 8 seconds; its first row replaces
    # the first attempt's row at 10, whose two seconds stay spent.
    second = recording(
        experiment="resume", run="same", directory=tmp_path, clocks=clocks, resume=True
    )
    second.log({"loss": 0.6}, training_seconds=10, learner_transitions=100)
    second.log({"loss": 0.5}, training_seconds=12, learner_transitions=120)
    second.finish()
    run = load_time_sources([str(tmp_path)]).runs[0]
    assert len(run.members) == 2
    assert run.active_seconds == 14
    assert run.learner_transitions == 120


def test_empty_attempts_preserve_observed_history_without_inventing_time(
    tmp_path: Path,
) -> None:
    clocks = [Time("training_seconds"), "updates", "learner_transitions"]
    with recording(experiment="resume", run="same", directory=tmp_path, clocks=clocks):
        pass
    (empty,) = load_time_sources([str(tmp_path)]).runs
    assert empty.active_seconds is None and empty.updates is None
    for seconds, update in [(10, 1), (20, 2)]:
        with recording(
            experiment="resume",
            run="same",
            directory=tmp_path,
            clocks=clocks,
            resume=True,
        ) as run:
            run.log(
                {}, training_seconds=seconds, updates=update, learner_transitions=100
            )
            run.log(
                {"loss": 0.5},
                training_seconds=seconds,
                updates=update,
                learner_transitions=100,
            )
        with recording(
            experiment="resume",
            run="same",
            directory=tmp_path,
            clocks=clocks,
            resume=True,
        ):
            pass
        (reported,) = load_time_sources([str(tmp_path)]).runs
        assert reported.active_seconds == seconds
        assert reported.updates == update
        assert reported.learner_transitions == 100
        assert len(reported.intervals) == update
        assert any("observed attempts only" in w for w in reported.warnings)


def test_local_recipe_is_a_named_smoke_excluded_from_default_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ETUDE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ETUDE_EXPERIMENT", "renamed-experiment")
    monkeypatch.setenv("ETUDE_RUN", "renamed-run")
    monkeypatch.setenv("ETUDE_TRACK_DIR", str(tmp_path / "track"))
    monkeypatch.setenv("ETUDE_OUT", str(tmp_path / "out"))
    train(LocalConfig(updates=1))
    cohort = load_time_sources([str(tmp_path / "track")])
    (run,) = cohort.runs
    assert run.kind == "smoke"
    assert run.recipe == "manabot.recipes.local"
    assert run.updates == 1 and run.active_seconds is not None
    write_cohort_report(cohort, tmp_path / "report.html")
    data = json.loads((tmp_path / "report-training-time.json").read_text())
    assert _js("T.largest(d.runs, 'active_seconds', 5, false)", data) == []
    assert _js("T.largest(d.runs, 'active_seconds', 5, true)", data) == [run.id]


def test_etude_continuation_adds_attempt_duration_and_uses_leaf_progress(
    tmp_path: Path,
) -> None:

    clocks = [
        Time("training_seconds", "active seconds"),
        Clock(
            name="learner_transitions", kind="step", meaning="absolute retained rows"
        ),
    ]
    parent = recording(
        experiment="continuation", run="parent", directory=tmp_path, clocks=clocks
    )
    parent.log({"loss": 1.0}, training_seconds=10, learner_transitions=100)
    parent.finish()
    declaration = read_all(tmp_path)[0].file
    assert declaration is not None
    child = recording(
        experiment="continuation",
        run="child",
        directory=tmp_path,
        clocks=clocks,
        continues=Continuation(
            parent=declaration.source_key, clock="training_seconds", value=8
        ),
    )
    child.log({"loss": 0.5}, training_seconds=12, learner_transitions=120)
    child.finish()
    cohort = load_time_sources([str(tmp_path)])
    assert len(cohort.runs) == 1
    assert cohort.runs[0].active_seconds == 14
    assert cohort.runs[0].learner_transitions == 120
    assert len(cohort.runs[0].members) == 2


def _cohort_run(identifier: str, **fields: object) -> CohortRun:
    values: dict[str, object] = {
        "id": identifier,
        "label": identifier,
        "experiment": "fixture",
        "kind": "training",
        "recipe": "recipe",
        "seed": 1,
        "status": "finished",
        "members": (identifier,),
        "active_seconds": 10.0,
        "learner_transitions": 100,
        "updates": 1,
        "environment_steps": None,
        "intervals": (),
        "warnings": (),
    }
    return CohortRun(**{**values, **fields})  # type: ignore[arg-type]


def test_weighted_aggregation_and_snapshot_selection(tmp_path: Path) -> None:
    first = _cohort_run("first")
    second = _cohort_run("second", active_seconds=100.0, learner_transitions=2000)
    missing = _cohort_run("missing", active_seconds=None, learner_transitions=None)
    benchmark = _cohort_run("benchmark", kind="benchmark", active_seconds=10000.0)
    cohort = TimeCohort((first, second, missing, benchmark))
    write_cohort_report(cohort, tmp_path / "report.html")
    data = json.loads((tmp_path / "report-training-time.json").read_text())
    summary = _js("T.aggregate(d.runs.slice(0,3))", data)
    assert summary["metrics"]["active_seconds"] == {
        "total": 110,
        "count": 2,
        "min": 10,
        "max": 100,
        "median": 55,
    }
    assert [row["rate"] for row in summary["rates"]] == [10, 20, None]
    assert _js("T.largest(d.runs, 'active_seconds', 5, false)", data) == [
        "second",
        "first",
    ]
    assert (
        _js("T.union([{start:1,end:5},{start:3,end:9},{start:12,end:15}])", data) == 11
    )
    assert _js("[...T.restore(d, T.selection(d, [d.runs[0].id]))]", data) == ["first"]
    assert (
        _js(
            "(()=>{try{T.restore(d,{schema_version:1,snapshot:'other',selected:[]});return false}catch(e){return true}})()",
            data,
        )
        is True
    )


def test_page_escapes_untrusted_run_data(tmp_path: Path) -> None:
    run = _cohort_run('</script><img src=x onerror="alert(1)">')
    html = write_cohort_report(TimeCohort((run,)), tmp_path / "time.html").read_text()
    assert "Select runs" in html and "Training attempts over time" in html
    assert "</script><img" not in html
    assert "&lt;/script&gt;" in html
    assert "\\u003c/script>" in html
    assert 'src="http' not in html


def test_a_saved_regime_record_is_named_and_holds_no_run(tmp_path: Path) -> None:
    (tmp_path / "run.json").write_text(json.dumps({"regime": {}, "stages": []}))
    cohort = load_time_sources([str(tmp_path)])
    assert not cohort.runs
    assert any("Unrecognized run record" in w for w in cohort.warnings)


def test_cli_creates_report_from_an_etude_source(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from manabot.cli import app

    source = tmp_path / "source"
    run = recording(
        experiment="real-training",
        run="seed-1",
        directory=source,
        clocks=[Time("training_seconds", "active training seconds")],
    )
    run.log({"loss": 0.5}, training_seconds=10.0)
    run.finish()
    output = tmp_path / "report.html"
    result = CliRunner().invoke(
        app, ["training-report", "--source", str(source), "--out", str(output)]
    )
    assert result.exit_code == 0, result.output
    assert "1 logical runs" in result.output
    assert "Select runs" in output.read_text()
