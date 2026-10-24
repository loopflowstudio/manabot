"""Selection, recovery accounting and real Etude logs feed one offline report."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess

import pytest

from manabot.reports.experiment import load_evidence, write_time_profile
from manabot.reports.records import RetainedRun
from manabot.reports.time_cohort import native_cohort, write_cohort_report
from manabot.reports.time_sources import load_time_sources
from manabot.training.records import StageRecord

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "tests/fixtures/experiment-demo"
JS = ROOT / "manabot/reports/time_cohort.js"


def _run(identifier: str = "first") -> RetainedRun:
    prototype = load_evidence(DEMO).runs[0]
    stage_id = prototype.regime.stages[0].id
    return prototype.model_copy(
        update={
            "id": identifier,
            "seconds": 10.0,
            "setup_seconds": 1.0,
            "last_recorded_wall_seconds": 110.0,
            "calendar_seconds": 10.0,
            "stages": [
                StageRecord(
                    id=stage_id,
                    status="interrupted",
                    seconds=8.0,
                    collection_seconds=5.0,
                    learning_seconds=2.0,
                    export_seconds=0.25,
                    diagnostic_seconds=0.0,
                    learner_transitions=100,
                    updates=1,
                )
            ],
        }
    )


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


def test_recovery_counts_retained_work_once_and_keeps_attempt_costs() -> None:
    parent = _run()
    child = parent.model_copy(
        update={
            "id": "second",
            "parent_run_id": parent.id,
            "seconds": 4.0,
            "prior_seconds": 10.0,
            "stages": [
                parent.stages[0].model_copy(
                    update={
                        "seconds": 3.0,
                        "collection_seconds": 1.0,
                        "learning_seconds": 1.0,
                        "learner_transitions": 150,
                        "updates": 2,
                    }
                )
            ],
        }
    )
    cohort = native_cohort([parent, child, parent])
    assert len(cohort.runs) == 1
    run = cohort.runs[0]
    assert run.elapsed_seconds == 14
    assert run.active_seconds == 9
    assert run.learner_transitions == 150
    assert sum(run.phases.values()) == 14
    # Rewind is retained trajectory, not parent + child or maximum endpoint.
    child.stages[0].learner_transitions = 50
    assert native_cohort([parent, child]).runs[0].learner_transitions == 50
    branch = child.model_copy(update={"id": "branch"})
    branched = native_cohort([parent, child, branch]).runs[0]
    assert branched.elapsed_seconds == 18
    assert branched.learner_transitions is None
    assert native_cohort([child]).runs[0].learner_transitions is None
    with pytest.raises(ValueError, match="Cyclic"):
        native_cohort([parent.model_copy(update={"parent_run_id": child.id}), child])


def test_missing_fields_duplicates_and_native_discovery(tmp_path: Path) -> None:
    run = _run()
    for folder in ("one", "copy"):
        target = tmp_path / folder / "run.json"
        target.parent.mkdir()
        target.write_text(run.model_dump_json(exclude_unset=True))
    cohort = load_time_sources([str(tmp_path)])
    assert len(cohort.runs) == 1
    assert len(cohort.sources) == 2
    assert not any("unavailable" in w for w in cohort.warnings)
    payload = run.model_dump(exclude_unset=True)
    del payload["stages"][0]["collection_seconds"]
    (tmp_path / "copy/run.json").write_text(json.dumps(payload))
    conflict = load_time_sources([str(tmp_path)])
    assert not conflict.runs
    assert any("Conflicting native" in w for w in conflict.warnings)
    absent = native_cohort([RetainedRun.model_validate(payload)]).runs[0]
    assert absent.active_seconds is None
    assert absent.phases["Unattributed"] > 0


def test_missing_elapsed_total_does_not_turn_observed_active_time_into_zero() -> None:
    payload = _run().model_dump(exclude_unset=True)
    del payload["seconds"]
    row = native_cohort([RetainedRun.model_validate(payload)]).runs[0]
    assert row.elapsed_seconds is None
    assert row.active_seconds == 7


def test_flamegraph_merges_hierarchies_without_counting_parent_time_twice() -> None:
    first = _run()
    second = _run("second")
    second.seconds = 20
    cohort = native_cohort([first, second])
    graph = _js("T.flamegraph(d.runs)", asdict(cohort))
    assert graph["covered"] == 2
    assert graph["seconds"] == 30
    nodes = graph["nodes"]
    collection = next(n for n in nodes if n["name"] == "Collection")
    assert collection["seconds"] == 10
    assert collection["width"] == pytest.approx(1 / 3)
    assert collection["depth"] == 2
    assert sum(n["seconds"] for n in nodes if n["depth"] == 1) == 30
    assert all(n["left"] + n["width"] <= 1.000001 for n in nodes)
    # Empty selection and missing phase evidence produce explicit empty graphs.
    assert _js("T.flamegraph([]).nodes", {}) == []
    missing = replace(cohort.runs[0], elapsed_seconds=None)
    assert _js("T.flamegraph(d.runs).covered", {"runs": [asdict(missing)]}) == 0


def test_page_displays_flamegraph_and_can_recover_from_empty_selection(
    tmp_path: Path,
) -> None:
    cohort = native_cohort([_run()])
    output = write_cohort_report(
        cohort, tmp_path / "report.html", selected=[cohort.runs[0].id]
    )
    assert output.read_text().index('id="flamegraph"') < output.read_text().index(
        'id="run-rows"'
    )
    # Exercise browser event wiring headlessly; this does not claim visual QA.
    program = r"""
const fs = require('fs'), vm = require('vm');
class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.style = {}; this.value = ''; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(name, value) { this[name] = value; }
}
const elements = new Map(), get = id => {
  if (!elements.has(id)) elements.set(id, new Element('div'));
  return elements.get(id);
};
get('cohort-data').textContent = fs.readFileSync(0, 'utf8');
get('sort').value = 'active_seconds';
const document = {getElementById:get, createElement:tag=>new Element(tag),
  createElementNS:(_,tag)=>new Element(tag), querySelectorAll:()=>[]};
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), {
  document, URLSearchParams, location:{hash:'',pathname:'/report.html',search:''},
  history:{replaceState:()=>{}}
});
const chart = () => get('flamegraph').children.some(n => n.tag === 'svg');
const initial = chart();
get('clear').onclick();
const cleared = !chart();
get('flamegraph').children.find(n => n.tag === 'button').onclick();
console.log(JSON.stringify({initial,cleared,restored:chart()}));
"""
    result = subprocess.run(
        ["node", "-e", program, str(JS)],
        input=(tmp_path / "report-training-time.json").read_text(),
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout) == {
        "initial": True,
        "cleared": True,
        "restored": True,
    }


def test_etude_written_logs_are_selectable_without_invented_phases(
    tmp_path: Path,
) -> None:
    from etude.train.records import Step, Time
    from etude.train.run import track

    for seed, seconds in [(1, 10.0), (2, 100.0)]:
        run = track(
            experiment="real-training",
            run=f"seed-{seed}",
            job="recipe",
            seed=seed,
            kind="training",
            directory=tmp_path,
            clocks=[
                Time("training_seconds", "active training seconds"),
                Step("learner_transitions", "retained learner rows"),
                Step("updates", "learning iterations"),
            ],
        )
        run.log(
            {"loss": 0.5},
            training_seconds=seconds,
            learner_transitions=seed * 100,
            updates=seed,
        )
        run.finish()
    cohort = load_time_sources([str(tmp_path)])
    assert len(cohort.runs) == 2
    assert sorted(r.active_seconds for r in cohort.runs) == [10, 100]
    assert sorted(r.learner_transitions for r in cohort.runs) == [100, 200]
    assert all(r.elapsed_seconds is None and not r.phases for r in cohort.runs)
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


def test_weighted_aggregation_and_snapshot_selection(tmp_path: Path) -> None:
    first = native_cohort([_run()]).runs[0]
    first = replace(
        first,
        hardware="GPU A",
        recipe="recipe",
        active_seconds=10.0,
        elapsed_seconds=10.0,
        learner_transitions=100,
        phases={"Collection": 9, "Learning": 1},
    )
    second = replace(
        first,
        id="second",
        active_seconds=100.0,
        elapsed_seconds=100.0,
        learner_transitions=2000,
        phases={"Collection": 10, "Learning": 90},
    )
    missing = replace(
        first,
        id="missing",
        active_seconds=None,
        elapsed_seconds=None,
        learner_transitions=None,
        phases={},
    )
    benchmark = replace(first, id="benchmark", kind="benchmark", active_seconds=10000)
    cohort = replace(native_cohort([]), runs=(first, second, missing, benchmark))
    write_cohort_report(cohort, tmp_path / "report.html")
    data = json.loads((tmp_path / "report-training-time.json").read_text())
    summary = _js("T.aggregate(d.runs.slice(0,3))", data)
    assert summary["metrics"]["elapsed_seconds"] == {
        "total": 110,
        "count": 2,
        "min": 10,
        "max": 100,
        "median": 55,
    }
    assert summary["phases"][0]["share"] == pytest.approx(19 / 110)
    assert summary["rates"][0]["rate"] == pytest.approx(2100 / 110)
    assert _js("T.largest(d.runs, 'active_seconds', 5, false)", data) == [
        "second",
        first.id,
    ]
    assert (
        _js("T.union([{start:1,end:5},{start:3,end:9},{start:12,end:15}])", data) == 11
    )
    assert _js("[...T.restore(d, T.selection(d, [d.runs[0].id]))]", data) == [first.id]
    assert (
        _js(
            "(()=>{try{T.restore(d,{schema_version:1,snapshot:'other',selected:[]});return false}catch(e){return true}})()",
            data,
        )
        is True
    )


def test_standalone_uses_cohort_page_and_escapes_untrusted_data(tmp_path: Path) -> None:
    run = _run('</script><img src=x onerror="alert(1)">')
    evidence = replace(load_evidence(DEMO), runs=(run,), executions=())
    html = write_time_profile(evidence, tmp_path / "time.html").read_text()
    assert "Select runs" in html and "Training attempts over time" in html
    assert "</script><img" not in html
    assert "&lt;/script&gt;" in html
    assert "\\u003c/script>" in html
    assert 'src="http' not in html


def test_etude_resume_keeps_sunk_time_but_not_duplicate_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from etude.train.records import Step, Time
    from etude.train.run import track

    clocks = [
        Time("training_seconds", "active seconds"),
        Step("learner_transitions", "retained rows"),
    ]
    first = track(experiment="resume", run="same", directory=tmp_path, clocks=clocks)
    first.log({"loss": 1.0}, training_seconds=10, learner_transitions=100)
    first.finish()
    monkeypatch.setenv("ETUDE_UNTIL_CLOCK", "training_seconds")
    monkeypatch.setenv("ETUDE_FROM_VALUE", "8")
    second = track(experiment="resume", run="same", directory=tmp_path, clocks=clocks)
    second.log({"loss": 0.5}, training_seconds=12, learner_transitions=120)
    second.finish()
    run = load_time_sources([str(tmp_path)]).runs[0]
    assert len(run.members) == 2
    assert run.active_seconds == 14
    assert run.learner_transitions == 120
    monkeypatch.delenv("ETUDE_FROM_VALUE")
    third = track(experiment="resume", run="same", directory=tmp_path, clocks=clocks)
    third.log({"loss": 0.4}, training_seconds=14, learner_transitions=140)
    third.finish()
    assert load_time_sources([str(tmp_path)]).runs[0].active_seconds is None


def test_failed_attempt_without_training_artifact_is_selectable() -> None:
    execution = load_evidence(DEMO).executions[0]
    attempt = execution.attempts[0].model_copy(
        update={
            "run_id": None,
            "status": "failed",
            "started_unix": 10,
            "finished_unix": 15,
        }
    )
    cohort = native_cohort([], [execution.model_copy(update={"attempts": [attempt]})])
    assert len(cohort.runs) == 1
    assert cohort.runs[0].status == "failed"
    assert cohort.runs[0].active_seconds is None
    assert cohort.runs[0].intervals[0].end == 15


def test_cli_creates_report_from_saved_native_records(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from manabot.cli import app

    source = tmp_path / "source"
    source.mkdir()
    (source / "run.json").write_text(_run().model_dump_json(exclude_unset=True))
    output = tmp_path / "report.html"
    result = CliRunner().invoke(
        app, ["training-report", "--source", str(source), "--out", str(output)]
    )
    assert result.exit_code == 0, result.output
    assert "1 logical runs" in result.output
    assert "Select runs" in output.read_text()


def test_etude_continuation_adds_attempt_duration_and_uses_leaf_progress(
    tmp_path: Path,
) -> None:
    from etude.run.records import Continuation
    from etude.train.read import read_all
    from etude.train.records import Step, Time
    from etude.train.run import track

    clocks = [
        Time("training_seconds", "active seconds"),
        Step("learner_transitions", "absolute retained rows"),
    ]
    parent = track(
        experiment="continuation", run="parent", directory=tmp_path, clocks=clocks
    )
    parent.log({"loss": 1.0}, training_seconds=10, learner_transitions=100)
    parent.finish()
    declaration = read_all(tmp_path)[0].file
    assert declaration is not None
    child = track(
        experiment="continuation",
        run="child",
        directory=tmp_path,
        clocks=clocks,
        continues=Continuation(
            parent=declaration.declaration.source_key, clock="training_seconds", value=8
        ),
    )
    child.log({"loss": 0.5}, training_seconds=12, learner_transitions=120)
    child.finish()
    cohort = load_time_sources([str(tmp_path)])
    assert len(cohort.runs) == 1
    assert cohort.runs[0].active_seconds == 14
    assert cohort.runs[0].learner_transitions == 120
    assert len(cohort.runs[0].members) == 2


def test_mixed_record_schemas_and_phase_coverage(tmp_path: Path) -> None:
    from etude.train.records import Step
    from etude.train.run import track

    native_dir = tmp_path / "native"
    native_dir.mkdir()
    (native_dir / "run.json").write_text(_run().model_dump_json(exclude_unset=True))
    tracked = track(
        experiment="smoke",
        run="logged",
        kind="smoke",
        directory=tmp_path,
        clocks=[Step("steps", "environment steps")],
    )
    tracked.log({"loss": 0.5}, steps=100)
    tracked.finish()
    cohort = load_time_sources([str(tmp_path)])
    assert len(cohort.runs) == 2
    native = next(r for r in cohort.runs if r.id.startswith("native:"))
    assert "Collection" in native.phase_coverage and native.source_uris
    logged = next(r for r in cohort.runs if r.id.startswith("etude:"))
    assert logged.kind == "smoke" and logged.active_seconds is None
    assert logged.environment_steps == 100
    write_cohort_report(cohort, tmp_path / "mixed.html")
    data = json.loads((tmp_path / "mixed-training-time.json").read_text())
    assert _js("T.largest(d.runs, 'active_seconds', 5, false)", data) == [native.id]
