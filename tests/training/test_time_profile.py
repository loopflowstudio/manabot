"""Training-time views reconcile costs without inventing execution or overlap."""

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from manabot.reports.experiment import load_evidence, write_time_profile
from manabot.reports.records import RetainedRun
from manabot.reports.time_cohort import native_cohort, phase_totals
from manabot.reports.time_profile import (
    training_intervals,
    training_profiles,
    training_time_report,
)
from manabot.training.records import PhaseTiming, StageRecord

DEMO = Path(__file__).resolve().parents[2] / "tests/fixtures/experiment-demo"


def _run() -> RetainedRun:
    return (
        load_evidence(DEMO)
        .runs[0]
        .model_copy(
            update={
                "id": "run-a",
                "seconds": 10.0,
                "setup_seconds": 1.0,
                "monitoring_export_seconds": 0.5,
                "recovery_seconds": 0.25,
                "last_recorded_wall_seconds": 110.0,
                "calendar_seconds": 12.0,
                "stages": [
                    StageRecord(
                        id="train",
                        status="completed",
                        seconds=8.0,
                        collection_seconds=5.0,
                        learning_seconds=2.0,
                        export_seconds=0.25,
                        diagnostic_seconds=0.0,
                    )
                ],
            }
        )
    )


def test_phase_costs_reconcile_and_nested_exports_are_not_added_twice() -> None:
    profile = training_profiles([_run()])[0]
    assert not profile.warnings
    assert profile.root is not None
    assert profile.root.seconds == 10.0
    assert sum(p.seconds for p in profile.root.children) == 10.0
    stage = next(p for p in profile.root.children if p.name == "train")
    assert sum(p.seconds for p in stage.children) == 8.0
    assert next(p.seconds for p in stage.children if p.name == "Unattributed") == 0.75
    assert (
        next(p.seconds for p in profile.root.children if p.name == "Unattributed")
        == 1.0
    )


def test_recovery_does_not_charge_completed_parent_stages_again() -> None:
    parent = _run()
    child = parent.model_copy(
        update={
            "id": "run-b",
            "parent_run_id": parent.id,
            "prior_seconds": 10.0,
            "seconds": 4.0,
            "stages": [
                *parent.stages,
                StageRecord(
                    id="next",
                    seconds=2.0,
                    learning_seconds=2.0,
                    collection_seconds=0.0,
                    export_seconds=0.0,
                    diagnostic_seconds=0.0,
                ),
            ],
        }
    )
    profile = training_profiles([parent, child])[1]
    assert not profile.warnings
    assert profile.root is not None
    assert profile.root.seconds == 4.0
    assert [p.name for p in profile.root.children] == ["Setup", "next", "Unattributed"]
    orphan = training_profiles([child])[0]
    assert "parent unavailable" in orphan.warnings[0]
    assert orphan.root is not None
    assert orphan.root.children[0].name == "Unattributed"


def test_initial_admission_wait_partitions_stage_time_without_active_work() -> None:
    run = _run()
    run.stages[0].admission_wait_seconds = 0.5
    profile = training_profiles([run])[0]
    assert profile.root is not None
    stage = next(p for p in profile.root.children if p.name == "train")
    assert sum(p.seconds for p in stage.children) == 8.0
    assert (
        next(p.seconds for p in stage.children if p.name == "Initial admission wait")
        == 0.5
    )
    assert next(p.seconds for p in stage.children if p.name == "Unattributed") == 0.25


def test_collection_subphases_preserve_parent_total_and_deeper_svg() -> None:
    run = _run()
    run.stages[0].collection_timings = {
        "Environment step and encoding": PhaseTiming(seconds=3, calls=12),
        "Policy inference and action selection": PhaseTiming(seconds=1, calls=10),
    }
    profile = training_profiles([run])[0]
    assert not profile.warnings
    assert profile.root is not None
    stage = next(p for p in profile.root.children if p.name == "train")
    collection = next(p for p in stage.children if p.name == "Collection")
    assert sum(p.seconds for p in collection.children) == collection.seconds == 5
    assert phase_totals([profile])["Collection"] == 5
    assert sum(phase_totals([profile]).values()) == 10
    html = training_time_report([run], [])
    assert 'viewBox="0 0 1000 156"' in html
    run.stages[0].collection_timings["Environment step and encoding"].seconds = 10
    contradictory = training_profiles([run])[0]
    assert "Collection: phase totals exceed recorded duration" in contradictory.warnings
    assert phase_totals([contradictory])["Collection"] == 5


def test_live_snapshot_keeps_observed_work_without_inventing_stage_duration() -> None:
    run = _run()
    run.status = "running"
    run.stages[0].status = "running"
    run.stages[0].seconds = 0
    profile = training_profiles([run])[0]
    assert profile.root is not None and profile.root.seconds == 10
    observed = next(p for p in profile.root.children if "observed phases" in p.name)
    assert observed.seconds == 7.25
    assert phase_totals([profile]) == {
        "Setup": 1,
        "Collection": 5,
        "Learning": 2,
        "Stage exports": 0.25,
        "Unattributed": 1.75,
    }
    assert "live stage duration unavailable" in profile.warnings[0]
    assert native_cohort([run]).runs[0].active_seconds == 7
    # An impossible enclosing clock still invalidates attribution.
    run.seconds = 3
    contradictory = training_profiles([run])[0]
    assert phase_totals([contradictory]) == {"Unattributed": 3}
    assert native_cohort([run]).runs[0].active_seconds is None
    # A completed stage cannot use the live-export explanation.
    run.seconds = 10
    run.stages[0].status = "completed"
    completed = training_profiles([run])[0]
    assert "Collection" not in phase_totals([completed])
    assert any("exceed" in warning for warning in completed.warnings)


def test_conflicting_costs_are_visible_instead_of_normalized() -> None:
    run = _run()
    run.stages[0].learning_seconds = 20
    profile = training_profiles([run])[0]
    assert profile.warnings == ("train: phase totals exceed recorded duration",)
    assert profile.root is not None
    stage = next(p for p in profile.root.children if p.name == "train")
    assert [(p.name, p.seconds) for p in stage.children] == [("Unattributed", 8.0)]
    conflict = run.model_copy(update={"seconds": 11.0})
    with pytest.raises(ValueError, match="conflicting retained"):
        training_profiles([run, conflict])


def test_absent_historical_counters_are_not_reported_as_measured_zero() -> None:
    run = _run().model_copy(update={"stages": [StageRecord(id="old", seconds=8.0)]})
    profile = training_profiles([run])[0]
    assert "old: Collection duration unavailable" in profile.warnings
    assert profile.root is not None
    stage = next(p for p in profile.root.children if p.name == "old")
    assert [(p.name, p.seconds) for p in stage.children] == [("Unattributed", 8.0)]
    payload = run.model_dump()
    del payload["seconds"]
    absent = RetainedRun.model_validate(payload)
    assert training_profiles([absent])[0].root is None


def test_calendar_intervals_preserve_overlap_and_stop_at_saved_time() -> None:
    first = _run()
    second = first.model_copy(
        update={
            "id": "run-b",
            "status": "running",
            "last_recorded_wall_seconds": 112.0,
        }
    )
    intervals, missing = training_intervals([first, second], [])
    assert not missing
    assert [(i.start, i.end) for i in intervals] == [(98.0, 110.0), (100.0, 112.0)]
    assert sum(i.end - i.start for i in intervals) == 24
    assert max(i.end for i in intervals) - min(i.start for i in intervals) == 14
    absent = second.model_copy(update={"calendar_seconds": 0.0})
    assert training_intervals([absent], [])[1]


def test_failed_attempt_without_run_and_pending_attempt_remain_visible() -> None:
    execution = load_evidence(DEMO).executions[0]
    failed = execution.attempts[0].model_copy(
        update={
            "run_id": None,
            "status": "failed",
            "started_unix": 10.0,
            "finished_unix": 12.0,
        }
    )
    pending = failed.model_copy(
        update={
            "ordinal": 99,
            "status": "pending",
            "started_unix": None,
            "finished_unix": None,
        }
    )
    execution = execution.model_copy(update={"attempts": [failed, pending]})
    intervals, missing = training_intervals([], [execution])
    assert len(intervals) == 1 and intervals[0].status == "failed"
    assert len(missing) == 1 and "pending" in missing[0]


def test_report_contains_escaped_training_hierarchy_and_valid_svg() -> None:
    run = _run().model_copy(update={"id": '<script>"unsafe"</script>'})
    html = training_time_report([run], [])
    assert "Where training time goes" in html
    assert "Unattributed" in html and "5.000000" in html
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "Evaluation lanes" in html
    for fragment in html.split("<svg ")[1:]:
        ET.fromstring("<svg " + fragment.split("</svg>", 1)[0] + "</svg>")


def test_standalone_report_omits_evaluation_panels(tmp_path: Path) -> None:
    evidence = load_evidence(DEMO)
    html = write_time_profile(evidence, tmp_path / "time.html").read_text()
    assert "Training attempts over time" in html
    assert "Win rate" not in html and "Scientific evaluation:" not in html
    assert 'href="time-training-time.json"' in html
    assert (tmp_path / "time-training-time.json").is_file()
