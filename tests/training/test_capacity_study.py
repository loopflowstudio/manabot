"""Synthetic observations test censoring and common support without games/training."""

import json
from pathlib import Path

import pytest

from experiments.runners.capacity_study import CapacityWorkload, declaration, plan
from experiments.runners.run_training_regimes import run_study
from manabot.arena.models import file_sha256
from manabot.training.analysis import report
from manabot.training.capacity_analysis import (
    Measurement,
    capacity_summary,
    threshold_crossing,
)


def point(regime: str, seconds: float, score: float, cutoff: int = 0) -> Measurement:
    return Measurement(
        regime=regime,
        seed=1031,
        cutoff=cutoff,
        opponent="random-smoke-anchor",
        checkpoint={"sha256": "a" * 64, "path": "synthetic", "bytes": 0},
        training_seconds=seconds,
        decisions=int(seconds * 10),
        optimizer_exposures=int(seconds * 20),
        score=score,
        complete=True,
    )


def test_censoring_and_first_observed_crossing() -> None:
    points = [point("a", 10, 0.4), point("a", 20, 0.65)]
    hit = threshold_crossing(points, 30, 0.6)
    assert (hit.status, hit.lower_seconds, hit.upper_seconds) == ("observed", 10, 20)
    miss = threshold_crossing(points, 15, 0.6)
    assert (miss.status, miss.lower_seconds, miss.upper_seconds) == (
        "right-censored",
        10,
        None,
    )
    assert threshold_crossing(points, 5, 0.6).status == "unavailable"
    assert threshold_crossing(points[1:], 30, 0.6).lower_seconds is None


def test_common_support_area_and_endpoint_separation() -> None:
    protocol = plan().protocol.model_copy(update={"early_progress_seconds": 25})
    points = [
        point("a", 10, 0.4),
        point("a", 30, 0.8, 1),
        point("b", 20, 0.5),
        point("b", 40, 0.9, 1),
    ]
    endpoint = point("a", 30, 0.1, 1).model_copy(update={"phase": "endpoint"})
    result = capacity_summary([*points, endpoint], protocol, ("a", "b"), True)
    curve = result["curves"]["training_seconds"]
    assert (curve["start"], curve["end"]) == (20, 30)
    assert [r["mean_score"] for r in curve["rows"]] == [0.4, 0.5]
    assert result["early_area"]["end_seconds"] == 25
    assert [r["score_seconds"] for r in result["early_area"]["rows"]] == [2, 2.5]
    assert result["terminal"][0]["score"] == 0.1
    assert (
        capacity_summary(points, protocol, ("a", "b", "missing"), True)["status"]
        == "unavailable"
    )
    assert (
        capacity_summary(points, protocol, ("a", "b"), False)["curves"]["decisions"][
            "status"
        ]
        == "unavailable"
    )


def test_no_overlap_or_missing_exposure_counter() -> None:
    protocol = plan().protocol
    points = [point("a", 10, 0.4), point("b", 20, 0.5)]
    result = capacity_summary(points, protocol, ("a", "b"), True)
    assert result["curves"]["training_seconds"]["status"] == "unavailable"
    points[0].optimizer_exposures = None
    assert (
        capacity_summary(points, protocol, ("a", "b"), True)["curves"][
            "optimizer_exposures"
        ]["status"]
        == "unavailable"
    )


def test_declaration_holds_every_noncapacity_field_fixed() -> None:
    cells = declaration().resolve()
    agents = []
    for recipe in cells.regimes.values():
        agent = recipe.agent.model_dump()
        for name in ("hidden_dim", "attention_layers"):
            agent.pop(name)
        agents.append(agent)
    assert agents[0] == agents[1] == agents[2]
    assert agents[0]["value_aggregation"] == "value_token"
    assert len(plan().recipes) == 3


def test_scientific_plan_binds_calibration_and_terminal_workload(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "calibration.json"
    evidence.write_text('{"synthetic": true}')
    workload = CapacityWorkload(
        updates_per_unit=(10, 8, 3),
        calibration_path=evidence,
        calibration_sha256=file_sha256(evidence),
        prior_campaign_seconds=0,
        runtime_identities={
            key: "a" * 64
            for key in (
                "engine_extension_sha256",
                "engine_source_sha256",
                "training_source_sha256",
                "content_manifest_sha256",
                "observation_abi_sha256",
                "action_abi_sha256",
                "matchup_sha256",
            )
        },
        projected_disk_bytes=1024,
        projected_evaluation_seconds=1000,
    )
    resolved = plan(workload)
    assert resolved.protocol.training_seeds == (1031, 1032, 1033)
    assert resolved.protocol.checkpoint_count == 4
    assert len(resolved.protocol.anchor_deals) * 4 == 100
    assert resolved.recipes[0]["stages"][-1]["updates"] == 80
    assert resolved.recipes[0]["stages"][-1]["initial"] == "checkpoint-3"
    evidence.write_text("changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        plan(workload)


@pytest.mark.parametrize("completed", [False, True])
def test_offline_capacity_report_reproduces_without_execution(
    tmp_path: Path, completed: bool
) -> None:
    pytest.importorskip("nbclient")

    resolved = plan()
    artifact = tmp_path / "synthetic.pt"
    artifact.write_bytes(b"synthetic evidence, never a loadable model")
    rows = []
    if completed:
        for recipe in resolved.recipes:
            for cutoff in range(2):
                row = point(
                    recipe["id"], 10 + 10 * cutoff, 0.4 + 0.2 * cutoff, cutoff
                ).model_dump()
                row["checkpoint"] = {
                    "path": str(artifact),
                    "sha256": file_sha256(artifact),
                    "bytes": artifact.stat().st_size,
                }
                row["games"] = 4
                rows.append(row)
    (tmp_path / "protocol.json").write_text(resolved.protocol.model_dump_json())
    (tmp_path / "recipes.json").write_text(json.dumps(resolved.recipes))
    (tmp_path / "study.json").write_text(
        json.dumps(
            {
                "study": "model-capacity",
                "profile": "workflow-smoke",
                "status": "completed" if completed else "failed",
                "limits": "Synthetic empty attempt; no training or games",
                "runs": [],
                "comparisons": [],
                "measurements": rows,
                "seconds": 0,
            }
        )
    )
    report(tmp_path)
    names = ("report.md", "metrics.json", "capacity-analysis.json")
    original = {n: (tmp_path / n).read_bytes() for n in names}
    report(tmp_path)
    assert original == {n: (tmp_path / n).read_bytes() for n in names}
    assert (tmp_path / "capacity-curves.png").exists()
    assert json.loads(original["capacity-analysis.json"])["status"] == (
        "available" if completed else "unavailable"
    )


def test_runner_requires_capacity_plan_before_execution(tmp_path: Path) -> None:

    with pytest.raises(ValueError, match="explicit separately resolved plan"):
        run_study("model-capacity", tmp_path / "absent")
    assert not (tmp_path / "absent").exists()
