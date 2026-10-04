"""Study evidence must preserve cost, cohort and artifact boundaries."""

import hashlib
import json

import pytest

from experiments.runners.training_protocol import EvaluationProtocol
from manabot.arena.models import canonical_sha256
from manabot.training.analysis import cost_comparison, verify_saved_inputs


def point(name, seconds, score):
    return dict(
        regime=name,
        seed=197,
        training_seconds=seconds,
        score=score,
        checkpoint={"path": f"{name}-{seconds}"},
        complete=True,
        opponent="random-smoke-anchor",
    )


def test_shared_cost_never_uses_future_checkpoint_or_backfills_collection():
    rows = [
        point("a", 10, 0.2),
        point("a", 20, 0.9),
        point("b", 5, 0.4),
        point("b", 15, 0.8),
    ]
    result = cost_comparison(rows)
    assert (result["start_seconds"], result["end_seconds"]) == (10, 15)
    assert [r["score"] for r in result["rows"]] == [0.2, 0.8]
    assert [r["mean_score"] for r in result["rows"]] == [0.2, 0.4]
    assert (
        cost_comparison([point("a", 20, 0.9), point("b", 5, 0.4)])["status"]
        == "unavailable"
    )


def test_failed_or_paired_measurements_cannot_enter_anchor_curve():
    bad = point("a", 1, 1)
    bad["complete"] = False
    paired = point("a", 2, 1)
    paired["opponent"] = "another-policy"
    assert cost_comparison([bad, paired])["status"] == "unavailable"


def test_protocol_rejects_overlap_and_unbound_recipes():
    kwargs = dict(study="learning-speed", regime_digests=("a" * 64, "b" * 64))
    EvaluationProtocol(**kwargs)
    with pytest.raises(ValueError, match="disjoint"):
        EvaluationProtocol(**kwargs, anchor_deals=(910001,))
    with pytest.raises(ValueError, match="reserved"):
        EvaluationProtocol(**kwargs, paired_deals=(10197,))
    with pytest.raises(ValueError, match="every resolved"):
        EvaluationProtocol(study="ataraxos-ablations", regime_digests=("a" * 64,))


def test_offline_reporting_rejects_changed_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"original")
    study = {
        "runs": [],
        "measurements": [
            {
                "checkpoint": {
                    "path": str(checkpoint),
                    "sha256": hashlib.sha256(b"original").hexdigest(),
                }
            }
        ],
    }
    verify_saved_inputs(tmp_path, study)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint digest"):
        verify_saved_inputs(tmp_path, study)


def test_offline_reporting_rejects_changed_recipe(tmp_path):
    recipe = {"id": "frozen"}
    protocol = {"regime_digests": [canonical_sha256(recipe)]}
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    (tmp_path / "recipes.json").write_text(json.dumps([recipe]))
    study = {
        "protocol_sha256": canonical_sha256(protocol),
        "runs": [],
        "measurements": [],
    }
    verify_saved_inputs(tmp_path, study)
    (tmp_path / "recipes.json").write_text('[{"id": "changed"}]')
    with pytest.raises(ValueError, match="recipe digest"):
        verify_saved_inputs(tmp_path, study)


def test_report_regeneration_preserves_metrics_without_training(tmp_path):
    pytest.importorskip("nbclient")
    from manabot.training.analysis import report

    study = dict(
        study="ataraxos-ablations",
        profile="smoke",
        status="failed",
        limits="Test fixture: no measured games",
        runs=[],
        comparisons=[],
        measurements=[],
        error="Failure before training",
    )
    (tmp_path / "study.json").write_text(json.dumps(study))
    report(tmp_path)
    first = {
        name: (tmp_path / name).read_bytes()
        for name in ("metrics.json", "cost-comparison.json", "report.md")
    }
    report(tmp_path)
    assert first == {name: (tmp_path / name).read_bytes() for name in first}
    notebook = json.loads((tmp_path / "analysis.ipynb").read_text())
    assert all(
        c["execution_count"] is not None
        for c in notebook["cells"]
        if c["cell_type"] == "code"
    )
    assert "No completed measurements" in first["report.md"].decode()
