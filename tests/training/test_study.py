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


def test_shared_cost_integrates_only_observed_intervals():
    rows = [
        point("a", 2, 0.1),
        point("a", 8, 0.6),
        point("a", 14, 0.9),
        point("b", 5, 0.4),
        point("b", 12, 0.8),
    ]
    result = cost_comparison(rows)
    assert (result["start_seconds"], result["end_seconds"]) == (5, 12)
    assert result["rows"][0]["mean_score"] == pytest.approx((3 * 0.1 + 4 * 0.6) / 7)
    assert result["rows"][1]["mean_score"] == pytest.approx(0.4)


def test_failed_or_paired_measurements_cannot_enter_anchor_curve():
    bad = point("a", 1, 1)
    bad["complete"] = False
    paired = point("a", 2, 1)
    paired["opponent"] = "another-policy"
    assert cost_comparison([bad, paired])["status"] == "unavailable"


def test_raw_and_averaged_checkpoints_remain_separate_cost_curves():
    raw = [point("a", 10, 0.2), point("a", 20, 0.3)]
    averaged = [dict(row, variant="ema", score=0.8) for row in raw]
    result = cost_comparison(raw + averaged)
    assert {(r["variant"], r["score"]) for r in result["rows"]} == {
        ("raw", 0.3),
        ("ema", 0.8),
    }


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
        comparisons=[
            dict(
                a="a",
                b="b",
                cutoff=0,
                evaluation_seconds=1,
                replay={"passed": False},
                trace=None,
                scheduled_games=4,
                rows=[
                    dict(
                        deal_seed=1,
                        leg=leg,
                        player_a_seat=0,
                        score_a=1,
                        failure=None,
                        terminated=True,
                        truncated=leg == 0,
                        replay_passed=leg != 1,
                    )
                    for leg in range(2)
                ],
            )
        ],
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
    rendered = first["report.md"].decode()
    assert "No completed measurements" in rendered
    assert "Unresolved-game bounds for A: [0.000, 1.000]" in rendered
    assert rendered.count("| unavailable | None |") == 2


def test_scientific_profile_requires_replicates_and_explicit_uncertainty():
    kwargs = dict(
        study="learning-speed",
        regime_digests=("a" * 64, "b" * 64),
        purpose="scientific",
        process_seconds=3600,
        cost_cutoffs_seconds=(600, 1200),
    )
    with pytest.raises(ValueError, match="three seeds"):
        EvaluationProtocol(**kwargs)
    protocol = EvaluationProtocol(
        **kwargs,
        training_seeds=(601, 1601, 2601),
        uncertainty="paired-seed-descriptive",
        anchors=("random", "scripted-greedy", "puct-64"),
        endpoint_paired_deals=(950001,),
        endpoint_anchor_deals=(960001,),
        endpoint_seed_pairs=tuple(
            (a, b) for a in (601, 1601, 2601) for b in (601, 1601, 2601)
        ),
    )
    assert len(protocol.training_seeds) == 3
    with pytest.raises(ValueError, match="900 seconds"):
        EvaluationProtocol(
            study="learning-speed",
            regime_digests=("a" * 64, "b" * 64),
            process_seconds=901,
        )


def test_cost_cutoffs_exclude_endpoint_and_future_weights():
    from manabot.training.analysis import cost_cutoffs

    rows = [
        point("a", 10, 0.2),
        point("a", 20, 0.9),
        point("b", 5, 0.3),
        point("b", 20, 0.4),
    ]
    rows.append(dict(point("a", 12, 1), phase="endpoint"))
    results = cost_cutoffs(rows, (1, 15, 25))
    assert results[0]["status"] == results[2]["status"] == "unavailable"
    assert [r["score"] for r in results[1]["rows"]] == [0.2, 0.3]


def test_uncertainty_clusters_training_seeds_and_four_leg_deals():
    from manabot.training.analysis import paired_uncertainty

    cells = [
        dict(
            a="a",
            b="b",
            cutoff=1,
            phase="endpoint",
            training_seed=a,
            b_training_seed=b,
            scheduled_games=8,
            replay={"passed": True},
            rows=[
                dict(
                    deal_seed=d,
                    score_a=a / 2,
                    failure=None,
                    terminated=True,
                    truncated=False,
                    replay_passed=True,
                )
                for d in (1000000, 1000001)
                for _ in range(4)
            ],
        )
        for a in range(3)
        for b in range(3)
    ]
    result = paired_uncertainty(cells)[0]
    assert result["status"] == "available"
    assert result["training_seeds"] == 3
    assert result["deal_blocks"] == 2
    assert result["score_a"] == 0.5
    assert "crossed" in result["method"]
    assert paired_uncertainty(cells[:3])[0]["status"] == "unavailable"


def test_calibrated_plan_scales_work_and_reserves_both_studies(tmp_path, monkeypatch):
    from experiments.runners.run_training_regimes import calibration_plan
    from experiments.runners.training_plan import scientific_plan

    plan = calibration_plan("learning-speed")
    entries = []
    for recipe in plan.recipes:
        stages = []
        for stage in recipe["stages"]:
            stages.append(dict(id=stage["id"], seconds=8, export_seconds=0.1))
        path = tmp_path / f"{recipe['id']}.json"
        path.write_text(
            json.dumps(
                dict(
                    status="completed",
                    regime=recipe,
                    stages=stages,
                    setup_seconds=1,
                    identities={
                        k: "a" * 64
                        for k in (
                            "engine_extension_sha256",
                            "engine_source_sha256",
                            "training_source_sha256",
                            "content_manifest_sha256",
                            "observation_abi_sha256",
                            "action_abi_sha256",
                            "matchup_sha256",
                        )
                    },
                )
            )
        )
        entries.append(dict(path=str(path)))
    study = dict(
        study="learning-speed",
        profile="calibration",
        status="completed",
        runs=entries,
        measurements=[],
        comparisons=[
            dict(
                b=b,
                replay={"passed": True},
                rows=[{}] * 4,
                scheduled_games=4,
                evaluation_seconds=4,
            )
            for b in (
                "direct-self-play",
                "random-smoke-anchor",
                "scripted-greedy-fixed-anchor",
                "puct-64-fixed-anchor",
            )
        ],
    )
    (tmp_path / "study.json").write_text(json.dumps(study))
    resolved = scientific_plan(tmp_path, 0)
    assert resolved.allocation_seconds == 132 * 3600
    assert resolved.prior_campaign_seconds == 25 * 3600
    assert resolved.protocol.checkpoint_count == 4
    assert len(resolved.protocol.endpoint_seed_pairs) == 9
    assert resolved.recipes[0]["stages"][0]["games"] > 4
    assert resolved.recipes[1]["stages"][0]["updates"] > 8
    assert resolved.recipes[1]["stages"][0]["transitions"] == 256
    assert resolved.recipes[0]["stages"][-1]["datasets"] == [
        f"collect-{i}" for i in range(4)
    ]
    from types import SimpleNamespace

    with monkeypatch.context() as patch:
        patch.setattr(
            "experiments.runners.training_plan.shutil.disk_usage",
            lambda _: SimpleNamespace(free=0),
        )
        with pytest.raises(ValueError, match="storage infeasible"):
            scientific_plan(tmp_path, 0)
    with pytest.raises(ValueError, match="insufficient budget"):
        scientific_plan(tmp_path, 30 * 3600)
    study["status"] = "failed"
    (tmp_path / "study.json").write_text(json.dumps(study))
    with pytest.raises(ValueError, match="completed integrated"):
        scientific_plan(tmp_path, 0)


def test_resume_evaluates_remaining_cells_without_retraining_or_retrying_failure(
    tmp_path, monkeypatch
):
    import torch

    from experiments.runners import run_training_regimes as runner
    from manabot.sim import teacher1_evidence
    from manabot.training.models import StageRecord, TrainingRun

    identities = {
        k: "a" * 64
        for k in (
            "engine_extension_sha256",
            "engine_source_sha256",
            "training_source_sha256",
            "content_manifest_sha256",
            "observation_abi_sha256",
            "action_abi_sha256",
            "matchup_sha256",
        )
    }
    monkeypatch.setattr(
        teacher1_evidence, "runtime_fingerprints", lambda *a, **kw: identities.copy()
    )
    monkeypatch.setattr(teacher1_evidence, "source_bundle_sha256", lambda *a: "a" * 64)
    monkeypatch.setattr(
        runner, "execute_regime", lambda *a: pytest.fail("resume must not retrain")
    )
    monkeypatch.setattr(
        runner, "load_checkpoint_agent", lambda *a: (torch.nn.Linear(1, 1), None)
    )
    monkeypatch.setattr(runner, "report", lambda *a: None)
    calls = []

    def play_cell(**kw):
        calls.append(kw)
        return (
            [
                dict(
                    failure=None,
                    terminated=True,
                    truncated=False,
                    replay_passed=True,
                    score_a=0.5,
                )
                for _ in range(4)
            ],
            None,
            {"passed": True},
        )

    monkeypatch.setattr(runner, "play_cell", play_cell)
    recipes = [runner.smoke_recipe(n) for n in runner.STUDIES["learning-speed"]]
    protocol = EvaluationProtocol(
        study="learning-speed",
        regime_digests=tuple(
            canonical_sha256(r.model_dump(mode="json")) for r in recipes
        ),
    )
    entries = []
    for recipe in recipes:
        checkpoint = tmp_path / f"{recipe.id}.pt"
        checkpoint.write_bytes(b"fixture")
        artifact = dict(
            path=str(checkpoint), sha256=hashlib.sha256(b"fixture").hexdigest(), bytes=7
        )
        run = TrainingRun(
            id=recipe.id,
            regime=recipe,
            regime_digest=canonical_sha256(recipe.model_dump(mode="json")),
            seed=197,
            seed_streams={},
            identities=identities,
            status="completed",
            stages=[
                StageRecord(
                    id=f"policy-{i}",
                    status="completed",
                    cumulative_seconds=i + 1,
                    artifacts={"raw": artifact},
                )
                for i in range(2)
            ],
        )
        path = tmp_path / f"{recipe.id}.json"
        path.write_text(run.model_dump_json())
        entries.append(
            dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        )
    failed = dict(
        a="search-distillation",
        b="direct-self-play",
        training_seed=197,
        b_training_seed=197,
        phase="development",
        cutoff=0,
        rows=[],
        scheduled_games=4,
        replay={"passed": False},
        evaluation_seconds=1,
        trace=None,
    )
    study = dict(
        study="learning-speed",
        profile="workflow-smoke",
        status="failed",
        seconds=1,
        runs=entries,
        measurements=[],
        comparisons=[failed],
        protocol_sha256=canonical_sha256(protocol.model_dump(mode="json")),
    )
    (tmp_path / "study.json").write_text(json.dumps(study))
    (tmp_path / "protocol.json").write_text(protocol.model_dump_json())
    (tmp_path / "recipes.json").write_text(
        json.dumps([r.model_dump(mode="json") for r in recipes])
    )
    with pytest.raises(RuntimeError, match="incomplete arena cohort"):
        runner.run_study("learning-speed", tmp_path, resume=True)
    saved = json.loads((tmp_path / "study.json").read_text())
    assert len(calls) == 5
    assert len(saved["comparisons"]) == 6
    assert saved["comparisons"][0] == failed
    assert saved["status"] == "failed"


def test_report_rejects_changed_command_trace(tmp_path):
    trace = tmp_path / "commands.gz"
    trace.write_bytes(b"changed")
    study = dict(
        runs=[],
        measurements=[],
        comparisons=[
            dict(
                trace=dict(
                    path=str(trace), sha256=hashlib.sha256(b"original").hexdigest()
                )
            )
        ],
    )
    with pytest.raises(ValueError, match="trace digest"):
        verify_saved_inputs(tmp_path, study)
