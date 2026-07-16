from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULT = Path("experiments/data/structural-semantic-katas-v1-discriminator.json")
PREREGISTRATION = "661392e7e09a0db902fea8de3e9392cad24ab6a1"


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def _runs(result: dict, arm: str) -> list[dict]:
    return sorted(
        [run for run in result["runs"] if run["arm"] == arm],
        key=lambda run: run["model_seed"],
    )


def test_checked_discriminator_result_selects_registered_capacity_branch() -> None:
    result = _result()

    assert result["decision"]["terminal"] == "KILL_REDESIGN structural_capacity"
    assert result["status"] == "redesign"
    assert result["stage_c_triggered"]
    assert len(result["runs"]) == 15
    assert result["budget"]["optimizer_steps"] == 44000
    assert result["budget"]["presented_examples"] == 2816000
    assert result["budget"]["wall_clock_seconds"] < 1800
    assert result["provenance"]["preregistration_revision"] == PREREGISTRATION
    assert result["provenance"]["measurement_code_revision"] == PREREGISTRATION


def test_bag_control_and_cached_equivalence_remain_exact() -> None:
    result = _result()
    bag_runs = _runs(result, "bag_v1")

    for run in result["runs"]:
        assert run["cached_equivalence"] == {
            "programs": 800,
            "tensor_equality": True,
            "logit_equality": True,
            "metric_equality": True,
        }
    for run in bag_runs:
        for split in ("train", "validation", "test"):
            assert run[split]["accuracy"] == 0.5
            assert all(
                family["accuracy"] == 0.5 for family in run[split]["by_family"].values()
            )
            assert run["symmetry"][split]["paired_prediction_disagreements"] == 0
            assert (
                run["symmetry"][split]["maximum_paired_probability_difference"] <= 1e-6
            )


def test_optimization_and_message_arms_do_not_fit_all_five_seeds() -> None:
    result = _result()
    optimization = _runs(result, "relational_semantic_encoder_v1_opt4000")
    message = _runs(result, "relational_message_encoder_v1")

    assert all(
        run["training"]["maximum_training_accuracy"] < 0.99 for run in optimization
    )
    assert [run["training"]["first_99_train_step"] for run in message] == [
        None,
        None,
        None,
        560,
        None,
    ]
    assert (
        sum(run["training"]["maximum_training_accuracy"] >= 0.99 for run in message)
        == 1
    )
    assert {run["performance"]["parameter_count"] for run in optimization} == {8838}
    assert {run["performance"]["parameter_count"] for run in message} == {9030}


def test_cost_attribution_is_derived_from_raw_receipts_without_changing_stop() -> None:
    result = _result()
    bag = _runs(result, "bag_v1")
    for arm in (
        "relational_semantic_encoder_v1_opt4000",
        "relational_message_encoder_v1",
    ):
        candidate = _runs(result, arm)
        receipt = result["cost_attribution"][arm]
        cached_p95 = [
            candidate_run["performance"]["cached_projector_e2e"]["batch1_latency_ns"][
                "p95"
            ]
            / bag_run["performance"]["cached_projector_e2e"]["batch1_latency_ns"]["p95"]
            for candidate_run, bag_run in zip(candidate, bag, strict=True)
        ]
        cached_throughput = [
            candidate_run["performance"]["cached_projector_e2e"][
                "batch128_examples_per_second"
            ]
            / bag_run["performance"]["cached_projector_e2e"][
                "batch128_examples_per_second"
            ]
            for candidate_run, bag_run in zip(candidate, bag, strict=True)
        ]
        assert receipt["worst_cached_batch1_p95_ratio"] == pytest.approx(
            max(cached_p95)
        )
        assert receipt["worst_cached_batch128_throughput_ratio"] == pytest.approx(
            min(cached_throughput)
        )
        assert receipt["cached_latency_gate"]
        assert not receipt["cached_throughput_gate"]
        assert not receipt["cached_cost_gate"]


def test_result_preserves_static_claim_boundary() -> None:
    result = _result()

    assert "does not start W2-213" in result["claim_boundary"]
    assert result["suite_audit"]["definition_references"] == 0
    assert result["suite_audit"]["identity_fields_tensorized"] == []
