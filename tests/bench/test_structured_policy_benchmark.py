from __future__ import annotations

import json
from pathlib import Path

import pytest

from manabot.sim.structured_benchmark import (
    evaluate_gates,
    load_workload,
    render_report,
    validate_workload,
)

WORKLOAD = Path("experiments/workloads/structured-policy-v1.json")


def test_checked_in_workload_is_fixed_seat_balanced_and_above_32() -> None:
    workload, digest = load_workload(WORKLOAD)

    assert workload["id"] == "structured-policy-v1"
    assert len(workload["evaluation"]["game_seeds"]) % 2 == 0
    assert workload["gates"]["minimum_legal_choices"] > 32
    assert len(digest) == 64


def test_workload_rejects_an_odd_game_list() -> None:
    workload = json.loads(WORKLOAD.read_text())
    workload["evaluation"]["game_seeds"] = [1]

    with pytest.raises(ValueError, match="even list"):
        validate_workload(workload)


def test_gate_evaluation_and_report_cover_all_required_metrics() -> None:
    workload, _ = load_workload(WORKLOAD)
    worker = {
        "games": 8,
        "cap_hits": 0,
        "game_records": [{"winner_seat": 0}],
        "decision_latency": {"p50": 1, "p95": 2},
        "throughput": {
            "games_per_second": 1.0,
            "legacy_equivalent_actions_per_second": 2.0,
        },
        "peak_rss_bytes": 1,
        "outcomes": {
            "win_rates": {"ur_lessons": 0.5, "gw_allies": 0.5},
            "wins": {"draw": 0},
            "seat_breakdown": {
                "ur_lessons_on_play": {"win_rate": 0.5},
                "ur_lessons_on_draw": {"win_rate": 0.5},
                "gw_allies_on_play": {"win_rate": 0.5},
                "gw_allies_on_draw": {"win_rate": 0.5},
            },
        },
    }
    result = {
        "status": "pass",
        "workload": {"id": "structured-policy-v1", "sha256": "a" * 64},
        "correctness": {
            "max_candidates": 35,
            "max_represented_legal_branches": 64,
            "observed_offer_verbs": [
                "pass_priority",
                "cast",
                "declare_attackers",
            ],
            "overflow_count": 0,
            "illegal_decode_count": 0,
            "trace_mismatch_count": 0,
            "action_agreement": {
                "matching": 10,
                "shared": 10,
                "supported_matching": 2,
                "supported_shared": 2,
            },
        },
        "focused_decision_latency": {
            "structured": {"p50": 1, "p95": 2},
            "legacy": {"p50": 1, "p95": 2},
        },
        "adapters": {"structured": worker, "legacy": worker},
    }

    assert all(evaluate_gates(result, workload).values())
    report = render_report(result)
    assert "Shared-state action agreement" in report
    assert "focused p50" in report
    assert "peak RSS" in report
