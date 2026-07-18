from __future__ import annotations

import json

import pytest

from etude.advice import FLIP_VERSIONED_FIXTURE_PATH
from experiments.runners.run_belief_recommendation_flip import (
    CANDIDATES,
    EXPECTED_CONDITION_IDS,
    RESULT_PATH,
    SEARCH_SEEDS,
    SELECTED_CANDIDATE_ID,
    SELECTED_SEED,
    build_conditioned_ranges,
    build_position,
    condition_receipts,
    run_search,
    summarize_result,
)
from manabot.sim.conditional_search import canonical_result_json, validate_result


def test_curated_suite_and_selection_are_fixed_and_disclosed() -> None:
    assert [spec.candidate_id for spec in CANDIDATES] == [
        "countered-wipe-four-wide-v1",
        "countered-wipe-three-wide-v1",
        "countered-wipe-buffered-v1",
    ]
    assert SEARCH_SEEDS == (197, 198, 199)
    assert SELECTED_CANDIDATE_ID == "countered-wipe-four-wide-v1"
    assert SELECTED_SEED == 197


def test_selected_position_uses_authoritative_has_and_lacks_receipts() -> None:
    selected = next(
        spec for spec in CANDIDATES if spec.candidate_id == SELECTED_CANDIDATE_ID
    )
    position = build_position(selected)
    try:
        conditions = build_conditioned_ranges(position)
        assert tuple(row.scenario_id for row in conditions) == EXPECTED_CONDITION_IDS
        assert [row.query.to_dict() for row in conditions] == [
            {"kind": "has", "card": "Counterspell", "at_least": 1},
            {"kind": "lacks", "card": "Counterspell", "fewer_than": 1},
        ]
        assert [row.receipt.support_size for row in conditions] == [4764, 5840]
        assert sum(row.receipt.support_size for row in conditions) == 10604
        assert all(row.receipt.space_identity for row in conditions)
        assert all(row.receipt.query_digest for row in conditions)
        assert all(row.receipt.canonical_digest for row in conditions)
        assert all(row.belief.normalization_error < 1e-12 for row in conditions)
        receipts = condition_receipts(position, conditions)
        assert sum(row["condition_mass"] for row in receipts) == pytest.approx(1.0)
        assert "actual_query_truth" not in json.dumps(receipts, sort_keys=True)
    finally:
        position.close()


def test_selected_current_provider_result_has_the_curated_flip() -> None:
    selected = next(
        spec for spec in CANDIDATES if spec.candidate_id == SELECTED_CANDIDATE_ID
    )
    position = build_position(selected)
    try:
        conditions = build_conditioned_ranges(position)
        first = run_search(position, conditions, seed=SELECTED_SEED, branch_audit=False)
        second = run_search(
            position, conditions, seed=SELECTED_SEED, branch_audit=False
        )
        validate_result(first, expected_condition_ids=EXPECTED_CONDITION_IDS)
        assert canonical_result_json(first) == canonical_result_json(second)
        summary = summarize_result(first)
        assert summary["top_action_changed"] == 1.0
        assert [row["top_action_index"] for row in summary["conditions"]] == [1, 0]
        assert [row["visit_margin"] for row in summary["conditions"]] == [66, 38]
        assert all(
            row["top_action_index"] == row["q_top_action_index"]
            for row in summary["conditions"]
        )
        serialized = canonical_result_json(first)
        assert "elapsed_seconds" not in serialized
        assert "peak_rss_bytes" not in serialized
    finally:
        position.close()


@pytest.mark.skipif(
    not FLIP_VERSIONED_FIXTURE_PATH.is_file() or not RESULT_PATH.is_file(),
    reason="INT-15 evidence is generated only after RUL-11 lands",
)
def test_checked_result_separates_deterministic_and_measurement_envelopes() -> None:
    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    deterministic = artifact["deterministic"]
    measurement = artifact["measurement"]
    assert deterministic["selection"] == {
        "kind": "post_hoc_fixture_curation",
        "candidate_id": SELECTED_CANDIDATE_ID,
        "seed": SELECTED_SEED,
        "all_observed_cells_retained": True,
        "prospective_or_stability_evidence": False,
    }
    assert len(deterministic["suite"]) == 9
    deterministic_json = json.dumps(deterministic, sort_keys=True)
    assert "elapsed_seconds" not in deterministic_json
    assert "peak_rss_bytes" not in deterministic_json
    assert measurement["excluded_from_deterministic_hashes_and_equality"] is True
    assert all("elapsed_seconds" in row for row in measurement["suite"])
    assert all("peak_rss_bytes" in row for row in measurement["suite"])
