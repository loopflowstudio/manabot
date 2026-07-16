"""Focused evidence-boundary tests for the bounded Teacher-1 admission gate."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from manabot.sim.teacher1_evidence import (
    _fresh_env,
    build_command,
    build_viewer_frame,
    evaluate_root_stability,
    record_teacher_trajectories,
    replay_teacher_trajectories,
    runtime_fingerprints,
)


def test_viewer_frame_redacts_private_hand_and_binds_legal_command() -> None:
    runtime = runtime_fingerprints(seed=101)
    env = _fresh_env(101)
    frame = build_viewer_frame(
        env.last_raw_obs,
        match_id="test-match",
        revision=0,
        content_hash=runtime["experience_content_hash"],
        asset_manifest_hash=runtime["asset_manifest_hash"],
    )

    assert frame["projection"]["opponent"]["hand"] == []
    assert frame["projection"]["opponent"]["hand_hidden_count"] > 0
    assert len(frame["offers"]) == env._engine.action_count()
    command = build_command(frame, int(frame["offers"][0]["id"]))
    assert command["expected_revision"] == frame["revision"]
    assert command["prompt_id"] == frame["prompt"]["id"]
    assert command["offer_id"] in {offer["id"] for offer in frame["offers"]}


def test_audit_trajectory_replays_and_preserves_target_invariants() -> None:
    runtime = runtime_fingerprints(seed=103)
    artifact = record_teacher_trajectories(
        games=1,
        simulations=1,
        worlds=1,
        c_puct=1.5,
        seed=103,
        content_hash=runtime["experience_content_hash"],
        asset_manifest_hash=runtime["asset_manifest_hash"],
    )
    receipt = replay_teacher_trajectories(
        artifact,
        content_hash=runtime["experience_content_hash"],
        asset_manifest_hash=runtime["asset_manifest_hash"],
        sampled_search_roots=[{"game_index": 0, "decision_index": 0}],
    )

    assert receipt.passed
    assert receipt.decisions > 0
    assert len(artifact["games"][0]["decisions"]) > 8
    assert receipt.sampled_search_roots == 1
    assert receipt.missing_sampled_search_roots == 0
    assert receipt.search_action_mismatches == 0
    assert receipt.search_visit_mismatches == 0
    assert receipt.search_q_mismatches == 0
    assert receipt.search_value_mismatches == 0
    for decision in artifact["games"][0]["decisions"]:
        search = decision["search"]
        assert sum(search["visit_counts"]) == 1
        assert len(search["visit_counts"]) == len(decision["frame"]["offers"])
        assert search["encoded_legal_count"] == len(decision["frame"]["offers"])
        assert search["root_unchanged"]
        assert search["call_index"] >= 1
        assert np.isfinite(search["root_value"])
        assert decision["seat"] == decision["actor"]
        assert decision["opponent_class"] == "search"

    tampered = deepcopy(artifact)
    tampered["games"][0]["decisions"][0]["search"]["q_values"][0] += 0.125
    tampered_receipt = replay_teacher_trajectories(
        tampered,
        content_hash=runtime["experience_content_hash"],
        asset_manifest_hash=runtime["asset_manifest_hash"],
        sampled_search_roots=[{"game_index": 0, "decision_index": 0}],
    )
    assert not tampered_receipt.passed
    assert tampered_receipt.search_q_mismatches == 1

    tampered = deepcopy(artifact)
    tampered["games"][0]["decisions"][0]["search"]["visit_counts"][0] += 1
    tampered_receipt = replay_teacher_trajectories(
        tampered,
        content_hash=runtime["experience_content_hash"],
        asset_manifest_hash=runtime["asset_manifest_hash"],
        sampled_search_roots=[{"game_index": 0, "decision_index": 0}],
    )
    assert not tampered_receipt.passed
    assert tampered_receipt.search_visit_mismatches == 1

    tampered = deepcopy(artifact)
    tampered["games"][0]["decisions"][0]["search"]["root_value"] += 0.125
    tampered_receipt = replay_teacher_trajectories(
        tampered,
        content_hash=runtime["experience_content_hash"],
        asset_manifest_hash=runtime["asset_manifest_hash"],
        sampled_search_roots=[{"game_index": 0, "decision_index": 0}],
    )
    assert not tampered_receipt.passed
    assert tampered_receipt.search_value_mismatches == 1

    tampered = deepcopy(artifact)
    first_command = tampered["games"][0]["decisions"][0]["command"]
    offer_ids = [
        offer["id"] for offer in tampered["games"][0]["decisions"][0]["frame"]["offers"]
    ]
    first_command["offer_id"] = next(
        offer_id for offer_id in offer_ids if offer_id != first_command["offer_id"]
    )
    tampered["games"][0]["decisions"] = tampered["games"][0]["decisions"][:1]
    tampered["games"][0]["winner"] = None
    tampered_receipt = replay_teacher_trajectories(
        tampered,
        content_hash=runtime["experience_content_hash"],
        asset_manifest_hash=runtime["asset_manifest_hash"],
        sampled_search_roots=[{"game_index": 0, "decision_index": 0}],
    )
    assert not tampered_receipt.passed
    assert tampered_receipt.search_action_mismatches == 1


def test_root_stability_reports_declared_budgets_without_mutating_roots() -> None:
    result = evaluate_root_stability(
        budgets=[1, 2],
        worlds=1,
        c_puct=1.5,
        roots=2,
        repeat_seeds=[7001, 7002],
        seed=107,
        max_steps=2000,
    )

    assert set(result) == {"1", "2"}
    for budget in result.values():
        assert budget["roots"] == 2
        assert budget["searches"] == 4
        assert budget["repeat_search_seeds"] == [7001, 7002]
        assert 0.0 <= budget["top_action_agreement"] <= 1.0
        assert budget["median_js_divergence"] >= 0.0
        assert budget["mean_tree_nodes"] > 0
        assert budget["mean_max_depth"] >= 1


def test_contract_runtime_fingerprints_are_frozen() -> None:
    import json

    repo_root = Path(__file__).resolve().parents[2]
    contract = json.loads(
        (repo_root / "experiments/contracts/w2-234-teacher1-pilot-v1.json").read_text()
    )
    runtime = runtime_fingerprints(seed=int(contract["seeds"]["runtime"]))

    assert all(
        runtime[key] == expected
        for key, expected in contract["expected_fingerprints"].items()
    )
    assert contract["seeds"]["future_training"] == [197, 419, 887]
    assert contract["teacher"]["budgets"] == [8, 32, 128]
    assert contract["evaluation"]["stability_repeat_search_seeds"] == [
        2197001,
        2197002,
        2197003,
    ]
    seed_blocks = contract["evaluation"]["matchup_seed_blocks"]
    assert [block["seed"] for block in seed_blocks] == [1197, 1419, 1887]
    assert sum(block["games"] for block in seed_blocks) == 48
