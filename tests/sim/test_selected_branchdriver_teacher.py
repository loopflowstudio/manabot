"""Production selected-BranchDriver integration tests for visit search."""

from __future__ import annotations

import json

import numpy as np
import pytest

from manabot.sim.mcts import determinized_puct
from manabot.sim.search_branch import (
    REFERENCE_BRANCH_DRIVER_ID,
    SELECTED_BRANCH_DRIVER_ID,
)
from manabot.verify.util import GW_ALLIES_DECK, UR_LESSONS_DECK
import managym


def _authored_engine(seed: int = 1197) -> managym.Env:
    env = managym.Env(seed=seed, skip_trivial=True)
    env.reset(
        [
            managym.PlayerConfig("ur", dict(UR_LESSONS_DECK)),
            managym.PlayerConfig("gw", dict(GW_ALLIES_DECK)),
        ]
    )
    return env


def _normalized_tape(receipt: dict[str, object]) -> list[dict[str, object]]:
    tapes = receipt["tapes"]
    assert isinstance(tapes, dict)
    normalized: list[dict[str, object]] = []
    for site in ("world", "child", "leaf"):
        rows = tapes[site]
        assert isinstance(rows, list)
        for row in rows:
            native = dict(row["native_receipt"])
            native.pop("driver_id")
            normalized.append(
                {
                    "site": row["site"],
                    "policy_index": row["policy_index"],
                    "offer_id": row["offer_id"],
                    "command": row["command"],
                    "source": row["source"],
                    "native_receipt": native,
                    "post_apply_witness": row["post_apply_witness"],
                }
            )
    return normalized


def test_selected_puct_uses_reconciled_world_child_and_leaf_commands() -> None:
    root = _authored_engine()
    before = root.search_witness_json()

    result = determinized_puct(
        root,
        simulations=8,
        worlds=1,
        seed=1419,
        max_steps=200,
        branch_audit=True,
        branch_match_id="rul-2-focused",
    )

    assert result.branch_driver_id == SELECTED_BRANCH_DRIVER_ID
    assert root.search_witness_json() == before
    counters = result.branch_receipt["counters"]
    tapes = result.branch_receipt["tapes"]
    for site in ("world", "child", "leaf"):
        assert counters["applies"][site] == len(tapes[site]) > 0
    assert sum(counters["applies"].values()) == sum(
        len(tapes[site]) for site in ("world", "child", "leaf")
    )
    assert result.branch_receipt["reconciliation"] == {
        "per_site_and_total": True,
        "zero_unmeasured_fallback": True,
    }
    for row in _normalized_tape(result.branch_receipt):
        command = row["command"]
        assert command["offer_id"] == row["offer_id"]
        assert command["expected_revision"] == row["source"]["expected_revision"]
        assert command["prompt_id"] == row["source"]["prompt_id"]


def test_selected_and_reference_backends_are_exact_on_real_puct() -> None:
    selected_root = _authored_engine(seed=1887)
    reference_root = _authored_engine(seed=1887)
    kwargs = {
        "simulations": 4,
        "worlds": 1,
        "seed": 2197,
        "max_steps": 120,
        "branch_audit": True,
        "branch_match_id": "rul-2-differential",
    }

    selected = determinized_puct(selected_root, **kwargs)
    reference = determinized_puct(
        reference_root,
        branch_driver_id=REFERENCE_BRANCH_DRIVER_ID,
        **kwargs,
    )

    assert np.array_equal(selected.visit_counts, reference.visit_counts)
    assert np.array_equal(selected.q_values, reference.q_values)
    assert selected.root_value == reference.root_value
    assert selected.cap_hits == reference.cap_hits
    assert selected.tree_nodes == reference.tree_nodes
    assert selected.max_depth == reference.max_depth
    assert _normalized_tape(selected.branch_receipt) == _normalized_tape(
        reference.branch_receipt
    )


def test_guarded_selected_branch_poisoned_indexed_paths_fail() -> None:
    root = _authored_engine()
    runtime = managym.SelectedBranchRuntime("rul-2-poison", True)
    branch = runtime.fork_exact(root, "world")
    offers = branch.structured_search_offers()
    submission = json.dumps({"offer_id": 0, "answers": []})

    poisoned = (
        lambda: branch.step(0),
        branch.clone_env,
        lambda: branch.step_structured(offers, submission),
        lambda: branch.step_legacy_submission(offers, submission),
        lambda: branch.random_playout(seed=7),
        lambda: branch.determinize(seed=7),
    )
    for operation in poisoned:
        with pytest.raises(
            managym.AgentError, match="selected production branch forbids"
        ):
            operation()

    runtime.determinize(branch, 7, int(branch.current_agent_index()))
    *_, receipt_json = runtime.apply_policy_choice(branch, "world", 0)
    snapshot = json.loads(runtime.snapshot_json())
    assert snapshot["counters"]["applies"]["world"] == 1
    assert snapshot["counters"]["indexed_fallbacks"] == 0
    receipt = json.loads(receipt_json)
    command = receipt["command"]
    assert command == snapshot["tapes"]["world"][0]["command"]
    assert command["match_id"] == "rul-2-poison"
    assert command["expected_revision"] == receipt["source"]["expected_revision"]
    assert command["prompt_id"] == receipt["source"]["prompt_id"]
    assert command["offer_id"] == receipt["offer_id"]
    assert command["answers"] == []


@pytest.mark.parametrize(
    ("policy_index", "override", "message"),
    [
        (999, None, "out of range"),
        (0, {"offer_id": 999}, "offer ID precondition mismatch"),
        (0, {"prompt_id": 999}, "prompt ID precondition mismatch"),
        (0, {"expected_revision": 999}, "revision precondition mismatch"),
        (0, {"authority_hash": "bad"}, "authority precondition mismatch"),
        (0, {"legal_surface_hash": "bad"}, "legal-surface precondition mismatch"),
    ],
)
def test_selected_preconditions_fail_closed_without_native_apply(
    policy_index: int, override: dict[str, object] | None, message: str
) -> None:
    root = _authored_engine()
    runtime = managym.SelectedBranchRuntime("rul-2-negative", True)
    branch = runtime.fork_exact(root, "world")
    before = branch.search_witness_json()
    before_counters = json.loads(runtime.snapshot_json())["counters"]

    with pytest.raises(managym.AgentError, match=message):
        runtime.apply_policy_choice(
            branch,
            "world",
            policy_index,
            None if override is None else json.dumps(override),
        )

    assert branch.search_witness_json() == before
    assert json.loads(runtime.snapshot_json())["counters"] == before_counters


def test_unknown_driver_cannot_fall_back() -> None:
    with pytest.raises(ValueError, match="unknown search branch driver"):
        determinized_puct(
            _authored_engine(),
            simulations=1,
            worlds=1,
            seed=1,
            branch_driver_id="unknown/driver",
        )
