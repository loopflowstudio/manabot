"""Tests for INT-13: determinized PUCT over conditional world priors.

Verifies the conditional search vertical slice: five aligned conditions at
one semantic root, identity-pinned checked fixture, fail-closed validation,
and viewer-safe layer with no hidden truth leakage.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from manabot.sim.conditional_search import (
    CONDITION_TRUE,
    ConditionalSearchError,
    ConditionalStrategyResult,
    HasCard,
    NotQuery,
    ScenarioWorldSpace,
    TrueQuery,
    WorldSpec,
    canonical_result_json,
    conditional_determinized_puct,
    make_prior,
    make_query_plan,
    result_sha256,
    serialize_result,
    validate_result,
)
from manabot.sim.search_branch import REFERENCE_BRANCH_DRIVER_ID
from manabot.verify.util import INTERACTIVE_DECK
import managym

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = (
    REPO_ROOT / "experiments" / "data" / "int-13-conditional-strategy-fixture-v1.json"
)

HERO_SEAT = 0
VILLAIN_SEAT = 1
SIMULATIONS = 8
WORLDS = 4
SEED = 197
MAX_STEPS = 200
FAST_SIMS = 4
FAST_MAX_STEPS = 100


def test_import_does_not_reach_training_only_dependencies() -> None:
    script = """
import builtins
import sys

real_import = builtins.__import__

def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split('.', 1)[0] == 'psutil':
        raise ModuleNotFoundError(f'blocked visual dependency: {name}')
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded
import manabot.sim.conditional_search
assert 'manabot.model.train' not in sys.modules
assert 'manabot.sim.flat_mc' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def _build_root_env(seed: int = 42) -> managym.Env:
    """Construct the same mid-game Priority position as the fixture generator."""

    env = managym.Env(seed=seed, skip_trivial=True)
    env.reset(
        [
            managym.PlayerConfig("hero", dict(INTERACTIVE_DECK)),
            managym.PlayerConfig("villain", dict(INTERACTIVE_DECK)),
        ]
    )
    env.scenario_clear_hand(HERO_SEAT)
    env.scenario_clear_hand(VILLAIN_SEAT)
    for card in ("Wind Drake", "Man-o'-War"):
        env.scenario_force_card_in_hand(HERO_SEAT, card)
    for _ in range(3):
        env.scenario_force_battlefield(HERO_SEAT, "Island", True)
    for _ in range(2):
        env.scenario_force_battlefield(HERO_SEAT, "Mountain", True)
    for _ in range(2):
        env.scenario_force_battlefield(VILLAIN_SEAT, "Island", True)
    env.scenario_force_battlefield(VILLAIN_SEAT, "Mountain", True)
    env.scenario_refresh()
    return env


def _build_world_space() -> ScenarioWorldSpace:
    """The same 2×2 factorial world space as the fixture generator."""

    worlds = (
        WorldSpec(0, "cs+lb", 0.25, ("Island", "Counterspell", "Lightning Bolt")),
        WorldSpec(1, "cs-only", 0.25, ("Island", "Counterspell", "Gray Ogre")),
        WorldSpec(2, "lb-only", 0.25, ("Island", "Lightning Bolt", "Gray Ogre")),
        WorldSpec(3, "neither", 0.25, ("Island", "Gray Ogre", "Wind Drake")),
    )
    return ScenarioWorldSpace(
        space_id="scenario-fixture-v1",
        viewer=HERO_SEAT,
        worlds=worlds,
        opponent_seat=VILLAIN_SEAT,
    )


def _run_search(seed: int = SEED) -> ConditionalStrategyResult:
    """Run the conditional search with fixture-matched parameters."""

    root_env = _build_root_env()
    ws = _build_world_space()
    prior = make_prior(ws, viewer=HERO_SEAT)
    plan = make_query_plan(
        has=HasCard("Counterspell"),
        q=HasCard("Lightning Bolt"),
    )
    result = conditional_determinized_puct(
        root_env,
        prior=prior,
        query_plan=plan,
        simulations=SIMULATIONS,
        worlds=WORLDS,
        seed=seed,
        c_puct=1.5,
        max_steps=MAX_STEPS,
        branch_driver_id=REFERENCE_BRANCH_DRIVER_ID,
        branch_audit=True,
        branch_match_id=f"int-13-fixture-{seed}",
    )
    validate_result(result)
    return result


def _run_search_fast(seed: int = SEED) -> ConditionalStrategyResult:
    """Run with small params for fast non-fixture tests."""

    root_env = _build_root_env()
    ws = _build_world_space()
    prior = make_prior(ws, viewer=HERO_SEAT)
    plan = make_query_plan(
        has=HasCard("Counterspell"),
        q=HasCard("Lightning Bolt"),
    )
    result = conditional_determinized_puct(
        root_env,
        prior=prior,
        query_plan=plan,
        simulations=FAST_SIMS,
        worlds=WORLDS,
        seed=seed,
        c_puct=1.5,
        max_steps=FAST_MAX_STEPS,
        branch_driver_id=REFERENCE_BRANCH_DRIVER_ID,
        branch_audit=False,
        branch_match_id=f"int-13-fast-{seed}",
    )
    validate_result(result)
    return result


# ---------------------------------------------------------------------------
# Checked fixture: deterministic and identity-pinned
# ---------------------------------------------------------------------------


def test_frozen_fixture_payload_is_stable_and_runtime_drift_is_explicit() -> None:
    """Keep INT-13 evidence frozen while checking its behavioral payload."""

    first = _run_search()
    second = _run_search()
    assert canonical_result_json(first) == canonical_result_json(second)

    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    regenerated = serialize_result(first)

    for key in set(regenerated) - {"identities"}:
        assert key in fixture, f"key {key!r} missing from checked fixture"
        assert fixture[key] == regenerated[key], (
            f"key {key!r} differs between regenerated and checked fixture"
        )

    frozen_identities = dict(fixture["identities"])
    runtime_identities = dict(regenerated["identities"])
    assert frozen_identities.pop("engine_source_sha256") == (
        "6acc1f17d1dec836fc0f1f2a1dc25cab740c449694d2c375862ba776fb3b6e03"
    )
    assert runtime_identities.pop("engine_source_sha256") == (
        "b1e8274ab7a7a3f447072c2ffddb537eafba8f57e3f625f3859b2eec7ba6127b"
    )
    assert frozen_identities == runtime_identities

    assert fixture["planner"] == "determinized_puct"
    assert fixture["fixture_sha256"] == (
        "97f01c893af7da31c6a340c482eef1ae9def8616dcd6b95c1efedfd55dc9f4e4"
    )
    assert fixture["root_state_digest"] == first.root_state_digest
    assert fixture["prior_sha256"] == first.prior_sha256
    assert fixture["plan_sha256"] == first.plan_sha256


# ---------------------------------------------------------------------------
# Condition alignment
# ---------------------------------------------------------------------------


def test_condition_results_are_aligned() -> None:
    """All 5 conditions share the same action count and action labels."""

    result = _run_search_fast()
    assert len(result.conditions) == 5
    for cr in result.conditions:
        assert len(cr.visit_counts) == result.action_count
        assert len(cr.q_values) == result.action_count
        assert cr.world_q_values.shape[1] == result.action_count
    assert len(result.action_labels) == result.action_count


# ---------------------------------------------------------------------------
# Condition mass and support
# ---------------------------------------------------------------------------


def test_condition_mass_and_support() -> None:
    """Mass and support match the 2×2 factorial world design."""

    result = _run_search_fast()
    by_id = result.condition_by_id

    assert by_id[CONDITION_TRUE].condition_mass == pytest.approx(1.0)
    assert by_id[CONDITION_TRUE].support == 4
    assert by_id[CONDITION_TRUE].sampled_worlds == 4

    assert by_id["has:Counterspell"].condition_mass == pytest.approx(0.5)
    assert by_id["has:Counterspell"].support == 2
    assert by_id["has:Counterspell"].sampled_worlds == 2

    assert by_id["not(has:Counterspell)"].condition_mass == pytest.approx(0.5)
    assert by_id["not(has:Counterspell)"].support == 2

    assert by_id["has:Lightning Bolt"].condition_mass == pytest.approx(0.5)
    assert by_id["has:Lightning Bolt"].support == 2

    assert by_id["not(has:Lightning Bolt)"].condition_mass == pytest.approx(0.5)
    assert by_id["not(has:Lightning Bolt)"].support == 2


# ---------------------------------------------------------------------------
# Comparison deltas
# ---------------------------------------------------------------------------


def test_comparison_deltas_are_nonzero_and_correct() -> None:
    """Deltas vs True are present and nonzero for divergent conditions."""

    result = _run_search_fast()
    true_cr = result.conditions[0]
    assert true_cr.condition_id == CONDITION_TRUE

    for cr in result.conditions[1:]:
        deltas = result.comparison_deltas[cr.condition_id]
        assert "root_value_delta" in deltas
        assert "uncertainty_delta" in deltas
        assert "visit_dist_l1" in deltas
        assert "q_max_abs_delta" in deltas
        assert "top_action_changed" in deltas
        expected_rv_delta = cr.root_value - true_cr.root_value
        assert deltas["root_value_delta"] == pytest.approx(expected_rv_delta)

    has_cs = result.comparison_deltas["has:Counterspell"]
    lacks_cs = result.comparison_deltas["not(has:Counterspell)"]
    assert has_cs["root_value_delta"] == pytest.approx(-lacks_cs["root_value_delta"])
    assert has_cs["visit_dist_l1"] >= 0.0
    assert has_cs["q_max_abs_delta"] >= 0.0


# ---------------------------------------------------------------------------
# Fail-closed: empty support
# ---------------------------------------------------------------------------


def test_fail_closed_on_empty_support() -> None:
    """A condition no world satisfies raises ConditionalSearchError."""

    root_env = _build_root_env()
    ws = _build_world_space()
    prior = make_prior(ws, viewer=HERO_SEAT)
    plan = make_query_plan(
        has=HasCard("Counterspell"),
        q=HasCard("Ancestral Recall"),
    )
    with pytest.raises(ConditionalSearchError, match="empty support"):
        conditional_determinized_puct(
            root_env,
            prior=prior,
            query_plan=plan,
            simulations=4,
            worlds=2,
            seed=SEED,
            max_steps=200,
            branch_driver_id=REFERENCE_BRANCH_DRIVER_ID,
        )


# ---------------------------------------------------------------------------
# Fail-closed: action misalignment
# ---------------------------------------------------------------------------


def test_fail_closed_on_action_misalignment() -> None:
    """If world configuration changes the root action count, the search raises."""

    root_env = _build_root_env()
    bad_worlds = (WorldSpec(0, "w0", 1.0, ("Wind Drake", "Man-o'-War", "Gray Ogre")),)
    ws = ScenarioWorldSpace(
        space_id="bad-fixture",
        viewer=HERO_SEAT,
        worlds=bad_worlds,
        opponent_seat=HERO_SEAT,
    )
    prior = make_prior(ws, viewer=HERO_SEAT)
    plan = make_query_plan(
        has=HasCard("Wind Drake"),
        q=HasCard("Gray Ogre"),
    )
    with pytest.raises(ConditionalSearchError, match="changed root action count"):
        conditional_determinized_puct(
            root_env,
            prior=prior,
            query_plan=plan,
            simulations=4,
            worlds=1,
            seed=SEED,
            max_steps=200,
            branch_driver_id=REFERENCE_BRANCH_DRIVER_ID,
        )


# ---------------------------------------------------------------------------
# Fail-closed: identity mismatch (tampered fixture)
# ---------------------------------------------------------------------------


def test_fail_closed_on_identity_mismatch() -> None:
    """A tampered fixture SHA-256 is detected by regeneration."""

    result = _run_search_fast()
    original_sha = result_sha256(result)
    tampered = serialize_result(result)
    tampered["action_count"] = result.action_count + 1
    tampered_sha = (
        __import__("hashlib")
        .sha256(
            json.dumps(
                {k: v for k, v in tampered.items() if k != "fixture_id"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        .hexdigest()
    )
    assert tampered_sha != original_sha


# ---------------------------------------------------------------------------
# World query partition
# ---------------------------------------------------------------------------


def test_world_query_partition_is_correct() -> None:
    """TrueQuery, HasCard, and NotQuery correctly partition the 4 worlds."""

    ws = _build_world_space()
    worlds = ws.worlds

    true_q = TrueQuery()
    has_cs = HasCard("Counterspell")
    lacks_cs = NotQuery(has_cs)
    has_lb = HasCard("Lightning Bolt")
    lacks_lb = NotQuery(has_lb)

    assert all(true_q.matches(w) for w in worlds)

    cs_worlds = [w for w in worlds if has_cs.matches(w)]
    assert {w.world_index for w in cs_worlds} == {0, 1}

    no_cs_worlds = [w for w in worlds if lacks_cs.matches(w)]
    assert {w.world_index for w in no_cs_worlds} == {2, 3}

    lb_worlds = [w for w in worlds if has_lb.matches(w)]
    assert {w.world_index for w in lb_worlds} == {0, 2}

    no_lb_worlds = [w for w in worlds if lacks_lb.matches(w)]
    assert {w.world_index for w in no_lb_worlds} == {1, 3}

    assert {w.world_index for w in cs_worlds} | {
        w.world_index for w in no_cs_worlds
    } == {0, 1, 2, 3}
    assert {w.world_index for w in cs_worlds} & {
        w.world_index for w in no_cs_worlds
    } == set()


# ---------------------------------------------------------------------------
# Viewer-safe layer has no hidden truth
# ---------------------------------------------------------------------------


def test_viewer_safe_layer_has_no_hidden_truth() -> None:
    """The viewer-safe projection contains no opponent hands or world labels.

    Condition IDs (e.g. ``has:Counterspell``) are query labels, not hidden
    truth — they name the hypothesis being tested, not the opponent's actual
    hand. The viewer-safe layer must not contain opponent hand contents,
    world labels, branch receipts, or per-world Q/value arrays.
    """

    result = _run_search_fast()
    payload = serialize_result(result)
    viewer_safe = payload["viewer_safe"]
    viewer_json = json.dumps(viewer_safe, sort_keys=True)

    for world in _build_world_space().worlds:
        assert world.label not in viewer_json, (
            f"world label '{world.label}' leaked into viewer-safe layer"
        )

    for condition in viewer_safe["conditions"]:
        assert "branch_receipt" not in condition
        assert "world_q_values" not in condition
        assert "world_root_values" not in condition
        assert "opponent_hand" not in condition
        assert "tree_nodes" not in condition
        assert "max_depth" not in condition
        assert "cap_hits" not in condition
        assert "branch_driver_id" not in condition

    assert "branch_receipt" not in viewer_json
    assert "opponent_hand" not in viewer_json


# ---------------------------------------------------------------------------
# Planner named determinized_puct
# ---------------------------------------------------------------------------


def test_planner_named_determinized_puct() -> None:
    """The planner is 'determinized_puct', no ISMCTS or equilibrium claim."""

    result = _run_search_fast()
    assert result.planner == "determinized_puct"

    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert fixture["planner"] == "determinized_puct"
    assert "No ISMCTS or equilibrium claim" in fixture["description"]


# ---------------------------------------------------------------------------
# Search does not mutate root
# ---------------------------------------------------------------------------


def test_search_does_not_mutate_root() -> None:
    """Root state digest is unchanged after all 5 conditions."""

    root_env = _build_root_env()
    digest_before = root_env.state_digest()
    action_count_before = root_env.action_count()
    agent_before = root_env.current_agent_index()

    ws = _build_world_space()
    prior = make_prior(ws, viewer=HERO_SEAT)
    plan = make_query_plan(
        has=HasCard("Counterspell"),
        q=HasCard("Lightning Bolt"),
    )
    conditional_determinized_puct(
        root_env,
        prior=prior,
        query_plan=plan,
        simulations=8,
        worlds=4,
        seed=SEED,
        max_steps=200,
        branch_driver_id=REFERENCE_BRANCH_DRIVER_ID,
    )

    assert root_env.state_digest() == digest_before
    assert root_env.action_count() == action_count_before
    assert root_env.current_agent_index() == agent_before


# ---------------------------------------------------------------------------
# Determinism: two calls with same seed produce identical results
# ---------------------------------------------------------------------------


def test_two_runs_same_seed_are_identical() -> None:
    """Same seed + same inputs → byte-identical output."""

    first = _run_search_fast(seed=SEED)
    second = _run_search_fast(seed=SEED)
    assert canonical_result_json(first) == canonical_result_json(second)


# ---------------------------------------------------------------------------
# Validate result fail-closed
# ---------------------------------------------------------------------------


def test_validate_result_rejects_tampered_planner() -> None:
    """validate_result raises on wrong planner name."""

    result = _run_search_fast()
    from dataclasses import replace

    tampered = replace(result, planner="ismcts")
    with pytest.raises(ConditionalSearchError, match="planner"):
        validate_result(tampered)


def test_validate_result_rejects_wrong_condition_count() -> None:
    """validate_result raises on wrong number of conditions."""

    result = _run_search_fast()
    from dataclasses import replace

    tampered = replace(result, conditions=result.conditions[:3])
    with pytest.raises(ConditionalSearchError, match="5 conditions"):
        validate_result(tampered)
