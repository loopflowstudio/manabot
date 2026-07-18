#!/usr/bin/env python
"""Generate the INT-13 conditional strategy checked fixture.

Constructs a mid-game Priority position, defines four exact worlds (2×2
factorial: Has/Not Counterspell × Has/Not Lightning Bolt), runs conditional
determinized PUCT for all five conditions, and writes an identity-pinned
JSON fixture. Runs twice and asserts byte-identical output before writing.

Usage:
    uv run python scripts/generate_conditional_strategy_fixture.py
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np

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
from manabot.sim.teacher1_evidence import canonical_sha256
from manabot.verify.util import INTERACTIVE_DECK
import managym

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = (
    REPO_ROOT / "experiments" / "data" / "int-13-conditional-strategy-fixture-v1.json"
)

SEED = 197
RESET_SEED = 42
SIMULATIONS = 8
WORLDS = 4
C_PUCT = 1.5
MAX_STEPS = 200
HERO_SEAT = 0
VILLAIN_SEAT = 1


def _build_root_env() -> managym.Env:
    """Construct the mid-game Priority position via scenario API."""

    env = managym.Env(seed=RESET_SEED, skip_trivial=True)
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
    """Define four exact worlds: 2×2 factorial of Counterspell × Lightning Bolt."""

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


def _run_search() -> ConditionalStrategyResult:
    """Run the conditional search and return the result."""

    root_env = _build_root_env()
    world_space = _build_world_space()
    prior = make_prior(world_space, viewer=HERO_SEAT)
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
        seed=SEED,
        c_puct=C_PUCT,
        max_steps=MAX_STEPS,
        branch_driver_id=REFERENCE_BRANCH_DRIVER_ID,
        branch_audit=True,
        branch_match_id=f"int-13-fixture-{SEED}",
    )
    validate_result(result)
    return result


def _build_fixture(result: ConditionalStrategyResult) -> dict:
    """Assemble the full fixture with identity metadata."""

    payload = serialize_result(result)
    payload["fixture_id"] = "int-13-conditional-strategy-fixture-v1"
    payload["fixture_sha256"] = result_sha256(result)
    payload["description"] = (
        "INT-13 vertical slice: determinized PUCT over conditional world priors. "
        "Five aligned conditions (True, Has, Lacks, Q, Not(Q)) at one semantic root. "
        "Planner: determinized_puct. No ISMCTS or equilibrium claim. "
        "Sampled worlds and hidden truth are audit-private."
    )
    payload["world_design"] = {
        "type": "2x2_factorial",
        "factors": ["has_count counterspell", "has_lightning_bolt"],
        "worlds": [
            {
                "index": 0,
                "label": "cs+lb",
                "opponent_hand": ["Island", "Counterspell", "Lightning Bolt"],
            },
            {
                "index": 1,
                "label": "cs-only",
                "opponent_hand": ["Island", "Counterspell", "Gray Ogre"],
            },
            {
                "index": 2,
                "label": "lb-only",
                "opponent_hand": ["Island", "Lightning Bolt", "Gray Ogre"],
            },
            {
                "index": 3,
                "label": "neither",
                "opponent_hand": ["Island", "Gray Ogre", "Wind Drake"],
            },
        ],
        "conditions": [
            {"id": "true", "query": "all worlds", "worlds": [0, 1, 2, 3]},
            {
                "id": "has:Counterspell",
                "query": "opponent has Counterspell",
                "worlds": [0, 1],
            },
            {
                "id": "not(has:Counterspell)",
                "query": "opponent lacks Counterspell",
                "worlds": [2, 3],
            },
            {
                "id": "has:Lightning Bolt",
                "query": "opponent has Lightning Bolt",
                "worlds": [0, 2],
            },
            {
                "id": "not(has:Lightning Bolt)",
                "query": "opponent lacks Lightning Bolt",
                "worlds": [1, 3],
            },
        ],
    }
    return payload


def main() -> int:
    print("INT-13: Generating conditional strategy fixture...")

    first = _run_search()
    second = _run_search()
    first_json = canonical_result_json(first)
    second_json = canonical_result_json(second)
    if first_json != second_json:
        print("ERROR: conditional search is not deterministic", file=sys.stderr)
        return 1
    print("  Determinism check passed (two runs byte-identical).")

    fixture = _build_fixture(first)
    fixture_text = json.dumps(fixture, sort_keys=True, separators=(",", ":"))
    fixture_bytes = fixture_text.encode("utf-8")

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if FIXTURE_PATH.exists():
        existing = FIXTURE_PATH.read_bytes()
        if existing == fixture_bytes:
            print(f"  Fixture unchanged: {FIXTURE_PATH}")
            return 0
        print(f"  Updating fixture: {FIXTURE_PATH}")
    else:
        print(f"  Writing fixture: {FIXTURE_PATH}")

    FIXTURE_PATH.write_bytes(fixture_bytes)
    print(f"  Fixture SHA-256: {result_sha256(first)}")
    print(f"  Action count: {first.action_count}")
    print(f"  Action labels: {list(first.action_labels)}")
    print(f"  Conditions: {[c.condition_id for c in first.conditions]}")
    for c in first.conditions:
        print(
            f"    {c.condition_id:30s}  mass={c.condition_mass:.3f}  "
            f"support={c.support}  sampled={c.sampled_worlds}  "
            f"root_value={c.root_value:.3f}  uncertainty={c.uncertainty:.4f}  "
            f"visits={c.visit_counts.tolist()}"
        )
    print(f"  Comparison deltas vs True:")
    for cid, deltas in first.comparison_deltas.items():
        print(f"    {cid:30s}  {deltas}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
