"""Current cards compile natively; qualifications cannot survive clause edits.

Clause-edit and missing-token rejection are native tests in
`managym/src/semantic/compile.rs`; these checks cover the Python consumers.
"""

import json
from pathlib import Path

from managym.decision import Command, DecisionFrame, Observation
from managym.experience_projection import describe_actions, serialize_observation
import pytest

from manabot.semantic.learning import (
    CURRENT_PACK_KEYS,
    BoundSemanticPack,
    current_ir_document,
)
from manabot.semantic.policy_catalog import bind_policy_catalog
import managym

CONTENT_ROOT = Path(__file__).resolve().parents[2] / "content/semantic"
SETUPS = json.loads((CONTENT_ROOT / "current/setups.json").read_text())


def test_all_current_packs_bind_learning_projections() -> None:
    pack_keys = {setup["pack_key"] for setup in SETUPS.values()} | {"tla-pool-v1"}
    assert CURRENT_PACK_KEYS <= pack_keys
    for pack_key in sorted(pack_keys):
        ir = current_ir_document(pack_key)
        assert ir["pack_key"] == pack_key
        manifest = {
            "schema_version": 1,
            "content_digest": "projection-test",
            "definitions": [
                {"card_def_id": i, "registry_name": d["content_pack_binding"]["value"]}
                for i, d in enumerate(ir["definitions"])
            ],
        }
        if pack_key in CURRENT_PACK_KEYS:
            bound = BoundSemanticPack.bind(
                {**manifest, "compiled_semantics": _receipt(ir)}
            )
            assert bound.ir.ir_hash == ir["ir_hash"]
            assert len(bound.definition_row_by_card_def_id) == len(ir["definitions"])


def _receipt(ir: dict) -> dict[str, str]:
    return {key: ir[key] for key in ("pack_key", "ir_hash", "source_hash")}


def test_unknown_pack_has_no_current_ir() -> None:
    with pytest.raises(ValueError, match="unknown content pack"):
        current_ir_document("missing-pack")


@pytest.mark.parametrize(
    "name",
    [
        "Accumulate Wisdom",
        "Cat-Gator",
        "Lightning Strike",
        "Suki, Kyoshi Warrior",
        "Water Tribe Rallier",
    ],
)
def test_cards_with_unresolved_clause_prerequisites_stay_out_of_pool(name: str) -> None:
    pool = managym.content_pool("tla-pool-v1")
    entry = next(row for row in pool["definitions"] if row["registry_name"] == name)
    assert entry["card_def_id"] is not None  # Preserved authored availability.
    assert entry["eligibility"] == "unsupported"
    assert entry["reason"]


@pytest.mark.parametrize(
    "name", ["Wartime Protestors", "Master Piandao", "Gran-Gran", "Allies at Last"]
)
def test_reviewed_cards_are_eligible_in_the_native_pool(name: str) -> None:
    pool = managym.content_pool("tla-pool-v1")
    entry = next(row for row in pool["definitions"] if row["registry_name"] == name)
    assert entry["card_def_id"] is not None
    assert entry["eligibility"] == "eligible"
    assert entry["reason"] is None


def test_current_projections_preserve_every_historical_setup_count() -> None:
    for preset, setup in SETUPS.items():
        if preset == "baseline":
            continue
        directory, filename = (
            ("w11", "elemental_world")
            if preset == "optional_payments"
            else ("v1", preset)
        )
        historical = json.loads(
            (CONTENT_ROOT / directory / "generated" / f"{filename}.ir.json").read_text()
        )
        current = current_ir_document(setup["pack_key"])
        for old, new in zip(historical["decks"], current["decks"], strict=True):
            assert old["key"] == new["key"]
            for field in ("cards", "sideboard"):
                expected = {
                    historical["definitions"][entry["definition_index"]][
                        "content_pack_binding"
                    ]["value"]: entry["count"]
                    for entry in old.get(field, [])
                }
                actual = {
                    current["definitions"][entry["definition_index"]][
                        "content_pack_binding"
                    ]["value"]: entry["count"]
                    for entry in new.get(field, [])
                }
                assert actual == expected, (preset, old["key"], field)


def test_baseline_card_executes_shared_commands_and_forks_with_learning_and_academy_views() -> (
    None
):
    env = managym.Env(seed=28, skip_trivial=False)
    obs, _ = env.reset(
        [
            managym.authored_deck_setup("tla-baseline-v1", key)
            for key in ("tla_red", "tla_white")
        ]
    )
    for _ in range(20):
        if int(obs.turn.step) == 3:
            break
        index = next(
            row["index"]
            for row in describe_actions(obs)
            if row["type"] == "PRIORITY_PASS_PRIORITY"
        )
        obs = env.step(index)[0]
    else:
        pytest.fail("fixture did not reach the first main phase")
    env.scenario_clear_hand(0)
    env.scenario_clear_hand(1)
    env.scenario_force_card_in_hand(0, "Treetop Freedom Fighters")
    for _ in range(3):
        env.scenario_force_battlefield(0, "Mountain", True)
    obs = env.scenario_refresh()
    bound = BoundSemanticPack.from_env(env)
    policy = bind_policy_catalog("tla-baseline-v1")
    assert bound.ir.ir_hash == policy.ir.ir_hash
    full_pool = bind_policy_catalog("tla-pool-v1")
    assert full_pool.ir.raw["pack_key"] == "tla-pool-v1"
    pool = managym.content_pool("tla-baseline-v1")
    assert pool["content_digest"] == env.content_pack_manifest()["content_digest"]
    assert (
        next(
            row
            for row in pool["definitions"]
            if row["registry_name"] == "Treetop Freedom Fighters"
        )["eligibility"]
        == "eligible"
    )
    assert Observation.from_json(
        env.semantic_observation_json(1)
    ).opponent_hand_is_hidden()

    root = env.state_digest()
    fork = env.clone_env()
    capture = env.capture_decision(0)
    frame = DecisionFrame.from_json(env.semantic_decision_frame_json())
    cast = next(
        row
        for row in describe_actions(obs)
        if row["description"] == "Cast Treetop Freedom Fighters"
    )
    command = Command(
        "tla-cast", frame.revision, frame.offers[cast["index"]]["id"]
    ).to_json()
    with pytest.raises(managym.AgentError, match="wrong_origin"):
        fork.execute_captured_command_json(capture, command)
    fork.execute_semantic_command_json(command)
    assert env.state_digest() == root
    env.execute_captured_command_json(capture, command)
    assert env.state_digest() == fork.state_digest()
    after_cast = env.state_digest()
    with pytest.raises(managym.AgentError, match="stale_capture"):
        env.execute_captured_command_json(capture, command)
    assert env.state_digest() == after_cast

    for ordinal in range(12):
        view = env.observation_for_player(0)
        names = [card.name for card in view.agent_cards if int(card.zone) == 2]
        if "Treetop Freedom Fighters" in names and "Ally" in names:
            break
        frame = DecisionFrame.from_json(env.semantic_decision_frame_json())
        offer = frame.find_verb("pass_priority")
        command = Command(f"resolve-{ordinal}", frame.revision, offer["id"]).to_json()
        env.execute_semantic_command_json(command)
        fork.execute_semantic_command_json(command)
        assert env.state_digest() == fork.state_digest()
    else:
        pytest.fail("Treetop did not resolve and create its Ally")
    projection = bound.project_observation(view, identity_mode="semantic_only")
    assert not projection.opaque_identity_valid.any()
    academy = serialize_observation(view)
    battlefield = academy["agent"]["battlefield"]
    assert sum(row["name"] == "Ally" for row in battlefield) == 1
    assert any(row["name"] == "Treetop Freedom Fighters" for row in battlefield)
