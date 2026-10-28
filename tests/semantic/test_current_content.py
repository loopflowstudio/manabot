"""Current authoring has one owner; qualifications cannot survive clause edits."""

import json
from pathlib import Path
import shutil

import pytest

from manabot.semantic.compiler import SemanticCompileError, compile_source
from manabot.semantic.content import CONTENT_ROOT, ContentSource, generate
from manabot.semantic.learning import BoundSemanticPack


def test_current_outputs_and_all_learning_projections() -> None:
    generate(check=True)
    source = ContentSource.load()
    for preset in [*source.setups, "pool"]:
        ir, _ = source.project(preset)
        manifest = {
            "schema_version": 1,
            "content_digest": "projection-test",
            "definitions": [
                {"card_def_id": i, "registry_name": d["content_pack_binding"]["value"]}
                for i, d in enumerate(ir["definitions"])
            ],
        }
        bound = BoundSemanticPack.bind(
            manifest,
            ir_path=CONTENT_ROOT / "generated" / f"{preset}.ir.json",
            schema_path=CONTENT_ROOT / "learning_schema.json",
        )
        assert len(bound.definition_row_by_card_def_id) == len(ir["definitions"])


def test_missing_token_and_changed_qualified_clause_fail_closed(tmp_path: Path) -> None:
    source = ContentSource.load()
    ally = next(
        k for k, d in source.definitions.items() if d["registry_name"] == "Ally"
    )
    del source.definitions[ally]
    with pytest.raises(SemanticCompileError, match="missing content definition"):
        source.project("two_deck")

    shutil.copytree(CONTENT_ROOT, tmp_path / "content")
    path = tmp_path / "content" / "definitions.json"
    rows = json.loads(path.read_text())
    enthusiast = next(d for d in rows if d["registry_name"] == "Avatar Enthusiasts")
    enthusiast["programs"] = []
    path.write_text(json.dumps(rows))
    with pytest.raises(SemanticCompileError, match="stale qualification"):
        generate(root=path.parent)


def test_new_definition_compiles_without_a_native_registration() -> None:
    ir, _ = compile_source(
        {
            "schema_version": 1,
            "pack_key": "authoring-proof",
            "decks": [],
            "pool": ["proof.draw"],
            "definitions": [
                {
                    "key": "proof.draw",
                    "registry_name": "Unregistered authoring witness",
                    "text_box": "Draw a card.",
                    "characteristics": {"types": ["sorcery"], "mana_cost": "U"},
                    "programs": [
                        {
                            "key": "draw",
                            "kind": "spell",
                            "cost": {"mana": "U"},
                            "ops": [{"op": "draw_cards", "count": 1}],
                        }
                    ],
                }
            ],
        }
    )
    assert ir["definitions"][0]["text_box"] == "Draw a card."
    assert ir["programs"][0]["instructions"][0]["op_name"] == "draw_cards"


def test_qualified_producer_rejects_changed_token_characteristics(
    tmp_path: Path,
) -> None:
    shutil.copytree(CONTENT_ROOT, tmp_path / "content")
    path = tmp_path / "content" / "definitions.json"
    rows = json.loads(path.read_text())
    ally = next(row for row in rows if row["registry_name"] == "Ally")
    ally["characteristics"]["power"] = 7
    path.write_text(json.dumps(rows))
    with pytest.raises(SemanticCompileError, match="stale qualification.*treetop"):
        generate(root=path.parent)


def test_current_projections_preserve_every_historical_setup_count() -> None:
    source = ContentSource.load()
    for preset in source.setups:
        if preset == "baseline":
            continue
        directory, filename = (
            ("w11", "elemental_world")
            if preset == "optional_payments"
            else ("v1", preset)
        )
        historical = json.loads(
            (
                CONTENT_ROOT.parent / directory / "generated" / f"{filename}.ir.json"
            ).read_text()
        )
        current, _ = source.project(preset)
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
    from managym.decision import Command, DecisionFrame, Observation
    from managym.experience_projection import describe_actions, serialize_observation

    from manabot.semantic.policy_catalog import bind_policy_catalog
    import managym

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
