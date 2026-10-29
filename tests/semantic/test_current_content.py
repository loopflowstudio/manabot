"""Current packs bind learning catalogs and the world pool reports support.

Setup and token-closure rejection are native tests in
`managym/src/semantic/setups.rs`; these checks cover the Python consumers.
"""

import json
from pathlib import Path

from managym.decision import Command, DecisionFrame, Observation
from managym.experience_projection import describe_actions, serialize_observation
import pytest

from manabot.semantic.learning import CURRENT_PACK_KEYS, BoundSemanticPack
from manabot.semantic.policy_catalog import bind_policy_catalog
import managym

CONTENT_ROOT = Path(__file__).resolve().parents[2] / "content/semantic"
SETUPS = json.loads((CONTENT_ROOT / "current/setups.json").read_text())


def _manifest(pack_key: str) -> dict:
    pool = managym.content_pool(pack_key)
    return {
        "schema_version": 1,
        "content_digest": pool["content_digest"],
        "pack_key": pack_key,
        "definitions": [
            {"card_def_id": row["card_def_id"], "registry_name": row["registry_name"]}
            for row in pool["definitions"]
        ],
    }


def test_all_current_packs_bind_learning_projections() -> None:
    pack_keys = {setup["pack_key"] for setup in SETUPS.values()} | {"tla-pool-v1"}
    assert CURRENT_PACK_KEYS <= pack_keys
    for pack_key in sorted(pack_keys):
        manifest = _manifest(pack_key)
        bound = BoundSemanticPack.bind(manifest)
        assert bound.pack_key == pack_key
        assert bound.definition_count == len(manifest["definitions"])
        assert len(bound.definition_row_by_card_def_id) == bound.definition_count


def test_catalog_must_match_the_live_pack_digest() -> None:
    manifest = {**_manifest("tla-baseline-v1"), "content_digest": "0" * 64}
    with pytest.raises(ValueError, match="differs from the live pack"):
        BoundSemanticPack.bind(manifest)


def test_unknown_pack_has_no_learning_catalog() -> None:
    with pytest.raises(ValueError, match="unknown content pool"):
        managym.learning_catalog("missing-pack")


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
def test_cards_marked_incomplete_are_unsupported(name: str) -> None:
    pool = managym.content_pool("tla-pool-v1")
    entry = next(row for row in pool["definitions"] if row["registry_name"] == name)
    assert entry["card_def_id"] is not None  # Fixed setups still run them.
    assert entry["eligibility"] == "unsupported"
    assert entry["reason"]


@pytest.mark.parametrize(
    "name",
    [
        "Wartime Protestors",
        "Master Piandao",
        "Gran-Gran",
        "Allies at Last",
        "Forest",
        "Pop Quiz",
    ],
)
def test_defined_cards_are_eligible_in_the_world_pool(name: str) -> None:
    pool = managym.content_pool("tla-pool-v1")
    entry = next(row for row in pool["definitions"] if row["registry_name"] == name)
    assert entry["card_def_id"] is not None
    assert entry["eligibility"] == "eligible"
    assert entry["reason"] is None


def test_set_filter_narrows_the_world_pool_by_key_prefix() -> None:
    tla = {row["registry_name"] for row in managym.set_cards(["tla"])}
    with_basics = {row["registry_name"] for row in managym.set_cards(["tla", "lea"])}
    assert "Avatar Enthusiasts" in tla and "Forest" not in tla
    assert "Forest" in with_basics and "Pop Quiz" not in with_basics
    assert all(
        row["semantic_key"].startswith(("tla.", "lea."))
        for row in managym.set_cards(["tla", "lea"])
    )


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
        pack_key = setup["pack_key"]
        for old in historical["decks"]:
            current = managym.authored_deck_setup(pack_key, old["key"])
            for field, actual in (
                ("cards", current.decklist),
                ("sideboard", current.sideboard),
            ):
                expected = {
                    historical["definitions"][entry["definition_index"]][
                        "content_pack_binding"
                    ]["value"]: entry["count"]
                    for entry in old.get(field, [])
                }
                assert dict(actual) == expected, (preset, old["key"], field)
        assert len(managym.authored_deck_keys(pack_key)) == len(historical["decks"])


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
    assert bound.semantic_pack_hash == policy.semantic_pack_hash
    full_pool = bind_policy_catalog("tla-pool-v1")
    assert full_pool.pack_key == "tla-pool-v1"
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
