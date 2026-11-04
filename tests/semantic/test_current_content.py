"""The world's one catalog binds learning projections and the world pool reports support.

Setup rejection is a native test in `managym/src/cards/setups.rs`; these
checks cover the Python consumers.
"""

import json
from pathlib import Path

from managym.decision import Command, DecisionFrame, Observation
from managym.experience_projection import describe_actions, serialize_observation
import pytest

from manabot.semantic.learning import FROZEN_VOCABULARY_LABELS, BoundSemanticPack
from manabot.semantic.policy_catalog import bind_policy_catalog
import managym

CONTENT_ROOT = Path(__file__).resolve().parents[2] / "content/semantic"
SETUPS = json.loads((CONTENT_ROOT / "current/setups.json").read_text())


def _manifest() -> dict:
    pool = managym.content_pool("world")
    return {
        "schema_version": 1,
        "content_digest": pool["content_digest"],
        "definitions": [
            {"card_def_id": row["card_def_id"], "registry_name": row["registry_name"]}
            for row in pool["definitions"]
        ],
    }


def test_the_catalog_binds_learning_projections() -> None:
    pack_keys = {setup["pack_key"] for setup in SETUPS.values()} | {"world"}
    assert FROZEN_VOCABULARY_LABELS <= pack_keys
    manifest = _manifest()
    bound = BoundSemanticPack.bind(manifest)
    assert bound.definition_count == len(manifest["definitions"])
    assert bound.definition_count == managym.CARD_DEFINITION_COUNT
    assert len(bound.definition_row_by_card_def_id) == bound.definition_count


def test_retired_authored_pack_manifests_do_not_bind() -> None:
    manifest = {**_manifest(), "pack_key": "tla-baseline-v1"}
    with pytest.raises(ValueError, match="retired"):
        BoundSemanticPack.bind(manifest)
    assert BoundSemanticPack.bind({**_manifest(), "pack_key": None})


def test_catalog_must_match_the_live_pack_digest() -> None:
    manifest = {**_manifest(), "content_digest": "0" * 64}
    with pytest.raises(ValueError, match="differs from the live pack"):
        BoundSemanticPack.bind(manifest)


def test_the_world_is_the_only_content_pool() -> None:
    for name in ("missing-pack", "tla-baseline-v1"):
        with pytest.raises(ValueError, match="unknown content pool"):
            managym.content_pool(name)


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
def test_five_completed_cards_are_supported(name: str) -> None:
    pool = managym.content_pool("world")
    entry = next(row for row in pool["definitions"] if row["registry_name"] == name)
    assert entry["eligibility"] == "eligible"
    assert entry["reason"] is None


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
    pool = managym.content_pool("world")
    entry = next(row for row in pool["definitions"] if row["registry_name"] == name)
    assert entry["eligibility"] == "eligible"
    assert entry["reason"] is None


def test_set_filter_narrows_the_world_pool_by_key_prefix() -> None:
    tla = {row["registry_name"] for row in managym.set_cards(["tla"])}
    with_basics = {row["registry_name"] for row in managym.set_cards(["tla", "lea"])}
    assert "Avatar Enthusiasts" in tla and "Forest" not in tla
    assert "Forest" in with_basics and "Pop Quiz" not in with_basics
    assert all(
        row["key"].startswith(("tla.", "lea."))
        for row in managym.set_cards(["tla", "lea"])
    )


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
    assert bind_policy_catalog("world").semantic_pack_hash == bound.semantic_pack_hash
    pool = managym.content_pool("world")
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
