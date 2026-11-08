"""Selected Commands bind current authority and retain full post-apply evidence."""

import json

import pytest

from manabot.env.match import Match
from manabot.infra.config import MatchConfig
import managym


@pytest.fixture
def players() -> list[managym.PlayerConfig]:
    return Match(
        MatchConfig(
            hero_deck={"Mountain": 8, "Gray Ogre": 8},
            villain_deck={"Forest": 8, "Llanowar Elves": 8},
        )
    ).to_rust()


def test_semantic_direct_hash_matches_composite_across_forks_and_commands(
    players: list[managym.PlayerConfig],
) -> None:
    root = managym.Env(seed=42)
    with pytest.raises(managym.AgentError, match="before reset"):
        root.semantic_viewer_state_hash(0)
    root.reset(players)
    before = root.search_witness_json()
    original = [root.semantic_viewer_state_hash(viewer) for viewer in (0, 1)]
    runtime = managym.SelectedBranchRuntime("direct-hash-test", True)
    branch = runtime.fork_exact(root, "child")
    for step in range(4):
        for viewer in (0, 1):
            composite = json.loads(branch.semantic_observation_json(viewer))
            assert (
                branch.semantic_viewer_state_hash(viewer)
                == composite["identity"]["viewer_state_hash"]
            )
            assert root.semantic_viewer_state_hash(viewer) == original[viewer]
        if step == 3:
            break
        runtime.apply_policy_choice(branch, "child", 0, "{}")
    assert branch.semantic_viewer_state_hash(0) != original[0]
    assert root.search_witness_json() == before
    with pytest.raises(managym.AgentError, match="viewer 2 is out of bounds"):
        branch.semantic_viewer_state_hash(2)


@pytest.mark.parametrize("audit", [False, True])
def test_selected_command_rejects_bad_identity_without_mutation(
    audit: bool, players: list[managym.PlayerConfig]
) -> None:
    root = managym.Env(seed=42)
    root.reset(players)
    runtime = managym.SelectedBranchRuntime("identity-test", audit)
    branch = runtime.fork_exact(root, "child")
    context = json.loads(branch.search_context_json(True))
    before = branch.search_witness_json()
    tape = runtime.snapshot_json()
    for field, value in (
        ("authority_hash", "incorrect"),
        ("legal_surface_hash", "incorrect"),
        ("expected_revision", context["revision"] + 1),
        ("prompt_id", context["prompt_id"] + 1),
        ("offer_id", 999999),
    ):
        with pytest.raises(managym.AgentError, match="precondition mismatch"):
            runtime.apply_policy_choice(branch, "child", 0, json.dumps({field: value}))
        assert branch.search_witness_json() == before
        assert runtime.snapshot_json() == tape

    # A random-stream change alters authority even at the same decision epoch.
    runtime.reseed_rollout(branch, 19)
    reseeded = branch.search_witness_json()
    with pytest.raises(managym.AgentError, match="authority precondition mismatch"):
        runtime.apply_policy_choice(
            branch,
            "child",
            0,
            json.dumps(
                {
                    "authority_hash": context["witness"]["authority_hash"],
                }
            ),
        )
    assert branch.search_witness_json() == reseeded
    assert runtime.snapshot_json() == tape

    current = json.loads(reseeded)
    *_, encoded = runtime.apply_policy_choice(
        branch,
        "child",
        0,
        json.dumps(
            {
                "authority_hash": current["authority_hash"],
                "legal_surface_hash": current["legal_surface_hash"],
            }
        ),
    )
    record = json.loads(encoded)
    assert record["source"]["authority_hash"] == current["authority_hash"]
    assert record["source"]["legal_surface_hash"] == current["legal_surface_hash"]
    assert record["native_receipt"]["native_apply_count"] == 1
    if audit:
        assert record["post_apply_witness"] == json.loads(branch.search_witness_json())
        assert json.loads(runtime.snapshot_json())["tapes"]["child"] == [record]
    else:
        assert record["post_apply_witness"] is None
    assert root.search_witness_json() == before
