"""Real Search and same-tape evidence cover Learn in both seat assignments."""

from copy import deepcopy
import random

import pytest

from etude.authored_match_receipt import (
    DeterministicServerOfferPolicy,
    UnsupportedAuthorityPrompt,
)
from etude.learn_lesson_evidence import attempt, verify_attempt
from etude.server import GameSession
from etude.villain import SearchVillain
from manabot.semantic.decision_contract import SemanticDecisionContract
import managym


@pytest.mark.parametrize("reverse", [False, True])
def test_configured_search_executes_learn_without_moving_source(reverse):
    keys = ["ur_lessons", "gw_allies"]
    if reverse:
        keys.reverse()
    engine = managym.Env(seed=0, skip_trivial=False)
    engine.reset(
        [managym.authored_deck_setup("ur-lessons-vs-gw-allies", key) for key in keys]
    )
    rng = random.Random(0)
    for _ in range(2000):
        contract = SemanticDecisionContract.from_env(engine)
        if any(offer["verb"] == "learn_take_lesson" for offer in contract.frame.offers):
            break
        contract.apply(engine, rng.choice(contract.frame.offers)["id"])
    else:
        pytest.fail("registered setup did not reach Learn")
    before = engine.search_witness_json()
    obs = engine.observation_for_player(contract.frame.actor)
    selected = SearchVillain(seed=0)(engine, obs)
    assert engine.search_witness_json() == before
    assert 0 <= selected < len(contract.frame.offers)
    branch = engine.clone_env()
    contract.apply(branch, contract.frame.offers[selected]["id"])
    assert branch.state_digest() != engine.state_digest()
    assert engine.search_witness_json() == before


def test_same_tape_parity_and_tampered_persisted_command():
    receipt = attempt(0, True)
    assert receipt["status"] == "passed", receipt.get("error")
    wrong_seed = deepcopy(receipt)
    wrong_seed["seed"] = 17
    with pytest.raises(RuntimeError, match="seed"):
        verify_attempt(wrong_seed)
    changed = deepcopy(receipt)
    changed["replay"]["decisions"][0]["command"]["offer_id"] = "forged"
    with pytest.raises((ValueError, RuntimeError)):
        verify_attempt(changed)


def test_total_command_cap_includes_opponent_turns():
    policy = DeterministicServerOfferPolicy(0, max_commands=1)
    with pytest.raises(UnsupportedAuthorityPrompt, match="cap"):
        policy.choose([], actor=1, revision=1, prompt_family="learn")


@pytest.mark.parametrize(
    "field,value",
    [("match_id", "different"), ("expected_revision", -1), ("offer_id", "forged")],
)
def test_learn_rejections_preserve_full_root(tmp_path, field, value):
    session = GameSession(trace_dir=tmp_path)
    frame = session.new_game({"demo": "learn"})["frame"]
    command = {
        "command_id": "invalid-learn",
        "match_id": frame["match_id"],
        "expected_revision": frame["revision"],
        "prompt_id": frame["prompt"]["id"],
        "offer_id": frame["offers"][0]["id"],
        "answers": [],
    }
    command[field] = value
    before = session.env.search_witness_json()
    result = session.hero_command(command)
    assert result["status"] == "rejected"
    assert session.env.search_witness_json() == before
    assert session.revision == frame["revision"]
