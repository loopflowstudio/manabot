from __future__ import annotations

import json

import pytest

from manabot.semantic.decision_contract import (
    SemanticContractError,
    SemanticDecisionContract,
)
import managym
from managym.decision import SEMANTIC_DECISION_VERSION


def _env() -> managym.Env:
    env = managym.Env(seed=23, skip_trivial=True)
    deck = {"Mountain": 10, "Gray Ogre": 20, "Lightning Bolt": 4}
    env.reset(
        [
            managym.PlayerConfig("hero", dict(deck)),
            managym.PlayerConfig("villain", dict(deck)),
        ]
    )
    return env


def test_contract_projects_a_revision_bound_frame() -> None:
    env = _env()
    contract = SemanticDecisionContract.from_env(env)

    frame = contract.frame
    authority_revision = json.loads(env.search_context_json())["revision"]
    assert frame.schema_version == SEMANTIC_DECISION_VERSION
    assert frame.revision == authority_revision
    assert len(frame.fingerprint) == 64
    assert any(offer["verb"] == "pass_priority" for offer in frame.offers)
    assert all(offer["actor"] == frame.actor for offer in frame.offers)


def test_contract_applies_a_command_and_the_receipt_advances_the_revision() -> None:
    env = _env()
    contract = SemanticDecisionContract.from_env(env)
    pass_offer = contract.frame.find_verb("pass_priority")

    transition = contract.apply(env, pass_offer["id"], command_id="cmd-1")
    receipt = transition.receipt

    assert receipt.schema_version == SEMANTIC_DECISION_VERSION
    assert receipt.command_id == "cmd-1"
    assert receipt.before_revision == contract.frame.revision
    assert receipt.after_revision > receipt.before_revision
    assert receipt.next_decision is not None
    assert transition.observation.revision == receipt.after_revision

    # The next frame projects at the after-revision and matches the receipt's
    # next_decision fingerprint (exact legal-action identity carries over).
    nxt = SemanticDecisionContract.from_env(env)
    assert nxt.frame.revision == receipt.after_revision
    assert nxt.frame.fingerprint == receipt.next_decision


def test_contract_fails_closed_on_stale_unknown_and_illegal_without_mutation() -> None:
    env = _env()
    contract = SemanticDecisionContract.from_env(env)
    pass_offer = contract.frame.find_verb("pass_priority")
    bound_revision = contract.frame.revision

    # Advance past the bound frame; the frozen contract now builds stale
    # commands because its frame is still at bound_revision.
    contract.apply(env, pass_offer["id"])

    with pytest.raises(SemanticContractError, match="stale"):
        contract.apply(env, pass_offer["id"], command_id="stale")

    current = SemanticDecisionContract.from_env(env)
    assert current.frame.revision > bound_revision

    # Unknown offer at the current revision: bypass the local offer check by
    # applying a raw command, exercising the engine's fail-closed path.
    unknown = {
        "command_id": "unknown",
        "expected_revision": current.frame.revision,
        "offer_id": 2**31 - 1,
        "answers": [],
    }
    with pytest.raises(SemanticContractError, match="absent|unknown"):
        current.apply_command(env, unknown)

    # Illegal answers on a choiceless action-aligned offer.
    illegal = {
        "command_id": "illegal",
        "expected_revision": current.frame.revision,
        "offer_id": current.frame.offers[0]["id"],
        "answers": [{"kind": "candidates", "role": 1, "candidates": [0]}],
    }
    with pytest.raises(SemanticContractError, match="illegal|unexpected"):
        current.apply_command(env, illegal)

    # No mutation on any rejected command: the revision is unchanged.
    assert (
        SemanticDecisionContract.from_env(env).frame.revision == current.frame.revision
    )


def test_contract_rejects_a_locally_fabricated_offer_id() -> None:
    env = _env()
    contract = SemanticDecisionContract.from_env(env)
    with pytest.raises(SemanticContractError, match="absent"):
        contract.command(2**31 - 1)
