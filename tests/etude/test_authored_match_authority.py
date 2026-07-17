"""Terminal release-stack authority evidence for the selected authored match."""

import json

from etude.authored_match_receipt import (
    DEFAULT_RECEIPT_PATH,
    SEMANTIC_IR_PATH,
    generate_authored_match_receipt,
)


def test_release_stack_authored_match_receipt_is_terminal_and_authoritative():
    checked = json.loads(DEFAULT_RECEIPT_PATH.read_text(encoding="utf-8"))
    generated = generate_authored_match_receipt()
    assert generated == checked

    summary = checked["summary"]
    decisions = checked["decisions"]
    assert summary["deliberate_commands"] == len(decisions) > 0
    assert summary["automatic_rules_actions"] == 0
    assert summary["fallback_counters"] == {
        "legacy_fixed_action": 0,
        "card_name_dispatch": 0,
        "candidate_cap": 0,
        "client_legality": 0,
    }
    assert [row["ordinal"] for row in decisions] == list(range(len(decisions)))
    assert {row["actor"] for row in decisions} == {0, 1}
    assert all(
        row["command"]["expected_revision"] == row["from_revision"] for row in decisions
    )
    assert all(row["command"]["prompt_id"] == row["prompt_id"] for row in decisions)
    assert all(row["command"]["offer_id"] == row["offer"]["id"] for row in decisions)
    assert all(row["offer_count"] == row["legal_action_count"] > 0 for row in decisions)
    assert all(
        row["command_receipt"]["command_id"] == row["command"]["command_id"]
        for row in decisions
    )
    assert all(
        row["command_receipt"]["resulting_revision"] == row["to_revision"]
        for row in decisions
    )
    assert all(
        row["state"]["after"] == row["command_receipt"]["resulting_state_digest"]
        for row in decisions
    )

    semantic_ir = json.loads(SEMANTIC_IR_PATH.read_text(encoding="utf-8"))
    admitted = {program["semantic_key"] for program in semantic_ir["programs"]}
    encountered = {
        program["semantic_key"] for program in checked["encountered_typed_programs"]
    }
    referenced = {
        semantic_key
        for decision in decisions
        for semantic_key in decision["encountered_programs"]
    }
    assert encountered == referenced
    assert encountered
    assert encountered <= admitted

    terminal = checked["terminal_witness"]
    assert terminal["terminal"] is True
    assert terminal["winner"] in (0, 1)
    assert terminal["revision"] == len(decisions)
    assert terminal["state_digest"] == decisions[-1]["state"]["after"]


def test_compatibility_action_is_counted_as_legacy_fallback(tmp_path):
    from etude.server import GameSession

    session = GameSession(
        tmp_path,
        villain_offer_policy=lambda context: int(context.offers[0]["id"]),
    )
    message = session.new_game(
        {
            "hero_deck": "ur_lessons",
            "villain_deck": "gw_allies",
            "villain_type": "passive",
            "seed": 0,
            "auto_pass": False,
        }
    )
    session.hero_action(message["actions"][0]["index"])
    assert session.authority_fallback_counters["legacy_fixed_action"] == 1
    assert session.authority_fallback_counters["client_legality"] == 1
