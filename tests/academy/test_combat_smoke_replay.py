"""A complete vanilla smoke-policy game, distinct from constructed combat proofs."""

from dataclasses import asdict
import json
from pathlib import Path

from conformance.combat_planning import declaration_command
from managym.academy.server import DecisionContext, GameSession
from managym.decision import Command as SemanticCommand
from managym.experience import ExperienceFrame


def _offer(context: DecisionContext) -> int:
    # This smoke policy plays creatures to exercise combat; it measures no strength.
    for kind in ("PRIORITY_PLAY_LAND", "PRIORITY_CAST_SPELL"):
        for action, offer in zip(context.actions, context.offers, strict=True):
            if action["type"] == kind:
                return int(offer["id"])
    for action, offer in zip(context.actions, context.offers, strict=True):
        if action.get("declared") is True:
            return int(offer["id"])
    for action, offer in zip(context.actions, context.offers, strict=True):
        if action["type"] == "PRIORITY_PASS_PRIORITY":
            return int(offer["id"])
    return int(context.offers[0]["id"])


def test_complete_smoke_game_replays_every_ordinary_child(tmp_path: Path) -> None:
    game = GameSession(
        tmp_path, villain_offer_policy=_offer, capture_authority_evidence=True
    )
    deck = {"Mountain": 24, "Gray Ogre": 16}
    game.new_game(
        {
            "hero_deck": deck,
            "villain_deck": deck,
            "villain_type": "passive",
            "seed": 364,
            "auto_pass": False,
        }
    )
    assert game.env is not None
    root = game.env.clone_env()
    multi: set[str] = set()
    for ordinal in range(400):
        assert game.obs is not None
        if game.obs.game_over:
            break
        context = game._publish_current_prompt()
        assert context is not None
        frame = ExperienceFrame.model_validate(context.frame)
        if frame.declaration is not None:
            stage = frame.declaration.stage
            count = sum(not row.fixed for row in frame.declaration.subjects)
            if count >= 2:
                multi.add(stage)
            command = declaration_command(
                frame,
                f"smoke.declare.{ordinal}",
                select=stage == "attackers" and count >= 2,
            )
            result = game.hero_declaration(command.model_dump(mode="json"))
        else:
            assert frame.prompt is not None
            result = game.hero_command(
                {
                    "command_id": f"smoke.choose.{ordinal}",
                    "match_id": frame.match_id,
                    "expected_revision": frame.revision,
                    "prompt_id": frame.prompt.id,
                    "offer_id": _offer(context),
                    "answers": [],
                }
            )
        assert result["status"] == "accepted"
    assert game.obs is not None and game.obs.game_over
    assert multi == {"attackers", "blockers"}
    for row in game.authority_transitions:
        assert row.command is not None and row.semantic_receipt is not None
        assert root.state_digest() == row.state_before
        command = SemanticCommand(
            str(row.command["command_id"]),
            row.semantic_receipt.before_revision,
            int(row.command["offer_id"]),
        )
        root.step_semantic_command(command.to_json())
        assert root.state_digest() == row.state_after
    assert root.search_witness_json() == game.env.search_witness_json()
    record = game.canonical_replay()
    record_path = tmp_path / "complete-combat-smoke.json"
    record_path.write_text(record.model_dump_json())
    assert record.command_receipts
    assert json.loads(record_path.read_text())["decisions"]

    (tmp_path / "complete-combat-smoke-authority.json").write_text(
        json.dumps(
            {
                "kind": "vanilla-smoke-policy-integration",
                "seed": 364,
                "deck": deck,
                "transitions": [asdict(row) for row in game.authority_transitions],
                "terminal_witness": json.loads(game.env.search_witness_json()),
            }
        )
    )
