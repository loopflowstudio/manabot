"""Forced play stays recorded and stops at the next actual decision."""

from pathlib import Path

from managym.experience import Command, ExperienceFrame
import pytest

from academy import villain
from academy.bot_decision import BotDecision
from academy.server import GameSession
import managym


def test_forced_actions_keep_commands_logs_and_real_choices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    choose = villain.choose_bot
    policy_calls: list[int] = []

    def choose_meaningful(
        policy: villain.VillainPolicy, env: managym.Env, obs: managym.Observation
    ) -> BotDecision:
        count = len(obs.action_space.actions)
        assert count > 1, "Forced actions must not invoke the opponent policy"
        policy_calls.append(count)
        return choose(policy, env, obs)

    monkeypatch.setattr(villain, "choose_bot", choose_meaningful)
    session = GameSession(tmp_path, capture_authority_evidence=True)
    opening = session.new_game(
        {
            "hero_deck": {"Mountain": 14, "Gray Ogre": 8},
            "villain_deck": {"Island": 20},
            "villain_type": "passive",
            "seed": 3,
        }
    )
    logs = list(opening.get("log", []))
    frame = ExperienceFrame.model_validate(
        session.current_recovery("reconnect")["frame"]
    )
    for index in range(30):
        assert len(frame.offers) > 1
        offer = next(
            (
                row
                for row in frame.offers
                if row.action_type in {"PRIORITY_PLAY_LAND", "PRIORITY_CAST_SPELL"}
            ),
            frame.offers[0],
        )
        assert frame.prompt is not None
        command = Command(
            command_id=f"forced-proof.{index}",
            match_id=frame.match_id,
            expected_revision=frame.revision,
            prompt_id=frame.prompt.id,
            offer_id=offer.id,
            answers=[],
        )
        result = session.hero_command(command.model_dump(mode="json"))
        assert result["status"] == "accepted"
        frame = ExperienceFrame.model_validate(result["update"]["frame"])
        logs.extend(result["update"].get("log", []))
        assert (
            session.hero_command(command.model_dump(mode="json"))["status"]
            == "duplicate"
        )
        recovered = ExperienceFrame.model_validate(
            session.current_recovery("reconnect")["frame"]
        )
        assert recovered.prompt == frame.prompt
        assert recovered.offers == frame.offers
        if any(
            row.automatic and row.action_space != "PRIORITY"
            for row in session.authority_transitions
        ):
            break
    else:
        raise AssertionError("Expected a forced non-priority action in ordinary play")

    automatic = [row for row in session.authority_transitions if row.automatic]
    assert any(row.action_space != "PRIORITY" for row in automatic)
    assert all(
        row.legal_action_count == 1 or row.action_type == "PRIORITY_PASS_PRIORITY"
        for row in automatic
    )
    assert all(
        row.source == "auto" and row.command and row.semantic_receipt
        for row in automatic
    )
    assert all(
        row.semantic_receipt.command_id == row.command["command_id"]
        for row in automatic
    )
    assert session.trace is not None
    forced = [
        event
        for event in session.trace.events
        if event.auto and len(event.actions) == 1
    ]
    for event in forced:
        prefix = "Auto: Opponent: " if event.actor == "villain" else "Auto: "
        assert f"{prefix}{event.action_description}" in logs
    assert any(event.actor == "villain" for event in forced)
    assert policy_calls
    auto_ids = {row.command["command_id"] for row in automatic}
    assert not any(row.command_id in auto_ids for row in session.canonical_decisions)


def test_disabling_automation_retains_forced_prompt(tmp_path: Path) -> None:
    session = GameSession(tmp_path)
    opening = session.new_game(
        {"villain_type": "passive", "seed": 3, "auto_pass": False}
    )
    assert len(opening["actions"]) == 1
    assert session.trace is not None
    assert not session.trace.events
    before = session.revision
    result = session.set_stops(None, None, True)
    assert session.revision > before + 1
    assert len(result["actions"]) > 1
    assert any(event.auto for event in session.trace.events)
