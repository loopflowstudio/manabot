"""Native skipping keeps forced decisions away from people and policies."""

from pathlib import Path

from managym.experience import Command, ExperienceFrame
import pytest

from academy import villain
from academy.bot_decision import BotDecision
from academy.server import GameSession
import managym


@pytest.mark.parametrize("auto_pass", [True, False])
def test_forced_decisions_never_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, auto_pass: bool
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
    session.new_game(
        {
            "hero_deck": {"Mountain": 14, "Gray Ogre": 8},
            "villain_deck": {"Island": 20},
            "villain_type": "passive",
            "seed": 3,
            "auto_pass": auto_pass,
        }
    )
    frame = ExperienceFrame.model_validate(
        session.current_recovery("reconnect")["frame"]
    )
    log: list[str] = []
    for index in range(30):
        if frame.prompt is None:
            break
        assert len(frame.offers) > 1
        offer = next(
            (
                row
                for row in frame.offers
                if row.action_type in {"PRIORITY_PLAY_LAND", "PRIORITY_CAST_SPELL"}
            ),
            frame.offers[0],
        )
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
        log.extend(result["update"].get("log", []))
        frame = ExperienceFrame.model_validate(result["update"]["frame"])

    assert session.env is not None
    assert session.env.skip_trivial_count() > 0
    assert session.trace is not None
    assert all(len(event.actions) > 1 for event in session.trace.events)
    automatic = [row for row in session.authority_transitions if row.automatic]
    assert all(row.action_type == "PRIORITY_PASS_PRIORITY" for row in automatic)
    assert log.count("Auto: Pass priority") == len(automatic)
    assert policy_calls


def test_priority_automation_is_logged(tmp_path: Path) -> None:
    """Passes outside configured stops are real choices, logged as automation."""
    session = GameSession(tmp_path, capture_authority_evidence=True)
    session.new_game(
        {
            "hero_deck": "elemental_ur",
            "villain_deck": "elemental_gw",
            "villain_type": "passive",
            "seed": 0,
        }
    )
    frame = ExperienceFrame.model_validate(
        session.current_recovery("reconnect")["frame"]
    )
    log: list[str] = []
    for index in range(6):
        assert frame.prompt is not None
        offer = next(
            (row for row in frame.offers if row.action_type == "PRIORITY_PLAY_LAND"),
            next(
                row
                for row in frame.offers
                if row.action_type == "PRIORITY_PASS_PRIORITY"
            ),
        )
        command = Command(
            command_id=f"automation-log.{index}",
            match_id=frame.match_id,
            expected_revision=frame.revision,
            prompt_id=frame.prompt.id,
            offer_id=offer.id,
            answers=[],
        )
        result = session.hero_command(command.model_dump(mode="json"))
        assert result["status"] == "accepted"
        log.extend(result["update"].get("log", []))
        frame = ExperienceFrame.model_validate(result["update"]["frame"])

    automatic = [row for row in session.authority_transitions if row.automatic]
    assert automatic
    assert log.count("Auto: Pass priority") == len(automatic)
