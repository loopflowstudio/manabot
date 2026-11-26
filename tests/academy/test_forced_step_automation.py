"""Native skipping keeps forced decisions away from people and policies."""

from pathlib import Path
from typing import Any

import pytest

import managym
from managym.academy import server
from managym.academy.server import GameSession
from managym.bots import BotDecision, BotPolicy
from managym.experience import Command, ExperienceFrame


def _auto_pass_turns(events: list[dict[str, Any]], human_seat: int) -> list[int | None]:
    """Turn number of each automated priority pass on the person's track."""
    turn: int | None = None
    turns: list[int | None] = []
    for event in events:
        kind = event["kind"]
        if kind["kind"] == "turn_started":
            turn = int(kind["turn_number"])
        elif (
            kind["kind"] == "passed"
            and int(kind["player"]) == human_seat
            and event["caused_by"] is None
        ):
            turns.append(turn)
    return turns


@pytest.mark.parametrize("auto_pass", [True, False])
def test_forced_decisions_never_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, auto_pass: bool
) -> None:
    choose = server.choose_bot
    policy_calls: list[int] = []

    def choose_meaningful(
        policy: BotPolicy, env: managym.Env, obs: managym.Observation
    ) -> BotDecision:
        count = len(obs.action_space.actions)
        assert count > 1, "Forced actions must not invoke the opponent policy"
        policy_calls.append(count)
        return choose(policy, env, obs)

    monkeypatch.setattr(server, "choose_bot", choose_meaningful)
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
    recovery = session.current_recovery("reconnect")
    frame = ExperienceFrame.model_validate(recovery["frame"])
    events: list[dict[str, Any]] = list(recovery["presentation_tail"])
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
        events.extend(result["update"]["presentation"])
        frame = ExperienceFrame.model_validate(result["update"]["frame"])

    assert session.env is not None
    assert session.env.skip_trivial_count() > 0
    assert session.trace is not None
    assert all(len(event.actions) > 1 for event in session.trace.events)
    automatic = [row for row in session.authority_transitions if row.automatic]
    assert all(row.action_type == "PRIORITY_PASS_PRIORITY" for row in automatic)
    assert len(_auto_pass_turns(events, session.human_seat)) == len(automatic)
    assert policy_calls


def test_priority_automation_is_logged(tmp_path: Path) -> None:
    """Passes outside configured stops are real choices, told as automation."""
    session = GameSession(tmp_path, capture_authority_evidence=True)
    session.new_game(
        {
            "hero_deck": {"Mountain": 32, "Lightning Bolt": 8},
            "villain_deck": {"Forest": 40},
            "villain_type": "passive",
            "seed": 1,
        }
    )
    recovery = session.current_recovery("reconnect")
    frame = ExperienceFrame.model_validate(recovery["frame"])
    events: list[dict[str, Any]] = list(recovery["presentation_tail"])
    for index, label in enumerate(("Keep hand", "Play Mountain", "Pass priority")):
        assert frame.prompt is not None
        offer = next(row for row in frame.offers if row.label == label)
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
        events.extend(result["update"]["presentation"])
        frame = ExperienceFrame.model_validate(result["update"]["frame"])

    automatic = [row for row in session.authority_transitions if row.automatic]
    assert automatic
    auto_turns = _auto_pass_turns(events, session.human_seat)
    # Each pass sits under the turn its action was taken in, never a later one.
    assert all(
        turn is not None and 1 <= turn <= frame.projection.turn.turn_number
        for turn in auto_turns
    )
    assert len(auto_turns) == len(automatic)
