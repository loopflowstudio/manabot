"""A submitted TLA deck reaches delegated play and retained-session recovery."""

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from academy import server, trace as trace_store
from manabot.env.env import Env
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import MatchConfig, RewardConfig
from manabot.managym.experience import ExperienceFrame, InteractionOffer
import managym

HERO = "56 Island\n4 Secret of Bloodbending"
VILLAIN = "56 Island\n4 Accumulate Wisdom"


def _request() -> dict[str, object]:
    return {
        "villain_type": "passive",
        "seed": 3,
        "auto_pass": True,
        "deck_import": {"hero": {"source": HERO}, "villain": {"source": VILLAIN}},
    }


def _command(
    frame: ExperienceFrame, offer: InteractionOffer, index: int
) -> dict[str, object]:
    assert frame.prompt is not None
    return {
        "command_id": f"bloodbending.{index}",
        "match_id": frame.match_id,
        "expected_revision": frame.revision,
        "prompt_id": frame.prompt.id,
        "offer_id": offer.id,
        "answers": [],
    }


def _next_offer(frame: ExperienceFrame) -> InteractionOffer:
    """Build fourteen Islands, then pay for whole-turn control; pass afterward."""
    offers = frame.offers
    own_lands = len(frame.projection.agent.battlefield)
    controls = frame.projection.player_controls
    for offer in offers:
        if offer.action_type == "PRIORITY_PLAY_LAND" and not controls:
            return offer
        if (
            offer.action_type == "PRIORITY_CAST_SPELL"
            and own_lands >= 14
            and not controls
        ):
            return offer
        if offer.action_type in {"PAY_COST", "CHOOSE_TARGET"}:
            return offer
    return next(
        (o for o in offers if o.action_type == "PRIORITY_PASS_PRIORITY"), offers[0]
    )


def test_bloodbending_pending_discard_recovers_and_commits_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/play") as socket:
            socket.send_json({"type": "new_game", "config": _request()})
            initial = socket.receive_json()
            frame = initial["frame"]
            for index in range(300):
                controls = frame["projection"].get("player_controls", [])
                if frame["action_space"] == "DISCARD" and any(
                    c["effective"] for c in controls
                ):
                    break
                parsed = ExperienceFrame.model_validate(frame)
                command = _command(parsed, _next_offer(parsed), index)
                socket.send_json({"type": "command", "command": command})
                result = socket.receive_json()
                assert result["status"] == "accepted", result
                frame = result["update"]["frame"]
            else:
                raise AssertionError("controlled cleanup discard was not reached")
            parsed = ExperienceFrame.model_validate(frame)
            assert parsed.prompt is not None and parsed.prompt.actor == 0
            assert parsed.projection.decision_context is not None
            assert (
                "chooses for Seat 2" in parsed.projection.decision_context.instruction
            )
            inspected = parsed.projection.agent.inspected
            assert inspected and len(inspected) == 8
            assert parsed.projection.opponent.hand == []
            before_hash = frame["frame_hash"]
            command = _command(parsed, parsed.offers[0], 301)
        with client.websocket_connect("/ws/play") as resumed:
            resumed.send_json(
                {
                    "type": "resume",
                    "session_id": initial["session_id"],
                    "resume_token": initial["resume_token"],
                }
            )
            recovery = resumed.receive_json()["recovery"]
            assert recovery["frame"]["frame_hash"] == before_hash
            assert recovery["frame"]["offers"] == frame["offers"]
            assert (
                recovery["frame"]["projection"]["agent"]["inspected"]
                == frame["projection"]["agent"]["inspected"]
            )
            resumed.send_json({"type": "command", "command": command})
            accepted = resumed.receive_json()
            assert accepted["status"] == "accepted"
            next_frame = accepted["update"]["frame"]
            assert not next_frame["projection"].get("player_controls")
            assert not next_frame["projection"]["agent"].get("inspected")
            resumed.send_json({"type": "command", "command": command})
            duplicate = resumed.receive_json()
            assert duplicate["status"] == "duplicate"
            stale = {**command, "command_id": "bloodbending.stale"}
            resumed.send_json({"type": "command", "command": stale})
            rejected = resumed.receive_json()
            assert rejected["status"] == "rejected"
            assert (
                rejected["recovery"]["frame"]["frame_hash"] == next_frame["frame_hash"]
            )
    session = next(iter(server.SESSION_REGISTRY.values())).game
    assert session.trace is not None
    assert session.env is not None
    assert session.env.skip_trivial_count() > 0
    assert any(event.auto for event in session.trace.events)
    session.close(end_reason="test_complete")
    server.SESSION_REGISTRY.clear()


def test_submitted_tla_decks_play_through_the_training_environment() -> None:
    imports = managym.admit_decklists(
        [("hero", managym.Decklist(HERO)), ("villain", managym.Decklist(VILLAIN))]
    )
    config = MatchConfig.from_import(*imports)
    env = Env(Match(config), ObservationSpace(), Reward(RewardConfig()), seed=3)
    env.reset()
    for _ in range(240):
        raw = env._last_obs
        assert raw is not None
        if any(frame.kind == 14 for frame in raw.continuation):
            assert raw.agent.player_index == 0
            assert env._engine.current_agent_index() == 0
            assert raw.opponent_cards
            break
        actions = raw.action_space.actions
        lands = len(raw.agent_permanents)
        preferred = next(
            (
                i
                for i, a in enumerate(actions)
                if a.action_type == managym.ActionEnum.PRIORITY_PLAY_LAND
            ),
            None,
        )
        if preferred is None and raw.agent.player_index == 0 and lands >= 14:
            preferred = next(
                (
                    i
                    for i, a in enumerate(actions)
                    if a.action_type == managym.ActionEnum.PRIORITY_CAST_SPELL
                ),
                None,
            )
        if preferred is None:
            preferred = next(
                (
                    i
                    for i, a in enumerate(actions)
                    if a.action_type
                    in (managym.ActionEnum.PAY_COST, managym.ActionEnum.CHOOSE_TARGET)
                ),
                None,
            )
        if preferred is None:
            preferred = next(
                (
                    i
                    for i, a in enumerate(actions)
                    if a.action_type == managym.ActionEnum.PRIORITY_PASS_PRIORITY
                ),
                0,
            )
        _, _, done, truncated, _ = env.step(preferred)
        assert not done and not truncated
    else:
        raise AssertionError("training consumer never reached delegated control")
    env.close()
