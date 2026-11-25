"""The native decision surface survives Commands, recovery, SQLite and Study."""

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError
import pytest

from manabot.verify.text_observation import priority_root
import managym
from managym.academy.replay_index import (
    CanonicalReplayV1,
    DecisionAddressV2,
    ReplayDecision,
    project_replay,
    restore_decision,
)
from managym.academy.server import GameSession, viewer_view
from managym.academy.trace import GameConfig, Trace, TraceEvent
from managym.bots import BeliefSummary, BotDecision, PolicyFactor
from managym.experience import Command, ExperienceFrame, InteractionOffer
from managym.experience_projection import serialize_observation


def _frame(session: GameSession) -> ExperienceFrame:
    return ExperienceFrame.model_validate(
        session.current_recovery("reconnect")["frame"]
    )


def _command(frame: ExperienceFrame, offer: InteractionOffer, identity: str) -> Command:
    assert frame.prompt is not None
    return Command(
        command_id=identity,
        match_id=frame.match_id,
        expected_revision=frame.revision,
        prompt_id=frame.prompt.id,
        offer_id=offer.id,
        answers=[],
    )


def _choose(
    session: GameSession, frame: ExperienceFrame, offer: InteractionOffer
) -> ExperienceFrame:
    command = _command(frame, offer, f"surface.{frame.revision}")
    result = session.hero_command(command.model_dump(mode="json"))
    assert result["status"] == "accepted"
    following = ExperienceFrame.model_validate(result["update"]["frame"])
    digest = session.env.state_digest()
    assert _frame(session) == following
    assert [entry.id for entry in following.projection.stack_objects] == [
        int(entry.stack_object_id)
        for entry in session.env.observation_for_player(0).stack_objects
    ]
    assert session.env.state_digest() == digest
    duplicate = session.hero_command(command.model_dump(mode="json"))
    assert duplicate["status"] == "duplicate"
    assert session.env.state_digest() == digest
    stale = command.model_copy(update={"command_id": f"stale.{frame.revision}"})
    rejected = session.hero_command(stale.model_dump(mode="json"))
    assert rejected["status"] == "rejected"
    assert rejected["rejection"]["code"] == "stale_revision"
    assert _frame(session) == following
    assert session.env.state_digest() == digest
    return following


def _label(frame: ExperienceFrame, label: str) -> InteractionOffer:
    return next(offer for offer in frame.offers if offer.label == label)


def _study_every_boundary(session: GameSession, revisions: list[int]) -> None:
    replay = session.canonical_replay()
    live_digest = session.env.state_digest()
    projection = project_replay(replay, 0)
    assert session.trace_id is not None
    for revision in revisions:
        row = next(row for row in projection.decisions if row.revision == revision)
        address = DecisionAddressV2.from_decision(projection, row).serialize()
        restored = restore_decision(replay, address, 0)
        assert restored.frame == row.frame
        result = session.retry_study(
            trace_id=session.trace_id,
            replay=replay,
            raw_address=address,
            raw_command=row.command.model_copy(
                update={"command_id": f"study.surface.{revision}"}
            ).model_dump(mode="json"),
        )
        assert result["retry"]["projection"]["opponent"]["hand"] == []
        following = next(
            (
                candidate
                for candidate in projection.decisions
                if candidate.revision == revision + 1
            ),
            None,
        )
        if following is not None:
            assert result["retry"][
                "projection"
            ] == following.frame.projection.model_dump(mode="json")
        returned = session.return_from_study(result["attempt_id"])
        assert returned["frame"] == row.frame.model_dump(mode="json")
        assert returned["command"] == row.command.model_dump(mode="json")
        assert returned["address"] == address
        assert session.env.state_digest() == live_digest
    # Persisted decisions carry the same optional details, not reconstructed
    # context from the current engine. The source remains authority-private.
    session._persist_attempt()
    saved = session.attempt_store.load(session.trace_id)
    assert saved is not None
    retained = CanonicalReplayV1.model_validate(saved["canonical_replay"])
    assert retained.decisions == replay.decisions


def test_precision_forced_treasure_payment(tmp_path: Path) -> None:
    session = GameSession(tmp_path)
    try:
        initial = session.new_game({"demo": "precision"})
        frame = ExperienceFrame.model_validate(initial["frame"])
        permanents = {
            permanent.name: permanent
            for permanent in frame.projection.agent.battlefield
        }
        assert permanents["Callous Inspector"].types.is_creature
        assert permanents["Swamp"].types.is_land
        assert permanents["Treasure"].types.is_artifact
        assert permanents["Treasure"].is_token
        assert (
            session.attempt_store.metadata(session.trace_id)["origin"]
            == "automated_validation"
        )
        assert initial["recovery"]["frame"] == initial["frame"]
        revisions: list[int] = []
        frame = _choose(session, frame, _label(frame, "Cast Deadly Precision"))
        assert frame.projection.decision_context.stage == "targets"
        targets = [
            offer
            for offer in frame.offers
            if offer.action_type == "CHOOSE_TARGET"
            and offer.subjects[0].object.label == "Callous Inspector"
        ]
        assert len(targets) >= 2
        assert len({offer.subjects[0].object.entity_id for offer in targets}) == len(
            targets
        )
        victim = targets[0].subjects[0].object.entity_id
        frame = _choose(session, frame, targets[0])
        context = frame.projection.decision_context
        assert context.stage == "additional_cost"
        assert context.targets[0].object.entity_id == victim
        sacrifice = next(
            offer
            for offer in frame.offers
            if offer.action_type == "SELECT_CARD"
            and offer.subjects[0].object.entity_id == victim
        )
        assert (
            sacrifice.subjects[0].object.display_id
            == context.targets[0].object.display_id
        )
        revisions.append(frame.revision)
        frame = _choose(session, frame, sacrifice)
        # Treasure is the only mana source left, so its activation, {B} and the
        # final payment are forced; native skipping commits them in order.
        context = frame.projection.decision_context
        assert context.stage not in {"payment", "mana_color"}
        assert not any(
            card.name == "Treasure" for card in frame.projection.agent.battlefield
        )
        facts = context.recent_facts
        treasure = next(
            i for i, fact in enumerate(facts) if "sacrificed Treasure" in fact
        )
        assert any(
            "Deadly Precision was cast" in fact for fact in facts[treasure + 1 :]
        )
        assert all(len(row.frame.offers) > 1 for row in session.canonical_decisions)
        # The other viewer receives public pools, no pending private context,
        # private hand, or inspection candidates from the acting player.
        other = viewer_view(
            session.env.observation_for_player(1), 1, winner=None, include_details=True
        )
        assert other["opponent"]["hand"] == []
        assert not other["opponent"].get("inspected")
        public_context = json.loads(session.env.decision_context_json(1))
        assert public_context["actor"] == 0
        assert public_context["source"] is None
        assert public_context["targets"] == []
        assert public_context["reserved_sacrifice"] is None
        _study_every_boundary(session, revisions)
    finally:
        session.close("test")


def test_precision_four_mana_branch(tmp_path: Path) -> None:
    session = GameSession(tmp_path)
    try:
        frame = ExperienceFrame.model_validate(
            session.new_game({"demo": "precision-mana"})["frame"]
        )
        frame = _choose(session, frame, _label(frame, "Cast Deadly Precision"))
        frame = _choose(
            session,
            frame,
            next(
                offer for offer in frame.offers if offer.action_type == "CHOOSE_TARGET"
            ),
        )
        before = len(frame.projection.agent.battlefield)
        revision = frame.revision
        alternative = next(
            offer for offer in frame.offers if offer.action_type == "PAY_COST"
        )
        assert "{4}" in alternative.label
        frame = _choose(session, frame, alternative)
        assert len(frame.projection.agent.battlefield) == before
        assert all(
            card.tapped
            for card in frame.projection.agent.battlefield
            if card.name == "Swamp"
        )
        _study_every_boundary(session, [revision])
    finally:
        session.close("test")


def test_imported_replay_rejects_private_inspection(tmp_path: Path) -> None:
    session = GameSession(tmp_path)
    try:
        frame = ExperienceFrame.model_validate(
            session.new_game({"demo": "precision"})["frame"]
        )
        _choose(session, frame, _label(frame, "Cast Deadly Precision"))
        row = session.canonical_replay().decisions[-1]
        payload: dict[str, Any] = deepcopy(row.model_dump(mode="json"))
        # Even a structurally valid card is forbidden in this viewer's other
        # player's private inspection area at every embedded replay reader.
        payload["frame"]["projection"]["opponent"]["inspected"] = [
            payload["frame"]["projection"]["agent"]["hand"][0]
        ]
        with pytest.raises(ValidationError, match="opponent-private"):
            ReplayDecision.model_validate(payload)
    finally:
        session.close("test")


def test_trace_record_preserves_nested_policy_evidence() -> None:
    trace = Trace(
        config=GameConfig({"Island": 40}, {"Plains": 40}, "random", seed=0),
        events=[
            TraceEvent(
                actor="hero",
                observation={"agent": {"hand": []}},
                actions=[],
                action=0,
                action_description="Pass",
                reward=0.0,
                bot_decision=BotDecision(
                    0,
                    factors=[PolicyFactor(0, [1.0], 0.5)],
                    belief=BeliefSummary("model", "digest", 1.0, 1),
                ),
            )
        ],
        final_observation={},
        winner=None,
        end_reason="active",
        timestamp="fixed",
        canonical_replay={"decisions": []},
    )
    expected = asdict(trace)
    expected.pop("recording_import")
    expected["config"].pop("mana_payment")
    expected["config"].pop("mana_floats")
    expected["config"].pop("hero_deck_pool")
    expected["config"].pop("villain_deck_pool")
    expected["mana_payment"] = ["manual", "manual"]
    expected["mana_floats"] = ["off", "off"]
    assert trace.to_record() == expected
    assert json.loads(json.dumps(trace.to_record())) == json.loads(json.dumps(expected))


def test_permanent_details_in_recovery_after_a_lesson_resolves(tmp_path: Path) -> None:
    env = priority_root(
        ("Firebending Lesson",),
        ("Platypus-Bear", "Mountain", "Fire Nation Cadets"),
        opponent=("Mai, Jaded Edge",),
    )
    session = GameSession(tmp_path)
    try:
        session.env = env
        session.obs = env.observation_for_player(0)
        initial = _frame(session).projection
        bear, _, cadets = initial.agent.battlefield
        assert bear.keyword_grants == {}  # Its Defender comes from its own face.
        assert not bear.attacks_despite_defender
        assert not cadets.keyword_abilities
        assert initial.opponent.battlefield[0].keyword_abilities[0].label == "Prowess"

        cast = next(
            i
            for i, action in enumerate(session.obs.action_space.actions)
            if action.action_type == managym.ActionEnum.PRIORITY_CAST_SPELL
        )
        targets = env.step(cast)[0].action_space.actions
        target = next(
            i for i, action in enumerate(targets) if action.focus == [bear.id]
        )
        session.obs = env.step(target)[0]
        for _ in range(2):
            (action,) = session.obs.action_space.actions
            assert action.action_type == managym.ActionEnum.PRIORITY_PASS_PRIORITY
            session.obs = env.step(0)[0]

        # Same public permanent ids and abilities from the other seat on reconnect.
        session.human_seat = 1
        board = _frame(session).projection.opponent.battlefield
        assert board[0].id == bear.id and board[0].attacks_despite_defender
        assert board[2].id == cadets.id
        assert [a.model_dump() for a in board[2].keyword_abilities] == [
            {"label": "Firebending 2"}
        ]

        # Bytes captured with the pre-change extension, at this resolved position.
        historical = serialize_observation(env.observation_for_player(0))
        blob = json.dumps(historical, separators=(",", ":"), sort_keys=True).encode()
        assert (
            sha256(blob).hexdigest()
            == "0c4bd6fa85670fb237252a5c48f4ec5cbb1b84654cbb07b9bdae7f12ba5ba7ae"
        )

        details = env.permanent_details_json(0)
        hidden = env.observation_for_player(1).toJSON()
        env.determinize(seed=71, perspective=0)
        assert env.observation_for_player(1).toJSON() != hidden
        assert env.permanent_details_json(0) == details
    finally:
        session.close("session_closed")
