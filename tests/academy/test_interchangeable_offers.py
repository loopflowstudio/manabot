"""End-to-end Academy contract for folded offers over indistinguishable objects."""

from pathlib import Path
from typing import Any

from managym.academy.server import GameSession
from managym.experience import ExperienceFrame, InteractionOffer, ObjectSubject


def _setup(
    tmp_path: Path,
    lands: list[str],
    *,
    hand: list[str],
    extra_deck: dict[str, int] | None = None,
) -> tuple[GameSession, ExperienceFrame]:
    hero_deck: dict[str, int] = {
        "Kyoshi Village": 4,
        "Plains": 24,
        "Youthful Knight": 8,
        "Pacifism": 4,
    }
    if extra_deck:
        hero_deck.update(extra_deck)
    session = GameSession(
        trace_dir=tmp_path,
        id_factory=lambda kind: f"interchangeable-{kind}",
    )
    frame = ExperienceFrame.model_validate(
        session.new_game(
            {
                "hero_deck": hero_deck,
                "villain_deck": {"Plains": 40},
                "villain_type": "passive",
                "seed": 347,
                "auto_pass": False,
            }
        )["frame"]
    )
    if frame.action_space == "MODAL":
        keep = next(offer for offer in frame.offers if offer.label == "Keep hand")
        frame = _choose(session, frame, keep)
    assert session.env is not None
    session.env.scenario_clear_hand(0)
    for card in hand:
        session.env.scenario_force_card_in_hand(0, card)
    for land in lands:
        session.env.scenario_force_battlefield(0, land, ready=True)
    session.obs = session.env.scenario_refresh()
    session.published_prompt = None
    return session, ExperienceFrame.model_validate(session._wire_message()["frame"])


def _send_command(
    session: GameSession,
    frame: ExperienceFrame,
    offer: InteractionOffer,
    *,
    command_id: str,
    expected_revision: int | None = None,
    offer_id: int | None = None,
) -> dict[str, Any]:
    assert frame.prompt is not None
    return session.hero_command(
        {
            "command_id": command_id,
            "match_id": frame.match_id,
            "expected_revision": (
                frame.revision if expected_revision is None else expected_revision
            ),
            "prompt_id": frame.prompt.id,
            "offer_id": offer.id if offer_id is None else offer_id,
            "answers": [],
        }
    )


def _choose(
    session: GameSession,
    frame: ExperienceFrame,
    offer: InteractionOffer,
) -> ExperienceFrame:
    outcome = _send_command(
        session, frame, offer, command_id=f"interchangeable.{frame.revision}.{offer.id}"
    )
    assert outcome["status"] == "accepted", outcome
    return ExperienceFrame.model_validate(outcome["update"]["frame"])


def test_dual_land_plus_three_plains_folds_with_aliases_replays_and_rejects_stale(
    tmp_path: Path,
) -> None:
    session, frame = _setup(
        tmp_path,
        ["Kyoshi Village", "Plains", "Plains", "Plains"],
        hand=["Youthful Knight", "Youthful Knight", "Plains"],
    )
    # Two identical castable copies in hand fold into one cast offer with one alias.
    cast_offers = [
        offer for offer in frame.offers if offer.label == "Cast Youthful Knight"
    ]
    assert len(cast_offers) == 1
    cast = cast_offers[0]
    hand_knights = [
        card.id
        for card in frame.projection.agent.hand
        if card.name == "Youthful Knight"
    ]
    assert len(hand_knights) == 2
    assert len(cast.subjects) == 1
    assert len(cast.aliases) == 1
    assert isinstance(cast.aliases[0], ObjectSubject)
    assert {cast.subjects[0].object.display_id, cast.aliases[0].id.entity} == set(
        hand_knights
    )

    payment = _choose(session, frame, cast)
    assert payment.action_space == "PAY_OR_NOT"
    assert [offer.label for offer in payment.offers] == [
        "Tap Kyoshi Village for {G}/{W}",
        "Tap Plains for {W}",
        "Pay the cost",
    ]
    # Unfolded offers omit aliases on the wire; the folded Plains offer names the other two.
    dumped = payment.model_dump(mode="json")
    assert "aliases" not in dumped["offers"][0]
    assert "aliases" not in dumped["offers"][2]

    plains_offer = payment.offers[1]
    plains_ids = [
        perm.id
        for perm in payment.projection.agent.battlefield
        if perm.name == "Plains" and not perm.tapped
    ]
    assert len(plains_ids) == 3
    assert len(plains_offer.subjects) == 1
    assert len(plains_offer.aliases) == 2
    assert all(isinstance(alias, ObjectSubject) for alias in plains_offer.aliases)
    assert {
        plains_offer.subjects[0].object.display_id,
        *(alias.id.entity for alias in plains_offer.aliases if alias.kind == "object"),
    } == set(plains_ids)

    # An offer id removed by folding is rejected as unknown.
    removed = _send_command(
        session,
        payment,
        plains_offer,
        command_id="interchangeable.unknown",
        offer_id=len(payment.offers),
    )
    assert removed["status"] == "rejected"
    assert removed["rejection"]["code"] == "unknown_offer"

    # Taking the canonical Plains offer commits the tap and resolves the remaining cost.
    after_tap = _choose(session, payment, plains_offer)
    assert after_tap.action_space == "PAY_OR_NOT"
    assert [offer.label for offer in after_tap.offers] == [
        "Tap Kyoshi Village for {G}/{W}",
        "Tap Plains for {W}",
        "Pay the cost",
    ]
    # Two untapped Plains remain, so one canonical offer carries one alias.
    assert len(after_tap.offers[1].aliases) == 1

    # Re-sending the previous revision's Command fails closed as stale.
    stale = _send_command(
        session,
        payment,
        plains_offer,
        command_id="interchangeable.stale",
        expected_revision=payment.revision,
    )
    assert stale["status"] == "rejected"
    assert stale["rejection"]["code"] == "stale_revision"

    # Finish paying and verify the recorded game replays to the same state digest.
    pay_rest = next(
        offer for offer in after_tap.offers if offer.label == "Pay the cost"
    )
    finished_cast = _choose(session, after_tap, pay_rest)
    assert finished_cast.action_space == "PRIORITY"
    assert any(
        perm.name == "Youthful Knight"
        for perm in finished_cast.projection.agent.battlefield
    )
    assert session.env is not None
    expected_digest = session.env.state_digest()
    assert session.canonical_decisions
    last_decision = session.canonical_decisions[-1]
    assert last_decision.offer.label == "Pay the cost"
    assert any(
        decision.offer.label == "Tap Plains for {W}"
        and len(decision.offer.aliases) == 2
        for decision in session.canonical_decisions
    )

    # Replay the exact same prefix from scratch to confirm determinism.
    replay_session, replay_frame = _setup(
        tmp_path / "replay",
        ["Kyoshi Village", "Plains", "Plains", "Plains"],
        hand=["Youthful Knight", "Youthful Knight", "Plains"],
    )
    for step_offer_id in (cast.id, plains_offer.id, pay_rest.id):
        offer = next(o for o in replay_frame.offers if o.id == step_offer_id)
        replay_frame = _choose(replay_session, replay_frame, offer)
    assert replay_session.env is not None
    assert replay_session.env.state_digest() == expected_digest
    assert replay_frame.frame_hash == finished_cast.frame_hash


def test_server_projects_decision_once_per_hero_prompt_and_skips_villain_autoplay(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    import managym.decision_group as decision_group_mod

    parse_calls: list[int] = []
    real_parse = decision_group_mod.parse_decision

    def spy_parse(raw: str) -> Any:
        parsed = real_parse(raw)
        parse_calls.append(parsed.address.viewer)
        return parsed

    monkeypatch.setattr(decision_group_mod, "parse_decision", spy_parse)

    session = GameSession(
        trace_dir=tmp_path,
        id_factory=lambda kind: f"proj-{kind}",
    )
    frame = ExperienceFrame.model_validate(
        session.new_game(
            {
                "hero_deck": {"Plains": 40},
                "villain_deck": {"Plains": 40},
                "villain_type": "passive",
                "seed": 11,
                "auto_pass": False,
            }
        )["frame"]
    )
    # Initial hero prompt computed once; live_decision_group reuses it without re-projecting.
    assert parse_calls == [session.human_seat]
    assert session.published_prompt is not None
    assert session.published_prompt.active_decision is not None
    assert session.live_decision_group() is None
    assert parse_calls == [session.human_seat]

    # Step past opening keep so villain autoplays its keep/turn without projecting env.decision().
    keep = next(offer for offer in frame.offers if offer.label == "Keep hand")
    parse_calls.clear()
    _choose(session, frame, keep)
    assert all(viewer == session.human_seat for viewer in parse_calls)
    assert len(parse_calls) == 1
    assert session.live_decision_group() is None
    assert len(parse_calls) == 1
