"""Academy manual dual-land payments skip when forced and label choice taps cleanly."""

import json
from pathlib import Path
from typing import Any

import pytest

from managym.academy.recording import Recording
from managym.academy.server import GameSession
from managym.experience import ExperienceFrame, InteractionOffer
from managym.payment import preview_payment


def _setup(
    tmp_path: Path,
    lands: list[str],
    *,
    spell: str = "Youthful Knight",
    extra_hand: tuple[str, ...] = ("Plains",),
    mana_floats: str | None = None,
) -> tuple[GameSession, ExperienceFrame]:
    config: dict[str, Any] = {
        "hero_deck": {
            "Kyoshi Village": 4,
            "Plains": 28,
            spell: 8,
        },
        "villain_deck": {"Plains": 40},
        "villain_type": "passive",
        "seed": 347,
        "auto_pass": False,
    }
    if mana_floats is not None:
        config["mana_floats"] = mana_floats
    session = GameSession(trace_dir=tmp_path)
    frame = ExperienceFrame.model_validate(session.new_game(config)["frame"])
    if frame.action_space == "MODAL":
        keep = next(offer for offer in frame.offers if offer.label == "Keep hand")
        _choose(session, frame, keep)
    assert session.env is not None
    session.env.scenario_clear_hand(0)
    session.env.scenario_force_card_in_hand(0, spell)
    for card in extra_hand:
        session.env.scenario_force_card_in_hand(0, card)
    for land in lands:
        session.env.scenario_force_battlefield(0, land, ready=True)
    session.obs = session.env.scenario_refresh()
    session.published_prompt = None
    return session, ExperienceFrame.model_validate(session._wire_message()["frame"])


def _choose(
    session: GameSession,
    frame: ExperienceFrame,
    offer: InteractionOffer,
) -> ExperienceFrame:
    assert frame.prompt is not None
    outcome = session.hero_command(
        {
            "command_id": f"dual.{frame.revision}.{offer.id}",
            "match_id": frame.match_id,
            "expected_revision": frame.revision,
            "prompt_id": frame.prompt.id,
            "offer_id": offer.id,
            "answers": [],
        }
    )
    assert outcome["status"] == "accepted"
    return ExperienceFrame.model_validate(outcome["update"]["frame"])


def test_exact_dual_land_two_drop_skips_payment(tmp_path: Path) -> None:
    session, frame = _setup(tmp_path, ["Kyoshi Village", "Plains"])
    assert session.env is not None
    cast = next(
        offer for offer in frame.offers if offer.label == "Cast Youthful Knight"
    )
    address = json.loads(session.env.semantic_decision_frame_json())
    preview = preview_payment(
        session.env,
        viewer=address["actor"],
        revision=address["revision"],
        fingerprint=address["fingerprint"],
        action_index=cast.id,
    )
    assert preview.state == "exact"
    assert preview.cost == "{1}{W}"
    assert {source.label for source in preview.sources} == {"Kyoshi Village", "Plains"}

    after = _choose(session, frame, cast)
    assert after.action_space == "PRIORITY"
    battlefield = after.projection.agent.battlefield
    assert any(permanent.name == "Youthful Knight" for permanent in battlefield)
    tapped = sorted(permanent.name for permanent in battlefield if permanent.tapped)
    assert tapped == ["Kyoshi Village", "Plains"]
    assert not any(after.projection.agent.mana)


@pytest.mark.parametrize(
    "play_third_plains", [False, True], ids=["ready", "played_this_turn"]
)
def test_dual_land_plus_three_plains_labels_and_skips_redundant_modal(
    tmp_path: Path, play_third_plains: bool
) -> None:
    lands = (
        ["Kyoshi Village", "Plains", "Plains"]
        if play_third_plains
        else ["Kyoshi Village", "Plains", "Plains", "Plains"]
    )
    # Keep a playable card in hand after the cast so priority stays in this main
    # phase instead of auto-passing into the next untap step.
    extra_hand = ("Plains", "Youthful Knight") if play_third_plains else ("Plains",)
    session, frame = _setup(tmp_path, lands, extra_hand=extra_hand)
    if play_third_plains:
        play_plains = next(
            offer for offer in frame.offers if offer.label == "Play Plains"
        )
        frame = _choose(session, frame, play_plains)
    cast = next(
        offer for offer in frame.offers if offer.label == "Cast Youthful Knight"
    )

    payment = _choose(session, frame, cast)
    assert payment.action_space == "PAY_OR_NOT"
    assert [offer.label for offer in payment.offers] == [
        "Tap Kyoshi Village for {G}/{W}",
        "Tap Plains for {W}",
        "Pay the cost",
    ]
    assert len(payment.offers[1].aliases) == 2

    # Tapping Kyoshi Village leaves 3 interchangeable Plains for {1}/{W} (even
    # when the 3rd Plains was played this turn), so the color modal skips too.
    after = _choose(session, payment, payment.offers[0])
    assert after.action_space == "PRIORITY"
    battlefield = after.projection.agent.battlefield
    assert any(permanent.name == "Youthful Knight" for permanent in battlefield)
    tapped = sorted(permanent.name for permanent in battlefield if permanent.tapped)
    untapped_plains = [
        permanent
        for permanent in battlefield
        if permanent.name == "Plains" and not permanent.tapped
    ]
    assert tapped == ["Kyoshi Village", "Plains"]
    assert len(untapped_plains) == 2
    assert not any(after.projection.agent.mana)


def test_priority_dual_land_float_narrates_resolved_color_not_raw_offer(
    tmp_path: Path,
) -> None:
    session, frame = _setup(tmp_path, ["Kyoshi Village"], mana_floats="all")
    assert session.trace is not None
    tap_index = len(session.trace.events)
    tap = next(
        offer
        for offer in frame.offers
        if offer.label == "Tap Kyoshi Village for {G}/{W}"
    )
    modal = _choose(session, frame, tap)
    assert modal.action_space == "MODAL"
    add_white = next(offer for offer in modal.offers if offer.label == "Add {W}")
    after = _choose(session, modal, add_white)
    assert after.action_space == "PRIORITY"
    assert after.projection.agent.mana[0] == 1

    payload = session.attempt_store.load(session.match_id)
    assert payload is not None
    recording = Recording(session.match_id, payload)
    consequences = {
        position.event_index: position
        for position in recording.positions
        if position.kind == "consequences"
    }
    assert recording.metadata(consequences[tap_index])["label"] is None
    resolved = consequences[tap_index + 1]
    assert recording.metadata(resolved)["label"] == "Tap Kyoshi Village for {W}"
    assert recording.read(resolved.key)["description"] == "Tap Kyoshi Village for {W}"
