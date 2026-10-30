"""Native representation proofs: visibility, complete choices and safe execution."""

from hashlib import sha256
import json
from pathlib import Path

from managym.decision import SemanticContractError
from managym.decision_capture import DecisionInput
import numpy as np
import pytest

from manabot.env.observation import (
    CARD_PERMISSION_INDEX,
    PERMANENT_ONCE_EACH_TURN_INDEX,
    ObservationSpace,
)
from manabot.infra.config import ObservationConfig
from manabot.semantic.text_observation import (
    RepresentationOverflow,
    TextDecision,
    TextEncodingError,
    capture_text_decision,
    render_decision,
    resolve_action,
    submit_response,
)
from manabot.verify.text_observation import collision_branches, priority_root, run_demo
import managym


def test_native_decision_demo_retains_all_alternatives_and_collision(
    tmp_path: Path,
) -> None:
    run_demo(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    counts = summary["legal_alternatives_executed_and_replayed"]
    assert all(
        counts[name] >= 2
        for name in (
            "priority",
            "target",
            "combat",
            "blockers",
            "selection",
            "mode",
            "optional_payment",
        )
    )
    assert summary["schema_6_collision_preserved"]
    assert summary["captured_continuations_and_text_distinct"]
    assert summary["equal_current_state_distinct_history"]
    assert summary["expired_history_links_explicit"]
    left = (tmp_path / "payment_left/prompt.txt").read_text()
    right = (tmp_path / "payment_right/prompt.txt").read_text()
    assert "Selected targets in order:" in left and left != right
    blockers = (tmp_path / "blockers/prompt.txt").read_text()
    actions = blockers.split("Legal actions:\n", 1)[1]
    assert "Do not block with" in actions and "Block " in actions
    assert "; declaration" not in actions
    assert "; mode none" not in actions and "; ability none" not in actions
    assert "Source: none" not in blockers and "Chosen mode: none" not in blockers
    assert "power/toughness unavailable" not in blockers
    assert "power/toughness 2/2" in blockers
    modes = (tmp_path / "mode/prompt.txt").read_text().split("Legal actions:\n", 1)[1]
    assert all(f"mode {index}" in modes for index in (1, 2, 3))
    assert "kicker {4}" in (tmp_path / "optional_payment/prompt.txt").read_text()


def test_hidden_hand_permutations_do_not_change_input_or_text() -> None:
    env = priority_root(("Lightning Bolt", "Lightning Bolt"), ("Mountain", "Gray Ogre"))
    initial = capture_text_decision(env, history_capacity=32)
    actor = DecisionInput.model_validate_json(initial.capture_json).observation.viewer
    changed = 0
    for seed in range(8):
        branch = env.clone_env()
        before = branch.state_digest()
        branch.determinize(seed, actor)
        changed += branch.state_digest() != before
        decision = capture_text_decision(branch, history_capacity=32)
        assert decision.capture_json == initial.capture_json
        assert decision.text == initial.text
    assert changed > 0


def test_reset_and_offline_capture_cannot_execute() -> None:
    env = managym.Env(seed=23, skip_trivial=False)
    players = [managym.PlayerConfig(name, {"Mountain": 40}) for name in ("a", "b")]
    env.reset(players)
    decision = capture_text_decision(env)
    env.reset(players)
    assert capture_text_decision(env).capture_digest == decision.capture_digest
    before = env.state_digest()
    with pytest.raises(SemanticContractError, match="reset"):
        submit_response(env, decision, "A0")
    with pytest.raises(SemanticContractError, match="no live origin"):
        submit_response(env, render_decision(decision.capture_json), "A0")
    assert env.state_digest() == before


def test_complete_prompt_budgets_fail_without_truncation_or_execution() -> None:
    env = priority_root(("Lightning Bolt",), ("Mountain",))
    decision = capture_text_decision(env)
    before = env.state_digest()
    with pytest.raises(RepresentationOverflow):
        capture_text_decision(env, max_bytes=len(decision.text.encode()) - 1)
    with pytest.raises(RepresentationOverflow):
        decision.check_token_budget(lambda text: len(text.split()), 1)
    assert decision.check_token_budget(lambda text: len(text.split()), 100_000) > 1
    assert env.state_digest() == before


@pytest.mark.parametrize(
    "mutation",
    (
        "version",
        "frame_version",
        "extra",
        "unbound",
        "action",
        "overflow_history",
        "digest",
    ),
)
def test_unsupported_or_inconsistent_capture_fails_closed(mutation: str) -> None:
    env = priority_root(("Lightning Bolt",), ("Mountain",))
    decision = capture_text_decision(env)
    payload = json.loads(decision.capture_json)
    if mutation == "version":
        payload["capture_version"] = 999
    elif mutation == "frame_version":
        payload["observation"]["decision"]["schema_version"] += 1
    elif mutation == "extra":
        payload["observation"]["viewer_state"]["agent"]["secret_future_field"] = 1
    elif mutation == "unbound":
        payload["definitions"] = {}
    elif mutation == "action":
        payload["observation"]["viewer_state"]["action_space"]["actions"].pop()
    elif mutation == "overflow_history":
        payload["history"]["capacity"] = 33
    del payload["digest"]
    digest = sha256(
        json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()
    payload["digest"] = "invalid" if mutation == "digest" else digest
    with pytest.raises(TextEncodingError):
        render_decision(json.dumps(payload))


def test_duplicate_card_choices_have_distinct_aliases_and_commands() -> None:
    env = priority_root(("Lightning Bolt", "Lightning Bolt"), ("Mountain", "Mountain"))
    decision = capture_text_decision(env)
    casts = [
        action for action in decision.actions if action.description.startswith("Cast ")
    ]
    assert len(casts) == 2
    assert casts[0].description != casts[1].description
    assert (
        resolve_action(decision, casts[0].reference).offer_id
        != resolve_action(decision, casts[1].reference).offer_id
    )


def test_digest_rejects_changed_input_even_when_revision_is_unchanged() -> None:
    env = priority_root(("Lightning Bolt",), ("Mountain",))
    decision = capture_text_decision(env)
    captured = DecisionInput.model_validate_json(decision.capture_json)
    env.scenario_set_life(captured.observation.viewer, 19)
    current = DecisionInput.model_validate_json(capture_text_decision(env).capture_json)
    assert current.observation.revision == captured.observation.revision
    before = env.state_digest()
    with pytest.raises(SemanticContractError, match="stale_capture"):
        submit_response(env, decision, "A0")
    assert env.state_digest() == before


def test_capture_object_key_order_does_not_change_text() -> None:
    env = priority_root(("Lightning Bolt",), ("Mountain", "Gray Ogre"))
    decision = capture_text_decision(env)
    reordered = json.dumps(json.loads(decision.capture_json), sort_keys=True)
    assert render_decision(reordered).text == decision.text


def test_stack_text_uses_exact_native_links_over_frozen_card_ids() -> None:
    _, env = collision_branches()
    decision = capture_text_decision(env)
    submit_response(env, decision, "A0")
    payload = json.loads(capture_text_decision(env).capture_json)
    assert payload["frames"][0]["kind"] == 1
    target = payload["frames"][0]["targets"][0]["reference"]
    assert target["object"] is not None and target["definition"] is not None
    # Retained schema 6 cannot distinguish an expired incarnation from the
    # same physical card's current alias. Current frames deliberately can.
    target["object"] = None
    del payload["digest"]
    payload["digest"] = sha256(
        json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()
    text = render_decision(json.dumps(payload)).text
    stack = text.split("Stack (native order, top to bottom):\n", 1)[1].split("\n\n", 1)[
        0
    ]
    assert "no current object link" in stack
    assert "targets clause 1: D" in stack


def test_ally_restricted_mana_is_not_described_as_lesson_mana() -> None:
    env = priority_root((), ("Jasmine Dragon Tea Shop",))
    submit_response(env, capture_text_decision(env), "A0")
    modes = capture_text_decision(env)
    green = next(a for a in modes.actions if "Add {G}" in a.description)
    submit_response(env, modes, green.reference)
    text = capture_text_decision(env).text
    assert "Restricted mana: W 0, U 0, B 0, R 0, G 1, C 0." in text
    assert "Lesson-only" not in text and "Lesson spending" not in text
    assert "Restricted mana for Ally spells or abilities of Ally sources: G 1." in text


def _viewer_observation(decision: TextDecision) -> managym.Observation:
    source = DecisionInput.model_validate_json(decision.capture_json)
    return managym.Observation.from_json(json.dumps(source.observation.viewer_state))


def test_foretold_card_and_its_permission_reach_only_its_owner() -> None:
    """Text and both encoders show the owner its foretold card and terms; the
    opponent's text, JSON and tensor hold no trace beyond the exile count."""
    env = priority_root(("Sozin's Comet",), ("Mountain", "Mountain"))
    decision = capture_text_decision(env)
    foretell = next(a for a in decision.actions if "Foretell" in a.description)
    submit_response(env, decision, foretell.reference)

    owner = capture_text_decision(env)
    assert "foretold, face down" in owner.text
    assert "may be played from exile by You: cast for {2}{R}, only after turn" in (
        owner.text
    )
    space = ObservationSpace(ObservationConfig())
    owner_rows = space.encode(_viewer_observation(owner))["agent_cards"]
    assert owner_rows[:, CARD_PERMISSION_INDEX + 11].sum() == 1

    pass_priority = next(a for a in owner.actions if "Pass" in a.description)
    submit_response(env, owner, pass_priority.reference)
    opponent = capture_text_decision(env)
    # The opponent's deck holds Comets too; none of the owner's may show.
    assert not any(
        "(Sozin's Comet)" in line and "owner Opponent" in line
        for line in opponent.text.splitlines()
    )
    assert "foretold" not in opponent.text
    observation = _viewer_observation(opponent)
    assert all(card.name != "Sozin's Comet" for card in observation.opponent_cards)
    assert observation.opponent.zone_counts[5] == 1
    encoded = space.encode(observation)
    for key in ("agent_cards", "opponent_cards"):
        assert not encoded[key][:, CARD_PERMISSION_INDEX].any()


def test_once_each_turn_use_reaches_text_and_both_encoders() -> None:
    """Barrels' once-each-turn mana ability, once used, is in the text and in
    the same v5 permanent column of the reference and native encoders."""
    env = priority_root((), ("Barrels of Blasting Jelly", "Mountain"))
    decision = capture_text_decision(env)
    barrels = next(
        a
        for a in decision.actions
        if "Barrels of Blasting Jelly" in a.description and "Activate" in a.description
    )
    submit_response(env, decision, barrels.reference)
    while "once-each-turn abilities used this turn 1" not in (
        decision := capture_text_decision(env)
    ).text:
        submit_response(env, decision, decision.actions[0].reference)

    observation = _viewer_observation(decision)
    space = ObservationSpace(ObservationConfig())
    encoded = space.encode(observation)
    row = next(
        index
        for index, permanent in enumerate(observation.agent_permanents)
        if permanent.once_each_turn_used
    )
    assert encoded["agent_permanents"][row, PERMANENT_ONCE_EACH_TURN_INDEX] == 1
    native = env.encode_observation(observation, policy_input_version=5)
    for key, value in encoded.items():
        np.testing.assert_array_equal(native[key], value, err_msg=key)


def _redigest(payload: dict[str, object]) -> str:
    payload.pop("digest")
    payload["digest"] = sha256(
        json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()
    return json.dumps(payload)


def test_effect_derived_characteristics_appear_only_when_present() -> None:
    env = priority_root((), ("Mountain", "Gray Ogre"))
    decision = capture_text_decision(env)
    payload = json.loads(decision.capture_json)
    mountain, ogre = payload["observation"]["viewer_state"]["agent_permanents"]
    assert render_decision(_redigest(json.loads(decision.capture_json))).text == (
        decision.text
    )
    ogre |= {"stun_counters": 2}
    mountain |= {
        "attached_to": ogre["id"],
        "name": "Citizen",
        "creature_types": ["Citizen"],
        "is_granted_land": True,
    }
    text = render_decision(_redigest(payload)).text
    before = decision.text.splitlines()
    after = text.splitlines()
    changed = [(old, new) for old, new in zip(before, after) if old != new]
    assert len(before) == len(after) and len(changed) == 2
    (mountain_before, mountain_after), (ogre_before, ogre_after) = changed
    assert mountain_after == mountain_before.replace(
        "; unlisted effective keywords false.",
        "; attached to C2 (Gray Ogre); name set to Citizen; creature types set to"
        " Citizen; land through an effect; unlisted effective keywords false.",
    )
    assert ogre_after == ogre_before.replace(
        "; unlisted effective keywords false.",
        "; stun counters 2; unlisted effective keywords false.",
    )


def test_stack_text_states_announced_x_only_when_present() -> None:
    _, env = collision_branches()
    submit_response(env, capture_text_decision(env), "A0")
    decision = capture_text_decision(env)
    assert "X = " not in decision.text
    payload = json.loads(decision.capture_json)
    payload["observation"]["viewer_state"]["stack_objects"][0]["x"] = 3
    stack = render_decision(_redigest(payload)).text.split(
        "Stack (native order, top to bottom):\n", 1
    )[1]
    assert stack.startswith("S1: ") and "; X = 3" in stack.split("\n", 1)[0]


def test_stack_text_states_granted_spell_keywords_only_when_present() -> None:
    _, env = collision_branches()
    submit_response(env, capture_text_decision(env), "A0")
    decision = capture_text_decision(env)
    assert "granted keywords" not in decision.text
    payload = json.loads(decision.capture_json)
    keywords = {name: name == "lifelink" for name in _KEYWORDS}
    payload["observation"]["viewer_state"]["stack_objects"][0]["granted_keywords"] = (
        keywords
    )
    stack = render_decision(_redigest(payload)).text.split(
        "Stack (native order, top to bottom):\n", 1
    )[1]
    assert "; granted keywords lifelink" in stack.split("\n", 1)[0]


def test_face_down_card_text_appears_only_when_present() -> None:
    env = priority_root(("Gray Ogre",), ())
    decision = capture_text_decision(env)
    assert "face down" not in decision.text
    payload = json.loads(decision.capture_json)
    payload["observation"]["viewer_state"]["agent_cards"][0]["face_down"] = True
    text = render_decision(_redigest(payload)).text
    changed = [
        (old, new)
        for old, new in zip(decision.text.splitlines(), text.splitlines())
        if old != new
    ]
    assert len(changed) == 1 and "face down" in changed[0][1]


def _card_line(text: str, alias: str) -> str:
    return next(line for line in text.splitlines() if line.startswith(f"{alias} ("))


def _foretell(env: managym.Env) -> None:
    decision = capture_text_decision(env)
    foretell = next(a for a in decision.actions if "Foretell" in a.description)
    submit_response(env, decision, foretell.reference)


def test_foretell_is_an_ability_offer_naming_the_card() -> None:
    env = priority_root(("Sozin's Comet",), ("Mountain",) * 6)
    actions = capture_text_decision(env).text.split("Legal actions:\n", 1)[1]
    assert actions.splitlines() == [
        "A0 — Cast C1 (Sozin's Comet).",
        "A1 — Activate ability of C1 (Sozin's Comet). Native description:"
        " Foretell Sozin's Comet.",
        "A2 — Pass priority.",
    ]
    _foretell(env)
    decision = capture_text_decision(env)
    assert "; zone exile; " in _card_line(decision.text, "C1")
    assert "; foretold, face down" in _card_line(decision.text, "C1")
    assert [a.description for a in decision.actions] == ["Pass priority"]


def test_a_foretold_card_is_cast_from_exile_on_a_later_turn() -> None:
    env = priority_root(("Sozin's Comet",), ("Mountain",) * 6)
    _foretell(env)
    for _ in range(200):
        decision = capture_text_decision(env)
        if (
            "Turn 3. Phase: first main." in decision.text
            and "Active player: You. Acting player: You." in decision.text
        ):
            break
        passing = next(
            (a for a in decision.actions if a.description == "Pass priority"),
            decision.actions[0],
        )
        submit_response(env, decision, passing.reference)
    else:
        raise AssertionError("the owner's next main phase was not reached")
    cast = next(a for a in decision.actions if a.description.startswith("Cast "))
    alias = cast.description.removeprefix("Cast ").split(" ", 1)[0]
    assert cast.description == f"Cast {alias} (Sozin's Comet)"
    line = _card_line(decision.text, alias)
    assert "; zone exile; " in line and "; foretold, face down" in line


def test_an_ally_is_cast_from_the_visible_top_of_the_library() -> None:
    env = priority_root((), ("Hakoda, Selfless Commander",) + ("Plains",) * 4)
    decision = capture_text_decision(env)
    assert decision.actions[0].description == "Cast C1 (Hakoda, Selfless Commander)"
    assert "; zone library; " in _card_line(decision.text, "C1")
    submit_response(env, decision, decision.actions[0].reference)
    stack = capture_text_decision(env).text.split(
        "Stack (native order, top to bottom):\n", 1
    )[1]
    assert stack.startswith("S1: spell [D2]; controller You; source C2 (Hakoda,")


_KEYWORDS = (
    "flying",
    "reach",
    "haste",
    "flash",
    "vigilance",
    "trample",
    "first_strike",
    "double_strike",
    "deathtouch",
    "lifelink",
    "defender",
    "menace",
    "hexproof",
    "indestructible",
)

# Every observed field and the LLM text that states it, or why none is needed.
# A field added to the native observation fails here until it is listed.
_TEXT_FIELDS: dict[type, dict[str, str]] = {
    managym.Permanent: {
        "id": "the permanent's alias",
        "registry_key": "[definition]",
        "controller_id": "controller",
        "tapped": "tapped",
        "damage": "damage",
        "is_summoning_sick": "summoning sick",
        "plus1_counters": "+1/+1 counters",
        "cant_be_blocked_this_turn": "cannot be blocked this turn",
        "power": "power/toughness",
        "toughness": "power/toughness",
        "is_animated": "animated",
        "has_exile_link": "holds exile link",
        "exile_if_dies_this_turn": "exile if dies this turn",
        "temp_subtypes": "temporary subtypes",
        "keywords": "effective keywords",
        "stun_counters": "stun counters",
        "fire_counters": "fire counters",
        "double_strike_counters": "double strike counters",
        "attached_to": "attached to",
        "name": "name set to",
        "creature_types": "creature types set to",
        "is_granted_land": "land through an effect",
        "attacking": "attacking",
        "exhausted_abilities": "exhaust abilities used",
        "once_each_turn_used": "once-each-turn abilities used this turn",
        "granted_firebending": "firebending",
        "blocking": "blocking",
        "abilities_lost": "has lost all abilities",
        "used_abilities": "used-up abilities",
    },
    managym.Card: {
        "id": "the card's alias",
        "registry_key": "[definition]",
        "name": "the definition line",
        "text_box": "Rules:",
        "owner_id": "owner",
        "zone": "zone",
        "power": "card power/toughness",
        "toughness": "card power/toughness",
        "mana_cost": "cost",
        "card_types": "type flags",
        "keywords": "printed keywords",
        "is_token": "token",
        "is_legendary": "legendary",
        "is_ally": "Ally",
        "is_lesson": "Lesson",
        "ward_cost": "ward mana value",
        "kicker_cost": "kicker mana value",
        "face_down": "face down",
        "cast_permission": "may be played from exile by",
        "x": "announced X",
    },
    managym.StackObject: {
        "stack_object_id": "S<n>",
        "kind": "spell/activated ability/triggered ability",
        "controller_id": "controller",
        "source_card_registry_key": "[definition]",
        "source_permanent_id": "source",
        "description": "the ability text",
        "chosen_mode": "mode",
        "ability_index": "ability",
        "targets": "targets",
        "x": "X =",
        "granted_keywords": "granted keywords",
    },
    managym.Player: {
        "id": "the player's alias",
        "player_index": "seat label",
        "is_agent": "the viewer's own section",
        "is_active": "active player",
        "life": "life",
        "zone_counts": "zone counts",
        "graveyard_lessons": "graveyard Lessons",
        "combat_mana": "combat mana",
        "mana": "mana pool",
        "lesson_mana": "Restricted mana",
        "known_hand": "known hand",
        "sideboard_counts": "Initial sideboard",
        "remaining_sideboard_counts": "Owned outside-game candidates",
        "keeps_red_mana": "Keeps unspent red mana",
        "no_maximum_hand_size": "Has no maximum hand size",
        "restricted_mana": "Restricted mana for",
        "hand_only_until": "Can cast spells only from hand",
    },
    managym.Turn: {
        "turn_number": "Turn",
        "phase": "Phase",
        "step": "Step",
        "active_player_id": "active player",
        "agent_player_id": "the viewer's own section",
    },
}


@pytest.mark.parametrize("kind", list(_TEXT_FIELDS), ids=lambda kind: kind.__name__)
def test_every_observed_field_has_llm_text(kind: type) -> None:
    observed = {name for name in dir(kind) if not name.startswith("_")}
    assert observed == set(_TEXT_FIELDS[kind])


def test_combat_lost_abilities_used_activations_and_player_state_text() -> None:
    env = priority_root((), ("Mountain", "Gray Ogre"))
    decision = capture_text_decision(env)
    payload = json.loads(decision.capture_json)
    state = payload["observation"]["viewer_state"]
    mountain, ogre = state["agent_permanents"]
    ogre |= {"attacking": True, "abilities_lost": True, "used_abilities": [0]}
    mountain |= {"blocking": ogre["id"]}
    state["agent"] |= {"keeps_red_mana": True, "no_maximum_hand_size": True}
    text = render_decision(_redigest(payload)).text
    assert "; blocking C2 (Gray Ogre);" in text
    assert "; attacking;" in text
    assert "; has lost all abilities; used-up abilities 1;" in text
    assert "Keeps unspent red mana until end of turn." in text
    assert "Has no maximum hand size for the rest of the game." in text
    for absent in (
        "attacking",
        "blocking",
        "lost all",
        "used-up",
        "Keeps",
        "hand size",
    ):
        assert absent not in decision.text
