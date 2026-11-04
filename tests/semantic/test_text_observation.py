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
from manabot.managym.experience_projection import decision_surface_view
from manabot.semantic.text_observation import (
    _OBLIGATIONS,
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


def test_every_native_decision_code_has_a_label() -> None:
    assert len(_OBLIGATIONS) == managym.MAX_DECISION_CODE + 1


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


@pytest.mark.parametrize(
    "config",
    [
        ObservationConfig(policy_input_version=5),
        ObservationConfig(policy_input_version=6, max_facts=256),
        ObservationConfig(),
    ],
    ids=["v5", "v6-pilot", "current-default"],
)
def test_once_each_turn_use_reaches_text_and_both_encoders(
    config: ObservationConfig,
) -> None:
    """Barrels' once-each-turn mana ability, once used, is in the text and in
    the same permanent column of the reference and native encoders."""
    env = priority_root((), ("Barrels of Blasting Jelly", "Mountain"))
    decision = capture_text_decision(env)
    barrels = next(
        a
        for a in decision.actions
        if "Barrels of Blasting Jelly" in a.description and "Activate" in a.description
    )
    submit_response(env, decision, barrels.reference)
    while (
        "once-each-turn abilities used this turn 1"
        not in (decision := capture_text_decision(env)).text
    ):
        submit_response(env, decision, decision.actions[0].reference)

    observation = _viewer_observation(decision)
    space = ObservationSpace(config)
    encoded = space.encode(observation)
    row = next(
        index
        for index, permanent in enumerate(observation.agent_permanents)
        if permanent.once_each_turn_used
    )
    assert encoded["agent_permanents"][row, PERMANENT_ONCE_EACH_TURN_INDEX] == 1
    native = env.encode_observation(
        observation,
        policy_input_version=config.policy_input_version,
        max_facts=config.max_facts,
    )
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
    ogre |= {"counters": {"stun": 2}}
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
        "; timestamp rank",
        "; attached to C2 (Gray Ogre); name set to Citizen; creature types set to"
        " Citizen; land through an effect; timestamp rank",
    )
    assert ogre_after == ogre_before.replace(
        "; timestamp rank",
        "; stun counters 2; timestamp rank",
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
    assert stack.startswith("S1: spell [D1]; controller You; source C2 (Hakoda,")


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
        "counters": "<kind> counters",
        "effects": "effect",
        "copy": "copy of D<n>, except ...",
        "timestamp": "effects are listed in timestamp order; a permanent's "
        "static abilities apply in its battlefield order",
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
        "dealt_damage_this_turn": "dealt damage this turn",
        "transforms": "transformed N times",
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
        "keywords": "keywords",
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
    managym.PlayerTurnHistory: {
        "attacked": "This turn: attacked",
        "permanents_sacrificed": "sacrificed",
        "land_entered": "had a land enter",
        "cards_drawn": "drew",
        "bent": "bent",
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
        "main_counts": "Main deck, or Decklist: closed",
        "unseen_counts": "Unseen cards",
        "sideboard_counts": "Initial sideboard",
        "remaining_sideboard_counts": "Remaining sideboard",
        "keeps_red_mana": "Keeps unspent red mana",
        "no_maximum_hand_size": "Has no maximum hand size",
        "turn_history": "This turn:",
        "extra_turns": "extra turn",
        "additional_combats": "additional combat phase",
        "restricted_mana": "Restricted mana for",
        "hand_only_until": "Can cast spells only from hand",
        "effects": "Effect:",
    },
    managym.ContinuousEffect: {
        "modification": "the effect phrase",
        "power": "+P/+T or base power/toughness",
        "toughness": "+P/+T or base power/toughness",
        "amount": "firebending N",
        "keywords": "the keywords gained",
        "subtypes": "is also",
        "color": "keeps unspent <color> mana",
        "duration": "until end of turn:",
        "source_registry_key": "(from D<n>)",
        "source_name": "the definition line of (from D<n>)",
        "timestamp": "listed in timestamp order",
    },
    managym.SharedEffect: {
        "effect": "one complete effect phrase and timestamp rank",
        "controller_id": "controller alias",
        "source": "exact source alias with capture incarnation admission",
        "members": "affects each named member, including duplicates",
    },
    managym.EffectObject: {
        "object_id": "public player/permanent alias",
        "incarnation": "checked against the captured current object",
    },
    managym.EffectMembership: {
        "target": "affected public object alias",
        "order": "record ordinal for exact reconstruction",
    },
    managym.CopyEffect: {
        "registry_key": "copy of D<n>",
        "name": "the definition line of D<n>",
        "exceptions": "except <effect phrase>",
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
    ogre |= {
        "attacking": True,
        "abilities_lost": True,
        "used_abilities": [0],
        "dealt_damage_this_turn": True,
    }
    mountain |= {"blocking": ogre["id"]}
    state["agent"] |= {
        "keeps_red_mana": True,
        "no_maximum_hand_size": True,
        "extra_turns": 1,
        "turn_history": {
            "attacked": True,
            "permanents_sacrificed": 2,
            "land_entered": True,
            "cards_drawn": 1,
        },
    }
    text = render_decision(_redigest(payload)).text
    assert "; blocking C2 (Gray Ogre);" in text
    assert "; attacking;" in text
    assert (
        "; has lost all abilities; used-up abilities 1; dealt damage this turn" in text
    )
    assert "Keeps unspent red mana until end of turn." in text
    assert "Has no maximum hand size for the rest of the game." in text
    assert "Will take 1 extra turn(s) after this one." in text
    assert (
        "This turn: attacked; sacrificed 2 permanents; had a land enter under "
        "their control; drew 1 card." in text
    )


@pytest.mark.parametrize("historical", (False, True))
def test_continuous_effect_records_are_stated_in_order_with_duration(
    historical: bool,
) -> None:
    env = priority_root((), ("Mountain", "Gray Ogre"))
    decision = capture_text_decision(env)
    payload = json.loads(decision.capture_json)
    state = payload["observation"]["viewer_state"]
    _, ogre = state["agent_permanents"]
    ogre["effects"] = [
        {"modification": "LoseAllAbilities", "duration": "EndOfTurn", "timestamp": 90},
        {
            "modification": "ModifyPt",
            "power": 2,
            "toughness": 0,
            "duration": "EndOfTurn",
            "timestamp": 91,
        },
        {
            "modification": "AddKeywords",
            "keywords": ["flying", "first_strike"],
            "duration": "EndOfTurn",
            "timestamp": 92,
        },
        {
            "modification": "BecomeCreature",
            "duration": "Indefinite",
            "timestamp": 93,
        },
    ]
    state["agent"]["effects"] = [
        {
            "modification": "KeepUnspentMana",
            "color": "Red",
            "duration": "EndOfTurn",
            "timestamp": 94,
        }
    ]
    if historical:
        state.pop("shared_effects")
    else:
        incarnation = next(
            link["incarnation"]
            for link in payload["objects"].values()
            if link.get("permanent_id") == ogre["id"]
        )
        state["shared_effects"] = [
            {
                "effect": effect,
                "controller_id": state["agent"]["id"],
                "source": None,
                "members": [
                    {
                        "target": {"object_id": ogre["id"], "incarnation": incarnation},
                        "order": order,
                    }
                ],
            }
            for order, effect in enumerate(ogre["effects"])
        ]
        state["shared_effects"].append(
            {
                "effect": state["agent"]["effects"][0],
                "controller_id": state["agent"]["id"],
                "source": None,
                "members": [
                    {
                        "target": {
                            "object_id": state["agent"]["id"],
                            "incarnation": None,
                        },
                        "order": 4,
                    }
                ],
            }
        )
    text = render_decision(_redigest(payload)).text
    if not historical:
        phrases = [
            "loses all abilities",
            "+2/+0",
            "flying, first strike",
            "is also a creature",
            "keeps unspent red mana",
        ]
        positions = [text.index(f"Effect {index}: ") for index in range(1, 6)]
        assert positions == sorted(positions)
        for position, phrase in zip(positions, phrases, strict=True):
            assert phrase in text[position:].splitlines()[0]
        assert "timestamp rank" in text and "controller You" in text
        return
    assert (
        "; effect until end of turn: loses all abilities"
        "; effect until end of turn: +2/+0"
        "; effect until end of turn: flying, first strike"
        "; effect is also a creature;" in text
    )
    assert "Effect: until end of turn: keeps unspent red mana." in text
    for absent in (
        "attacking",
        "blocking",
        "lost all",
        "used-up",
        "Keeps",
        "hand size",
        "This turn:",
        "dealt damage",
    ):
        assert absent not in decision.text


@pytest.mark.parametrize("frequency", [None, 1, 2])
@pytest.mark.parametrize("this_turn", [False, True])
def test_waiting_delayed_trigger_relation_names_its_event_and_duration(
    frequency: int | None, this_turn: bool
) -> None:
    env = priority_root((), ("Mountain", "Gray Ogre"))
    payload = json.loads(capture_text_decision(env).capture_json)
    mountain, ogre = payload["observation"]["viewer_state"]["agent_permanents"]
    viewer = payload["observation"]["viewer_state"]["agent"]["player_index"]
    payload["frames"].append(
        {
            "kind": 10,
            "decision": 1,
            "parent": None,
            "controller": viewer,
            "source": {
                "object": None,
                "definition": ogre["registry_key"],
                "player": None,
            },
            "mode": None,
            "ability": None,
            "kicked": False,
            "active_clause": None,
            "targets": [],
            "context": {
                "object": ogre["id"],
                "definition": ogre["registry_key"],
                "player": None,
            },
            "sacrifice": None,
            "cost": None,
            "hybrid": [],
            "lesson": False,
            "waterbend": False,
            "can_decline": False,
            "mana_outputs": [],
            "resolutions_this_turn": 0,
            "this_turn": this_turn,
            "delayed_frequency": frequency,
            "remembered": [
                {
                    "object": mountain["id"],
                    "definition": mountain["registry_key"],
                    "player": None,
                }
            ],
        }
    )
    if frequency is None:
        del payload["frames"][-1]["delayed_frequency"]
    if not this_turn:
        del payload["frames"][-1]["this_turn"]
    text = render_decision(_redigest(payload)).text
    relation = next(
        line for line in text.splitlines() if "waiting delayed trigger" in line
    )
    assert "waits for the watched object to die" in relation
    assert "watching C" in relation and "(Gray Ogre)" in relation
    assert "remembered C" in relation and "(Mountain)" in relation
    assert (
        "waiting ends at this turn's cleanup" if this_turn else "no stated expiry"
    ) in relation
    if frequency is None:
        assert "fires once" not in relation and "fires on each" not in relation
    else:
        assert (
            "fires once" if frequency == 1 else "fires on each matching event"
        ) in relation


@pytest.mark.parametrize("frequency,kind", [(0, 10), (3, 10), (1, 3)])
def test_delayed_frequency_rejects_invalid_capture_codes(
    frequency: int, kind: int
) -> None:
    env = priority_root((), ("Mountain", "Gray Ogre"))
    payload = json.loads(capture_text_decision(env).capture_json)
    frame = {
        "kind": kind,
        "decision": 1 if kind == 10 else 0,
        "parent": None,
        "controller": 0,
        "source": {"object": None, "definition": None, "player": None},
        "mode": None,
        "ability": None,
        "kicked": False,
        "active_clause": None,
        "targets": [],
        "context": None,
        "sacrifice": None,
        "cost": None,
        "hybrid": [],
        "lesson": False,
        "waterbend": False,
        "can_decline": False,
        "mana_outputs": [],
        "resolutions_this_turn": 0,
        "delayed_frequency": frequency,
    }
    payload["frames"].append(frame)
    with pytest.raises(ValueError, match="invalid continuation frame relationship"):
        managym.parse_continuation_frames_json(json.dumps(payload["frames"]))


def test_a_copy_and_sacrifices_this_turn_are_stated() -> None:
    env = priority_root((), ("Mountain", "Gray Ogre"))
    decision = capture_text_decision(env)
    payload = json.loads(decision.capture_json)
    state = payload["observation"]["viewer_state"]
    _, ogre = state["agent_permanents"]
    ogre["copy"] = {
        "registry_key": ogre["registry_key"],
        "name": "Gray Ogre",
        "exceptions": [
            {
                "modification": "Nonlegendary",
                "duration": "Indefinite",
                "timestamp": ogre["id"],
            }
        ],
    }
    state["agent"]["turn_history"] = {"permanents_sacrificed": 2}
    text = render_decision(_redigest(payload)).text
    assert "; copy of D" in text and ", except isn't legendary" in text
    assert "This turn: sacrificed 2 permanents." in text
    assert "copy of" not in decision.text
    assert "This turn:" not in decision.text


@pytest.mark.parametrize("historical", (False, True))
def test_card_line_uses_the_observed_flag_without_rederiving_it_from_effects(
    historical: bool,
) -> None:
    """Conflicting injected data proves the renderer reads the engine's flag.

    Rust observability tests prove real Nonlegendary effects clear that flag;
    text rendering must not recalculate it from the permanent's effect list.
    """
    env = priority_root((), ("Mountain", "Gray Ogre"))
    decision = capture_text_decision(env)
    payload = json.loads(decision.capture_json)
    state = payload["observation"]["viewer_state"]
    _, ogre = state["agent_permanents"]
    ogre["effects"] = [
        {"modification": "Nonlegendary", "duration": "EndOfTurn", "timestamp": 90}
    ]
    if historical:
        state.pop("shared_effects")
    else:
        incarnation = next(
            link["incarnation"]
            for link in payload["objects"].values()
            if link.get("permanent_id") == ogre["id"]
        )
        state["shared_effects"] = [
            {
                "effect": ogre["effects"][0],
                "controller_id": state["agent"]["id"],
                "source": None,
                "members": [
                    {
                        "target": {"object_id": ogre["id"], "incarnation": incarnation},
                        "order": 0,
                    }
                ],
            }
        ]
    ogre_card = next(
        card
        for card in state["agent_cards"]
        if card["registry_key"] == ogre["registry_key"] and card["zone"] == 2
    )
    assert "legendary no" in decision.text
    ogre_card["is_legendary"] = True
    text = render_decision(_redigest(payload)).text
    card_line = next(
        line
        for line in text.splitlines()
        if "[D" in line and "Gray Ogre" in line and "owner" in line
    )
    assert "legendary yes" in card_line
    prefix = "; effect" if historical else "Effect 1:"
    assert f"{prefix} until end of turn: isn't legendary" in text


def test_joo_dee_surveil_and_copy_play_through_the_text_decision() -> None:
    """A surveil obligation (code 22) captures as text; resolving Joo Dee
    leaves a token stated as a copy and one sacrifice this turn."""
    env = priority_root((), ("Swamp", "Joo Dee, One of Many"))
    decision = capture_text_decision(env)
    activate = next(
        a
        for a in decision.actions
        if "Activate" in a.description and "Joo Dee" in a.description
    )
    submit_response(env, decision, activate.reference)
    for _ in range(8):
        decision = capture_text_decision(env)
        if "Stage: surveil." in decision.text:
            break
        submit_response(env, decision, decision.actions[0].reference)
    else:
        raise AssertionError("surveil decision not reached")
    assert "suspended decision" in decision.text and "; surveil" in decision.text
    keep = next(a for a in decision.actions if "Keep on top" in a.description)
    submit_response(env, decision, keep.reference)

    decision = capture_text_decision(env)
    source = decision.text.split("Source: ", 1)[1].split(" ", 1)[0]
    original = next(
        a
        for a in decision.actions
        if f"Select {source} " in a.description and "Sacrifice" in a.description
    )
    submit_response(env, decision, original.reference)
    text = capture_text_decision(env).text
    assert "; copy of D" in text
    assert "This turn: sacrificed 1 permanent." in text


def test_t6_payment_search_and_timing_context_have_text_meaning() -> None:
    env = priority_root((), ("Mountain", "Gray Ogre"))
    payload = json.loads(capture_text_decision(env).capture_json)
    state = payload["observation"]["viewer_state"]
    _, ogre = state["agent_permanents"]
    ogre["transforms"] = 2
    source = {"object": ogre["id"], "definition": ogre["registry_key"], "player": None}
    frame: dict[str, object] = {
        "kind": 3,
        "decision": 0,
        "parent": None,
        "controller": state["agent"]["player_index"],
        "source": source,
        "mode": None,
        "ability": 1,
        "kicked": False,
        "active_clause": None,
        "targets": [],
        "context": None,
        "sacrifice": None,
        "cost": None,
        "hybrid": [],
        "lesson": False,
        "waterbend": False,
        "can_decline": False,
        "mana_outputs": [],
        "resolutions_this_turn": 0,
    }
    payload["frames"] = [
        frame | {"x": 6, "source_transforms": 1, "inherited_from": source},
        frame | {"kind": 6, "decision": 24, "can_decline": True},
        frame | {"kind": 6, "decision": 25, "mode": 2, "can_decline": True},
        frame | {"kind": 10, "decision": 7, "delayed_frequency": 1},
        frame | {"kind": 11, "mode": 0},
        frame | {"kind": 12, "mode": 1, "attackers": "land creatures"},
    ]
    text = render_decision(_redigest(payload)).text
    for meaning in (
        "X = 6",
        "transformed 2 times",
        "source inherits abilities from",
        "choose optional payment X",
        "may exile up to 2 more cards",
        "its controller's end step",
        "fires once; consumed on matching event",
        "no stated expiry",
        "scheduled turn 1",
        "only land creatures may attack",
    ):
        assert meaning in text


def test_sozin_search_plays_through_text_policy_and_academy_projection() -> None:
    env = priority_root(("The Rise of Sozin",), ("Swamp",) * 6, opponent=("Gray Ogre",))
    raw = env.scenario_refresh()
    casted = False
    for _ in range(100):
        actions = raw.action_space.actions
        name = next(
            (
                i
                for i, action in enumerate(actions)
                if action.description == "Name Gray Ogre"
            ),
            None,
        )
        if name is not None:
            decision = capture_text_decision(env)
            assert len(decision.actions) == len(actions) > 64
            submit_response(env, decision, decision.actions[name].reference)
            break
        cast = next(
            (
                i
                for i, action in enumerate(actions)
                if not casted
                and action.action_type == managym.ActionEnum.PRIORITY_CAST_SPELL
            ),
            None,
        )
        target = next(
            (
                i
                for i, action in enumerate(actions)
                if action.action_type == managym.ActionEnum.CHOOSE_TARGET
                and raw.opponent.id in action.focus
            ),
            None,
        )
        quiet = next(
            (
                i
                for i, action in enumerate(actions)
                if action.action_type
                in (
                    managym.ActionEnum.PRIORITY_PASS_PRIORITY,
                    managym.ActionEnum.DECLINE_CHOICE,
                )
            ),
            0,
        )
        casted |= cast is not None
        raw, *_ = env.step(
            cast if cast is not None else target if target is not None else quiet
        )
    else:
        raise AssertionError("Sozin's name choice was not reached")
    viewer = env.current_agent_index()
    raw = env.observation_for_player(viewer)
    decision = capture_text_decision(env)
    assert "search named cards" in decision.text
    inspected_ids = {card.id for card in raw.opponent_cards if int(card.zone) in (0, 1)}
    assert inspected_ids
    projection = decision_surface_view(env, viewer)
    assert {card["id"] for card in projection["agent"]["inspected"]} == inspected_ids
    assert not projection["opponent"]["hand"]
    assert not decision_surface_view(env, 1 - viewer)["agent"].get("inspected")
    expected = ObservationSpace().encode(raw)
    for key, actual in env.encode_observation(raw).items():
        np.testing.assert_array_equal(actual, expected[key], err_msg=key)
    stop = next(
        action
        for action in decision.actions
        if "Stop searching and shuffle" in action.description
    )
    submit_response(env, decision, stop.reference)
    assert not decision_surface_view(env, viewer)["agent"].get("inspected")


def _assert_native_observation_parity(
    env: managym.Env, decision: TextDecision
) -> managym.Observation:
    # Read the native observation: captured viewer JSON omits continuation frames.
    viewer = DecisionInput.model_validate_json(decision.capture_json).observation.viewer
    observation = env.observation_for_player(viewer)
    encoded = ObservationSpace(ObservationConfig(policy_input_version=5)).encode(
        observation
    )
    native = env.encode_observation(observation, policy_input_version=5)
    for key, value in encoded.items():
        np.testing.assert_array_equal(native[key], value, err_msg=key)
    return observation


def test_accumulate_wisdom_ordering_reaches_text_commands_and_both_encoders() -> None:
    env = priority_root(("Accumulate Wisdom",), ("Island", "Island"))
    decision = capture_text_decision(env)
    cast = next(a for a in decision.actions if "Cast" in a.description)
    submit_response(env, decision, cast.reference)
    for _ in range(8):
        decision = capture_text_decision(env)
        if "Stage: private inspection." in decision.text:
            break
        submit_response(env, decision, decision.actions[0].reference)
    else:
        raise AssertionError("inspection decision not reached")
    assert len(decision.actions) == 3
    submit_response(env, decision, decision.actions[0].reference)
    decision = capture_text_decision(env)
    assert "Stage: order library bottom." in decision.text
    assert "; order library bottom" in decision.text
    assert "Choose which remaining card to draw first." in decision.text
    assert "Each choice commits immediately" in decision.text
    assert "the last card is placed automatically" in decision.text
    assert len(decision.actions) == 2
    assert all(
        "next on the bottom" in action.description for action in decision.actions
    )
    observation = _assert_native_observation_parity(env, decision)
    assert any(
        frame.kind == 6 and frame.decision == 23 for frame in observation.continuation
    )
    # Capture/Command execution preserves root isolation, replay and stale rejection.
    before = env.state_digest()
    for action in decision.actions:
        branch = env.clone_env()
        replay = env.clone_env()
        branch_decision = capture_text_decision(branch)
        submit_response(branch, branch_decision, action.reference)
        submit_response(replay, capture_text_decision(replay), action.reference)
        assert branch.state_digest() == replay.state_digest()
        with pytest.raises(SemanticContractError):
            submit_response(branch, branch_decision, action.reference)
    assert env.state_digest() == before


def test_suki_hand_power_reaches_text_and_both_encoders() -> None:
    env = priority_root(("Suki, Kyoshi Warrior",), ("Plains", "Gray Ogre", "Gray Ogre"))
    decision = capture_text_decision(env)
    assert any(
        "Suki, Kyoshi Warrior" in line and "power/toughness 2/4" in line
        for line in decision.text.splitlines()
    )
    _assert_native_observation_parity(env, decision)
