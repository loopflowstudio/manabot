"""Native representation proofs: visibility, complete choices and safe execution."""

from hashlib import sha256
import json
from pathlib import Path

from managym.decision import SemanticContractError
from managym.decision_capture import DecisionInput
import pytest

from manabot.semantic.text_observation import (
    RepresentationOverflow,
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
    ogre |= {"stun_counters": 2, "firebending": 1}
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
        "; stun counters 2; firebending 1; unlisted effective keywords false.",
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
