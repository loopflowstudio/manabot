"""A manifest's public target alias and private printing in acting-seat captures."""

import re

from manabot.semantic.text_observation import capture_text_decision, submit_response
from manabot.verify.text_observation import priority_root


def test_manifest_target_alias_from_both_seats() -> None:
    env = priority_root(
        ("Manifest Dread", "Lightning Bolt"),
        ("Forest", "Forest", "Mountain"),
        opponent=("Mountain",),
    )
    owner = env.current_agent_index()
    assert owner is not None
    env.scenario_force_card_in_hand(1 - owner, "Lightning Bolt")
    env.scenario_refresh()

    def take(pattern: str) -> str:
        decision = capture_text_decision(env)
        action = next(a for a in decision.actions if re.search(pattern, a.description))
        submit_response(env, decision, action.reference)
        return action.description

    take(r"Cast .*Manifest Dread")
    take(r"Tap for mana .*Forest")
    take(r"Tap for mana .*Forest")
    take("Pass priority")
    take("Pass priority")
    choice = take("Select")
    selected = re.search(r"Native description: Manifest (.+)", choice)
    assert selected is not None

    for viewer in (owner, 1 - owner):
        assert env.current_agent_index() == viewer
        decision = capture_text_decision(env)
        row = next(
            line
            for line in decision.text.splitlines()
            if "(nameless permanent)" in line and "power/toughness 2/2" in line
        )
        alias = re.search(r"P\d+ \(nameless permanent\)", row)
        assert alias is not None
        printing = re.search(r"\[(D\d+)\]", row)
        if viewer == owner:
            assert printing is not None
            assert f"{printing[1]} — {selected[1]}." in decision.text
        else:
            assert printing is None

        branch = env.clone_env()
        priority = capture_text_decision(branch)
        bolt = next(
            a
            for a in priority.actions
            if "Cast" in a.description and "Lightning Bolt" in a.description
        )
        submit_response(branch, priority, bolt.reference)
        targets = capture_text_decision(branch)
        assert any(alias[0] in a.description for a in targets.actions)
        if viewer == owner:
            take("Pass priority")
