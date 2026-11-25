"""The two Equipment payments remain distinguishable in text and Academy Commands."""

from pathlib import Path

import pytest

from manabot.semantic.text_observation import capture_text_decision, submit_response
from manabot.verify.text_observation import priority_root
import managym
from managym.academy.server import GameSession
from managym.experience import ExperienceFrame

BRACELET = "The Dominion Bracelet"


def step_action(
    env: managym.Env, kind: managym.ActionEnum, focus: list[int] | None = None
) -> None:
    actor = env.current_agent_index()
    assert actor is not None
    actions = env.observation_for_player(actor).action_space.actions
    env.step(
        next(
            i
            for i, action in enumerate(actions)
            if action.action_type == kind and (focus is None or action.focus == focus)
        )
    )


def equip_both() -> managym.Env:
    env = priority_root(
        (),
        ("Sakashima of a Thousand Faces", "Gray Ogre", BRACELET, BRACELET)
        + ("Island",) * 14,
    )
    permanents = env.observation_for_player(0).agent_permanents
    ogre = next(p.id for p in permanents if p.characteristics.name == "Gray Ogre")

    for permanent in permanents:
        if permanent.characteristics.name == BRACELET:
            step_action(
                env, managym.ActionEnum.PRIORITY_ACTIVATE_ABILITY, [permanent.id]
            )
            step_action(env, managym.ActionEnum.CHOOSE_TARGET, [ogre])
            step_action(env, managym.ActionEnum.PAY_COST)
            step_action(env, managym.ActionEnum.PRIORITY_PASS_PRIORITY)
            step_action(env, managym.ActionEnum.PRIORITY_PASS_PRIORITY)
    return env


@pytest.mark.parametrize("selected", [0, 1])
def test_bracelet_cost_labels_bind_the_selected_permanent(
    tmp_path: Path, selected: int
) -> None:
    env = equip_both()
    session = GameSession(trace_dir=tmp_path)
    session.new_game(
        {
            "hero_deck": {"Island": 40},
            "villain_deck": {"Island": 40},
            "villain_type": "passive",
            "auto_pass": False,
            "seed": 23,
        }
    )
    session.published_prompt = None
    session.env = env.clone_env()
    session.obs = session.env.observation_for_player(0)
    prompt = session._publish_current_prompt()
    assert prompt is not None
    frame = ExperienceFrame.model_validate(prompt.frame)
    decision = capture_text_decision(env)
    permanents = session.obs.agent_permanents
    bracelets = [p for p in permanents if p.characteristics.name == BRACELET]
    assert len(bracelets) == 2
    grantor = bracelets[selected]
    ogre = next(p for p in permanents if p.characteristics.name == "Gray Ogre")
    offer = next(o for o in frame.offers if o.focus == [ogre.id, grantor.id])
    assert offer.label.startswith("Activate ability of Gray Ogre:")
    assert f"exile {BRACELET} (permanent {grantor.id})" in offer.label
    rows = {p.id: i for i, p in enumerate(permanents, 1)}
    label = f"Activate ability of P{rows[ogre.id]} (Gray Ogre); exile P{rows[grantor.id]} ({BRACELET})"
    action = next(a for a in decision.actions if a.description.startswith(label))
    indexed = env.clone_env()
    step_action(
        indexed, managym.ActionEnum.PRIORITY_ACTIVATE_ABILITY, [ogre.id, grantor.id]
    )
    submit_response(env, decision, action.reference)
    result = session.hero_command(
        {
            "command_id": f"bracelet-{selected}",
            "match_id": session.match_id,
            "expected_revision": prompt.revision,
            "prompt_id": prompt.prompt_id,
            "offer_id": offer.id,
            "answers": [],
        }
    )
    assert result["status"] == "accepted"
    assert session.env.state_digest() == env.state_digest() == indexed.state_digest()
    for branch in (indexed, env, session.env):
        step_action(branch, managym.ActionEnum.CHOOSE_TARGET)
        step_action(branch, managym.ActionEnum.PAY_COST)
        paid = branch.observation_for_player(0)
        assert paid.stack_objects[-1].source_permanent_id == ogre.id
        assert {
            p.id for p in paid.agent_permanents if p.characteristics.name == BRACELET
        } == {bracelets[1 - selected].id}
        assert [c.id for c in paid.agent_cards if c.zone == managym.ZoneEnum.EXILE] == [
            grantor.object_id
        ]
    assert session.env.state_digest() == env.state_digest() == indexed.state_digest()
    session.close(end_reason="test_complete")
