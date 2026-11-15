"""Constructed combat positions for protocol and browser proofs.

These roots bypass setup events and are not complete-game replay evidence.
The terminal smoke game is exercised separately through ordinary Commands.
"""

from __future__ import annotations

import json
from typing import Literal

import managym
from managym.experience import (
    DeclarationAnswer,
    DeclarationCommand,
    DeclarationPlan,
    ExperienceFrame,
    ObjectSubject,
)

CombatStage = Literal["attackers", "blockers"]
DECK = {
    "Mountain": 24,
    "Gray Ogre": 32,
    "Boggart Brute": 4,
    "Wind Drake": 4,
    "Giant Spider": 4,
}


def combat_config() -> dict[str, object]:
    return {
        "hero_deck": dict(DECK),
        "villain_deck": dict(DECK),
        "villain_type": "passive",
        "seed": 364,
        "auto_pass": False,
    }


def configure_combat(
    env: managym.Env, stage: CombatStage, *, count: int = 3, human_seat: int = 0
) -> managym.Observation:
    for seat in (0, 1):
        env.scenario_clear_hand(seat)
    attacking_seat = human_seat if stage == "attackers" else 1 - human_seat
    for name in ("Boggart Brute", "Boggart Brute", "Wind Drake"):
        env.scenario_force_battlefield(attacking_seat, name, ready=True)
    for index in range(count):
        env.scenario_force_battlefield(
            1 - attacking_seat,
            "Giant Spider" if index == 2 else "Gray Ogre",
            ready=True,
        )
    observation = env.scenario_refresh()
    for ordinal in range(32):
        projection = json.loads(env.declaration_projection_json(human_seat))
        if projection is not None and projection["stage"] == stage:
            return observation
        frame = json.loads(env.semantic_decision_frame_json())
        actions = observation.action_space.actions
        index = next(
            (
                index
                for index, action in enumerate(actions)
                if int(action.action_type)
                == int(managym.ActionEnum.PRIORITY_PASS_PRIORITY)
            ),
            None,
        )
        if index is None:
            index = next(
                (
                    index
                    for index, action in enumerate(actions)
                    if action.declared is (frame["actor"] != human_seat)
                ),
                0,
            )
        _, observation, _, _, _, _ = env.step_semantic_command(
            json.dumps(
                {
                    "command_id": f"constructed-combat.{ordinal}",
                    "expected_revision": frame["revision"],
                    "offer_id": frame["offers"][index]["id"],
                    "answers": [],
                }
            )
        )
    raise AssertionError("constructed combat did not reach its requested declaration")


def declaration_command(
    frame: ExperienceFrame, identity: str, *, select: bool = False
) -> DeclarationCommand:
    projection = frame.declaration
    assert projection is not None and frame.prompt is not None
    assignments: list[DeclarationAnswer] = []
    for row in projection.subjects:
        assert isinstance(row.object.subject, ObjectSubject)
        if not row.fixed:
            assignments.append(
                DeclarationAnswer(
                    subject=row.object.subject.id,
                    destination=row.destinations[0]
                    if select and row.destinations
                    else None,
                )
            )
    return DeclarationCommand(
        command_id=identity,
        match_id=frame.match_id,
        expected_revision=frame.revision,
        prompt_id=frame.prompt.id,
        plan=DeclarationPlan(binding=projection.binding, assignments=assignments),
    )
