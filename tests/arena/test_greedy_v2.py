"""Polarity-aware greedy targeting over synthetic viewer observations."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from academy.enums import ActionEnum, ActionSpaceEnum
from manabot.arena.greedy_v2 import ScriptedGreedyV2Player
from manabot.arena.players import ScriptedGreedyPlayer
from manabot.env import Env

OWN, OPPONENT = 10, 20
SOURCE = 1
OWN_CREATURE, OPPONENT_CREATURE = 30, 40
OWN_SPELL, OPPONENT_SPELL = 2, 3


def offer(target: int) -> SimpleNamespace:
    return SimpleNamespace(action_type=ActionEnum.CHOOSE_TARGET, focus=[target])


DECLINE = SimpleNamespace(action_type=ActionEnum.DECLINE_CHOICE, focus=[])


def env(source: str, actions: list[SimpleNamespace]) -> Env:
    """Seat-0 offer order: own objects before the opponent's."""
    raw = SimpleNamespace(
        agent=SimpleNamespace(id=OWN),
        opponent=SimpleNamespace(id=OPPONENT),
        agent_cards=[
            SimpleNamespace(id=SOURCE, name=source),
            SimpleNamespace(id=OWN_SPELL, name="Own spell"),
        ],
        opponent_cards=[SimpleNamespace(id=OPPONENT_SPELL, name="Opposing spell")],
        agent_permanents=[SimpleNamespace(id=OWN_CREATURE, controller_id=OWN)],
        opponent_permanents=[
            SimpleNamespace(id=OPPONENT_CREATURE, controller_id=OPPONENT)
        ],
        action_space=SimpleNamespace(
            action_space_type=ActionSpaceEnum.CHOOSE_TARGET,
            focus=[SOURCE],
            actions=actions,
        ),
    )
    return cast(Env, SimpleNamespace(last_raw_obs=raw))


def choose(player: ScriptedGreedyPlayer, source: str, targets: list[int]) -> int:
    return player.act(env(source, [offer(t) for t in targets]), {})


@pytest.mark.parametrize(
    ("source", "targets", "expected"),
    [
        ("Igneous Inspiration", [OWN, OPPONENT, OWN_CREATURE], OPPONENT),
        ("Firebending Lesson", [OWN_CREATURE, OPPONENT_CREATURE], OPPONENT_CREATURE),
        ("Divide by Zero", [OWN_SPELL, OPPONENT_SPELL, OWN_CREATURE], OPPONENT_SPELL),
        ("It'll Quench Ya!", [OWN_SPELL, OPPONENT_SPELL], OPPONENT_SPELL),
        ("Fancy Footwork", [OPPONENT_CREATURE, OWN_CREATURE], OWN_CREATURE),
    ],
)
def test_v2_targets_by_effect_polarity(
    source: str, targets: list[int], expected: int
) -> None:
    assert targets[choose(ScriptedGreedyV2Player(), source, targets)] == expected
    assert choose(ScriptedGreedyPlayer(), source, targets) == 0


def test_v2_declines_rather_than_harm_itself() -> None:
    actions = [offer(OWN_CREATURE), DECLINE]
    assert ScriptedGreedyV2Player().act(env("Firebending Lesson", actions), {}) == 1


def test_v2_falls_back_to_v1_when_forced_or_unlisted() -> None:
    player = ScriptedGreedyV2Player()
    assert choose(player, "Firebending Lesson", [OWN_CREATURE]) == 0
    assert choose(player, "Earth Kingdom Jailer", [OWN_CREATURE, OPPONENT]) == 0
    unknown_source = env("Igneous Inspiration", [offer(OWN), offer(OPPONENT)])
    unknown_source.last_raw_obs.action_space.focus = []
    assert player.act(unknown_source, {}) == 0
