"""Polarity-aware scripted greedy anchor, `scripted_greedy_v2`.

`ScriptedGreedyPlayer` (v1) takes the first non-decline target offer. Target
offers arrive in absolute engine order (player 0's objects first), so v1 picks
a target side by seat rather than by effect: from seat 0 it burns itself with
Igneous Inspiration. v2 changes only those side-ambiguous target prompts.

This lives outside `players.py` because arena registrations pin that file's
bytes as the v1 and demo-search source identity; v1 stays frozen.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from manabot.env import Env
from manabot.study.targeting import inspect_target
import managym

from .players import ScriptedGreedyPlayer

# Side-ambiguous targeted cards in the Allies/Lessons pool, keyed by engine
# card name. Side-restricted cards (Earth Kingdom Jailer, Yip Yip, ...) are
# already correct under v1 and need no entry.
TARGET_POLARITY: dict[str, Literal["harmful", "beneficial"]] = {
    "Firebending Lesson": "harmful",
    "Igneous Inspiration": "harmful",
    "Divide by Zero": "harmful",
    "It'll Quench Ya!": "harmful",
    "Fancy Footwork": "beneficial",
}


def _offer_side(
    raw: managym.Observation, index: int
) -> Literal["own", "opponent"] | None:
    """Return the viewer-relative side of one target offer.

    Players and permanents resolve through `inspect_target`. Stack spells,
    which that measurement leaves unresolved, use card ownership as their
    controller here. That proxy is a policy choice for this pool, which has no
    control-changing spells; it is not evidence.
    """
    choice = inspect_target(raw, index)
    if choice is None:
        return None
    if choice.target_kind != "unresolved":
        return choice.side
    focus = raw.action_space.actions[index].focus
    if len(focus) != 1:
        return None
    target = int(focus[0])
    if any(int(card.id) == target for card in raw.agent_cards):
        return "own"
    if any(int(card.id) == target for card in raw.opponent_cards):
        return "opponent"
    return None


class ScriptedGreedyV2Player(ScriptedGreedyPlayer):
    """`scripted_greedy` with effect-polarity targeting.

    For cards in `TARGET_POLARITY`, harmful effects take the first
    opponent-side offer and beneficial effects the first own-side offer. With
    no such offer the player declines when allowed, and otherwise falls back to
    v1. Every other prompt, including unknown-source triggers, is v1 behavior.
    """

    def act(self, env: Env, obs: dict[str, np.ndarray]) -> int:
        raw = env.last_raw_obs
        prompt = inspect_target(raw, 0)
        polarity = TARGET_POLARITY.get(prompt.source or "") if prompt else None
        if polarity is not None:
            wanted = "opponent" if polarity == "harmful" else "own"
            decline = None
            for index, action in enumerate(raw.action_space.actions):
                if int(action.action_type) == int(managym.ActionEnum.DECLINE_CHOICE):
                    decline = index
                elif _offer_side(raw, index) == wanted:
                    return index
            if decline is not None:
                return decline
        return super().act(env, obs)
