"""Explicit stream strata over a regime's two complete deck/sideboard setups.

Each stream contributes the same learner transitions per update. Fixed strata
therefore balance transition exposure, not completed games or wall time. Native
auto-reset retains the stream's setup; recovery replays the same assignment.
<<<<<<< HEAD
Elemental uniform sampling instead belongs to native game roots, not stream strata.
=======
>>>>>>> 20194feb758cd6d1df3e63b8d4fdd567d3ebbe6f
"""

import json
from typing import Literal

from manabot.env import Match
from manabot.infra.hypers import MatchHypers
<<<<<<< HEAD
import managym

Curriculum = Literal["fixed", "cross-balanced", "mirrors-balanced", "elemental-uniform"]


def stream_matches(match: Match, streams: int, curriculum: Curriculum) -> list[Match]:
    if curriculum == "elemental-uniform":
        raise ValueError("elemental-uniform requires native per-game sampling")
=======

Curriculum = Literal["fixed", "cross-balanced", "mirrors-balanced"]


def stream_matches(match: Match, streams: int, curriculum: Curriculum) -> list[Match]:
>>>>>>> 20194feb758cd6d1df3e63b8d4fdd567d3ebbe6f
    if curriculum == "fixed":
        return [match] * streams
    width = 12 if curriculum == "mirrors-balanced" else 4
    if streams % width:
        raise ValueError(f"{curriculum} requires streams divisible by {width}")
    reverse = match.swapped()
    cross = [match, match, reverse, reverse]
    if curriculum == "cross-balanced":
        return cross * (streams // width)
    roster = Match(
        MatchHypers.authored("ur-lessons-vs-gw-allies", "ur_lessons", "gw_allies")
    )

    def setups(value: Match) -> set[str]:
        return {
            json.dumps(
                {"deck": dict(p.decklist), "sideboard": dict(p.sideboard)},
                sort_keys=True,
            )
            for p in value.to_rust()
        }

    if setups(match) != setups(roster):
        raise ValueError(
            "mirror curriculum requires the exact selected authored roster"
        )
    first = match.hypers.model_copy(deep=True)
    first.content_pack = "ur-lessons-vs-gw-allies"
    first.villain_deck = dict(first.hero_deck)
    first.villain_sideboard = dict(first.hero_sideboard)
    second = reverse.hypers.model_copy(deep=True)
    second.content_pack = "ur-lessons-vs-gw-allies"
    second.villain_deck = dict(second.hero_deck)
    second.villain_sideboard = dict(second.hero_sideboard)
    return ([Match(first)] * 4 + [Match(second)] * 4 + cross) * (streams // width)
<<<<<<< HEAD


def elemental_roster(match: Match) -> list[managym.PlayerConfig]:
    """Admit a complete native Elemental roster and an exact representative pair."""
    pack = "elemental-world-v1"
    roster = [
        managym.authored_deck_setup(pack, key)
        for key in managym.authored_deck_keys(pack)
    ]
    if len(roster) != 10 or any(
        not any(
            actual.decklist == setup.decklist
            and actual.sideboard == setup.sideboard
            and actual.content_pack == setup.content_pack
            for setup in roster
        )
        for actual in match.to_rust()
    ):
        raise ValueError("elemental-uniform requires an exact Elemental authored match")
    return roster
=======
>>>>>>> 20194feb758cd6d1df3e63b8d4fdd567d3ebbe6f
