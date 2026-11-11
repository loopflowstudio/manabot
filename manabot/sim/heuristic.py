"""Rule-of-thumb arena anchor, `heuristic`.

`ScriptedGreedyPlayer` (v1) plays a land, casts the first castable spell,
attacks and blocks with everything, and takes the first target offer. Target
offers arrive in absolute engine order (player 0's objects first), so v1 picks
a target side by seat: from seat 0 it burns itself with Igneous Inspiration.

`HeuristicPlayer` keeps v1 for every prompt it has no rule for and adds:

- Mana: play a land, crack fetch lands, then cast the castable spell with the
  highest mana value that has a worthwhile use, one spell at a time (its own
  spell resolves before the next). The engine only offers affordable casts, so
  repeated highest-first casting spends as much mana as the offers allow.
  Kicker is paid when untapped lands cover spell plus kicker.
- Instant timing: delay instants in its own upkeep or draw step. Instants without a
  target that matters now (card draw, selection) and draw or life abilities
  wait for the opponent's end step, so the turn's mana goes to permanents first.
- Targets: effect polarity is derived from the engine's card definitions
  (native card definitions: damage, destroy, bounce, counter and exile
  are harmful; pumps, untaps and keyword grants are beneficial). Harmful
  effects go to the opponent's side, beneficial ones to its own. Removal is
  held until it kills something (with the damage the available mana and
  graveyard Lessons allow), burn that can hit players until it kills a creature
  or the opponent, counters until the opponent has a spell on the stack.
- Card choices: Learn takes a Lesson (removal while the opponent has creatures);
  discards are spare lands or else the costliest card, optional ones declined;
  a look-and-select keeps a land while short of four.
- Combat: hold back the toughest attackers while the opponent's swing back
  would be lethal; attack with a creature unless the opponent has an untapped
  creature that can block it, kill it and survive, or with everything when
  unblockable damage is lethal. Blocks are planned once per combat, one
  blocker per attacker, largest attackers first: kill and survive, trade up,
  absorb for free; then chump the largest attackers while the rest is lethal.
- Own combat tricks: a pump in hand with mana to cast it lets one dominated
  creature attack anyway. After blocks, pumps are cast only where they change
  a one-on-one fight or make unblocked damage lethal, and self-pump abilities
  go on unblocked attackers. The opponent's tricks are ignored: the Elemental
  lists hold too few for a hidden-hand estimate to change a decision.
- Optional costs (ward, "unless pays"): pay when offered.

Combat assignments come from the observation's attacker/blocker events, whose
object IDs are the current permanent IDs; the player remembers them for the
turn. Mana is approximated by untapped lands, ignoring colors and other
sources. Creature identity uses the visible card definitions; a permanent
without one counts as a creature when its toughness is positive.

This lives outside `anchors.py` because arena registrations pin that file's
bytes as the v1 source identity; v1 stays frozen. The rules themselves live in
`managym/src/agent/heuristic`; this module only adapts the scalar player API.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from manabot import managym
from manabot.env.env import Env


def heuristic_source_paths() -> list[Path]:
    """Native rules and adapter bytes identifying newly registered opponents."""
    root = Path(__file__).resolve().parents[2]
    return [
        Path(__file__).resolve(),
        root / "managym/src/agent/heuristic/mod.rs",
        root / "managym/src/agent/heuristic/knowledge.rs",
        root / "managym/src/python/heuristic_bindings.rs",
    ]


class HeuristicPlayer:
    """Stateful adapter to the engine's viewer-safe heuristic selector."""

    def __init__(self, *, max_actions: int | None = None) -> None:
        if max_actions is not None and (
            type(max_actions) is not int or max_actions < 1
        ):
            raise ValueError("heuristic max_actions must be a positive integer")
        self._native = managym.Heuristic(max_actions=max_actions)

    def act(self, env: Env, obs: dict[str, np.ndarray]) -> int:
        del obs
        return self._native.select_action(env.last_raw_obs)
