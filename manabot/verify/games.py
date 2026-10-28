"""Torch-free deck fixtures and outcome helpers.

Game children, replay and provenance import these without the verification
runner's policy helpers, which need Torch.
"""

from __future__ import annotations

from typing import Any

import managym

STANDARD_DECK = {
    "Mountain": 12,
    "Forest": 12,
    "Llanowar Elves": 18,
    "Gray Ogre": 18,
}

# UR deck with real interaction: instant-speed removal (Lightning Bolt),
# permission (Counterspell), card draw (Ancestral Recall), and mass removal
# (Pyroclasm). Two colors for mana stability; 60 cards.
INTERACTIVE_DECK = {
    "Island": 12,
    "Mountain": 12,
    "Gray Ogre": 6,
    "Wind Drake": 6,
    "Man-o'-War": 4,
    "Raging Goblin": 4,
    "Lightning Bolt": 6,
    "Counterspell": 4,
    "Ancestral Recall": 3,
    "Pyroclasm": 3,
}

# Main-deck projections for callers that inspect the selected matchup. Match
# construction uses MatchConfig.authored to carry sideboards as well.
UR_LESSONS_DECK = dict(
    managym.authored_deck_setup("ur-lessons-vs-gw-allies", "ur_lessons").decklist
)
GW_ALLIES_DECK = dict(
    managym.authored_deck_setup("ur-lessons-vs-gw-allies", "gw_allies").decklist
)


def winner_from_info_or_obs(info: dict[str, Any], raw_obs) -> int | None:
    """Return winner player index when available, else None."""

    if "winner_index" in info:
        try:
            return int(info["winner_index"])
        except (TypeError, ValueError):
            pass

    agent_idx = int(raw_obs.agent.player_index)
    opp_idx = int(raw_obs.opponent.player_index)

    if raw_obs.agent.life <= 0:
        return opp_idx
    if raw_obs.opponent.life <= 0:
        return agent_idx

    return None
