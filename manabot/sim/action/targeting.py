"""Inspect public target identity for simulation policies.

Source identity comes only from action-space focus; target side uses public
player/permanent IDs and current controllers, never labels or hidden engine
state. Own-side targeting is a measurement, not a judgment of strategic value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from managym.enums import ActionEnum, ActionSpaceEnum

if TYPE_CHECKING:
    from managym import Observation


@dataclass(frozen=True)
class TargetChoice:
    source: str | None
    target_kind: Literal["player", "permanent", "unresolved", "declined"]
    side: Literal["own", "opponent"] | None


def inspect_target(raw: Observation, action_index: int) -> TargetChoice | None:
    """Describe one target prompt; missing source/target identity stays unknown."""
    if int(raw.action_space.action_space_type) != ActionSpaceEnum.CHOOSE_TARGET:
        return None
    action = raw.action_space.actions[action_index]
    cards = {int(card.id): card.name for card in raw.agent_cards + raw.opponent_cards}
    source = next(
        (cards[int(i)] for i in raw.action_space.focus if int(i) in cards), None
    )
    if int(action.action_type) == ActionEnum.DECLINE_CHOICE:
        return TargetChoice(source, "declined", None)
    if int(action.action_type) != ActionEnum.CHOOSE_TARGET:
        raise ValueError("unexpected action in target prompt")
    if len(action.focus) != 1:
        return TargetChoice(source, "unresolved", None)
    target_id = int(action.focus[0])
    sides: dict[int, Literal["own", "opponent"]] = {
        int(raw.agent.id): "own",
        int(raw.opponent.id): "opponent",
    }
    if target_id in sides:
        return TargetChoice(source, "player", sides[target_id])
    for permanent in raw.agent_permanents + raw.opponent_permanents:
        if int(permanent.id) == target_id:
            side = sides.get(int(permanent.controller_id))
            return TargetChoice(
                source, "permanent" if side is not None else "unresolved", side
            )
    # Stack spell action focus is a card ID, not a StackObject ID. Do not infer
    # its controller from card ownership or conflate the two identity domains.
    return TargetChoice(source, "unresolved", None)
