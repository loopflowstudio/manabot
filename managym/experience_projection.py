"""Protocol-v1 viewer projection of a native managym ``Observation``.

``serialize_observation`` turns an engine observation into the JSON
``projection`` carried by an experience frame, ``redact_observation`` removes
the opponent's hidden hand so the projection is viewer-safe, and
``describe_actions`` / ``offer_verb`` name the legal actions as protocol-v1
offers. Academy uses this to render live tables and traces; manabot uses the
same functions to record replay and teacher evidence, so recorded frame hashes
depend on this module byte for byte. Change it only together with the frozen
evidence it reproduces.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import managym
from managym.enums import (
    ActionEnum,
    ActionSpaceEnum,
    PhaseEnum,
    StepEnum,
    ZoneEnum,
)

# The protocol-v1 projection labels seat 0 "Hero" and seat 1 "Villain".
HERO_PLAYER_INDEX = 0

ACTION_LABELS = {
    "PRIORITY_PLAY_LAND": "Play land",
    "PRIORITY_CAST_SPELL": "Cast spell",
    "PRIORITY_ACTIVATE_ABILITY": "Activate ability",
    "DECLARE_ATTACKER": "Declare attacker",
    "DECLARE_BLOCKER": "Declare blocker",
    "CHOOSE_TARGET": "Choose target",
}


def enum_name(enum_type: type[IntEnum], value: Any) -> str:
    """Name a native enum value, or its integer when the mirror lacks it."""
    try:
        return enum_type(int(value)).name
    except Exception:
        return str(int(value))


def _serialize_card(card: managym.Card) -> dict[str, Any]:
    return {
        "id": int(card.id),
        "registry_key": int(card.registry_key),
        "name": card.name,
        "text_box": card.text_box,
        "zone": enum_name(ZoneEnum, card.zone),
        "owner_id": int(card.owner_id),
        "power": int(card.power),
        "toughness": int(card.toughness),
        "mana_value": int(card.mana_cost.mana_value),
        "types": {
            "is_creature": bool(card.card_types.is_creature),
            "is_land": bool(card.card_types.is_land),
            "is_spell": bool(card.card_types.is_spell),
            "is_artifact": bool(card.card_types.is_artifact),
            "is_enchantment": bool(card.card_types.is_enchantment),
            "is_planeswalker": bool(card.card_types.is_planeswalker),
            "is_battle": bool(card.card_types.is_battle),
        },
    }


def _serialize_permanent(
    permanent: managym.Permanent,
    card: managym.Card | None,
) -> dict[str, Any]:
    return {
        "id": int(permanent.id),
        "name": card.name if card else None,
        "text_box": card.text_box if card else "",
        "exile_if_dies_this_turn": bool(permanent.exile_if_dies_this_turn),
        "controller_id": int(permanent.controller_id),
        "tapped": bool(permanent.tapped),
        "damage": int(permanent.damage),
        "summoning_sick": bool(permanent.is_summoning_sick),
        # Effective P/T (statics, until-EOT buffs, +1/+1 counters) — what the
        # SBA actually checks. Printed values ride alongside so the UI can
        # show "4/4 (2/2)" on a buffed permanent.
        "power": int(permanent.power),
        "toughness": int(permanent.toughness),
        "base_power": int(card.power) if card else None,
        "base_toughness": int(card.toughness) if card else None,
        "plus1_counters": int(permanent.plus1_counters),
    }


def _serialize_player(
    player: managym.Player,
    cards: list[managym.Card],
    permanents: list[managym.Permanent],
) -> dict[str, Any]:
    grouped_cards: dict[str, list[dict[str, Any]]] = {
        "HAND": [],
        "GRAVEYARD": [],
        "EXILE": [],
        "STACK": [],
    }
    battlefield_cards: list[managym.Card] = []
    for card in cards:
        zone_name = enum_name(ZoneEnum, card.zone)
        if zone_name == "BATTLEFIELD":
            battlefield_cards.append(card)
        elif zone_name in grouped_cards:
            grouped_cards[zone_name].append(_serialize_card(card))

    # Permanents and battlefield cards correspond 1:1 in order.
    # Fall back to None if the lists don't align.
    serialized_permanents = []
    for i, permanent in enumerate(permanents):
        card = battlefield_cards[i] if i < len(battlefield_cards) else None
        serialized_permanents.append(_serialize_permanent(permanent, card))

    zone_counts = {
        enum_name(ZoneEnum, index): int(count)
        for index, count in enumerate(player.zone_counts)
    }

    return {
        "player_index": int(player.player_index),
        "id": int(player.id),
        "is_active": bool(player.is_active),
        "is_agent": bool(player.is_agent),
        "life": int(player.life),
        "zone_counts": zone_counts,
        "library_count": zone_counts.get("LIBRARY", 0),
        "known_hand": {
            str(key): int(count) for key, count in player.known_hand.items()
        },
        "sideboard_counts": {
            str(key): int(count) for key, count in player.sideboard_counts.items()
        },
        "remaining_sideboard_counts": {
            str(key): int(count)
            for key, count in player.remaining_sideboard_counts.items()
        },
        "hand": grouped_cards["HAND"],
        "graveyard": grouped_cards["GRAVEYARD"],
        "exile": grouped_cards["EXILE"],
        "stack": grouped_cards["STACK"],
        "battlefield": serialized_permanents,
    }


def serialize_observation(obs: managym.Observation) -> dict[str, Any]:
    data = {
        "game_over": bool(obs.game_over),
        "won": bool(obs.won),
        "turn": {
            "turn_number": int(obs.turn.turn_number),
            "phase": enum_name(PhaseEnum, obs.turn.phase),
            "step": enum_name(StepEnum, obs.turn.step),
            "active_player_id": int(obs.turn.active_player_id),
            "agent_player_id": int(obs.turn.agent_player_id),
        },
        "agent": _serialize_player(obs.agent, obs.agent_cards, obs.agent_permanents),
        "opponent": _serialize_player(
            obs.opponent,
            obs.opponent_cards,
            obs.opponent_permanents,
        ),
    }

    data["agent"]["sideboard"] = [
        {
            "candidate_id": int(card.candidate_id),
            "owner_id": int(card.owner_id),
            "registry_key": int(card.registry_key),
            "name": card.name,
            "text_box": card.text_box,
        }
        for card in obs.agent_sideboard
    ]
    return data


def _player_label(player: managym.Player) -> str:
    return "Hero" if int(player.player_index) == HERO_PLAYER_INDEX else "Villain"


def _permanent_names(
    cards: list[managym.Card],
    permanents: list[managym.Permanent],
) -> dict[int, str]:
    """Map permanent ids to card names.

    Permanents carry their own object ids (distinct from card ids), but the
    observation lists permanents and battlefield cards in the same order —
    the same alignment _serialize_player relies on.
    """
    battlefield_cards = [
        card for card in cards if enum_name(ZoneEnum, card.zone) == "BATTLEFIELD"
    ]
    names: dict[int, str] = {}
    for index, permanent in enumerate(permanents):
        if index < len(battlefield_cards):
            names[int(permanent.id)] = battlefield_cards[index].name
    return names


def _build_id_to_name(obs: managym.Observation) -> dict[int, str]:
    """Map object ids (players, cards, permanents) to display names."""
    names: dict[int, str] = {
        int(obs.agent.id): _player_label(obs.agent),
        int(obs.opponent.id): _player_label(obs.opponent),
    }
    names.update(
        {int(card.id): card.name for card in [*obs.agent_cards, *obs.opponent_cards]}
    )
    names.update({int(card.candidate_id): card.name for card in obs.agent_sideboard})
    names.update(_permanent_names(obs.agent_cards, obs.agent_permanents))
    names.update(_permanent_names(obs.opponent_cards, obs.opponent_permanents))
    return names


def format_action(
    action: managym.Action,
    names: dict[int, str],
    space_kind: str = "",
    declared: bool | None = None,
) -> str:
    """Render a legal action in Magic terms with card names.

    Focus semantics (managym observation.rs action_focus):
      PLAY_LAND / CAST_SPELL: [card_id]
      DECLARE_ATTACKER:       [attacker_permanent_id]
      DECLARE_BLOCKER:        [blocker_permanent_id, attacker_permanent_id?]
                              (one entry means "this creature does not block")
      CHOOSE_TARGET:          [player_id or permanent_id]
      SCRY_* / SELECT_CARD:   [card_id] (library cards revealed by the
                              pending decision are in the observation)
      TAP_FOR_COST:           [permanent_id]
      DECLINE/PAY/MODE:       [resolving or cast card_id]
      PASS_PRIORITY:          []
    """
    action_name = enum_name(ActionEnum, action.action_type)
    if declared is None:
        declared = getattr(action, "declared", None)
    focus_names = [names.get(int(value)) for value in action.focus]
    first = focus_names[0] if focus_names else None

    if action_name == "PRIORITY_PASS_PRIORITY":
        return "Pass priority"
    if action_name == "PRIORITY_PLAY_LAND" and first:
        return f"Play {first}"
    if action_name == "PRIORITY_CAST_SPELL" and first:
        return f"Cast {first}"
    if action_name == "DECLARE_ATTACKER" and first:
        return (
            f"Do not attack with {first}"
            if declared is False
            else f"Attack with {first}"
        )
    if action_name == "DECLARE_BLOCKER" and first:
        if len(focus_names) >= 2 and focus_names[1]:
            return f"Block {focus_names[1]} with {first}"
        return f"{first}: do not block"
    if action_name == "CHOOSE_TARGET" and first:
        return f"Target {first}"
    # Mid-resolution decision actions (scry / look-and-select / learn /
    # pay-or-not / modal / waterbend) in Magic terms.
    if action_name == "SCRY_KEEP" and first:
        return f"Keep {first} on top"
    if action_name == "SCRY_BOTTOM" and first:
        return f"Put {first} on the bottom"
    if action_name == "LEARN_TAKE_LESSON" and first:
        return f"Take {first}"
    if action_name == "LEARN_DISCARD" and first:
        return f"Discard {first}, then draw a card"
    if action_name == "SELECT_CARD" and first:
        if space_kind == "DISCARD":
            return f"Discard {first}"
        if space_kind == "LEGEND_RULE":
            return f"Keep {first} (legend rule)"
        return f"Put {first} into your hand"
    if action_name == "DECLINE_CHOICE":
        if space_kind == "LEARN":
            return "Decline Learn"
        return "Decline"
    if action_name == "PAY_COST":
        return f"Pay the cost ({first})" if first else "Pay the cost"
    if action_name == "CHOOSE_MODE":
        return f"Choose a mode ({first})" if first else "Choose a mode"
    if action_name == "TAP_FOR_COST" and first:
        return f"Tap {first} to help pay"

    label = ACTION_LABELS.get(action_name, action_name)
    if first:
        return f"{label}: {first}"
    return label


def describe_actions(obs: managym.Observation) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    names = _build_id_to_name(obs)
    space_kind = enum_name(ActionSpaceEnum, int(obs.action_space.action_space_type))
    for index, action in enumerate(obs.action_space.actions):
        focus = [int(value) for value in action.focus]
        results.append(
            {
                "index": index,
                "type": enum_name(ActionEnum, action.action_type),
                "focus": focus,
                "declared": action.declared,
                "description": format_action(
                    action, names, space_kind, action.declared
                ),
            }
        )
    return results


def offer_verb(action_type: str) -> str:
    return {
        "PRIORITY_CAST_SPELL": "cast",
        "PRIORITY_PLAY_LAND": "play_land",
        "PRIORITY_ACTIVATE_ABILITY": "activate",
        "PRIORITY_PASS_PRIORITY": "pass_priority",
        "DECLARE_ATTACKER": "declare_attackers",
        "DECLARE_BLOCKER": "declare_blockers",
        "PAY_COST": "pay",
    }.get(action_type, "choose")


def _redact_hand(player_state: dict[str, Any]) -> None:
    hand = player_state.get("hand")
    if not isinstance(hand, list):
        return
    zone_counts = player_state.get("zone_counts")
    if isinstance(zone_counts, dict) and "HAND" in zone_counts:
        player_state["hand_hidden_count"] = int(zone_counts["HAND"])
    else:
        player_state["hand_hidden_count"] = len(hand)
    player_state["hand"] = []


def redact_observation(observation: dict[str, Any]) -> None:
    """Strip the opponent's hidden information from a serialized observation.

    Mutates in place. Libraries are never serialized (only counts), so the
    opponent's hand is the only hidden zone to remove.
    """
    if not isinstance(observation, dict):
        return

    opponent = observation.get("opponent")
    if isinstance(opponent, dict):
        _redact_hand(opponent)
        opponent.pop("sideboard", None)
