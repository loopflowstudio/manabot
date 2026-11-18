"""Printed card faces by name, for the table's card frames.

`card_faces(names)` answers, for each requested name the engine defines, what a
printed card shows besides its name, art and rules text: the mana cost in
symbols, its colours, the type line, the printed keywords and printed
power/toughness. Every
value comes from the engine's own card definitions
(`managym.card_definitions_json`); nothing here is game state. A permanent's
current power, toughness and continuous effects stay in the frame's projection,
and the browser asks only for names a viewer already sees, as with card art.
"""

from __future__ import annotations

from functools import cache
import json
from typing import Any, TypedDict

import managym

# `ManaCost.cost` is indexed by the engine's `Color`: W, U, B, R, G, C, generic.
COLOR_SYMBOLS = ("W", "U", "B", "R", "G", "C")
GENERIC = 6
HYBRID_SYMBOLS = {"White": "W", "Blue": "U", "Black": "B", "Red": "R", "Green": "G"}
COLOR_NAMES = ("White", "Blue", "Black", "Red", "Green")


class CardFace(TypedDict):
    """What a printed face shows besides its name, art and rules text."""

    cost: str
    # Its colours as letters in printed order (`["G", "W"]`): a colour
    # indicator or token colour when the definition sets one, else the cost's.
    # Empty for colourless cards and lands.
    colors: list[str]
    type_line: str
    keywords: list[str]
    power: int | None
    toughness: int | None


def cost_symbols(mana_cost: dict[str, Any] | None) -> str:
    """`{X}{2}{W/U}{R}`: X, generic, hybrid, colored symbols in printed order, then C."""
    if not mana_cost:
        return ""
    cost: list[int] = mana_cost["cost"]
    symbols = ["{X}"] * int(mana_cost.get("x", 0))
    if cost[GENERIC]:
        symbols.append(f"{{{cost[GENERIC]}}}")
    symbols += [
        f"{{{HYBRID_SYMBOLS[left]}/{HYBRID_SYMBOLS[right]}}}"
        for left, right in mana_cost.get("hybrid", [])
    ]
    for color in printed_order([index for index in range(5) if cost[index]]):
        symbols += [f"{{{COLOR_SYMBOLS[color]}}}"] * cost[color]
    symbols += ["{C}"] * cost[5]
    if not symbols and mana_cost.get("mana_value") == 0:
        return "{0}"
    return "".join(symbols)


def printed_order(colors: list[int]) -> list[int]:
    """Colors in printed order: around the WUBRG wheel from the start that
    covers them in the shortest arc, so {G}{W}, {W}{B} and {R}{W}."""
    if len(colors) < 2:
        return colors
    starts = [(max((color - start) % 5 for color in colors), start) for start in colors]
    _, start = min(starts)
    return sorted(colors, key=lambda color: (color - start) % 5)


def colors(definition: dict[str, Any]) -> list[str]:
    """The printed colours (CR 105.2, 202.2): an override, else the cost's symbols."""
    override = definition.get("color_override")
    if override is not None:
        found = {COLOR_NAMES.index(name) for name in override}
    else:
        mana_cost = definition.get("mana_cost") or {"cost": [0] * 7}
        found = {index for index in range(5) if mana_cost["cost"][index]}
        for pair in mana_cost.get("hybrid", []):
            found |= {COLOR_NAMES.index(name) for name in pair}
    return [COLOR_SYMBOLS[index] for index in printed_order(sorted(found))]


def type_line(definition: dict[str, Any]) -> str:
    front = " ".join(
        [s.capitalize() for s in definition["supertypes"]]
        + list(definition["types"]["types"])
    )
    if definition.get("is_token"):
        front = f"Token {front}"
    subtypes = " ".join(definition["subtypes"])
    return f"{front} — {subtypes}" if subtypes else front


@cache
def faces() -> dict[str, CardFace]:
    """Every defined face by printed name; a card wins over a token of its name."""
    out: dict[str, CardFace] = {}
    definitions = sorted(
        json.loads(managym.card_definitions_json()), key=lambda d: d["is_token"]
    )
    for definition in definitions:
        out.setdefault(
            definition["name"],
            CardFace(
                cost=cost_symbols(definition.get("mana_cost")),
                colors=colors(definition),
                type_line=type_line(definition),
                keywords=list(definition["keywords"]),
                power=definition.get("power"),
                toughness=definition.get("toughness"),
            ),
        )
    return out


def card_faces(names: list[str]) -> dict[str, CardFace]:
    known = faces()
    return {name: known[name] for name in names if name in known}
