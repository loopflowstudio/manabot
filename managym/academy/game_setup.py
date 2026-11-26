"""The setup a game was started from, as a request that starts it again.

`setup_draft` reads it from a game's recorded config; `shown_setup` is the
part of it the person playing may see; `continuation_config` applies a
rematch's seat swap and edits to it.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from managym.academy.trace import GameConfig


def setup_draft(config: GameConfig) -> dict[str, Any]:
    """Re-admittable setup, without a resolved random seed or player credentials."""
    draft: dict[str, Any] = {
        "hero_deck": config.hero_deck_name
        if config.hero_deck_name != "custom"
        else config.hero_deck,
        "villain_deck": config.villain_deck_name
        if config.villain_deck_name != "custom"
        else config.villain_deck,
        "villain_type": config.villain_type,
        "villain_sims": config.villain_sims,
        "human_seat": config.human_seat,
        "mana_floats": config.mana_floats,
        "mana_payment": config.mana_payment,
        "decklists": config.decklists,
        "stops": config.stops,
        "auto_pass": config.auto_pass,
        "stop_on_stack": config.stop_on_stack,
        "seed": str(config.seed) if config.seed_was_requested else None,
        "mirror": asdict(config.mirror) if config.mirror else None,
    }
    if config.opponent and "sha256" in config.opponent:
        draft["opponent_sha256"] = config.opponent.get(
            "base_sha256", config.opponent["sha256"]
        )
    if config.hero_source is not None and config.villain_source is not None:
        draft["deck_import"] = {
            side: {
                "source": source["source"],
                "filename": source["filename"],
                **(
                    {"name": deck_name}
                    if deck_name not in ("imported", "custom")
                    else {}
                ),
            }
            for side, source, deck_name in (
                ("hero", config.hero_source, config.hero_deck_name),
                ("villain", config.villain_source, config.villain_deck_name),
            )
        }
    if config.hero_deck_pool is not None:
        draft["hero_deck_pool"] = config.hero_deck_pool
    if config.villain_deck_pool is not None:
        draft["villain_deck_pool"] = config.villain_deck_pool
    return draft


def shown_setup(config: GameConfig, game_over: bool) -> dict[str, Any]:
    """The setup as the browser gets it: what to select to start this matchup
    again, never the cards a selection resolved to.

    A side dealt from a deck pool is its pool key, without the dealt list, so
    starting it again deals again; a rematch reads the dealt decks from the
    stored game. While decklists are closed and the game is not over, the
    opponent's deck and imported list are left out (a pool it was dealt from
    stays)."""
    draft = setup_draft(config)
    imported = dict(draft.pop("deck_import", {}))
    for side in ("hero", "villain"):
        pool = draft.get(f"{side}_deck_pool")
        if pool is not None:
            draft[f"{side}_deck"] = pool
            imported.pop(side, None)
    if config.decklists == "closed" and not game_over:
        imported.pop("villain", None)
        if config.villain_deck_pool is None:
            del draft["villain_deck"]
    if imported:
        draft["deck_import"] = imported
    return draft


def continuation_config(
    draft: dict[str, Any],
    overrides: dict[str, Any],
    intent: Literal["rematch", "swap_sides"],
) -> dict[str, Any]:
    """Apply explicit edits once, after deriving seat changes from retained setup."""
    resolved = dict(draft)
    if intent == "swap_sides":
        resolved["human_seat"] = 1 - draft.get("human_seat", 0)
    if "hero_deck" in overrides or "villain_deck" in overrides:
        resolved.pop("deck_import", None)
        resolved.pop("hero_deck_pool", None)
        resolved.pop("villain_deck_pool", None)
    resolved.update(overrides)
    return resolved
