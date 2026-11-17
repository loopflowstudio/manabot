"""Constructed live Academy table shared by WebSocket and browser proofs."""

from typing import Any

from fastapi import FastAPI

from managym.academy import server

new_game = server.GameSession.new_game


def mana_game(
    self: server.GameSession,
    raw_config: Any,
    *,
    continuation_fingerprint: str | None = None,
) -> dict[str, Any]:
    if not isinstance(raw_config, dict) or not raw_config.get("mana_controls_fixture"):
        return new_game(
            self, raw_config, continuation_fingerprint=continuation_fingerprint
        )
    new_game(
        self,
        {
            "hero_deck": {
                "Forest": 12,
                "Mountain": 20,
                "Jasmine Dragon Tea Shop": 4,
                "Gray Ogre": 4,
            },
            "villain_deck": {"Plains": 40},
            "villain_type": "passive",
            "auto_pass": False,
            "seed": 165,
        },
    )
    assert self.env is not None
    self.env.scenario_clear_hand(0)
    self.env.scenario_force_card_in_hand(0, "Gray Ogre")
    self.env.scenario_force_card_in_hand(0, "Mountain")
    for name in ("Forest", "Mountain", "Jasmine Dragon Tea Shop"):
        self.env.scenario_force_battlefield(0, name, ready=True)
    self.env.scenario_create_token(0, "token.treasure")
    self.obs = self.env.scenario_refresh()
    self.published_prompt = None
    # Resolve the token's entry events before the first reversible tap.
    setup = self._wire_message()
    land = next(a for a in setup["actions"] if a["description"] == "Play Mountain")
    return self.hero_action(land["index"])


def create_app() -> FastAPI:
    server.GameSession.new_game = mana_game
    return server.app
