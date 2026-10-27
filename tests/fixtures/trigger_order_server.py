"""Release-browser ASGI fixture; production Academy has no state-injection route."""

from typing import Any

from academy import server
from manabot.verify.trigger_order import DECK, SEED, configure_root

_new_game = server.GameSession.new_game


def new_game(self: server.GameSession, raw_config: Any = None) -> dict[str, Any]:
    # Only the dedicated browser proof opts in; other release cases use the
    # ordinary server unchanged. Any is confined to the existing JSON boundary.
    fixture = (
        isinstance(raw_config, dict) and raw_config.get("trigger_order_fixture") is True
    )
    if not fixture:
        return _new_game(self, raw_config)
    _new_game(
        self,
        {
            "hero_deck": DECK,
            "villain_deck": {"Swamp": 40},
            "villain_type": "passive",
            "seed": SEED,
            "auto_pass": False,
        },
    )
    assert self.env is not None
    self.obs = configure_root(self.env)
    self.published_prompt = None
    self.presentation.reset()
    return self._wire_message()


server.GameSession.new_game = new_game
app = server.app
