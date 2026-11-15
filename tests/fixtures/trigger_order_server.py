"""Release-browser ASGI fixture; production Academy has no state-injection route."""

from typing import Any

from manabot.verify.trigger_order import DECK, SEED, configure_root
from managym.academy import server

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


@app.get("/api/__fixtures/recorded-evaluation")
def recorded_evaluation() -> dict[str, Any]:
    """One verified source game for the release replay import proof."""
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from manabot.eval.match import play_cell
    from manabot.sim.replay import read_trace
    from tests.eval.arena_fixture import arena_fixture

    fixture = arena_fixture()
    with TemporaryDirectory(prefix="academy-replay-source-") as directory:
        _, trace, receipt = play_cell(
            key=fixture.key,
            player_a=fixture.anchors[0],
            player_b=fixture.anchors[1],
            deal_seeds=(777,),
            out_dir=Path(directory),
        )
        assert receipt["passed"]
        return read_trace(Path(trace["path"]))[0]
