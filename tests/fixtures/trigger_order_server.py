"""Release-browser ASGI fixture; production Academy has no state-injection route."""

from collections.abc import Sequence
from typing import Any

from manabot.verify.text_observation import keep_opening_hands
from manabot.verify.trigger_order import DECK, SEED, configure_root
import managym
from managym.academy import server
from managym.academy.opponent import policy_registry
from managym.academy.trace import GameConfig
from managym.policy import PolicyDecision, PolicyDescriptor, PolicyView

_new_game = server.GameSession.new_game
_parse_game_config = server._parse_game_config


def parse_game_config(config: Any) -> GameConfig:
    # Fixed eight-card decks finish through native failed-draw rules. Only the
    # dedicated timeline proof requests this name; submitted decks stay unchanged.
    if isinstance(config, dict) and config.get("hero_deck") == "watch-timeline-fixture":
        config = {**config, "hero_deck": {"Island": 8}, "villain_deck": {"Mountain": 8}}
    return _parse_game_config(config)


server._parse_game_config = parse_game_config


def new_game(
    self: server.GameSession,
    raw_config: Any = None,
    *,
    continuation_fingerprint: str | None = None,
) -> dict[str, Any]:
    # Only the dedicated browser proof opts in; other release cases use the
    # ordinary server unchanged. Any is confined to the existing JSON boundary.
    if isinstance(raw_config, dict) and raw_config.get("payment_fixture") is True:
        _new_game(
            self,
            {
                "hero_deck": {
                    "Lightning Strike": 4,
                    "White Lotus Hideout": 4,
                    "Mountain": 20,
                    "Island": 12,
                    "Prophetic Prism": 4,
                },
                "villain_deck": {"Plains": 40},
                "villain_type": "passive",
                "seed": 165,
                "auto_pass": False,
                # The proof floats restricted mana from a choice-output land
                # at priority, which a default (off) seat is never offered.
                "mana_floats": "usable",
            },
        )
        assert self.env is not None and self.obs is not None
        keep_opening_hands(self.env, self.obs)
        self.env.scenario_set_life(1, 3)
        self.env.scenario_clear_hand(0)
        self.env.scenario_force_card_in_hand(0, "Lightning Strike")
        for name in (
            "Island",
            "Mountain",
            "Mountain",
            "Island",
            "Island",
            "Mountain",
            "Mountain",
            "Prophetic Prism",
            "White Lotus Hideout",
        ):
            self.env.scenario_force_battlefield(0, name, ready=True)
        self.obs = self.env.scenario_refresh()
        self.published_prompt = None
        self.presentation.reset()
        return self._wire_message()
    fixture = (
        isinstance(raw_config, dict) and raw_config.get("trigger_order_fixture") is True
    )
    if not fixture:
        return _new_game(
            self, raw_config, continuation_fingerprint=continuation_fingerprint
        )
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


# Registered, action-only policies for repeated-session browser coverage. These
# identities describe workflow fixtures and make no trained-strength claim.


class SessionSmokePolicy:
    def observe(self, view: PolicyView) -> None:
        pass

    def choose(self, view: PolicyView) -> PolicyDecision:
        choice = next(
            (
                str(index)
                for index, action in enumerate(view.observation.action_space.actions)
                if int(action.action_type)
                == int(managym.ActionEnum.PRIORITY_PASS_PRIORITY)
            ),
            view.choices[0],
        )
        return PolicyDecision(choice, view.binding)

    def close(self) -> None:
        pass


class SessionSmokeFactory:
    def __init__(self, name: str) -> None:
        self.name = name

    def describe(self) -> PolicyDescriptor:
        return PolicyDescriptor(
            self.name,
            "release-session-fixture",
            "pass-priority-1",
            config={"fixture": self.name},
        )

    def admit(self, setup: Sequence[managym.PlayerConfig]) -> None:
        if len(setup) != 2:
            raise ValueError("Two native seats required.")

    def load(self, *, seat: int, seed: int) -> SessionSmokePolicy:
        return SessionSmokePolicy()


for _name in ("Session smoke A", "Session smoke B"):
    policy_registry().register(SessionSmokeFactory(_name))
