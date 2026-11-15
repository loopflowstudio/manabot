"""Release-browser ASGI fixture; production Academy has no state-injection route."""

from collections.abc import Sequence
from typing import Any

from conformance.combat_planning import combat_config, configure_combat
from manabot.verify.trigger_order import DECK, SEED, configure_root
import managym
from managym.academy import server
from managym.academy.opponent import policy_registry
from managym.policy import PolicyDecision, PolicyDescriptor, PolicyView

_new_game = server.GameSession.new_game


def new_game(
    self: server.GameSession,
    raw_config: Any = None,
    *,
    continuation_fingerprint: str | None = None,
) -> dict[str, Any]:
    # Only the dedicated browser proof opts in; other release cases use the
    # ordinary server unchanged. Any is confined to the existing JSON boundary.
    fixture = (
        isinstance(raw_config, dict) and raw_config.get("trigger_order_fixture") is True
    )
    combat = raw_config.get("combat_fixture") if isinstance(raw_config, dict) else None
    if combat in ("attackers", "blockers"):
        _new_game(self, combat_config())
        assert self.env is not None
        self.obs = configure_combat(
            self.env, combat, count=24 if raw_config.get("dense") else 3
        )
        self.published_prompt = None
        self.presentation.reset()
        return self._wire_message()
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
