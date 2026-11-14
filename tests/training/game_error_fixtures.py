"""One deterministic payment defect for capture, replay and repaired continuation.

These adapters use native Mountain/Gray Ogre games. Their faulty/repaired payment
validator is fixture code, not a reproduction or repair of MTG-175. Keep this
module independent of remote test owners so bounded replay needs only game inputs.

The `*_worker` functions are `ChildProcess` game-worker targets. Their children
import this module under the game's wall-clock budget, so keep it Torch-free;
collector adapters belong with their training tests.
"""

import json
import multiprocessing.connection
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from manabot.infra.config import MatchConfig
from manabot.sim import native_game
from manabot.sim.registration import PlayerRegistration
import managym

_NATIVE_ENV = managym.Env


def _payment_match() -> MatchConfig:
    # A narrow native fixture: every mana source is Mountain, every spell is 2R.
    return MatchConfig(
        hero_deck={"Mountain": 24, "Gray Ogre": 36},
        villain_deck={"Mountain": 24, "Gray Ogre": 36},
    )


def _broken_payment(red: int) -> bool:
    # Fault: consume every available R when satisfying the one colored symbol,
    # leaving nothing for the two generic mana. Native offers remain correct.
    return red >= 1 and red - red >= 2


def _repaired_payment(red: int) -> bool:
    # Spend just one R; the remainder can pay generic mana.
    return red >= 1 and red - 1 >= 2


class PaymentMismatchEnv:
    payment = staticmethod(_broken_payment)

    def __init__(self, seed: int = 0, **kwargs: Any) -> None:
        self.seed = seed
        self.native = _NATIVE_ENV(seed=seed, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # Test adapter forwards the native extension's untyped interface.
        return getattr(self.native, name)

    def set_seed(self, seed: int) -> None:
        self.seed = seed
        self.native.set_seed(seed)

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        frame = json.loads(self.native.semantic_decision_frame_json())
        offer = frame["offers"][action]
        if (
            self.seed % 4 == 0
            and offer.get("public_commitment", {}).get("card") == "Gray Ogre"
            and offer["verb"] == "cast"
        ):
            raw = self.native.observation_for_player(frame["actor"])
            mountains = {
                card.registry_key for card in raw.agent_cards if card.name == "Mountain"
            }
            red = sum(raw.agent.mana) + sum(
                not permanent.tapped and permanent.registry_key in mountains
                for permanent in raw.agent_permanents
            )
            assert _repaired_payment(red), (
                "fixture must select an actually payable offer"
            )
            if not self.payment(red):
                raise managym.AgentError(
                    f"fixture payment mismatch: not enough producible mana (R={red}; cost=2R)"
                )
        return self.native.step(action)


def _payment_game_worker(
    connection: multiprocessing.connection.Connection,
    game: dict[str, Any],
    registrations: list[PlayerRegistration],
    checkpoint_paths: dict[str, str],
    max_commands: int,
    inference: int | None,
    deadline_unix: float,
) -> None:
    managym.Env = PaymentMismatchEnv
    native_game.game_worker(
        connection,
        game,
        registrations,
        checkpoint_paths,
        max_commands,
        inference,
        deadline_unix,
    )


def _lost_payment_worker(
    connection: multiprocessing.connection.Connection,
    game: dict[str, Any],
    registrations: list[PlayerRegistration],
    checkpoint_paths: dict[str, str],
    max_commands: int,
    inference: int | None,
    deadline_unix: float,
) -> None:
    """Die or hang only after the parent acknowledges retained native inputs."""

    def send(event: tuple[str, Any]) -> None:
        connection.send(event)
        if event[0] == "pending":
            assert connection.recv() == "retained"
            if game.get("test_hang"):
                time.sleep(60)
            os._exit(7)

    native_game.execute_game(game, registrations, checkpoint_paths, max_commands, send)


def _bounded_fixture_replay(
    module: str, path: Path, selector: str, index: int, *, repaired: bool = False
) -> dict[str, Any]:
    script = """
import importlib, runpy, sys
import managym
fixture = runpy.run_path(sys.argv[1])
managym.Env = fixture['PaymentMismatchEnv']
if sys.argv[6] == 'repaired':
    managym.Env.payment = staticmethod(fixture['_repaired_payment'])
main = importlib.import_module(sys.argv[2]).main
sys.argv = ['replay', sys.argv[3], sys.argv[4], sys.argv[5], '--worker']
main()
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            __file__,
            module,
            str(path),
            selector,
            str(index),
            "repaired" if repaired else "original",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)
