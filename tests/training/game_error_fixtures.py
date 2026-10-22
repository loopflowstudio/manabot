"""One deterministic payment defect for capture, replay and repaired continuation.

These adapters use native Mountain/Gray Ogre games. Their faulty/repaired payment
validator is fixture code, not a reproduction or repair of MTG-175. Keep this
module independent of remote test owners so bounded replay needs only game inputs.
"""

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from manabot.infra.config import MatchConfig
from manabot.sim.net_opponent import SeatRoutedCollector
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


class PaymentMismatchVector:
    def __init__(self, collector: SeatRoutedCollector) -> None:
        self.native = collector._env
        self.configs = [match.to_rust() for match in collector.stream_matches]
        self.shadows = [self._new_shadow(row) for row in range(len(self.configs))]
        self.infos: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.native, name)

    def _new_shadow(self, row: int) -> PaymentMismatchEnv:
        shadow = PaymentMismatchEnv(
            seed=self.native.current_game_seeds()[row], skip_trivial=True
        )
        shadow.reset(self.configs[row])
        return shadow

    def reset_seeds_into_buffers(self, seeds: list[int]) -> None:
        self.native.reset_seeds_into_buffers(seeds)
        self.shadows = [self._new_shadow(row) for row in range(len(self.configs))]

    def step_into_buffers(self, actions: list[int], active: list[bool]) -> None:
        self.native.step_into_buffers(actions, active)
        for row, enabled in enumerate(active):
            if enabled:
                self.shadows[row].step(actions[row])

    def step_isolated_into_buffers(
        self, actions: list[int], active: list[bool]
    ) -> None:
        enabled = list(active)
        failures: dict[int, dict[str, Any]] = {}
        done_rows: list[int] = []
        for row, shadow in enumerate(self.shadows):
            if not active[row]:
                continue
            frame = shadow.semantic_decision_frame_json()
            try:
                _, _, done, truncated, _ = shadow.step(actions[row])
                if done or truncated:
                    done_rows.append(row)
            except managym.AgentError as error:
                enabled[row] = False
                failures[row] = {
                    "game_error": str(error),
                    "game_error_class": type(error).__name__,
                    "game_error_frame": frame,
                    "game_error_trace": "deterministic offered-cast payment fixture",
                }
        self.native.step_isolated_into_buffers(actions, enabled)
        self.infos = self.native.get_last_info()
        for row, failure in failures.items():
            self.infos[row] = failure
        for row in done_rows:
            self.shadows[row] = self._new_shadow(row)

    def get_last_info(self) -> list[dict[str, Any]]:
        return self.infos

    def replace_failed_into_buffers(self, failed: list[bool]) -> None:
        self.native.replace_failed_into_buffers(failed)
        for row, replace in enumerate(failed):
            if replace:
                self.shadows[row] = self._new_shadow(row)


class PaymentMismatchCollector(SeatRoutedCollector):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._env = PaymentMismatchVector(self)


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
