"""Controlled parent-side fixtures behind Etude's real game process boundary.

Persistence fixtures keep injected outcomes and counters in their pytest process.
The game process calls this importable bridge rather than importing a closure.
"""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from multiprocessing.connection import Client, Listener
import os
from pathlib import Path
import tempfile
from threading import Thread

from etude.eval.records import Match
from etude.eval.result import EvalResult, Result, Tables
from etude.eval.run import EvalRun
from etude.jobs import Agents
import pytest


def play_callback(match: Match, agents: Sequence[object]) -> Tables | Result:
    """Run the fixture in its owning process and retain SDK execution in a child."""
    with Client(os.environ["MANABOT_TEST_CALLBACK"], authkey=b"fixture") as connection:
        connection.send(match)
        result, error = connection.recv()
    if error is not None:
        raise error
    return result


@contextmanager
def callback_bridge(
    callback: Callable[..., Tables | Result],
) -> Iterator[None]:
    with tempfile.TemporaryDirectory(prefix="eval-fixture-") as folder:
        address = str(Path(folder) / "callback.sock")
        with Listener(address, backlog=128, authkey=b"fixture") as listener:

            def serve() -> None:
                while True:
                    with listener.accept() as connection:
                        try:
                            request = connection.recv()
                        except EOFError:
                            continue
                        if request is None:
                            return
                        try:
                            result = callback(request, ())
                        except Exception as error:
                            connection.send((None, error))
                        else:
                            connection.send((result, None))

            thread = Thread(target=serve, daemon=True)
            thread.start()
            previous = os.environ.get("MANABOT_TEST_CALLBACK")
            os.environ["MANABOT_TEST_CALLBACK"] = address
            try:
                yield
            finally:
                if previous is None:
                    os.environ.pop("MANABOT_TEST_CALLBACK", None)
                else:
                    os.environ["MANABOT_TEST_CALLBACK"] = previous
                with Client(address, authkey=b"fixture") as connection:
                    connection.send(None)
                thread.join()


def bridge_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a local fixture's injected outcomes while using Etude's real scheduler."""
    original = EvalRun.play

    def play(
        run: EvalRun,
        callback: Callable[..., Tables | Result],
        *,
        agents: Agents | None = None,
    ) -> EvalResult:
        if "<locals>" not in callback.__qualname__:
            return original(run, callback, agents=agents)
        with callback_bridge(callback):
            return original(run, play_callback, agents=agents)

    monkeypatch.setattr(EvalRun, "play", play)


class PanicException(BaseException):
    """Controlled PyO3-shaped panic for exercising BaseException retention."""


def play_native_case(match: Match, agents: Sequence[object]) -> Result:
    """Run real native games with one injected player failure or clock expiry."""
    import sys
    import time
    from types import SimpleNamespace
    from typing import cast
    from unittest.mock import patch

    from manabot.eval.card_evaluation import verification_spec
    from manabot.eval.match import play_cell
    from manabot.sim.agent import GamePlayer
    import managym

    assert "torch" not in sys.modules, "the Etude game callback imported Torch"
    spec = verification_spec(8)
    case = str(match.config["case"])
    first = cast(GamePlayer, agents[0])

    class Injected:
        space = first.space
        called = False

        def admit(self, match: object) -> None:
            first.admit(match)

        def choose(self, env: object, obs: object, seed: int) -> int:
            if not self.called:
                self.called = True
                if case == "agent":
                    raise managym.AgentError("controlled agent failure")
                if case == "panic":
                    raise PanicException("controlled native panic")
                if case == "retry":
                    from etude.errors import RetryRequested

                    raise RetryRequested("controlled agent transport failure")
            return first.choose(env, obs, seed)

    built = {
        spec.players[0].player_id: Injected(),
        spec.players[1].player_id: cast(GamePlayer, agents[1]),
    }
    root = Path(os.environ["ETUDE_OUTPUT_DIR"]) / match.key
    clock = SimpleNamespace(
        monotonic=lambda: float("inf"), perf_counter=time.perf_counter
    )
    with (
        patch("manabot.sim.native_game.time", clock)
        if case == "timeout"
        else nullcontext()
    ):
        rows, _, _ = play_cell(
            key=spec.key,
            player_a=spec.players[0],
            player_b=spec.players[1],
            deal_seeds=spec.deals,
            out_dir=root,
            game_seconds=spec.game_seconds,
            max_commands=1 if case == "commands" else spec.max_commands,
            authored_matchup=spec.authored_matchup,
            matchup_mode="all",
            agents=built,
        )
    assert "torch" not in sys.modules, "native game execution imported Torch"
    return Result(
        tables={
            "scores": [
                {
                    "seat": 0,
                    "game": row["leg"],
                    "score": row["score_a"],
                    "failure": row["failure"],
                    "termination": row["termination_reason"],
                }
                for row in rows
            ]
        }
    )
