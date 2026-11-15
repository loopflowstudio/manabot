"""Real Play/Watch proof with independent and untrained baseline search policies."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import threading
from types import FrameType

from managym.academy import server
from managym.academy.launch import serve
from managym.academy.opponent import policy_registry
from managym.academy.trace import load_trace
from managym.bots import passive_policy
from managym.mirror_search import SearchSpec
from managym.policy import PolicySpec
from tests.academy.test_mirror_baseline import write_search_checkpoint
from tests.academy.test_mirror_play import SearchFactory
from tests.academy.test_policy_runtime import IndependentFactory


def _finished_play(key: str) -> str:
    game = server.GameSession()
    try:
        game.new_game(
            {
                "hero_deck": {"Mountain": 8, "Gray Ogre": 8},
                "villain_deck": {"Forest": 8, "Llanowar Elves": 8},
                "villain_type": "policy",
                "opponent_sha256": key,
                "search": [None, asdict(SearchSpec(worlds=1, depth=1))],
                "seed": 42,
                "auto_pass": False,
            }
        )
        for _ in range(200):
            assert game.env is not None and game.obs is not None
            if game.obs.game_over:
                return game.match_id
            game.hero_action(passive_policy(game.env, game.obs))
        raise RuntimeError("Fixture game did not finish")
    finally:
        game.close("proof")


def main() -> None:
    output = Path(os.environ["ACADEMY_MIRROR_PROOF"])
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = output.parent / "untrained-smoke.pt"
    write_search_checkpoint(checkpoint)
    shutdown = threading.Event()

    def request_shutdown(signum: int, frame: FrameType | None) -> None:
        shutdown.set()

    # A supervisor may forward SIGINT twice. Keep later signals from interrupting
    # retained-record persistence while the context closes the real server.
    signal.signal(signal.SIGINT, request_shutdown)
    with serve(
        port=int(os.environ["ACADEMY_MIRROR_PORT"]),
        traces_dir=output.parent / "traces",
        policies={
            "independent": SearchFactory(),
            "baseline": PolicySpec("manabot", str(checkpoint)),
            "action-only": IndependentFactory("Action-only"),
        },
    ) as academy:
        spec = SearchSpec(worlds=1, depth=1)
        timeout = SearchSpec(worlds=1, depth=1, decision_seconds=1e-9)
        saved = _finished_play(policy_registry().descriptor("independent").id)
        trace = load_trace(saved)
        ordinal = next(
            index
            for index, event in enumerate(trace["events"])
            if event.get("bot_decision")
        )
        output.write_text(
            json.dumps(
                {
                    "play": academy.new_game(
                        opponent="baseline", search=spec, seed=7
                    ).url,
                    "timeout": academy.new_game(
                        opponent="baseline", search=timeout, seed=7
                    ).url,
                    "watch": academy.watch(
                        players=("independent", "baseline"), search=(spec, None)
                    ).url,
                    "watch_timeout": academy.watch(
                        players=("independent", "baseline"), search=(timeout, None)
                    ).url,
                    "saved": f"{academy.url}/replay?trace={saved}",
                    "saved_ordinal": ordinal,
                    "ids": {
                        name: policy_registry().descriptor(name).id
                        for name in ("independent", "baseline", "action-only")
                    },
                }
            )
        )
        print("ACADEMY_MIRROR_READY", flush=True)
        shutdown.wait()


if __name__ == "__main__":
    main()
