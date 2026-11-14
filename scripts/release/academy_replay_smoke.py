"""Prove configured Python and CLI replay launches from the installed pair.

A tiny complete native game supplies fresh evaluation-format Commands; no policy
or checkpoint is loaded. The same fixture feeds the configured browser proof.
"""

import json
from pathlib import Path
from queue import Queue
import signal
import subprocess
import sys
import tempfile
from threading import Thread
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

from manabot.env.failures import GameInputs, PlayerSetup
from manabot.sim.recording import record_actions
from manabot.sim.replay import write_trace
from managym.academy.launch import serve


def write_example_trace(path: Path) -> None:
    """Write one complete evaluation-format game with explicit policy input."""
    setup = GameInputs(
        seed=7,
        players=[
            PlayerSetup(
                name="first-legal-a",
                decklist={"Mountain": 1, "Gray Ogre": 1},
                sideboard={},
            ),
            PlayerSetup(
                name="first-legal-b",
                decklist={"Forest": 1, "Llanowar Elves": 1},
                sideboard={},
            ),
        ],
    )
    # Tiny decks lose to the first draw after their opening hand.
    game = record_actions(
        setup,
        [0],
        players=("first-legal-a", "first-legal-b"),
        source={"kind": "evaluation"},
    )
    write_trace(path, [game])


def _check_replay(url: str) -> None:
    parsed = urlparse(url)
    trace_id = parse_qs(parsed.query)["trace"][0]
    with urlopen(
        f"{parsed.scheme}://{parsed.netloc}/api/watch/{trace_id}/decisions/0",
        timeout=15,
    ) as response:
        decision = json.load(response)
    assert decision["count"] > 0
    assert decision["decision"]["ordinal"] == 0


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="academy-replay-") as directory:
        root = Path(directory)
        source = root / "game.commands.jsonl.gz"
        write_example_trace(source)
        with serve(port=0, traces_dir=root / "python") as academy:
            assert "launch=" in academy.new_game(opponent="passive", seed=7).url
            replay = academy.replay(source)
            _check_replay(replay.url)
            assert academy.replay(source) == replay
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "managym.academy",
                "replay",
                str(source),
                "--port",
                "0",
                "--traces-dir",
                str(root / "cli"),
                "--no-open",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        ready: Queue[str] = Queue()
        log: list[str] = []

        def output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                log.append(line)
                if line.startswith("ACADEMY_PLAY_READY "):
                    ready.put(line)

        reader = Thread(target=output, daemon=True)
        reader.start()
        try:
            record = json.loads(
                ready.get(timeout=30).removeprefix("ACADEMY_PLAY_READY ")
            )
            _check_replay(record["open"])
        finally:
            process.send_signal(signal.SIGINT)
            try:
                assert process.wait(timeout=15) == 0, "".join(log)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise
            reader.join(timeout=5)
    print("ACADEMY_REPLAY_PROOF passed")


if __name__ == "__main__":
    main()
