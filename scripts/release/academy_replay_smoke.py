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
from typing import Any
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

from manabot.infra.config import MatchConfig, ObservationConfig
from manabot.infra.identity import canonical_sha256
from manabot.infra.provenance import RECORDED_ASSET_MANIFEST_HASH
from manabot.sim.frames import build_command, build_viewer_frame
from manabot.sim.replay import replay_environment, write_trace
from managym.academy.launch import serve
from managym.experience import UNVERSIONED_CONTENT_HASH


def write_example_trace(path: Path) -> None:
    """Write one complete evaluation-format game with explicit policy input."""
    game: dict[str, Any] = {
        "match_id": "academy-release-proof",
        "deal_seed": 7,
        "leg": 0,
        "seat_players": ["first-legal-a", "first-legal-b"],
        "match_hypers": MatchConfig(
            hero="first-legal-a",
            villain="first-legal-b",
            hero_deck={"Mountain": 1, "Gray Ogre": 1},
            villain_deck={"Forest": 1, "Llanowar Elves": 1},
        ).model_dump(),
        "observation_hypers": ObservationConfig().model_dump(),
        "winner": None,
        "terminated": False,
        "truncated": False,
        "failure": None,
        "decisions": [],
    }
    env, _ = replay_environment(game)
    try:
        for revision in range(100):
            frame = build_viewer_frame(
                env.last_raw_obs,
                match_id=game["match_id"],
                revision=revision,
                content_hash=UNVERSIONED_CONTENT_HASH,
                asset_manifest_hash=RECORDED_ASSET_MANIFEST_HASH,
            )
            offer = frame["offers"][0]
            command = build_command(frame, offer["id"])
            actor = int(env.last_raw_obs.agent.player_index)
            before = env._engine.state_digest()
            _, _, terminated, truncated, _ = env.step(offer["id"])
            game["decisions"].append(
                {
                    "actor": actor,
                    "player_id": game["seat_players"][actor],
                    "frame_sha256": canonical_sha256(frame),
                    "command": command,
                    "command_sha256": canonical_sha256(command),
                    "chosen_offer": offer,
                    "action_space_kind": frame["action_space"],
                    "pre_state_digest": before,
                    "post_state_digest": env._engine.state_digest(),
                }
            )
            assert not truncated
            if terminated:
                game["winner"] = env._engine.winner_index()
                game["terminated"] = True
                break
        else:
            raise AssertionError("Example game exceeded its decision bound")
    finally:
        env.close()
    game["game_trace_sha256"] = canonical_sha256(game)
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
        with serve(port=0, traces_dir=root / "python", open_browser=False) as academy:
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
