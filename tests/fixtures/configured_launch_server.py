"""Browser proof using the public launch API, with no production injection route."""

import json
import os
from pathlib import Path

from managym.academy.launch import serve
from scripts.release.academy_replay_smoke import write_example_trace


def main() -> None:
    output = Path(os.environ["ACADEMY_LAUNCH_PROOF"])
    output.parent.mkdir(parents=True, exist_ok=True)
    source = output.parent / "example.commands.jsonl.gz"
    write_example_trace(source)
    with serve(
        port=int(os.environ["ACADEMY_LAUNCH_PORT"]),
        traces_dir=output.parent / "traces",
    ) as academy:
        game = academy.new_game(
            deck="gw_allies_jeong",
            opponent_deck="ur_lessons",
            opponent="passive",
            seed=7,
        )
        deck_file = output.parent / "my-deck.txt"
        deck_file.write_text("20 Mountain\n")
        imported_game = academy.new_game(
            deck="ur_lessons",
            opponent_deck=deck_file,
            opponent="passive",
            seed=7,
        )
        replay = academy.replay(source)
        output.write_text(
            json.dumps(
                {
                    "game": game.url,
                    "imported_game": imported_game.url,
                    "replay": replay.url,
                }
            )
        )
        print("ACADEMY_LAUNCH_PROOF_READY", flush=True)
        academy.wait()


if __name__ == "__main__":
    main()
