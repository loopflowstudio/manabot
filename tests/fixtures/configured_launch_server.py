"""Browser proof using the public launch API, with no production injection route."""

import base64
import json
import os
from pathlib import Path

import torch

from manabot.env.failures import CaptureScope, capture_scope
from manabot.infra.config import ModelConfig
from manabot.infra.run_settings import RunSettings
from manabot.training.loops.minimal import MinimalConfig, train
from managym.academy.card_art import printings
from managym.academy.launch import serve
from scripts.release.academy_replay_smoke import write_example_trace


def main() -> None:
    output = Path(os.environ["ACADEMY_LAUNCH_PROOF"])
    output.parent.mkdir(parents=True, exist_ok=True)
    source = output.parent / "example.commands.jsonl.gz"
    write_example_trace(source)
    # A real bounded self-play update supplies the producer-to-browser proof.
    torch.set_num_threads(1)
    # The identity `etude train` hands a local run, set for this in-process call.
    os.environ.update(
        ETUDE_OUT=str(output.parent / "training-run" / "out"),
        ETUDE_EXPERIMENT="academy-launch-proof",
        ETUDE_RUN="academy-launch-proof-seed-3",
        ETUDE_SEED="3",
        ETUDE_TRACK_DIR=str(output.parent / "training-run" / "track"),
    )
    training_run = output.parent / "training-games"
    with capture_scope(
        CaptureScope(
            training_run / "failures",
            settings=RunSettings(record_games=1),
        )
    ):
        train(
            MinimalConfig(
                updates=1,
                streams=2,
                decisions=16,
                model=ModelConfig(hidden_dim=8, num_attention_heads=2),
            )
        )
    training_source = training_run / "games"
    # Valid one-pixel JPEG: seeded cache, no Scryfall request from browser proofs.
    art_dir = output.parent / "art"
    art_dir.mkdir(exist_ok=True)
    jpeg = base64.b64decode(
        "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAABQb/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="
    )
    for printing in printings().values():
        (art_dir / f"{printing}-art.jpg").write_bytes(jpeg)
        (art_dir / f"{printing}-full.jpg").write_bytes(jpeg)
        (art_dir / f"{printing}.json").write_text(
            json.dumps(
                {
                    "artist": "Proof Artist",
                    "source": "https://scryfall.com",
                    "copyright": "© Wizards of the Coast LLC",
                }
            )
        )
    (art_dir / "back-back.jpg").write_bytes(jpeg)
    os.environ["ACADEMY_CARD_ART"] = "on"
    os.environ["ACADEMY_CARD_ART_DIR"] = str(art_dir)
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
        training_replay = academy.replay(training_source, game=0)
        output.write_text(
            json.dumps(
                {
                    "game": game.url,
                    "imported_game": imported_game.url,
                    "replay": replay.url,
                    "training_replay": training_replay.url,
                }
            )
        )
        print("ACADEMY_LAUNCH_PROOF_READY", flush=True)
        academy.wait()


if __name__ == "__main__":
    main()
