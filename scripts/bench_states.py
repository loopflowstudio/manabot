"""Time Python consumers of the native stress-state corpus.

The Rust harness exports recipes together with their exact acting-player
observation and decision capture. Decoding and encoder allocation are outside
three individually timed samples; no Python board builder can drift from Rust.
Run from the repository root with ``uv run --extra play python scripts/bench_states.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
from time import perf_counter
from typing import Callable

from pydantic import BaseModel

from manabot.env.observation import ObservationEncoder
from manabot.infra.config import ObservationConfig
from manabot.semantic.text_observation import TextEncodingError, render_decision
import managym


class Recipe(BaseModel):
    state: str
    size: int


class Checkpoint(BaseModel):
    recipe: Recipe
    checkpoint: str
    observation: str
    continuation: str
    capture: str


def _measure(work: Callable[[], object]) -> float:
    samples: list[float] = []
    for _ in range(3):
        start = perf_counter()
        work()
        samples.append(1000 * (perf_counter() - start))
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("states", nargs="*")
    parser.add_argument("--sizes", help="comma-separated; default: each state's own")
    parser.add_argument("--layers", default="encode-py,text")
    parser.add_argument("--recipes-file", type=Path)
    args = parser.parse_args()
    layers = args.layers.split(",")
    if unknown := set(layers) - {"encode-py", "text"}:
        parser.error(f"unknown layers: {sorted(unknown)}")
    if args.recipes_file:
        corpus = args.recipes_file.read_text()
    else:
        corpus = subprocess.run(
            [
                "cargo",
                "run",
                "--release",
                "--features",
                "benchmark",
                "--example",
                "bench_states",
                "--manifest-path",
                "managym/Cargo.toml",
                "--",
                "--recipes",
                *(["--sizes", args.sizes] if args.sizes else []),
                *args.states,
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            timeout=180,
        ).stdout
    for line in corpus.splitlines():
        point = Checkpoint.model_validate_json(line)
        if args.states and point.recipe.state not in args.states:
            continue
        obs = managym.Observation.from_json(
            point.observation, continuation_json=point.continuation
        )
        encoder = ObservationEncoder(
            ObservationConfig(
                policy_input_version=10,
                max_cards_per_player=max(
                    60, len(obs.agent_cards), len(obs.opponent_cards)
                ),
                max_permanents_per_player=max(
                    40, len(obs.agent_permanents), len(obs.opponent_permanents)
                ),
                max_actions=max(512, len(obs.action_space.actions)),
                max_facts=4096,
                max_events=max(32, len(obs.recent_events)),
            )
        )
        consumers: dict[str, Callable[[], object]] = {
            "encode-py": lambda: encoder.encode(obs),
            "text": lambda: render_decision(point.capture, max_bytes=4 * 1024 * 1024),
        }
        for layer in layers:
            row: dict[str, str | int | float] = {
                "state": point.recipe.state,
                "size": point.recipe.size,
                "checkpoint": point.checkpoint,
                "layer": layer,
            }
            try:
                row["ms"] = _measure(consumers[layer])
            except TextEncodingError as error:
                # An existing renderer gap is an audit finding. Malformed
                # captures, capacity failures and other faults still fail the run.
                if str(error) != "unsupported continuation stage":
                    raise
                row["unavailable"] = str(error)
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
