"""Time the LLM text renderer on the native stress-state corpus.

The Rust harness exports recipes together with their exact acting-player
observation and decision capture. Decoding is outside three individually
timed samples; no Python board builder can drift from Rust. Policy-input
encoding is native and timed by the Rust harness itself.
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

from manabot.semantic.text_observation import render_decision


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
    parser.add_argument("--recipes-file", type=Path)
    args = parser.parse_args()
    if args.recipes_file:
        corpus = args.recipes_file.read_text()
    else:
        harness = subprocess.run(
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
            text=True,
            stdout=subprocess.PIPE,
            timeout=180,
        )
        # The harness prints its own error (unknown state, bad size) to stderr.
        if harness.returncode:
            raise SystemExit(harness.returncode)
        corpus = harness.stdout
    for line in corpus.splitlines():
        point = Checkpoint.model_validate_json(line)
        if args.states and point.recipe.state not in args.states:
            continue
        row = {
            "state": point.recipe.state,
            "size": point.recipe.size,
            "checkpoint": point.checkpoint,
            "layer": "text",
            "ms": _measure(
                lambda: render_decision(point.capture, max_bytes=4 * 1024 * 1024)
            ),
        }
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
