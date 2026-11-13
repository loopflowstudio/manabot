#!/usr/bin/env python3
"""Measure a few self-play loop updates on this machine.

`--recipe module:function` returns SelfPlay settings without arguments; the
Elemental builder is the default. Regime JSON and TrainingRun receipts are no
longer supported. `--json` writes per-update rows and a summary; `--out` retains
state and immutable policies, otherwise they are temporary. Output directories
must be fresh so the measurement covers exactly the requested updates.
CUDA counts cover the entire run and change timing. Compare matched workloads
on the same machine; CPU timings do not establish CUDA throughput.

    uv run python scripts/bench_training_update.py --streams 4 --updates 2
"""

import argparse
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
import importlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict, cast
import warnings

from manabot.training.loops.self_play import SelfPlay, self_play

RECIPE = "manabot.recipes.builders:elemental_self_play"
_LAUNCHES = frozenset({"cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC"})


class Summary(TypedDict):
    seconds_per_update: float
    decisions_per_update: float
    games_per_update: float
    phase_microseconds_per_decision: dict[str, float]


class Measurement(TypedDict):
    recipe_settings: dict[str, object]
    seed: int
    rows: list[dict[str, object]]
    summary: Summary


def load_recipe(spec: str) -> SelfPlay:
    """Load an ordinary zero-argument settings builder, validating its result."""
    module, name = spec.split(":", 1)
    declare = cast(
        Callable[[], SelfPlay], getattr(importlib.import_module(module), name)
    )
    return SelfPlay.model_validate(declare())


@dataclass
class Rows:
    """An in-memory Tracker for a bounded untracked measurement."""

    rows: list[dict[str, object]] = field(default_factory=list)

    def log(self, values: dict[str, object], /, **clocks: float) -> None:
        self.rows.append(values | clocks)

    def log_table(
        self, name: str, rows: list[dict[str, object]], /, **clocks: float
    ) -> None:
        pass

    def checkpoint(
        self, path: Path, /, *, tags: Sequence[str] = (), **clocks: float
    ) -> None:
        pass

    def stopping(self) -> bool:
        return False


@contextmanager
def _cuda_counts(out: Path) -> Iterator[None]:
    """Count kernel launches and synchronizing calls inside the block."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    if not torch.cuda.is_available():
        raise SystemExit("--cuda-counts needs CUDA")
    syncs = 0
    shown = warnings.showwarning

    def count(message: Warning | str, *args: object, **kwargs: object) -> None:
        nonlocal syncs
        if "synchroniz" in str(message):
            syncs += 1
            return
        shown(message, *args, **kwargs)  # type: ignore[arg-type]

    # catch_warnings restores the filters and `showwarning` on exit.
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = count  # type: ignore[assignment]
        torch.cuda.set_sync_debug_mode("warn")
        try:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as trace:
                yield
        finally:
            torch.cuda.set_sync_debug_mode("default")
    launches = sum(e.count for e in trace.key_averages() if e.key in _LAUNCHES)
    out.write_text(json.dumps({"kernel_launches": launches, "synchronizations": syncs}))


def measure(settings: SelfPlay, out: Path, *, seed: int, updates: int) -> Measurement:
    """Return the loop's rows and summary, retaining policies at a fresh path."""
    if (out / "state.pt").exists():
        raise ValueError("benchmark requires a fresh --out directory")
    tracker = Rows()
    self_play(settings, out, seed=seed, updates=updates, run=tracker)
    last = tracker.rows[-1]
    decisions = int(cast(int, last["coordinates/environment_decisions"]))
    phases = {
        key: sum(float(cast(float, row.get(key, 0))) for row in tracker.rows)
        for key in {
            key for row in tracker.rows for key in row if key.startswith("time/")
        }
    }
    return {
        "recipe_settings": settings.model_dump(mode="json"),
        "seed": seed,
        "rows": tracker.rows,
        "summary": {
            "seconds_per_update": float(cast(float, last["training_seconds"]))
            / updates,
            "decisions_per_update": decisions / updates,
            "games_per_update": int(cast(int, last["coordinates/games"])) / updates,
            "phase_microseconds_per_decision": {
                key: seconds * 1e6 / decisions
                for key, seconds in sorted(phases.items())
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recipe", default=RECIPE)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--streams", type=int, default=256)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=25201)
    parser.add_argument("--max-facts", type=int)
    parser.add_argument("--json", type=Path, help="write update rows and summary")
    parser.add_argument("--out", type=Path, help="keep state and checkpoints here")
    parser.add_argument("--cuda-counts", type=Path, help="write CUDA launch counts")
    args = parser.parse_args()
    settings = load_recipe(args.recipe)
    values = settings.model_dump()
    values.update(streams=args.streams, device=args.device)
    if args.max_facts is not None:
        values["observation"]["max_facts"] = args.max_facts
    settings = SelfPlay.model_validate(values)
    with TemporaryDirectory(prefix="manabot-update-") as temporary:
        with _cuda_counts(args.cuda_counts) if args.cuda_counts else nullcontext():
            result = measure(
                settings,
                args.out or Path(temporary),
                seed=args.seed,
                updates=args.updates,
            )
    if args.json is not None:
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
