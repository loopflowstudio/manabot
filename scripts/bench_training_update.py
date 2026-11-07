#!/usr/bin/env python3
"""Where one training update's time goes, from the real stage on this machine.

Runs a recipe's self-play stage for a few updates through `execute_regime` and
prints its receipt per environment decision. `--recipe` is a regime JSON file or
a declaration `module:function` called as `function(regime_id, updates)`, such
as `experiments.elemental_variants:elemental_recipe`. The default is the Elemental
183 file re-pinned to the current world, so a checkout of any commit with that
regime file gives a comparable row on the same machine; that re-pin has no
capacity recovery, so one overflowing game ends the run. `--max-facts` narrows
the fact axis for attribution only; it never names a recipe to train.

`--json OUT` writes the whole `TrainingRun` receipt. `--cuda-counts OUT` counts
CUDA kernel launches (`torch.profiler`) and synchronizing calls
(`torch.cuda.set_sync_debug_mode("warn")`) over the run; divide by the receipt's
phase `calls` for per-forward or per-minibatch rates. The instrumentation slows
the run, so its timings are not comparable with uninstrumented rows.

Host-side phases are comparable across machines; model phases on CPU are not,
because rented GPUs spend them on launch latency. On a shared machine, alternate
the checkouts being compared and read minimum, not mean, durations.

    uv run python scripts/bench_training_update.py --streams 256 --updates 3
"""

import argparse
from collections.abc import Callable, Iterator
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import tempfile
import warnings

from manabot.training.execution import execute_regime, validate_regime
from manabot.training.records import TrainingRun
from manabot.training.regime import TrainingRegime, TrainSelfPlay
from manabot.verify.store import VerifyStore
import managym

RECIPE = Path(__file__).parents[1] / "experiments/regimes/elemental-long-183-v1.json"
REGIME_ID = "bench-training-update"
# Driver-side names of a kernel launch in the profiler's CPU events.
_LAUNCHES = frozenset({"cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC"})


def load_recipe(spec: str, updates: int) -> TrainingRegime:
    """A regime file, or a declaration `module:function(regime_id, updates)`."""
    if ":" in spec and not Path(spec).exists():
        module, name = spec.split(":", 1)
        declare: Callable[[str, int], TrainingRegime] = getattr(
            importlib.import_module(module), name
        )
        return declare(REGIME_ID, updates)
    return TrainingRegime.model_validate_json(Path(spec).read_bytes())


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
            warnings.showwarning = shown
    launches = sum(e.count for e in trace.key_averages() if e.key in _LAUNCHES)
    out.write_text(json.dumps({"kernel_launches": launches, "synchronizations": syncs}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recipe", default=str(RECIPE))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--streams", type=int, default=256)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=25201)
    parser.add_argument("--max-facts", type=int)
    parser.add_argument("--json", type=Path, help="write the TrainingRun receipt")
    parser.add_argument("--cuda-counts", type=Path, help="write CUDA launch counts")
    args = parser.parse_args()

    saved = load_recipe(args.recipe, args.updates)
    regime = validate_regime(
        saved.model_copy(update={"id": REGIME_ID, "world": managym.WORLD_VERSION})
    )
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.updates = args.updates
    stage.streams = args.streams
    stage.execution.device = args.device
    stage.execution.wall_seconds = regime.wall_seconds = 3600.0
    if args.max_facts is not None:
        regime.observation = regime.observation.model_copy(
            update={"max_facts": args.max_facts}
        )

    run = _execute(regime, args.seed, args.cuda_counts)
    if args.json is not None:
        args.json.write_text(run.model_dump_json())
    if run.status != "completed":
        raise SystemExit(f"benchmark run {run.status}: {run.error}")

    record = run.stages[0]
    decisions = record.environment_decisions
    print(
        json.dumps(
            {
                "seconds_per_update": round(
                    (record.collection_seconds + record.learning_seconds)
                    / record.updates,
                    2,
                ),
                "decisions_per_update": decisions // record.updates,
                "games_per_update": record.games // record.updates,
                "host": run.host.model_dump() if run.host else None,
            }
        )
    )
    for label, timings in (
        ("collection", record.collection_timings or {}),
        ("learning", record.learning_timings or {}),
    ):
        for path, timing in timings.items():
            micros = timing.seconds * 1e6 / decisions
            print(f"{label:10} {path:64} {micros:9.1f} us/decision")


def _execute(regime: TrainingRegime, seed: int, counts: Path | None) -> TrainingRun:
    with tempfile.TemporaryDirectory(prefix="bench-update-") as folder:
        out = Path(folder)
        with VerifyStore(out / "runs.sqlite") as store:
            if counts is None:
                return execute_regime(regime, seed, out / "run", store)
            with _cuda_counts(counts):
                return execute_regime(regime, seed, out / "run", store)


if __name__ == "__main__":
    main()
