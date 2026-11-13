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

`--json OUT` writes the whole `TrainingRun` receipt. `--out DIR` keeps the run's
folder (its verify store, checkpoints and the Etude metrics run that holds each
update's rows, which the receipt's `metrics_run` names); without it the folder is
temporary and deleted. `--cuda-counts OUT` counts CUDA kernel launches
(`torch.profiler`) and synchronizing calls (`torch.cuda.set_sync_debug_mode("warn")`)
as one total over the run. Divided by the receipt's updates it is a per-update
rate; divided by one phase's `calls` it is only an upper bound for that phase,
because every phase's launches are in the sum. The instrumentation slows the run,
so its timings are not comparable with uninstrumented rows.

Compare phase and whole-update timings on the same machine with matched workloads.
CPU model timings do not establish CUDA throughput. On a shared machine, alternate
the checkouts being compared and retain the timing spread.

    uv run python scripts/bench_training_update.py --streams 256 --updates 3
"""

import argparse
from collections.abc import Callable, Iterator
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import warnings

from manabot.training.regime import TrainingRegime, TrainSelfPlay
from manabot.training.validation import validate_regime
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recipe", default=str(RECIPE))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--streams", type=int, default=256)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=25201)
    parser.add_argument("--max-facts", type=int)
    parser.add_argument("--json", type=Path, help="write the TrainingRun receipt")
    parser.add_argument("--out", type=Path, help="keep the run folder here")
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

    run = _execute(regime, args.seed, args.out, args.cuda_counts)
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


def _execute(*args: object, **kwargs: object) -> None:
    """Retired frozen execution entry point; retained data and evaluation still read."""
    raise RuntimeError(
        "Frozen execution requires revision f09f5c2919b8b1a5cf6d72c4f4f4721dac68f653"
    )


if __name__ == "__main__":
    main()
