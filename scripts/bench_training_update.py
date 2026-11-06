#!/usr/bin/env python3
"""Where one training update's time goes, from the real stage on this machine.

Runs a recipe's self-play stage for a few updates through `execute_regime` and
prints its receipt per environment decision. The default is the Elemental 183
recipe re-pinned to the current world, so a checkout of any commit with that
regime file gives a comparable row on the same machine. `--max-facts` narrows the
fact axis for attribution only; it never names a recipe to train.

Host-side phases are comparable across machines; model phases on CPU are not,
because rented GPUs spend them on launch latency. On a shared machine, alternate
the checkouts being compared and read minimum, not mean, durations.

    uv run python scripts/bench_training_update.py --streams 256 --updates 3
"""

import argparse
import json
from pathlib import Path
import tempfile

from manabot.training.execution import execute_regime, validate_regime
from manabot.training.regime import TrainingRegime, TrainSelfPlay
from manabot.verify.store import VerifyStore
import managym

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "experiments/regimes/elemental-long-183-v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recipe", type=Path, default=RECIPE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--streams", type=int, default=256)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=25201)
    parser.add_argument("--max-facts", type=int)
    args = parser.parse_args()

    saved = TrainingRegime.model_validate_json(args.recipe.read_bytes())
    regime = validate_regime(
        saved.model_copy(
            update={"id": "bench-training-update", "world": managym.WORLD_VERSION}
        )
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

    with tempfile.TemporaryDirectory(prefix="bench-update-") as folder:
        out = Path(folder)
        with VerifyStore(out / "runs.sqlite") as store:
            run = execute_regime(regime, args.seed, out / "run", store)
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


if __name__ == "__main__":
    main()
