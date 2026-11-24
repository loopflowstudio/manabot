"""Measure retained buffers and a real CPU frontier update in a fresh process."""

import argparse
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import resource
from tempfile import TemporaryDirectory
from time import perf_counter, process_time, thread_time
from unittest.mock import patch

import numpy as np

from manabot.training.loops import self_play as loop
from manabot.training.rollout.admission import AdmittedRows
from manabot.training.rollout.collection import RolloutBatch
from scratch.bench_tla import frontier_v11


def retained_bytes(batch: RolloutBatch) -> int:
    """Unique backing bytes in rollout arrays, immutable CSR and offsets, excluding Python objects."""
    owners: dict[int, int] = {}
    seen_batches: set[int] = set()

    def count(value: object) -> None:
        if isinstance(value, np.ndarray):
            while isinstance(value.base, np.ndarray):
                value = value.base
            owners[id(value)] = value.nbytes
        elif hasattr(value, "coordinates"):
            count(vars(value).get("action_counts"))
            for source, _ in value.coordinates:
                if id(source) in seen_batches:
                    continue
                seen_batches.add(id(source))
                for _, offsets, records in source.tables().values():
                    owners[id(offsets)] = len(offsets)
                    owners[id(records)] = len(records)
        elif isinstance(value, Mapping):
            for array in value.values():
                count(array)

    for record in (batch, batch.recorded, batch.action_data):
        for field in fields(record):
            count(getattr(record, field.name))
    return sum(owners.values())


def rollout_reading(batch: RolloutBatch) -> dict[str, object]:
    return {
        "rows": batch.actions.size,
        "action_digest": hashlib.sha256(batch.actions.tobytes()).hexdigest(),
        "retained_bytes": retained_bytes(batch),
        "peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


@contextmanager
def learner_times() -> Iterator[dict[str, float]]:
    """Include gather in admission and preparation, even on a prefetch thread."""
    times: dict[str, float] = {}

    def timed(name: str) -> Callable[..., object]:
        function = getattr(AdmittedRows, name)

        def measure(*args: object, **kwargs: object) -> object:
            start = thread_time()
            result = function(*args, **kwargs)
            times[name] = times.get(name, 0.0) + thread_time() - start
            return result

        return measure

    with (
        patch.object(AdmittedRows, "admit_rows", timed("admit_rows")),
        patch.object(AdmittedRows, "prepare_rows", timed("prepare_rows")),
    ):
        yield times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streams", type=int, default=4)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument(
        "--report-collection",
        action="store_true",
        help="emit a retained-buffer reading before learning",
    )
    parser.add_argument("--collection-only", action="store_true")
    options = parser.parse_args()
    seed = 24
    settings = frontier_v11().model_copy(
        update={"streams": options.streams, "threads": 1}
    )
    counts = loop.Counts()
    rng = np.random.default_rng(seed)
    with TemporaryDirectory() as tmp, loop.device_settings(settings):
        learner, ema = loop.create(settings, seed=seed)
        learner.profiler.enabled = True
        run = loop.FreshRun(Path(tmp), seed)
        with loop.self_play_games(settings, run, counts) as collector:
            if options.collection_only:
                policy = loop.behavior(settings, learner, ema)
                wall, cpu = perf_counter(), process_time()
                batch = loop.collect(collector, policy, settings.transitions)
                reading = {
                    "stage": "collected",
                    "streams": settings.streams,
                    "seed": seed,
                    "wall_seconds": perf_counter() - wall,
                    "cpu_seconds": process_time() - cpu,
                    **rollout_reading(batch),
                }
                print(json.dumps(reading), flush=True)
                return
            for _ in range(options.updates):
                batches: list[RolloutBatch] = []
                collect = loop.collect

                def record(*args: object, **kwargs: object) -> RolloutBatch:
                    batch = collect(*args, **kwargs)
                    if options.report_collection:
                        print(
                            json.dumps(
                                {
                                    "stage": "collected",
                                    **rollout_reading(batch),
                                }
                            ),
                            flush=True,
                        )
                    batches.append(batch)
                    return batch

                with (
                    patch.object(loop, "collect", record),
                    learner_times() as learner_cpu,
                ):
                    wall, cpu = perf_counter(), process_time()
                    row = loop.step(
                        settings, learner, ema, collector, counts, rng, out=run.out
                    )
                    wall, cpu = perf_counter() - wall, process_time() - cpu
                batch = batches.pop()
                print(
                    json.dumps(
                        {
                            "streams": settings.streams,
                            "seed": seed,
                            "update": counts.updates,
                            **rollout_reading(batch),
                            "wall_seconds": wall,
                            "cpu_seconds": cpu,
                            "learner_thread_seconds": learner_cpu,
                            "phases": {
                                k: v for k, v in row.items() if k.startswith("time/")
                            },
                        }
                    ),
                    flush=True,
                )
                del batch


if __name__ == "__main__":
    main()
