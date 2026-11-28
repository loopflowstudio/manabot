"""Time four-worker steps and refills. Hash packed legal offers, choices and outcomes."""

import argparse
from contextlib import closing
import hashlib
import json
import random
from time import perf_counter

import numpy as np

from manabot.env.observation import ObservationSpace, step_buffers
from manabot.training.rollout.producers import ShardedVectorEnv
import managym


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-allocator", choices=("system", "mimalloc"))
    args = parser.parse_args()
    if managym.ALLOCATION_COUNTING:
        parser.error(
            "allocation counting is enabled; rebuild without the benchmark feature"
        )
    if args.expect_allocator and args.expect_allocator != managym.NATIVE_ALLOCATOR:
        parser.error(
            f"expected allocator {args.expect_allocator}, loaded {managym.NATIVE_ALLOCATOR}"
        )

    streams = 16
    batches = 128
    seed = 24
    rng = random.Random(seed)
    space = ObservationSpace()
    buffers = step_buffers(streams)
    setup = managym.GameSetup(
        [
            managym.authored_deck_setup("ur-lessons-vs-gw-allies", key)
            for key in ("ur_lessons", "gw_allies")
        ]
    )
    digest = hashlib.sha256()
    seconds = 0.0
    with closing(ShardedVectorEnv(streams, seed, workers=4)) as env:
        env.set_observation_layout(
            buffers, action_overflow=space.config.action_overflow, layout=space.layout
        )
        env.reset_all_into_buffers(setup)
        for _ in range(batches):
            done = np.flatnonzero(
                (buffers["terminated"] > 0) | (buffers["truncated"] > 0)
            )
            if done.size:
                start = perf_counter()
                env.load_slots_into_buffers(
                    [managym.SlotLoad.fresh(int(s)) for s in done]
                )
                seconds += perf_counter() - start
            rows = env.observation_rows()
            batch = rows[0][0].gather(rows)
            actions = [rng.randrange(count) for count in batch.action_counts]
            _, offsets, values = batch.tables()["actions"]
            digest.update(offsets)
            digest.update(values)
            digest.update(np.asarray(actions, dtype="<u4"))
            start = perf_counter()
            env.step_into_buffers(actions)
            seconds += perf_counter() - start
            digest.update(buffers["terminated"])
            digest.update(buffers["rewards"])
    steps = batches * streams
    print(
        json.dumps(
            {
                "allocator": managym.NATIVE_ALLOCATOR,
                "steps": steps,
                "seconds": seconds,
                "steps_per_s": steps / seconds,
                "digest": digest.hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
