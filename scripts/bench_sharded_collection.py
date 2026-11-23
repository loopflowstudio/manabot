"""Time four-worker buffered steps and hash the contiguous arrays between calls."""

from contextlib import closing
import hashlib
import json
import random
from time import perf_counter

import numpy as np

from manabot.env.observation import ObservationSpace, allocate_vector_buffers
from manabot.training.rollout.producers import ShardedVectorEnv
import managym


def main() -> None:
    streams = 16
    batches = 128
    seed = 24
    rng = random.Random(seed)
    space = ObservationSpace()
    buffers = allocate_vector_buffers(space, streams)
    setup = managym.GameSetup(
        [
            managym.authored_deck_setup("ur-lessons-vs-gw-allies", key)
            for key in ("ur_lessons", "gw_allies")
        ]
    )
    digest = hashlib.sha256()
    seconds = 0.0
    with closing(ShardedVectorEnv(streams, seed, workers=4)) as env:
        env.set_buffers(buffers, layout=space.layout)
        env.reset_all_into_buffers(setup)
        for _ in range(batches):
            actions = [
                int(rng.choice(np.flatnonzero(mask)))
                for mask in buffers["actions_valid"]
            ]
            digest.update(buffers["actions"])
            digest.update(buffers["actions_valid"])
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
                "steps": steps,
                "seconds": seconds,
                "steps_per_s": steps / seconds,
                "digest": digest.hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
