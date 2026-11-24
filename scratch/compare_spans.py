"""Compare both v11 transports on the eight seed-24 positions from stages.py."""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import fields, is_dataclass
import random
import statistics
import subprocess
import sys
import time
from types import ModuleType
from unittest.mock import patch

import numpy as np
import torch

from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match
from manabot.env.observation import ObservationSpace, allocate_vector_buffers
from manabot.policy.spans import observation_spans
from manabot.training.loops.self_play import root_match
import managym
from scratch.bench_tla import frontier_v11


def equal(left: object, right: object, path: str) -> None:
    if is_dataclass(left):
        for field in fields(left):
            equal(
                getattr(left, field.name),
                getattr(right, field.name),
                f"{path}.{field.name}",
            )
    elif isinstance(left, tuple):
        assert isinstance(right, tuple)
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            equal(a, b, f"{path}[{index}]")
    elif isinstance(left, np.ndarray):
        assert (
            isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and np.array_equal(left, right)
        ), path
    else:
        assert left == right, path


def best(call: Callable[[], object]) -> float:
    samples = []
    for _ in range(5):
        start = time.process_time()
        call()
        samples.append(1000 * (time.process_time() - start))
    return min(samples)


def compare() -> None:
    torch.set_num_threads(1)
    source = subprocess.check_output(
        ["git", "show", "fe46e6b375:manabot/policy/prepare.py"]
    )
    baseline = ModuleType("slice_one")
    sys.modules[baseline.__name__] = baseline
    exec(compile(source, "fe46e6b375:prepare.py", "exec"), baseline.__dict__)
    settings = frontier_v11()
    policy = ActorCritic(ObservationSpace(settings.observation), settings.model)
    space = policy.observation_space
    streams, seed = 256, 24
    sampler = random.Random(seed)
    deck_source = settings.deck_source
    setup = managym.GameSetup(
        Match(root_match(settings)).to_rust(), decklists=deck_source.decklists
    )
    policy_overflow = space.config.action_overflow
    dense, packed = [
        managym.VectorEnv(streams, seed=seed, seed_stride=streams) for _ in range(2)
    ]
    buffers = allocate_vector_buffers(space, streams)
    for env in (dense, packed):
        # Both copies follow identical seed succession. The packed copy writes
        # into its own scratch buffers on steps omitted by the sampling schedule.
        env.set_buffers(
            buffers if env is dense else allocate_vector_buffers(space, streams),
            policy_overflow,
            layout=space.layout,
        )
        env.set_training_decks(
            deck_source.players(),
            distinct=deck_source.sampling == "uniform-distinct-v1",
        )
        env.reset_all_into_buffers(setup)
    observations = {k: buffers[k] for k in space.shapes}
    results: dict[str, list[float]] = defaultdict(list)
    for step in range(1, 321):
        actions = [
            int(sampler.choice(np.flatnonzero(mask)))
            for mask in buffers["actions_valid"]
        ]
        start = time.process_time()
        dense.step_into_buffers(actions)
        elapsed = 1000 * (time.process_time() - start)
        if step % 40:
            packed.step_into_buffers(actions)
            continue
        results["dense_step_ms"].append(elapsed)
        start = time.process_time()
        batch, rewards, terminated, truncated = packed.step_observation_batch(
            actions, space.layout, policy_overflow
        )
        results["packed_step_ms"].append(1000 * (time.process_time() - start))
        for key, rows in (
            ("rewards", rewards),
            ("terminated", terminated),
            ("truncated", truncated),
        ):
            np.testing.assert_array_equal(buffers[key], rows)
        rows = observation_spans(batch, space, bucketed=False, objects=False)
        for key, expected in observations.items():
            np.testing.assert_array_equal(rows.dense(key), expected, err_msg=key)
        results["bytes"].append(
            sum(len(o) + len(v) for _, o, v in batch.tables().values())
        )
        results["dense_admit"].append(
            best(lambda: policy.check_observation(observations))
        )
        results["native_admit"].append(best(lambda: policy.check_observation(batch)))
        for inference in (False, True):

            def dense_prepare() -> object:
                return policy.prepare_admitted(observations, inference=inference)

            def packed_prepare() -> object:
                return policy.prepare_admitted(batch, inference=inference)

            with patch("manabot.backends.torch.policy.prepare", baseline.prepare):
                expected = dense_prepare()
                results[f"dense_prepare_{inference}"].append(best(dense_prepare))
            equal(expected, packed_prepare(), f"step {step}, inference={inference}")
            results[f"native_prepare_{inference}"].append(best(packed_prepare))
        print(f"Step {step} matched", flush=True)
    print(
        {
            name: round(statistics.median(samples), 2)
            for name, samples in results.items()
        }
    )


if __name__ == "__main__":
    compare()
