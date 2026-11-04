"""Bounded CPU layer-input costs, without optimizer updates or experiment launch.

Run each version/state in a fresh process for an interpretable peak RSS. The
recipe supplies the real model configuration; observations are constructed
positions. Warmups and individual samples are retained, not inferred tails.
"""

import argparse
from collections.abc import Callable
import hashlib
import json
import platform
import resource
import statistics
import time

import numpy as np
import torch

from manabot.env.observation import ObservationSpace
from manabot.infra.config import ObservationConfig
from manabot.model.actor_critic import ActorCritic
from manabot.recipes.baseline import baseline
from manabot.verify.representation import kellan_trigger_root, shared_effect_root
from manabot.verify.text_observation import priority_root


def measure(operation: Callable[[], object], samples: int) -> dict[str, object]:
    for _ in range(2):
        operation()
    timings = []
    for _ in range(samples):
        start = time.perf_counter()
        operation()
        timings.append(time.perf_counter() - start)
    return {
        "samples_s": timings,
        "median_s": statistics.median(timings),
        "max_s": max(timings),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, choices=(6, 7, 8), required=True)
    parser.add_argument(
        "--state", choices=("ordinary", "shared", "program"), required=True
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument(
        "--inventory",
        action="store_true",
        help="Validate parity and count storage without timing or policy execution",
    )
    args = parser.parse_args()
    if args.batch < 1 or args.samples < 1:
        parser.error("batch and samples must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(244)
    if args.state == "program":
        env = kellan_trigger_root()
    elif args.state == "shared":
        env = shared_effect_root(1)
    else:
        env = priority_root((), ("Mountain",) * 5 + ("Gray Ogre",))
    obs = env.observation_for_player(env.current_agent_index())
    # Match all configurable capacities. V7's published default is 128 facts;
    # this comparator explicitly uses 512, like v6/v8.
    space = ObservationSpace(
        ObservationConfig(policy_input_version=args.version, max_facts=512)
    )
    buffers = space.encoder.allocate()
    reference = space.encode(obs)
    env.encode_observation_into(obs, buffers)
    for name in reference:
        np.testing.assert_array_equal(reference[name], buffers[name])

    def sample(operation: Callable[[], object]) -> dict[str, object]:
        return {} if args.inventory else measure(operation, args.samples)

    native = sample(lambda: env.encode_observation_into(obs, buffers))
    python = sample(lambda: space.encode(obs))
    observation = sample(lambda: env.observation_for_player(env.current_agent_index()))
    config = baseline(1).regime.agent
    model = ActorCritic(space, config)
    values = {
        name: torch.from_numpy(value)
        .unsqueeze(0)
        .repeat(args.batch, *([1] * value.ndim))
        for name, value in buffers.items()
    }
    model.eval()
    with torch.inference_mode():
        inference = sample(lambda: model(values))
    model.train()

    def backward() -> None:
        model.zero_grad(set_to_none=True)
        logits, value = model(values)
        loss = (
            logits[values["actions_valid"].bool()].square().mean()
            + value.square().mean()
        )
        loss.backward()

    learning = sample(backward)
    details = buffers.get("effect_details")
    facts = buffers.get("facts")
    print(
        json.dumps(
            {
                "version": args.version,
                "inventory_only": args.inventory,
                "state": args.state,
                "batch": args.batch,
                "device": "cpu",
                "dtype": "float32",
                "threads": torch.get_num_threads(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "observation_config": space.encoder.config.model_dump(),
                "model_config": config.model_dump(),
                "observation_sha256": hashlib.sha256(obs.toJSON().encode()).hexdigest(),
                "observation_bytes": sum(array.nbytes for array in buffers.values()),
                "batch_observation_bytes": sum(
                    t.numel() * t.element_size() for t in values.values()
                ),
                "parameter_bytes": sum(
                    p.numel() * p.element_size() for p in model.parameters()
                ),
                "live_details": int(np.count_nonzero(details[:, 0]))
                if details is not None
                else 0,
                "live_facts": int(np.count_nonzero(facts[:, -1]))
                if facts is not None
                else 0,
                "native_encode": native,
                "python_encode": python,
                "observation": observation,
                "inference": inference,
                "forward_backward": learning,
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if platform.system() == "Darwin" else 1024),
            }
        )
    )


if __name__ == "__main__":
    main()
