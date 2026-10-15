"""Profile one existing semantic collector call without an optimizer or exports."""

import cProfile
from dataclasses import fields
import json
from pathlib import Path
import time

import numpy as np
import torch

from manabot.env import Match, ObservationSpace, Reward
from manabot.infra import RewardHypers
from manabot.model.agent import Agent
from manabot.sim.net_opponent import SeatRoutedCollector
from manabot.training.models import TrainingRegime


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(10831)
    recipe = TrainingRegime.model_validate_json(Path("scratch/cpu-probes/update-recipe.json").read_text())
    space = ObservationSpace(recipe.observation)
    agent = Agent(space, recipe.agent).eval()
    collector = SeatRoutedCollector(space, Match(recipe.match), Reward(RewardHypers()),
                                   num_envs=4, seed=10831, opponent_mode="self")
    profiler = cProfile.Profile()
    start = time.perf_counter()
    batch = profiler.runcall(collector.collect, agent, 64)
    elapsed = time.perf_counter() - start
    profiler.dump_stats("scratch/cpu-probes/collection.pstats")
    array_bytes: dict[str, int] = {}
    for field in fields(batch):
        value = getattr(batch, field.name)
        if isinstance(value, np.ndarray):
            array_bytes[field.name] = value.nbytes
        elif isinstance(value, dict):
            array_bytes[field.name] = sum(item.nbytes for item in value.values())
    payload = {"seconds": elapsed, "learner_transitions": 256,
               "parameters": sum(p.numel() for p in agent.parameters()),
               "collector_stats": collector.stats.to_dict(),
               "batch_array_bytes": array_bytes, "batch_total_bytes": sum(array_bytes.values()),
               "native_buffer_bytes": sum(array.nbytes for array in collector._buffers.values()),
               "seed":10831, "checkpoint":None, "weights":"untrained; fixed throughout collect",
               "limits":["One collection; cProfile overhead; first forward not warmed",
                         "Array bytes exclude Python objects, transient arrays and allocator overhead"]}
    Path("scratch/cpu-probes/collection.json").write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
