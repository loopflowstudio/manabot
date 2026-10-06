"""CPU workloads for the bounded ETU-108 supervisor; no remote orchestration.

Simulator timing includes Python/tensor transport and random action selection.
Inference uses a fixed batch of real collected observations; train reports the
existing executor's collection/update/export clocks. Complete reuses calibration.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from experiments.runners.calibrate_training import calibrate
from manabot.env import Match, ObservationSpace, Reward, VectorEnv
from manabot.infra import RewardHypers
from manabot.model.agent import Agent
from manabot.model.world import checkpoint_world
from manabot.sim.net_opponent import SeatRoutedCollector
from manabot.training.execution import execute_regime
from manabot.training.models import TrainingRegime, TrainSelfPlay
from manabot.verify.store import VerifyStore
import managym
from managym import _managym as native


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("simulator", "inference", "train", "complete"), required=True
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(108)
    if args.mode == "complete":
        calibrate(args.out)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    recipe = TrainingRegime.model_validate_json(
        Path("experiments/regimes/ataraxos-move.json").read_text()
    )
    recipe.id = "etu108-benchmark"
    recipe.wall_seconds = 60
    for stage in recipe.stages:
        if not isinstance(stage, TrainSelfPlay):
            raise ValueError("benchmark requires self-play stages")
        stage.execution.wall_seconds = 25
        stage.execution.threads = 1
        stage.transitions = 16
        stage.updates = 1
    recipe = TrainingRegime.model_validate(recipe.model_dump())
    (args.out / "recipe.json").write_text(recipe.model_dump_json(indent=2) + "\n")
    match = Match(recipe.match)
    space = ObservationSpace(recipe.observation)
    reward = Reward(RewardHypers())
    binding = checkpoint_world(match.to_rust(), space)
    if native.__file__ is None:
        raise RuntimeError("native module has no artifact path")
    (args.out / "identity.json").write_text(
        json.dumps(
            {
                "world_binding": binding,
                "torch": torch.__version__,
                "native_world": managym.WORLD_VERSION,
                "native_sha256": hashlib.sha256(
                    Path(native.__file__).read_bytes()
                ).hexdigest(),
                "seed": 108,
                "device": "cpu",
                "threads": torch.get_num_threads(),
                "recipe_sha256": hashlib.sha256(
                    recipe.model_dump_json().encode()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    if args.mode == "train":
        with VerifyStore(args.out / "training.sqlite") as store:
            run = execute_regime(recipe, 108, args.out / "run", store)
        (args.out / "training-result.json").write_text(
            run.model_dump_json(indent=2) + "\n"
        )
        if run.status != "completed":
            raise RuntimeError(f"training status: {run.status}")
        return
    agent = Agent(space, recipe.agent).eval()
    if args.mode == "inference":
        collector = SeatRoutedCollector(
            space, match, reward, num_envs=4, seed=108, opponent_mode="self"
        )
        try:
            batch = collector.collect(
                agent, 16, deadline_monotonic=time.perf_counter() + 20
            )
        finally:
            collector.close()
        observations = {
            key: torch.from_numpy(value.reshape((-1, *value.shape[2:])).copy())
            for key, value in batch.obs.items()
        }
        with torch.inference_mode():
            agent(observations)
            start = time.perf_counter()
            iterations = 0
            while time.perf_counter() - start < args.seconds:
                agent(observations)
                iterations += 1
        units = iterations * 64
        unit = "observations (fixed real batch of 64; forward only)"
    else:
        env = VectorEnv(
            4, match, space, reward, "cpu", seed=108, opponent_policy="none"
        )
        observations, _ = env.reset()
        start = time.perf_counter()
        units = 0
        try:
            while time.perf_counter() - start < args.seconds:
                actions = torch.multinomial(
                    observations["actions_valid"].float(), 1
                ).squeeze(-1)
                observations, _, _, truncated, _ = env.step(actions)
                if truncated.any():
                    raise RuntimeError("truncated game in simulator benchmark")
                units += 4
        finally:
            env.close()
        unit = "surfaced native decisions including wrapper/random-sampling overhead"
    elapsed = time.perf_counter() - start
    (args.out / "measurement.json").write_text(
        json.dumps(
            {
                "units": units,
                "unit": unit,
                "seconds": elapsed,
                "units_per_second": units / elapsed,
                "parameters": sum(p.numel() for p in agent.parameters()),
                "initialization_and_warmup_excluded": True,
                "strength_or_useful_learning_progress_measured": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
