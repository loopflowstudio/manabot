"""CPU workloads for the bounded ETU-108 supervisor; no remote orchestration.

Simulator timing includes Python/tensor transport and random action selection.
Inference uses a fixed batch of real collected observations; train reports the
existing executor's collection/update/export clocks. Complete reuses calibration.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import time

import torch

from experiments.runners.calibrate_training import calibrate
from manabot.arena.models import canonical_sha256
from manabot.env import Match, ObservationSpace, Reward, VectorEnv
from manabot.infra import RewardHypers
from manabot.model.agent import Agent
from manabot.model.world import checkpoint_world
from manabot.sim.net_opponent import SeatRoutedCollector
from manabot.sim.teacher1_evidence import source_bundle_sha256
from manabot.training.execution import execute_regime, validate_regime
from manabot.training.models import TrainingRegime, TrainingRun, TrainSelfPlay
from manabot.verify.store import VerifyStore
import managym
from managym import _managym as native


def _recipe(path: Path | None, timeout: float) -> TrainingRegime:
    """Admit a resolved self-play recipe unchanged, or construct the tiny smoke."""
    recipe = TrainingRegime.model_validate_json(
        (path or Path("experiments/regimes/ataraxos-move.json")).read_bytes()
    )
    if path is None:
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
    recipe = validate_regime(recipe)
    if recipe.wall_seconds > timeout:
        raise ValueError("recipe wall_seconds exceeds supervisor timeout")
    for stage in recipe.stages:
        if not isinstance(stage, TrainSelfPlay) or stage.behavior != "current-self":
            raise ValueError("benchmark requires current-self self-play stages")
        if stage.execution.threads != 1:
            raise ValueError("benchmark requires one CPU thread")
        if stage.execution.wall_seconds > recipe.wall_seconds:
            raise ValueError("stage wall_seconds exceeds recipe budget")
    return recipe


@dataclass(frozen=True)
class TrainingMeasurement:
    """Derived counters, including iterations that performed no optimization."""

    status: str
    learner_transitions: int
    completed_iterations: int
    optimizer_exposures: int
    empty_filter_skips: int
    training_seconds: float
    collection_seconds: float
    learning_seconds: float
    export_seconds: float


def _training_measurement(run: TrainingRun) -> TrainingMeasurement:
    return TrainingMeasurement(
        status=run.status,
        learner_transitions=sum(s.learner_transitions for s in run.stages),
        completed_iterations=run.updates_through(),
        optimizer_exposures=sum(s.optimizer_exposures for s in run.stages),
        empty_filter_skips=sum(
            d.get("skipped") == "empty advantage filter"
            for s in run.stages
            for d in s.diagnostics
        ),
        training_seconds=run.seconds,
        collection_seconds=sum(s.collection_seconds for s in run.stages),
        learning_seconds=sum(s.learning_seconds for s in run.stages),
        export_seconds=sum(s.export_seconds for s in run.stages),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("simulator", "inference", "train", "complete"), required=True
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--timeout", type=float, default=110)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--seed", type=int, default=108)
    args = parser.parse_args()
    if not 0 < args.seconds <= 10 or not 0 < args.timeout <= 240:
        parser.error("seconds must be in (0,10]; timeout must be in (0,240]")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0,2**32)")
    if args.recipe is not None and args.mode not in {"inference", "train"}:
        parser.error("--recipe is supported only for inference/train")
    if args.mode == "complete" and args.seed != 108:
        parser.error("complete uses the calibration runner's own seed")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    if args.mode == "complete":
        calibrate(args.out)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    recipe = _recipe(args.recipe, args.timeout)
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
                "python": platform.python_version(),
                "training_source_sha256": source_bundle_sha256(
                    sorted(Path("manabot").resolve().rglob("*.py"))
                ),
                "native_world": managym.WORLD_VERSION,
                "native_sha256": hashlib.sha256(
                    Path(native.__file__).read_bytes()
                ).hexdigest(),
                "seed": args.seed,
                "device": "cpu",
                "threads": torch.get_num_threads(),
                "regime_digest": canonical_sha256(recipe.model_dump(mode="json")),
                "requested_memory_bytes_by_stage": {
                    stage.id: stage.execution.memory_bytes for stage in recipe.stages
                },
                "memory_limit_enforced": False,
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
            run = execute_regime(recipe, args.seed, args.out / "run", store)
        (args.out / "training-result.json").write_text(
            run.model_dump_json(indent=2) + "\n"
        )
        (args.out / "measurement.json").write_text(
            json.dumps(asdict(_training_measurement(run)), indent=2) + "\n"
        )
        if run.status != "completed":
            raise RuntimeError(f"training status: {run.status}")
        return
    agent = Agent(space, recipe.agent).eval()
    if args.mode == "inference":
        stage = recipe.stages[0]
        assert isinstance(stage, TrainSelfPlay)
        collector = SeatRoutedCollector(
            space,
            match,
            reward,
            num_envs=stage.streams,
            seed=args.seed,
            opponent_mode="self",
        )
        try:
            batch = collector.collect(
                agent,
                stage.transitions,
                deadline_monotonic=time.perf_counter()
                + (20 if args.recipe is None else stage.execution.wall_seconds),
            )
        finally:
            # The collector owns an in-process PyO3 environment, not workers.
            # Releasing it drops native games/buffers; there is no close API.
            del collector
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
        units = iterations * stage.streams * stage.transitions
        unit = f"observations (fixed real batch of {stage.streams * stage.transitions}; forward only)"
    else:
        env = VectorEnv(
            4, match, space, reward, "cpu", seed=args.seed, opponent_policy="none"
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
