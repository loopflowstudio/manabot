"""MTG-325 speed check (steer 449b1a8e): one arm's torch timings, base or head.

    uv run python scratch/speed/bench.py OUT.json [--device cuda] [--streams 256]

Runs whichever `manabot/` is first on the path (main at the merge-base or this
branch) and resolves the module homes of either. One process: one Elemental
self-play batch from the current recipe (elemental-long-183 re-pinned to the
current world: w64l2 FiLM-off, 256 streams x 64 transitions, both seats,
elemental-uniform), with `bucketed_shapes`, `live_tokens` and tf32 on CUDA.
Then, from the same initial weights each time, a warm-up and `--repeats` timed
damped updates and PPO updates (optimizer steps per second, counted by a global
optimizer hook), `--inference` timed collection forwards through
`PolicyInference` over the collector's last observation rows (prepare, forward,
legal logs, sampling, host read), and `--checkpoints` save/load round trips.
"""

import argparse
import cProfile
from copy import deepcopy
import json
from pathlib import Path
import statistics
import tempfile
import time

import numpy as np
import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import RewardConfig
from manabot.model.inference import PolicyInference
from manabot.training.execution import validate_regime
from manabot.training.regime import Learning, TrainingRegime, TrainSelfPlay
from manabot.training.rollout.collection import SeatRoutedCollector
import managym
import manabot

try:
    from manabot.backends.torch import rl as _rl
    from manabot.backends.torch.checkpoint import load_checkpoint, save_checkpoint
    from manabot.backends.torch.learner import (
        Learner,
        apply_matmul_precision,
        forward_precision,
    )
    from manabot.backends.torch.policy import ActorCritic
    import manabot.backends.torch as _backend
    from manabot.training.updates.damped_policy import update_damped_iteration
    from manabot.training.updates.ppo import update_iteration

    ARM = "head"
    LEAD: tuple[object, ...] = (_backend,)
    legal_mask, policy_logs = _rl.legal_mask, _rl.policy_logs
except ImportError:  # main before this change (a343904ee)
    from manabot.model.actor_critic import ActorCritic  # type: ignore[no-redef]
    from manabot.model.checkpoint import (  # type: ignore[no-redef]
        load_checkpoint,
        save_checkpoint,
    )
    from manabot.model.policy_distribution import (  # type: ignore[no-redef]
        legal_mask,
        policy_logs,
    )
    from manabot.training.precision import (  # type: ignore[no-redef]
        apply_matmul_precision,
        forward_precision,
    )
    from manabot.training.updates.damped_policy import (  # type: ignore[no-redef]
        update_damped_iteration,
    )
    from manabot.training.updates.learner import Learner  # type: ignore[no-redef]
    from manabot.training.updates.ppo import (  # type: ignore[no-redef]
        update_iteration,
    )

    ARM = "base"
    LEAD = ()

RECIPE = Path(__file__).parents[2] / "experiments/regimes/elemental-long-183-v1.json"
_steps = 0


def _count_step(*_: object) -> None:
    global _steps
    _steps += 1


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summary(values: list[float]) -> dict[str, float | list[float]]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "all": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--streams", type=int, default=256)
    parser.add_argument("--transitions", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--inference", type=int, default=50)
    parser.add_argument("--checkpoints", type=int, default=5)
    parser.add_argument("--profile", type=Path, help="cProfile one update of each kind")
    args = parser.parse_args()
    device = torch.device(args.device)
    precision = "tf32" if device.type == "cuda" else "float32"
    apply_matmul_precision(precision)
    register_optimizer_step_post_hook(_count_step)

    regime = validate_regime(
        TrainingRegime.model_validate_json(RECIPE.read_bytes()).model_copy(
            update={"id": "mtg325-speed", "world": managym.WORLD_VERSION}
        )
    )
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    space = ObservationSpace(regime.observation)
    match = Match(regime.root_match)
    torch.manual_seed(25201)
    agent = ActorCritic(space, regime.agent, bucketed_shapes=True, live_tokens=True).to(
        device
    )
    initial = deepcopy(agent.state_dict())
    collector = SeatRoutedCollector(
        space,
        match,
        Reward(RewardConfig()),
        num_envs=args.streams,
        seed=25201,
        opponent_mode="self",
        learning_seats="both",
        batch_closure=stage.batch_closure,
        games_per_stream=stage.games_per_stream,
        recovery_max_microsteps=regime.recovery_max_microsteps,
        matchup_curriculum="elemental-uniform",
        device=args.device,
        precision=precision,
    )
    tick = time.perf_counter()
    batch = collector.collect(agent, args.transitions)
    collection_seconds = time.perf_counter() - tick
    result: dict[str, object] = {
        "arm": ARM,
        "manabot": str(Path(manabot.__file__).parent),
        "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "torch": torch.__version__,
        "collection_seconds": collection_seconds,
        "learner_rows": int(np.asarray(batch.rewards).size),
    }

    def updates(name: str, run: "callable") -> None:  # type: ignore[valid-type]
        global _steps
        rates, seconds, steps = [], [], []
        for repeat in range(args.repeats + 1):
            agent.load_state_dict(initial)
            learner = Learner(
                agent, device, minibatch_rows=args.streams, precision=precision
            )
            copy = deepcopy(batch)
            _sync(device)
            _steps = 0
            tick = time.perf_counter()
            if args.profile is not None and repeat == 1:
                profiler = cProfile.Profile()
                profiler.runcall(run, learner, copy)
                profiler.dump_stats(f"{args.profile}.{name}")
            else:
                run(learner, copy)
            _sync(device)
            elapsed = time.perf_counter() - tick
            if repeat:
                rates.append(_steps / elapsed)
                seconds.append(elapsed)
                steps.append(_steps)
        result[name] = {
            "steps_per_second": _summary(rates),
            "seconds": _summary(seconds),
            "steps": steps,
        }

    updates(
        "damped",
        lambda learner, b: update_damped_iteration(
            *LEAD, learner, b, stage.learning, 1
        ),
    )
    ppo = Learning(reference="action_type_uniform")
    updates(
        "ppo",
        lambda learner, b: update_iteration(
            *LEAD, learner, b, ppo, 0.3, np.random.default_rng(3), iteration=1
        ),
    )

    agent.load_state_dict(initial)
    agent.eval()
    rows = np.arange(args.streams)
    rates = []
    generator = torch.Generator(device=device).manual_seed(5)
    with PolicyInference(agent) as inference:
        for repeat in range(args.inference + 5):
            _sync(device)
            tick = time.perf_counter()
            obs = collector._slice_obs_tensors(rows, inference)
            with torch.inference_mode():
                with forward_precision(precision, device):
                    logits, _ = agent.forward(obs)
                probs = policy_logs(logits.float(), legal_mask(obs["actions_valid"]))
                actions = torch.multinomial(probs.exp(), 1, generator=generator)
            actions.cpu().numpy()
            elapsed = time.perf_counter() - tick
            if repeat >= 5:
                rates.append(len(rows) / elapsed)
    result["inference_obs_per_second"] = _summary(rates)

    saves, loads = [], []
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "speed.pt"
        for _ in range(args.checkpoints):
            tick = time.perf_counter()
            save_checkpoint(agent, space, path, player_configs=match.to_rust())
            saves.append(time.perf_counter() - tick)
            tick = time.perf_counter()
            load_checkpoint(str(path), bucketed_shapes=True, live_tokens=True)
            loads.append(time.perf_counter() - tick)
    result["save_seconds"] = _summary(saves)
    result["load_seconds"] = _summary(loads)
    collector.close()
    args.out.write_text(json.dumps(result, indent=1))
    print(
        json.dumps({k: v for k, v in result.items() if k != "gpu"}, default=str)[:2000]
    )


if __name__ == "__main__":
    main()
