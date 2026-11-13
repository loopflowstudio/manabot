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
`--ppo-only` omits the other operations. `--profile` adds an untimed update
after the samples; its operator trace covers the first optimizer step.
"""

import argparse
from collections.abc import Callable
from copy import deepcopy
import cProfile
import faulthandler
import hashlib
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import pstats
import statistics
import tempfile
import time

import numpy as np
import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

import manabot
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import RewardConfig
from manabot.infra.runtime import seed_random
from manabot.model.inference import PolicyInference
from manabot.training.execution import validate_regime
from manabot.training.regime import Learning, TrainingRegime, TrainSelfPlay
from manabot.training.rollout.collection import RolloutBatch, SeatRoutedCollector
from manabot.training.updates.selection import UpdateDiagnostics
import managym


def _first(*homes: str) -> object:
    """The first importable `module:name` of `homes` (the arms move names)."""
    for home in homes:
        module, name = home.split(":")
        try:
            return getattr(importlib.import_module(module), name)
        except (ImportError, AttributeError):
            continue
    raise ImportError(f"none of {homes}")


ActorCritic = _first(
    "manabot.backends.torch.policy:ActorCritic",
    "manabot.model.actor_critic:ActorCritic",
)
load_checkpoint = _first(
    "manabot.backends.torch.checkpoint:load_checkpoint",
    "manabot.model.checkpoint:load_checkpoint",
)
save_checkpoint = _first(
    "manabot.backends.torch.checkpoint:save_checkpoint",
    "manabot.model.checkpoint:save_checkpoint",
)
legal_mask = _first(
    "manabot.backends.torch.rl:legal_mask",
    "manabot.model.policy_distribution:legal_mask",
)
policy_logs = _first(
    "manabot.backends.torch.rl:policy_logs",
    "manabot.model.policy_distribution:policy_logs",
)
Learner = _first(
    "manabot.backends.torch.learner:Learner", "manabot.training.updates.learner:Learner"
)
apply_matmul_precision = _first(
    "manabot.backends.torch.learner:apply_matmul_precision",
    "manabot.training.precision:apply_matmul_precision",
)
forward_precision = _first(
    "manabot.backends.torch.learner:forward_precision",
    "manabot.training.precision:forward_precision",
)
update_damped_iteration = _first(
    "manabot.training.updates.damped_policy:update_damped_iteration"
)
update_iteration = _first("manabot.training.updates.ppo:update_iteration")
# The updates take the backend module first on this branch (#501) only.
TAKES_BACKEND = "backend" in inspect.signature(update_iteration).parameters
LEAD: tuple[object, ...] = (
    (importlib.import_module("manabot.backends.torch"),) if TAKES_BACKEND else ()
)
if TAKES_BACKEND:
    ARM = "head"
elif ActorCritic.__module__ == "manabot.backends.torch.policy":
    ARM = "backend"
else:
    ARM = "base"

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


def _write_result(path: Path, result: dict[str, object]) -> None:
    """Retain each completed sample even if a later update reaches the job cap."""
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(result, indent=1))
    partial.replace(path)


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
    parser.add_argument("--ppo-only", action="store_true")
    args = parser.parse_args()
    if args.profile is not None:
        faulthandler.dump_traceback_later(120, repeat=True)
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
    torch.set_num_threads(stage.execution.threads)
    space = ObservationSpace(regime.observation)
    match = Match(regime.root_match)
    seed_random(25201)
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
    print("collection started", flush=True)
    batch = collector.collect(agent, args.transitions)
    collection_seconds = time.perf_counter() - tick
    print(
        f"collection seconds={collection_seconds:.3f} rows={batch.rewards.size}",
        flush=True,
    )
    result: dict[str, object] = {
        "arm": ARM,
        "manabot": str(Path(manabot.__file__).parent),
        "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda": torch.version.cuda,
        "precision": precision,
        "streams": args.streams,
        "transitions": args.transitions,
        "learning": Learning(reference="action_type_uniform").model_dump(mode="json"),
        "collection_seconds": collection_seconds,
        "learner_rows": int(np.asarray(batch.rewards).size),
    }

    def updates(
        name: str, run: Callable[[object, RolloutBatch], UpdateDiagnostics]
    ) -> None:
        global _steps
        rates, seconds, steps = [], [], []
        receipts: list[dict[str, object]] = []
        reference_weights: dict[str, torch.Tensor] | None = None
        sections_by_repeat: list[object] = []
        for repeat in range(args.repeats + 1 + int(args.profile is not None)):
            print(f"{name} starting repeat={repeat}", flush=True)
            agent.load_state_dict(initial)
            learner = Learner(
                agent, device, minibatch_rows=args.streams, precision=precision
            )
            copy = deepcopy(batch)
            _sync(device)
            _steps = 0
            tick = time.perf_counter()
            profiled = args.profile is not None and repeat == args.repeats + 1
            if profiled:
                profiler = cProfile.Profile()
                activities = [torch.profiler.ProfilerActivity.CPU]
                if device.type == "cuda":
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                # Record one complete minibatch, then stop recording. The rest
                # of this update completes normally but is never a timing sample.
                with torch.profiler.profile(
                    activities=activities,
                    schedule=torch.profiler.schedule(
                        wait=0, warmup=0, active=1, repeat=1
                    ),
                    record_shapes=True,
                    with_stack=True,
                    on_trace_ready=lambda p: p.export_chrome_trace(
                        f"{args.profile}.{name}.trace.json"
                    ),
                ) as trace:
                    handle = register_optimizer_step_post_hook(lambda *_: trace.step())
                    try:
                        diagnostics = profiler.runcall(run, learner, copy)
                    finally:
                        handle.remove()
                profiler.dump_stats(f"{args.profile}.{name}")
                with Path(f"{args.profile}.{name}.txt").open("w") as stream:
                    pstats.Stats(profiler, stream=stream).sort_stats(
                        "cumtime"
                    ).print_stats(60)
            else:
                diagnostics = run(learner, copy)
            _sync(device)
            elapsed = time.perf_counter() - tick
            if repeat and not profiled:
                sections = learner.profiler.get_stats()
                sections_by_repeat.append(sections)
                rates.append(_steps / elapsed)
                seconds.append(elapsed)
                steps.append(_steps)
                digest = hashlib.sha256()
                weights = {
                    key: value.detach().cpu().clone()
                    for key, value in agent.state_dict().items()
                }
                if reference_weights is None:
                    reference_weights = weights
                for key, value in sorted(weights.items()):
                    digest.update(key.encode())
                    digest.update(value.numpy().tobytes())
                health = diagnostics["numerical"]
                assert health["optimizer_steps"] == _steps > 0
                assert health["rejected_steps"] == health["skipped_steps"] == 0
                receipts.append(
                    {
                        "weights_sha256": digest.hexdigest(),
                        "max_abs_weight_difference_from_first": max(
                            float((value - reference_weights[key]).abs().max())
                            for key, value in weights.items()
                            if value.is_floating_point() and value.numel()
                        ),
                        "optimizer_exposures": diagnostics["optimizer_exposures"],
                        "errors": health["rejected_steps"],
                        "diagnostics": diagnostics,
                    }
                )
                print(
                    f"{name} repeat={repeat} seconds={elapsed:.3f} steps={_steps}",
                    flush=True,
                )
                result[name] = {
                    "sections": sections,
                    "sections_by_repeat": sections_by_repeat,
                    "receipts": receipts,
                    "steps_per_second": _summary(rates),
                    "seconds": _summary(seconds),
                    "steps": steps,
                }
                _write_result(args.out, result)

    if not args.ppo_only:
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
    if args.ppo_only:
        if args.profile is not None:
            faulthandler.cancel_dump_traceback_later()
        collector.close()
        _write_result(args.out, result)
        print(
            json.dumps(
                {
                    "ppo": result["ppo"]["steps_per_second"],
                    "learner_rows": result["learner_rows"],
                }
            )
        )
        return

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
    _write_result(args.out, result)
    print(
        json.dumps({k: v for k, v in result.items() if k != "gpu"}, default=str)[:2000]
    )


if __name__ == "__main__":
    main()
