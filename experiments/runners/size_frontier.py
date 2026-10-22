"""Bounded capacity, inference-cost and fixed-batch numerical experiments.

The ordinary TrainingRun owns learning, checkpoints and scientific counters.
This runner only chooses frozen recipes and observes diagnostic boundaries.
Each Etude job owns one GPU; packaging follows measurement. CPU mode is solely
for local workflow checks, never a substitute for a CUDA result.
"""

import argparse
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import random
import signal
import time
from unittest.mock import patch

import numpy as np
import torch
from torch import Tensor

from experiments.recipes import gpu_training_profile
from experiments.runners.etude_progress import ProgressStore
from experiments.runners.inference_preparation import state_digest
from experiments.runners.learner_witnesses import learner_witnesses
from experiments.runners.profile_artifacts import package
from manabot.arena.models import file_sha256
from manabot.env import Match, Reward
from manabot.infra.hypers import RewardHypers
from manabot.model.agent import Agent
from manabot.model.inference import PolicyInference
from manabot.model.policy_distribution import legal_mask, policy_logs
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.sim.net_opponent import (
    NetOpponentTrainer,
    RolloutBatch,
    SeatRoutedCollector,
)
from manabot.training import objectives
from manabot.training.execution import atomic_json, execute_regime
from manabot.training.models import (
    AtaraxosMoveLearning,
    TrainingRegime,
    TrainingRun,
    TrainSelfPlay,
)
from manabot.training.selection import UpdateDiagnostics
from manabot.training.source_bundle import SourceBundle, source_commit


@dataclass(frozen=True)
class Size:
    name: str
    width: int
    depth: int
    feedforward: int


SIZES = (
    Size("small", 64, 1, 256),
    Size("middle", 192, 4, 768),
    Size("large", 384, 8, 1536),
)
MILESTONES = (128, 512, 1024, 2048)


def recipe(
    size: Size, updates: int, streams: int, device: str, seconds: float
) -> TrainingRegime:
    result = gpu_training_profile()
    result.id = f"size-frontier-{size.name}"
    result.agent.hidden_dim = size.width
    result.agent.attention_layers = size.depth
    result.agent.attention_feedforward_dim = size.feedforward
    stage = result.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.updates, stage.streams, stage.transitions = updates, streams, 8
    stage.execution.device, stage.execution.threads, stage.execution.workers = (
        device,
        1,
        1,
    )
    stage.execution.profile_collection = False
    stage.execution.wall_seconds, result.wall_seconds = seconds, seconds + 30
    return TrainingRegime.model_validate(result.model_dump())


def train(
    out: Path,
    spec: TrainingRegime,
    seed: int,
    *,
    milestones: bool = False,
    checkpoint_updates: tuple[int, ...] | None = None,
) -> TrainingRun:
    if milestones and checkpoint_updates is not None:
        raise ValueError("Choose legacy milestones or explicit checkpoint updates")
    if milestones:
        checkpoint_updates = tuple(n for n in MILESTONES if n < spec.stages[0].updates)
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "regime.json", spec.model_dump(mode="json"))
    output = os.environ.get("ETUDE_OUTPUT_DIR")
    with ProgressStore(
        out / "training.sqlite", Path(output) if output else None
    ) as store:
        run = execute_regime(
            spec,
            seed,
            out / "run",
            store,
            checkpoint_updates=checkpoint_updates,
        )
    atomic_json(
        out / "result.json",
        {
            "status": run.status,
            "updates": run.updates_through(),
            "error": run.error,
            "peak_cuda_allocated": torch.cuda.max_memory_allocated()
            if torch.cuda.is_available()
            else None,
        },
    )
    if run.status != "completed":
        raise RuntimeError(f"Training incomplete: {run.error}")
    return run


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def windows(
    call: Callable[[], None], device: str, seconds: float
) -> list[dict[str, float | int]]:
    for _ in range(3):
        call()
    synchronize(device)
    result = []
    for _ in range(3):
        start, calls = time.perf_counter(), 0
        while time.perf_counter() - start < seconds or calls < 5:
            call()
            synchronize(device)
            calls += 1
        result.append({"calls": calls, "seconds": time.perf_counter() - start})
    return result


def profile_checkpoint(
    path: Path, spec: TrainingRegime, out: Path, device: str, *, smoke: bool
) -> None:
    """Capture real requests separately, then measure a fixed observation pool."""
    out.mkdir()
    agent, space = load_checkpoint_agent(str(path))
    agent.to(device).eval()
    measurements: list[dict[str, object]] = []
    inputs: list[dict[str, Tensor]] = []
    prepare = PolicyInference.prepare
    for streams in (2,) if smoke else (64, 512):
        collector = SeatRoutedCollector(
            space,
            Match(spec.match),
            Reward(RewardHypers()),
            num_envs=streams,
            seed=13580,
            opponent_mode="self",
            device=device,
        )
        requests: list[dict[str, float | int]] = []
        start = time.perf_counter()

        def observe(
            scope: PolicyInference, obs: dict[str, Tensor]
        ) -> dict[str, Tensor]:
            requests.append(
                {
                    "rows": len(obs["actions_valid"]),
                    "elapsed_seconds": time.perf_counter() - start,
                }
            )
            # Only the 64-stream capture supplies the sweep; bounded real rows.
            if len(inputs) < 16 and streams == (2 if smoke else 64):
                inputs.append(
                    {key: value.detach().cpu().clone() for key, value in obs.items()}
                )
            return prepare(scope, obs)

        with patch.object(PolicyInference, "prepare", observe):
            collector.collect(agent, 1 if smoke else 8)
        synchronize(device)
        atomic_json(
            out / f"requests-s{streams}.json",
            {
                "requests": requests,
                "capture_seconds": time.perf_counter() - start,
                "stats": collector.stats.to_dict(),
                "timing": "observation capture adds overhead; not throughput",
            },
        )
        collector.close()
        collector = SeatRoutedCollector(
            space,
            Match(spec.match),
            Reward(RewardHypers()),
            num_envs=streams,
            seed=13580,
            opponent_mode="self",
            device=device,
        )
        clean = []
        try:
            for repeat in range(3):
                synchronize(device)
                before = collector.stats.to_dict()
                started = time.perf_counter()
                collector.collect(agent, 1 if smoke else 8)
                synchronize(device)
                clean.append(
                    {
                        "repeat": repeat,
                        "seconds": time.perf_counter() - started,
                        "before": before,
                        "after": collector.stats.to_dict(),
                    }
                )
        finally:
            collector.close()
        atomic_json(out / f"collection-s{streams}.json", clean)
    cpu = {key: torch.cat([part[key] for part in inputs]) for key in inputs[0]}
    torch.save(cpu, out / "observations.pt")
    for count in (1, 4) if smoke else (1, 4, 16, 64, 128, 256, 512):
        rows = torch.arange(count) % len(cpu["actions_valid"])
        obs = {key: value[rows] for key, value in cpu.items()}
        rng = torch.Generator(device=device).manual_seed(13581)
        with torch.no_grad(), PolicyInference(agent) as scope:
            prepared = scope.prepare(obs)

            def resident() -> None:
                agent.forward_distribution(prepared)

            def end_to_end() -> None:
                value = scope.prepare(obs)
                logits, values = agent.forward_distribution(value)
                logs = policy_logs(logits, legal_mask(value["actions_valid"]))
                probabilities = logs.exp()
                actions = torch.multinomial(probabilities, 1, generator=rng)
                # Include ordinary learner-path return materialization, not merely forward.
                actions.cpu().numpy()
                logs.gather(-1, actions).cpu().numpy()
                probabilities.cpu().numpy()
                logs.cpu().numpy()
                values.cpu().numpy()

            measurements.append(
                {
                    "batch": count,
                    "unique_rows_available": len(cpu["actions_valid"]),
                    "resident_forward": windows(
                        resident, device, 0.01 if smoke else 0.25
                    ),
                    "prepare_forward_sample_return": windows(
                        end_to_end, device, 0.01 if smoke else 0.25
                    ),
                }
            )
        atomic_json(out / "timings.json", measurements)
    atomic_json(
        out / "identity.json",
        {
            "checkpoint": str(path),
            "sha256": file_sha256(path),
            "parameters": sum(p.numel() for p in agent.parameters()),
            "input_sha256": file_sha256(out / "observations.pt"),
            "sweep_rows": "cycle retained real rows when batch exceeds available rows",
            "timing": "synchronized wall latency, three warmups and three repeated windows; not GPU utilization",
        },
    )


def profile(out: Path, device: str, *, smoke: bool) -> None:
    for size in SIZES[:1] if smoke else SIZES:
        spec = recipe(size, 1 if smoke else 128, 2 if smoke else 64, device, 500)
        cell = out / size.name
        train(cell, spec, 13570)
        for label in ("initial", "early"):
            path = (
                cell
                / "run"
                / ("policy-initial-raw.pt" if label == "initial" else "policy-raw.pt")
            )
            profile_checkpoint(path, spec, cell / label, device, smoke=smoke)


@dataclass
class ReplayState:
    model: dict[str, Tensor]
    optimizer: dict
    torch_rng: Tensor
    cuda_rng: list[Tensor]
    numpy_rng: tuple
    python_rng: tuple

    @classmethod
    def capture(cls, trainer: NetOpponentTrainer) -> "ReplayState":
        return cls(
            {
                key: tensor.detach().clone()
                for key, tensor in trainer.agent.state_dict().items()
            },
            deepcopy(trainer.optimizer.state_dict()),
            torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            np.random.get_state(),
            random.getstate(),
        )

    def restore(self, trainer: NetOpponentTrainer) -> None:
        trainer.agent.load_state_dict(self.model)
        trainer.optimizer.load_state_dict(deepcopy(self.optimizer))
        trainer.optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(self.torch_rng)
        if self.cuda_rng:
            torch.cuda.set_rng_state_all(self.cuda_rng)
        np.random.set_state(self.numpy_rng)
        random.setstate(self.python_rng)


def numerical(out: Path, device: str, *, smoke: bool) -> None:
    """Replay exactly one saved learner batch at a fixed, warmed optimizer state.

    Collection runs once. Every replay restores model, Adam and RNG; EMA remains
    frozen during this diagnostic and is never advanced between comparisons.
    Instrumented witnesses synchronize; this experiment makes no speed claim.
    """
    target = 1 if smoke else 16
    for streams in (2,) if smoke else (64, 512):
        size = SIZES[0] if smoke else SIZES[-1]
        spec = recipe(size, target, streams, device, 600)
        cell = out / f"s{streams}"
        original = objectives.update_move_iteration

        def replay(
            trainer: NetOpponentTrainer,
            batch: RolloutBatch,
            learning: AtaraxosMoveLearning,
            iteration: int,
            *,
            bootstrap_agent: Agent | None = None,
        ) -> UpdateDiagnostics:
            if iteration != target:
                return original(
                    trainer, batch, learning, iteration, bootstrap_agent=bootstrap_agent
                )
            state = ReplayState.capture(trainer)
            torch.save(
                {
                    "state": state,
                    "batch": batch,
                    "learning": learning.model_dump(),
                    "iteration": iteration,
                    "bootstrap": bootstrap_agent.state_dict()
                    if bootstrap_agent
                    else None,
                },
                cell / "replay-input.pt",
            )
            results: list[dict[str, object]] = []
            was_deterministic = torch.are_deterministic_algorithms_enabled()
            try:
                for deterministic in (False, True):
                    torch.use_deterministic_algorithms(deterministic)
                    for repeat in range(3):
                        state.restore(trainer)
                        child = (
                            cell
                            / f"{'deterministic' if deterministic else 'ordinary'}-{repeat}"
                        )
                        child.mkdir()
                        # Bind witnesses to the ordinary update, not this replay wrapper.
                        with (
                            patch.object(objectives, "update_move_iteration", original),
                            learner_witnesses(child),
                        ):
                            diagnostics = objectives.update_move_iteration(
                                trainer,
                                batch,
                                learning,
                                iteration,
                                bootstrap_agent=bootstrap_agent,
                            )
                        torch.save(
                            {
                                "model": trainer.agent.state_dict(),
                                "optimizer": trainer.optimizer.state_dict(),
                            },
                            child / "after.pt",
                        )
                        results.append(
                            {
                                "name": child.name,
                                "model": state_digest(trainer.agent.state_dict()),
                                "optimizer": state_digest(
                                    trainer.optimizer.state_dict()
                                ),
                                "diagnostics": diagnostics,
                            }
                        )
                        atomic_json(cell / "replays.json", results)
            finally:
                torch.use_deterministic_algorithms(was_deterministic)
                state.restore(trainer)
            return original(
                trainer, batch, learning, iteration, bootstrap_agent=bootstrap_agent
            )

        with patch.object(objectives, "update_move_iteration", replay):
            train(cell, spec, 10349)


def admit_source(out: Path, device: str) -> None:
    if "ETUDE_CONFIG" in os.environ:
        payload = json.loads(Path(os.environ["ETUDE_CONFIG"]).read_text())
        bundle = SourceBundle.model_validate(payload["source"])
        for name, digest in bundle.files.items():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or file_sha256(path) != digest:
                raise ValueError(f"Source mismatch: {name}")
        for path in Path("managym").glob("*.so"):
            bundle.files[str(path)] = file_sha256(path)
        atomic_json(".manabot-source.json", bundle.model_dump(mode="json"))
        atomic_json(out / "source-bundle.json", bundle.model_dump(mode="json"))
    if device == "cuda" and (
        not torch.cuda.is_available()
        or torch.cuda.get_device_name() != "NVIDIA GeForce RTX 5090"
    ):
        raise ValueError("Study requires one RTX 5090; no fallback")
    atomic_json(
        out / "hardware.json",
        {
            "source_commit": source_commit(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
            "platform": platform.platform(),
            "cpuinfo": Path("/proc/cpuinfo").read_text()
            if Path("/proc/cpuinfo").exists()
            else None,
            "gpu_bytes": torch.cuda.get_device_properties(0).total_memory
            if device == "cuda"
            else None,
            "tf32": False,
            "cpu_max": Path("/sys/fs/cgroup/cpu.max").read_text()
            if Path("/sys/fs/cgroup/cpu.max").exists()
            else None,
        },
    )


def main() -> None:
    def terminate(signum: int, frame: object) -> None:
        raise TimeoutError(f"Worker received signal {signum}; retain partial evidence")

    signal.signal(signal.SIGTERM, terminate)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("learning", "profile", "numerical"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--size", choices=[s.name for s in SIZES], default="small")
    parser.add_argument("--seed", type=int, default=13561)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--canary",
        action="store_true",
        help="Two CUDA updates; never a scientific trajectory",
    )
    args = parser.parse_args()
    device = "cpu" if args.smoke else "cuda"
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    error: str | None = None
    try:
        admit_source(args.out, device)
        if args.mode == "learning":
            size = next(s for s in SIZES if s.name == args.size)
            spec = recipe(
                size,
                2 if args.smoke or args.canary else 2048,
                2 if args.smoke else 64,
                device,
                3_600,
            )
            train(args.out / "training", spec, args.seed, milestones=True)
        elif args.mode == "profile":
            profile(args.out, device, smoke=args.smoke)
        else:
            numerical(args.out, device, smoke=args.smoke)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        atomic_json(
            args.out / "status.json",
            {"status": "failed" if error else "completed", "error": error},
        )
        atomic_json(
            args.out / "manifest.json",
            [
                {
                    "path": str(p.relative_to(args.out)),
                    "bytes": p.stat().st_size,
                    "sha256": file_sha256(p),
                }
                for p in sorted(args.out.rglob("*"))
                if p.is_file() and p.name != "manifest.json"
            ],
        )
        if args.artifacts:
            package(args.out, args.artifacts)


if __name__ == "__main__":
    main()
