"""Joint capacity/collection grid anchored to MTG-122's corrected baseline.

Each process runs one ordinary TrainingRun at a fixed transition budget. The
existing iteration schedule maps its endpoints to the four-stream reference;
larger batches still change optimizer frequency and EMA age per transition.
Files stay in Etude outputs throughout execution, including interrupted runs.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import signal

import torch

from experiments.runners.size_frontier import admit_source, train
from manabot.arena.models import file_sha256
from manabot.model.agent import Agent
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.training.execution import atomic_json
from manabot.training.models import (
    AtaraxosMoveLearning,
    IterationSchedule,
    TrainingRegime,
    TrainingRun,
    TrainSelfPlay,
)
import managym


@dataclass(frozen=True)
class Capacity:
    name: str
    width: int
    depth: int


CAPACITIES = (
    Capacity("small", 64, 2),
    Capacity("middle", 192, 4),
    Capacity("large", 384, 8),
)
STREAMS = (4, 64, 128, 256, 512)
TRANSITIONS_PER_STREAM = 64
FEASIBILITY_TRANSITIONS = 131_072


def recipe(
    capacity: Capacity,
    streams: int,
    transitions: int,
    *,
    device: str = "cuda",
    seconds: float = 1800,
) -> TrainingRegime:
    if streams not in STREAMS or transitions % (streams * TRANSITIONS_PER_STREAM):
        raise ValueError("Grid budget must contain whole iterations for every cell")
    if transitions < streams * TRANSITIONS_PER_STREAM:
        raise ValueError("Grid budget must contain at least one iteration")
    result = TrainingRegime.model_validate_json(
        Path("experiments/regimes/early-learning-rate-mini-source.json").read_text()
    )
    result.id = f"batching-grid-{capacity.name}-s{streams}"
    result.world = "w6"
    result.agent.hidden_dim = capacity.width
    result.agent.attention_layers = capacity.depth
    # The baseline omits FF width: four attention heads imply 4*hidden_dim.
    # Keep that same resolved expansion rule across all three capacities.
    result.agent.attention_feedforward_dim = None
    stage = result.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    assert isinstance(stage.learning, AtaraxosMoveLearning)
    stage.streams, stage.transitions = streams, TRANSITIONS_PER_STREAM
    stage.updates = transitions // (streams * TRANSITIONS_PER_STREAM)
    stage.matchup_curriculum = "cross-balanced"
    stage.execution.device = device
    stage.execution.threads = stage.execution.workers = 1
    stage.execution.wall_seconds = seconds
    stage.execution.memory_bytes = 48 * 1024**3
    result.wall_seconds = seconds + 30
    stage.learning.iteration_schedule = (
        None
        if streams == 4
        else IterationSchedule(
            steps=stage.updates,
            reference_steps=transitions // (4 * TRANSITIONS_PER_STREAM),
        )
    )
    return TrainingRegime.model_validate(result.model_dump())


def summarize(run: TrainingRun, directory: Path) -> dict[str, object]:
    """Use completed run clocks; never infer progress from rental duration."""
    stage = run.stages[0]
    timings = {
        "collection_seconds": stage.collection_seconds,
        "learning_seconds": stage.learning_seconds,
        "run_seconds": run.seconds,
    }
    active = stage.collection_seconds + stage.learning_seconds
    result: dict[str, object] = {
        "status": run.status,
        "error": run.error,
        "seed": run.seed,
        "updates": run.updates_through(),
        "learner_transitions": stage.learner_transitions,
        "environment_decisions": stage.environment_decisions,
        "optimizer_exposures": stage.optimizer_exposures,
        **timings,
        "active_transitions_per_second": stage.learner_transitions / active
        if active > 0
        else None,
        "inclusive_transitions_per_second": stage.learner_transitions / run.seconds
        if run.seconds > 0
        else None,
        "peak_cuda_allocated": torch.cuda.max_memory_allocated()
        if torch.cuda.is_available()
        else None,
    }
    if run.status == "completed":
        agent, _ = load_checkpoint_agent(str(directory / "self-play-raw.pt"))
        assert isinstance(agent, Agent)
        result["parameters"] = sum(p.numel() for p in agent.parameters())
        result["checkpoint_finite"] = all(
            bool(torch.isfinite(p).all()) for p in agent.parameters()
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=[c.name for c in CAPACITIES], required=True)
    parser.add_argument("--streams", type=int, choices=STREAMS, required=True)
    parser.add_argument("--transitions", type=int, default=FEASIBILITY_TRANSITIONS)
    parser.add_argument("--seed", type=int, default=13571)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=1800)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    device = "cpu" if args.smoke else "cuda"
    if managym.WORLD_VERSION != "w6":
        raise ValueError("Grid requires corrected w6")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    def terminate(signum: int, frame: object) -> None:
        raise TimeoutError(f"Worker interrupted: signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    error: str | None = None
    try:
        admit_source(args.out, device)
        capacity = next(c for c in CAPACITIES if c.name == args.size)
        spec = recipe(
            capacity,
            args.streams,
            args.transitions,
            device=device,
            seconds=args.seconds,
        )
        trained = train(args.out / "training", spec, args.seed, milestones=True)
        atomic_json(
            args.out / "summary.json", summarize(trained, args.out / "training/run")
        )
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


if __name__ == "__main__":
    main()
