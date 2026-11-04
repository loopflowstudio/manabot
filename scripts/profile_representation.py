"""Bounded operator diagnosis through the ordinary representation-study learner.

One excluded update retains normal training receipts, Python call costs and one
operator trace per inference/learning forward. These are diagnosis, not study
scores or uninstrumented throughput admission. CUDA runs require the caller's
existing quoted rental, deadline, stop file and teardown owner.
"""

import argparse
from contextlib import ExitStack
import cProfile
from datetime import datetime
from decimal import Decimal
import os
from pathlib import Path
import pstats
import sys
import time
from typing import Callable
from unittest.mock import patch

from etude.launch.records import Run
import torch

from experiments.state_representation_gpu import (
    ADMISSION_SEED,
    ProgressWatch,
    StudyJob,
    _run_child,
    recipe,
    runs as admission_runs,
)
from manabot.infra.files import atomic_json
from manabot.infra.source import bind_etude_source
from manabot.model.actor_critic import ActorCritic
from manabot.model.shared_effects import SharedEffectEncoder
from manabot.model.state_records import TypedContinuationEncoder
from manabot.training.execution import execute_regime
from manabot.training.regime import TrainingRegime, TrainSelfPlay
from manabot.verify.store import VerifyStore


def runs(revision: str) -> list[Run]:
    """Declare one capped diagnostic rental; no scientific seed is used."""
    (run,) = admission_runs(revision)
    job = run.job.model_copy(
        update={
            "name": "representation-operator-profile",
            "command": run.job.command.replace(
                "-m experiments.state_representation_gpu",
                "scripts/profile_representation.py --worker",
            ),
            "until": "updates=1",
            "max_hours": 0.5,
            "max_usd": Decimal("0.50"),
        }
    )
    return job.runs(0)


def worker() -> None:
    """Use the existing progress/stop owner and rental deadline for each child."""
    bind_etude_source()
    spec = StudyJob.model_validate_json(Path(os.environ["ETUDE_CONFIG"]).read_bytes())
    out = Path(os.environ["ETUDE_OUTPUT_DIR"])
    deadline = datetime.fromisoformat(os.environ["ETUDE_DEADLINE"]).timestamp()
    stop = Path(os.environ["ETUDE_STOP_FILE"])
    for arm in ("typed_pool", "typed_gru"):
        logs = out / f"{arm}-supervision"
        logs.mkdir(parents=True)
        child_stop = logs / "stop"
        _run_child(
            [
                sys.executable,
                "scripts/profile_representation.py",
                "--arm",
                arm,
                "--device",
                "cuda",
                "--out",
                str(out / arm),
                "--stop",
                str(child_stop),
            ],
            logs,
            stop,
            min(deadline - 120, time.time() + 720),
            watch=ProgressWatch(
                arm,
                out / arm / "training/run.json",
                spec.admission_limits,
                time.monotonic(),
            ),
            child_stop=child_stop,
        )
    atomic_json(out / "etude-result.json", {"clocks": {"updates": 1}})


def main() -> None:
    if sys.argv[1:] == ["--worker"]:
        worker()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("typed_pool", "typed_gru"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stop", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=600)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    plan = recipe(args.arm)
    plan.id = f"excluded-operator-diagnosis-{args.arm}"
    stage = plan.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.updates = 1
    stage.execution.device = args.device
    stage.execution.wall_seconds = plan.wall_seconds = args.seconds
    plan = TrainingRegime.model_validate(plan.model_dump())
    activities = [torch.profiler.ProfilerActivity.CPU]
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA diagnosis requires CUDA")
        torch.cuda.set_per_process_memory_fraction(0.8)
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    original = ActorCritic.forward_distribution
    captured: set[str] = set()

    def forward(
        model: ActorCritic, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        phase = "learning" if torch.is_grad_enabled() else "inference"
        if phase in captured:
            return original(model, obs)
        captured.add(phase)
        with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
            result = original(model, obs)
        prof.export_chrome_trace(str(args.out / f"{phase}.trace.json"))
        (args.out / f"{phase}.operators.txt").write_text(
            prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=40)
        )
        return result

    def labelled[**P, T](method: Callable[P, T], name: str) -> Callable[P, T]:
        # Diagnostic wrappers retain arbitrary encoder signatures; the model's
        # public typed methods and parameters are unchanged.
        def call(*values: P.args, **kwargs: P.kwargs) -> T:
            with torch.profiler.record_function(name):
                return method(*values, **kwargs)

        return call

    calls = cProfile.Profile()
    with ExitStack() as stack:
        stack.enter_context(patch.object(ActorCritic, "forward_distribution", forward))
        for encoder in (SharedEffectEncoder, TypedContinuationEncoder):
            stack.enter_context(
                patch.object(
                    encoder, "forward", labelled(encoder.forward, encoder.__name__)
                )
            )
        store = stack.enter_context(VerifyStore(args.out / "training.sqlite"))
        try:
            with calls:
                run = execute_regime(
                    plan,
                    ADMISSION_SEED,
                    args.out / "training",
                    store,
                    stop_path=args.stop,
                    stop_reserve_seconds=120,
                )
        finally:
            calls.dump_stats(str(args.out / "python.prof"))
            with (args.out / "python.txt").open("w") as output:
                pstats.Stats(calls, stream=output).strip_dirs().sort_stats(
                    "cumulative"
                ).print_stats(80)
    atomic_json(
        args.out / "diagnosis.json",
        {
            "status": run.status,
            "updates": run.updates_through(),
            "device": args.device,
            "arm": args.arm,
            "phases": sorted(captured),
            "scientific": False,
        },
    )
    if run.status != "completed":
        raise RuntimeError(f"diagnosis incomplete: {run.status}: {run.error}")


if __name__ == "__main__":
    main()
