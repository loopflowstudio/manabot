"""Bounded operator diagnosis through the ordinary representation-study learner.

One excluded update retains normal training receipts, Python call costs, one
inference trace and one complete learning minibatch through Adam. These are
diagnosis, not study scores or uninstrumented throughput admission. CUDA runs
require the caller's existing quoted rental, deadline, stop file and teardown owner.
"""

import argparse
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
import cProfile
from datetime import datetime
from decimal import Decimal
import os
from pathlib import Path
import pstats
import subprocess
import sys
import time
from unittest.mock import patch

from etude.launch.records import Input, Job, Run
import torch

from experiments.state_representation_gpu import (
    ADMISSION_SEED,
    ARMS,
    EXPERIMENT,
    AdmissionLimits,
    ProgressWatch,
    StudyJob,
    _run_child,
    recipe,
)
from manabot.data import load_manifest
from manabot.experiment.job import TOOLCHAIN_ENV, machine_setup
from manabot.infra.files import atomic_json
from manabot.infra.source import bind_etude_source, bundle_at
from manabot.model.actor_critic import ActorCritic
from manabot.model.shared_effects import SharedEffectEncoder
from manabot.model.state_records import TypedContinuationEncoder
from manabot.training.execution import execute_regime
from manabot.training.learning.health import OptimizerHealth
from manabot.training.regime import TrainingRegime, TrainSelfPlay
from manabot.verify.store import VerifyStore

REFERENCE = "experiments/data/state-representation-gpu-5da5f4/encoder-reference"


def check_references(root: Path, out: Path, device: str) -> None:
    """Check retained pre-batching CPU outputs and gradients on this device.

    Inputs are frozen tensor fixtures, not an admission of their historical game
    world. Etude verifies their manifest digests before starting the worker.
    """
    torch.set_num_threads(1)
    results: list[dict[str, str | float | int]] = []
    for arm in ARMS:
        for kind in ("effects", "continuation"):
            path = root / f"{arm}-{kind}.pt"
            # The trusted retained boundary contains only tensors/collections;
            # weights_only prevents unpickling arbitrary fixture code.
            saved = torch.load(path, map_location="cpu", weights_only=True)
            assert isinstance(saved, dict)
            inputs = saved["inputs"]
            assert isinstance(inputs, list) and all(
                isinstance(t, torch.Tensor) for t in inputs
            )
            tensors = [t.to(device).detach().clone() for t in inputs]
            object_index = 3 if kind == "effects" else 1
            tensors[object_index].requires_grad_()
            if kind == "continuation":
                tensors[2].requires_grad_()
            encoder = (
                SharedEffectEncoder(8, arm)
                if kind == "effects"
                else TypedContinuationEncoder(8, arm)
            ).to(device)
            encoder.load_state_dict(saved["state"], strict=True)
            result = encoder(*tensors)
            outputs = [result] if isinstance(result, torch.Tensor) else list(result)
            max_error = 0.0

            def compare(
                actual: torch.Tensor | None, expected: torch.Tensor | None
            ) -> None:
                nonlocal max_error
                if expected is None:
                    assert actual is None
                    return
                assert actual is not None
                actual = actual.detach().cpu()
                # Float32 CUDA/CPU accumulation order differs. The absolute
                # floor also admits near-zero gradients; record the actual error.
                torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
                max_error = max(max_error, float((actual - expected).abs().max()))

            for actual, expected in zip(outputs, saved["outputs"], strict=True):
                compare(actual, expected)
            sum(t.square().sum() for t in outputs).backward()
            compare(tensors[object_index].grad, saved["object_grad"])
            if kind == "continuation":
                compare(tensors[2].grad, saved["definition_grad"])
            for name, parameter in encoder.named_parameters():
                compare(parameter.grad, saved["gradients"][name])
            results.append(
                {
                    "arm": arm,
                    "kind": kind,
                    "max_absolute_error": max_error,
                    "absent_parameter_gradients": sum(
                        p.grad is None for p in encoder.parameters()
                    ),
                }
            )
    atomic_json(out, {"device": device, "rtol": 2e-4, "atol": 2e-4, "checks": results})


class OperatorTraces:
    """Capture bounded ordinary work, retaining partial traces on failure.

    Learning starts at the first grad-enabled policy forward and ends only after
    the corresponding OptimizerHealth step (backward, clipping, Adam and health
    checks). No loss, gradient, optimizer or random state is changed. Collection
    and later minibatches run outside the operator profiler.
    """

    def __init__(self, out: Path, device: str) -> None:
        self.out = out
        self.activities = [torch.profiler.ProfilerActivity.CPU]
        if device == "cuda":
            self.activities.append(torch.profiler.ProfilerActivity.CUDA)
        self.completed: set[str] = set()
        self.active: str | None = None
        self.profiler: torch.profiler.profile | None = None

    def _start(self, phase: str) -> bool:
        if phase in self.completed or self.active is not None:
            return False
        self.profiler = torch.profiler.profile(
            activities=self.activities, record_shapes=True
        )
        self.profiler.__enter__()
        self.active = phase
        return True

    def _finish(self, *, complete: bool) -> None:
        if self.profiler is None:
            return
        prof, phase = self.profiler, self.active
        self.profiler, self.active = None, None
        prof.__exit__(None, None, None)
        if complete:
            assert phase is not None
            self.completed.add(phase)
        name = phase if complete else f"{phase}.partial"
        prof.export_chrome_trace(str(self.out / f"{name}.trace.json"))
        averages = prof.key_averages()
        tables = averages.table(sort_by="self_cpu_time_total", row_limit=40)
        if torch.profiler.ProfilerActivity.CUDA in self.activities:
            tables += "\n" + averages.table(
                sort_by="self_cuda_time_total", row_limit=40
            )
        (self.out / f"{name}.operators.txt").write_text(tables)

    @contextmanager
    def capture(self) -> Iterator[None]:
        original_forward = ActorCritic.forward_distribution
        original_step = OptimizerHealth.step

        def forward(
            model: ActorCritic, obs: dict[str, torch.Tensor]
        ) -> tuple[torch.Tensor, torch.Tensor]:
            phase = "learning" if torch.is_grad_enabled() else "inference"
            started = self._start(phase)
            with torch.profiler.record_function(f"{phase}.forward"):
                result = original_forward(model, obs)
            if started and phase == "inference":
                self._finish(complete=True)
            return result

        def step(
            observer: OptimizerHealth,
            loss: torch.Tensor,
            max_norm: float,
            *,
            checked: bool = False,
        ) -> torch.Tensor:
            result = original_step(observer, loss, max_norm, checked=checked)
            if self.active == "learning":
                self._finish(complete=True)
            return result

        def labelled[**P, T](method: Callable[P, T], name: str) -> Callable[P, T]:
            def call(*values: P.args, **kwargs: P.kwargs) -> T:
                with torch.profiler.record_function(name):
                    return method(*values, **kwargs)

            return call

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(ActorCritic, "forward_distribution", forward)
            )
            stack.enter_context(patch.object(OptimizerHealth, "step", step))
            for owner, method, name in (
                (SharedEffectEncoder, "forward", "SharedEffectEncoder"),
                (TypedContinuationEncoder, "forward", "TypedContinuationEncoder"),
                (OptimizerHealth, "_backward", "learning.backward_and_clipping"),
                (OptimizerHealth, "_optimize", "learning.optimizer_and_health"),
            ):
                stack.enter_context(
                    patch.object(owner, method, labelled(getattr(owner, method), name))
                )
            try:
                yield
            finally:
                self._finish(complete=False)


def runs(revision: str) -> list[Run]:
    """Declare one capped diagnostic rental; no scientific seed is used."""
    root = Path(__file__).resolve().parents[1]
    spec = StudyJob(
        source_bundle=bundle_at(root, revision),
        admission_limits=AdmissionLimits(scientific_usd=Decimal("15.50")),
    )
    manifest = load_manifest()
    inputs = tuple(
        Input(
            name=Path(path).name,
            key=f"sha256/{manifest.files[path].sha256}",
            sha256=manifest.files[path].sha256,
            size_bytes=manifest.files[path].bytes,
        )
        for path in manifest.select([REFERENCE])
    )
    # This job must deliver its traces before separate throughput admission.
    job = Job(
        name="representation-operator-profile",
        experiment=EXPERIMENT,
        setup=machine_setup(" --extra dev --extra artifacts --extra evaluation"),
        command=f"{TOOLCHAIN_ENV}\nuv run --locked --extra dev --extra artifacts --extra evaluation python scripts/profile_representation.py --worker",
        repo=str(root),
        revision=revision,
        config=spec.model_dump_json(),
        inputs=inputs,
        until="updates=1",
        max_hours=0.5,
        max_usd=Decimal("0.50"),
        max_attempts=1,
    )
    return job.runs(0)


def worker() -> None:
    """Use the existing progress/stop owner and rental deadline for each child."""
    bind_etude_source()
    spec = StudyJob.model_validate_json(Path(os.environ["ETUDE_CONFIG"]).read_bytes())
    out = Path(os.environ["ETUDE_OUTPUT_DIR"])
    deadline = datetime.fromisoformat(os.environ["ETUDE_DEADLINE"]).timestamp()
    stop = Path(os.environ["ETUDE_STOP_FILE"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA diagnosis requires CUDA")
    check_references(
        Path(os.environ["ETUDE_INPUTS_DIR"]), out / "cuda-parity.json", "cuda"
    )
    print((out / "cuda-parity.json").read_text(), flush=True)
    with (out / "cuda-routing-tests.txt").open("w") as log:
        routing = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/model/test_shared_effects.py",
                "tests/model/test_state_records.py",
                "-k",
                "cuda and (batched_effect_routing or continuation_batches or unused_continuation)",
                "-q",
            ],
            check=False,
            timeout=120,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    print((out / "cuda-routing-tests.txt").read_text(), flush=True)
    routing.check_returncode()
    for arm in ARMS:
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
        print((out / arm / "diagnosis.json").read_text(), flush=True)
        for phase in ("inference", "learning"):
            print(f"{arm} {phase} operators:", flush=True)
            print((out / arm / f"{phase}.operators.txt").read_text(), flush=True)
    # Finish this job so Etude publishes the trace inventory before any separate
    # throughput admission. A lost admission must not erase completed diagnosis.
    atomic_json(out / "etude-result.json", {"clocks": {"updates": 1}})


def main() -> None:
    if sys.argv[1:] == ["--worker"]:
        worker()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
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
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA diagnosis requires CUDA")
        torch.cuda.set_per_process_memory_fraction(0.8)
    traces = OperatorTraces(args.out, args.device)
    calls = cProfile.Profile()
    with traces.capture(), VerifyStore(args.out / "training.sqlite") as store:
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
            "phases": sorted(traces.completed),
            "learning_scope": "first forward/loss/backward/clipping/Adam/health minibatch",
            "scientific": False,
        },
    )
    if run.status != "completed":
        raise RuntimeError(f"diagnosis incomplete: {run.status}: {run.error}")
    if traces.completed != {"inference", "learning"}:
        raise RuntimeError(f"operator capture incomplete: {traces.completed}")


if __name__ == "__main__":
    main()
