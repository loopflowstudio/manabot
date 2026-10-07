"""Bounded CUDA capacity measurements on a shared, fixed real observation batch.

Model-only Adam measures a diagnostic legal-policy/value loss, not RL. Separate
ordinary TrainingRuns measure collection, actual filtered learning, persistence,
and export. Each cell has a fresh process, preserving failures and OOMs without
carrying a poisoned CUDA context into the next cell.
"""

import argparse
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Literal

from pydantic import BaseModel, Field
import torch

from experiments.runners.model_capacity import regimes
from manabot.arena.models import file_sha256
from manabot.env import ObservationSpace
from manabot.model.agent import Agent
from manabot.model.architecture import ArchitectureReceipt, architecture_receipt
from manabot.training.execution import atomic_json, execute_regime
from manabot.training.models import TrainingRegime, TrainSelfPlay
from manabot.verify.store import VerifyStore


class Cell(BaseModel):
    capacity: str
    kind: Literal["model", "loop"]
    batch: int = Field(gt=0)
    streams: int = Field(default=0, ge=0)

    @property
    def name(self) -> str:
        return f"{self.capacity}-{self.kind}-b{self.batch}-s{self.streams}"


class Attempt(BaseModel):
    cell: Cell
    status: str = "running"
    seconds: float = 0
    exit_code: int | None = None
    error: str | None = None


class Sweep(BaseModel):
    status: str = "running"
    input_sha256: str
    source_commit: str
    deadline_unix: float
    started_unix: float = Field(default_factory=time.time)
    seconds: float = 0
    attempts: list[Attempt] = Field(default_factory=list)
    expected_cells: list[Cell] = Field(default_factory=list)


class Window(BaseModel):
    calls: int
    observations: int
    seconds: float


class PhaseResult(BaseModel):
    construction_seconds: float
    first_call_seconds: float
    windows: list[Window]
    peak_allocated_bytes: int
    peak_reserved_bytes: int


class ModelResult(BaseModel):
    batch: int
    unique_input_rows_available: int
    input_sha256: str
    input_shapes: dict[str, list[int]]
    valid_counts: dict[str, float]
    padding: str
    warmups: int = 2
    windows: int = 3
    target_window_seconds: float = 1
    precision: str = "float32; TF32 disabled"
    device: str
    total_memory_bytes: int
    torch: str
    python: str
    native_sha256: dict[str, str]
    architecture: ArchitectureReceipt | None = None
    attention_slots: int | None = None
    inference: PhaseResult | None = None
    optimizer: PhaseResult | None = None


def load_observations(path: Path) -> dict[str, torch.Tensor]:
    value: object = torch.load(path, weights_only=True, map_location="cpu")
    if not isinstance(value, dict) or not value:
        raise ValueError("observation artifact must contain named tensors")
    result: dict[str, torch.Tensor] = {}
    for key, tensor in value.items():
        if (
            not isinstance(key, str)
            or not isinstance(tensor, torch.Tensor)
            or tensor.ndim < 1
        ):
            raise ValueError("observation artifact contains a non-tensor row")
        result[key] = tensor
    if "actions_valid" not in result or len({t.shape[0] for t in result.values()}) != 1:
        raise ValueError("observation rows are not aligned")
    if result["actions_valid"].shape[0] == 0:
        raise ValueError("observation artifact is empty")
    return result


def cells() -> list[Cell]:
    capacities = ("w64-d1", "w64-d2", "w128-d2", "w384-d8")
    return [
        Cell(capacity=c, kind="model", batch=b)
        for c in capacities
        for b in (1, 4, 16, 64, 256, 1024)
    ] + [
        Cell(capacity=c, kind="loop", batch=b, streams=s)
        for c in capacities
        for b in (128, 512)
        for s in (4, 16, 64)
    ]


def _model_step(
    agent: Agent, obs: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer | None
) -> None:
    if optimizer is None:
        with torch.inference_mode():
            agent(obs)
    else:
        optimizer.zero_grad(set_to_none=True)
        logits, value = agent(obs)
        valid = obs["actions_valid"].bool()
        loss = (
            -(
                logits.log_softmax(-1).masked_fill(~valid, 0).sum(-1) / valid.sum(-1)
            ).mean()
            + value.square().mean()
        )
        loss.backward()
        optimizer.step()


def _model(recipe: TrainingRegime, input_path: Path, batch: int, out: Path) -> None:
    cpu = load_observations(input_path)
    available = cpu["actions_valid"].shape[0]
    indexes = torch.arange(batch) % available
    obs = {key: value[indexes].to("cuda") for key, value in cpu.items()}
    result = ModelResult.model_validate(
        {
            "batch": batch,
            "unique_input_rows_available": available,
            "input_sha256": file_sha256(input_path),
            "input_shapes": {k: list(v.shape) for k, v in obs.items()},
            "valid_counts": {
                k: float(v.sum()) for k, v in obs.items() if k.endswith("_valid")
            },
            "padding": "unchanged fixed slots; no packing; batches above available rows cycle exact real rows",
            "warmups": 2,
            "windows": 3,
            "target_window_seconds": 1.0,
            "precision": "float32; TF32 disabled",
            "device": torch.cuda.get_device_name(),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "native_sha256": {
                p.name: file_sha256(p) for p in Path("managym").glob("*.so")
            },
        }
    )
    for phase in ("inference", "optimizer"):
        torch.manual_seed(10349)
        began = time.perf_counter()
        agent = Agent(ObservationSpace(recipe.observation), recipe.agent).to("cuda")
        optimizer = (
            torch.optim.Adam(agent.parameters(), lr=1e-4)
            if phase == "optimizer"
            else None
        )
        agent.train(optimizer is not None)
        torch.cuda.synchronize()
        construction = time.perf_counter() - began
        result.architecture = architecture_receipt(agent)
        lengths: list[int] = []

        def capture(module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            lengths.append(args[0].shape[1])

        hook = agent.attention.register_forward_pre_hook(capture)

        torch.cuda.reset_peak_memory_stats()
        cold = time.perf_counter()
        _model_step(agent, obs, optimizer)
        torch.cuda.synchronize()
        cold_seconds = time.perf_counter() - cold
        hook.remove()
        result.attention_slots = lengths[0]
        for _ in range(2):
            _model_step(agent, obs, optimizer)
        windows: list[Window] = []
        for _ in range(3):
            torch.cuda.synchronize()
            began, calls = time.perf_counter(), 0
            while calls < 1000:
                _model_step(agent, obs, optimizer)
                torch.cuda.synchronize()
                calls += 1
                if time.perf_counter() - began >= 1:
                    break
            windows.append(
                Window(
                    calls=calls,
                    observations=calls * batch,
                    seconds=time.perf_counter() - began,
                )
            )
        measured = PhaseResult(
            construction_seconds=construction,
            first_call_seconds=cold_seconds,
            windows=windows,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        if phase == "inference":
            result.inference = measured
        else:
            result.optimizer = measured
        atomic_json(out / "result.json", result.model_dump(mode="json"))
        del agent, optimizer
        torch.cuda.empty_cache()


def child(root: Path, cell: Cell) -> None:
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; no CPU fallback")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base = TrainingRegime.model_validate_json((root / "baseline.json").read_text())
    recipe = regimes(base, include_ataraxos=True)[cell.capacity]
    out = root / cell.name
    if cell.kind == "model":
        _model(recipe, root / "observations.pt", cell.batch, out)
    else:
        recipe.stages = recipe.stages[:1]
        stage = recipe.stages[0]
        assert isinstance(stage, TrainSelfPlay)
        stage.streams, stage.transitions, stage.updates = (
            cell.streams,
            cell.batch // cell.streams,
            3,
        )
        stage.execution.device, stage.execution.threads = "cuda", 1
        stage.execution.wall_seconds, recipe.wall_seconds = 55, 60
        recipe = TrainingRegime.model_validate(recipe.model_dump())
        atomic_json(out / "regime.json", recipe.model_dump(mode="json"))
        torch.cuda.reset_peak_memory_stats()
        with VerifyStore(out / "training.sqlite") as store:
            run = execute_regime(recipe, 10349, out / "run", store)
        atomic_json(
            out / "result.json",
            {
                "status": run.status,
                "run_id": run.id,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "warmup": "first ordinary collect/update is reported separately; next two are repeated steady observations",
            },
        )
        if run.status != "completed":
            raise ValueError(
                "complete-loop cell failed; retained TrainingRun owns failure"
            )


def sweep(root: Path, seconds: float) -> None:
    started = time.time()
    report = Sweep(
        input_sha256=file_sha256(root / "observations.pt"),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        deadline_unix=started + seconds,
        expected_cells=cells(),
    )
    try:
        for cell in cells():
            remaining = report.deadline_unix - time.time()
            if remaining < 75:
                report.status = "budget-exhausted"
                break
            out = root / cell.name
            out.mkdir()
            attempt = Attempt(cell=cell)
            report.attempts.append(attempt)
            atomic_json(root / "sweep.json", report.model_dump(mode="json"))
            began = time.time()
            try:
                with (out / "worker.log").open("w") as log:
                    process = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "experiments.runners.cuda_performance",
                            "--root",
                            str(root),
                            "--cell",
                            cell.model_dump_json(),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=min(70, remaining),
                    )
                attempt.exit_code = process.returncode
                attempt.status = "completed" if process.returncode == 0 else "failed"
                if (
                    process.returncode
                    and "OutOfMemoryError" in (out / "worker.log").read_text()
                ):
                    attempt.status = "oom"
                    attempt.error = (
                        "CUDA allocation failed; partial phase results and log retained"
                    )
            except subprocess.TimeoutExpired:
                attempt.status, attempt.error = (
                    "timeout",
                    "cell exceeded 70-second process allowance",
                )
            attempt.seconds = time.time() - began
            report.seconds = time.time() - started
            atomic_json(root / "sweep.json", report.model_dump(mode="json"))
        else:
            report.status = (
                "completed-with-retained-failures"
                if any(a.status != "completed" for a in report.attempts)
                else "completed"
            )
    finally:
        report.seconds = time.time() - started
        atomic_json(root / "sweep.json", report.model_dump(mode="json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=1200)
    parser.add_argument("--cell")
    args = parser.parse_args()
    if args.cell:
        child(args.root, Cell.model_validate_json(args.cell))
    else:
        sweep(args.root, args.seconds)


if __name__ == "__main__":
    main()
