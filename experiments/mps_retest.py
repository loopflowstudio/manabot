"""Bucketed MPS retest: real capture, isolated arm measurements and report.

Run `capture` on the rented M2, then `matrix` for isolated correctness and
per-call blocks, and `replays` for the three whole-update repetitions. `report`
keeps incomplete matrices and unsupported arms visible. Every worker records its
host, source revision, capture, dtype and load. Backend recommendations belong
to the report's interpretation, never to a hard-coded threshold.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Literal, Protocol
import uuid
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if TYPE_CHECKING:
    import torch

    from manabot.model.actor_critic import ActorCritic
    from manabot.policy.protocol import Settings, Targets

Arm = Literal["cpu8", "mps-f32", "mps-bf16", "mps-f16", "mlx-f32"]
ARMS: tuple[Arm, ...] = ("cpu8", "mps-f32", "mps-bf16", "mps-f16", "mlx-f32")
HALF = ("mps-bf16", "mps-f16")
WIDTHS = (64, 96, 160, 384, 512, 768, 1024)
STREAMS = (256, 512, 1024)
LAYERS = {64: 2, 96: 4, 160: 8, 384: 8, 512: 8, 768: 8, 1024: 8}
SIZES = {"learner": (64, 256, 512, 1024, 2048), "collection": (64, 256, 512)}
LABELS = {
    "cpu8": "PyTorch CPU8 · float32",
    "mps-f32": "PyTorch MPS · float32",
    "mps-bf16": "PyTorch MPS · bfloat16",
    "mps-f16": "PyTorch MPS · float16",
    "mlx-f32": "MLX · float32",
}


@dataclass(frozen=True)
class Batch:
    observations: dict[str, np.ndarray]
    targets: Targets | None = None
    settings: Settings | None = None

    @property
    def rows(self) -> int:
        return len(self.observations["actions_valid"])


@dataclass(frozen=True)
class Evaluation:
    logits: np.ndarray
    values: np.ndarray
    loss: float
    gradients: dict[str, np.ndarray]


class Runner(Protocol):
    def evaluate(self, batch: Batch) -> Evaluation: ...
    def step(self, batch: Batch) -> float: ...
    def collect(self, batch: Batch) -> np.ndarray: ...
    def begin_collection(self) -> None: ...
    def end_collection(self) -> None: ...
    def reset(self) -> None: ...
    def sync(self) -> None: ...
    def memory(self) -> dict[str, int]: ...
    def close(self) -> None: ...


def _read(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        return {k: arrays[k] for k in arrays.files}


def _emit(path: Path, row: dict[str, Any]) -> None:
    """JSON is the evidence boundary; reject NaN instead of emitting invalid JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
    print(json.dumps(row, allow_nan=False), flush=True)


def _rows(path: Path) -> list[dict[str, Any]]:
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def _host() -> dict[str, Any]:
    from importlib.metadata import version

    from etude.eval.source import source_commit
    import torch

    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "chip": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        "memory_bytes": int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        ),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mlx": version("mlx"),
        "cpu_threads": torch.get_num_threads(),
        "revision": source_commit([__file__]),
    }


def _quiet(laptop: bool) -> float:
    load = os.getloadavg()[0]
    if laptop and load >= 8:
        raise RuntimeError(f"not measured: laptop load {load:.2f} is not below 8")
    return load


def _agent(capture: Path, width: int) -> ActorCritic:
    import torch

    from manabot.model.actor_critic import ActorCritic
    from manabot.model.checkpoint import load_checkpoint

    agent, space = load_checkpoint(str(capture / "initial.pt"), bucketed_shapes=True)
    if agent.config.hidden_dim != width:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(21501)
            agent = ActorCritic(
                space,
                agent.config.model_copy(
                    update={
                        "hidden_dim": width,
                        "attention_layers": LAYERS[width],
                        "num_attention_heads": 4 if width == 64 else 8,
                    }
                ),
                bucketed_shapes=True,
            )
    return agent


def _batches(capture: Path, phase: str) -> Iterator[Batch]:
    from manabot.backends.parity import minibatches

    if phase == "learner":
        for obs, targets, settings, _ in minibatches(capture, None):
            yield Batch(obs, targets, settings)
    else:
        for path in sorted((capture / "collection").glob("*.npz")):
            yield Batch(_read(path))


def _batch(capture: Path, phase: str, rows: int) -> Batch:
    """Consecutive recorded rows, without wrapping/repeating a short capture."""
    from manabot.policy.protocol import Targets

    parts: list[Batch] = []
    remaining = rows
    for batch in _batches(capture, phase):
        take = min(remaining, batch.rows)
        target = batch.targets
        if target is not None:
            target = Targets(**{k: v[:take] for k, v in vars(target).items()})
        parts.append(
            Batch(
                {k: v[:take] for k, v in batch.observations.items()},
                target,
                batch.settings,
            )
        )
        remaining -= take
        if not remaining:
            break
    if remaining:
        raise ValueError(f"capture has fewer than {rows} {phase} rows")
    obs = {
        k: np.concatenate([p.observations[k] for p in parts])
        for k in parts[0].observations
    }
    targets = None
    if phase == "learner":
        if any(p.settings != parts[0].settings for p in parts):
            raise ValueError("cannot combine rows with different learner settings")
        targets = Targets(
            **{
                k: np.concatenate([getattr(p.targets, k) for p in parts])
                for k in vars(parts[0].targets)
            }
        )
    return Batch(obs, targets, parts[0].settings)


class TorchRunner:
    """Production forward/loss, with harness-only MPS autocast and f16 scaling."""

    def __init__(self, agent: ActorCritic, arm: Arm) -> None:
        import torch

        self.agent = agent.to("cpu" if arm == "cpu8" else "mps").train()
        self.device = next(agent.parameters()).device
        self.dtype = {"mps-bf16": torch.bfloat16, "mps-f16": torch.float16}.get(arm)
        self.initial = {k: v.detach().clone() for k, v in agent.state_dict().items()}
        self.scaled = arm == "mps-f16"
        self.inference = None
        self._peak = 0
        self._stop = threading.Event()
        self._monitor: threading.Thread | None = None
        if self.device.type == "mps":
            self._monitor = threading.Thread(target=self._sample_memory, daemon=True)
            self._monitor.start()
        self.reset()

    def _sample_memory(self) -> None:
        import torch

        while not self._stop.wait(0.05):
            self._peak = max(self._peak, torch.mps.driver_allocated_memory())

    def close(self) -> None:
        self._stop.set()
        if self._monitor is not None:
            self._monitor.join()

    def precision(self) -> AbstractContextManager[Any]:
        import torch

        return (
            nullcontext()
            if self.dtype is None
            else torch.autocast("mps", dtype=self.dtype)
        )

    def reset(self) -> None:
        import torch

        self.agent.load_state_dict(self.initial)
        self.agent.zero_grad(set_to_none=True)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=1e-4, eps=1e-5)
        # The scaler's overflow check must run before clipping. A skipped update
        # is an error receipt, never a deceptively fast learner step.
        self.scaler = torch.amp.GradScaler("mps", init_scale=128, enabled=self.scaled)

    def sync(self) -> None:
        import torch

        if self.device.type == "mps":
            torch.mps.synchronize()

    def _loss(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        import torch

        from manabot.model.admission import bind_admitted
        from manabot.model.inference import transfer_admitted
        from manabot.rl.numerics import float64_operand
        from manabot.rl.selection import selected_mean
        from manabot.training.updates.damped_policy import damped_policy_loss

        targets, settings = batch.targets, batch.settings
        assert targets is not None and settings is not None
        # These are already-admitted rollout rows retained by capture, just
        # as the production learner rebinds its selected minibatches.
        admitted = bind_admitted(
            self.agent, {k: torch.from_numpy(v) for k, v in batch.observations.items()}
        )
        obs, _ = transfer_admitted(admitted, self.device)
        with self.precision():
            logits, values = self.agent.forward_distribution(obs)
        logits, values = logits.float(), values.float()

        def tensor(value: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(value).to(self.device)

        policy = damped_policy_loss(
            logits,
            tensor(targets.actions),
            tensor(targets.advantages),
            tensor(targets.behavior_logs),
            tensor(targets.reference),
            obs["actions_valid"].bool(),
            clip=settings.clip,
            collection_kl=settings.collection_kl,
            tau=settings.tau,
        )
        policy = selected_mean(policy, tensor(targets.actor_mask))
        value = (values.squeeze(-1) - tensor(targets.value_targets)).square().mean()
        return policy + float64_operand(value), logits, values

    def _backward(self, loss: torch.Tensor) -> None:

        from manabot.rl.step import check_objective

        check_objective(loss)
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        if self.scaled:
            self.scaler.unscale_(self.optimizer)
        if not any(p.grad is not None for p in self.agent.parameters()):
            raise ValueError("missing gradients; no optimizer step")

    def evaluate(self, batch: Batch) -> Evaluation:
        from manabot.policy.checkpoint import canonical_parameters

        loss, logits, values = self._loss(batch)
        self._backward(loss)
        grads = canonical_parameters(
            {
                name: np.zeros(tuple(p.shape), np.float32)
                if p.grad is None
                else p.grad.detach().float().cpu().numpy().copy()
                for name, p in self.agent.named_parameters()
            }
        )
        return Evaluation(
            logits.detach().cpu().numpy(),
            values.detach().cpu().numpy(),
            float(loss.detach()),
            grads,
        )

    def step(self, batch: Batch) -> float:
        import torch

        assert batch.settings is not None
        for group in self.optimizer.param_groups:
            group["lr"] = batch.settings.learning_rate
        loss, _, _ = self._loss(batch)
        self._backward(loss)
        torch.nn.utils.clip_grad_norm_(
            self.agent.parameters(),
            batch.settings.max_grad_norm,
            error_if_nonfinite=True,
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if not bool(
            torch.isfinite(
                torch.cat([p.detach().flatten() for p in self.agent.parameters()])
            ).all()
        ):
            raise ValueError("nonfinite parameters after step")
        self.sync()
        return float(loss.detach())

    def begin_collection(self) -> None:
        import torch

        from manabot.model.inference import PolicyInference

        self.agent.eval()
        self.inference = PolicyInference(self.agent).__enter__()
        with torch.no_grad(), self.precision():
            self.inference.catalog()
        self.sync()

    def end_collection(self) -> None:
        assert self.inference is not None
        self.inference.__exit__(None, None, None)
        self.inference = None
        self.agent.train()

    def collect(self, batch: Batch) -> np.ndarray:
        import torch

        assert self.inference is not None
        with torch.no_grad(), self.precision():
            obs = self.inference.prepare(
                {k: torch.from_numpy(v) for k, v in batch.observations.items()}
            )
            logits, _ = self.agent(obs)
            result = logits.float().cpu().numpy()
        self.sync()
        return result

    def memory(self) -> dict[str, int]:
        import torch

        result = {
            "process_peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        }
        if self.device.type == "mps":
            result["mps_driver_sampled_peak"] = max(
                self._peak, torch.mps.driver_allocated_memory()
            )
        return result


class MLXRunner:
    def __init__(self, agent: ActorCritic) -> None:
        import mlx.core as mx

        from manabot.backends.mlx.checkpoint import build

        self.encoder = agent.observation_space.encoder
        self.policy, _ = build(
            agent.config,
            self.encoder.config,
            {k: v.detach().numpy().copy() for k, v in agent.state_dict().items()},
        )
        self.initial = self.policy.parameters()
        mx.eval(self.initial)
        mx.reset_peak_memory()
        self.reset()

    def close(self) -> None:
        self.policy.clear_cache()

    def reset(self) -> None:
        from manabot.backends.mlx.learner import MLXLearner

        self.policy.update(self.initial)
        self.learner = MLXLearner(self.policy)

    def sync(self) -> None:
        import mlx.core as mx

        mx.synchronize()

    def _trim(self, batch: Batch) -> Batch:
        import torch

        from manabot.model.buckets import bucket_sizes, trim_observation

        obs = {k: torch.from_numpy(v) for k, v in batch.observations.items()}
        sizes = bucket_sizes(obs, self.encoder, objects=True)
        if sizes is None:
            raise ValueError("capture does not match the checkpoint observation layout")
        trimmed = {k: v.numpy() for k, v in trim_observation(obs, sizes).items()}
        targets = batch.targets
        if targets is not None:
            actions = sizes.sizes["actions"]
            targets = replace(
                targets,
                behavior_logs=targets.behavior_logs[:, :actions],
                reference=targets.reference[:, :actions],
            )
        return Batch(trimmed, targets, batch.settings)

    def evaluate(self, batch: Batch) -> Evaluation:
        trimmed = self._trim(batch)
        assert trimmed.targets is not None and trimmed.settings is not None
        prepared = self.policy.prepare(trimmed.observations)
        logits, values = self.policy.evaluate(prepared)
        loss, gradients = self.learner.gradients(
            prepared, trimmed.targets, trimmed.settings
        )
        return Evaluation(_pad(logits, batch), values.reshape(-1, 1), loss, gradients)

    def step(self, batch: Batch) -> float:
        import mlx.core as mx
        from mlx.utils import tree_flatten

        trimmed = self._trim(batch)
        assert trimmed.targets is not None and trimmed.settings is not None
        loss, norm = self.learner.step(
            self.policy.prepare(trimmed.observations), trimmed.targets, trimmed.settings
        )
        self.sync()
        parameters = [p.reshape(-1) for _, p in tree_flatten(self.policy.parameters())]
        if not np.isfinite([loss, norm]).all() or not bool(
            mx.all(mx.isfinite(mx.concatenate(parameters)))
        ):
            raise ValueError("nonfinite MLX loss, gradient norm or parameters")
        return loss

    def begin_collection(self) -> None:
        self.policy.cache_programs()
        self.sync()

    def end_collection(self) -> None:
        self.policy.clear_cache()

    def collect(self, batch: Batch) -> np.ndarray:
        trimmed = self._trim(batch)
        # GPU arms attend the same bucket shape. CPU additionally compacts its
        # inference columns, as production PolicyInference does.
        logits, _ = self.policy.evaluate(self.policy.prepare(trimmed.observations))
        return _pad(logits, batch)

    def memory(self) -> dict[str, int]:
        import mlx.core as mx

        return {
            "process_peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "mlx_peak": mx.get_peak_memory(),
        }


def _pad(logits: np.ndarray, batch: Batch) -> np.ndarray:
    result = np.full(batch.observations["actions_valid"].shape, -1e8, np.float32)
    result[:, : logits.shape[1]] = logits
    return result


@contextmanager
def _runner(capture: Path, width: int, arm: Arm) -> Iterator[Runner]:
    agent = _agent(capture, width)
    runner = MLXRunner(agent) if arm == "mlx-f32" else TorchRunner(agent, arm)
    # MLX owns its converted weights; do not keep a second Torch model resident.
    del agent
    try:
        yield runner
    finally:
        runner.close()


def _compare(
    actual: Evaluation, reference: Evaluation, legal: np.ndarray, arm: Arm
) -> dict[str, Any]:
    expected = reference.gradients
    if arm == "mlx-f32":
        from manabot.backends.mlx.checkpoint import mlx_gradients

        expected = mlx_gradients(expected)
    if expected.keys() != actual.gradients.keys():
        raise ValueError("gradient parameter sets differ")
    dot = left = right = 0.0
    norms = {
        k: float(np.linalg.norm(v.astype(np.float64))) for k, v in expected.items()
    }
    floor = 1e-6 * np.sqrt(sum(n * n for n in norms.values()))
    cosines: dict[str, float] = {}
    for key, want in expected.items():
        a, b = (
            actual.gradients[key].astype(np.float64).ravel(),
            want.astype(np.float64).ravel(),
        )
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"invalid gradient: {key}")
        aa, bb, ab = float(a @ a), float(b @ b), float(a @ b)
        left += aa
        right += bb
        dot += ab
        if max(np.sqrt(aa), np.sqrt(bb)) > floor:
            cosines[key] = ab / np.sqrt(aa * bb) if aa * bb else 0.0
    cosine = dot / np.sqrt(left * right) if left * right else 0.0
    worst = min(cosines, key=cosines.__getitem__)
    return {
        "logits_max_abs": float(
            np.abs(actual.logits[legal] - reference.logits[legal]).max()
        ),
        "values_max_abs": float(np.abs(actual.values - reference.values).max()),
        "loss": actual.loss,
        "reference_loss": reference.loss,
        "gradient_cosine": float(cosine),
        "gradient_min_cosine": cosines[worst],
        "gradient_worst": worst,
        "passes_gradient": bool(cosine > 0.999),
    }


def _reference_path(out: Path, capture: Path, width: int, rows: int) -> Path:
    ident = json.loads((capture / "provenance.json").read_text())["capture_id"]
    return out / "references" / ident / platform.node() / f"w{width}-b{rows}.npz"


def _choices(logits: np.ndarray, legal: np.ndarray, seed: int) -> np.ndarray:
    """Common-uniform sampling check, separate from measured policy forwards."""
    shifted = np.where(legal, logits, -np.inf).astype(np.float64)
    weights = np.exp(shifted - shifted.max(1, keepdims=True))
    cdf = np.cumsum(weights / weights.sum(1, keepdims=True), axis=1)
    cdf[:, -1] = 1
    return (cdf < np.random.default_rng(seed).random((len(logits), 1))).sum(1)


def _load_reference(path: Path) -> Evaluation:
    arrays = _read(path)
    return Evaluation(
        arrays["logits"],
        arrays["values"],
        float(arrays["loss"]),
        {k[5:]: v for k, v in arrays.items() if k.startswith("grad/")},
    )


def _provenance(capture: Path, laptop: bool) -> dict[str, Any]:
    return {
        "host": _host(),
        "laptop": laptop,
        "capture": str(capture.resolve()),
        "capture_id": json.loads((capture / "provenance.json").read_text())[
            "capture_id"
        ],
        "streams": json.loads((capture / "regime.json").read_text())["stages"][0][
            "streams"
        ],
        "capture_revision": json.loads((capture / "provenance.json").read_text())[
            "revision"
        ],
    }


def _parity_records(
    out: Path, metadata: dict[str, Any], width: int, arm: Arm
) -> list[dict[str, Any]]:
    return [
        r
        for r in _rows(out / "bench.jsonl")
        if r["event"] == "parity"
        and r.get("status") == "ok"
        and r["width"] == width
        and r["arm"] == arm
        and r["host"] == metadata["host"]
        and r["capture_id"] == metadata["capture_id"]
    ]


def bench(
    out: Path,
    capture: Path,
    phase: str,
    width: int,
    size: int,
    arm: Arm,
    laptop: bool,
    repeats: int,
) -> None:
    metadata = _provenance(capture, laptop)
    row: dict[str, Any] = {
        "event": phase,
        "width": width,
        "rows": size,
        "arm": arm,
        "etude_run": os.environ.get("ETUDE_RUN"),
        "layers": LAYERS[width],
        **metadata,
        "started": time.time(),
        "load_before": _quiet(laptop),
    }
    batch = _batch(capture, "learner" if phase == "parity" else phase, size)
    with _runner(capture, width, arm) as runner:
        if phase == "parity":
            result = runner.evaluate(batch)
            path = _reference_path(out, capture, width, size)
            collection = _batch(capture, "collection", size)
            runner.reset()
            runner.begin_collection()
            try:
                collection_logits = runner.collect(collection)
            finally:
                runner.end_collection()
            if arm == "cpu8":
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    path,
                    collection_logits=collection_logits,
                    logits=result.logits,
                    values=result.values,
                    loss=result.loss,
                    **{f"grad/{k}": v for k, v in result.gradients.items()},
                )
                path.with_suffix(".json").write_text(json.dumps(metadata))
            if json.loads(path.with_suffix(".json").read_text()) != metadata:
                raise ValueError("CPU reference provenance differs")
            collection_reference = _read(path)["collection_logits"]
            legal_collection = collection.observations["actions_valid"] > 0
            row["collection_logits_max_abs"] = float(
                np.abs(
                    collection_logits[legal_collection]
                    - collection_reference[legal_collection]
                ).max()
            )
            row["sample_disagreements"] = int(
                np.count_nonzero(
                    _choices(collection_logits, legal_collection, 21501)
                    != _choices(collection_reference, legal_collection, 21501)
                )
            )
            reference = _load_reference(path)
            row.update(
                _compare(
                    result, reference, batch.observations["actions_valid"] > 0, arm
                )
            )
        else:
            parity = [
                r
                for r in _parity_records(out, metadata, width, arm)
                if r["rows"] == size
            ]
            if not parity:
                refused = [
                    r
                    for r in _rows(out / "bench.jsonl")
                    if r["event"] == "parity"
                    and r["status"] == "unsupported"
                    and r["width"] == width
                    and r["rows"] == size
                    and r["arm"] == arm
                    and r["host"] == metadata["host"]
                    and r["capture_id"] == metadata["capture_id"]
                ]
                if refused:
                    row.update(
                        status="unsupported",
                        error=refused[-1]["error"],
                        failure_phase="parity",
                        load_after=os.getloadavg()[0],
                        ended=time.time(),
                    )
                    _emit(out / "bench.jsonl", row)
                    return
                raise ValueError(
                    "run parity for this host, revision, capture, width, rows and arm first"
                )
            row["parity"] = {
                k: parity[-1][k]
                for k in (
                    "gradient_cosine",
                    "gradient_min_cosine",
                    "logits_max_abs",
                    "passes_gradient",
                )
            }
            times: list[float] = []
            if phase == "collection":
                _quiet(laptop)
                tick = time.perf_counter()
                runner.begin_collection()
                row["catalog_seconds"] = time.perf_counter() - tick
            try:
                for rep in range(repeats + 1):
                    if phase == "learner":
                        runner.reset()
                    runner.sync()
                    _quiet(laptop)
                    tick = time.perf_counter()
                    if phase == "learner":
                        runner.step(batch)
                    else:
                        runner.collect(batch)
                    runner.sync()
                    times.append(time.perf_counter() - tick)
            finally:
                if phase == "collection":
                    runner.end_collection()
            row.update(
                first_seconds=times[0],
                seconds=times[1:],
                median_seconds=statistics.median(times[1:]),
                rows_per_second=size / statistics.median(times[1:]),
            )
        row.update(
            status="ok",
            memory=runner.memory(),
            load_after=os.getloadavg()[0],
            ended=time.time(),
        )
        _emit(out / "bench.jsonl", row)


def update(
    out: Path, capture: Path, width: int, arm: Arm, laptop: bool, rep: int
) -> None:
    """Replay original calls and minibatches; weights evolve only in learning."""
    meta = json.loads((capture / "capture.json").read_text())
    provenance = _provenance(capture, laptop)
    _quiet(laptop)
    with _runner(capture, width, arm) as runner:
        row: dict[str, Any] = {
            "event": "update",
            "arm": arm,
            "width": width,
            "rep": rep,
            "etude_run": os.environ.get("ETUDE_RUN"),
            "layers": LAYERS[width],
            **provenance,
            "started": time.time(),
            "load_before": _quiet(laptop),
        }
        _quiet(laptop)
        tick = time.perf_counter()
        runner.begin_collection()
        policy_seconds = time.perf_counter() - tick
        calls = decisions = 0
        sampled: list[np.ndarray] = []
        try:
            for batch in _batches(capture, "collection"):
                _quiet(laptop)
                tick = time.perf_counter()
                logits = runner.collect(batch)
                policy_seconds += time.perf_counter() - tick
                sampled.append(
                    _choices(
                        logits, batch.observations["actions_valid"] > 0, 21501 + calls
                    )
                )
                calls += 1
                decisions += batch.rows
        finally:
            runner.end_collection()
        actions = np.concatenate(sampled)
        sample_path = _reference_path(out, capture, width, 0)
        if arm == "cpu8":
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(sample_path, actions=actions)
            sample_path.with_suffix(".json").write_text(
                json.dumps(_provenance(capture, laptop))
            )
        if json.loads(sample_path.with_suffix(".json").read_text()) != _provenance(
            capture, laptop
        ):
            raise ValueError("whole-update CPU sampling reference provenance differs")
        expected_actions = _read(sample_path)["actions"]
        if actions.shape != expected_actions.shape:
            raise ValueError("whole-update call rows differ")
        row["sample_disagreements"] = int(np.count_nonzero(actions != expected_actions))
        learning_seconds = 0.0
        steps = learner_rows = 0
        for batch in _batches(capture, "learner"):
            _quiet(laptop)
            tick = time.perf_counter()
            runner.step(batch)
            learning_seconds += time.perf_counter() - tick
            steps += 1
            learner_rows += batch.rows
        engine = float(meta["engine"])
        row.update(
            status="ok",
            engine=engine,
            collection_policy=policy_seconds,
            learning=learning_seconds,
            seconds=engine + policy_seconds + learning_seconds,
            collection_rows=decisions,
            learner_rows=learner_rows,
            learner_decisions=int(meta["learner_decisions"]),
            calls=calls,
            steps=steps,
            memory=runner.memory(),
            ended=time.time(),
            load_after=os.getloadavg()[0],
        )
        _emit(out / "updates.jsonl", row)


def capture(out: Path, streams: int) -> None:
    from experiments.tla_frontier import frozen
    from manabot.backends.capture import capture as record
    from manabot.training.regime import TrainingRegime

    regimes, protocol = frozen()
    data = regimes["w160l8"].model_dump()
    data["id"] = "mps-retest"
    data["agent"]["hidden_dim"] = 384
    data["stages"][0]["updates"] = 1
    data["stages"][0]["streams"] = streams
    data["stages"][0]["execution"].update(
        device="cpu",
        workers=1,
        threads=4,
        bucketed_shapes=True,
        memory_bytes=20 * 2**30,
        wall_seconds=7200,
    )
    data["wall_seconds"] = 7200
    provenance = {
        **_host(),
        "capture_id": str(uuid.uuid4()),
        "capture_threads": 4,
        "etude_run": os.environ.get("ETUDE_RUN"),
    }
    record(
        out,
        TrainingRegime.model_validate(data),
        protocol.training_seed,
        experiment="mps_retest",
    )
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))


def _child(
    args: argparse.Namespace,
    command: str,
    width: int,
    size: int,
    arm: Arm,
    *,
    phase: str = "",
    rep: int = 0,
) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        command,
        str(args.out),
        "--capture",
        str(args.capture),
        "--width",
        str(width),
        "--arm",
        arm,
    ]
    if command == "bench":
        cmd += ["--phase", phase, "--rows", str(size), "--repeats", str(args.repeats)]
    else:
        cmd += ["--rep", str(rep)]
    if args.laptop:
        cmd += ["--laptop"]
    ledger = args.out / ("bench.jsonl" if command == "bench" else "updates.jsonl")
    before = len(_rows(ledger))
    # A hard timeout is retained as a failed cell, never inferred to be OOM.
    try:
        result = subprocess.run(cmd, timeout=args.timeout, check=False)
        if result.returncode == 0:
            return
        if len(_rows(ledger)) > before:
            return
        reason = f"worker exit {result.returncode}"
    except subprocess.TimeoutExpired:
        reason = f"worker timeout after {args.timeout}s"
    _emit(
        args.out / ("bench.jsonl" if command == "bench" else "updates.jsonl"),
        {
            "event": phase if command == "bench" else "update",
            "width": width,
            "rows": size,
            "arm": arm,
            "rep": rep,
            "status": "failed",
            "error": reason,
            **_provenance(args.capture, args.laptop),
        },
    )


def matrix(args: argparse.Namespace) -> None:
    for width in args.widths:
        for size in SIZES["learner"]:
            for arm in ARMS:
                _quiet(args.laptop)
                _child(args, "bench", width, size, arm, phase="parity")
                _child(args, "bench", width, size, arm, phase="learner")
                if size in SIZES["collection"]:
                    _child(args, "bench", width, size, arm, phase="collection")


def replays(args: argparse.Namespace) -> None:
    # Interleave arms by repetition to reduce temporal drift. All four arm
    # families are required, and both half dtypes are attempted.
    for width in args.widths:
        for arm in ARMS:
            if not _parity_records(
                args.out, _provenance(args.capture, args.laptop), width, arm
            ):
                _quiet(args.laptop)
                _child(args, "bench", width, 64, arm, phase="parity")
        for rep in range(3):
            for arm in ARMS:
                _quiet(args.laptop)
                _child(args, "update", width, 0, arm, rep=rep)


def report(bench_dir: Path, out: Path) -> None:
    """Keep one host/source/capture set per report; mark every absent bar."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _rows(bench_dir / "bench.jsonl")
    updates = _rows(bench_dir / "updates.jsonl")
    out.mkdir(parents=True, exist_ok=True)
    hosts = sorted({r["host"]["hostname"] for r in rows + updates})
    if not hosts:
        raise ValueError("no measurement receipts; refusing an empty report")
    summaries: list[dict[str, Any]] = []
    for hostname in hosts:
        local = [r for r in rows if r["host"]["hostname"] == hostname]
        replay = [r for r in updates if r["host"]["hostname"] == hostname]
        evidence = local + replay
        host = evidence[0]["host"]
        if any(r["host"] != host for r in evidence):
            raise ValueError(
                "host/software/source changed; use separate report directories"
            )
        for streams in STREAMS:
            captures = {r["capture_id"] for r in evidence if r["streams"] == streams}
            if len(captures) > 1:
                raise ValueError(
                    f"multiple captures for {hostname}/{streams}; select one explicitly"
                )
        latest = {
            (r["event"], r["width"], r["streams"], r["rows"], r["arm"]): r
            for r in local
        }
        replay_latest = {
            (r["width"], r["streams"], r["arm"], r["rep"]): r for r in replay
        }
        missing: list[str] = []
        # Per-call matrix uses the 256-stream capture; the other captures are
        # whole-update workloads, not additional per-call comparison datasets.
        for width in WIDTHS:
            for phase, sizes in SIZES.items():
                for size in sizes:
                    available_half = False
                    for arm in ARMS:
                        entry = latest.get((phase, width, 256, size, arm))
                        status = entry.get("status") if entry else None
                        if status not in ("ok", "oom", "unsupported") or (
                            status == "unsupported" and arm not in HALF
                        ):
                            missing.append(f"{phase}/w{width}/b{size}/{arm}")
                        if arm in HALF and status in ("ok", "oom"):
                            available_half = True
                    if not available_half:
                        missing.append(f"{phase}/w{width}/b{size}/no executed half arm")
        groups = {
            (width, streams, arm): [
                replay_latest.get((width, streams, arm, rep)) for rep in range(3)
            ]
            for width in WIDTHS
            for streams in STREAMS
            for arm in ARMS
        }
        for (width, streams, arm), attempts in groups.items():
            if not all(
                r and r["status"] in ("ok", "oom", "unsupported") for r in attempts
            ):
                missing.append(f"update/w{width}/streams{streams}/{arm}")
        summary: dict[str, Any] = {
            "host": host,
            "missing": missing,
            "complete": not missing,
        }
        summaries.append(summary)
        is_m2 = host["chip"] == "Apple M2"
        suffix = "" if is_m2 else "-" + hostname.replace("/", "_")
        if not is_m2:
            summary["supplemental_host"] = True
        for phase in SIZES:
            fig, axes = plt.subplots(2, 4, figsize=(17, 8))
            for width, axis in zip(WIDTHS, axes.flat):
                for arm in ARMS:
                    points = [
                        latest[(phase, width, 256, b, arm)]
                        for b in SIZES[phase]
                        if latest.get((phase, width, 256, b, arm), {}).get("status")
                        == "ok"
                    ]
                    axis.plot(
                        [r["rows"] for r in points],
                        [r["rows_per_second"] for r in points],
                        marker="o",
                        label=LABELS[arm],
                    )
                axis.set(
                    title=f"Width {width} · {LAYERS[width]} layers",
                    xlabel="Rows per call",
                    ylabel="Rows per second",
                    xscale="log",
                )
            axes.flat[-1].axis("off")
            axes.flat[-1].legend(
                *axes.flat[0].get_legend_handles_labels(), loc="center", fontsize=9
            )
            fig.suptitle(
                f"{phase.capitalize()} · {host['chip']} · {host['memory_bytes'] / 2**30:g} GiB\n{hostname} · torch {host['torch']} · MLX {host['mlx']} · 256-stream capture"
            )
            fig.tight_layout()
            fig.savefig(out / f"{phase}{suffix}.png", dpi=180, facecolor="white")
            plt.close(fig)
        # One panel per model shape; three stream groups each contain all arms.
        fig = plt.figure(figsize=(19, 22))
        grid = fig.add_gridspec(4, 2, hspace=0.5, wspace=0.22)
        medians: list[dict[str, Any]] = []
        labels = ("CPU8\nf32", "MPS\nf32", "MPS\nbf16", "MPS\nf16", "MLX\nf32")
        spans: list[dict[str, int | float]] = []
        for index, width in enumerate(WIDTHS):
            sub = grid[index // 2, index % 2].subgridspec(
                2, 1, height_ratios=(3, 1), hspace=0.12
            )
            axis = fig.add_subplot(sub[0])
            rate = fig.add_subplot(sub[1], sharex=axis)
            for si, streams in enumerate(STREAMS):
                valid_replays = [
                    r
                    for (w, s, _, _), r in replay_latest.items()
                    if w == width and s == streams and r["status"] == "ok"
                ]
                if valid_replays:
                    span = max(r["ended"] for r in valid_replays) - min(
                        r["started"] for r in valid_replays
                    )
                    spans.append({"width": width, "streams": streams, "seconds": span})
                for ai, arm in enumerate(ARMS):
                    x = si * 6 + ai
                    attempts = groups[(width, streams, arm)]
                    complete = [r for r in attempts if r and r["status"] == "ok"]
                    if len(complete) != 3:
                        statuses = {
                            r["status"] if r else "not_measured" for r in attempts
                        }
                        note = (
                            "does not fit"
                            if statuses == {"oom"}
                            else "unsupported"
                            if statuses == {"unsupported"}
                            else "not measured"
                        )
                        axis.text(
                            x,
                            0.03,
                            note,
                            rotation=90,
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            transform=axis.get_xaxis_transform(),
                        )
                        continue
                    # Select the median-total replay's phases: stacks sum to
                    # the median update, unlike independently medianed phases.
                    middle = sorted(complete, key=lambda r: r["seconds"])[1]
                    medians.append(middle)
                    bottom = 0.0
                    for key, label, color in (
                        ("engine", "Engine + host work", "#808080"),
                        ("collection_policy", "Collection policy calls", "#539caf"),
                        ("learning", "Learning", "#db9250"),
                    ):
                        axis.bar(
                            x,
                            middle[key],
                            0.83,
                            bottom=bottom,
                            color=color,
                            edgecolor="#333333",
                            linewidth=0.4,
                            label=label if index == 0 and x == 0 else None,
                        )
                        bottom += middle[key]
                    rate.bar(
                        x,
                        statistics.median(
                            r["learner_decisions"] / r["seconds"] for r in complete
                        ),
                        0.83,
                        color="#539caf",
                        edgecolor="#333333",
                        linewidth=0.4,
                    )
                    parity = [
                        r
                        for r in local
                        if r["event"] == "parity"
                        and r["width"] == width
                        and r["arm"] == arm
                        and r["status"] == "ok"
                    ]
                    worst = min((r["gradient_cosine"] for r in parity), default=None)
                    same = all(r["sample_disagreements"] == 0 for r in complete)
                    text = f"cos {worst:.4f}" if worst is not None else "parity missing"
                    if not same:
                        text += " · choices differ"
                    axis.annotate(
                        text,
                        (x, bottom),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center",
                        rotation=90,
                        fontsize=6,
                    )
            axis.set(
                title=f"Width {width} · {LAYERS[width]} attention layers",
                ylabel="Seconds per update",
            )
            axis.margins(y=0.35)
            axis.tick_params(axis="x", labelbottom=False)
            rate.set_ylabel("Learner decisions /\nwhole-update second", fontsize=8)
            rate.set_xticks(
                [si * 6 + ai for si in range(3) for ai in range(5)],
                list(labels) * 3,
                fontsize=7,
            )
            for si, streams in enumerate(STREAMS):
                rate.text(
                    si * 6 + 2,
                    -0.6,
                    f"{streams} streams",
                    ha="center",
                    fontsize=9,
                    transform=rate.get_xaxis_transform(),
                )
        summary["whole_updates"] = medians
        summary["replay_windows"] = spans
        summary["same_hour"] = bool(spans) and all(r["seconds"] <= 3600 for r in spans)
        if not summary["same_hour"] and is_m2:
            summary["complete"] = False
        legend = fig.add_subplot(grid[3, 1])
        legend.axis("off")
        legend.text(
            0,
            0.95,
            "Full context: the current training recipe",
            fontsize=16,
            weight="bold",
            va="top",
        )
        legend.text(
            0,
            0.84,
            f"{host['chip']} · {host['memory_bytes'] / 2**30:g} GiB\n{hostname}\ntorch {host['torch']} · MLX {host['mlx']}\nBoth-seat self-play; one real capture per stream count.\nMedian of three replays; same inputs and recorded decisions.\nHalf precision: bfloat16 and float16 autocast.\nFloat16 uses a gradient scaler.\n\nStacks show the median-total replay's phase times.\nEngine time includes non-policy collection host work.\nRate counts collected learner decisions before filtering.\nCosine labels compare gradients with CPU float32.\nCommon-uniform sampling mismatches are labelled.\nParity is not a proof of identical trained policies.\n\nMissing and failed arms remain visible.\nPer-group same-hour condition: "
            + ("met" if summary["same_hour"] else "NOT MET"),
            fontsize=10,
            va="top",
            linespacing=1.6,
        )
        handles, names = fig.axes[0].get_legend_handles_labels()
        legend.legend(handles, names, loc="lower left", fontsize=10)
        fig.suptitle("Whole-update performance · " + host["chip"], fontsize=20, y=0.995)
        fig.savefig(
            out / f"full_context{suffix}.png",
            dpi=180,
            facecolor="white",
            bbox_inches="tight",
        )
        plt.close(fig)
    (out / "summary.json").write_text(
        json.dumps(
            {"hosts": summaries, "measurements": rows, "updates": updates},
            indent=2,
            allow_nan=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("capture", "bench", "matrix", "update", "replays", "report"):
        command = commands.add_parser(name)
        command.add_argument("out", type=Path)
        if name == "report":
            command.add_argument("--bench", type=Path, required=True)
            continue
        command.add_argument("--laptop", action="store_true")
        if name == "capture":
            command.add_argument("--streams", type=int, choices=STREAMS, required=True)
        if name != "capture":
            command.add_argument("--capture", type=Path, required=True)
        if name in ("bench", "update"):
            command.add_argument("--width", type=int, choices=WIDTHS, required=True)
            command.add_argument("--arm", choices=ARMS, required=True)
        if name == "bench":
            command.add_argument(
                "--phase", choices=("parity", "learner", "collection"), required=True
            )
            command.add_argument(
                "--rows", type=int, choices=SIZES["learner"], required=True
            )
        if name in ("bench", "matrix", "replays"):
            command.add_argument("--repeats", type=int, default=3)
        if name in ("matrix", "replays"):
            command.add_argument("--timeout", type=float, default=900)
            command.add_argument("--widths", nargs="+", type=int, default=list(WIDTHS))
        if name == "update":
            command.add_argument("--rep", type=int, choices=range(3), required=True)
    args = parser.parse_args()
    if args.command == "report":
        report(args.bench, args.out)
        return
    import torch

    torch.set_num_threads(8)
    if not str(torch.__version__).startswith("2.15."):
        parser.error("measurements require pinned torch 2.15")
    if platform.system() != "Darwin":
        parser.error("this experiment measures Apple hosts")
    if not os.environ.get("ETUDE_RUN"):
        parser.error("measurements require the launcher's ETUDE_RUN identity")
    if args.laptop and args.command == "capture":
        parser.error(
            "main captures and whole-update comparisons belong on the rented M2"
        )
    if (
        args.laptop
        and args.command == "bench"
        and args.width < 512
        and args.rows < 2048
    ):
        parser.error("laptop is only for wider/larger overflow arms")
    if getattr(args, "repeats", 3) < 1:
        parser.error("at least one measured repeat is required")
    host = _host()
    if host["chip"] != "Apple M2" and not args.laptop:
        parser.error(
            "main measurements require Apple M2; M4 overflow runs require --laptop"
        )
    if not args.laptop and platform.python_version_tuple()[:2] != ("3", "12"):
        parser.error("the rented M2 requires a fresh Python 3.12 environment")
    if (
        args.laptop
        and args.command in ("matrix", "replays")
        and any(w < 512 for w in args.widths)
    ):
        parser.error(
            "laptop matrices/replays require explicit --widths 512 768 1024 (or a subset)"
        )
    if args.laptop and args.command == "update" and args.width < 512:
        parser.error("laptop whole updates are only for widths the M2 cannot hold")
    if (
        args.command == "bench"
        and args.phase == "collection"
        and args.rows not in SIZES["collection"]
    ):
        parser.error("collection rows are 64, 256 or 512")
    # New evidence must name committed executable source. Report edits may
    # proceed independently, but source edits require a new freeze.
    if not host["revision"]:
        parser.error(
            "no source commit; run from the frozen checkout or an Etude source archive"
        )
    if (ROOT / ".git").exists():
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "experiments/mps_retest.py"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        dirty = subprocess.run(
            [
                "git",
                "diff",
                "HEAD",
                "--quiet",
                "--",
                "*.py",
                "*.rs",
                "*.toml",
                "uv.lock",
            ],
            cwd=ROOT,
        )
        if tracked.returncode or dirty.returncode:
            parser.error(
                "commit experiment and executable source before capture or scoring"
            )
    try:
        with warnings.catch_warnings():
            # Unsupported autocast often warns and silently becomes float32.
            warnings.filterwarnings(
                "error", message=".*autocast.*", category=UserWarning
            )
            if args.command == "capture":
                capture(args.out, args.streams)
            elif args.command == "bench":
                bench(
                    args.out,
                    args.capture,
                    args.phase,
                    args.width,
                    args.rows,
                    args.arm,
                    args.laptop,
                    args.repeats,
                )
            elif args.command == "update":
                update(
                    args.out, args.capture, args.width, args.arm, args.laptop, args.rep
                )
            elif args.command == "matrix":
                matrix(args)
            else:
                replays(args)
    except Exception as error:
        if args.command not in ("bench", "update"):
            raise
        message = str(error)
        lower = message.lower()
        status = "failed"
        if isinstance(error, MemoryError) or "out of memory" in lower:
            status = "oom"
        elif args.arm in HALF and (
            ("autocast" in lower and isinstance(error, UserWarning))
            or (
                any(dtype in lower for dtype in ("bfloat16", "float16", "half"))
                and any(
                    term in lower
                    for term in ("not supported", "unsupported", "not implemented")
                )
            )
        ):
            status = "unsupported"
        elif "laptop load" in lower:
            status = "not_measured"
        _emit(
            args.out / ("bench.jsonl" if args.command == "bench" else "updates.jsonl"),
            {
                "event": getattr(args, "phase", "update"),
                "width": args.width,
                "rows": getattr(args, "rows", 0),
                "arm": args.arm,
                "rep": getattr(args, "rep", 0),
                "status": status,
                "error": f"{type(error).__name__}: {message}",
                "load_after": os.getloadavg()[0],
                **_provenance(args.capture, args.laptop),
            },
        )
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
