"""Bucketed MPS retest: real capture, isolated arm measurements and report.

Run `capture` on the rented M2 or quiet laptop, then `matrix` for correctness and
per-call blocks, and `replays` for the three whole-update repetitions. `report`
keeps incomplete matrices and unsupported arms visible. Every worker records its
host, source revision, capture, dtype and load. Backend recommendations belong
to the report's interpretation, never to a hard-coded threshold. `_host_report`
selects and summarizes receipts once; tables and charts render that same evidence.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
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


Collect = Callable[[Batch], np.ndarray]


class Runner(Protocol):
    def evaluate(self, batch: Batch) -> Evaluation: ...
    def step(self, batch: Batch) -> float: ...
    def collection(self) -> AbstractContextManager[Collect]: ...
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
        self._peak = 0
        self._stop = threading.Event()
        self._monitor: threading.Thread | None = None
        self.reset()
        if self.device.type == "mps":
            self._monitor = threading.Thread(target=self._sample_memory, daemon=True)
            self._monitor.start()

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

    @contextmanager
    def collection(self) -> Iterator[Collect]:
        import torch

        from manabot.model.inference import PolicyInference

        self.agent.eval()
        try:
            with PolicyInference(self.agent) as inference:
                with torch.no_grad(), self.precision():
                    inference.catalog()
                self.sync()

                def collect(batch: Batch) -> np.ndarray:
                    with torch.no_grad(), self.precision():
                        obs = inference.prepare(
                            {
                                k: torch.from_numpy(v)
                                for k, v in batch.observations.items()
                            }
                        )
                        logits, _ = self.agent(obs)
                        result = logits.float().cpu().numpy()
                    self.sync()
                    return result

                yield collect
        finally:
            self.agent.train()

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

    @contextmanager
    def collection(self) -> Iterator[Collect]:
        try:
            self.policy.cache_programs()
            self.sync()

            def collect(batch: Batch) -> np.ndarray:
                trimmed = self._trim(batch)
                # GPU arms attend the same bucket shape. CPU additionally
                # compacts inference columns, as production PolicyInference does.
                logits, _ = self.policy.evaluate(
                    self.policy.prepare(trimmed.observations)
                )
                return _pad(logits, batch)

            yield collect
        finally:
            self.policy.clear_cache()

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


def _reference_path(out: Path, metadata: dict[str, Any], width: int, rows: int) -> Path:
    return (
        out
        / "references"
        / metadata["capture_id"]
        / metadata["host"]["hostname"]
        / f"w{width}-b{rows}.npz"
    )


def _choices(logits: np.ndarray, legal: np.ndarray, seed: int) -> np.ndarray:
    """Common-uniform sampling check, separate from measured policy forwards."""
    shifted = np.where(legal, logits, -np.inf).astype(np.float64)
    weights = np.exp(shifted - shifted.max(1, keepdims=True))
    cdf = np.cumsum(weights / weights.sum(1, keepdims=True), axis=1)
    cdf[:, -1] = 1
    return (cdf < np.random.default_rng(seed).random((len(logits), 1))).sum(1)


def _reference(
    path: Path,
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Save CPU arrays when supplied, then read only a matching reference."""
    if arrays is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
        path.with_suffix(".json").write_text(json.dumps(metadata))
    if json.loads(path.with_suffix(".json").read_text()) != metadata:
        raise ValueError("CPU reference provenance differs")
    return _read(path)


def _provenance(capture: Path, laptop: bool) -> dict[str, Any]:
    provenance = json.loads((capture / "provenance.json").read_text())
    return {
        "host": _host(),
        "laptop": laptop,
        "capture": str(capture.resolve()),
        "capture_id": provenance["capture_id"],
        "streams": json.loads((capture / "regime.json").read_text())["stages"][0][
            "streams"
        ],
        "capture_revision": provenance["revision"],
        # The shared engine component was measured here, not on the replay
        # host. Capture execution sets its own thread count after _host().
        "capture_host": {
            **{
                key: value
                for key, value in provenance.items()
                if key not in ("capture_id", "capture_threads", "etude_run")
            },
            "cpu_threads": provenance["capture_threads"],
        },
        "capture_threads": provenance["capture_threads"],
    }


def _latest_parity(
    out: Path, metadata: dict[str, Any], width: int, arm: Arm
) -> dict[int, dict[str, Any]]:
    """Latest attempt per row count; failures supersede earlier successes too."""
    return {
        r["rows"]: r
        for r in _rows(out / "bench.jsonl")
        if r["event"] == "parity"
        and r["width"] == width
        and r["arm"] == arm
        and r["host"] == metadata["host"]
        and r["capture_id"] == metadata["capture_id"]
    }


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
    # Receipt admission needs no model, device allocation or capture arrays.
    if phase != "parity":
        parity = _latest_parity(out, metadata, width, arm).get(size)
        if parity is None or parity["status"] != "ok":
            if parity is not None and parity["status"] == "unsupported":
                row.update(
                    status="unsupported",
                    error=parity["error"],
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
            k: parity[k]
            for k in (
                "gradient_cosine",
                "gradient_min_cosine",
                "logits_max_abs",
                "passes_gradient",
            )
        }
    batch = _batch(capture, "learner" if phase == "parity" else phase, size)
    with _runner(capture, width, arm) as runner:
        if phase == "parity":
            result = runner.evaluate(batch)
            path = _reference_path(out, metadata, width, size)
            collection = _batch(capture, "collection", size)
            runner.reset()
            with runner.collection() as collect:
                collection_logits = collect(collection)
            arrays = _reference(
                path,
                metadata,
                {
                    "collection_logits": collection_logits,
                    "logits": result.logits,
                    "values": result.values,
                    "loss": np.asarray(result.loss),
                    **{f"grad/{k}": v for k, v in result.gradients.items()},
                }
                if arm == "cpu8"
                else None,
            )
            collection_reference = arrays["collection_logits"]
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
            reference = Evaluation(
                arrays["logits"],
                arrays["values"],
                float(arrays["loss"]),
                {k[5:]: v for k, v in arrays.items() if k.startswith("grad/")},
            )
            row.update(
                _compare(
                    result, reference, batch.observations["actions_valid"] > 0, arm
                )
            )
        else:
            times: list[float] = []
            _quiet(laptop)
            tick = time.perf_counter()
            scope = (
                runner.collection()
                if phase == "collection"
                else nullcontext(runner.step)
            )
            with scope as call:
                if phase == "collection":
                    row["catalog_seconds"] = time.perf_counter() - tick
                for rep in range(repeats + 1):
                    if phase == "learner":
                        runner.reset()
                    runner.sync()
                    _quiet(laptop)
                    tick = time.perf_counter()
                    call(batch)
                    runner.sync()
                    times.append(time.perf_counter() - tick)
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
        calls = decisions = 0
        sampled: list[np.ndarray] = []
        with runner.collection() as collect:
            policy_seconds = time.perf_counter() - tick
            for batch in _batches(capture, "collection"):
                _quiet(laptop)
                tick = time.perf_counter()
                logits = collect(batch)
                policy_seconds += time.perf_counter() - tick
                sampled.append(
                    _choices(
                        logits, batch.observations["actions_valid"] > 0, 21501 + calls
                    )
                )
                calls += 1
                decisions += batch.rows
        actions = np.concatenate(sampled)
        sample_path = _reference_path(out, provenance, width, 0)
        expected_actions = _reference(
            sample_path, provenance, {"actions": actions} if arm == "cpu8" else None
        )["actions"]
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


def capture(out: Path, streams: int, laptop: bool) -> None:
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
        "laptop": laptop,
        "load_before": _quiet(laptop),
    }
    record(
        out,
        TrainingRegime.model_validate(data),
        protocol.training_seed,
        experiment="mps_retest",
    )
    provenance["load_after"] = os.getloadavg()[0]
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))


def _child(
    args: argparse.Namespace,
    phase: Literal["parity", "learner", "collection", "update"],
    width: int,
    arm: Arm,
    *,
    size: int = 0,
    rep: int = 0,
) -> None:
    command = "update" if phase == "update" else "bench"
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
        cmd += ["--phase", phase, "--rows", str(size)]
        if phase != "parity":
            cmd += ["--repeats", str(args.repeats)]
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
        ledger,
        {
            "event": phase,
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
                _child(args, "parity", width, arm, size=size)
                _child(args, "learner", width, arm, size=size)
                if size in SIZES["collection"]:
                    _child(args, "collection", width, arm, size=size)


def replays(args: argparse.Namespace) -> None:
    # Interleave arms by repetition to reduce temporal drift. All four arm
    # families are required, and both half dtypes are attempted.
    for width in args.widths:
        for arm in ARMS:
            if not any(
                r["status"] == "ok"
                for r in _latest_parity(
                    args.out, _provenance(args.capture, args.laptop), width, arm
                ).values()
            ):
                _quiet(args.laptop)
                _child(args, "parity", width, arm, size=64)
        for rep in range(3):
            for arm in ARMS:
                _quiet(args.laptop)
                _child(args, "update", width, arm, rep=rep)


@dataclass(frozen=True)
class HostReport:
    """One validated receipt set, shared by the summary, tables and charts."""

    calls: dict[tuple[str, int, int, int, str], dict[str, Any]]
    updates: dict[tuple[int, int, str], list[dict[str, Any] | None]]
    summary: dict[str, Any]


def _status(row: dict[str, Any] | None) -> str:
    if row is None:
        return "not measured"
    return {"oom": "does not fit", "ok": "measured"}.get(
        row["status"], row["status"].replace("_", " ")
    )


def _number(value: float | None) -> str:
    return "—" if value is None else f"{value:.6g}"


def _memory_cells(row: dict[str, Any]) -> list[str]:
    memory = row.get("memory", {})
    return [
        _number(memory[key] / 2**30) if key in memory else "—"
        for key in ("process_peak_rss", "mps_driver_sampled_peak", "mlx_peak")
    ]


def _table(headers: list[str], rows: list[list[str]]) -> str:
    def line(cells: list[str]) -> str:
        return (
            "| "
            + " | ".join(c.replace("|", "\\|").replace("\n", " ") for c in cells)
            + " |"
        )

    return "\n".join(
        [line(headers), line(["---"] * len(headers)), *(line(r) for r in rows)]
    )


def _details(
    out: Path,
    suffix: str,
    data: HostReport,
) -> None:
    """Readable comparisons retain failures and absent values, never zero-fill."""
    summary, latest, groups = data.summary, data.calls, data.updates
    host = summary["host"]
    text = [
        f"# Measurements · {host['chip']} · {host['hostname']}",
        f"Replay: {host['memory_bytes'] / 2**30:g} GiB, "
        f"{host['cpu_threads']} CPU threads, Python {host['python']}, "
        f"torch {host['torch']}, MLX {host['mlx']}. Source: `{host['revision']}`.",
        "Coverage is not a correctness verdict. Timings remain visible when gradients "
        "or choices disagree. Missing measurements are not zero. Memory columns are "
        "GiB: lifetime process RSS, sampled MPS driver peak (50 ms; lower bound), "
        "and MLX allocator peak. They measure different scopes and must not be summed.",
        "## Shared engine component",
        _table(
            [
                "Streams",
                "Capture",
                "Capture machine",
                "Capture threads",
                "Capture source",
            ],
            [
                [
                    str(c["streams"]),
                    c["capture_id"],
                    f"{c['host']['chip']} · {c['host']['hostname']} · {c['host']['memory_bytes'] / 2**30:g} GiB"
                    if c["host"]
                    else "unknown",
                    str(c["threads"]) if c["threads"] is not None else "unknown",
                    c["revision"],
                ]
                for c in summary["captures"]
            ],
        ),
        "Engine/host seconds are reused from capture at its stated thread count. "
        "Collection policy and learning are measured on the replay machine. "
        "A cross-host sum is not a measured whole update on the replay machine.",
    ]
    for phase, sizes in SIZES.items():
        cells: list[list[str]] = []
        for width in WIDTHS:
            for size in sizes:
                for arm in ARMS:
                    row = latest.get((phase, width, 256, size, arm))
                    r = row or {}
                    cells.append(
                        [
                            f"w{width}l{LAYERS[width]}",
                            str(size),
                            LABELS[arm],
                            _status(row),
                            *[
                                _number(r.get(k))
                                for k in (
                                    "first_seconds",
                                    "catalog_seconds",
                                    "median_seconds",
                                    "rows_per_second",
                                )
                            ],
                            *_memory_cells(r),
                            r.get("error", ""),
                        ]
                    )
        text.extend(
            [
                f"## {phase.capitalize()} · 256-stream capture",
                "First call and catalog setup are separate from warmed median seconds. "
                "Rates are learner rows/s or collection decisions/s.",
                _table(
                    [
                        "Model",
                        "Rows",
                        "Arm",
                        "Status",
                        "First s",
                        "Catalog s",
                        "Warm median s",
                        "Rate /s",
                        "RSS GiB",
                        "MPS GiB ≥",
                        "MLX GiB",
                        "Error",
                    ],
                    cells,
                ),
            ]
        )
    parity_cells: list[list[str]] = []
    for width in WIDTHS:
        for streams in STREAMS:
            # Other captures require a parity block before replay; replays uses
            # 64 rows unless an existing block already covers this arm.
            sizes = sorted(
                {
                    r["rows"]
                    for r in latest.values()
                    if r["event"] == "parity"
                    and r["width"] == width
                    and r["streams"] == streams
                }
                | (set(SIZES["learner"]) if streams == 256 else {64})
            )
            for size in sizes:
                for arm in ARMS:
                    row = latest.get(("parity", width, streams, size, arm))
                    r = row or {}
                    loss_delta = (
                        abs(r["loss"] - r["reference_loss"]) if "loss" in r else None
                    )
                    parity_cells.append(
                        [
                            f"w{width}l{LAYERS[width]}",
                            str(streams),
                            str(size),
                            LABELS[arm],
                            _status(row),
                            *[
                                _number(r.get(k))
                                for k in (
                                    "logits_max_abs",
                                    "collection_logits_max_abs",
                                    "values_max_abs",
                                )
                            ],
                            _number(loss_delta),
                            _number(r.get("reference_loss")),
                            _number(r.get("gradient_cosine")),
                            _number(r.get("gradient_min_cosine")),
                            r.get("gradient_worst", "—"),
                            str(r["sample_disagreements"])
                            if "sample_disagreements" in r
                            else "—",
                            r.get("error", ""),
                        ]
                    )
    text.extend(
        [
            "## Correctness against CPU float32",
            "Cosines cover the full gradient and the worst tensor after the global-norm floor. "
            "The historical 0.999 cosine reference is descriptive, not an adoption rule. "
            "Logit/value/loss differences and common-uniform sampled-choice disagreements "
            "need interpretation; zero disagreements cover only the recorded rows.",
            _table(
                [
                    "Model",
                    "Streams",
                    "Rows",
                    "Arm",
                    "Status",
                    "Learner logit Δ",
                    "Collection logit Δ",
                    "Value Δ",
                    "Loss abs Δ",
                    "CPU loss",
                    "Gradient cosine",
                    "Min tensor cosine",
                    "Worst tensor",
                    "Choice mismatches",
                    "Error",
                ],
                parity_cells,
            ),
            "## Whole-update repetitions",
            "All repetitions are listed, including failures. Charts select the median-total "
            "replay's phases. First-call overhead is included in replay phases. "
            "Choice mismatches here cover the complete replayed collection.",
        ]
    )
    replay_cells: list[list[str]] = []
    for (width, streams, arm), attempts in groups.items():
        for rep, row in enumerate(attempts):
            r = row or {}
            replay_cells.append(
                [
                    f"w{width}l{LAYERS[width]}",
                    str(streams),
                    LABELS[arm],
                    str(rep + 1),
                    _status(row),
                    *[
                        _number(r.get(k))
                        for k in ("engine", "collection_policy", "learning", "seconds")
                    ],
                    *_memory_cells(r),
                    str(r["sample_disagreements"])
                    if "sample_disagreements" in r
                    else "—",
                    r.get("error", ""),
                ]
            )
    text.append(
        _table(
            [
                "Model",
                "Streams",
                "Arm",
                "Repeat",
                "Status",
                "Shared engine s",
                "Collection s",
                "Learning s",
                "Sum s",
                "RSS GiB",
                "MPS GiB ≥",
                "MLX GiB",
                "Choice mismatches",
                "Error",
            ],
            replay_cells,
        )
    )
    (out / f"comparisons{suffix}.md").write_text("\n\n".join(text) + "\n")


def _missing_arms(
    prefix: str, attempts: dict[str, list[dict[str, Any] | None]]
) -> list[str]:
    """Coverage accepts OOM, but unsupported halves cannot replace both dtypes."""
    statuses = {
        arm: {r["status"] if r else "not_measured" for r in rows}
        for arm, rows in attempts.items()
    }
    executed = {"ok", "oom"}
    missing = [
        f"{prefix}/{arm}"
        for arm, states in statuses.items()
        if not states <= (executed | {"unsupported"} if arm in HALF else executed)
    ]
    if not any(statuses[arm] <= executed for arm in HALF):
        missing.append(f"{prefix}/no executed half arm")
    return missing


def _host_report(
    local: list[dict[str, Any]], replay: list[dict[str, Any]]
) -> HostReport:
    """Select latest attempts and summarize evidence before any rendering."""
    evidence = local + replay
    hostname = evidence[0]["host"]["hostname"]
    host = evidence[0]["host"]
    if any(r["host"] != host for r in evidence):
        raise ValueError(
            "host/software/source changed; use separate report directories"
        )
    latest = {
        (r["event"], r["width"], r["streams"], r["rows"], r["arm"]): r for r in local
    }
    replay_latest = {(r["width"], r["streams"], r["arm"], r["rep"]): r for r in replay}
    missing: list[str] = []
    captures: list[dict[str, Any]] = []
    for streams in STREAMS:
        captured = [r for r in evidence if r["streams"] == streams]
        if not captured:
            continue
        if len({r["capture_id"] for r in captured}) > 1:
            raise ValueError(
                f"multiple captures for {hostname}/{streams}; select one explicitly"
            )
        origins = [
            {
                "streams": streams,
                "capture_id": r["capture_id"],
                "host": r.get("capture_host"),
                "threads": r.get("capture_threads"),
                "revision": r["capture_revision"],
            }
            for r in captured
        ]
        if any(origin != origins[0] for origin in origins):
            raise ValueError("capture host/source/thread provenance changed")
        captures.append(origins[0])
        if origins[0]["host"] is None or origins[0]["threads"] is None:
            missing.append(f"capture/streams{streams}/host or threads unknown")
    groups = {
        (width, streams, arm): [
            replay_latest.get((width, streams, arm, rep)) for rep in range(3)
        ]
        for width in WIDTHS
        for streams in STREAMS
        for arm in ARMS
    }
    # Per-call comparisons use only the 256-stream capture.
    for width in WIDTHS:
        for phase, sizes in SIZES.items():
            for size in sizes:
                missing.extend(
                    _missing_arms(
                        f"{phase}/w{width}/b{size}",
                        {
                            arm: [latest.get((phase, width, 256, size, arm))]
                            for arm in ARMS
                        },
                    )
                )
    # The median-total replay owns the phase stack, so it sums to that total.
    medians: list[dict[str, Any]] = []
    spans: list[dict[str, int | float]] = []
    for width in WIDTHS:
        for streams in STREAMS:
            missing.extend(
                _missing_arms(
                    f"update/w{width}/streams{streams}",
                    {arm: groups[(width, streams, arm)] for arm in ARMS},
                )
            )
            valid_replays: list[dict[str, Any]] = []
            for arm in ARMS:
                complete = [
                    r
                    for r in groups[(width, streams, arm)]
                    if r and r["status"] == "ok"
                ]
                valid_replays.extend(complete)
                if len(complete) == 3:
                    medians.append(sorted(complete, key=lambda r: r["seconds"])[1])
            if valid_replays:
                span = max(r["ended"] for r in valid_replays) - min(
                    r["started"] for r in valid_replays
                )
                spans.append({"width": width, "streams": streams, "seconds": span})
    same_hour = bool(spans) and all(r["seconds"] <= 3600 for r in spans)
    summary = {
        "host": host,
        "missing": missing,
        "complete": not missing and same_hour,
        "captures": captures,
        "whole_updates": medians,
        "replay_windows": spans,
        "same_hour": same_hour,
    }
    if host["chip"] != "Apple M2":
        summary["supplemental_host"] = True
    return HostReport(latest, groups, summary)


def _call_charts(out: Path, suffix: str, data: HostReport) -> None:
    import matplotlib.pyplot as plt

    host = data.summary["host"]
    hostname = host["hostname"]
    for phase in SIZES:
        fig, axes = plt.subplots(4, 2, figsize=(17, 21))
        for width, axis in zip(WIDTHS, axes.flat):
            cells = []
            for arm in ARMS:
                points = [
                    data.calls.get((phase, width, 256, b, arm)) for b in SIZES[phase]
                ]
                axis.plot(
                    SIZES[phase],
                    [
                        r["rows_per_second"] if r and r["status"] == "ok" else np.nan
                        for r in points
                    ],
                    marker="o",
                    label=LABELS[arm],
                )
                cells.append([_status(r) for r in points])
            axis.set(
                title=f"Width {width} · {LAYERS[width]} layers",
                xlabel="Rows per call",
                ylabel="Decisions per second"
                if phase == "collection"
                else "Rows per second",
                xscale="log",
            )
            axis.set_xticks(SIZES[phase], [str(b) for b in SIZES[phase]])
            table = axis.table(
                cellText=cells,
                rowLabels=[LABELS[a] for a in ARMS],
                colLabels=[str(b) for b in SIZES[phase]],
                cellLoc="center",
                bbox=(0.32, -0.72, 0.68, 0.48),
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7)
        axes.flat[-1].axis("off")
        axes.flat[-1].legend(
            *axes.flat[0].get_legend_handles_labels(), loc="center", fontsize=9
        )
        axes.flat[-1].text(
            0,
            0.15,
            "Warmed medians; gaps are not zero.\n"
            "Measured means execution completed, not correctness passed.\n"
            f"First calls, setup, memory and correctness: comparisons{suffix}.md",
            fontsize=9,
            va="top",
            wrap=True,
        )
        fig.suptitle(
            f"{phase.capitalize()} · {host['chip']} · {host['memory_bytes'] / 2**30:g} GiB\n{hostname} · torch {host['torch']} · MLX {host['mlx']} · 256-stream capture"
        )
        fig.subplots_adjust(top=0.93, bottom=0.07, hspace=1.25, wspace=0.3)
        fig.savefig(
            out / f"{phase}{suffix}.png",
            dpi=180,
            facecolor="white",
            bbox_inches="tight",
        )
        plt.close(fig)


def _update_chart(out: Path, suffix: str, data: HostReport) -> None:
    from matplotlib.patches import Patch
    import matplotlib.pyplot as plt

    summary = data.summary
    host, captures = summary["host"], summary["captures"]
    hostname = host["hostname"]
    medians = {
        (r["width"], r["streams"], r["arm"]): r for r in summary["whole_updates"]
    }
    # One panel per model shape; three stream groups each contain all arms.
    fig = plt.figure(figsize=(19, 22))
    grid = fig.add_gridspec(4, 2, hspace=0.5, wspace=0.22)
    phases = (
        ("engine", "Shared capture engine + host work", "#808080"),
        ("collection_policy", "Collection policy calls", "#539caf"),
        ("learning", "Learning", "#db9250"),
    )
    labels = ("CPU8\nf32", "MPS\nf32", "MPS\nbf16", "MPS\nf16", "MLX\nf32")
    for index, width in enumerate(WIDTHS):
        sub = grid[index // 2, index % 2].subgridspec(
            2, 1, height_ratios=(3, 1), hspace=0.12
        )
        axis = fig.add_subplot(sub[0])
        rate = fig.add_subplot(sub[1], sharex=axis)
        for si, streams in enumerate(STREAMS):
            for ai, arm in enumerate(ARMS):
                x = si * 6 + ai
                attempts = data.updates[(width, streams, arm)]
                complete = [r for r in attempts if r and r["status"] == "ok"]
                middle = medians.get((width, streams, arm))
                if middle is None:
                    note = "/".join(sorted({_status(r) for r in attempts}))
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
                bottom = 0.0
                for key, _, color in phases:
                    axis.bar(
                        x,
                        middle[key],
                        0.83,
                        bottom=bottom,
                        color=color,
                        edgecolor="#333333",
                        linewidth=0.4,
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
                    for r in data.calls.values()
                    if r["event"] == "parity"
                    and r["width"] == width
                    and r["streams"] == streams
                    and r["arm"] == arm
                    and r["status"] == "ok"
                ]
                worst = min((r["gradient_cosine"] for r in parity), default=None)
                same = all(r["sample_disagreements"] == 0 for r in complete)
                text = f"cos {worst:.4f}" if worst is not None else "parity missing"
                if not same:
                    text += "\nchoices differ"
                else:
                    text += "\nchoices match"
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
        axis.margins(y=0.65)
        axis.set_ylim(bottom=0)
        axis.set_xlim(-0.7, 16.7)
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
    origins = sorted(
        {
            f"{c['host']['chip']} / {c['host']['hostname']} / {c['threads']} CPU threads"
            if c["host"]
            else "unknown capture host/threads"
            for c in captures
        }
    )
    origin_caption = "\n".join(origins) or "No capture provenance"
    mixed_host = any(
        c["host"] is None or c["host"]["hostname"] != hostname for c in captures
    )
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
        f"Replay: {host['chip']} · {host['memory_bytes'] / 2**30:g} GiB\n{hostname} · {host['cpu_threads']} CPU threads\ntorch {host['torch']} · MLX {host['mlx']}\nShared engine/host component from capture:\n{origin_caption}\n"
        + (
            "Cross-host phase sum, not a measured replay-host update.\n"
            if mixed_host
            else ""
        )
        + "Both-seat self-play; one real capture per stream count.\nMedian of three replays; same inputs and recorded decisions.\nHalf arms: bfloat16; float16 with gradient scaler.\nBars identify executed dtype and median-total phase times.\nRate counts learner decisions before filtering.\nCosines vs CPU; choices use common-uniform samples.\nNeither proves identical trained policies.\nFirst calls, memory and correctness: comparisons"
        + suffix
        + ".md\nMissing and failed arms remain visible.\nSame-hour condition for measured groups: "
        + ("met" if summary["same_hour"] else "NOT MET")
        + (
            "\nReceipt coverage: complete"
            if summary["complete"]
            else "\nReceipt coverage: incomplete"
        ),
        fontsize=9,
        va="top",
        linespacing=1.2,
    )
    legend.legend(
        handles=[Patch(facecolor=color, label=label) for _, label, color in phases],
        loc="upper left",
        bbox_to_anchor=(0, -0.12),
        fontsize=9,
    )
    fig.suptitle(
        ("Cross-host phase sum · " if mixed_host else "Whole-update replay · ")
        + host["chip"],
        fontsize=20,
        y=0.995,
    )
    fig.savefig(
        out / f"full_context{suffix}.png",
        dpi=180,
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(fig)


def report(bench_dir: Path, out: Path) -> None:
    """Keep one host/source/capture set per report; mark every absent bar."""
    import matplotlib

    matplotlib.use("Agg")

    rows = _rows(bench_dir / "bench.jsonl")
    updates = _rows(bench_dir / "updates.jsonl")
    out.mkdir(parents=True, exist_ok=True)
    hosts = sorted({r["host"]["hostname"] for r in rows + updates})
    if not hosts:
        raise ValueError("no measurement receipts; refusing an empty report")
    if (
        len(
            {
                r["host"]["hostname"]
                for r in rows + updates
                if r["host"]["chip"] == "Apple M2"
            }
        )
        > 1
    ):
        raise ValueError("multiple main M2 hosts; use separate report directories")
    summaries: list[dict[str, Any]] = []
    for hostname in hosts:
        data = _host_report(
            [r for r in rows if r["host"]["hostname"] == hostname],
            [r for r in updates if r["host"]["hostname"] == hostname],
        )
        summaries.append(data.summary)
        suffix = (
            ""
            if data.summary["host"]["chip"] == "Apple M2"
            else "-" + hostname.replace("/", "_")
        )
        _call_charts(out, suffix, data)
        _update_chart(out, suffix, data)
        _details(out, suffix, data)
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
        if name in ("bench", "matrix"):
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
    if getattr(args, "repeats", 3) < 1:
        parser.error("at least one measured repeat is required")
    host = _host()
    if host["chip"] != "Apple M2" and not args.laptop:
        parser.error("measurements require Apple M2; laptop runs require --laptop")
    if not args.laptop and platform.python_version_tuple()[:2] != ("3", "12"):
        parser.error("the rented M2 requires a fresh Python 3.12 environment")
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
                capture(args.out, args.streams, args.laptop)
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
