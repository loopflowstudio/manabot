"""Bucketed MPS retest: real capture, isolated arm measurements and report.

Run `capture` on the rented M2 or laptop, then `matrix` for correctness and
per-call blocks, and `replays` for the three whole-update repetitions. `report`
keeps incomplete matrices and unsupported arms visible. Every worker records its
host, source revision, capture, dtype and load. Backend recommendations belong
to the report's interpretation, never to a hard-coded threshold. `_host_report`
selects and summarizes receipts once; tables and charts render that same evidence.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, closing, contextmanager, nullcontext
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import platform
import re
import resource
import select
import statistics
import subprocess
import sys
import textwrap
import threading
import time
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast
import uuid
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if TYPE_CHECKING:
    import torch

    from manabot.backends.torch.policy import ActorCritic
    from manabot.policy.prepare import Prepared
    from manabot.policy.protocol import Settings, Targets

Arm = Literal["cpu", "mps-f32", "mps-bf16", "mps-f16", "mlx-f32", "mlx-bf16", "mlx-f16"]
ARMS: tuple[Arm, ...] = (
    "cpu",
    "mps-f32",
    "mlx-f32",
    "mps-bf16",
    "mlx-bf16",
    "mps-f16",
    "mlx-f16",
)
HALF = ("mps-bf16", "mps-f16", "mlx-bf16", "mlx-f16")
WIDTHS = (64, 96, 160, 384, 512, 768, 1024)
STREAMS = (256, 512, 1024)
LAYERS = {64: 2, 96: 4, 160: 8, 384: 8, 512: 8, 768: 8, 1024: 8}
SIZES = {"learner": (64, 256, 512, 1024, 2048), "collection": (64, 256, 512)}
# Parity and per-call receipts share one ledger; whole-update replays have their own.
LEDGERS = {"bench": "bench.jsonl", "update": "updates.jsonl"}
LABELS = {
    "cpu": "PyTorch CPU · float32",
    "mps-f32": "PyTorch MPS · float32",
    "mps-bf16": "PyTorch MPS · bfloat16",
    "mps-f16": "PyTorch MPS · float16",
    "mlx-f32": "MLX · float32",
    "mlx-bf16": "MLX · bfloat16",
    "mlx-f16": "MLX · float16",
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
    def parameters(self) -> dict[str, np.ndarray]: ...
    def sync(self) -> None: ...
    def memory(self) -> dict[str, int]: ...
    def close(self) -> None: ...


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


def _agent(capture: Path, width: int) -> ActorCritic:
    import torch

    from manabot.backends.torch.checkpoint import load_checkpoint
    from manabot.backends.torch.policy import ActorCritic

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
    from manabot.backends.parity import minibatches, read_npz

    if phase == "learner":
        for obs, targets, settings, _ in minibatches(capture, None):
            yield Batch(obs, targets, settings)
    else:
        for path in sorted((capture / "collection").glob("*.npz")):
            yield Batch(read_npz(path))


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

        self.agent = agent.to("cpu" if arm == "cpu" else "mps").train()
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

    def parameters(self) -> dict[str, np.ndarray]:
        from manabot.policy.checkpoint import canonical_parameters

        return canonical_parameters(
            {
                name: p.detach().cpu().numpy().copy()
                for name, p in self.agent.named_parameters()
            }
        )

    def sync(self) -> None:
        import torch

        if self.device.type == "mps":
            torch.mps.synchronize()

    def _loss(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        import torch

        from manabot.backends.torch import rl
        from manabot.model.admission import bind_admitted
        from manabot.model.inference import transfer_admitted
        from manabot.training.updates.damped_policy import minibatch_objective

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

        loss, _, _ = minibatch_objective(
            rl,
            logits,
            values,
            obs["actions_valid"].bool(),
            actions=tensor(targets.actions),
            advantages=tensor(targets.advantages),
            value_targets=tensor(targets.value_targets),
            behavior_logs=tensor(targets.behavior_logs),
            reference=tensor(targets.reference),
            actor_mask=tensor(targets.actor_mask),
            categorical=self.agent.config.value_kind == "categorical_wdl",
            clip=settings.clip,
            collection_kl=settings.collection_kl,
            tau=settings.tau,
        )
        return loss, logits, values

    def _backward(self, loss: torch.Tensor) -> None:
        from manabot.backends.torch.learner import check_objective

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
    def __init__(self, agent: ActorCritic, arm: Arm = "mlx-f32") -> None:
        import mlx.core as mx

        from manabot.backends.mlx.checkpoint import build
        from manabot.policy.protocol import Precision

        # The backend no longer offers float16: that arm is refused as an
        # unsupported precision, which the ledger records as `unsupported`.
        precision = cast(
            Precision,
            {"mlx-f32": "float32", "mlx-bf16": "bfloat16", "mlx-f16": "float16"}[arm],
        )
        self.encoder = agent.observation_space.encoder
        self.policy, _ = build(
            agent.config,
            self.encoder.config,
            {k: v.detach().numpy().copy() for k, v in agent.state_dict().items()},
            precision=precision,
        )
        self.initial = self.policy.parameters()
        mx.eval(self.initial)
        mx.reset_peak_memory()
        self.reset()

    def close(self) -> None:
        self.policy.clear_cache()

    def reset(self) -> None:
        from manabot.backends.mlx.learner import Learner

        self.policy.update(self.initial)
        self.learner = Learner(self.policy, minibatch_rows=1)

    def parameters(self) -> dict[str, np.ndarray]:
        return self.learner.parameters()

    def sync(self) -> None:
        import mlx.core as mx

        mx.synchronize()

    def _trim(self, batch: Batch) -> Batch:
        import torch

        from manabot.model.inference import trim_observation
        from manabot.policy.buckets import bucket_sizes

        self.policy.check_observation(batch.observations)
        obs = {k: torch.from_numpy(v) for k, v in batch.observations.items()}
        sizes = bucket_sizes(batch.observations, self.encoder, objects=True)
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

    def _loss(self, batch: Batch) -> tuple[Any, Prepared]:
        """The shared update's loss on a trimmed batch, with its gradients pending."""
        import manabot.backends.mlx as backend
        from manabot.backends.parity import minibatch_loss

        assert batch.targets is not None and batch.settings is not None
        prepared = self.policy.prepare_admitted(batch.observations)
        loss = minibatch_loss(
            backend, self.policy, self.learner, prepared, batch.targets, batch.settings
        )
        return loss, prepared

    def evaluate(self, batch: Batch) -> Evaluation:
        loss, prepared = self._loss(self._trim(batch))
        logits, values = self.policy.evaluate(prepared)
        return Evaluation(
            _pad(logits, batch),
            values.reshape(-1, 1),
            float(loss),
            self.learner.gradients(loss),
        )

    def step(self, batch: Batch) -> float:
        import mlx.core as mx
        from mlx.utils import tree_flatten

        assert batch.settings is not None
        loss, _ = self._loss(self._trim(batch))
        self.learner.set_learning_rate(batch.settings.learning_rate)
        norm = self.learner.step(loss, batch.settings.max_grad_norm)
        self.sync()
        parameters = [p.reshape(-1) for _, p in tree_flatten(self.policy.parameters())]
        if not np.isfinite([loss, norm]).all() or not bool(
            mx.all(mx.isfinite(mx.concatenate(parameters)))
        ):
            raise ValueError("nonfinite MLX loss, gradient norm or parameters")
        return float(loss)

    @contextmanager
    def collection(self) -> Iterator[Collect]:
        with self.policy.forward_precision():
            try:
                self.policy.cache_programs()
                self.sync()

                def collect(batch: Batch) -> np.ndarray:
                    trimmed = self._trim(batch)
                    logits, _ = self.policy.evaluate(
                        self.policy.prepare_admitted(trimmed.observations)
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
    runner = (
        MLXRunner(agent, arm) if arm.startswith("mlx-") else TorchRunner(agent, arm)
    )
    # MLX owns its converted weights; do not keep a second Torch model resident.
    del agent
    try:
        yield runner
    finally:
        runner.close()


def _compare(
    actual: Evaluation, reference: Evaluation, legal: np.ndarray, arm: Arm
) -> dict[str, Any]:
    from manabot.backends.parity import gradient_agreement

    expected = reference.gradients
    if arm.startswith("mlx-"):
        from manabot.backends.mlx.checkpoint import mlx_gradients

        expected = mlx_gradients(expected)
    agreement = gradient_agreement(actual.gradients, expected)
    return {
        "logits_max_abs": float(
            np.abs(actual.logits[legal] - reference.logits[legal]).max()
        ),
        "values_max_abs": float(np.abs(actual.values - reference.values).max()),
        "loss": actual.loss,
        "reference_loss": reference.loss,
        "loss_relative": abs(actual.loss - reference.loss)
        / max(abs(reference.loss), 1e-12),
        "gradient_cosine": agreement.cosine,
        "gradient_min_cosine": agreement.min_cosine,
        "gradient_worst": agreement.worst,
        "passes_gradient": agreement.cosine > 0.999,
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
    """Retain CPU arrays, or load a reference with matching provenance."""
    from manabot.backends.parity import read_npz

    if arrays is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
        path.with_suffix(".json").write_text(json.dumps(metadata))
        return arrays
    if json.loads(path.with_suffix(".json").read_text()) != metadata:
        raise ValueError("CPU reference provenance differs")
    return read_npz(path)


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
    out: Path, metadata: dict[str, Any], width: int
) -> dict[Arm, dict[int, dict[str, Any]]]:
    """Read all arms once; latest failures supersede earlier successes too."""
    latest: dict[Arm, dict[int, dict[str, Any]]] = {arm: {} for arm in ARMS}
    for row in _rows(out / LEDGERS["bench"]):
        if (
            row["event"] == "parity"
            and row["width"] == width
            and row["arm"] in latest
            and row["host"] == metadata["host"]
            and row["capture_id"] == metadata["capture_id"]
        ):
            latest[row["arm"]][row["rows"]] = row
    return latest


def _parity_receipt(
    runner: Runner,
    out: Path,
    capture: Path,
    metadata: dict[str, Any],
    width: int,
    size: int,
    arm: Arm,
    batch: Batch,
) -> dict[str, Any]:
    """Gradients, one clipped step and collection logits against CPU float32.

    The CPU arm retains the reference arrays every other arm compares with, so
    it runs first for each host, capture, width and row count.
    """
    assert batch.settings is not None
    result = runner.evaluate(batch)
    runner.reset()
    runner.step(batch)
    parameters = runner.parameters()
    collection = _batch(capture, "collection", size)
    runner.reset()
    with runner.collection() as collect:
        collection_logits = collect(collection)
    arrays = _reference(
        _reference_path(out, metadata, width, size),
        metadata,
        {
            "collection_logits": collection_logits,
            "logits": result.logits,
            "values": result.values,
            "loss": np.asarray(result.loss),
            **{f"grad/{k}": v for k, v in result.gradients.items()},
            **{f"step/{k}": v for k, v in parameters.items()},
        }
        if arm == "cpu"
        else None,
    )

    def stored(prefix: str) -> dict[str, np.ndarray]:
        return {k[len(prefix) :]: v for k, v in arrays.items() if k.startswith(prefix)}

    expected_parameters = stored("step/")
    if arm.startswith("mlx-"):
        from manabot.backends.mlx.checkpoint import mlx_parameters

        expected_parameters = mlx_parameters(expected_parameters)
    if parameters.keys() != expected_parameters.keys():
        raise ValueError("one-step parameter sets differ")
    step_difference = max(
        float(np.abs(value - expected_parameters[name]).max())
        for name, value in parameters.items()
    )
    legal = collection.observations["actions_valid"] > 0
    reference = Evaluation(
        arrays["logits"], arrays["values"], float(arrays["loss"]), stored("grad/")
    )
    return {
        "step_parameter_max_abs": step_difference,
        "step_max_grad_norm": batch.settings.max_grad_norm,
        "passes_step": step_difference <= 1e-6,
        "collection_logits_max_abs": float(
            np.abs(collection_logits[legal] - arrays["collection_logits"][legal]).max()
        ),
        "sample_disagreements": int(
            np.count_nonzero(
                _choices(collection_logits, legal, 21501)
                != _choices(arrays["collection_logits"], legal, 21501)
            )
        ),
        **_compare(result, reference, batch.observations["actions_valid"] > 0, arm),
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
) -> Iterator[dict[str, Any] | None]:
    """Evaluate parity, or time calls admitted by the interleaving coordinator."""
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
        "load_before": os.getloadavg()[0],
    }
    batch = _batch(capture, "learner" if phase == "parity" else phase, size)
    with _runner(capture, width, arm) as runner:
        yield None  # All workers allocate before the coordinator admits the block.
        if phase == "parity":
            row.update(
                _parity_receipt(runner, out, capture, metadata, width, size, arm, batch)
            )
        else:
            times: list[float] = []
            tick = time.perf_counter()
            scope = (
                runner.collection()
                if phase == "collection"
                else nullcontext(runner.step)
            )
            with scope as call:
                if phase == "collection":
                    row["catalog_seconds"] = time.perf_counter() - tick
                yield None
                for rep in range(repeats + 1):
                    if phase == "learner":
                        runner.reset()
                    runner.sync()
                    tick = time.perf_counter()
                    call(batch)
                    runner.sync()
                    times.append(time.perf_counter() - tick)
                    yield None
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
    yield row


def update(
    out: Path, capture: Path, width: int, arm: Arm, laptop: bool, rep: int
) -> Iterator[dict[str, Any] | None]:
    """Replay original calls and minibatches; weights evolve only in learning."""
    meta = json.loads((capture / "capture.json").read_text())
    provenance = _provenance(capture, laptop)
    with _runner(capture, width, arm) as runner:
        yield None
        row: dict[str, Any] = {
            "event": "update",
            "arm": arm,
            "width": width,
            "rep": rep,
            "etude_run": os.environ.get("ETUDE_RUN"),
            "layers": LAYERS[width],
            **provenance,
            "started": time.time(),
            "load_before": os.getloadavg()[0],
        }
        tick = time.perf_counter()
        calls = decisions = 0
        sampled: list[np.ndarray] = []
        with runner.collection() as collect:
            policy_seconds = time.perf_counter() - tick
            yield None
            for batch in _batches(capture, "collection"):
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
                yield None
        actions = np.concatenate(sampled)
        sample_path = _reference_path(out, provenance, width, 0)
        expected_actions = _reference(
            sample_path, provenance, {"actions": actions} if arm == "cpu" else None
        )["actions"]
        if actions.shape != expected_actions.shape:
            raise ValueError("whole-update call rows differ")
        row["sample_disagreements"] = int(np.count_nonzero(actions != expected_actions))
        learning_seconds = 0.0
        steps = learner_rows = 0
        for batch in _batches(capture, "learner"):
            tick = time.perf_counter()
            runner.step(batch)
            learning_seconds += time.perf_counter() - tick
            steps += 1
            learner_rows += batch.rows
            yield None
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
    yield row


def capture(args: argparse.Namespace) -> None:
    """Frozen capture requires the original executor and world."""
    from manabot.training.regime import refuse_frozen_execution

    refuse_frozen_execution()


def _parity(args: argparse.Namespace, width: int, arm: Arm, size: int) -> None:
    """Correctness runs in isolation before the timed resident comparison."""
    try:
        # Use the same replies and cleanup as timing, but finish this arm
        # before allocating the next. The timeout covers the whole parity run.
        deadline = time.monotonic() + args.timeout
        paging = _paging_guard(args)
        with closing(CallWorker(args, "parity", width, arm, size, 0)) as worker:
            worker.paging = paging
            worker.timeout = max(0, deadline - time.monotonic())
            row = worker.receive()
            while row is None:
                worker.timeout = max(0, deadline - time.monotonic())
                row = worker.advance()
    except Exception as error:
        row = {
            "event": "parity",
            "width": width,
            "rows": size,
            "arm": arm,
            **_failure(error, arm),
            **_provenance(args.capture, args.laptop),
        }
    _emit(args.out / LEDGERS["bench"], row)
    if row["status"] == "paging":
        raise PagingDetected(row["error"])


def _unstable(before: float, after: float) -> bool:
    return after > 2 * before


def _retry_pending(row: dict[str, Any]) -> bool:
    """A block whose load doubled and whose repeat has not been selected yet."""
    return bool(row.get("load_retry_required") and not row.get("load_retry_complete"))


def _failure(error: Exception, arm: Arm) -> dict[str, str]:
    message = str(error)
    lower = message.lower()
    status = "failed"
    if isinstance(error, PagingDetected):
        status = "paging"
    elif isinstance(error, MemoryError) or "out of memory" in lower:
        status = "oom"
    elif arm in HALF and (
        ("autocast" in lower and isinstance(error, UserWarning))
        or (
            any(dtype in lower for dtype in ("bfloat16", "float16", "half"))
            and any(
                t in lower for t in ("not supported", "unsupported", "not implemented")
            )
        )
    ):
        status = "unsupported"
    return {"status": status, "error": f"{type(error).__name__}: {message}"}


class PagingDetected(RuntimeError):
    """The dedicated host swapped during this cell; no timing is selected."""


class PagingGuard:
    """Sample real macOS swap counters; used only on the dedicated M2.

    Poll while a worker is computing as well as between calls, so a paging
    cell stops without waiting for a long kernel or the worker timeout.
    Existing occupied swap alone does not count as new paging.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.next_sample = 0.0
        self.initial: tuple[int, int] | None = None
        self()

    def __call__(self) -> None:
        now = time.monotonic()
        if now < self.next_sample:
            return
        self.next_sample = now + 1.0
        output = subprocess.check_output(["vm_stat"], text=True)
        counts = tuple(
            int(re.search(rf"{name}:\s+(\d+)", output)[1])
            for name in ("Swapins", "Swapouts")
        )
        _emit(
            self.path,
            {
                "time": time.time(),
                "swapins_pages": counts[0],
                "swapouts_pages": counts[1],
                "raw": output,
            },
        )
        if self.initial is None:
            self.initial = (counts[0], counts[1])
        elif any(n > old for n, old in zip(counts, self.initial, strict=True)):
            raise PagingDetected(f"swap pages {self.initial} → {counts}; cell stopped")


def _paging_guard(args: argparse.Namespace) -> PagingGuard | None:
    """The dedicated M2 stops a cell on new swap; the shared laptop only labels load."""
    return None if args.laptop else PagingGuard(args.out / "host-observations.jsonl")


class CallWorker:
    """One isolated arm, blocked on stdin between synchronized calls.

    Each arm owns its model, optimizer, inference scope and allocator. Only one
    worker computes at a time; retained models still share physical host memory.
    Generators close their runner before sending the final receipt.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        phase: str,
        width: int,
        arm: Arm,
        size: int,
        rep: int,
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
            "--worker",
        ]
        if command == "bench":
            cmd += ["--phase", phase, "--rows", str(size)]
            if phase != "parity":
                cmd += ["--repeats", str(args.repeats)]
        else:
            cmd += ["--rep", str(rep)]
        if args.laptop:
            cmd += ["--laptop"]
        self.process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )
        self.timeout = args.timeout
        self.paging: PagingGuard | None = None

    def receive(self) -> dict[str, Any] | None:
        assert self.process.stdout is not None
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if self.paging is not None:
                self.paging()
            if remaining <= 0:
                raise TimeoutError(f"worker call timeout after {self.timeout}s")
            wait = min(remaining, 1.0) if self.paging is not None else remaining
            if not select.select([self.process.stdout], [], [], wait)[0]:
                continue
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f"worker exited {self.process.poll()}")
            if line.startswith("MPS_REPLY "):
                return json.loads(line.removeprefix("MPS_REPLY "))
            print(line, end="", file=sys.stderr)

    def advance(self) -> dict[str, Any] | None:
        assert self.process.stdin is not None
        self.process.stdin.write("next\n")
        self.process.stdin.flush()
        return self.receive()

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        for pipe in (self.process.stdin, self.process.stdout):
            if pipe is not None:
                pipe.close()


def _admitted(parity: dict[int, dict[str, Any]], size: int) -> dict[str, Any] | None:
    """The parity cell that admits a timed one: its row count for per-call
    timing; for a whole update any executed cell, else any failure to report."""
    if size:
        return parity.get(size)
    return next(
        (r for r in parity.values() if r["status"] == "ok"),
        next(iter(parity.values()), None),
    )


def _interleaved(
    args: argparse.Namespace,
    phase: str,
    width: int,
    *,
    size: int = 0,
    rep: int = 0,
    paired_arm: Arm | None = None,
) -> None:
    """Repeat a drifting block once; select one entire quieter comparison.

    Quieter means the lower arithmetic mean of endpoint loads. Both attempts
    remain in the ledger; a selection receipt republishes the chosen rows last
    so existing latest-attempt readers never combine arms from different tries.
    """
    metadata = _provenance(args.capture, args.laptop)
    arms = ARMS if paired_arm is None else ("cpu", paired_arm)
    # Admission belongs here, before any worker reads inputs or allocates a
    # model. Both retry attempts use the same parity evidence for every arm.
    parities = _latest_parity(args.out, metadata, width)
    attempts: list[list[dict[str, Any]]] = []
    ledger = args.out / LEDGERS["update" if phase == "update" else "bench"]
    for attempt in range(2):
        block = str(uuid.uuid4())
        workers: dict[Arm, CallWorker] = {}
        rows: dict[Arm, dict[str, Any]] = {}
        before, started = os.getloadavg()[0], time.time()
        paging = _paging_guard(args)
        paging_error: PagingDetected | None = None
        try:
            for arm in arms:
                rows[arm] = {
                    **metadata,
                    "event": phase,
                    "width": width,
                    "layers": LAYERS[width],
                    "rows": size,
                    "arm": arm,
                    "rep": rep,
                    "etude_run": os.environ.get("ETUDE_RUN"),
                    "block_id": block,
                    "attempt": attempt,
                    "interleaved": True,
                    "memory_scope": "per-process; CPU and one accelerator resident"
                    if paired_arm
                    else "per-process; all arm models resident",
                    "comparison": "paired" if paired_arm else "resident",
                    "paired_arm": paired_arm,
                }
                admitted = _admitted(parities[arm], size)
                if admitted is None or admitted["status"] != "ok":
                    rows[arm].update(
                        status=admitted["status"] if admitted else "failed",
                        error="parity not executable: "
                        + str(admitted and admitted.get("error")),
                    )
                    continue
                if phase != "update":
                    rows[arm]["parity"] = {
                        key: admitted[key]
                        for key in (
                            "gradient_cosine",
                            "gradient_min_cosine",
                            "logits_max_abs",
                            "passes_gradient",
                        )
                    }
                try:
                    worker = CallWorker(args, phase, width, arm, size, rep)
                    workers[arm] = worker
                    worker.paging = paging
                    ready = worker.receive()
                    if ready is not None:
                        rows[arm].update(ready)
                        worker.close()
                        del workers[arm]
                except PagingDetected:
                    raise
                except Exception as error:
                    rows[arm].update(_failure(error, arm))
                    if arm in workers:
                        workers.pop(arm).close()
            before = os.getloadavg()[0]
            started = time.time()
            while workers:
                for arm, worker in list(workers.items()):
                    try:
                        result = worker.advance()
                        if result is None:
                            continue
                        rows[arm].update(result)
                    except PagingDetected:
                        raise
                    except Exception as error:
                        rows[arm].update(_failure(error, arm))
                    worker.close()
                    del workers[arm]
            after = os.getloadavg()[0]
            ended = time.time()
        except PagingDetected as error:
            paging_error = error
            after, ended = os.getloadavg()[0], time.time()
            for row in rows.values():
                row.update(status="paging", error=str(error))
        finally:
            for worker in workers.values():
                worker.close()
        for row in rows.values():
            row.update(
                load_before=before,
                load_after=after,
                started=started,
                ended=ended,
                load_retry=attempt > 0,
                load_retry_required=_unstable(before, after),
            )
            _emit(ledger, row)
        if paging_error is not None:
            raise paging_error
        attempts.append(list(rows.values()))
        if not _unstable(before, after) or attempt:
            break
    if len(attempts) == 2:
        chosen = min(
            range(2),
            key=lambda i: attempts[i][0]["load_before"] + attempts[i][0]["load_after"],
        )
        for row in attempts[chosen]:
            _emit(
                ledger,
                {
                    **row,
                    "load_retry": True,
                    "selected_attempt": chosen,
                    "load_retry_complete": True,
                    "selection": "lower mean endpoint load",
                    "attempt_blocks": [a[0]["block_id"] for a in attempts],
                },
            )


def matrix(args: argparse.Namespace) -> None:
    for width in args.widths:
        for size in SIZES["learner"]:
            for arm in ARMS:
                _parity(args, width, arm, size)
            _interleaved(args, "learner", width, size=size)
            if size in SIZES["collection"]:
                _interleaved(args, "collection", width, size=size)


def replays(args: argparse.Namespace) -> None:
    metadata = _provenance(args.capture, args.laptop)
    for width in args.widths:
        parities = _latest_parity(args.out, metadata, width)
        for arm in ARMS:
            if not any(r["status"] == "ok" for r in parities[arm].values()):
                _parity(args, width, arm, 64)
        for rep in range(3):
            pairs = ARMS[1:] if args.pairwise else (None,)
            for paired_arm in pairs:
                _interleaved(args, "update", width, rep=rep, paired_arm=paired_arm)


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


def _load_label(row: dict[str, Any]) -> str:
    if row.get("paired_references"):
        return "six pairs; see references"
    before, after = row.get("load_before"), row.get("load_after")
    label = (
        f"{before:.2f}→{after:.2f}"
        if before is not None and after is not None
        else "load unknown"
    )
    if row.get("load_retry"):
        label += " (quieter of 2)" if row.get("load_retry_complete") else " (retry)"
    elif row.get("load_retry_required"):
        label += " (retry pending)"
    return label


def _arm_label(arm: Arm, host: dict[str, Any]) -> str:
    return (
        f"PyTorch CPU{host['cpu_threads']} · float32" if arm == "cpu" else LABELS[arm]
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
        "and MLX allocator peak. They measure different scopes and must not be summed. "
        "Resident comparisons keep every arm; paired comparisons keep CPU and one accelerator. Allocation failures describe this "
        "shared-memory setup, not standalone arm capacity. Endpoint loads label every "
        "measurement; quieter of 2 selects the lower mean endpoint load.",
        "## Shared engine component",
        _table(
            [
                "Streams",
                "Capture",
                "Capture machine",
                "Capture threads",
                "Capture source",
                "Load start→end",
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
                    _load_label(c["host"] or {}),
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
                            _arm_label(arm, host),
                            _status(row),
                            _load_label(r),
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
                        "Load start→end",
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
                            _arm_label(arm, host),
                            _status(row),
                            _load_label(r),
                            *[
                                _number(r.get(k))
                                for k in (
                                    "logits_max_abs",
                                    "collection_logits_max_abs",
                                    "values_max_abs",
                                )
                            ],
                            _number(loss_delta),
                            _number(r.get("loss_relative")),
                            _number(r.get("step_parameter_max_abs")),
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
                    "Load start→end",
                    "Learner logit Δ",
                    "Collection logit Δ",
                    "Value Δ",
                    "Loss abs Δ",
                    "Loss relative Δ",
                    "One-step parameter Δ",
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
                    _arm_label(arm, host),
                    str(rep + 1),
                    *[
                        time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(r[k]))
                        if k in r
                        else "—"
                        for k in ("started", "ended")
                    ],
                    _status(row),
                    _load_label(r),
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
                "Started UTC",
                "Ended UTC",
                "Status",
                "Load start→end",
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
    references = [
        reference
        for attempts in groups.values()
        for row in attempts
        if row
        for reference in row.get("paired_references", [])
    ]
    if references:
        text.extend(
            [
                "## Paired CPU references",
                "Each repetition runs CPU plus one accelerator at a time. The displayed CPU "
                "reference is the mean of six paired CPU timings within each repetition, then "
                "the median of those three means. Accelerator bars are medians of their three "
                "paired replays. All individual references and their loads remain below.",
                _table(
                    [
                        "Model",
                        "Streams",
                        "Rep",
                        "Paired accelerator",
                        "CPU seconds",
                        "Load start→end",
                    ],
                    [
                        [
                            f"w{r['width']}",
                            str(r["streams"]),
                            str(r["rep"]),
                            LABELS[r["paired_arm"]],
                            _number(r["seconds"]),
                            _load_label(r),
                        ]
                        for r in references
                    ],
                ),
            ]
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
    replay_latest = {
        (r["width"], r["streams"], r["arm"], r["rep"], r.get("paired_arm")): r
        for r in replay
    }

    def selected(width: int, streams: int, arm: Arm, rep: int) -> dict[str, Any] | None:
        matching = [
            r
            for r in replay_latest.values()
            if (r["width"], r["streams"], r["arm"], r["rep"])
            == (width, streams, arm, rep)
        ]
        if not matching:
            return None
        if len({r.get("comparison", "resident") for r in matching}) != 1:
            raise ValueError(
                "mixed resident and paired methods; use separate report directories"
            )
        if arm != "cpu" or matching[0].get("comparison") != "paired":
            if len(matching) != 1:
                raise ValueError("multiple references for one accelerator replay")
            return matching[0]
        # Each of the three repetitions contains all six CPU/accelerator pairs.
        # The CPU bar represents the arithmetic mean of those six references;
        # the report retains every paired CPU timing, not only this aggregate.
        if {r["paired_arm"] for r in matching} != set(ARMS[1:]) or any(
            r["status"] != "ok" or _retry_pending(r) for r in matching
        ):
            return None
        if len({r["learner_decisions"] for r in matching}) != 1:
            raise ValueError("paired CPU decision counts differ")
        result = dict(matching[0])
        result.update(
            comparison="paired-reference-mean",
            paired_arm=None,
            paired_references=matching,
            started=min(r["started"] for r in matching),
            ended=max(r["ended"] for r in matching),
            load_before=None,
            load_after=None,
            load_retry=False,
            load_retry_required=False,
        )
        for key in ("seconds", "engine", "collection_policy", "learning"):
            result[key] = statistics.mean(r[key] for r in matching)
        result["memory"] = {
            key: max(r["memory"].get(key, 0) for r in matching)
            for key in set().union(*(r["memory"] for r in matching))
        }
        return result

    missing: list[str] = [
        f"{r['event']}/w{r['width']}/{r['arm']}/load retry pending"
        for r in [*latest.values(), *replay_latest.values()]
        if _retry_pending(r)
    ]
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
        (width, streams, arm): [selected(width, streams, arm, rep) for rep in range(3)]
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
                    if r and r["status"] == "ok" and not _retry_pending(r)
                ]
                valid_replays.extend(complete)
                if len(complete) == 3:
                    medians.append(sorted(complete, key=lambda r: r["seconds"])[1])
            if valid_replays:
                span = max(r["ended"] for r in valid_replays) - min(
                    r["started"] for r in valid_replays
                )
                spans.append({"width": width, "streams": streams, "seconds": span})
    summary = {
        "host": host,
        "missing": missing,
        "complete": not missing,
        "captures": captures,
        "whole_updates": medians,
        "replay_windows": spans,
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
                    label=_arm_label(arm, host),
                )
                cells.append(
                    [
                        _status(r)
                        + (
                            "\nL " + _load_label(r).replace(" (quieter of 2)", "*")
                            if r
                            else ""
                        )
                        for r in points
                    ]
                )
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
                rowLabels=[_arm_label(a, host) for a in ARMS],
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
            "L = block start→end 1-minute load, shown per point.\n"
            "* = quieter of two blocks (lower mean endpoint load).\n"
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


def _update_chart(out: Path, suffix: str, hosts: list[HostReport]) -> None:
    """Measured host/shape panels only; missing cells belong in the matrix."""
    from matplotlib.patches import Patch
    import matplotlib.pyplot as plt

    panels = [
        (data, width, streams)
        for data in hosts
        for width, streams in sorted(
            {(row["width"], row["streams"]) for row in data.summary["whole_updates"]}
        )
    ]
    if not panels:
        return
    phases = (
        ("engine", "Shared engine + host work", "#808080"),
        ("collection_policy", "Collection policy", "#539caf"),
        ("learning", "Learning", "#db9250"),
    )
    figure_width = max(8.5, 5.5 * len(panels))
    fig = plt.figure(figsize=(figure_width, 7.5), facecolor="white")
    grid = fig.add_gridspec(
        2,
        len(panels),
        height_ratios=(3, 1),
        hspace=0.15,
        wspace=0.35,
        bottom=0.30,
        top=0.84,
    )
    for index, (data, width, streams) in enumerate(panels):
        host = data.summary["host"]
        medians = {
            r["arm"]: r
            for r in data.summary["whole_updates"]
            if r["width"] == width and r["streams"] == streams
        }
        axis = fig.add_subplot(grid[0, index])
        rate = fig.add_subplot(grid[1, index], sharex=axis)
        arms = [arm for arm in ARMS if arm in medians]
        for x, arm in enumerate(arms):
            middle = medians[arm]
            bottom = 0.0
            for key, _, color in phases:
                axis.bar(
                    x,
                    middle[key],
                    0.75,
                    bottom=bottom,
                    color=color,
                    edgecolor="#333333",
                    linewidth=0.4,
                )
                bottom += middle[key]
            axis.annotate(
                f"{bottom:.0f}",
                (x, bottom),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=10,
            )
            complete = data.updates[(width, streams, arm)]
            rate.bar(
                x,
                statistics.median(
                    r["learner_decisions"] / r["seconds"] for r in complete if r
                ),
                0.75,
                color="#539caf",
                edgecolor="#333333",
                linewidth=0.4,
            )
        axis.set(
            title=f"{host['chip']} · {host['memory_bytes'] / 2**30:g} GiB\n"
            f"w{width}l{LAYERS[width]} · {streams} streams",
            ylabel="Seconds per update",
        )
        axis.margins(y=0.15)
        axis.set_ylim(bottom=0)
        axis.tick_params(axis="x", labelbottom=False)
        labels = {
            "cpu": f"CPU{host['cpu_threads']}\nf32",
            "mps-f32": "MPS\nf32",
            "mps-bf16": "MPS\nbf16",
            "mps-f16": "MPS\nf16",
            "mlx-f32": "MLX\nf32",
            "mlx-bf16": "MLX\nbf16",
            "mlx-f16": "MLX\nf16",
        }
        rate.set_xticks(range(len(arms)), [labels[arm] for arm in arms], fontsize=9)
        rate.set_ylabel("Learner decisions/s", fontsize=9)
        for ax in (axis, rate):
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)
            ax.grid(axis="y", alpha=0.15)
    provenance = []
    for data in hosts:
        if not data.summary["whole_updates"]:
            continue
        host = data.summary["host"]
        provenance.append(
            f"{host['chip']}: PyTorch {host['torch']}, MLX {host['mlx']}, "
            f"source {host['revision'][:9]}"
        )
    same = all(
        r["sample_disagreements"] == 0
        for data, width, streams in panels
        for arm in ARMS
        for r in data.updates[(width, streams, arm)]
        if r and r["status"] == "ok"
    )
    cross_host = any(
        c["host"] is None or c["host"]["hostname"] != data.summary["host"]["hostname"]
        for data in hosts
        for c in data.summary["captures"]
    )
    caption = "\n".join(provenance)
    paired = any(
        r.get("comparison", "resident").startswith("paired")
        for data in hosts
        for r in data.summary["whole_updates"]
    )
    caption += (
        "\nThree CPU-paired replays per accelerator; CPU = median of three six-reference means. "
        if paired
        else "\nMedian of three interleaved replays; phase stack from the median-total replay. "
    )
    caption += (
        "Same sampled decisions as CPU."
        if same
        else "Sampled decisions differ; see report."
    )
    caption += "\nHalf has dtype-specific gates; post-step deltas exceed 1e-6. Training equivalence is unmeasured. "
    caption += (
        "Cross-host engine component; see report."
        if cross_host
        else "Engine component from same-host capture."
    )
    caption += "\nShared host. Loads, memory, gradients and missing cells: experiments/mps_retest.md."
    caption = "\n".join(
        textwrap.fill(line, width=int(figure_width * 13))
        for line in caption.splitlines()
    )
    fig.suptitle("Whole training update · measured cells", fontsize=18, y=0.97)
    fig.legend(
        handles=[Patch(facecolor=color, label=label) for _, label, color in phases],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
        frameon=False,
    )
    fig.text(0.04, 0.02, caption, fontsize=9, va="bottom", linespacing=1.4)
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

    rows = _rows(bench_dir / LEDGERS["bench"])
    updates = _rows(bench_dir / LEDGERS["update"])
    out.mkdir(parents=True, exist_ok=True)
    chips = {r["host"]["hostname"]: r["host"]["chip"] for r in rows + updates}
    hosts = sorted(chips)
    if not hosts:
        raise ValueError("no measurement receipts; refusing an empty report")
    if list(chips.values()).count("Apple M2") > 1:
        raise ValueError("multiple main M2 hosts; use separate report directories")
    reports: list[HostReport] = []
    for hostname in hosts:
        data = _host_report(
            [r for r in rows if r["host"]["hostname"] == hostname],
            [r for r in updates if r["host"]["hostname"] == hostname],
        )
        reports.append(data)
        suffix = (
            ""
            if len(hosts) == 1 or data.summary["host"]["chip"] == "Apple M2"
            else "-" + hostname.replace("/", "_")
        )
        _call_charts(out, suffix, data)
        if suffix:
            _update_chart(out, suffix, [data])
        _details(out, suffix, data)
    # The unsuffixed chart holds every host's measured panels, never one axis.
    _update_chart(out, "", reports)
    (out / "summary.json").write_text(
        json.dumps(
            {
                "hosts": [data.summary for data in reports],
                "measurements": rows,
                "updates": updates,
            },
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
            command.add_argument(
                "--worker", action="store_true", help=argparse.SUPPRESS
            )
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
    commands.choices["replays"].add_argument(
        "--pairwise",
        action="store_true",
        help="interleave CPU with one accelerator at a time; retain every CPU reference",
    )
    args = parser.parse_args()
    if args.command == "report":
        report(args.bench, args.out)
        return
    import torch

    torch.set_num_threads(4 if args.laptop else 8)
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
    if (
        args.command in ("bench", "update")
        and not args.worker
        and (args.command == "update" or args.phase != "parity")
    ):
        parser.error("timed comparisons use matrix or replays, interleaved across arms")
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
    if args.command not in ("bench", "update"):
        {"capture": capture, "matrix": matrix, "replays": replays}[args.command](args)
        return
    ledger = args.out / LEDGERS[args.command]

    def reply(row: dict[str, Any] | None) -> None:
        """Workers answer the coordinator; a direct parity cell owns its receipt."""
        if args.worker:
            print("MPS_REPLY " + json.dumps(row, allow_nan=False), flush=True)
        elif row is not None:
            _emit(ledger, row)

    try:
        with warnings.catch_warnings():
            # Unsupported autocast often warns and silently becomes float32.
            warnings.filterwarnings(
                "error", message=".*autocast.*", category=UserWarning
            )
            work = (
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
                if args.command == "bench"
                else update(
                    args.out, args.capture, args.width, args.arm, args.laptop, args.rep
                )
            )
            with closing(work):
                for row in work:
                    reply(row)
                    # Each `None` is a pause; the coordinator admits the next call.
                    if (
                        args.worker
                        and row is None
                        and sys.stdin.readline().strip() != "next"
                    ):
                        break
    except Exception as error:
        reply(
            {
                "event": getattr(args, "phase", "update"),
                "width": args.width,
                "rows": getattr(args, "rows", 0),
                "arm": args.arm,
                "rep": getattr(args, "rep", 0),
                **_failure(error, args.arm),
                "load_after": os.getloadavg()[0],
                **_provenance(args.capture, args.laptop),
            }
        )
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
