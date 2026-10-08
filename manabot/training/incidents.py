"""Bounded private numerical incidents registered by the regime executor.

These are debugging artifacts, never admitted policies or process-recovery
snapshots. Keep one terminal incident per stage, at most 32 MiB of evidence and
512 MiB of model/Adam/EMA state. No environment variables or arbitrary locals.
"""

from collections.abc import Mapping
import json
from pathlib import Path
import random
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

from manabot.model.policy_distribution import NumericalError

if TYPE_CHECKING:
    from manabot.sim.net_opponent import NetOpponentTrainer, RolloutBatch
    from manabot.training.models import TrainingRun

EVIDENCE_BYTES = 32 * 1024**2
STATE_BYTES = 512 * 1024**2
MAX_ROWS = 16


def _tensor_bytes(value: object) -> int:
    if isinstance(value, Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, Mapping):
        return sum(_tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(v) for v in value)
    return 0


def save_incident(
    directory: Path,
    error: NumericalError,
    trainer: "NetOpponentTrainer",
    batch: "RolloutBatch | None",
    run: "TrainingRun",
    iteration: int,
    ema: torch.nn.Module | None,
    rng: np.random.Generator,
) -> list[Path]:
    """Write unique terminal evidence; propagate capture errors to caller notes.

    RNG is captured at failure, not reconstructed at collection time. Exact
    offending rows and the full current learner/Adam are enough to inspect or
    repeat a failing loss; this is NOT a promise to replay an entire iteration.
    """
    directory.mkdir(exist_ok=False)
    tensors = dict(error.tensors)
    batch_tensors: dict[str, Tensor] = {}
    if batch is not None:
        for name in (
            "actions",
            "logprobs",
            "log_probabilities",
            "probabilities",
            "values",
            "rewards",
            "dones",
        ):
            batch_tensors[f"batch/{name}"] = torch.as_tensor(getattr(batch, name))
        batch_tensors.update(
            {f"observation/{k}": torch.as_tensor(v) for k, v in batch.obs.items()}
        )
        batch_tensors.update(
            {
                f"next_observation/{k}": torch.as_tensor(v)
                for k, v in batch.next_obs.items()
            }
        )
    tensors.update(batch_tensors)
    full_batch = _tensor_bytes(batch_tensors) <= EVIDENCE_BYTES // 2
    evidence: dict[str, Tensor] = {}
    shapes: dict[str, object] = {}
    remaining = EVIDENCE_BYTES
    failed = tensors.get("failed")
    row_shape = failed.shape if failed is not None else None
    row_indexes: Tensor | None = None
    # Most invariants reduce actions last. Retain the offending row indexes,
    # including a late row; never only save the first rows of a failed batch.
    valid = tensors.get("valid")
    if valid is not None:
        row_shape = valid.shape[:-1]
        if failed is not None and failed.shape == valid.shape:
            failed = failed.any(-1)
        if failed is not None and failed.shape == row_shape:
            row_indexes = failed.flatten().nonzero().flatten()[:MAX_ROWS]
        if row_indexes is None or not row_indexes.numel():
            row_indexes = torch.arange(min(MAX_ROWS, valid.numel() // valid.shape[-1]))
    for name, tensor in tensors.items():
        finite = torch.isfinite(tensor)
        finite_values = tensor[finite]
        shapes[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "nonfinite_count": int((~finite).sum()),
            "zero_count": int((tensor == 0).sum()),
            "finite_min": float(finite_values.min()) if finite_values.numel() else None,
            "finite_max": float(finite_values.max()) if finite_values.numel() else None,
        }
        if name in batch_tensors and full_batch:
            sample = tensor
        elif (
            row_shape is not None
            and row_indexes is not None
            and tuple(tensor.shape[: len(row_shape)]) == tuple(row_shape)
        ):
            sample = tensor.reshape(-1, *tensor.shape[len(row_shape) :])[
                row_indexes.to(tensor.device)
            ]
        else:
            sample = tensor.flatten()[:2048]
        needed = sample.numel() * sample.element_size()
        if needed <= remaining:
            evidence[name] = sample.detach().cpu().clone()
            remaining -= needed
        else:
            shapes[name]["omitted"] = "evidence byte budget"
    if row_indexes is not None:
        evidence["row_indexes"] = row_indexes.cpu()
    evidence_path = directory / "tensors.pt"
    torch.save(evidence, evidence_path)
    # torch optimizer/RNG state is a heterogeneous library boundary. Nothing
    # from process locals, credentials or arbitrary objects enters this payload.
    state: dict[str, Any] = {
        "model": trainer.agent.state_dict(),
        "gradients": {
            name: p.grad
            for name, p in trainer.agent.named_parameters()
            if p.grad is not None
        },
        "optimizer": trainer.optimizer.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": {
            str(device): torch.cuda.get_rng_state(device)
            for device in {p.device for p in trainer.agent.parameters() if p.is_cuda}
        },
        "collector_rng": trainer.collector._self_rng.get_state(),
        "numpy_rng": rng.bit_generator.state,
        "python_rng": random.getstate(),
    }
    size = _tensor_bytes(state)
    paths = [evidence_path]
    metadata = {
        "schema": 1,
        "invariant": error.invariant,
        "health": error.health,
        "run": run.id,
        "recipe": run.regime_digest,
        "stage": run.stages[-1].id,
        "iteration": iteration,
        "completed_updates": len(run.stages[-1].diagnostics),
        "source": {
            k: run.identities.get(k)
            for k in ("source_commit", "training_source_sha256", "torch", "cuda")
        },
        "tensors": shapes,
        "state_tensor_bytes": size,
        "state_limit_bytes": STATE_BYTES,
        "evidence_limit_bytes": EVIDENCE_BYTES,
        "full_batch_captured": bool(batch_tensors)
        and all(
            name in evidence and evidence[name].shape == value.shape
            for name, value in batch_tensors.items()
        ),
        "sampling_contract": "masked-log-softmax-v1",
        "attempted_collection": trainer.collector.stats.to_dict(),
        "last_valid_exports": [stage.artifacts for stage in run.stages],
        "monitoring_exports": [c.artifact for c in run.monitoring_checkpoints],
        "recovery": "diagnostic only; no engine journal or collection-time RNG",
    }
    if size <= STATE_BYTES:
        state_path = directory / "state.pt"
        torch.save(state, state_path)
        paths.append(state_path)
        metadata["state"] = "captured at failure (may itself be invalid)"
    else:
        metadata["state"] = "omitted: exceeds explicit state byte budget"
    metadata_path = directory / "incident.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    paths.append(metadata_path)
    return paths
