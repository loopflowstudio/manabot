"""Torch tensors for a numerical incident, and their serialization.

`capture_incident` bounds the evidence it keeps; the host owns the incident
directory and its metadata (`training/state/incidents.py`).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import random
from typing import TYPE_CHECKING, Any, BinaryIO

import numpy as np
import torch
from torch import Tensor

from manabot.backends.torch.learner import Learner
from manabot.policy.protocol import BackendPolicy
from manabot.training.state.backend import IncidentCapture, NumericalIncident

if TYPE_CHECKING:
    from manabot.training.rollout.collection import Collector, RolloutBatch


@contextmanager
def preserve_rng() -> Iterator[None]:
    with torch.random.fork_rng(devices=[]):
        yield


def save(value: object, stream: BinaryIO) -> None:
    torch.save(value, stream)


def _tensor_bytes(value: object) -> int:
    if isinstance(value, Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, Mapping):
        return sum(_tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(v) for v in value)
    return 0


def capture_incident(
    error: NumericalIncident,
    learner: Learner,
    collector: Collector,
    batch: RolloutBatch | None,
    ema: BackendPolicy | None,
    rng: np.random.Generator,
    *,
    evidence_bytes: int,
    max_rows: int,
) -> IncidentCapture:
    tensors = dict(error.tensors)
    batch_tensors: dict[str, Tensor] = {}
    if batch is not None:
        for name in (
            "actions",
            "logprobs",
            "values",
            "rewards",
            "dones",
        ):
            batch_tensors[f"batch/{name}"] = torch.as_tensor(getattr(batch, name))
        # Packed rows remain compact in incident artifacts. Offsets and slots
        # are required to reconstruct original Commands and full distributions.
        packed = batch.action_data
        for name in ("offsets", "slots", "log_probabilities"):
            batch_tensors[f"packed_actions/{name}"] = torch.as_tensor(
                getattr(packed, name)
            )
        batch_tensors["packed_actions/capacity"] = torch.tensor(packed.capacity)
        batch_tensors.update(
            {
                f"packed_actions/{key}": torch.as_tensor(value)
                for key, value in packed.fields.items()
            }
        )
        batch_tensors.update(
            {
                f"observation/{k}": torch.as_tensor(v)
                for k, v in batch.ordered_obs().items()
            }
        )
        batch_tensors.update(
            {
                f"next_observation/{k}": torch.as_tensor(v)
                for k, v in batch.next_obs.items()
            }
        )
    tensors.update(batch_tensors)
    full_batch = _tensor_bytes(batch_tensors) <= evidence_bytes // 2
    evidence: dict[str, Tensor] = {}
    shapes: dict[str, object] = {}
    remaining = evidence_bytes
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
            row_indexes = failed.flatten().nonzero().flatten()[:max_rows]
        if row_indexes is None or not row_indexes.numel():
            row_indexes = torch.arange(min(max_rows, valid.numel() // valid.shape[-1]))
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
    # torch optimizer/RNG state is a heterogeneous library boundary. Nothing
    # from process locals, credentials or arbitrary objects enters this payload.
    state: dict[str, Any] = {
        "model": learner.agent.state_dict(),
        "gradients": {
            name: p.grad
            for name, p in learner.agent.named_parameters()
            if p.grad is not None
        },
        "optimizer": learner.optimizer.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": {
            str(device): torch.cuda.get_rng_state(device)
            for device in {p.device for p in learner.agent.parameters() if p.is_cuda}
        },
        "collector_rng": collector.numerics.snapshot_rng(),
        "numpy_rng": rng.bit_generator.state,
        "python_rng": random.getstate(),
    }
    if any(p.device.type == "mps" for p in learner.agent.parameters()):
        state["mps_rng"] = torch.mps.get_rng_state()
    size = _tensor_bytes(state)
    return IncidentCapture(
        evidence,
        state,
        shapes,
        size,
        bool(batch_tensors)
        and all(
            name in evidence and evidence[name].shape == value.shape
            for name, value in batch_tensors.items()
        ),
    )
