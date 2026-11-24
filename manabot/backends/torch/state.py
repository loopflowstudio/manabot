"""Torch policy and Adam continuation, and bounded numerical incident evidence.

`capture_incident` bounds the evidence it keeps; the host owns the incident
directory and its metadata (`training/state/incidents.py`).
"""

from __future__ import annotations

from collections.abc import Mapping
import random
from typing import TYPE_CHECKING, Any, BinaryIO

import numpy as np
import torch
from torch import Tensor

from manabot.backends.torch.checkpoint import (
    policy_arrays as policy_arrays,
    policy_buffers as policy_buffers,
)
from manabot.backends.torch.learner import Learner
from manabot.backends.torch.policy import ActorCritic
from manabot.env.decision_memory import MEMORY_FIELDS
from manabot.env.memory_store import PrefixObservation, archive_memory
from manabot.policy.checkpoint import canonical_name
from manabot.policy.protocol import BackendPolicy
from manabot.training.state.backend import IncidentCapture, NumericalIncident
from manabot.training.state.optimizer import AdamState, admit_adam, admit_policy_arrays

if TYPE_CHECKING:
    from etude.train.tracking import TrainingRun

    from manabot.training.loops.self_play import Counts
    from manabot.training.rollout.collection import Collector, RolloutBatch


def parameter_shapes(agent: ActorCritic) -> dict[str, tuple[int, ...]]:
    return {
        canonical_name(name): tuple(p.shape) for name, p in agent.named_parameters()
    }


def load_policy_arrays(agent: ActorCritic, arrays: Mapping[str, np.ndarray]) -> None:
    """Copy into the live parameters so the optimizer keeps its references."""
    admit_policy_arrays(arrays, parameter_shapes(agent))
    with torch.no_grad():
        for name, parameter in agent.named_parameters():
            parameter.copy_(torch.from_numpy(arrays[canonical_name(name)]))
    agent.clear_cache()


def adam_state(learner: Learner, *, step: int) -> AdamState:
    steps: dict[str, int] = {}
    first: dict[str, np.ndarray] = {}
    second: dict[str, np.ndarray] = {}
    for name, parameter in learner.agent.named_parameters():
        key = canonical_name(name)
        entry = learner.optimizer.state.get(parameter)
        if entry:
            steps[key] = int(entry["step"].item())
            first[key] = entry["exp_avg"].detach().cpu().numpy()
            second[key] = entry["exp_avg_sq"].detach().cpu().numpy()
        elif step:
            steps[key] = 0
    adam = AdamState(step, steps, first, second)
    admit_adam(adam, parameter_shapes(learner.agent), step=step)
    return adam


def load_adam(learner: Learner, adam: AdamState) -> int:
    """Replace live Adam history while preserving optimizer groups and rate."""
    admit_adam(adam, parameter_shapes(learner.agent), step=adam.step)
    entries = {}
    for name, parameter in learner.agent.named_parameters():
        key = canonical_name(name)
        if count := adam.param_steps.get(key, 0):
            entries[parameter] = {
                "step": torch.tensor(float(count), dtype=torch.float32, device="cpu"),
                "exp_avg": torch.tensor(adam.exp_avg[key], device=parameter.device),
                "exp_avg_sq": torch.tensor(
                    adam.exp_avg_sq[key], device=parameter.device
                ),
            }
    learner.optimizer.state.clear()
    learner.optimizer.state.update(entries)
    return 0


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
    memory_tensors: dict[str, Tensor] = {}
    memory_metadata: dict[str, object] = {}
    memory_bytes = 0
    selected_all = True
    if batch is not None:
        # Select before gathering any observation. Failed tensors may describe
        # either learner rows or action slots; reduce only the latter.
        failure = tensors.get("failed")
        selected = np.arange(min(max_rows, batch.actions.size), dtype=np.int64)
        if "indexes" in tensors:
            indexes = tensors["indexes"].detach().cpu().flatten()
            if ((indexes < 0) | (indexes >= batch.actions.size)).any():
                raise ValueError("invalid incident rollout indexes")
            selected = indexes.numpy()[:max_rows]
            if (
                failure is not None
                and failure.ndim
                and failure.shape[0] == len(indexes)
            ):
                reduced = failure.detach().cpu().reshape(len(indexes), -1).any(-1)
                offending = indexes[reduced].numpy()[:max_rows]
                if len(offending):
                    selected = offending
        elif failure is not None:
            failed_rows = failure.detach().cpu()
            if tuple(failed_rows.shape) == batch.actions.shape:
                offending = failed_rows.flatten().nonzero().flatten().numpy()[:max_rows]
            elif tuple(failed_rows.shape[:-1]) == batch.actions.shape:
                offending = (
                    failed_rows.any(-1).flatten().nonzero().flatten().numpy()[:max_rows]
                )
            else:
                offending = np.array([], dtype=np.int64)
            if len(offending):
                selected = offending
        selected_all = (
            np.array_equal(selected, np.arange(batch.actions.size))
            and len(batch.next_done) <= max_rows
        )
        batch_tensors["batch_row_indexes"] = torch.as_tensor(selected)
        for name in ("actions", "logprobs", "values", "rewards", "dones"):
            batch_tensors[f"batch/{name}"] = torch.as_tensor(
                getattr(batch, name).reshape(-1)[selected]
            )
        packed = type(batch.action_data).concatenate(
            [batch.action_data.row(int(row)) for row in selected]
        )
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
        recorded = batch.recorded.index[selected]
        batch_tensors.update(
            {
                f"observation/{key}": torch.as_tensor(value[recorded])
                for key, value in batch.recorded.rows.items()
            }
        )
        batch_tensors.update(
            {
                f"next_observation/{key}": torch.as_tensor(value[:max_rows])
                for key, value in batch.next_obs.items()
                if key not in MEMORY_FIELDS
            }
        )
        if batch.recorded.memory_capacities is not None:
            prefixes = [batch.recorded.memory[int(row)] for row in recorded]
            next_count = 0
            if isinstance(batch.next_obs, PrefixObservation):
                next_prefixes = batch.next_obs.prefixes[:max_rows]
                next_count = len(next_prefixes)
                prefixes.extend(next_prefixes)
            archive = archive_memory(prefixes, batch.recorded.memory_capacities)
            memory_metadata = {
                "required_bytes": archive.nbytes,
                "capacities": list(archive.capacities),
                "learner_rows": len(selected),
                "bootstrap_rows": next_count,
                "identities": [
                    {"game": p.game, "viewer": p.viewer, "branch": p.branch}
                    for p in archive.identities
                ],
                "keys": list(archive.keys),
            }
            # Atomic prefix capture: no partial detail rows presented as exact.
            if archive.nbytes <= evidence_bytes:
                memory_bytes = archive.nbytes
                memory_tensors = {
                    f"memory/chunks/{i}/{field}": torch.from_numpy(rows.copy())
                    for i, chunk in enumerate(archive.chunks)
                    for field, rows in zip(MEMORY_FIELDS, chunk, strict=True)
                }
                memory_tensors.update(
                    {
                        f"memory/{name}": torch.from_numpy(
                            getattr(archive, name).copy()
                        )
                        for name in ("offsets", "indexes")
                    }
                )
                memory_metadata["complete"] = True
            else:
                memory_metadata["complete"] = False
                memory_metadata["omitted"] = (
                    "complete prefixes exceed evidence byte budget"
                )
    tensors.update(batch_tensors)
    evidence: dict[str, Tensor] = dict(memory_tensors)
    shapes: dict[str, object] = (
        {"decision_memory": memory_metadata} if memory_metadata else {}
    )
    remaining = evidence_bytes - memory_bytes
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
        if name in batch_tensors:
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
    if batch is not None:
        shapes["batch_layout"] = {
            "trajectory_shape": list(batch.actions.shape),
            "selected_flat_indexes": selected.tolist(),
        }
    if row_indexes is not None:
        needed = row_indexes.numel() * row_indexes.element_size()
        if needed <= remaining:
            evidence["row_indexes"] = row_indexes.cpu()
            remaining -= needed
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
        and selected_all
        and bool(memory_metadata.get("complete", True))
        and all(
            name in evidence and evidence[name].shape == value.shape
            for name, value in batch_tensors.items()
        ),
    )


def restore(
    run: TrainingRun, learner: Learner, ema: BackendPolicy | None, counts: Counts
) -> None:
    objects: dict[str, object] = {
        "learner": learner.agent,
        "optimizer": learner.optimizer,
    }
    if ema is not None:
        objects["ema"] = ema
    run.resume(**objects, extra=counts)


def framework() -> dict[str, str | None]:
    return {"torch": str(torch.__version__), "cuda": torch.version.cuda}
