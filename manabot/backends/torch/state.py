"""Torch training-state codec, Adam admission and exact RNG continuation.

Payloads retain their historical nested Torch schema. The host owns artifact
hashing, immutable publication, lineage and restoration order.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
import random
from typing import TYPE_CHECKING, Any, BinaryIO, cast

import numpy as np
import torch
from torch import Tensor

from manabot.backends.torch.learner import Learner
from manabot.backends.torch.policy import ActorCritic
from manabot.env.memory_store import MEMORY_FIELDS, PrefixObservation, archive_memory
from manabot.infra.config import ModelConfig
from manabot.policy.checkpoint import pretrained_memory_branch
from manabot.policy.protocol import BackendPolicy
from manabot.training.state.backend import IncidentCapture, NumericalIncident

if TYPE_CHECKING:
    from manabot.training.rollout.collection import RolloutBatch, SeatRoutedCollector


@contextmanager
def preserve_rng() -> Iterator[None]:
    with torch.random.fork_rng(devices=[]):
        yield


def policy_state(policy: BackendPolicy) -> dict[str, Any]:
    """Serialization view; snapshot owners copy it before publishing."""
    return cast(ActorCritic, policy).state_dict()


def restore_policy(
    policy: BackendPolicy,
    state: dict[str, Any],
    *,
    source_config: ModelConfig | None = None,
) -> None:
    """Strict restoration; pretrained branches alone may initialize memory keys."""
    agent = cast(ActorCritic, policy)
    branch = source_config is not None and pretrained_memory_branch(
        source_config, agent.config, agent.observation_space.encoder.config
    )
    if not branch:
        agent.load_state_dict(state, strict=True)
        return
    target = agent.state_dict()
    allowed = {
        name
        for name in target
        if name.startswith("decision_memory.") or name.endswith(".memory_gate")
    }
    if set(target) - set(state) != allowed or set(state) - set(target):
        raise ValueError(
            "pretrained copy would leave non-memory parameters unused or uninitialized"
        )
    # Strict loading also checks every shared buffer and parameter shape.
    agent.load_state_dict({**target, **state}, strict=True)


def optimizer_state(learner: object) -> dict[str, Any]:
    """Serialization view; ordinary synchronous exports need no tensor copy."""
    return cast(Learner, learner).optimizer.state_dict()


def restore_optimizer(learner: object, state: dict[str, Any]) -> None:
    cast(Learner, learner).optimizer.load_state_dict(state)


def framework_rng() -> object:
    return torch.get_rng_state()


def restore_rng(state: object) -> None:
    if not isinstance(state, torch.Tensor):
        raise TypeError("Torch RNG state requires a tensor")
    torch.set_rng_state(state)


def save(value: object, stream: BinaryIO) -> None:
    torch.save(value, stream)


def load(stream: BinaryIO) -> object:
    return torch.load(stream, map_location="cpu", weights_only=False)


def parameter_names(policy: BackendPolicy) -> list[str]:
    return list(dict(cast(ActorCritic, policy).named_parameters()))


def validate_policy(policy: BackendPolicy, *, label: str = "learning-state") -> None:
    if any(
        not torch.isfinite(value).all()
        for value in cast(ActorCritic, policy).state_dict().values()
    ):
        raise ValueError(f"{label} policy contains nonfinite tensors")


def clone_policy(policy: BackendPolicy) -> BackendPolicy:
    return deepcopy(policy)


def validate_adam(optimizer: torch.optim.Adam) -> None:
    """Reject absent/corrupt moments; unused parameters may legitimately lack state."""
    if not optimizer.state:
        raise ValueError("learning-state Adam has no populated state")
    for parameter, state in optimizer.state.items():
        if not isinstance(parameter, torch.Tensor) or not isinstance(state, dict):
            raise ValueError("invalid Adam parameter state")
        step = state.get("step")
        if (
            not isinstance(step, torch.Tensor)
            or step.numel() != 1
            or not torch.isfinite(step).all()
            or step.item() <= 0
            or step.item() != int(step.item())
        ):
            raise ValueError("invalid Adam step counter")
        for name in ("exp_avg", "exp_avg_sq"):
            value = state.get(name)
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != parameter.shape
                or not torch.isfinite(value).all()
                or (name == "exp_avg_sq" and (value < 0).any())
            ):
                raise ValueError(f"invalid Adam moment: {name}")


def admit_adam(path: str, policy: BackendPolicy) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or set(state) != {"state", "param_groups"}:
        raise ValueError("invalid Adam artifact")
    optimizer = torch.optim.Adam(cast(ActorCritic, policy).parameters(), eps=1e-5)
    groups = state["param_groups"]
    expected_group = optimizer.state_dict()["param_groups"][0]
    if (
        not isinstance(groups, list)
        or len(groups) != 1
        or not isinstance(groups[0], dict)
        or groups[0].get("params") != expected_group["params"]
        or not isinstance(state["state"], dict)
        or not set(state["state"]).issubset(expected_group["params"])
        or any(
            groups[0].get(key) != expected_group[key]
            for key in (
                "betas",
                "eps",
                "weight_decay",
                "amsgrad",
                "maximize",
                "capturable",
                "differentiable",
            )
        )
    ):
        raise ValueError("learning-state Adam parameter groups or settings mismatch")
    optimizer.load_state_dict(state)
    validate_adam(optimizer)
    return state


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
    collector: SeatRoutedCollector,
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
                    f"memory/chunks/{i}": torch.from_numpy(rows.copy())
                    for i, rows in enumerate(archive.chunks)
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


def validate_optimizer(learner: object) -> None:
    validate_adam(cast(Learner, learner).optimizer)
