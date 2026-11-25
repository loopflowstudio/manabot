"""MLX policy, Adam and sampling arrays for portable self-play recovery.

Numerical incidents retain native MLX tensors and failure-time sampling keys.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import random
from typing import TYPE_CHECKING, BinaryIO

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

from manabot.backends.mlx.checkpoint import (
    canonical_from_mlx,
    canonical_groups,
    mlx_names,
    mlx_parameters,
    policy_arrays as policy_arrays,
    policy_buffers as policy_buffers,
)
from manabot.backends.mlx.learner import Learner
from manabot.backends.mlx.policy import MLXActorCritic
from manabot.training.rollout.numerics import CollectionSession
from manabot.training.state.backend import IncidentCapture, NumericalIncident
from manabot.training.state.optimizer import AdamState, admit_adam, admit_policy_arrays

if TYPE_CHECKING:
    from manabot.training.rollout.collection import Collector, RolloutBatch


def parameter_shapes(agent: MLXActorCritic) -> dict[str, tuple[int, ...]]:
    native = dict(tree_flatten(agent.trainable_parameters()))
    shapes = {}
    for name, leaves in canonical_groups(native.keys()).items():
        shape = native[leaves[0]].shape
        shapes[name] = (
            shape if len(leaves) == 1 else (len(leaves) * shape[0], *shape[1:])
        )
    return shapes


def load_policy_arrays(agent: MLXActorCritic, arrays: Mapping[str, np.ndarray]) -> None:
    """Install validated master weights and drop cached inference programs."""
    admit_policy_arrays(arrays, parameter_shapes(agent))
    agent.update(
        tree_unflatten([(k, mx.array(v)) for k, v in mlx_parameters(arrays).items()])
    )
    agent.clear_cache()
    mx.eval(agent.parameters())


def adam_state(learner: Learner, *, step: int) -> AdamState:
    state = dict(tree_flatten(learner.optimizer.state))
    if int(state["step"]) != step:
        raise ValueError("adam/step differs from the optimizer count")
    steps: dict[str, int] = {}
    first: dict[str, np.ndarray] = {}
    second: dict[str, np.ndarray] = {}
    shapes = parameter_shapes(learner.agent)
    if step:
        for name in shapes:
            leaves = mlx_names(name)
            ages = {step - int(state.get(f"{leaf}.lag", 0)) for leaf in leaves}
            if len(ages) != 1:
                raise ValueError(f"adam/param_steps/{name}: split projections differ")
            steps[name] = ages.pop()
            if steps[name]:
                for leaf in leaves:
                    first[leaf] = np.asarray(state[f"{leaf}.m"])
                    second[leaf] = np.asarray(state[f"{leaf}.v"])
    adam = AdamState(step, steps, canonical_from_mlx(first), canonical_from_mlx(second))
    admit_adam(adam, shapes, step=step)
    return adam


def load_adam(learner: Learner, adam: AdamState) -> int:
    """Retain each leaf's bias-correction clock across dense MLX updates."""
    shapes = parameter_shapes(learner.agent)
    admit_adam(adam, shapes, step=adam.step)
    native = dict(tree_flatten(learner.agent.trainable_parameters()))
    first, second = mlx_parameters(adam.exp_avg), mlx_parameters(adam.exp_avg_sq)
    state = {
        "step": mx.array(adam.step, dtype=mx.uint64),
        "learning_rate": learner.optimizer.learning_rate,
    }
    for name, count in adam.param_steps.items():
        for leaf in mlx_names(name):
            state[f"{leaf}.m"] = (
                mx.array(first[leaf]) if count else mx.zeros_like(native[leaf])
            )
            state[f"{leaf}.v"] = (
                mx.array(second[leaf]) if count else mx.zeros_like(native[leaf])
            )
            if count < adam.step:
                state[f"{leaf}.lag"] = mx.array(adam.step - count, dtype=mx.uint64)
    learner.optimizer.state = tree_unflatten(state)
    return sum(count < adam.step for count in adam.param_steps.values())


def framework() -> dict[str, str | None]:
    return {"mlx": mx.__version__}


def training_arrays(
    learner: Learner, ema: MLXActorCritic | None
) -> dict[str, mx.array]:
    trees = {"learner": learner.agent.parameters(), "adam": learner.optimizer.state}
    if ema is not None:
        trees["ema"] = ema.parameters()
    return dict(tree_flatten(trees))


def sampling_domain(device: str) -> str:
    return "mlx"


def sampling_state(session: CollectionSession) -> dict[str, np.ndarray]:
    return {
        "rng.collector": np.asarray(session.snapshot_rng()),
        "rng.mlx": np.asarray(mx.random.state[0]),
    }


def validate_sampling(device: str, arrays: Mapping[str, np.ndarray]) -> None:
    if differing := arrays.keys() ^ {"rng.collector", "rng.mlx"}:
        raise ValueError(f"sampling.safetensors: incompatible keys {sorted(differing)}")
    for name, key in arrays.items():
        if key.shape != (2,) or key.dtype != np.uint32:
            raise ValueError(f"sampling.safetensors: {name} needs a uint32 key pair")


def restore_sampling(
    session: CollectionSession, arrays: Mapping[str, np.ndarray]
) -> None:
    session.restore_rng(arrays["rng.collector"])
    high, low = arrays["rng.mlx"].tolist()
    mx.random.seed((high << 32) | low)


def restore_legacy(
    folder: Path, learner: Learner, ema: MLXActorCritic | None, *, step: int
) -> None:
    raise ValueError(
        "learner.pt: this save was written by Torch object registration; "
        "resume once on Torch to write a portable save"
    )


@dataclass
class IncidentTensors:
    arrays: dict[str, mx.array] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)


def save(value: object, stream: BinaryIO) -> None:
    if not isinstance(value, IncidentTensors):
        raise TypeError("expected MLX incident tensors")
    mx.save_safetensors(stream, value.arrays, value.metadata)


def capture_incident(
    error: NumericalIncident,
    learner: Learner,
    collector: Collector,
    batch: RolloutBatch | None,
    ema: MLXActorCritic | None,
    rng: np.random.Generator,
    *,
    evidence_bytes: int,
    max_rows: int,
) -> IncidentCapture:
    """Retain bounded failure samples, their rollout coordinates, and failure-time state."""
    evidence = IncidentTensors()
    shapes: dict[str, object] = {}
    remaining = evidence_bytes

    def retain(
        name: str, value: mx.array | np.ndarray, sample: mx.array | np.ndarray
    ) -> None:
        nonlocal remaining
        detail = {"shape": list(value.shape), "dtype": str(value.dtype)}
        if sample.nbytes <= remaining:
            evidence.arrays[name] = mx.array(sample)
            remaining -= sample.nbytes
        else:
            detail["omitted"] = "evidence byte budget"
        shapes[name] = detail

    # The gradient error can carry the whole model. Only sample a bounded
    # prefix of each gradient. Action masks instead locate the offending rows.
    failed = error.tensors.get("failed")
    valid = error.tensors.get("valid")
    indexes = error.tensors.get("minibatch_indexes")
    positions = None
    row_shape: tuple[int, ...] = ()
    if failed is not None:
        if valid is not None and failed.shape == valid.shape:
            failed = failed.any(axis=-1)
        row_shape = failed.shape
        if row_shape:
            positions = np.flatnonzero(np.asarray(failed))[:max_rows]
    # An optimizer refusal has no failed-row mask. Its minibatch indexes
    # still identify the rows of the logits, actions and observation.
    if positions is None and indexes is not None:
        row_shape = indexes.shape
    for name, value in error.tensors.items():
        if (
            row_shape
            and value.shape[: len(row_shape)] == row_shape
            and not name.startswith(("parameter/", "gradient/"))
        ):
            row_values = value.reshape(
                math.prod(row_shape), *value.shape[len(row_shape) :]
            )
            sample = (
                row_values[:max_rows]
                if positions is None
                else row_values[mx.array(positions)]
            )
        else:
            sample = value.reshape(-1)[:2048]
        retain(name, value, sample)

    if batch is not None:
        rows = np.arange(min(batch.actions.size, max_rows))
        if indexes is not None:
            mapped = np.asarray(indexes).reshape(-1)
            if np.any((mapped < 0) | (mapped >= batch.actions.size)):
                raise ValueError("invalid incident rollout indexes")
            rows = (
                mapped[positions]
                if positions is not None and failed.size == mapped.size
                else mapped[:max_rows]
            )
        elif positions is not None and failed.size == batch.actions.size:
            rows = positions
        shapes["batch_layout"] = {
            "trajectory_shape": list(batch.actions.shape),
            "selected_flat_indexes": rows.tolist(),
        }
        for name in ("actions", "logprobs", "values", "rewards", "dones"):
            selected = getattr(batch, name).reshape(-1)[rows]
            retain(f"batch/{name}", selected, selected)
        if len(rows):
            packed = type(batch.action_data).concatenate(
                [batch.action_data.row(int(row)) for row in rows]
            )
            for name, value in {
                "offsets": packed.offsets,
                "slots": packed.slots,
                "capacity": np.array(packed.capacity),
                "log_probabilities": packed.log_probabilities,
            }.items():
                retain(f"packed_actions/{name}", value, value)
        source = batch.recorded.rows
        selected_obs = source.select(batch.recorded.index[rows])
        for name in source.fields:
            size = (
                np.prod(source.space.shapes[name])
                * np.dtype(source.space.dtypes[name]).itemsize
            )
            if len(rows) * size > remaining:
                shapes[f"observation/{name}"] = {"omitted": "evidence byte budget"}
                continue
            sample = selected_obs.project(name)
            retain(f"observation/{name}", sample, sample)

    state = IncidentTensors(
        training_arrays(learner, ema),
        {
            "numpy_rng": json.dumps(rng.bit_generator.state),
            "python_rng": json.dumps(random.getstate()),
        },
    )
    state.arrays["rng.collector"] = mx.array(collector.numerics.snapshot_rng())
    state.arrays["rng.mlx"] = mx.random.state[0]
    return IncidentCapture(
        evidence,
        state,
        shapes,
        sum(v.nbytes for v in state.arrays.values()),
        False,
        ".safetensors",
    )
