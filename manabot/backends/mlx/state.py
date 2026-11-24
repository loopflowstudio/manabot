"""MLX continuation and numerical evidence for tracked self-play.

Etude commits the host continuation, backend tensors and state.json together.
Its state.json owns the counts, model settings and column readings. Restore checks the
complete array layout before changing either policy or Adam.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
from typing import TYPE_CHECKING, BinaryIO

from etude.train.state import read_state
import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

from manabot.backends.mlx.arrays import all_finite
from manabot.backends.mlx.learner import Learner
from manabot.backends.mlx.policy import MLXActorCritic
from manabot.infra.identity import canonical_sha256
from manabot.policy.native_rows import ObservationSelection
from manabot.training.rollout.numerics import CollectionSession
from manabot.training.state.backend import IncidentCapture, NumericalIncident
from manabot.training.state.continuation import (
    HOST_FILE,
    HostContinuation,
    StagedContinuation,
    host_sha256,
    read_host,
)

if TYPE_CHECKING:
    from etude.train.tracking import TrainingRun

    from manabot.training.loops.self_play import Counts
    from manabot.training.rollout.collection import Collector, RolloutBatch


def framework() -> dict[str, str | None]:
    return {"mlx": mx.__version__}


def training_arrays(
    learner: Learner, ema: MLXActorCritic | None
) -> dict[str, mx.array]:
    trees = {"learner": learner.agent.parameters(), "adam": learner.optimizer.state}
    if ema is not None:
        trees["ema"] = ema.parameters()
    return dict(tree_flatten(trees))


def restore(
    run: TrainingRun, learner: Learner, ema: MLXActorCritic | None, counts: Counts
) -> StagedContinuation:
    """Admit the committed snapshot and supply the writer for Etude's staged folder."""
    filename = "training.safetensors"
    agent = learner.agent
    # These lookup tables are reconstructed by policy creation. A table with the
    # same shape but a changed vocabulary would silently reinterpret the weights.
    binding = canonical_sha256(
        {
            "observation": agent.observation_space.config.model_dump(mode="json"),
            "precision": agent.precision,
            "buffers": {
                name: [
                    str(value.dtype),
                    value.shape,
                    hashlib.sha256(value.tobytes()).hexdigest(),
                ]
                for name, value in agent.buffers().items()
            },
        }
    )

    def metadata() -> dict[str, str]:
        return {
            "backend": "mlx",
            "binding": binding,
            "counts": canonical_sha256(counts.model_dump(mode="json")),
        }

    host = None
    sampling: dict[str, mx.array] = {}
    found = read_state(run.out / "state")
    if found is not None:
        path = found.path / filename
        if found.record.files or not path.is_file():
            raise ValueError("expected an MLX callback save for tracked self-play")
        if (
            counts.updates != found.record.at["updates"]
            or counts.model is None
            or counts.readings is None
        ):
            raise ValueError("incompatible MLX saved counts or input contract")
        arrays, saved = mx.load(path, return_metadata=True)
        header = metadata() | {"schema": "1"}
        payload = None
        if saved.get("schema") == "2":
            payload = read_host(found.path)
            if payload is None:
                raise ValueError(f"missing MLX {HOST_FILE}")
            for key in ("rng.collector", "rng.mlx"):
                value = arrays.pop(key, None)
                if value is None or value.shape != (2,) or value.dtype != mx.uint32:
                    raise ValueError(
                        f"incompatible MLX snapshot RNG shape or dtype for {key}"
                    )
                sampling[key] = value
            header |= {
                "schema": "2",
                "continuation_sha256": host_sha256(payload),
                "rng_sha256": sampling_digest(sampling),
            }
        elif (found.path / HOST_FILE).exists():
            raise ValueError("incompatible MLX snapshot schema with host continuation")
        # The header binds the host bytes, so a damaged file fails here by name
        # rather than as a JSON decoding error.
        verify_header(saved, header)
        if payload is not None:
            host = HostContinuation.model_validate_json(payload)
        expected = training_arrays(learner, ema)
        # Adam creates its two moments lazily on the first accepted step.
        if counts.optimizer_updates:
            for name, value in tree_flatten(agent.trainable_parameters()):
                expected[f"adam.{name}.m"] = value
                expected[f"adam.{name}.v"] = value
        if arrays.keys() != expected.keys():
            raise ValueError("incompatible MLX snapshot parameter or Adam keys")
        for name, value in arrays.items():
            if (
                value.shape != expected[name].shape
                or value.dtype != expected[name].dtype
            ):
                raise ValueError(f"incompatible MLX snapshot shape or dtype for {name}")
        if int(arrays["adam.step"]) != counts.optimizer_updates:
            raise ValueError("MLX Adam step differs from the saved optimizer count")
        if not all_finite(arrays.values()):
            raise ValueError("nonfinite MLX training snapshot")
        trees = tree_unflatten(arrays)
        agent.update(trees["learner"])
        agent.clear_cache()
        if ema is not None:
            ema.update(trees["ema"])
            ema.clear_cache()
        learner.optimizer.state = trees["adam"]

    def restore_sampling(session: CollectionSession) -> None:
        if sampling:
            session.restore_rng(sampling["rng.collector"])
            high, low = sampling["rng.mlx"].tolist()
            mx.random.seed((high << 32) | low)

    def export(folder: Path, host_sha256: str, session: CollectionSession) -> None:
        keys = {
            "rng.collector": mx.array(session.snapshot_rng()),
            "rng.mlx": mx.random.state[0],
        }
        header = metadata() | {
            "schema": "2",
            "continuation_sha256": host_sha256,
            "rng_sha256": sampling_digest(keys),
        }
        mx.save_safetensors(
            folder / filename, training_arrays(learner, ema) | keys, header
        )

    return StagedContinuation(host, restore_sampling, export)


def verify_header(saved: dict[str, str], expected: dict[str, str]) -> None:
    if saved != expected:
        differing = sorted(
            key
            for key in saved.keys() | expected.keys()
            if saved.get(key) != expected.get(key)
        )
        raise ValueError(
            "incompatible MLX snapshot: saved header differs in " + ", ".join(differing)
        )


def sampling_digest(keys: dict[str, mx.array]) -> str:
    """Bind the two small RNG payloads, including same-shape bit corruption."""
    return hashlib.sha256(
        b"".join(np.asarray(keys[k]).tobytes() for k in sorted(keys))
    ).hexdigest()


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
                **packed.fields,
            }.items():
                retain(f"packed_actions/{name}", value, value)
        source = batch.recorded.rows
        if isinstance(source, ObservationSelection):
            selected_obs = source.select(batch.recorded.index[rows])
        for name in source:
            if isinstance(source, ObservationSelection):
                size = (
                    np.prod(source.space.shapes[name])
                    * np.dtype(source.space.dtypes[name]).itemsize
                )
            else:
                size = source[name][0].nbytes
            if len(rows) * size > remaining:
                shapes[f"observation/{name}"] = {"omitted": "evidence byte budget"}
                continue
            sample = (
                selected_obs[name]
                if isinstance(source, ObservationSelection)
                else source[name][batch.recorded.index[rows]]
            )
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
