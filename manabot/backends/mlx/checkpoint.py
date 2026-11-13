"""Load the shared checkpoint container into the MLX policy.

`mlx_parameters` is the one name and layout mapping from canonical parameters
(`manabot.policy.checkpoint`) to MLX's, and `mlx_gradients` the matching map
for gradients. Three layouts differ:

- Attention: the canonical packed in-projection `qkv_weight` [3D,D] /
  `qkv_bias` [3D] splits, in that order, into `query_proj`, `key_proj` and
  `value_proj` of `mlx.nn.MultiHeadAttention`.
- GRU: gates are reset, update, new in both. `input_weight` is `Wx` and
  `hidden_weight` is `Wh`; `input_bias` and `hidden_bias` keep their names
  and remain independent optimizer parameters.
- Containers: `nn.Sequential` children `N` become `layers.N`.

Every canonical parameter maps to MLX arrays or loading fails, naming the key.
"""

import json
from pathlib import Path
import re
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

from manabot.backends.mlx.arrays import to_mlx
from manabot.backends.mlx.learner import Learner
from manabot.backends.mlx.policy import MLXActorCritic
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.policy.checkpoint import (
    GRUS,
    ParameterArrays,
    PolicyCheckpoint,
    UnsupportedPolicy,
    read_container,
    read_metadata,
)


def require_supported(model: ModelConfig, observation: ObservationConfig) -> None:
    """Refuse configurations the host preparation and backends do not
    implement, naming every one, rather than running them partially."""
    reasons = []
    if model.world is None or not model.catalog_records():
        reasons.append("a catalog world with record_vocabulary='catalog' is required")
    if observation.policy_input_version != 9:
        reasons.append(f"policy_input_version={observation.policy_input_version}")
    if model.belief_count_buckets:
        reasons.append("belief inputs")
    if model.recent_events:
        reasons.append("policy history input (recent_events)")
    if model.value_aggregation != "masked_mean":
        reasons.append(f"value_aggregation={model.value_aggregation!r}")
    if model.value_kind != "scalar":
        reasons.append(f"value_kind={model.value_kind!r}")
    if model.policy_film:
        reasons.append("policy_film")
    if not model.attention_on:
        reasons.append("attention_on=False")
    if reasons:
        raise UnsupportedPolicy(
            "the framework backends do not support " + "; ".join(reasons)
        )


_SEQUENTIAL = re.compile(r"(^|\.)(projection|policy_head|value_head|mlp)\.(\d+)\.")
_GRU = {
    "input_weight": "Wx",
    "hidden_weight": "Wh",
    "input_bias": "input_bias",
    "hidden_bias": "hidden_bias",
}


def mlx_parameters(canonical: ParameterArrays) -> dict[str, np.ndarray]:
    """Canonical arrays in MLX names; only renames and splits, preserving norms."""
    out: dict[str, np.ndarray] = {}

    def put(key: str, value: np.ndarray) -> None:
        if key in out:
            raise KeyError(f"two canonical tensors map to {key}")
        out[key] = np.ascontiguousarray(value, dtype=np.float32)

    for name, tensor in canonical.items():
        tensor = np.asarray(tensor)
        prefix = next((p for p in GRUS if name.startswith(p + ".")), None)
        if prefix is not None:
            leaf = name[len(prefix) + 1 :]
            if leaf in _GRU:
                put(f"{prefix}.{_GRU[leaf]}", tensor)
            else:
                raise KeyError(f"unmapped GRU tensor {name}")
            continue
        name = _SEQUENTIAL.sub(
            lambda m: f"{m.group(1)}{m.group(2)}.layers.{m.group(3)}.", name
        )
        if name.endswith((".mha.qkv_weight", ".mha.qkv_bias")):
            stem, leaf = name.rsplit(".", 1)
            leaf = leaf.removeprefix("qkv_")
            for part, value in zip(("query", "key", "value"), np.split(tensor, 3)):
                put(f"{stem}.{part}_proj.{leaf}", value)
            continue
        put(name, tensor)
    return out


def mlx_gradients(canonical: ParameterArrays) -> dict[str, np.ndarray]:
    """Gradients of canonical parameters as gradients of MLX's."""
    return mlx_parameters(canonical)


def assemble(checkpoint: PolicyCheckpoint) -> MLXActorCritic:
    """An MLX policy holding the checkpoint's weights; names and shapes must
    match the MLX model exactly."""
    require_supported(checkpoint.model, checkpoint.space.encoder.config)
    parameters = mlx_parameters(checkpoint.parameters)
    policy = MLXActorCritic(
        checkpoint.dimensions(), checkpoint.tables(), checkpoint.model, checkpoint.space
    )
    expected = dict(tree_flatten(policy.parameters()))
    unmapped = sorted(set(parameters) ^ set(expected))
    if unmapped:
        raise KeyError(f"canonical and MLX parameters differ: {unmapped}")
    for key, value in parameters.items():
        if tuple(expected[key].shape) != value.shape:
            raise ValueError(
                f"{key}: canonical {value.shape}, MLX {expected[key].shape}"
            )
    policy.update(tree_unflatten([(k, to_mlx(v)) for k, v in parameters.items()]))
    mx.eval(policy.parameters())
    return policy


def load_mlx_policy(path: str | Path) -> tuple[MLXActorCritic, ObservationSpace]:
    """Load a checkpoint container, validating its world binding."""
    checkpoint = read_container(path)
    policy = assemble(checkpoint)
    # read_container has already validated the world. Keep it on the loaded
    # policy so simulation's common deck admission cannot silently bypass it.
    policy.world_binding = json.loads(checkpoint.metadata["world_binding"])
    record = json.loads(checkpoint.metadata.get("bc", "{}"))
    semantic = record.get("value_semantic")
    if semantic not in (None, "signed_outcome", "win_logit"):
        raise ValueError("unsupported checkpoint value semantic")
    policy.value_semantic = semantic
    return policy, checkpoint.space


class _MLX:
    """`manabot.policy.protocol.Backend` for MLX."""

    name = "mlx"

    @staticmethod
    def load_policy(
        path: str, *, bucketed_shapes: bool = False, live_tokens: bool = False
    ) -> tuple[MLXActorCritic, ObservationSpace]:
        if bucketed_shapes or live_tokens:
            raise ValueError("MLX does not support bucketed or live-token inference")
        return load_mlx_policy(path)

    @staticmethod
    def metadata(path: str) -> dict[str, Any]:
        return read_metadata(path)

    @staticmethod
    def seed(seed: int) -> None:
        mx.random.seed(seed)

    def load(self, checkpoint: PolicyCheckpoint) -> tuple[MLXActorCritic, Learner]:
        policy = assemble(checkpoint)
        return policy, Learner(policy, minibatch_rows=1)

    def parameter_layout(self, canonical: ParameterArrays) -> dict[str, np.ndarray]:
        return mlx_parameters(canonical)

    def gradient_layout(self, canonical: ParameterArrays) -> dict[str, np.ndarray]:
        return mlx_gradients(canonical)


BACKEND = _MLX()
