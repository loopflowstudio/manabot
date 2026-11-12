"""Load the shared checkpoint container into the MLX policy.

`mlx_parameters` is the one name and layout mapping from canonical parameters
(`manabot.policy.checkpoint`) to MLX's, and `mlx_gradients` the matching map
for gradients. Three layouts differ:

- Attention: the canonical packed in-projection `qkv_weight` [3D,D] /
  `qkv_bias` [3D] splits, in that order, into `query_proj`, `key_proj` and
  `value_proj` of `mlx.nn.MultiHeadAttention`.
- GRU: gates are reset, update, new in both. `input_weight` is `Wx` and
  `hidden_weight` is `Wh`. MLX keeps one input bias `b` and a hidden bias
  `bhn` for the new gate only: `b = input_bias + [hidden_bias_r,
  hidden_bias_z, 0]`, exact because those gates add both biases before the
  sigmoid; `bhn = hidden_bias_n`. The gradient of `b` is therefore the
  gradient of `input_bias` (the reset/update parts of `hidden_bias` have the
  same one), not a sum.
- Containers: `nn.Sequential` children `N` become `layers.N`.

Every canonical parameter maps to MLX arrays or loading fails, naming the key.
"""

from collections.abc import Callable, Mapping
from pathlib import Path
import re

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

from manabot.backends.mlx.arrays import to_mlx
from manabot.backends.mlx.learner import MLXLearner
from manabot.backends.mlx.policy import MLXActorCritic
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.policy.checkpoint import (
    GRUS,
    PolicyCheckpoint,
    from_torch_state,
    read_container,
)

_SEQUENTIAL = re.compile(r"(^|\.)(projection|policy_head|value_head|mlp)\.(\d+)\.")
_GRU = {"input_weight": "Wx", "hidden_weight": "Wh"}

Fold = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _layout(canonical: Mapping[str, np.ndarray], fold: Fold) -> dict[str, np.ndarray]:
    """Canonical arrays in MLX names; `fold(input_bias, hidden_bias)` gives `b`."""
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
            elif leaf == "hidden_bias":
                put(f"{prefix}.bhn", tensor[2 * (tensor.shape[0] // 3) :])
                put(
                    f"{prefix}.b",
                    fold(np.asarray(canonical[f"{prefix}.input_bias"]), tensor),
                )
            elif leaf != "input_bias":
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


def _fold_biases(input_bias: np.ndarray, hidden_bias: np.ndarray) -> np.ndarray:
    folded = np.array(input_bias, dtype=np.float32)
    gates = 2 * (hidden_bias.shape[0] // 3)
    folded[:gates] += hidden_bias[:gates]
    return folded


def mlx_parameters(canonical: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """MLX float32 parameters of canonical ones. Linear, so it maps deltas too."""
    return _layout(canonical, _fold_biases)


def mlx_gradients(canonical: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Gradients of canonical parameters as gradients of MLX's."""
    return _layout(canonical, lambda input_bias, _: input_bias)


def assemble(checkpoint: PolicyCheckpoint) -> MLXActorCritic:
    """An MLX policy holding the checkpoint's weights; names and shapes must
    match the MLX model exactly."""
    parameters = mlx_parameters(checkpoint.parameters)
    policy = MLXActorCritic(checkpoint.dimensions(), checkpoint.tables())
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


def build(
    model: ModelConfig,
    observation: ObservationConfig,
    state: Mapping[str, np.ndarray],
) -> tuple[MLXActorCritic, ObservationSpace]:
    """An MLX policy holding `state` (a PyTorch state dict as NumPy arrays)."""
    checkpoint = from_torch_state(model, observation, state)
    return assemble(checkpoint), checkpoint.space


def load_mlx_policy(path: str | Path) -> tuple[MLXActorCritic, ObservationSpace]:
    """Load a checkpoint container, validating its world binding."""
    checkpoint = read_container(path)
    return assemble(checkpoint), checkpoint.space


class _MLX:
    """`manabot.policy.protocol.Backend` for MLX."""

    name = "mlx"

    def load(self, checkpoint: PolicyCheckpoint) -> tuple[MLXActorCritic, MLXLearner]:
        policy = assemble(checkpoint)
        return policy, MLXLearner(policy)

    def parameter_layout(
        self, canonical: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        return mlx_parameters(canonical)

    def gradient_layout(
        self, canonical: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        return mlx_gradients(canonical)


BACKEND = _MLX()
