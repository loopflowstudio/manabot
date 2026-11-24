"""Load the shared checkpoint container into the MLX policy.

`mlx_parameters` maps canonical parameters and their gradients
(`manabot.policy.checkpoint`) to MLX names and layouts. The map is linear
and preserves norms. Three layouts differ:

- Attention: the canonical packed in-projection `qkv_weight` [3D,D] /
  `qkv_bias` [3D] splits, in that order, into `query_proj`, `key_proj` and
  `value_proj` of `mlx.nn.MultiHeadAttention`.
- GRU: gates are reset, update, new in both. `input_weight` is `Wx` and
  `hidden_weight` is `Wh`; `input_bias` and `hidden_bias` keep their names
  and remain independent optimizer parameters.
- Containers: `nn.Sequential` children `N` become `layers.N`.

Every canonical parameter maps to MLX arrays or loading fails, naming the key.
"""

from collections.abc import Collection, Iterator, Sequence
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import re
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

from manabot.backends.mlx import arrays, rl
from manabot.backends.mlx.arrays import flat_numpy, precision_dtype, to_mlx
from manabot.backends.mlx.policy import MLXActorCritic
from manabot.belief.encoding import BeliefEncodingSchema
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.model.architecture import architecture_receipt
from manabot.model.world import validate_policy_input
from manabot.policy.checkpoint import (
    GRUS,
    ParameterArrays,
    PolicyCheckpoint,
    UnsupportedPolicy,
    read_container,
    read_metadata,
    restore_value_semantic,
    save_policy_checkpoint,
)
from manabot.policy.protocol import Precision

if TYPE_CHECKING:
    from manabot.backends.mlx.learner import Learner
    from manabot.backends.mlx.runtime import Runtime
    import managym


def require_supported(model: ModelConfig, observation: ObservationConfig) -> None:
    """Refuse configurations the MLX backend does not yet implement, naming
    every one, rather than running them partially."""
    reasons = []
    if model.world is None or not model.catalog_records():
        reasons.append("a catalog world with record_vocabulary='catalog' is required")
    if observation.policy_input_version not in (9, 10, 11):
        reasons.append(f"policy_input_version={observation.policy_input_version}")
    if model.belief_count_buckets:
        reasons.append("belief inputs")
    if model.decision_memory != "none":
        reasons.append("decision memory")
    if model.recent_events:
        reasons.append("policy history input (recent_events)")
    if model.value_aggregation != "masked_mean":
        reasons.append(f"value_aggregation={model.value_aggregation!r}")
    if model.value_kind != "scalar":
        reasons.append(f"value_kind={model.value_kind!r}")
    if not model.attention_on:
        reasons.append("attention_on=False")
    if reasons:
        raise UnsupportedPolicy(
            "the framework backends do not support " + "; ".join(reasons)
        )


_SEQUENTIAL = re.compile(r"(^|\.)(projection|policy_head|value_head|mlp)\.(\d+)\.")
_MLX_SEQUENTIAL = re.compile(
    r"(^|\.)(projection|policy_head|value_head|mlp)\.layers\.(\d+)\."
)
_GRU = {
    "input_weight": "Wx",
    "hidden_weight": "Wh",
    "input_bias": "input_bias",
    "hidden_bias": "hidden_bias",
}
_GRU_INVERSE = {value: key for key, value in _GRU.items()}
_QKV = ("query", "key", "value")


def gru_name(name: str, leaves: dict[str, str]) -> str:
    """Both frameworks keep the GRU prefix and rename its four leaves."""
    for prefix in GRUS:
        if name.startswith(prefix + "."):
            leaf = name[len(prefix) + 1 :]
            if leaf not in leaves:
                raise KeyError(f"unmapped GRU tensor {name}")
            return f"{prefix}.{leaves[leaf]}"
    return name


def put_parameter(out: dict[str, np.ndarray], name: str, value: np.ndarray) -> None:
    if name in out:
        raise KeyError(f"two tensors map to {name}")
    out[name] = np.asarray(value, dtype=np.float32, order="C")


def mlx_names(canonical: str) -> tuple[str, ...]:
    """Native leaves of a canonical parameter, in packed tensor order."""
    name = _SEQUENTIAL.sub(r"\1\2.layers.\3.", gru_name(canonical, _GRU))
    if name.endswith((".mha.qkv_weight", ".mha.qkv_bias")):
        stem, leaf = name.rsplit(".", 1)
        return tuple(f"{stem}.{part}_proj.{leaf.removeprefix('qkv_')}" for part in _QKV)
    return (name,)


def mlx_parameters(canonical: ParameterArrays) -> dict[str, np.ndarray]:
    """Canonical arrays in MLX names; only renames and splits, preserving norms."""
    out: dict[str, np.ndarray] = {}
    for name, tensor in canonical.items():
        names = mlx_names(name)
        parts = (tensor,) if len(names) == 1 else np.split(tensor, len(names))
        for leaf, value in zip(names, parts, strict=True):
            put_parameter(out, leaf, value)
    return out


def canonical_groups(names: Collection[str]) -> dict[str, tuple[str, ...]]:
    """Group a native policy tree by canonical parameter, including packed QKV."""
    groups: dict[str, tuple[str, ...]] = {}
    for name in names:
        canonical = _MLX_SEQUENTIAL.sub(r"\1\2.\3.", gru_name(name, _GRU_INVERSE))
        matched = re.fullmatch(
            r"(.*\.mha)\.(query|key|value)_proj\.(weight|bias)", canonical
        )
        if matched:
            stem, _, leaf = matched.groups()
            canonical = f"{stem}.qkv_{leaf}"
        if canonical not in groups:
            leaves = mlx_names(canonical)
            if not all(leaf in names for leaf in leaves):
                raise KeyError(f"incomplete MLX projections for {canonical}")
            groups[canonical] = leaves
    return groups


def canonical_from_mlx(mlx_named: ParameterArrays) -> dict[str, np.ndarray]:
    """Inverse of `mlx_parameters`: MLX names and split QKV back to canonical."""
    out: dict[str, np.ndarray] = {}
    for name, leaves in canonical_groups(mlx_named.keys()).items():
        parts = [mlx_named[leaf] for leaf in leaves]
        put_parameter(out, name, parts[0] if len(parts) == 1 else np.concatenate(parts))
    return out


def assemble(
    checkpoint: PolicyCheckpoint,
    *,
    precision: Precision | None = None,
    bucketed_shapes: bool = False,
) -> MLXActorCritic:
    """An MLX policy holding the checkpoint's weights; names and shapes must
    match the MLX model exactly. An omitted precision follows the model's
    width (`manabot.backends.mlx.arrays.default_precision`)."""
    require_supported(checkpoint.model, checkpoint.space.config)
    parameters = mlx_parameters(checkpoint.parameters)
    policy = MLXActorCritic(
        checkpoint.dimensions(),
        checkpoint.tables(),
        checkpoint.model,
        checkpoint.space,
        precision=precision,
        bucketed_shapes=bucketed_shapes,
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


def policy_arrays(agent: MLXActorCritic) -> dict[str, np.ndarray]:
    if agent.compute_dtype != mx.float32:
        raise ValueError("close the forward precision scope before saving")
    return canonical_from_mlx(flat_numpy(agent.parameters()))


def policy_buffers(agent: MLXActorCritic) -> dict[str, np.ndarray]:
    return agent.buffers()


def save_checkpoint(
    agent: MLXActorCritic,
    obs_space: ObservationSpace,
    path: str | Path,
    *,
    player_configs: "Sequence[managym.PlayerConfig] | None",
    extra: dict[str, Any] | None = None,
    belief_schema: BeliefEncodingSchema | None = None,
) -> None:
    """Save an MLX policy into the shared portable safetensors container."""
    parameters = policy_arrays(agent)
    receipt = architecture_receipt(
        agent.config,
        obs_space,
        {name: (int(value.size), True) for name, value in parameters.items()},
    )
    save_policy_checkpoint(
        agent,
        obs_space,
        path,
        parameters=parameters,
        buffers=policy_buffers(agent),
        readings=agent.readings,
        architecture=receipt,
        player_configs=player_configs,
        extra=extra,
        belief_schema=belief_schema,
    )


class MLXBackend:
    """`manabot.policy.protocol.Backend` for MLX.

    An omitted precision lets each loaded policy follow its width
    (`manabot.backends.mlx.arrays.default_precision`).
    """

    name: Literal["mlx"] = "mlx"
    arrays = arrays
    rl = rl

    @staticmethod
    def configure_cpu_inference() -> None:
        mx.set_default_device(mx.cpu)

    @property
    def learner(self) -> ModuleType:
        return importlib.import_module("manabot.backends.mlx.learner")

    @property
    def runtime(self) -> "Runtime":
        from manabot.backends.mlx.runtime import Runtime

        return Runtime(self.precision)

    @property
    def state(self) -> ModuleType:
        return importlib.import_module("manabot.backends.mlx.state")

    save_policy = staticmethod(save_checkpoint)

    @property
    def collection(self) -> ModuleType:
        return importlib.import_module("manabot.backends.mlx.collection")

    @staticmethod
    @contextmanager
    def seeded_inference(seed: int, *, cpu_only: bool = False) -> Iterator[None]:
        stream = mx.cpu if cpu_only else mx.default_device()
        from manabot.backends.mlx.runtime import preserve_rng

        with preserve_rng("cpu" if cpu_only else "mps"), mx.stream(stream):
            mx.random.seed(seed)
            yield

    @property
    def rollout(self) -> ModuleType:
        return importlib.import_module("manabot.backends.mlx.rollout")

    def __init__(self, precision: Precision | None = None) -> None:
        if precision is not None:
            precision_dtype(precision)
        self.precision = precision

    def load_policy(
        self,
        path: str | Path | PolicyCheckpoint,
        *,
        bucketed_shapes: bool = False,
        live_tokens: bool = False,
    ) -> tuple[MLXActorCritic, ObservationSpace]:
        if live_tokens:
            raise ValueError("MLX does not support live-token inference")
        checkpoint = (
            path if isinstance(path, PolicyCheckpoint) else read_container(str(path))
        )
        policy = assemble(
            checkpoint, precision=self.precision, bucketed_shapes=bucketed_shapes
        )
        # read_container has already validated the world. Keep it on the loaded
        # policy so simulation's common deck admission cannot silently bypass it.
        world_binding = json.loads(checkpoint.metadata["world_binding"])
        validate_policy_input(policy, world_binding)
        policy.world_binding = world_binding
        training = json.loads(checkpoint.metadata.get("bc", "{}"))
        restore_value_semantic(policy, training.get("value_semantic"))
        return policy, checkpoint.space

    read_policy = staticmethod(read_container)

    metadata = staticmethod(read_metadata)

    @staticmethod
    def seed(seed: int, *, deterministic: bool | None = None) -> None:
        mx.random.seed(seed)

    def load(
        self,
        checkpoint: PolicyCheckpoint,
        *,
        device: str | None = None,
        precision: Precision | None = None,
        bucketed_shapes: bool = False,
        minibatch_rows: int = 1,
        execution_rows: int | None = None,
        bootstrap_rows: int | None = None,
    ) -> tuple[MLXActorCritic, "Learner"]:
        device = "mps" if device is None else device
        with mx.stream(arrays.mlx_device(device)):
            policy = assemble(
                checkpoint,
                precision=self.precision if precision is None else precision,
                bucketed_shapes=bucketed_shapes,
            ).to(device)
        return policy, self.learner.Learner(
            policy,
            device,
            minibatch_rows=minibatch_rows,
            execution_rows=execution_rows,
            bootstrap_rows=bootstrap_rows,
        )

    parameter_layout = staticmethod(mlx_parameters)
    gradient_layout = staticmethod(mlx_parameters)


BACKEND = MLXBackend()
