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

from collections.abc import Iterator, Sequence
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
from manabot.belief.encoding import BeliefCheckpointBinding, BeliefEncodingSchema
from manabot.env.observation import ObservationSpace
from manabot.model.architecture import architecture_receipt
from manabot.model.world import validate_policy_input
from manabot.policy.checkpoint import (
    GRUS,
    ParameterArrays,
    PolicyCheckpoint,
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


_SEQUENCES = r"(^|\.)(projection|policy_head|value_head|mlp|detail_map|reference_map|member_map|record_output)"
_SEQUENTIAL = re.compile(_SEQUENCES + r"\.(\d+)\.")
_MLX_SEQUENTIAL = re.compile(_SEQUENCES + r"\.layers\.(\d+)\.")
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
    # Preserve rank-zero memory gates when requiring contiguous storage.
    out[name] = np.asarray(value, dtype=np.float32, order="C")


def mlx_parameters(canonical: ParameterArrays) -> dict[str, np.ndarray]:
    """Canonical arrays in MLX names; only renames and splits, preserving norms."""
    out: dict[str, np.ndarray] = {}

    for name, tensor in canonical.items():
        name = _SEQUENTIAL.sub(r"\1\2.layers.\3.", gru_name(name, _GRU))
        if name.endswith(
            (
                ".mha.qkv_weight",
                ".mha.qkv_bias",
                ".self_attn.qkv_weight",
                ".self_attn.qkv_bias",
            )
        ):
            stem, leaf = name.rsplit(".", 1)
            leaf = leaf.removeprefix("qkv_")
            for part, value in zip(_QKV, np.split(tensor, 3), strict=True):
                put_parameter(out, f"{stem}.{part}_proj.{leaf}", value)
            continue
        put_parameter(out, name, tensor)
    return out


def canonical_from_mlx(mlx_named: ParameterArrays) -> dict[str, np.ndarray]:
    """Inverse of `mlx_parameters`: MLX names and split QKV back to canonical."""
    out: dict[str, np.ndarray] = {}
    qkv: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    for name, tensor in mlx_named.items():
        canonical = _MLX_SEQUENTIAL.sub(r"\1\2.\3.", gru_name(name, _GRU_INVERSE))
        matched = re.match(
            r"^(.*\.(?:mha|self_attn))\.(query|key|value)_proj\.(weight|bias)$",
            canonical,
        )
        if matched is not None:
            stem, part, leaf = matched.groups()
            qkv.setdefault((stem, leaf), {})[part] = tensor
            continue
        put_parameter(out, canonical, tensor)
    for (stem, leaf), parts in qkv.items():
        if parts.keys() != set(_QKV):
            raise KeyError(f"incomplete MLX attention projections for {stem}.{leaf}")
        put_parameter(
            out,
            f"{stem}.qkv_{leaf}",
            np.concatenate([parts[part] for part in _QKV], axis=0),
        )
    return out


def assemble(
    checkpoint: PolicyCheckpoint,
    *,
    precision: Precision | None = None,
    bucketed_shapes: bool = False,
    live_tokens: bool = False,
) -> MLXActorCritic:
    """An MLX policy holding the checkpoint's weights; names and shapes must
    match the MLX model exactly. An omitted precision follows the model's
    width (`manabot.backends.mlx.arrays.default_precision`)."""
    parameters = mlx_parameters(checkpoint.parameters)
    policy = MLXActorCritic(
        checkpoint.dimensions(),
        checkpoint.tables(),
        checkpoint.model,
        checkpoint.space,
        precision=precision,
        bucketed_shapes=bucketed_shapes,
        live_tokens=live_tokens,
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
    if agent.compute_dtype != mx.float32:
        raise ValueError("close the forward precision scope before saving")
    parameters = canonical_from_mlx(flat_numpy(agent.parameters()))
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
        buffers=agent.buffers(),
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
    def belief_fit(self) -> ModuleType:
        return importlib.import_module("manabot.backends.mlx.belief_fit")

    @property
    def belief(self) -> ModuleType:
        return importlib.import_module("manabot.backends.mlx.belief")

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
        checkpoint = (
            path if isinstance(path, PolicyCheckpoint) else read_container(str(path))
        )
        policy = assemble(
            checkpoint,
            precision=self.precision,
            bucketed_shapes=bucketed_shapes,
            live_tokens=live_tokens,
        )
        # read_container has already validated the world. Keep it on the loaded
        # policy so simulation's common deck admission cannot silently bypass it.
        world_binding = json.loads(checkpoint.metadata["world_binding"])
        validate_policy_input(policy, world_binding)
        policy.world_binding = world_binding
        if policy.belief_count_buckets:
            policy.belief_binding = BeliefCheckpointBinding.from_checkpoint(
                checkpoint.record()
            )
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
