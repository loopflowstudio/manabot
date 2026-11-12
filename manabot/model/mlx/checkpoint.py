"""Convert PyTorch policy checkpoints to MLX and load them back.

A converted checkpoint is two files beside the original:

- `<stem>.mlx.safetensors`: one float32 array per MLX parameter, plus the
  integer lookup buffers under `buffer/<torch name>`;
- `<stem>.mlx.json`: the checkpoint's `hypers`, `world_binding` and
  `architecture`, so loading validates the same world binding as
  `manabot.model.checkpoint.load_checkpoint`.

`mlx_parameters` is the one name and layout mapping (see `torch_to_mlx`). Every
torch tensor maps to exactly one MLX array or a lookup buffer; anything else
fails, naming the key. Conversion needs PyTorch only to read the `.pt` file;
`load_mlx_policy` does not import it.
"""

from collections.abc import Mapping
import json
from pathlib import Path
import re
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

from manabot.env.observation import (
    FACT_KINDS,
    PERMANENT_DEFINITION_INDEX,
    PERMANENT_LINK_INDEXES,
    ObservationSpace,
)
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.model.mlx.model import Dimensions, MLXActorCritic
from manabot.model.mlx.prepare import Layout, Tables
from manabot.model.world import validate_checkpoint_world

# GRUs of the policy: PyTorch `nn.GRU` modules, MLX `PackedGRU`.
GRUS = (
    "semantic_cards.encoder",
    "facts.program_sequence",
    "shared_effects.reducer.gru",
)
# PyTorch nn.Sequential containers; MLX keeps their children under `layers`.
_SEQUENTIAL = re.compile(r"(^|\.)(projection|policy_head|value_head|mlp)\.(\d+)\.")


class UnsupportedForMLX(ValueError):
    """The checkpoint enables something the MLX port does not implement."""


def require_supported(model: ModelConfig, observation: ObservationConfig) -> None:
    """Refuse configurations the port does not implement, naming every one,
    rather than running them partially."""
    reasons = []
    if model.world is None or not model.catalog_records():
        reasons.append("a catalog world with record_vocabulary='catalog' is required")
    if observation.policy_input_version != 9:
        reasons.append(f"policy_input_version={observation.policy_input_version}")
    if model.belief_count_buckets:
        reasons.append("belief inputs")
    if model.recent_events:
        reasons.append("policy history input (recent_events)")
    if model.token_stream != "tables":
        reasons.append(f"token_stream={model.token_stream!r}")
    if model.value_aggregation != "masked_mean":
        reasons.append(f"value_aggregation={model.value_aggregation!r}")
    if model.value_kind != "scalar":
        reasons.append(f"value_kind={model.value_kind!r}")
    if model.policy_film:
        reasons.append("policy_film")
    if not model.attention_on:
        reasons.append("attention_on=False")
    if reasons:
        raise UnsupportedForMLX("the MLX port does not support " + "; ".join(reasons))


def layout(space: ObservationSpace) -> Layout:
    encoder = space.encoder
    joins = (PERMANENT_DEFINITION_INDEX, *PERMANENT_LINK_INDEXES)
    return Layout(
        cards_per_player=encoder.cards_per_player,
        perms_per_player=encoder.perms_per_player,
        max_actions=encoder.max_actions,
        numeric_permanent_columns=tuple(
            c for c in range(encoder.permanent_dim) if c not in joins
        ),
        link_columns=PERMANENT_LINK_INDEXES,
    )


def dimensions(
    model: ModelConfig, space: ObservationSpace, state: Mapping[str, Any]
) -> Dimensions:
    encoder = space.encoder
    return Dimensions(
        hidden=model.hidden_dim,
        heads=model.num_attention_heads,
        layers=model.attention_layers,
        feedforward=model.attention_feedforward_dim
        or model.num_attention_heads * model.hidden_dim,
        player=encoder.player_dim,
        card=encoder.card_dim,
        permanent=len(layout(space).numeric_permanent_columns),
        action=encoder.action_dim,
        focus=encoder.max_focus_objects,
        relation=encoder.relation_dim,
        fact_kinds=FACT_KINDS,
        token_rows=int(state["semantic_cards.token_embedding.weight"].shape[0]),
        name_rows=int(state["name_candidates.embedding.weight"].shape[0]),
        links=len(PERMANENT_LINK_INDEXES),
    )


def torch_to_mlx(name: str, tensor: np.ndarray, hidden: int) -> dict[str, np.ndarray]:
    """MLX arrays for one PyTorch parameter.

    - `nn.MultiheadAttention.in_proj_{weight,bias}` packs query, key and value
      ([3D, D] / [3D]); they split, in that order, into `query_proj`,
      `key_proj` and `value_proj`.
    - `nn.GRU` gates are reset, update, new in both frameworks. `weight_ih_l0`
      is `Wx` and `weight_hh_l0` is `Wh`. MLX keeps one input bias `b` and a
      hidden bias `bhn` for the new gate only: `b = bias_ih + [bias_hh_r,
      bias_hh_z, 0]`, exact because those gates add both biases before the
      sigmoid; `bhn = bias_hh_n`.
    - `nn.Sequential` children `N` become `layers.N`.

    `bias_ih_l0` maps to nothing here; `mlx_parameters` folds it into `b`.
    """
    for prefix in GRUS:
        if name.startswith(prefix + "."):
            leaf = name[len(prefix) + 1 :]
            if leaf == "weight_ih_l0":
                return {f"{prefix}.Wx": tensor}
            if leaf == "weight_hh_l0":
                return {f"{prefix}.Wh": tensor}
            if leaf == "bias_hh_l0":
                return {f"{prefix}.bhn": tensor[2 * hidden :]}
            if leaf == "bias_ih_l0":
                return {}
            raise KeyError(f"unmapped GRU tensor {name}")
    name = _SEQUENTIAL.sub(
        lambda m: f"{m.group(1)}{m.group(2)}.layers.{m.group(3)}.", name
    )
    for leaf in ("weight", "bias"):
        packed = f".mha.in_proj_{leaf}"
        if name.endswith(packed):
            stem = name[: -len(packed)] + ".mha"
            query, key, value = np.split(tensor, 3, axis=0)
            return {
                f"{stem}.query_proj.{leaf}": query,
                f"{stem}.key_proj.{leaf}": key,
                f"{stem}.value_proj.{leaf}": value,
            }
    return {name: tensor}


def mlx_parameters(
    state: Mapping[str, np.ndarray], hidden: int
) -> dict[str, np.ndarray]:
    """MLX float32 parameters of PyTorch parameters (`torch_to_mlx`, plus each
    GRU's folded input bias). Linear, so it maps parameter deltas too."""
    parameters: dict[str, np.ndarray] = {}
    for name, tensor in state.items():
        for key, value in torch_to_mlx(name, np.asarray(tensor), hidden).items():
            if key in parameters:
                raise KeyError(f"two torch tensors map to {key}")
            parameters[key] = np.ascontiguousarray(value, dtype=np.float32)
    for prefix in GRUS:
        if f"{prefix}.bias_ih_l0" not in state:
            continue
        folded = np.array(state[f"{prefix}.bias_ih_l0"], dtype=np.float32)
        folded[: 2 * hidden] += np.asarray(state[f"{prefix}.bias_hh_l0"])[: 2 * hidden]
        parameters[f"{prefix}.b"] = folded
    return parameters


def convert_state(
    state: Mapping[str, np.ndarray], hidden: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Split a PyTorch state dict into MLX parameters and lookup buffers."""
    buffer_names = set(Tables.BUFFERS.values())
    missing = sorted(buffer_names - set(state))
    if missing:
        raise KeyError(f"checkpoint lacks lookup buffers {missing}")
    buffers = {name: np.asarray(state[name]) for name in buffer_names}
    parameters = mlx_parameters(
        {k: v for k, v in state.items() if k not in buffer_names}, hidden
    )
    return parameters, buffers


def build(
    model: ModelConfig,
    observation: ObservationConfig,
    state: Mapping[str, np.ndarray],
) -> tuple[MLXActorCritic, ObservationSpace]:
    """An MLX policy holding `state` (a PyTorch state dict as numpy arrays)."""
    require_supported(model, observation)
    space = ObservationSpace(observation)
    parameters, buffers = convert_state(state, model.hidden_dim)
    return _assemble(model, space, parameters, buffers), space


def _assemble(
    model: ModelConfig,
    space: ObservationSpace,
    parameters: Mapping[str, np.ndarray],
    buffers: Mapping[str, np.ndarray],
) -> MLXActorCritic:
    state = {**parameters, **buffers}
    tables = Tables.from_buffers(buffers, layout(space))
    dims = dimensions(model, space, state)
    policy = MLXActorCritic(dims, tables)
    expected = dict(tree_flatten(policy.parameters()))
    unmapped = sorted(set(parameters) ^ set(expected))
    if unmapped:
        raise KeyError(f"torch and MLX parameters differ: {unmapped}")
    for key, value in parameters.items():
        if tuple(expected[key].shape) != value.shape:
            raise ValueError(f"{key}: torch {value.shape}, MLX {expected[key].shape}")
    policy.update(tree_unflatten([(k, mx.array(v)) for k, v in parameters.items()]))
    mx.eval(policy.parameters())
    return policy


# -- files ---------------------------------------------------------------------------


def converted_paths(checkpoint: Path) -> tuple[Path, Path]:
    stem = checkpoint.with_suffix("")
    return stem.with_suffix(".mlx.safetensors"), stem.with_suffix(".mlx.json")


def convert(checkpoint: Path) -> tuple[Path, Path, int]:
    """Write the MLX files for a PyTorch checkpoint; return them and the
    number of torch tensors read."""
    import torch

    from manabot.model.checkpoint import load_checkpoint

    agent, space = load_checkpoint(str(checkpoint))
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {k: v.detach().numpy() for k, v in agent.state_dict().items()}
    require_supported(agent.config, space.encoder.config)
    parameters, buffers = convert_state(state, agent.config.hidden_dim)
    _assemble(agent.config, space, parameters, buffers)  # validates names and shapes
    weights, sidecar = converted_paths(checkpoint)
    arrays = {k: mx.array(v) for k, v in parameters.items()}
    arrays.update({f"buffer/{k}": mx.array(v) for k, v in buffers.items()})
    mx.save_safetensors(str(weights), arrays)
    sidecar.write_text(
        json.dumps(
            {
                "hypers": raw["hypers"],
                "world_binding": raw["world_binding"],
                "architecture": raw.get("architecture"),
                "source": str(checkpoint),
            },
            indent=2,
            default=str,
        )
    )
    return weights, sidecar, len(state)


def load_mlx_policy(path: str | Path) -> tuple[MLXActorCritic, ObservationSpace]:
    """Load a converted checkpoint, validating its world binding."""
    weights = Path(path)
    sidecar = weights.with_name(weights.name.replace(".safetensors", ".json"))
    meta = json.loads(sidecar.read_text())
    hypers = meta["hypers"]
    space = ObservationSpace(
        ObservationConfig.from_checkpoint(hypers["observation_hypers"])
    )
    validate_checkpoint_world(meta, space)
    model = ModelConfig.from_checkpoint(hypers["agent_hypers"])
    require_supported(model, space.encoder.config)
    arrays = mx.load(str(weights))
    assert isinstance(arrays, dict)
    parameters = {
        k: np.array(v) for k, v in arrays.items() if not k.startswith("buffer/")
    }
    buffers = {k[7:]: np.array(v) for k, v in arrays.items() if k.startswith("buffer/")}
    return _assemble(model, space, parameters, buffers), space
