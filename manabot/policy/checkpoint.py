"""The policy checkpoint container every framework backend reads.

A container is one [safetensors](https://huggingface.co/docs/safetensors) file:
float32 parameters under canonical, framework-neutral names, the integer
lookup buffers under `buffer/<name>`, and string metadata holding the
checkpoint's `hypers`, `world_binding`, `architecture` and `source` record as
JSON (`_METADATA_KEYS`) plus the PyTorch file it came from (`converted_from`).
`manabot.backends.convert` writes one from a PyTorch `.pt` checkpoint, which
stays readable as before; `read_container` validates the same world
binding as `manabot.backends.torch.checkpoint.load_checkpoint`, so a refusal names the
revision that wrote the original.

Canonical names are the PyTorch state-dict names with the framework's own
idioms renamed (`canonical_name`): a GRU holds `input_weight` [3H,I],
`hidden_weight` [3H,H], `input_bias` [3H] and `hidden_bias` [3H], gates reset,
update, new; attention keeps its packed in-projection as `qkv_weight` [3D,D]
and `qkv_bias` [3D], query then key then value. A backend maps these to its own
layout (`manabot.backends.<name>.checkpoint`).

This module reads and writes the file format with NumPy; it imports no array
framework.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import struct
from typing import Any

import numpy as np

from manabot.env.observation import (
    FACT_KINDS,
    PERMANENT_DEFINITION_INDEX,
    PERMANENT_LINK_INDEXES,
    ObservationSpace,
)
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.model.world import validate_checkpoint_world
from manabot.policy.prepare import Layout, Tables

type ParameterArrays = Mapping[str, np.ndarray]

# GRUs of the policy (PyTorch `nn.GRU` modules) by parameter prefix.
GRUS = (
    "semantic_cards.encoder",
    "facts.program_sequence",
    "shared_effects.reducer.gru",
)
_GRU_LEAVES = {
    "weight_ih_l0": "input_weight",
    "weight_hh_l0": "hidden_weight",
    "bias_ih_l0": "input_bias",
    "bias_hh_l0": "hidden_bias",
}
_ATTENTION_LEAVES = {
    ".mha.in_proj_weight": ".mha.qkv_weight",
    ".mha.in_proj_bias": ".mha.qkv_bias",
}
# Checkpoint fields carried, JSON-encoded, in the container metadata.
_METADATA_KEYS = ("hypers", "world_binding", "architecture", "source")
_OPTIONAL_METADATA_KEYS = (
    "bc",
    "global_step",
    "belief_schema_identity",
    "belief_content_manifest_identity",
)
_BUFFER = "buffer/"


class NotAContainer(ValueError):
    """The file is not a policy checkpoint container (for example a PyTorch
    `.pt` file, or a conversion that predates the container metadata)."""


class UnsupportedPolicy(ValueError):
    """The checkpoint enables something the framework backends do not implement."""


# -- architecture --------------------------------------------------------------------


@dataclass(frozen=True)
class Dimensions:
    """Input widths and architecture of one checkpoint."""

    hidden: int
    heads: int
    layers: int
    feedforward: int
    player: int
    card: int
    permanent: int
    action: int
    focus: int
    relation: int
    fact_kinds: int
    token_rows: int
    name_rows: int
    links: int


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


# -- canonical names -----------------------------------------------------------------


def canonical_name(torch_name: str) -> str:
    """The canonical name of one PyTorch parameter (or of its gradient)."""
    for prefix in GRUS:
        if torch_name.startswith(prefix + "."):
            leaf = torch_name[len(prefix) + 1 :]
            if leaf not in _GRU_LEAVES:
                raise KeyError(f"unmapped GRU tensor {torch_name}")
            return f"{prefix}.{_GRU_LEAVES[leaf]}"
    for packed, canonical in _ATTENTION_LEAVES.items():
        if torch_name.endswith(packed):
            return torch_name[: -len(packed)] + canonical
    return torch_name


def torch_name(canonical: str) -> str:
    """The PyTorch name of one canonical parameter: `canonical_name`'s inverse."""
    for prefix in GRUS:
        if canonical.startswith(prefix + "."):
            leaf = canonical[len(prefix) + 1 :]
            for torch_leaf, canonical_leaf in _GRU_LEAVES.items():
                if leaf == canonical_leaf:
                    return f"{prefix}.{torch_leaf}"
            raise KeyError(f"unmapped GRU tensor {canonical}")
    for packed, renamed in _ATTENTION_LEAVES.items():
        if canonical.endswith(renamed):
            return canonical[: -len(renamed)] + packed
    return canonical


def canonical_parameters(
    torch_named: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """PyTorch parameters (or gradients, or deltas) under canonical names.
    Only names change, so the map is linear."""
    return {
        canonical_name(name): np.asarray(value) for name, value in torch_named.items()
    }


@dataclass(frozen=True)
class PolicyCheckpoint:
    """A policy's configuration, canonical float32 parameters and lookup
    buffers, as every backend loads them."""

    model: ModelConfig
    space: ObservationSpace
    parameters: ParameterArrays
    buffers: Mapping[str, np.ndarray]
    metadata: Mapping[str, str]

    def tables(self) -> Tables:
        return Tables.from_buffers(self.buffers, layout(self.space))

    def dimensions(self) -> Dimensions:
        encoder = self.space.encoder
        model = self.model
        return Dimensions(
            hidden=model.hidden_dim,
            heads=model.num_attention_heads,
            layers=model.attention_layers,
            feedforward=model.attention_feedforward_dim
            or model.num_attention_heads * model.hidden_dim,
            player=encoder.player_dim,
            card=encoder.card_dim,
            permanent=len(layout(self.space).numeric_permanent_columns),
            action=encoder.action_dim,
            focus=encoder.max_focus_objects,
            relation=encoder.relation_dim,
            fact_kinds=FACT_KINDS,
            token_rows=int(
                self.parameters["semantic_cards.token_embedding.weight"].shape[0]
            ),
            name_rows=int(self.parameters["name_candidates.embedding.weight"].shape[0]),
            links=len(PERMANENT_LINK_INDEXES),
        )


def from_torch_state(
    model: ModelConfig,
    observation: ObservationConfig,
    state: Mapping[str, np.ndarray],
    metadata: Mapping[str, str] | None = None,
    *,
    parameter_names: Iterable[str],
) -> PolicyCheckpoint:
    """The container contents of a PyTorch state dict (as NumPy arrays).
    Every tensor becomes one canonical parameter or one lookup buffer."""
    names = tuple(parameter_names)
    missing = sorted(set(names) - set(state))
    if missing:
        raise KeyError(f"checkpoint lacks parameters {missing}")
    parameters: dict[str, np.ndarray] = {}
    for name in names:
        value = np.asarray(state[name])
        if value.dtype != np.float32:
            raise TypeError(f"parameter {name}: expected float32, got {value.dtype}")
        key = canonical_name(name)
        if key in parameters:
            raise KeyError(f"two torch parameters share canonical name {key}")
        parameters[key] = (
            value if value.flags.c_contiguous else np.ascontiguousarray(value)
        )
    parameter_set = set(names)
    meta = dict(metadata or {})
    # Adam snapshots address parameters positionally. A safetensors header sorts
    # tensor names, so preserve the writer's order explicitly for warm starts.
    meta["parameter_names"] = json.dumps(names)
    return PolicyCheckpoint(
        model=model,
        space=ObservationSpace(observation),
        parameters=parameters,
        buffers={
            name: np.asarray(value)
            for name, value in state.items()
            if name not in parameter_set
        },
        metadata=meta,
    )


# -- files ---------------------------------------------------------------------------


def checkpoint_metadata(
    record: Mapping[str, Any], converted_from: Path | None = None
) -> dict[str, str]:
    """Container metadata of a PyTorch checkpoint record (`torch.load` output)."""
    metadata = {key: json.dumps(record.get(key), default=str) for key in _METADATA_KEYS}
    metadata.update(
        {
            key: json.dumps(record[key])
            for key in _OPTIONAL_METADATA_KEYS
            if key in record
        }
    )
    if converted_from is not None:
        metadata["converted_from"] = str(converted_from)
    return metadata


def write_container(path: Path, checkpoint: PolicyCheckpoint) -> None:
    arrays = dict(checkpoint.parameters)
    arrays.update({_BUFFER + k: v for k, v in checkpoint.buffers.items()})
    write_safetensors(path, arrays, checkpoint.metadata)


def read_metadata(path: str | Path) -> dict[str, Any]:
    """Decode a container's JSON metadata without reading its tensor payload.

    The four original binding records are required; optional records remain
    absent on earlier containers. Legacy pickle files belong to a backend reader.
    """
    path = Path(path)
    try:
        with path.open("rb") as file:
            (length,) = struct.unpack("<Q", file.read(8))
            if length > path.stat().st_size - 8:
                raise ValueError("truncated header")
            header = json.loads(file.read(length))
        metadata = header.get("__metadata__", {})
        missing = [key for key in _METADATA_KEYS if key not in metadata]
        if missing:
            raise NotAContainer(f"{path} has no {', '.join(missing)} metadata")
        return {
            key: value if key == "converted_from" else json.loads(value)
            for key, value in metadata.items()
        }
    except (struct.error, ValueError, KeyError, TypeError) as error:
        if isinstance(error, NotAContainer):
            raise
        raise NotAContainer(f"{path} is not a policy checkpoint container") from error


def read_container(path: str | Path) -> PolicyCheckpoint:
    """Read a container, validating its world binding. Backend support is separate.

    Raises `NotAContainer` for a file that is not safetensors or lacks the
    container metadata; convert its PyTorch checkpoint again.
    """
    path = Path(path)
    remedy = "convert the PyTorch .pt with `uv run python -m manabot.backends.convert`"
    try:
        arrays, metadata = read_safetensors(path)
    except (struct.error, ValueError, KeyError) as error:
        raise NotAContainer(f"{path} is not a safetensors file; {remedy}") from error
    missing = [key for key in _METADATA_KEYS if key not in metadata]
    if missing:
        raise NotAContainer(
            f"{path} has no {', '.join(missing)} metadata, so it is not a policy "
            f"checkpoint container; {remedy}"
        )
    meta = {key: json.loads(metadata[key]) for key in _METADATA_KEYS}
    hypers = meta["hypers"]
    space = ObservationSpace(
        ObservationConfig.from_checkpoint(hypers["observation_hypers"])
    )
    validate_checkpoint_world(meta, space)
    model = ModelConfig.from_checkpoint(hypers["agent_hypers"])
    parameters = {k: v for k, v in arrays.items() if not k.startswith(_BUFFER)}
    for name, value in parameters.items():
        if value.dtype != np.float32:
            raise TypeError(f"parameter {name}: expected float32, got {value.dtype}")
    return PolicyCheckpoint(
        model=model,
        space=space,
        parameters=parameters,
        buffers={
            k[len(_BUFFER) :]: v for k, v in arrays.items() if k.startswith(_BUFFER)
        },
        metadata=metadata,
    )


# safetensors: an 8-byte little-endian header length, a JSON header naming each
# tensor's dtype, shape and byte range, then the little-endian tensor bytes.
_DTYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}
_CODES = {np.dtype(dtype): code for code, dtype in _DTYPES.items()}


def write_safetensors(
    path: Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, str]
) -> None:
    header: dict[str, Any] = {"__metadata__": dict(metadata)}
    blobs: list[bytes] = []
    offset = 0
    for name in sorted(arrays):
        array = np.asarray(arrays[name])
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        code = _CODES.get(array.dtype)
        if code is None:
            raise TypeError(f"{name}: dtype {array.dtype} has no safetensors code")
        blob = array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes()
        header[name] = {
            "dtype": code,
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        blobs.append(blob)
        offset += len(blob)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with path.open("wb") as file:
        file.write(struct.pack("<Q", len(encoded)))
        file.write(encoded)
        for blob in blobs:
            file.write(blob)


def read_safetensors(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    data = path.read_bytes()
    (length,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + length])
    metadata = header.pop("__metadata__", None) or {}
    body = memoryview(data)[8 + length :]
    arrays = {}
    for name, entry in header.items():
        start, end = entry["data_offsets"]
        dtype = np.dtype(_DTYPES[entry["dtype"]]).newbyteorder("<")
        arrays[name] = (
            np.frombuffer(body[start:end], dtype=dtype)
            .reshape(entry["shape"])
            .astype(dtype.newbyteorder("="))
        )
    return arrays, metadata
