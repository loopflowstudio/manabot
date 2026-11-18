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
from dataclasses import dataclass, replace
import json
import logging
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
from manabot.model.architecture import (
    ArchitectureReceipt,
    architecture_receipt,
    validate_architecture_receipt,
)
from manabot.model.interface import model_interface, world_fields
from manabot.model.world import checkpoint_source, written_at
from manabot.policy.binding import BindingReport, bind_catalog
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
    "writer_catalog",
    "input_columns",
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


def layout(
    space: ObservationSpace, column_widths: dict[str, int] | None = None
) -> Layout:
    encoder = space.encoder
    width = (column_widths or {}).get("agent_permanents", encoder.permanent_dim)
    links = tuple(c for c in PERMANENT_LINK_INDEXES if c < width)
    joins = (PERMANENT_DEFINITION_INDEX, *links)
    return Layout(
        numeric_permanent_columns=tuple(c for c in range(width) if c not in joins),
        link_columns=links,
        column_widths=column_widths,
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
    column_widths: dict[str, int] | None = None
    report: BindingReport | None = None

    def receipt(self) -> ArchitectureReceipt:
        return architecture_receipt(
            self.model,
            self.space,
            {k: (int(v.size), True) for k, v in self.parameters.items()},
        )

    def record(self) -> dict[str, Any]:
        return {
            key: json.loads(value)
            for key, value in self.metadata.items()
            if key != "converted_from"
        }

    @property
    def writer_revision(self) -> str | None:
        source = json.loads(self.metadata.get("source", "null"))
        return source.get("commit") if isinstance(source, dict) else None

    @property
    def stored_programs(self) -> int:
        """Count writer programs in use, excluding new runtime definitions."""
        if self.report is not None:
            return self.report.stored_programs
        # With no catalog rebind the backend uses the saved buffers directly.
        programs = self.buffers.get("semantic_cards.tokens")
        return len(programs) if programs is not None else 0

    def tables(self) -> Tables:
        return Tables.from_buffers(self.buffers, layout(self.space, self.column_widths))

    def dimensions(self) -> Dimensions:
        encoder = self.space.encoder
        model = self.model
        width = (self.column_widths or {}).get
        columns = layout(self.space, self.column_widths)
        return Dimensions(
            hidden=model.hidden_dim,
            heads=model.num_attention_heads,
            layers=model.attention_layers,
            feedforward=model.attention_feedforward_dim
            or model.num_attention_heads * model.hidden_dim,
            player=width("agent_player", encoder.player_dim),
            card=width("agent_cards", encoder.card_dim),
            permanent=len(columns.numeric_permanent_columns),
            action=width("actions", encoder.action_dim),
            focus=encoder.max_focus_objects,
            relation=width("relations", encoder.relation_dim),
            fact_kinds=FACT_KINDS,
            token_rows=int(
                self.parameters["semantic_cards.token_embedding.weight"].shape[0]
            ),
            name_rows=int(self.parameters["name_candidates.embedding.weight"].shape[0]),
            links=len(columns.link_columns),
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
    metadata = dict(checkpoint.metadata)
    if checkpoint.column_widths is not None:
        # read_container returns runtime-bound arrays. Serializing those arrays
        # makes this runtime their writer, while conversion uses the originals.
        record = {
            **world_fields(checkpoint.space, checkpoint.column_widths),
            "architecture": checkpoint.receipt().model_dump(mode="json"),
            "source": checkpoint_source(),
        }
        metadata.update({key: json.dumps(value) for key, value in record.items()})
    write_safetensors(path, arrays, metadata)


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
    return admit_checkpoint(read_saved_container(path))


def read_saved_container(path: str | Path) -> PolicyCheckpoint:
    """A container's tensors and metadata as written, before admission."""
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


def admit_checkpoint(checkpoint: PolicyCheckpoint) -> PolicyCheckpoint:
    """Verify the receipt against the tensors that were written, before adding rows."""
    record = checkpoint.record()
    interface = model_interface(record, checkpoint.space)
    if record.get("architecture") is not None:
        validate_architecture_receipt(checkpoint.receipt(), record["architecture"])
    if interface is None:
        return checkpoint
    width = interface.widths.get("agent_player", checkpoint.space.encoder.player_dim)
    projected = checkpoint.parameters["player_embedding.projection.0.weight"].shape[1]
    extra = (
        interface.widths.get(
            "player_turn_state", checkpoint.space.shapes["player_turn_state"][-1]
        )
        if checkpoint.space.encoder.turn_state
        else 0
    )
    if projected != width + extra:
        raise ValueError("checkpoint player columns differ from its projection")
    if not checkpoint.model.catalog_records() and interface.writer == interface.current:
        return replace(checkpoint, column_widths=interface.widths)
    try:
        bound, report = bind_catalog(checkpoint, interface.writer, interface.current)
    except ValueError as error:
        raise ValueError(f"{error}; {written_at(record)}") from error
    logging.getLogger(__name__).info(
        "Policy input v%s (%s). Player columns %d/%d. %s.",
        checkpoint.space.encoder.config.policy_input_version,
        written_at(record),
        width,
        checkpoint.space.encoder.player_dim,
        report.summary(),
    )
    return replace(bound, column_widths=interface.widths, report=report)


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
