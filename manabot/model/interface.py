"""Compare the checkpoint's policy inputs with the engine that will serve it."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.infra.identity import canonical_sha256
from manabot.model.world import (
    WORLD,
    checkpoint_world,
    deck_world,
    validate_checkpoint_world,
    written_at,
)
from manabot.model.writer_catalog import WriterCatalog, writer_catalog
import managym


def input_columns(space: ObservationSpace) -> dict[str, list[str]]:
    encoder = space.encoder
    layout = managym.observation_layout(
        policy_history=bool(encoder.config.policy_history_version),
        turn_state=encoder.turn_state,
        legacy=not encoder.one_token,
    )
    return {
        key: sorted(row["columns"], key=lambda name: row["columns"][name]["index"])
        for key, row in layout.items()
        if row["columns"]
    }


def world_fields(
    space: ObservationSpace,
    widths: Mapping[str, int],
    setups: Sequence[managym.PlayerConfig] | None = None,
) -> dict[str, Any]:
    """Record the named input prefixes and catalog rows used by the saved policy."""
    binding = deck_world(space) if setups is None else checkpoint_world(setups, space)
    columns = input_columns(space)
    for key, width in widths.items():
        binding["input_schema"][key]["shape"][-1] = width
        columns[key] = columns[key][:width]
    return {
        "world_binding": binding,
        "writer_catalog": writer_catalog(),
        "input_columns": columns,
    }


@dataclass(frozen=True)
class ModelInterface:
    """A checkpoint's writer vocabulary, the running one, and its saved column
    prefixes (only the arrays narrower than the runtime's)."""

    writer: WriterCatalog
    current: WriterCatalog
    widths: dict[str, int]


def model_interface(
    checkpoint: Mapping[str, Any], space: ObservationSpace
) -> ModelInterface | None:
    """Resolve the writer vocabulary. Return None after validating a binding
    that records exact decks."""
    saved = checkpoint.get("world_binding")
    if not isinstance(saved, dict):
        raise ValueError(f"trained without a world binding; engine is {WORLD}")
    if "world_digest" in saved and ("setups" in saved or "content_manifest" in saved):
        raise ValueError("checkpoint binding mixes a world and a fixed setup")
    if "world_digest" not in saved:
        validate_checkpoint_world(checkpoint, space)
        return None
    model = ModelConfig.from_checkpoint(checkpoint["hypers"]["agent_hypers"])
    if model.world != managym.WORLD_SETUP:
        raise ValueError(
            "checkpoint requires the complete semantic program input"
        )
    expected = deck_world(space)
    location = written_at(checkpoint)
    for key in (
        "policy_input_version",
        "learning_schema_sha256",
        "naming_representation",
        "naming_branching",
    ):
        if saved.get(key) != expected[key]:
            raise ValueError(f"incompatible checkpoint binding for {key}; {location}")
    rules = saved.get("rules")
    required = ["decision_version"]
    if model.belief_count_buckets:
        required.append("possible_world_version")
    for key in required:
        if not isinstance(rules, dict) or rules.get(key) != expected["rules"][key]:
            raise ValueError(
                f"incompatible checkpoint rules binding for {key}; {location}"
            )

    widths = saved_widths(checkpoint, expected, space, location)
    current = writer_catalog()
    writer = checkpoint.get("writer_catalog")
    if writer is None:
        writer = recorded_writer(saved, expected, current, location)
    if (
        not isinstance(writer, dict)
        or not isinstance(writer.get("definition_keys"), list)
        or not all(isinstance(key, str) for key in writer["definition_keys"])
    ):
        raise ValueError(f"writer catalog requires ordered definition keys; {location}")
    if writer.get("world_digest") != saved["world_digest"] or canonical_sha256(
        writer.get("naming_catalog")
    ) != saved.get("naming_vocabulary"):
        raise ValueError(
            f"writer catalog does not match checkpoint binding for naming vocabulary or world digest; {location}"
        )
    if not model.catalog_records() and writer != current:
        raise ValueError(
            f"rebinding card programs requires the catalog record vocabulary; {location}"
        )
    if (
        model.belief_count_buckets
        and writer["definition_keys"] != current["definition_keys"]
    ):
        raise ValueError(
            f"belief card embedding requires a bound belief vocabulary; {location}"
        )
    return ModelInterface(writer, current, widths)


def saved_widths(
    checkpoint: Mapping[str, Any],
    expected: Mapping[str, Any],
    space: ObservationSpace,
    location: str,
) -> dict[str, int]:
    """The saved width of each named array narrower than the runtime's.

    Columns are only appended, so a checkpoint reads the runtime's first N
    columns; recorded names, when present, must equal that prefix."""
    schema = checkpoint["world_binding"].get("input_schema")
    if not isinstance(schema, dict) or schema.keys() != expected["input_schema"].keys():
        raise ValueError(f"checkpoint input buffers differ; {location}")
    columns = input_columns(space)
    recorded = checkpoint.get("input_columns")
    if recorded is not None and (
        not isinstance(recorded, dict) or recorded.keys() != columns.keys()
    ):
        raise ValueError(f"checkpoint input column declarations differ; {location}")
    widths: dict[str, int] = {}
    for key, actual in expected["input_schema"].items():
        entry = schema[key]
        if not isinstance(entry, dict) or entry.get("dtype") != actual["dtype"]:
            raise ValueError(f"incompatible {key} dtype; {location}")
        shape = entry.get("shape")
        current_shape = actual["shape"]
        if key in columns:
            if (
                not isinstance(shape, list)
                or shape[:-1] != current_shape[:-1]
                or not shape
                or type(shape[-1]) is not int
                or not 0 < shape[-1] <= current_shape[-1]
            ):
                raise ValueError(f"incompatible {key} columns {shape}; {location}")
            width = shape[-1]
            wanted = columns[key][:width]
            if recorded is not None and recorded[key] != wanted:
                raise ValueError(
                    f"incompatible {key} columns: checkpoint {recorded[key]}, runtime {wanted}; {location}"
                )
            if width != current_shape[-1]:
                widths[key] = width
        elif shape != current_shape:
            raise ValueError(f"incompatible {key} shape {shape}; {location}")
    for group in ("player", "cards", "permanents"):
        if widths.get(f"agent_{group}") != widths.get(f"opponent_{group}"):
            raise ValueError(f"{group} column counts differ between seats; {location}")
    return widths


def recorded_writer(
    saved: Mapping[str, Any],
    expected: Mapping[str, Any],
    current: WriterCatalog,
    location: str,
) -> Any:
    """The writer catalog of a checkpoint saved before it recorded one: the
    running catalog when the card and naming digests agree, otherwise the
    committed file for its card digest. The caller validates the contents."""
    digest = saved["world_digest"]
    if (
        digest == current["world_digest"]
        and saved.get("naming_vocabulary") == expected["naming_vocabulary"]
    ):
        return current
    path = Path(__file__).with_name("writer_catalogs") / f"{digest}.json"
    if (
        not isinstance(digest, str)
        or not re.fullmatch("[0-9a-f]{64}", digest)
        or not path.is_file()
    ):
        raise ValueError(
            f"missing writer catalog for {digest}; at the writing revision run "
            f"uv run python -m manabot.model.writer_catalog and save its output as "
            f"manabot/model/writer_catalogs/{digest}.json; {location}"
        )
    return json.loads(path.read_bytes())
