"""Compare the checkpoint's policy inputs with the engine that will serve it."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.infra.identity import canonical_sha256
from manabot.model.world import (
    checkpoint_world,
    deck_world,
    validate_checkpoint_world,
    written_at,
)
from manabot.model.writer_catalog import WriterCatalog, archived_catalog, writer_catalog
from manabot.policy.readings import (
    Reading,
    Readings,
    fresh_readings,
    legacy_readings,
    published_columns,
    recorded_readings,
)
import managym


def input_columns(space: ObservationSpace) -> dict[str, list[str]]:
    return {
        key: [column.name for column in columns]
        for key, columns in published_columns(space).items()
    }


def world_fields(
    space: ObservationSpace,
    readings: Readings,
    setups: Sequence[managym.PlayerConfig] | None = None,
) -> dict[str, Any]:
    """Record the named input prefixes, their readings and the catalog rows
    used by the saved policy. Each buffer's saved width is its readings' length."""
    binding = deck_world(space) if setups is None else checkpoint_world(setups, space)
    columns = input_columns(space)
    for key, values in readings.items():
        binding["input_schema"][key]["shape"][-1] = len(values)
        columns[key] = columns[key][: len(values)]
    return {
        "world_binding": binding,
        "writer_catalog": writer_catalog(),
        "input_columns": columns,
        "input_readings": {
            key: [value.value for value in values] for key, values in readings.items()
        },
    }


def saved_readings(
    checkpoint: Mapping[str, Any], space: ObservationSpace
) -> dict[str, tuple[Reading, ...]]:
    """The readings a checkpoint record was written with.

    A record without `input_readings` predates them and gets
    `legacy_readings`, cut to its saved column prefixes. A record without a
    world binding is a model never saved, which reads as a fresh one.
    """
    binding = checkpoint.get("world_binding")
    recorded = checkpoint.get("input_readings")
    if recorded is not None:
        columns = checkpoint.get("input_columns")
        if not isinstance(columns, dict):
            raise ValueError(
                f"checkpoint input readings have no columns; {written_at(checkpoint)}"
            )
        return recorded_readings(recorded, columns, written_at(checkpoint))
    if not isinstance(binding, dict):
        return fresh_readings(space)
    schema = binding.get("input_schema")
    widths = {
        key: entry["shape"][-1]
        for key, entry in (schema if isinstance(schema, dict) else {}).items()
        if isinstance(entry, dict) and isinstance(entry.get("shape"), list)
    }
    return {
        key: values[: widths.get(key)] for key, values in legacy_readings(space).items()
    }


@dataclass(frozen=True)
class ModelInterface:
    """A checkpoint's writer vocabulary and the running one. Its saved column
    prefixes are the lengths of its readings (`saved_readings`)."""

    writer: WriterCatalog
    current: WriterCatalog


def model_interface(
    checkpoint: Mapping[str, Any], space: ObservationSpace
) -> ModelInterface | None:
    """Resolve the writer vocabulary. Return None after validating a binding
    that records exact decks."""
    saved = checkpoint.get("world_binding")
    if not isinstance(saved, dict):
        raise ValueError("trained without a world binding")
    if "world_digest" in saved and ("setups" in saved or "content_manifest" in saved):
        raise ValueError("checkpoint binding mixes a world and a fixed setup")
    if "world_digest" not in saved:
        validate_checkpoint_world(checkpoint, space)
        return None
    model = ModelConfig.from_checkpoint(checkpoint["hypers"]["agent_hypers"])
    if model.world != managym.WORLD_SETUP:
        raise ValueError("checkpoint requires the complete semantic program input")
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

    validate_saved_columns(checkpoint, expected, space, location)
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
    return ModelInterface(writer, current)


def validate_saved_columns(
    checkpoint: Mapping[str, Any],
    expected: Mapping[str, Any],
    space: ObservationSpace,
    location: str,
) -> None:
    """Check each saved array against the runtime's.

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
            widths[key] = width
        elif shape != current_shape:
            raise ValueError(f"incompatible {key} shape {shape}; {location}")
    for group in ("player", "cards", "permanents"):
        if widths.get(f"agent_{group}") != widths.get(f"opponent_{group}"):
            raise ValueError(f"{group} column counts differ between seats; {location}")


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
    return archived_catalog(digest, location)
