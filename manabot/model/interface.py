"""Compare the checkpoint's policy inputs with the engine that will serve it."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from manabot.env.decision_memory import MEMORY_FIELDS
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.infra.identity import canonical_sha256
from manabot.model.world import (
    checkpoint_world,
    deck_world,
    input_columns,
    validate_checkpoint_world,
    validate_saved_columns,
    written_at,
)
from manabot.model.writer_catalog import WriterCatalog, archived_catalog, writer_catalog
from manabot.policy.readings import (
    ColumnReading,
    Reading,
    Readings,
    fresh_readings,
    legacy_readings,
    recorded_readings,
)
import managym


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
            key: [str(value) for value in values] for key, values in readings.items()
        },
    }


def saved_readings(
    checkpoint: Mapping[str, Any], space: ObservationSpace
) -> dict[str, tuple[ColumnReading, ...]]:
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
        readings = recorded_readings(recorded, columns, written_at(checkpoint))
        if space.config.max_decision_records and not columns.keys() & set(
            MEMORY_FIELDS
        ):
            for key in MEMORY_FIELDS:
                readings[key] = (Reading.OWN,) * space.shapes[key][-1]
        return readings
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

    validate_saved_columns(checkpoint, expected["input_schema"], space, location)
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
