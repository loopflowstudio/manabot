"""Compare the checkpoint's policy inputs with the engine that will serve it."""

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.infra.identity import canonical_sha256
from manabot.model.world import WORLD, deck_world, validate_checkpoint_world, written_at
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


def world_fields(space: ObservationSpace, player_width: int) -> dict[str, Any]:
    """Record the rows this model actually reads, including a rebound player's width."""
    binding = deck_world(space)
    columns = input_columns(space)
    for key in ("agent_player", "opponent_player"):
        binding["input_schema"][key]["shape"][-1] = player_width
        columns[key] = columns[key][:player_width]
    return {
        "world_binding": binding,
        "writer_catalog": writer_catalog(),
        "input_columns": columns,
    }


@dataclass(frozen=True)
class ModelInterface:
    writer: WriterCatalog | None
    current: WriterCatalog | None
    player_width: int


def model_interface(
    checkpoint: Mapping[str, Any], space: ObservationSpace
) -> ModelInterface:
    """Validate setup identity or resolve the world pilot's writer vocabulary."""
    saved = checkpoint.get("world_binding")
    width = space.encoder.player_dim
    if not isinstance(saved, dict):
        raise ValueError(f"trained without a world binding; engine is {WORLD}")
    if "world_digest" in saved and ("setups" in saved or "content_manifest" in saved):
        raise ValueError("checkpoint binding mixes a world and a fixed setup")
    if "world_digest" not in saved:
        validate_checkpoint_world(checkpoint, space)
        return ModelInterface(None, None, width)
    model = ModelConfig.from_checkpoint(checkpoint["hypers"]["agent_hypers"])
    if model.world != managym.WORLD_SETUP:
        raise ValueError(
            "world checkpoint requires its complete semantic program input"
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

    schema = saved.get("input_schema")
    if not isinstance(schema, dict) or schema.keys() != expected["input_schema"].keys():
        raise ValueError(f"checkpoint input buffers differ; {location}")
    columns = input_columns(space)
    recorded = checkpoint.get("input_columns")
    if recorded is not None and (
        not isinstance(recorded, dict) or recorded.keys() != columns.keys()
    ):
        raise ValueError(f"checkpoint input column declarations differ; {location}")
    widths = []
    for key, actual in expected["input_schema"].items():
        entry = schema[key]
        if not isinstance(entry, dict) or entry.get("dtype") != actual["dtype"]:
            raise ValueError(f"incompatible {key} dtype; {location}")
        shape = entry.get("shape")
        if key in ("agent_player", "opponent_player"):
            if (
                not isinstance(shape, list)
                or len(shape) != 2
                or shape[0] != 1
                or type(shape[1]) is not int
                or not 0 < shape[1] <= width
            ):
                raise ValueError(f"incompatible {key} columns {shape}; {location}")
            widths.append(shape[1])
        elif shape != actual["shape"]:
            raise ValueError(f"incompatible {key} shape {shape}; {location}")
        if recorded is not None and key in columns:
            wanted = (
                columns[key][: shape[-1]]
                if key in ("agent_player", "opponent_player")
                else columns[key]
            )
            if recorded[key] != wanted:
                raise ValueError(
                    f"incompatible {key} columns: checkpoint {recorded[key]}, runtime {wanted}; {location}"
                )
    if widths[0] != widths[1]:
        raise ValueError(f"player column counts differ between seats; {location}")
    current = writer_catalog()
    digest = saved["world_digest"]
    writer = checkpoint.get("writer_catalog")
    if (
        writer is None
        and digest == current["world_digest"]
        and saved.get("naming_vocabulary") == expected["naming_vocabulary"]
    ):
        writer = current
    if writer is None:
        path = Path(__file__).with_name("writer_catalogs") / f"{digest}.json"
        if (
            not isinstance(digest, str)
            or not re.fullmatch("[0-9a-f]{64}", digest)
            or not path.is_file()
        ):
            raise ValueError(
                f"missing writer catalog for {digest}; export with uv run python -m manabot.model.writer_catalog at the writing revision; {location}"
            )
        writer = json.loads(path.read_bytes())
    if (
        not isinstance(writer, dict)
        or not isinstance(writer.get("definition_keys"), list)
        or not all(isinstance(key, str) for key in writer["definition_keys"])
    ):
        raise ValueError(f"writer catalog requires ordered definition keys; {location}")
    if writer.get("world_digest") != digest or canonical_sha256(
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
    return ModelInterface(writer, current, widths[0])
