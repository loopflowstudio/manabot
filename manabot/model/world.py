"""Mandatory rules, setup and input binding for ordinary policy checkpoints.

Two binding forms exist. A world binding (`deck_world`) names the installed
world and its card digest (`world_digest`) beside the shared rules and input
contract: the model may play any deck the world admits, whatever decks trained
it. A setup binding (`checkpoint_world`) names the exact setups of a match on
the world's one catalog; a match within the Elemental roster binds the
complete roster, and a binding of more than two setups admits any pair of
them. Neither form records how a model trained.

A refused checkpoint is run from the revision that wrote it: the checkpoint
records that revision (`checkpoint_source`), and a mismatch names it with the
fields and input buffers that differ.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
import subprocess
from typing import Any

from manabot.env.observation import ObservationSpace
from manabot.infra.source import source_commit, source_dirty
from manabot.model.writer_catalog import naming_entries
from manabot.policy.readings import published_columns
from manabot.semantic.learning import DEFAULT_SCHEMA_PATH, BoundSemanticPack
import managym
from managym.decision import SEMANTIC_DECISION_VERSION
from managym.possible_worlds import POSSIBLE_WORLD_SPACE_VERSION

# The authored setup whose decks form the Elemental roster (`elemental-world-v1`
# names the same decks).
_ROSTER_SETUP = "optional_payments"


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _setups(player_configs) -> list[dict]:
    return sorted(
        [
            dict(deck=dict(p.decklist), sideboard=dict(p.sideboard))
            for p in player_configs
        ],
        key=_digest,
    )


def _binds_world(binding: Mapping[str, Any]) -> bool:
    return "world_digest" in binding


def validate_agent_setup(
    agent: Any,
    player_configs: Sequence[managym.PlayerConfig],
    *,
    allow_deck_repetition: bool = False,
) -> None:
    """Check an admitted policy against the actual match at execution time."""
    validate_setup_binding(
        getattr(agent, "world_binding", None),
        player_configs,
        allow_deck_repetition=allow_deck_repetition,
    )


def validate_setup_binding(
    binding: Mapping[str, Any] | None,
    player_configs: Sequence[managym.PlayerConfig],
    *,
    allow_deck_repetition: bool = False,
) -> None:
    """Admit a checkpoint's binding without constructing its array framework."""
    if binding is None:
        return
    if _binds_world(binding):
        if len(player_configs) != 2:
            raise ValueError("execution requires two decks")
        admit_world_decks(player_configs)
        return
    actual = _setups(player_configs)
    roster_world = len(binding["setups"]) > 2
    if not roster_world and not allow_deck_repetition:
        if actual != binding["setups"]:
            raise ValueError("checkpoint setup differs from the execution match")
        return
    if len(actual) != 2 or any(setup not in binding["setups"] for setup in actual):
        raise ValueError("execution deck is outside checkpoint's exact setup roster")


def validate_agent_decks(
    agent: Any, player_configs: Sequence[managym.PlayerConfig]
) -> None:
    """Check that a checkpoint may play every one of these decks, in any pairing.

    Exact decks in the binding restrict play to that roster. Otherwise native
    admission checks each submitted deck.
    """
    binding = getattr(agent, "world_binding", None)
    if binding is None:
        return
    if _binds_world(binding):
        admit_world_decks(player_configs)
        return
    if any(setup not in binding["setups"] for setup in _setups(player_configs)):
        raise ValueError("execution deck is outside checkpoint's exact setup roster")


def admit_world_decks(player_configs: Sequence[managym.PlayerConfig]) -> None:
    """Admit every deck by the world's native validation; membership grants nothing."""
    if any(p.world != managym.WORLD_SETUP for p in player_configs):
        raise ValueError(
            "Supply a DeckMatchup with submitted decks for this checkpoint."
        )
    checked = managym.check_decks([player.deck for player in player_configs])
    for player, issues in zip(player_configs, checked, strict=True):
        if issues:
            raise ValueError(
                f"deck {player.name} is not admitted: "
                + "; ".join(issue.message for issue in issues)
            )


def validate_policy_input(agent: Any, binding: dict) -> None:
    """Check the semantic program input required by the recorded binding."""
    if _binds_world(binding) and agent.config.world != managym.WORLD_SETUP:
        raise ValueError("checkpoint requires the complete semantic program input")


def _input_binding(observation_space: Any) -> dict:
    return {
        "policy_input_version": observation_space.encoder.config.policy_input_version,
        # The input binds the native group partition through its branching
        # bound; changing it reinterprets every group route.
        "naming_representation": "public-groups-v1",
        "naming_branching": managym.NAME_BRANCHING,
        "naming_vocabulary": _digest(naming_entries()),
        "rules": {
            "decision_version": SEMANTIC_DECISION_VERSION,
            "possible_world_version": POSSIBLE_WORLD_SPACE_VERSION,
        },
        "input_schema": {
            key: {
                "shape": list(shape),
                "dtype": str(observation_space.encoder.dtypes[key]),
            }
            for key, shape in observation_space.shapes.items()
        },
    }


def _schema_file_digest() -> str:
    return hashlib.sha256(DEFAULT_SCHEMA_PATH.read_bytes()).hexdigest()


def deck_world(observation_space: Any) -> dict:
    """Bind the installed world and input contract, independent of any deck.

    `world_digest` changes whenever the world's cards change.
    """
    return {
        **_input_binding(observation_space),
        "world_digest": managym.world_identity()["card_digest"],
        "learning_schema_sha256": _schema_file_digest(),
    }


def checkpoint_world(player_configs: Sequence[Any], observation_space: Any) -> dict:
    """Bind the actual configured match, never a default-deck surrogate.

    Seats may reverse without changing compatibility; each deck stays paired
    with its own sideboard. Names and shuffle seeds are not rules identity.
    """
    configs = list(player_configs)
    if len(configs) != 2:
        raise ValueError("checkpoint world requires exactly two player setups")
    engine = managym.Env(seed=0)
    engine.reset(configs)
    manifest = engine.content_pack_manifest()
    roster = _setups(
        [
            managym.authored_deck_setup(_ROSTER_SETUP, key)
            for key in managym.authored_deck_keys(_ROSTER_SETUP)
        ]
    )
    actual = _setups(configs)
    in_roster = all(setup in roster for setup in actual)
    # The Elemental roster admits every pair, including mirrors. Bind the
    # complete roster, while collection still uses its actual pair.
    setups = roster if in_roster else actual
    return {
        **_input_binding(observation_space),
        "setups": setups,
        "content_manifest": manifest,
        "learning_schema_sha256": (
            _digest(BoundSemanticPack.bind(manifest).schema.raw)
            if in_roster
            else _schema_file_digest()
        ),
    }


def checkpoint_configs(binding: Mapping[str, Any]) -> list[managym.PlayerConfig]:
    """Reconstruct a representative root of the binding's setups."""
    return [
        managym.PlayerConfig(str(index), setup["deck"], setup["sideboard"])
        for index, setup in enumerate(binding["setups"][:2])
    ]


def expected_world(
    binding: Mapping[str, Any],
    observation_space: Any,
    player_configs: Sequence[managym.PlayerConfig] | None = None,
) -> dict:
    """Rebuild a binding's form from this runtime; world bindings need no deck."""
    if _binds_world(binding):
        return deck_world(observation_space)
    configs = player_configs
    if configs is None:
        configs = checkpoint_configs(binding)
    return checkpoint_world(configs, observation_space)


def current_binding_form(binding: Mapping[str, Any]) -> dict:
    """A retained binding in the form this runtime writes.

    Earlier revisions recorded a world label (`world`, and `world_version` in
    the content manifest); nothing compares it now, so an earlier-world
    checkpoint is refused by its input version or by the fields that differ.
    Bindings of the world's catalog recorded `pack_key: None` in their content
    manifest before the catalog was the only pack; that field carries nothing
    either. An authored pack's non-null key stays, so its binding never matches.
    """
    current = {key: value for key, value in binding.items() if key != "world"}
    manifest = current.get("content_manifest")
    if isinstance(manifest, Mapping):
        current["content_manifest"] = {
            key: value
            for key, value in manifest.items()
            if key != "world_version" and not (key == "pack_key" and value is None)
        }
    return current


def input_columns(space: ObservationSpace) -> dict[str, list[str]]:
    return {
        key: [column.name for column in columns]
        for key, columns in published_columns(space).items()
    }


def validate_saved_columns(
    checkpoint: Mapping[str, Any],
    expected_schema: Mapping[str, Any],
    space: ObservationSpace,
    location: str,
) -> None:
    """Check each saved array against the runtime's.

    Columns are only appended, so a checkpoint reads the runtime's first N
    columns; recorded names, when present, must equal that prefix."""
    schema = checkpoint["world_binding"].get("input_schema")
    if not isinstance(schema, dict) or schema.keys() != expected_schema.keys():
        raise ValueError(f"checkpoint input buffers differ; {location}")
    columns = input_columns(space)
    recorded = checkpoint.get("input_columns")
    if recorded is not None and (
        not isinstance(recorded, dict) or recorded.keys() != columns.keys()
    ):
        raise ValueError(f"checkpoint input column declarations differ; {location}")
    widths: dict[str, int] = {}
    for key, actual in expected_schema.items():
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


def validate_checkpoint_world(
    checkpoint: Mapping[str, Any],
    observation_space: ObservationSpace,
    player_configs: Sequence[managym.PlayerConfig] | None = None,
) -> dict[str, Any]:
    binding = checkpoint.get("world_binding")
    if not isinstance(binding, dict):
        raise ValueError("checkpoint requires an explicit world binding")
    try:
        expected = expected_world(binding, observation_space, player_configs)
    except (KeyError, TypeError, ValueError, managym.AgentError) as error:
        raise ValueError("checkpoint world binding has invalid setup") from error
    current = current_binding_form(binding)
    current.pop("input_schema", None)
    expected_schema = expected.pop("input_schema")
    if _digest(current) != _digest(expected):
        manifest = current.get("content_manifest")
        if isinstance(manifest, dict) and manifest != expected.get("content_manifest"):
            raise ValueError(
                "checkpoint content binding differs: trained on content "
                f"{str(manifest.get('content_digest', 'unknown'))[:12]}; {written_at(checkpoint)}"
            )
        raise ValueError(
            "checkpoint world/setup/input binding differs from this runtime in "
            f"{binding_difference(current, expected)}; {written_at(checkpoint)}"
        )
    validate_saved_columns(
        checkpoint, expected_schema, observation_space, written_at(checkpoint)
    )
    return binding


def checkpoint_source() -> dict[str, str | bool | None]:
    """The revision and Etude run writing a checkpoint, for a later mismatch to name."""
    try:
        commit: str | None = source_commit()
        dirty: bool | None = source_dirty()
    except (OSError, subprocess.SubprocessError, ValueError):
        commit, dirty = None, None
    return {
        "commit": commit,
        "dirty": dirty,
        "etude_experiment": os.environ.get("ETUDE_EXPERIMENT"),
        "etude_run": os.environ.get("ETUDE_RUN"),
    }


def written_at(checkpoint: Mapping[str, Any]) -> str:
    """Where to run a checkpoint this runtime refuses."""
    source = checkpoint.get("source")
    commit = source.get("commit") if isinstance(source, Mapping) else None
    if not commit:
        return "it records no source revision"
    run = f", Etude run {source['etude_run']}" if source.get("etude_run") else ""
    dirty = " with uncommitted changes" if source.get("dirty") else ""
    return f"it was written at revision {commit}{dirty}{run}; run it from there"


def binding_difference(binding: Mapping[str, Any], expected: Mapping[str, Any]) -> str:
    """Name differing fields outside the input schema."""
    return ", ".join(
        key
        for key in sorted(binding.keys() | expected.keys())
        if key not in binding or key not in expected or binding[key] != expected[key]
    )
