"""Mandatory rules, setup and input binding for ordinary policy checkpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import managym
from managym.decision import SEMANTIC_DECISION_VERSION
from managym.possible_worlds import POSSIBLE_WORLD_SPACE_VERSION

WORLD = managym.WORLD_VERSION
POLICY_INPUT_VERSION = 1
ROOT = Path(__file__).resolve().parents[2]


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


def validate_agent_setup(agent: Any, player_configs) -> None:
    """Check an admitted policy against the actual match at execution time."""
    binding = getattr(agent, "world_binding", None)
    if binding is not None and binding["setups"] != _setups(player_configs):
        raise ValueError("checkpoint setup differs from the execution match")


def validate_policy_input(agent: Any, binding: dict) -> None:
    compiled = binding["content_manifest"].get("compiled_semantics")
    if compiled is not None and agent.hypers.semantic_pack != compiled["pack_key"]:
        raise ValueError(
            "compiled matchup checkpoint requires its complete semantic program input"
        )


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
    setups = _setups(configs)
    schema = {
        key: {"shape": list(shape), "dtype": str(observation_space.encoder.dtypes[key])}
        for key, shape in observation_space.shapes.items()
    }
    return {
        "world": WORLD,
        "policy_input_version": POLICY_INPUT_VERSION,
        "rules": {
            "decision_version": SEMANTIC_DECISION_VERSION,
            "possible_world_version": POSSIBLE_WORLD_SPACE_VERSION,
        },
        "setups": setups,
        "content_manifest": engine.content_pack_manifest(),
        "input_schema": schema,
        "learning_schema_sha256": hashlib.sha256(
            (ROOT / "content/semantic/v1/learning_schema.json").read_bytes()
        ).hexdigest(),
    }


def validate_checkpoint_world(
    checkpoint: Mapping[str, Any], observation_space: Any, player_configs=None
) -> dict:
    binding = checkpoint.get("world_binding")
    if not isinstance(binding, dict) or binding.get("world") != WORLD:
        raise ValueError(
            f"checkpoint requires an explicit compatible {WORLD} world binding"
        )
    try:
        configs = player_configs
        if configs is None:
            configs = [
                managym.PlayerConfig(str(index), setup["deck"], setup["sideboard"])
                for index, setup in enumerate(binding["setups"])
            ]
        expected = checkpoint_world(configs, observation_space)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("checkpoint world binding has invalid setup") from error
    if _digest(binding) != _digest(expected):
        raise ValueError(
            "checkpoint world/setup/input binding differs from this runtime"
        )
    return binding
