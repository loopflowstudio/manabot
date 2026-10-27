"""Mandatory rules, setup and input binding for ordinary policy checkpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
<<<<<<< HEAD
from pathlib import Path
from typing import Any

import managym
from managym.decision import SEMANTIC_DECISION_VERSION
from managym.possible_worlds import POSSIBLE_WORLD_SPACE_VERSION

WORLD = managym.WORLD_VERSION
POLICY_INPUT_VERSION = 1
ROOT = Path(__file__).resolve().parents[2]
=======
from typing import Any

from managym.decision import SEMANTIC_DECISION_VERSION
from managym.possible_worlds import POSSIBLE_WORLD_SPACE_VERSION

from manabot.semantic.learning import BoundSemanticPack
from manabot.semantic.policy_catalog import learning_schema_path
import managym

WORLD = managym.WORLD_VERSION
_ROSTER_PACKS = {"elemental-world-v1", "elemental-world-v2"}
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f


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


def validate_agent_setup(
    agent: Any,
    player_configs: Sequence[managym.PlayerConfig],
    *,
    allow_deck_repetition: bool = False,
) -> None:
    """Check an admitted policy against the actual match at execution time."""
    binding = getattr(agent, "world_binding", None)
    if binding is None:
        return
    actual = _setups(player_configs)
<<<<<<< HEAD
    if not allow_deck_repetition:
=======
    compiled = binding["content_manifest"].get("compiled_semantics")
    pack_key = compiled["pack_key"] if compiled is not None else None
    roster_world = pack_key in _ROSTER_PACKS
    if not roster_world and not allow_deck_repetition:
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
        if actual != binding["setups"]:
            raise ValueError("checkpoint setup differs from the execution match")
        return

    if len(actual) != 2 or any(setup not in binding["setups"] for setup in actual):
        raise ValueError("execution deck is outside checkpoint's exact setup roster")
<<<<<<< HEAD
    compiled = binding["content_manifest"].get("compiled_semantics")
    expected_pack = compiled["pack_key"] if compiled is not None else None
    if any(p.content_pack not in (None, expected_pack) for p in player_configs):
        raise ValueError("execution content pack differs from the checkpoint")
    if actual != binding["setups"] and any(
        p.content_pack != expected_pack for p in player_configs
=======
    if roster_world:
        if any(p.content_pack != pack_key for p in player_configs):
            raise ValueError("Elemental execution requires its explicit content pack")
        return
    if any(p.content_pack not in (None, pack_key) for p in player_configs):
        raise ValueError("execution content pack differs from the checkpoint")
    if actual != binding["setups"] and any(
        p.content_pack != pack_key for p in player_configs
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    ):
        raise ValueError(
            "repeated decks require the checkpoint's explicit content pack"
        )


def validate_policy_input(agent: Any, binding: dict) -> None:
    compiled = binding["content_manifest"].get("compiled_semantics")
<<<<<<< HEAD
    if compiled is not None and agent.hypers.semantic_pack != compiled["pack_key"]:
=======
    if compiled is not None and agent.config.semantic_pack != compiled["pack_key"]:
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
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
<<<<<<< HEAD
    setups = _setups(configs)
=======
    manifest = engine.content_pack_manifest()
    compiled = manifest.get("compiled_semantics")
    if compiled is not None and compiled["pack_key"] in _ROSTER_PACKS:
        # The authored world admits every pair, including mirrors. Bind the
        # complete native roster, while collection still uses its actual pair.
        setups = _setups(
            [
                managym.authored_deck_setup(compiled["pack_key"], key)
                for key in managym.authored_deck_keys(compiled["pack_key"])
            ]
        )
    else:
        setups = _setups(configs)
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    schema = {
        key: {"shape": list(shape), "dtype": str(observation_space.encoder.dtypes[key])}
        for key, shape in observation_space.shapes.items()
    }
<<<<<<< HEAD
    return {
        "world": WORLD,
        "policy_input_version": POLICY_INPUT_VERSION,
=======
    schema_path = learning_schema_path(compiled["pack_key"] if compiled else None)
    return {
        "world": WORLD,
        "policy_input_version": observation_space.encoder.config.policy_input_version,
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
        "rules": {
            "decision_version": SEMANTIC_DECISION_VERSION,
            "possible_world_version": POSSIBLE_WORLD_SPACE_VERSION,
        },
        "setups": setups,
<<<<<<< HEAD
        "content_manifest": engine.content_pack_manifest(),
        "input_schema": schema,
        "learning_schema_sha256": hashlib.sha256(
            (ROOT / "content/semantic/v1/learning_schema.json").read_bytes()
        ).hexdigest(),
    }


=======
        "content_manifest": manifest,
        "input_schema": schema,
        "learning_schema_sha256": (
            _digest(BoundSemanticPack.bind(manifest).schema.raw)
            if compiled is not None and compiled["pack_key"] in _ROSTER_PACKS
            else hashlib.sha256(schema_path.read_bytes()).hexdigest()
        ),
    }


def checkpoint_configs(binding: Mapping[str, Any]) -> list[managym.PlayerConfig]:
    """Reconstruct a representative root without dropping authored pack identity."""
    compiled = binding["content_manifest"].get("compiled_semantics")
    pack_key = None
    if compiled is not None:
        if not isinstance(compiled, dict) or not isinstance(
            compiled.get("pack_key"), str
        ):
            raise ValueError("invalid compiled content pack selection")
        pack_key = compiled["pack_key"]
    return [
        managym.PlayerConfig(str(index), setup["deck"], setup["sideboard"], pack_key)
        for index, setup in enumerate(binding["setups"][:2])
    ]


>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
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
<<<<<<< HEAD
            configs = [
                managym.PlayerConfig(str(index), setup["deck"], setup["sideboard"])
                for index, setup in enumerate(binding["setups"])
            ]
        expected = checkpoint_world(configs, observation_space)
    except (KeyError, TypeError, ValueError) as error:
=======
            configs = checkpoint_configs(binding)
        expected = checkpoint_world(configs, observation_space)
    except (KeyError, TypeError, ValueError, managym.AgentError) as error:
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
        raise ValueError("checkpoint world binding has invalid setup") from error
    if _digest(binding) != _digest(expected):
        raise ValueError(
            "checkpoint world/setup/input binding differs from this runtime"
        )
    return binding
