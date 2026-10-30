"""Mandatory rules, setup and input binding for ordinary policy checkpoints.

Two binding forms exist. A world binding (`deck_world`) names the installed
world and its card digest (`world_digest`) beside the shared rules and input
contract: the model may play any deck the world admits, whatever decks trained
it. A setup binding (`checkpoint_world`) names the exact setups of an authored
or default match; the Elemental packs bind their complete authored roster.
Neither form records how a model trained.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any

from managym.decision import SEMANTIC_DECISION_VERSION
from managym.possible_worlds import POSSIBLE_WORLD_SPACE_VERSION

from manabot.semantic.learning import BoundSemanticPack
from manabot.semantic.policy_catalog import learning_schema_path
import managym

WORLD = managym.WORLD_VERSION
_ROSTER_PACKS = {"elemental-world-v1", "elemental-world-v2"}


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
    binding = getattr(agent, "world_binding", None)
    if binding is None:
        return
    if _binds_world(binding):
        if len(player_configs) != 2:
            raise ValueError("execution requires two decks")
        validate_agent_decks(agent, player_configs)
        return
    actual = _setups(player_configs)
    pack_key = binding["content_manifest"].get("pack_key")
    roster_world = pack_key in _ROSTER_PACKS
    if not roster_world and not allow_deck_repetition:
        if actual != binding["setups"]:
            raise ValueError("checkpoint setup differs from the execution match")
        return

    if len(actual) != 2 or any(setup not in binding["setups"] for setup in actual):
        raise ValueError("execution deck is outside checkpoint's exact setup roster")
    if roster_world:
        if any(p.content_pack != pack_key for p in player_configs):
            raise ValueError("Elemental execution requires its explicit content pack")
        return
    if any(p.content_pack not in (None, pack_key) for p in player_configs):
        raise ValueError("execution content pack differs from the checkpoint")
    if actual != binding["setups"] and any(
        p.content_pack != pack_key for p in player_configs
    ):
        raise ValueError(
            "repeated decks require the checkpoint's explicit content pack"
        )


def validate_agent_decks(
    agent: Any, player_configs: Sequence[managym.PlayerConfig]
) -> None:
    """Check that a checkpoint may play every one of these decks, in any pairing.

    A world binding admits each deck the world admits natively; a setup binding
    admits only the decks of its setups.
    """
    binding = getattr(agent, "world_binding", None)
    if binding is None:
        return
    if _binds_world(binding):
        if any(p.content_pack != managym.WORLD_SETUP for p in player_configs):
            raise ValueError("execution deck is not submitted to the world")
        admit_world_decks(player_configs)
        return
    if any(setup not in binding["setups"] for setup in _setups(player_configs)):
        raise ValueError("execution deck is outside checkpoint's exact setup roster")


def admit_world_decks(player_configs: Sequence[managym.PlayerConfig]) -> None:
    """Admit every deck by the world's native validation; membership grants nothing."""
    checked = managym.check_decks([player.deck for player in player_configs])
    for player, issues in zip(player_configs, checked, strict=True):
        if issues:
            raise ValueError(
                f"deck {player.name} is not admitted by world {WORLD}: "
                + "; ".join(issue.message for issue in issues)
            )


def validate_policy_input(agent: Any, binding: dict) -> None:
    pack_key = (
        managym.WORLD_SETUP
        if _binds_world(binding)
        else binding["content_manifest"].get("pack_key")
    )
    if pack_key is not None and agent.config.semantic_pack != pack_key:
        raise ValueError(
            "authored matchup checkpoint requires its complete semantic program input"
        )


def _input_binding(observation_space: Any) -> dict:
    return {
        "world": WORLD,
        "policy_input_version": observation_space.encoder.config.policy_input_version,
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


def _schema_file_digest(pack_key: str | None) -> str:
    return hashlib.sha256(learning_schema_path(pack_key).read_bytes()).hexdigest()


def deck_world(observation_space: Any) -> dict:
    """Bind the installed world and input contract, independent of any deck.

    `world_digest` changes whenever the world's cards change.
    """
    return {
        **_input_binding(observation_space),
        "world_digest": managym.content_pool(managym.WORLD_SETUP)["content_digest"],
        "learning_schema_sha256": _schema_file_digest(managym.WORLD_SETUP),
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
    pack_key = manifest.get("pack_key")
    if pack_key in _ROSTER_PACKS:
        # The authored world admits every pair, including mirrors. Bind the
        # complete native roster, while collection still uses its actual pair.
        setups = _setups(
            [
                managym.authored_deck_setup(pack_key, key)
                for key in managym.authored_deck_keys(pack_key)
            ]
        )
    else:
        setups = _setups(configs)
    return {
        **_input_binding(observation_space),
        "setups": setups,
        "content_manifest": manifest,
        "learning_schema_sha256": (
            _digest(BoundSemanticPack.bind(manifest).schema.raw)
            if pack_key in _ROSTER_PACKS
            else _schema_file_digest(pack_key)
        ),
    }


def checkpoint_configs(binding: Mapping[str, Any]) -> list[managym.PlayerConfig]:
    """Reconstruct a representative root without dropping authored pack identity."""
    pack_key = binding["content_manifest"].get("pack_key")
    if pack_key is not None and not isinstance(pack_key, str):
        raise ValueError("invalid authored content pack selection")
    return [
        managym.PlayerConfig(str(index), setup["deck"], setup["sideboard"], pack_key)
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


def validate_checkpoint_world(
    checkpoint: Mapping[str, Any], observation_space: Any, player_configs=None
) -> dict:
    binding = checkpoint.get("world_binding")
    if not isinstance(binding, dict) or binding.get("world") != WORLD:
        raise ValueError(
            f"checkpoint requires an explicit compatible {WORLD} world binding"
        )
    try:
        expected = expected_world(binding, observation_space, player_configs)
    except (KeyError, TypeError, ValueError, managym.AgentError) as error:
        raise ValueError("checkpoint world binding has invalid setup") from error
    if _digest(binding) != _digest(expected):
        raise ValueError(
            "checkpoint world/setup/input binding differs from this runtime"
        )
    return binding
