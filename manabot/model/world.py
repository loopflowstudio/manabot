"""Mandatory rules, content and input binding for ordinary policy checkpoints.

Two binding forms exist. A pool binding (`pool_world`) names a qualified
content pool by its native descriptor: the model may play any deck that pool
admits, whatever decks trained it. A setup binding (`checkpoint_world`) names
exact setups. Over a compiled authored pack it names the pack's complete
authored roster, whichever pair trained it; the Elemental packs did so before
the form was versioned, and new bindings over other authored packs carry
`setup_binding: authored-pack-v1`. Unversioned bindings over other packs keep
their historical pair semantics (`_pair_world`), as does default content,
which has no native deck admission. Neither form records how a model trained.
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
# Packs whose roster binding predates the versioned form; their artifacts
# carry no marker and must keep their exact digests.
_ROSTER_PACKS = {"elemental-world-v1", "elemental-world-v2"}
AUTHORED_PACK_BINDING = "authored-pack-v1"


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
    pack_key = bound_pack(binding)
    if "content_pool" in binding:
        # Any deck the bound pool admits may execute; unsupported cards never do.
        if len(player_configs) != 2 or any(
            p.content_pack != pack_key for p in player_configs
        ):
            raise ValueError(
                "execution requires two decks in the checkpoint's content pool"
            )
        admit_pool_decks(pack_key, player_configs)
        return
    actual = _setups(player_configs)
    roster_world = _binds_roster(binding)
    if not roster_world and not allow_deck_repetition:
        if actual != binding["setups"]:
            raise ValueError("checkpoint setup differs from the execution match")
        return

    if len(actual) != 2 or any(setup not in binding["setups"] for setup in actual):
        raise ValueError(
            "execution deck is outside checkpoint's exact setup roster "
            "(setup differs from every authored setup)"
        )
    # Native reset admits any pair of a pack's authored decks, mirrors included;
    # the Python side checks roster membership and the selected pack only.
    if pack_key in _ROSTER_PACKS:
        if any(p.content_pack != pack_key for p in player_configs):
            raise ValueError("Elemental execution requires its explicit content pack")
        return
    if any(p.content_pack not in (None, pack_key) for p in player_configs):
        raise ValueError("execution content pack differs from the checkpoint")
    # Native reset infers a pack only from its own distinct two-deck pair.
    if actual != binding["setups"] and any(
        p.content_pack != pack_key for p in player_configs
    ):
        raise ValueError(
            "decks other than the bound pair require the checkpoint's explicit "
            "content pack"
        )


def admit_pool_decks(
    pack_key: str | None, player_configs: Sequence[managym.PlayerConfig]
) -> None:
    """Admit every deck by the pool's native validation; membership grants nothing."""
    for player in player_configs:
        issues = managym.admit_deck(pack_key, player.decklist, player.sideboard)
        if issues:
            raise ValueError(
                f"deck {player.name} is not admitted by {pack_key}: "
                + "; ".join(issue["message"] for issue in issues)
            )


def _binds_roster(binding: Mapping[str, Any]) -> bool:
    return (
        binding.get("setup_binding") == AUTHORED_PACK_BINDING
        or bound_pack(binding) in _ROSTER_PACKS
    )


def _authored_roster(pack_key: str) -> list[dict] | None:
    """The pack's complete authored setups, or None when it authors no decks."""
    try:
        keys = managym.authored_deck_keys(pack_key)
    except ValueError:  # pool packs author no decks
        return None
    return _setups([managym.authored_deck_setup(pack_key, key) for key in keys])


def bound_pack(binding: Mapping[str, Any]) -> str | None:
    """Compiled content pack a binding names, or None for default content."""
    if "content_pool" in binding:
        return binding["content_pool"]["pack_key"]
    compiled = binding["content_manifest"].get("compiled_semantics")
    return compiled["pack_key"] if compiled is not None else None


def validate_policy_input(agent: Any, binding: dict) -> None:
    pack_key = bound_pack(binding)
    if pack_key is not None and agent.config.semantic_pack != pack_key:
        raise ValueError(
            "compiled matchup checkpoint requires its complete semantic program input"
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


def pool_world(pack_key: str, observation_space: Any) -> dict:
    """Bind a qualified content pool's descriptor, independent of any deck."""
    return {
        **_input_binding(observation_space),
        "content_pool": managym.content_pool(pack_key),
        "learning_schema_sha256": _schema_file_digest(pack_key),
    }


def checkpoint_world(player_configs: Sequence[Any], observation_space: Any) -> dict:
    """Bind the actual configured world, never a default-deck surrogate.

    A match of authored decks from one compiled pack binds that pack's
    complete roster, so the model may execute any pair native reset admits.
    Other setups bind their exact pair. Seats may reverse without changing
    compatibility; each deck stays paired with its own sideboard. Names and
    shuffle seeds are not rules identity.
    """
    configs, manifest = _configured(player_configs)
    compiled = manifest.get("compiled_semantics")
    pack_key = compiled["pack_key"] if compiled is not None else None
    roster = _authored_roster(pack_key) if pack_key is not None else None
    if pack_key in _ROSTER_PACKS:
        return {
            **_input_binding(observation_space),
            "setups": roster,
            "content_manifest": manifest,
            "learning_schema_sha256": _digest(
                BoundSemanticPack.bind(manifest).schema.raw
            ),
        }
    if roster is None or any(setup not in roster for setup in _setups(configs)):
        return _pair_world(configs, manifest, observation_space)
    return {
        **_input_binding(observation_space),
        "setup_binding": AUTHORED_PACK_BINDING,
        "setups": roster,
        "content_manifest": manifest,
        "learning_schema_sha256": _schema_file_digest(pack_key),
    }


def _pair_world(configs: Sequence[Any], manifest: dict, observation_space: Any) -> dict:
    """Exact-pair binding: default content, and historical authored bindings."""
    compiled = manifest.get("compiled_semantics")
    return {
        **_input_binding(observation_space),
        "setups": _setups(configs),
        "content_manifest": manifest,
        "learning_schema_sha256": _schema_file_digest(
            compiled["pack_key"] if compiled else None
        ),
    }


def _configured(player_configs: Sequence[Any]) -> tuple[list[Any], dict]:
    configs = list(player_configs)
    if len(configs) != 2:
        raise ValueError("checkpoint world requires exactly two player setups")
    engine = managym.Env(seed=0)
    engine.reset(configs)
    return configs, engine.content_pack_manifest()


def _schema_file_digest(pack_key: str | None) -> str:
    return hashlib.sha256(learning_schema_path(pack_key).read_bytes()).hexdigest()


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


def expected_world(
    binding: Mapping[str, Any],
    observation_space: Any,
    player_configs: Sequence[managym.PlayerConfig] | None = None,
) -> dict:
    """Rebuild a binding's form from this runtime; pool bindings need no deck."""
    if "content_pool" in binding:
        return pool_world(binding["content_pool"]["pack_key"], observation_space)
    configs = player_configs
    if configs is None:
        configs = checkpoint_configs(binding)
    if _binds_roster(binding) or bound_pack(binding) is None:
        return checkpoint_world(configs, observation_space)
    return _pair_world(configs, _configured(configs)[1], observation_space)


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
