"""Server-scoped selectable policies. Providers own artifact discovery and admission.

The historical identity envelope (name/producer/sha256) remains readable by
existing tables and traces; new records bind the complete runtime descriptor.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from managym.policy import PolicyDescriptor, PolicyRegistry
from managym.search import (
    SearchConfig,
    descriptor_search_unavailable,
    search_descriptor,
)
from managym.settings_fit import (
    SettingsFitReport,
    compare_settings,
    distributions_from_counts,
)

from . import advice

BUILTIN_OPPONENTS: tuple[dict[str, Any], ...] = (
    {
        "id": "search-256",
        "name": "Search 256",
        "kind": "search",
        "group": "Search",
        "summary": "256 random playouts per legal action",
        "requires_open_decklists": True,
    },
    {
        "id": "search-64",
        "name": "Search 64",
        "kind": "search",
        "group": "Search",
        "summary": "64 random playouts per legal action",
        "requires_open_decklists": True,
    },
    {
        "id": "search-16",
        "name": "Search 16",
        "kind": "search",
        "group": "Search",
        "summary": "16 random playouts per legal action",
        "requires_open_decklists": True,
    },
    {
        "id": "random",
        "name": "Random",
        "kind": "scripted",
        "group": "Scripted",
        "summary": "Uniform over legal actions",
        "requires_open_decklists": False,
    },
    {
        "id": "passive",
        "name": "Passive",
        "kind": "scripted",
        "group": "Scripted",
        "summary": "Passes when possible",
        "requires_open_decklists": False,
    },
)

_registry: PolicyRegistry | None = None


def policy_registry() -> PolicyRegistry:
    global _registry
    if _registry is None:
        _registry = PolicyRegistry()
        _registry.discover()
    return _registry


def set_policy_registry(registry: PolicyRegistry | None) -> None:
    global _registry
    _registry = registry


@dataclass(frozen=True)
class Opponent:
    descriptor: PolicyDescriptor

    @property
    def name(self) -> str:
        name_parts = {
            part.strip().casefold() for part in self.descriptor.name.split("·")
        }
        aliases = sorted(
            alias
            for alias, key in policy_registry().aliases.items()
            if key == self.descriptor.id and alias.casefold() not in name_parts
        )
        return " · ".join((self.descriptor.name, *aliases))

    def identity(self, mirror: SearchConfig | None = None) -> dict[str, str]:
        base = replace(self.descriptor, name=self.name)
        descriptor = base if mirror is None else search_descriptor(base, mirror)
        return {
            "name": descriptor.name,
            "producer": descriptor.provider,
            "sha256": descriptor.id,
            "bot_id": f"policy:{descriptor.id}",
            **({"base_sha256": self.descriptor.id} if mirror is not None else {}),
        }

    def record(self, mirror: SearchConfig | None = None) -> dict[str, Any]:
        descriptor = (
            self.descriptor
            if mirror is None
            else search_descriptor(self.descriptor, mirror)
        )
        return {
            **self.identity(mirror),
            "schema": "managym-policy-1",
            "policy": descriptor.record(),
        }


def opponent_kind(policy: dict[str, Any]) -> str:
    """How a recorded policy (`PolicyDescriptor.record()`) decides, for display:
    "search" over another policy, "scripted" with no artifact, otherwise
    "checkpoint" and the world it was trained in when its details name one."""
    if policy.get("base_policy"):
        return "search"
    if not policy.get("artifacts"):
        return "scripted"
    world = next(
        (
            detail.get("value")
            for detail in policy.get("details") or []
            if detail.get("label") == "World" and detail.get("value") != "not recorded"
        ),
        None,
    )
    return f"checkpoint {world}" if world else "checkpoint"


def default_opponent_id() -> str:
    """Preferred default opponent: recommended hosted policy -> heuristic -> search-64."""
    registry = policy_registry()
    descriptors = registry.descriptors()
    for descriptor in descriptors:
        if descriptor.recommended and descriptor.incompatibility is None:
            return descriptor.alias or f"policy:{descriptor.id}"
    if "heuristic" in registry.aliases:
        try:
            if registry.descriptor("heuristic").incompatibility is None:
                return "heuristic"
        except ValueError:
            pass
    return "search-64"


def policy_settings_fit(
    key: str,
    planned: Mapping[str, str],
    *,
    seat: int | None = None,
) -> SettingsFitReport | None:
    """Informational MTG-399 settings comparison for a registered policy."""
    try:
        descriptor = policy_registry().descriptor(key.removeprefix("policy:"))
    except ValueError:
        return None
    if not descriptor.artifacts and descriptor.trained_settings is None:
        return None
    trained = distributions_from_counts(descriptor.trained_settings, seat=seat)
    return compare_settings(trained, planned)


def opponent_catalog() -> dict[str, Any]:
    entries = []
    error = None
    provider_names: list[str] = []
    default_id = "search-64"
    try:
        registry = policy_registry()
        provider_names = sorted(registry.providers)
        error = "; ".join(registry.discovery_errors) or None
        default_id = default_opponent_id()
        for descriptor in registry.descriptors():
            record = descriptor.record()
            search_reason = descriptor_search_unavailable(descriptor)
            aliases = sorted(
                alias
                for alias, target in registry.aliases.items()
                if target == descriptor.id
            )
            entries.append(
                {
                    "sha256": descriptor.id,
                    "name": Opponent(descriptor).name,
                    "kind": opponent_kind(record),
                    "producer": descriptor.provider,
                    "group": descriptor.group or descriptor.provider,
                    "details": record["details"],
                    "available": descriptor.incompatibility is None,
                    "reason": descriptor.incompatibility,
                    "supports_search": search_reason is None,
                    "search_reason": search_reason,
                    "recommended": descriptor.recommended,
                    "summary": descriptor.summary,
                    "trained_settings": descriptor.trained_settings,
                    "aliases": aliases,
                    "policy": record,
                }
            )
    except (ImportError, OSError, ValueError) as exc:
        error = f"Policies could not be listed: {exc}"
    return {
        "opponents": entries,
        "builtins": list(BUILTIN_OPPONENTS),
        "default_opponent": default_id,
        "catalog_error": error,
        "advice_available": advice.FIXTURE_PATH.is_file(),
        "extension": ", ".join(provider_names) or None,
        "watch_players": ["random", "search"],
    }


def catalog_opponent(key: str) -> Opponent | None:
    try:
        descriptor = policy_registry().descriptor(key)
    except ValueError:
        matches = [
            d
            for d in policy_registry().descriptors()
            if d.artifacts.get("actor") == key
        ]
        if len(matches) != 1:
            return None
        descriptor = matches[0]
    if descriptor.incompatibility:
        raise ValueError(
            f"{descriptor.name} cannot play here: {descriptor.incompatibility}"
        )
    return Opponent(descriptor)
