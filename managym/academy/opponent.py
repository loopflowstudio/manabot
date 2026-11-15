"""Server-scoped selectable policies. Providers own artifact discovery and admission.

The historical identity envelope (name/producer/sha256) remains readable by
existing tables and traces; new records bind the complete runtime descriptor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from managym.mirror_search import (
    SearchSpec,
    descriptor_search_unavailable,
    searched_descriptor,
)
from managym.policy import PolicyDescriptor, PolicyRegistry

from . import advice

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
        aliases = sorted(
            alias
            for alias, key in policy_registry().aliases.items()
            if key == self.descriptor.id
            and alias.casefold() != self.descriptor.name.casefold()
        )
        return " · ".join((self.descriptor.name, *aliases))

    def identity(self, search: SearchSpec | None = None) -> dict[str, str]:
        descriptor = (
            self.descriptor
            if search is None
            else searched_descriptor(self.descriptor, search)
        )
        return {
            "name": descriptor.name,
            "producer": descriptor.provider,
            "sha256": descriptor.id,
            "bot_id": f"policy:{descriptor.id}",
            **({"base_sha256": self.descriptor.id} if search is not None else {}),
        }

    def record(self, search: SearchSpec | None = None) -> dict[str, Any]:
        descriptor = (
            self.descriptor
            if search is None
            else searched_descriptor(self.descriptor, search)
        )
        return {
            **self.identity(search),
            "schema": "managym-policy-1",
            "policy": descriptor.record(),
        }


def opponent_catalog() -> dict[str, Any]:
    entries = []
    error = None
    provider_names: list[str] = []
    try:
        registry = policy_registry()
        provider_names = sorted(registry.providers)
        error = "; ".join(registry.discovery_errors) or None
        for descriptor in registry.descriptors():
            record = descriptor.record()
            entries.append(
                {
                    "sha256": descriptor.id,
                    "name": Opponent(descriptor).name,
                    "producer": descriptor.provider,
                    "group": descriptor.group or descriptor.provider,
                    "details": record["details"],
                    "available": descriptor.incompatibility is None,
                    "reason": descriptor.incompatibility,
                    "supports_search": descriptor_search_unavailable(descriptor)
                    is None,
                    "search_reason": descriptor_search_unavailable(descriptor),
                    "policy": record,
                }
            )
    except (ImportError, OSError, ValueError) as exc:
        error = f"Policies could not be listed: {exc}"
    return {
        "opponents": entries,
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
