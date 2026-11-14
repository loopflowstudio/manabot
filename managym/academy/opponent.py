"""Server-scoped selectable policies. Providers own artifact discovery and admission.

The historical identity envelope (name/producer/sha256) remains readable by
existing tables and traces; new records bind the complete runtime descriptor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from managym.policy import PolicyDescriptor, PolicyRegistry

from . import advice

_registry: PolicyRegistry | None = None


def policy_registry() -> PolicyRegistry:
    global _registry
    if _registry is None:
        _registry = PolicyRegistry()
        # Default discovery is opt-in at each installed provider, including its
        # own environment configuration. Core knows no model directory layout.
        for name, provider in _registry.providers.items():
            defaults = getattr(provider, "defaults", None)
            if defaults is not None:
                try:
                    for factory in defaults():
                        alias = factory.describe().metadata.get("alias")
                        _registry.register(
                            factory, name=alias if isinstance(alias, str) else None
                        )
                except (ImportError, OSError, ValueError) as exc:
                    _registry.discovery_errors.append(f"{name}: {exc}")
    return _registry


def set_policy_registry(registry: PolicyRegistry | None) -> None:
    global _registry
    _registry = registry


@dataclass(frozen=True)
class Opponent:
    descriptor: PolicyDescriptor

    @property
    def name(self) -> str:
        return self.descriptor.name

    def identity(self) -> dict[str, str]:
        return {
            "name": self.name,
            "producer": self.descriptor.provider,
            "sha256": self.descriptor.id,
            "bot_id": f"policy:{self.descriptor.id}",
        }

    def record(self) -> dict[str, Any]:
        return {
            **self.identity(),
            "schema": "managym-policy-1",
            "policy": self.descriptor.record(),
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
            metadata = descriptor.metadata
            entries.append(
                {
                    "sha256": descriptor.id,
                    "name": descriptor.name,
                    "producer": descriptor.provider,
                    "source": metadata.get("source", "provider"),
                    **{
                        key: metadata.get(key)
                        for key in (
                            "world",
                            "model",
                            "parameters",
                            "step",
                            "run",
                            "run_id",
                            "stage",
                            "weights",
                            "updates",
                            "transitions",
                        )
                    },
                    "setups": metadata.get("setups", []),
                    "available": descriptor.incompatibility is None,
                    "reason": descriptor.incompatibility,
                    "supports_search": descriptor.supports_search,
                    "policy": descriptor.record(),
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
