"""Optional bot capabilities, discovered from installed distributions.

The single provider answers in engine types (managym.bots) and the shapes
declared here; Academy has no training imports.
Installing manabot adds its entry point without application configuration.
"""

from collections.abc import Sequence
from functools import cache
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import managym
from managym.bots import BotPolicy, VerifiedArena


class CheckpointSummary(Protocol):
    """Catalog metadata for one trained checkpoint."""

    path: Path
    sha256: str
    label: str
    source: Literal["path", "runs", "tournament"]
    producer: str
    world: str | None
    model: str
    parameters: int
    step: int | None
    setups: tuple[managym.Deck, ...]
    incompatibility: str | None

    @property
    def compatible(self) -> bool: ...


class UnavailableEntry(Protocol):
    sha256: str
    label: str
    producer: str
    reason: str


class CheckpointRegistration(Protocol):
    checkpoint_id: str
    checkpoint_sha256: str
    checkpoint_bytes: int
    manifest_sha256: str
    observation_abi_sha256: str
    action_abi_sha256: str
    value_mode: str
    world_id: str


@runtime_checkable
class BotExtension(Protocol):
    name: str
    # Additional player kinds; Academy owns the native players.
    watch_players: tuple[str, ...]

    def discover(
        self, paths: list[Path], runs_dir: Path
    ) -> tuple[Sequence[CheckpointSummary], Sequence[UnavailableEntry]]: ...
    def checkpoint_policy(
        self,
        path: str,
        *,
        deterministic: bool,
        player_configs: Sequence[managym.PlayerConfig],
    ) -> BotPolicy: ...
    def watch_policy(self, kind: str) -> BotPolicy: ...
    def checkpoint_registration(self, seed: int) -> CheckpointRegistration: ...
    def verify_arena(self, record: dict[str, Any]) -> VerifiedArena: ...


@cache
def bot_extension() -> BotExtension | None:
    providers = list(entry_points(group="managym.academy.bots"))
    if len(providers) > 1:
        raise RuntimeError(
            "Multiple Academy bot extensions: " + ", ".join(p.name for p in providers)
        )
    if not providers:
        return None
    provider = providers[0].load()
    if not isinstance(provider, BotExtension):
        raise TypeError(f"Invalid Academy bot extension: {providers[0].name}")
    return provider
