"""Optional bot capabilities, discovered from installed distributions.

Optional replay/advice extensions answer in engine types and the shapes
declared here; Academy has no training imports.
Installing manabot adds its entry point without application configuration.
"""

from functools import cache
from importlib.metadata import entry_points
from typing import Any, Protocol, runtime_checkable

from managym.bots import VerifiedArena


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

    def checkpoint_registration(self, seed: int) -> CheckpointRegistration: ...
    def verify_arena(self, record: dict[str, Any]) -> VerifiedArena: ...


@cache
def bot_extension() -> BotExtension | None:
    providers = sorted(entry_points(group="managym.academy.bots"), key=lambda p: p.name)
    if not providers:
        return None
    # These legacy capabilities are selected separately from playing policies.
    provider = providers[0].load()
    if not isinstance(provider, BotExtension):
        raise TypeError(f"Invalid Academy bot extension: {providers[0].name}")
    return provider
