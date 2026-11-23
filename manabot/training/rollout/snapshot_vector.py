"""Root installation on a buffered training vector.

The vector restores the retained setup and committed actions (or a hot handle),
checks the resulting game digest, and publishes the chooser into its execution
buffers. Every installation must start a new native game generation.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from manabot.infra.config import MatchConfig
import managym
from managym.recording import GameInputs


@dataclass(frozen=True)
class SnapshotRecipe:
    match: MatchConfig
    seed: int
    actions: tuple[int, ...]
    root_digest: str
    root_chooser: Literal[0, 1]
    decision_memory: bool
    decision_memory_include_own: bool

    @classmethod
    def from_inputs(
        cls, game: GameInputs, root_digest: str, root_chooser: Literal[0, 1]
    ) -> "SnapshotRecipe":
        """Bind only the committed self-play prefix and its resolved native setup."""
        prefix = game.actions[: game.committed]
        if game.committed is not None and game.committed > len(game.actions):
            raise ValueError("snapshot inputs missing committed actions")
        actions = tuple(action for action in prefix if isinstance(action, int))
        if len(actions) != len(prefix):
            raise ValueError("snapshot inputs require externally chosen actions")
        setup = game.setup()
        return cls(
            MatchConfig.from_players(
                *setup,
                decklists=setup.decklists,
                mana_floats=setup.mana_floats,
                mana_payment=setup.mana_payment,
            ),
            game.seed,
            actions,
            root_digest,
            root_chooser,
            game.decision_memory,
            game.decision_memory_include_own,
        )


@dataclass(frozen=True)
class SnapshotBranch:
    snapshot_id: int
    recipe: SnapshotRecipe
    horizon: int | None = None
    branch_group_id: int = -1
    branch_index: int = 0
    sample_weight: float = 1.0
    hot_handle: object | None = None

    def __post_init__(self) -> None:
        if self.snapshot_id < 0 or self.branch_group_id < -1 or self.branch_index < 0:
            raise ValueError("invalid snapshot or branch identity")
        if self.horizon is not None and self.horizon < 1:
            raise ValueError("snapshot horizon must be positive")
        if not np.isfinite(self.sample_weight) or self.sample_weight < 0:
            raise ValueError("sample_weight must be finite and nonnegative")


@dataclass(frozen=True)
class SlotInstall:
    """A missing recipe starts a game from the vector's configured fresh source."""

    slot: int
    snapshot_id: int
    recipe: SnapshotRecipe | None
    hot_handle: object | None = None


@dataclass(frozen=True)
class InstalledSlot:
    slot: int
    root_digest: str
    root_chooser: int


@dataclass(frozen=True)
class ObservationRead:
    """A fixed-seat read with the slot in the producer's global coordinates."""

    slot: int
    player: int
    game_id: int
    seed: int
    revision: int


@runtime_checkable
class SnapshotVector(Protocol):
    """Installation and viewer reads in addition to ordinary buffered stepping."""

    def install_slots(self, installs: Sequence[SlotInstall]) -> list[InstalledSlot]: ...

    def write_observations_for_players(
        self,
        players: list[int],
        buffers: dict[str, np.ndarray],
        action_overflow: Literal["error", "first-legal"] = "error",
        *,
        layout: managym.ObservationLayout,
    ) -> Sequence[ObservationRead | managym.ObservationRead]: ...
