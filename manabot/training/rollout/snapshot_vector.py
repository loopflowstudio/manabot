"""Root installation on a buffered training vector.

The vector restores the retained setup and committed actions (or a hot handle),
checks the resulting game digest, and publishes the chooser into its execution
buffers. Every installation must advance the slot generation.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import json
from typing import TYPE_CHECKING, Annotated, Literal, Protocol, runtime_checkable
import warnings

import numpy as np
from pydantic import Field

from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
from manabot.infra.config import MatchConfig
import managym
from managym.recording import GameInputs, record_from_row

if TYPE_CHECKING:
    from manabot.training.rollout.producers import BufferedStreams, ObservationRead
    from managym.recording import RecordedVectorEnv


@dataclass(frozen=True)
class SnapshotRecipe:
    match: MatchConfig
    seed: int
    actions: tuple[int | str, ...]
    root_digest: str
    root_chooser: Literal[0, 1]
    decision_memory: bool
    decision_memory_include_own: bool
    replay_json: Annotated[
        str | None, Field(exclude_if=lambda value: value is None)
    ] = None

    @classmethod
    def from_inputs(
        cls, game: GameInputs, root_digest: str, root_chooser: Literal[0, 1]
    ) -> "SnapshotRecipe":
        """Bind only the committed self-play prefix and its resolved native setup."""
        if game.interventions or game.object_candidate_frames:
            raise ValueError(
                "snapshot interventions and object frames require native replay JSON"
            )
        prefix = game.actions[: game.committed]
        if game.committed is not None and game.committed > len(game.actions):
            raise ValueError("snapshot inputs missing committed actions")
        actions = tuple(action for action in prefix if isinstance(action, int))
        if len(actions) != len(prefix):
            raise ValueError("snapshot inputs require externally chosen actions")
        return cls._from_game(game, actions, root_digest, root_chooser, None)

    @classmethod
    def from_replay_json(cls, raw: str, chooser: Literal[0, 1]) -> "SnapshotRecipe":
        """Retain a native replay, including Commands and boundary interventions."""
        row = json.loads(raw)
        if not isinstance(row, dict) or row.get("schema") != "managym-snapshot-v1":
            raise ValueError("expected a managym-snapshot-v1 replay")
        [game] = record_from_row(row).games
        if game.end is None or game.end.digest is None or game.end.status == "complete":
            raise ValueError("snapshot requires a live decision")
        return cls._from_game(game, tuple(game.actions), game.end.digest, chooser, raw)

    @classmethod
    def _from_game(
        cls,
        game: GameInputs,
        actions: tuple[int | str, ...],
        digest: str,
        chooser: Literal[0, 1],
        raw: str | None,
    ) -> "SnapshotRecipe":
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
            digest,
            chooser,
            game.decision_memory,
            game.decision_memory_include_own,
            raw,
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
    restore_options: managym.RestoreOptions | None = None

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
    restore_options: managym.RestoreOptions | None = None


@dataclass(frozen=True)
class InstalledSlot:
    slot: int
    root_digest: str
    root_chooser: int


@dataclass(frozen=True)
class CapturedSnapshot:
    slot: int
    recipe: SnapshotRecipe
    hot_handle: object | None
    root_digest: str
    root_chooser: int


@runtime_checkable
class SnapshotVector(Protocol):
    """Capture returns one owned handle per slot. Release relinquishes those references.

    Vectors used with game-error recovery also supply game_inputs(slot) to
    materialize the failed branch's committed replay and attempted actions.
    """

    def capture_slots(self, slots: Sequence[int]) -> list[CapturedSnapshot]: ...

    def release_handles(self, handles: Sequence[object]) -> None: ...

    def install_slots(self, installs: Sequence[SlotInstall]) -> list[InstalledSlot]: ...

    def observations_for_players(
        self, players: Sequence[int | None]
    ) -> tuple[
        Sequence["ObservationRead | managym.ObservationRead"], managym.ObservationBatch
    ]: ...


SnapshotVectorFactory = Callable[[ObservationSpace, Match, int, int], SnapshotVector]


def replay_snapshot(recipe: SnapshotRecipe) -> managym.GameSnapshot:
    """Rebuild a detached source once from native or legacy positional history."""
    if recipe.replay_json is not None:
        return managym.GameSnapshot.from_replay_json(recipe.replay_json)
    env = managym.Env(
        seed=recipe.seed,
        decision_memory=recipe.decision_memory,
        decision_memory_include_own=recipe.decision_memory_include_own,
    )
    env.reset(Match(recipe.match).to_rust())
    for action in recipe.actions:
        if not isinstance(action, int):
            raise ValueError("legacy snapshot requires positional actions")
        _, _, ended, truncated, _ = env.step(action)
        if ended or truncated:
            raise ValueError("action prefix crossed a game terminal")
    return env.snapshot()


def replay_prefix(
    live: "RecordedVectorEnv", recipe: SnapshotRecipe, hot_handle: object | None
) -> None:
    """Reset a replay stream and apply the committed decisions of a single game."""
    setup = (
        hot_handle
        if isinstance(hot_handle, managym.GameSetup)
        else Match(recipe.match).to_rust()
    )
    replay_prefixes(live, [setup], [recipe])


def replay_prefixes(
    live: "RecordedVectorEnv | BufferedStreams",
    setups: list[managym.GameSetup],
    recipes: list[SnapshotRecipe],
) -> None:
    """Install roots and advance each stream only through its committed prefix."""
    for game, recipe in zip(live.games, recipes, strict=True):
        if (game.decision_memory, game.decision_memory_include_own) != (
            recipe.decision_memory,
            recipe.decision_memory_include_own,
        ):
            raise ValueError("decision memory settings differ")
    live.reset_each_into_buffers(setups, seeds=[r.seed for r in recipes])
    generation = live.current_game_generations()
    prefixes = [r.actions for r in recipes]
    for offset in range(max(map(len, prefixes))):
        active = [offset < len(prefix) for prefix in prefixes]
        actions = [
            prefix[offset] if on else 0
            for prefix, on in zip(prefixes, active, strict=True)
        ]
        live.step_into_buffers(actions, active)
        if live.current_game_generations() != generation:
            raise ValueError("action prefix crossed a game terminal")


def bind_cold_start(
    request: SlotInstall,
    replayed: dict[int, tuple[SnapshotRecipe, managym.GameSnapshot]],
) -> SlotInstall:
    """Keep one replay per cold root in the caller's allocation."""
    recipe = request.recipe
    if recipe is None or request.hot_handle is not None:
        return request
    cached = replayed.get(request.snapshot_id)
    if cached is None:
        source = replay_snapshot(recipe)
        replayed[request.snapshot_id] = recipe, source
    else:
        existing, source = cached
        if existing is not recipe and existing != recipe:
            raise ValueError(f"snapshot {request.snapshot_id} has conflicting recipes")
    return replace(request, hot_handle=source)


class InMemorySnapshotVector:
    """Deprecated native one-worker factory, removed in 0.3.0."""

    @staticmethod
    def factory(
        observation_space: ObservationSpace, match: Match, num_envs: int, seed: int
    ) -> "BufferedStreams":
        return InMemorySnapshotVector(observation_space, num_envs, seed)

    def __new__(
        cls, observation_space: ObservationSpace, num_envs: int, seed: int
    ) -> "BufferedStreams":
        from manabot.training.rollout.producers import open_native_streams

        warnings.warn(
            "InMemorySnapshotVector is deprecated and will be removed in 0.3.0; "
            "use open_collector(..., snapshots=True)",
            DeprecationWarning,
            stacklevel=2,
        )
        memory = observation_space.config
        return open_native_streams(
            num_envs,
            seed,
            workers=1,
            decision_memory=memory.max_decision_records > 0,
            decision_memory_include_own=memory.decision_memory_include_own,
        )
