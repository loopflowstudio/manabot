"""Root installation on a buffered training vector.

The vector restores the retained setup and committed actions (or a hot handle),
checks the resulting game digest, and publishes the chooser into its execution
buffers. Every installation must advance the slot generation.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, Self, runtime_checkable

import numpy as np

from manabot.env.failures import CapturedVector
from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
from manabot.infra.config import MatchConfig
from manabot.training.rollout.producers import (
    BufferedStreams,
    ObservationRead,
    StreamShard,
)
import managym
from managym.recording import GameInputs, RecordingOptions


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
class CapturedSnapshot:
    slot: int
    recipe: SnapshotRecipe
    hot_handle: object | None
    root_digest: str
    root_chooser: int


@runtime_checkable
class SnapshotVector(Protocol):
    """Capture returns one owned handle per slot. Release relinquishes those references."""

    def capture_slots(self, slots: Sequence[int]) -> list[CapturedSnapshot]: ...

    def release_handles(self, handles: Sequence[object]) -> None: ...

    def install_slots(self, installs: Sequence[SlotInstall]) -> list[InstalledSlot]: ...

    def write_observations_for_players(
        self,
        players: list[int],
        buffers: dict[str, np.ndarray],
        action_overflow: Literal["error", "first-legal"] = "error",
        *,
        layout: managym.ObservationLayout,
    ) -> Sequence[ObservationRead | managym.ObservationRead]: ...


SnapshotVectorFactory = Callable[[ObservationSpace, Match, int, int], SnapshotVector]


class InMemorySnapshotVector(BufferedStreams):
    """Serial reference producer with independently resettable one-slot vectors.

    Each slot parks its fresh native source during replay. A hot handle keeps
    the resolved native setup; both paths replay the committed action prefix.
    Slot generations advance on native resets and when a parked source resumes.
    Only fresh sources write game records and statistics: a continuation
    replays a prefix its fresh game already recorded, so it is not a game.
    """

    @classmethod
    def factory(
        cls, observation_space: ObservationSpace, match: Match, num_envs: int, seed: int
    ) -> Self:
        # open_collector supplies the resolved match at its ordinary reset boundary.
        return cls(observation_space, num_envs, seed)

    def __init__(
        self,
        observation_space: ObservationSpace,
        num_envs: int,
        seed: int,
    ) -> None:
        memory = observation_space.config

        def streams(recording: RecordingOptions) -> list[StreamShard]:
            return [
                StreamShard(
                    i,
                    i + 1,
                    CapturedVector(
                        num_envs=1,
                        seed=min(seed + i, 2**64 - 1),
                        seed_stride=num_envs,
                        stream_offset=i,
                        decision_memory=memory.max_decision_records > 0,
                        decision_memory_include_own=memory.decision_memory_include_own,
                        player_policies=("agent", "agent"),
                        **recording,
                    ),
                )
                for i in range(num_envs)
            ]

        self._generation_bumps = [0] * num_envs
        self._fresh = streams({})
        self._replay = streams({"record_fraction": 0.0, "statistics_path": None})
        super().__init__(num_envs, self._fresh.copy())

    def set_buffers(
        self,
        buffers: dict[str, np.ndarray],
        action_overflow: Literal["error", "first-legal"] = "error",
        *,
        layout: managym.ObservationLayout,
    ) -> None:
        super().set_buffers(buffers, action_overflow, layout=layout)
        self._buffers = buffers
        self._layout = layout
        self._action_overflow = action_overflow
        for slot, active in enumerate(self._shards):
            shard = (
                self._fresh[slot]
                if active is self._replay[slot]
                else self._replay[slot]
            )
            shard.native.set_buffers(
                {k: v[shard.start : shard.stop] for k, v in buffers.items()},
                action_overflow,
                layout=layout,
            )

    def capture_slots(self, slots: Sequence[int]) -> list[CapturedSnapshot]:
        captures = []
        for slot in slots:
            if not 0 <= slot < self.num_envs:
                raise ValueError(f"invalid capture slot {slot}")
            live = self._shards[slot].native
            [digest] = live.native.state_digests()
            [chooser] = live.current_agent_indices()
            if digest is None or chooser not in (0, 1):
                raise ValueError(f"slot {slot} has no decision to capture")
            game = live.games[0]
            recipe = SnapshotRecipe.from_inputs(game, digest, chooser)
            captures.append(
                CapturedSnapshot(slot, recipe, game.setup(), digest, chooser)
            )
        return captures

    def release_handles(self, handles: Sequence[object]) -> None:
        """Native setup references are owned and released by their Python holders."""

    def install_slots(self, installs: Sequence[SlotInstall]) -> list[InstalledSlot]:
        results = []
        for request in installs:
            slot, recipe = request.slot, request.recipe
            if not 0 <= slot < self.num_envs:
                raise ValueError(
                    f"snapshot {request.snapshot_id} has invalid slot {slot}"
                )
            shard = (self._fresh if recipe is None else self._replay)[slot]
            live = shard.native
            if recipe is None and self._shards[slot] is self._replay[slot]:
                # Replay overwrote the shared buffers. The parked fresh game
                # still owns its original seed, state and recording.
                result_keys = ("terminated", "truncated", "rewards")
                live.write_observations_for_players(
                    live.current_agent_indices(),
                    {
                        k: v[slot : slot + 1]
                        for k, v in self._buffers.items()
                        if k not in result_keys
                    },
                    self._action_overflow,
                    layout=self._layout,
                )
                for key in result_keys:
                    self._buffers[key][slot] = 0
                self._generation_bumps[slot] += 1
            elif recipe is None:
                live.reset_seeds_into_buffers(live.current_game_seeds())
            else:
                game = live.games[0]
                try:
                    if (game.decision_memory, game.decision_memory_include_own) != (
                        recipe.decision_memory,
                        recipe.decision_memory_include_own,
                    ):
                        raise ValueError("decision memory settings differ")
                    setup = (
                        request.hot_handle
                        if isinstance(request.hot_handle, managym.GameSetup)
                        else Match(recipe.match).to_rust()
                    )
                    live.reset_all_into_buffers(setup)
                    live.reset_seeds_into_buffers([recipe.seed])
                    generation = live.current_game_generations()
                    for action in recipe.actions:
                        live.step_into_buffers([action])
                    if live.current_game_generations() != generation:
                        raise ValueError("action prefix crossed a game terminal")
                except Exception as exc:
                    raise ValueError(
                        f"snapshot {request.snapshot_id} installation failed: {exc}"
                    ) from exc
            [digest] = live.native.state_digests()
            [chooser] = live.current_agent_indices()
            if digest is None or (
                recipe is not None
                and (digest, chooser) != (recipe.root_digest, recipe.root_chooser)
            ):
                raise ValueError(
                    f"snapshot {request.snapshot_id} root identity mismatch in slot {slot}"
                )
            self._shards[slot] = shard
            results.append(InstalledSlot(slot, digest, chooser))
        return results

    def current_game_generations(self) -> list[int]:
        return [
            a.native.current_game_generations()[0]
            + b.native.current_game_generations()[0]
            + bump
            for a, b, bump in zip(
                self._fresh, self._replay, self._generation_bumps, strict=True
            )
        ]

    def write_observations_for_players(
        self,
        players: list[int],
        buffers: dict[str, np.ndarray],
        action_overflow: Literal["error", "first-legal"] = "error",
        *,
        layout: managym.ObservationLayout,
    ) -> list[ObservationRead]:
        reads = super().write_observations_for_players(
            players, buffers, action_overflow, layout=layout
        )
        return [
            ObservationRead(r.slot, r.player, generation, r.seed, r.revision)
            for r, generation in zip(
                reads, self.current_game_generations(), strict=True
            )
        ]

    def close(self) -> None:
        if not self._closed:
            for fresh, replay, active in zip(
                self._fresh, self._replay, self._shards, strict=True
            ):
                (replay if active is fresh else fresh).native.close()
            super().close()
