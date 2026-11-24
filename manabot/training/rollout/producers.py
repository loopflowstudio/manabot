"""Persistent CPU observation producers for one GPU-owned collector.

ShardedVectorEnv gives each worker disjoint native streams and NumPy row views.
Native buffered calls release the GIL; a barrier completes every shard before
returning to the collector. The parent retains global batching, policy sampling,
rollout ordering and recovery. Global reset strides make stream identity
independent of worker count. Sampled rosters (deck sources, elemental-uniform)
draw each game's pair natively from that game's seed, so draws match for any
worker count. No policy or CUDA state enters a producer thread.
"""

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from manabot.env.failures import CapturedVector
from manabot.training.rollout.snapshot_vector import (
    CapturedSnapshot,
    InstalledSlot,
    SlotInstall,
    SnapshotRecipe,
    bind_cold_start,
)
import managym
from managym.recording import GameInputs

Array = NDArray[np.generic]


@dataclass(frozen=True)
class StreamShard:
    start: int
    stop: int
    native: CapturedVector

    @property
    def rows(self) -> slice:
        return slice(self.start, self.stop)


@dataclass(frozen=True)
class ObservationRead:
    """A fixed-seat read in the producer's global coordinates."""

    slot: int
    player: int
    game_id: int
    seed: int
    revision: int


def native_slot_load(request: SlotInstall, local_slot: int) -> managym.SlotLoad:
    source, recipe = request.hot_handle, request.recipe
    if recipe is None:
        if source is not None or request.restore_options is not None:
            raise ValueError("fresh slot cannot carry a snapshot or restore options")
        return managym.SlotLoad.fresh(local_slot)
    if not isinstance(source, managym.GameSnapshot) or (
        source.state_digest,
        source.seed,
        source.acting_player,
    ) != (recipe.root_digest, recipe.seed, recipe.root_chooser):
        raise ValueError(
            f"snapshot {request.snapshot_id} root identity mismatch in slot {request.slot}"
        )
    return managym.SlotLoad.snapshot(local_slot, source, request.restore_options)


class BufferedStreams:
    """Present disjoint native shards and their buffer views as one vector."""

    skip_trivial = True

    def __init__(self, num_envs: int, shards: list[StreamShard]) -> None:
        self.num_envs = num_envs
        self._shards = shards
        self._closed = False

    def _run[T](self, calls: Sequence[Callable[[], T]]) -> list[T]:
        self._check()
        return [call() for call in calls]

    def _check(self) -> None:
        if self._closed:
            raise RuntimeError("producer pool is closed")

    def _rows(self, count: int) -> None:
        self._check()
        if count != self.num_envs:
            raise ValueError("producer row count differs from global streams")

    def set_buffers(
        self,
        buffers: dict[str, Array],
        action_overflow: str = "error",
        *,
        layout: managym.ObservationLayout,
    ) -> None:
        self._check()
        for value in buffers.values():
            self._rows(len(value))
            if not value.flags.c_contiguous or not value.flags.writeable:
                raise ValueError("producer buffers must be writable and contiguous")
        self._run(
            [
                partial(
                    shard.native.set_buffers,
                    {key: value[shard.rows] for key, value in buffers.items()},
                    action_overflow,
                    layout=layout,
                )
                for shard in self._shards
            ]
        )

    def write_observations_for_players(
        self,
        players: Sequence[int | None],
        buffers: dict[str, Array],
        action_overflow: Literal["error", "first-legal"] = "error",
        *,
        layout: managym.ObservationLayout,
    ) -> list[ObservationRead]:
        """Project native seats into disjoint row slices and return global identities."""
        self._rows(len(players))
        for value in buffers.values():
            self._rows(len(value))
        selected = [
            shard
            for shard in self._shards
            if any(player is not None for player in players[shard.rows])
        ]
        reads = self._run(
            [
                partial(
                    shard.native.write_observations_for_players,
                    players[shard.rows],
                    {key: value[shard.rows] for key, value in buffers.items()},
                    action_overflow,
                    layout=layout,
                )
                for shard in selected
            ]
        )
        return [
            ObservationRead(
                shard.start + read.slot,
                read.player,
                read.game_id,
                read.seed,
                read.revision,
            )
            for shard, local in zip(selected, reads, strict=True)
            for read in local
        ]

    # The two roster setters run serially, before reset: native admission is
    # input-determined, so a rejection comes from the first shard and leaves
    # every shard unchanged.

    def set_uniform_roster(self, roster: list[managym.PlayerConfig]) -> None:
        self._check()
        for shard in self._shards:
            shard.native.set_uniform_roster(roster)

    def set_training_decks(
        self,
        decks: list[managym.PlayerConfig],
        digest: str,
        sampling: Literal[
            "uniform-distinct-v1", "uniform-independent-v1"
        ] = "uniform-distinct-v1",
    ) -> None:
        """Replace later roots' decks on every shard."""
        self._check()
        for shard in self._shards:
            shard.native.set_training_decks(decks, digest, sampling)

    def _read_rows[T](self, read: Callable[[CapturedVector], list[T]]) -> list[T]:
        self._check()
        return [row for shard in self._shards for row in read(shard.native)]

    def current_player_configs(self) -> list[managym.GameSetup]:
        """Return each current game with resolved open/closed disclosure."""
        return self._read_rows(lambda native: native.current_player_configs())

    def current_matchup_indices(self) -> list[tuple[int, int]]:
        pairs = self._read_rows(lambda native: native.current_matchup_indices())
        return pairs if len(pairs) == self.num_envs else []

    def reset_all_into_buffers(self, players: managym.GameSetup) -> None:
        self._run(
            [
                partial(shard.native.reset_all_into_buffers, players)
                for shard in self._shards
            ]
        )

    def reset_each_into_buffers(self, players: list[managym.GameSetup]) -> None:
        self._rows(len(players))
        self._run(
            [
                partial(
                    shard.native.reset_each_into_buffers,
                    players[shard.rows],
                )
                for shard in self._shards
            ]
        )

    def reset_seeds_into_buffers(self, seeds: list[int]) -> None:
        self._rows(len(seeds))
        self._run(
            [
                partial(
                    shard.native.reset_seeds_into_buffers,
                    seeds[shard.rows],
                )
                for shard in self._shards
            ]
        )

    def step_into_buffers(
        self,
        actions: list[int],
        active: list[bool] | None = None,
        *,
        annotations: list[dict[str, object] | None] | None = None,
    ) -> None:
        self._rows(len(actions))
        if active is not None:
            self._rows(len(active))
        self._run(
            [
                partial(
                    shard.native.step_into_buffers,
                    actions[shard.rows],
                    None if active is None else active[shard.rows],
                    annotations=None
                    if annotations is None
                    else annotations[shard.rows],
                )
                for shard in self._shards
            ]
        )

    def recording_mask(self) -> list[bool]:
        return [
            selected
            for shard in self._shards
            for selected in shard.native.recording_mask()
        ]

    def current_action_summaries(self) -> list[tuple[int, str]]:
        return [
            summary
            for shard in self._shards
            for summary in shard.native.current_action_summaries()
        ]

    def current_agent_indices(self) -> list[int]:
        return self._read_rows(lambda native: native.current_agent_indices())

    def step_isolated_into_buffers(
        self,
        actions: list[int],
        active: list[bool],
        recover_capacity: bool = False,
        *,
        annotations: list[dict[str, object] | None] | None = None,
    ) -> None:
        """Gameplay errors are row outcomes; infrastructure errors still stop."""
        self._rows(len(actions))
        self._rows(len(active))
        self._run(
            [
                partial(
                    shard.native.step_isolated_into_buffers,
                    actions[shard.rows],
                    active[shard.rows],
                    recover_capacity,
                    annotations=None
                    if annotations is None
                    else annotations[shard.rows],
                )
                for shard in self._shards
            ]
        )

    def replace_failed_into_buffers(self, failed: list[bool]) -> None:
        self._rows(len(failed))
        self._run(
            [
                partial(
                    shard.native.replace_failed_into_buffers,
                    failed[shard.rows],
                )
                for shard in self._shards
            ]
        )

    def current_game_seeds(self) -> list[int]:
        return self._read_rows(lambda native: native.current_game_seeds())

    def next_fresh_seeds(self) -> list[int]:
        return self._read_rows(lambda native: native.next_fresh_seeds())

    def _read_slots[T](
        self,
        mask: Sequence[bool],
        read: Callable[[CapturedVector, Sequence[bool]], list[T | None]],
    ) -> list[T | None]:
        self._rows(len(mask))
        selected = [shard for shard in self._shards if any(mask[shard.rows])]
        batches = self._run(
            [partial(read, shard.native, mask[shard.rows]) for shard in selected]
        )
        result: list[T | None] = [None] * self.num_envs
        for shard, rows in zip(selected, batches, strict=True):
            result[shard.rows] = rows
        return result

    def snapshot_slots(self, mask: Sequence[bool]) -> list[managym.GameSnapshot | None]:
        return self._read_slots(
            mask, lambda native, selected: native.snapshot_slots(selected)
        )

    def slot_metadata(
        self, mask: Sequence[bool]
    ) -> list[managym.SnapshotMetadata | None]:
        return self._read_slots(
            mask, lambda native, selected: native.slot_metadata(selected)
        )

    def _validate_slots(self, slots: Sequence[int]) -> set[int]:
        self._check()
        selected = set(slots)
        if len(selected) != len(slots) or any(
            slot < 0 or slot >= self.num_envs for slot in slots
        ):
            raise ValueError("invalid or duplicate snapshot slots")
        return selected

    def capture_slots(self, slots: Sequence[int]) -> list[CapturedSnapshot]:
        """Freeze only requested decisions, preserving caller order across shards."""
        selected = self._validate_slots(slots)
        handles = self.snapshot_slots([i in selected for i in range(self.num_envs)])
        captures = []
        for slot in slots:
            source = handles[slot]
            assert source is not None
            chooser = source.acting_player
            if chooser not in (0, 1):
                raise ValueError(f"slot {slot} has no decision to capture")
            recipe = SnapshotRecipe.from_replay_json(source.to_replay_json(), chooser)
            captures.append(
                CapturedSnapshot(slot, recipe, source, source.state_digest, chooser)
            )
        return captures

    def release_handles(self, handles: Sequence[object]) -> None:
        """Detached native snapshots are released by their last Python owner."""

    def install_slots(self, installs: Sequence[SlotInstall]) -> list[InstalledSlot]:
        """Validate global source bindings, then submit one mixed load per active shard."""
        slots = [r.slot for r in installs]
        self._validate_slots(slots)
        replayed: dict[int, tuple[SnapshotRecipe, managym.GameSnapshot]] = {}
        calls = []
        offsets = []
        for shard in self._shards:
            loads = [
                native_slot_load(
                    bind_cold_start(request, replayed), request.slot - shard.start
                )
                for request in installs
                if shard.start <= request.slot < shard.stop
            ]
            if loads:
                calls.append(partial(shard.native.load_slots_into_buffers, loads))
                offsets.append(shard.start)
        results = {
            offset + result.slot: InstalledSlot(
                offset + result.slot,
                result.source_digest or "",
                result.acting_player if result.acting_player is not None else -1,
            )
            for offset, batch in zip(offsets, self._run(calls), strict=True)
            for result in batch
        }
        return [results[slot] for slot in slots]

    def current_game_generations(self) -> list[int]:
        return self._read_rows(lambda native: native.current_game_generations())

    def decision_memory_cursors(self) -> list[list[list[int]]]:
        return self._read_rows(lambda native: native.decision_memory_cursors())

    def decision_memory_counts(self) -> list[list[list[int]]]:
        return self._read_rows(lambda native: native.decision_memory_counts())

    def get_last_info(self) -> list[dict[str, Any]]:
        return self._read_rows(lambda native: native.get_last_info())

    def buffered_call_timings(self) -> tuple[float, float]:
        """Mean shard stepping and encoding seconds: shards run at once."""
        shards = [shard.native.buffered_call_timings() for shard in self._shards]
        return (
            sum(stepping for stepping, _ in shards) / len(shards),
            sum(encoding for _, encoding in shards) / len(shards),
        )

    @property
    def auto_reset(self) -> bool:
        return self._shards[0].native.auto_reset

    def game_inputs(self, slot: int) -> GameInputs:
        """Materialize one failed game's replay on the producer that owns it."""
        for shard in self._shards:
            if shard.start <= slot < shard.stop:
                return shard.native.game_inputs(slot - shard.start)
        raise IndexError(slot)

    def capture_failure(self, error: BaseException) -> None:
        # Capture from the owning thread so live run coordinates remain available.
        for shard in self._shards:
            shard.native.capture_failure(error)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            for shard in self._shards:
                shard.native.close()
            self._shards.clear()


class ShardedVectorEnv(BufferedStreams):
    """Coarse native producers with a synchronous, globally ordered interface.

    The caller owns the arrays exclusively during every method. Native calls are
    cooperative: an outer process supervisor supplies a hard execution deadline.
    On a worker failure, all submitted calls settle before the pool is closed and
    the earliest shard error is raised. Partial buffers must then be discarded.
    Call close at the end of the owning training run, including failure paths.
    """

    def __init__(
        self,
        num_envs: int,
        seed: int,
        workers: int,
        *,
        auto_reset: bool = True,
        decision_memory_include_own: bool = False,
        decision_memory: bool = False,
        games_path: Path | None = None,
        record_fraction: float = 1.0,
    ) -> None:
        if workers < 2 or workers > num_envs:
            raise ValueError(
                "parallel producers require at least 2 workers and at least one stream each"
            )
        if not 0 <= seed < 2**64:
            raise ValueError("native seed must fit uint64")
        self.workers = workers
        shards = []
        for index in range(workers):
            start = index * num_envs // workers
            stop = (index + 1) * num_envs // workers
            native = CapturedVector(
                num_envs=stop - start,
                seed=min(seed + start, 2**64 - 1),
                seed_stride=num_envs,
                stream_offset=start,
                skip_trivial=self.skip_trivial,
                auto_reset=auto_reset,
                decision_memory=decision_memory,
                decision_memory_include_own=decision_memory_include_own,
                player_policies=("agent", "agent"),
                **(
                    {
                        "games_path": games_path.with_name(
                            f"{games_path.stem}-{index}.jsonl"
                        ),
                        "record_fraction": record_fraction,
                    }
                    if games_path is not None
                    else {}
                ),
            )
            shards.append(StreamShard(start, stop, native))
        super().__init__(num_envs, shards)
        self._pool = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="manabot-producer"
        )

    def _run[T](self, calls: Sequence[Callable[[], T]]) -> list[T]:
        self._check()
        try:
            # Freeze owner coordinates before dispatch; ContextVars do not cross threads.
            for shard in self._shards:
                if shard.native.scope is not None and shard.native.scope.detailed:
                    shard.native.frozen_context = shard.native.scope.context().copy()
            futures = [self._pool.submit(call) for call in calls]
            # Settle every writer before propagating failure or exposing buffers.
            wait(futures)
            return [future.result() for future in futures]
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)
        super().close()


def open_native_streams(
    num_envs: int,
    seed: int,
    workers: int,
    *,
    auto_reset: bool,
    decision_memory: bool,
    decision_memory_include_own: bool,
    games_path: Path | None = None,
    record_fraction: float = 1.0,
) -> BufferedStreams | CapturedVector:
    """Select the collector transport while keeping fresh recording on its native slots."""
    if workers != 1:
        return ShardedVectorEnv(
            num_envs,
            seed,
            workers,
            auto_reset=auto_reset,
            decision_memory=decision_memory,
            decision_memory_include_own=decision_memory_include_own,
            games_path=games_path,
            record_fraction=record_fraction,
        )
    native = CapturedVector(
        num_envs=num_envs,
        seed=seed,
        auto_reset=auto_reset,
        decision_memory=decision_memory,
        decision_memory_include_own=decision_memory_include_own,
        skip_trivial=True,
        player_policies=("agent", "agent"),
        **(
            {"games_path": games_path, "record_fraction": record_fraction}
            if games_path is not None
            else {}
        ),
    )
    return (
        native
        if auto_reset
        else BufferedStreams(num_envs, [StreamShard(0, num_envs, native)])
    )
