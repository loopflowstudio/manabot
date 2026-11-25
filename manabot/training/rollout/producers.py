"""Persistent CPU observation producers for one GPU-owned collector.

BufferedStreams gives each worker disjoint native streams and outcome buffers.
V11 observation coordinates keep their native owners across the shard barrier.
Native buffered calls release the GIL; a barrier completes every shard before
returning to the collector. The parent retains global batching, policy sampling,
rollout ordering and recovery. Global reset strides make stream identity
independent of worker count. Sampled rosters (deck sources, elemental-uniform)
draw each game's pair natively from that game's seed, so draws match for any
worker count. No policy or CUDA state enters a producer thread.
"""

from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import partial
import operator
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from manabot.env.failures import RecordedVectorEnv
from manabot.training.rollout.snapshot_vector import (
    CapturedSnapshot,
    InstalledSlot,
    SlotInstall,
    SnapshotRecipe,
    bind_cold_start,
)
import managym
from managym.recording import GameInputs, load_recorded_slots

Array = NDArray[np.generic]


@dataclass(frozen=True)
class StreamShard:
    start: int
    stop: int
    native: RecordedVectorEnv

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


def native_slot_load(request: SlotInstall) -> managym.SlotLoad:
    source, recipe = request.hot_handle, request.recipe
    if recipe is None:
        if source is not None or request.restore_options is not None:
            raise ValueError("fresh slot cannot carry a snapshot or restore options")
        return managym.SlotLoad.fresh(request.slot)
    if not isinstance(source, managym.GameSnapshot) or (
        source.state_digest,
        source.seed,
        source.acting_player,
    ) != (recipe.root_digest, recipe.seed, recipe.root_chooser):
        raise ValueError(
            f"snapshot {request.snapshot_id} root identity mismatch in slot {request.slot}"
        )
    return managym.SlotLoad.snapshot(request.slot, source, request.restore_options)


class BufferedStreams:
    """Present disjoint native shards and their buffer views as one vector."""

    skip_trivial = True

    def __init__(self, num_envs: int, shards: list[StreamShard]) -> None:
        self.num_envs = num_envs
        self._shards = shards
        self._shard_by_slot = [
            shard for shard in shards for _ in range(shard.start, shard.stop)
        ]
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

    @property
    def games(self) -> list[GameInputs]:
        """Current replay inputs in global slot order."""
        return [game for shard in self._shards for game in shard.native.games]

    @contextmanager
    def reconstructing(self) -> Iterator[None]:
        with ExitStack() as stack:
            for shard in self._shards:
                stack.enter_context(shard.native.reconstructing())
            yield

    def set_observation_layout(
        self,
        buffers: dict[str, Array],
        action_overflow: str,
        *,
        layout: managym.ObservationLayout,
    ) -> None:
        for value in buffers.values():
            self._rows(len(value))
        self._run(
            [
                partial(
                    shard.native.set_observation_layout,
                    {key: value[shard.rows] for key, value in buffers.items()},
                    action_overflow,
                    layout=layout,
                )
                for shard in self._shards
            ]
        )

    def observation_rows(self) -> list[tuple[managym.ObservationBatch, int]]:
        return self._read_rows(lambda native: native.observation_rows())

    def observations_for_players(
        self, players: Sequence[int | None]
    ) -> tuple[list[ObservationRead], managym.ObservationBatch]:
        self._rows(len(players))
        selected = [
            s for s in self._shards if any(p is not None for p in players[s.rows])
        ]
        if not selected:
            selected = self._shards[:1]
        results = self._run(
            [
                partial(s.native.observations_for_players, players[s.rows])
                for s in selected
            ]
        )
        reads = [
            ObservationRead(s.start + r.slot, r.player, r.game_id, r.seed, r.revision)
            for s, (local, _) in zip(selected, results, strict=True)
            for r in local
        ]
        first = results[0][1]
        return reads, first if len(results) == 1 else first.gather(
            [(batch, i) for _, batch in results for i in range(batch.batch_size)]
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

    def _read_rows[T](self, read: Callable[[RecordedVectorEnv], list[T]]) -> list[T]:
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

    def reset_each_into_buffers(
        self, players: list[managym.GameSetup], *, seeds: list[int] | None = None
    ) -> None:
        self._rows(len(players))
        if seeds is not None:
            self._rows(len(seeds))
            # Admit the whole list before any producer replaces its streams.
            seeds = [operator.index(seed) for seed in seeds]
            if any(not 0 <= seed < 2**64 for seed in seeds):
                raise ValueError("native seed must fit uint64")
        self._run(
            [
                partial(
                    shard.native.reset_each_into_buffers,
                    players[shard.rows],
                    seeds=None if seeds is None else seeds[shard.rows],
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

    def current_game_seeds(self) -> list[int]:
        return self._read_rows(lambda native: native.current_game_seeds())

    def next_fresh_seeds(self) -> list[int]:
        return self._read_rows(lambda native: native.next_fresh_seeds())

    def _read_slots[T](
        self,
        mask: Sequence[bool],
        read: Callable[[RecordedVectorEnv, Sequence[bool]], list[T | None]],
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
            raise ValueError("invalid or duplicate slots")
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

    def load_slots_into_buffers(
        self, requests: Sequence[managym.SlotLoad]
    ) -> list[managym.SlotLoadResult]:
        """Load sparse global slots as one native transaction, in request order."""
        slots = [request.slot for request in requests]
        self._validate_slots(slots)
        if not slots:
            return []
        grouped: dict[StreamShard, list[managym.SlotLoad]] = {}
        for request in requests:
            shard = self._shard_by_slot[request.slot]
            grouped.setdefault(shard, []).append(
                request.with_slot(request.slot - shard.start)
            )
        # This coordinated call runs on the owner thread, with live run coordinates.
        for shard in grouped:
            shard.native.frozen_context = None
        loaded = load_recorded_slots(
            [(shard.native, loads) for shard, loads in grouped.items()]
        )
        results = {
            shard.start + result.slot: result.with_slot(shard.start + result.slot)
            for shard, batch in zip(grouped, loaded, strict=True)
            for result in batch
        }
        return [results[slot] for slot in slots]

    def install_slots(self, installs: Sequence[SlotInstall]) -> list[InstalledSlot]:
        """Bind cold roots once, then load fresh games and snapshots together."""
        replayed: dict[int, tuple[SnapshotRecipe, managym.GameSnapshot]] = {}
        loads = [
            native_slot_load(bind_cold_start(request, replayed)) for request in installs
        ]
        return [
            InstalledSlot(
                result.slot,
                result.source_digest or "",
                result.acting_player if result.acting_player is not None else -1,
            )
            for result in self.load_slots_into_buffers(loads)
        ]

    def state_digests(self, active: list[bool] | None = None) -> list[str | None]:
        """Hash selected games in global slot order, leaving masked slots empty."""
        self._check()
        if active is not None:
            self._rows(len(active))
        return [
            digest
            for shard in self._shards
            for digest in shard.native.state_digests(
                None if active is None else active[shard.rows]
            )
        ]

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

    def game_inputs(self, slot: int) -> GameInputs:
        """Materialize one failed game's replay on the producer that owns it."""
        if not 0 <= slot < len(self._shard_by_slot):
            raise IndexError(slot)
        shard = self._shard_by_slot[slot]
        return shard.native.game_inputs(slot - shard.start)

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
            self._shard_by_slot.clear()


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
            native = RecordedVectorEnv(
                num_envs=stop - start,
                seed=min(seed + start, 2**64 - 1),
                seed_stride=num_envs,
                stream_offset=start,
                skip_trivial=self.skip_trivial,
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
    decision_memory: bool,
    decision_memory_include_own: bool,
    games_path: Path | None = None,
    record_fraction: float = 1.0,
) -> BufferedStreams:
    """Select the collector transport while keeping fresh recording on its native slots."""
    if workers != 1:
        return ShardedVectorEnv(
            num_envs,
            seed,
            workers,
            decision_memory=decision_memory,
            decision_memory_include_own=decision_memory_include_own,
            games_path=games_path,
            record_fraction=record_fraction,
        )
    native = RecordedVectorEnv(
        num_envs=num_envs,
        seed=seed,
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
    return BufferedStreams(num_envs, [StreamShard(0, num_envs, native)])
