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
import managym

Array = NDArray[np.generic]


@dataclass(frozen=True)
class StreamShard:
    start: int
    stop: int
    native: CapturedVector


@dataclass(frozen=True)
class ObservationRead:
    """A fixed-seat read in the producer's global coordinates."""

    slot: int
    player: int
    game_id: int
    seed: int
    revision: int


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

    def _owned_shards(self) -> Sequence[StreamShard]:
        """Buffer binding and cleanup include parked sources as well as active streams."""
        return self._shards

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
                    {
                        key: value[shard.start : shard.stop]
                        for key, value in buffers.items()
                    },
                    action_overflow,
                    layout=layout,
                )
                for shard in self._owned_shards()
            ]
        )

    def write_observations_for_players(
        self,
        players: list[int],
        buffers: dict[str, Array],
        action_overflow: Literal["error", "first-legal"] = "error",
        *,
        layout: managym.ObservationLayout,
    ) -> list[ObservationRead]:
        """Project native seats into disjoint row slices and return global identities."""
        self._rows(len(players))
        for value in buffers.values():
            self._rows(len(value))
        reads = self._run(
            [
                partial(
                    shard.native.write_observations_for_players,
                    players[shard.start : shard.stop],
                    {
                        key: value[shard.start : shard.stop]
                        for key, value in buffers.items()
                    },
                    action_overflow,
                    layout=layout,
                )
                for shard in self._shards
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
            for shard, local in zip(self._shards, reads, strict=True)
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

    def current_player_configs(self) -> list[managym.GameSetup]:
        """Return each current game with resolved open/closed disclosure."""
        self._check()
        return [
            players
            for shard in self._shards
            for players in shard.native.current_player_configs()
        ]

    def current_matchup_indices(self) -> list[tuple[int, int]]:
        self._check()
        return [
            pair
            for shard in self._shards
            for pair in shard.native.current_matchup_indices()
        ]

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
        self._run(
            [
                partial(
                    shard.native.reset_each_into_buffers,
                    players[shard.start : shard.stop],
                    seeds=None if seeds is None else seeds[shard.start : shard.stop],
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
                    seeds[shard.start : shard.stop],
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
                    actions[shard.start : shard.stop],
                    None if active is None else active[shard.start : shard.stop],
                    annotations=None
                    if annotations is None
                    else annotations[shard.start : shard.stop],
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
        self._check()
        return [
            index
            for shard in self._shards
            for index in shard.native.current_agent_indices()
        ]

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
                    actions[shard.start : shard.stop],
                    active[shard.start : shard.stop],
                    recover_capacity,
                    annotations=None
                    if annotations is None
                    else annotations[shard.start : shard.stop],
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
                    failed[shard.start : shard.stop],
                )
                for shard in self._shards
            ]
        )

    def current_game_seeds(self) -> list[int]:
        self._check()
        return [
            seed for shard in self._shards for seed in shard.native.current_game_seeds()
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
                None if active is None else active[shard.start : shard.stop]
            )
        ]

    def current_game_generations(self) -> list[int]:
        self._check()
        return [
            epoch
            for shard in self._shards
            for epoch in shard.native.current_game_generations()
        ]

    def decision_memory_cursors(self) -> list[list[list[int]]]:
        self._check()
        return [
            cursor
            for shard in self._shards
            for cursor in shard.native.decision_memory_cursors()
        ]

    def decision_memory_counts(self) -> list[list[list[int]]]:
        self._check()
        return [
            counts
            for shard in self._shards
            for counts in shard.native.decision_memory_counts()
        ]

    def get_last_info(self) -> list[dict[str, Any]]:
        self._check()
        return [info for shard in self._shards for info in shard.native.get_last_info()]

    def buffered_call_timings(self) -> tuple[float, float]:
        """Mean shard stepping and encoding seconds: shards run at once."""
        shards = [shard.native.buffered_call_timings() for shard in self._shards]
        return (
            sum(stepping for stepping, _ in shards) / len(shards),
            sum(encoding for _, encoding in shards) / len(shards),
        )

    def capture_failure(self, error: BaseException) -> None:
        # Capture from the owning thread so live run coordinates remain available.
        for shard in self._shards:
            shard.native.capture_failure(error)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            for shard in self._owned_shards():
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
        decision_memory_include_own: bool = False,
        decision_memory: bool = False,
        games_path: Path | None = None,
        record_fraction: float = 1.0,
    ) -> None:
        if workers not in (2, 4) or workers > num_envs:
            raise ValueError(
                "parallel producers require 2 or 4 workers and at least one stream each"
            )
        if not 0 <= seed < 2**64:
            raise ValueError("native seed must fit uint64")
        self.workers = workers
        shards = [
            StreamShard(
                start=index * num_envs // workers,
                stop=(index + 1) * num_envs // workers,
                native=CapturedVector(
                    num_envs=(index + 1) * num_envs // workers
                    - index * num_envs // workers,
                    seed=min(seed + index * num_envs // workers, 2**64 - 1),
                    seed_stride=num_envs,
                    stream_offset=index * num_envs // workers,
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
                ),
            )
            for index in range(workers)
        ]
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
