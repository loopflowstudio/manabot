"""Persistent CPU observation producers for one GPU-owned collector.

ShardedVectorEnv gives each worker disjoint native streams and NumPy row views.
Native buffered calls release the GIL; a barrier completes every shard before
returning to the collector. The parent retains global batching, policy sampling,
rollout ordering and recovery. Global reset strides make stream identity
independent of worker count. No policy or CUDA state enters a producer thread.
"""

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
from numpy.typing import NDArray

from manabot.env.failures import CapturedVector
import managym

Array = NDArray[np.generic]


@dataclass(frozen=True)
class _Shard:
    start: int
    stop: int
    native: CapturedVector


class ShardedVectorEnv:
    """Coarse native producers with a synchronous, globally ordered interface.

    The caller owns the arrays exclusively during every method. Native calls are
    cooperative: an outer process supervisor supplies a hard execution deadline.
    On a worker failure, all submitted calls settle before the pool is closed and
    the earliest shard error is raised. Partial buffers must then be discarded.
    Call close at the end of the owning training run, including failure paths.
    """

    def __init__(self, num_envs: int, seed: int, workers: int) -> None:
        if workers not in (2, 4) or workers > num_envs:
            raise ValueError(
                "parallel producers require 2 or 4 workers and at least one stream each"
            )
        if not 0 <= seed < 2**64:
            raise ValueError("native seed must fit uint64")
        self.num_envs = num_envs
        self.workers = workers
        self._closed = False
        self._shards = tuple(
            _Shard(
                start=index * num_envs // workers,
                stop=(index + 1) * num_envs // workers,
                native=CapturedVector(
                    num_envs=(index + 1) * num_envs // workers
                    - index * num_envs // workers,
                    seed=min(seed + index * num_envs // workers, 2**64 - 1),
                    seed_stride=num_envs,
                    stream_offset=index * num_envs // workers,
                    skip_trivial=True,
                    opponent_policy="none",
                ),
            )
            for index in range(workers)
        )
        self._pool = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="manabot-producer"
        )

    def _check(self) -> None:
        if self._closed:
            raise RuntimeError("producer pool is closed")

    def _rows(self, count: int) -> None:
        self._check()
        if count != self.num_envs:
            raise ValueError("producer row count differs from global streams")

    def _run[T](self, calls: Sequence[Callable[[], T]]) -> list[T]:
        self._check()
        try:
            futures = [self._pool.submit(call) for call in calls]
            # Settle every writer before propagating failure or exposing buffers.
            wait(futures)
            return [future.result() for future in futures]
        except BaseException:
            self.close()
            raise

    def set_buffers(self, buffers: dict[str, Array]) -> None:
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
                )
                for shard in self._shards
            ]
        )

    def reset_all_into_buffers(self, players: list[managym.PlayerConfig]) -> None:
        self._run(
            [
                partial(shard.native.reset_all_into_buffers, players)
                for shard in self._shards
            ]
        )

    def reset_each_into_buffers(
        self, players: list[list[managym.PlayerConfig]]
    ) -> None:
        self._rows(len(players))
        self._run(
            [
                partial(
                    shard.native.reset_each_into_buffers,
                    players[shard.start : shard.stop],
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
        self, actions: list[int], active: list[bool] | None = None
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
                )
                for shard in self._shards
            ]
        )

    def current_agent_indices(self) -> list[int]:
        self._check()
        return [
            index
            for shard in self._shards
            for index in shard.native.current_agent_indices()
        ]

    def step_isolated_into_buffers(
        self, actions: list[int], active: list[bool]
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

    def get_last_info(self) -> list[dict[str, Any]]:
        self._check()
        return [info for shard in self._shards for info in shard.native.get_last_info()]

    def capture_failure(self, error: BaseException) -> None:
        # Capture from the owning thread so live run coordinates remain available.
        for shard in self._shards:
            shard.native.capture_failure(error)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._shards = ()
