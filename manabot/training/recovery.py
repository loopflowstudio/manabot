"""Private, digest-bound update snapshots for the regime executor.

These trusted local artifacts contain hidden collector state and Python RNG
objects; never accept downloaded pickle files or expose them as play checkpoints.
The ordinary model export remains the only serving artifact. Replay restores
native auto-reset state from the original seed, then checks every output buffer.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import fcntl
import os
from pathlib import Path
import random
import socket
import time
from typing import Any, Literal

import numpy as np
import torch

from manabot.arena.models import file_sha256
from manabot.model.agent import Agent
from manabot.sim.net_opponent import (
    CollectorSnapshot,
    CollectorStats,
    NetOpponentTrainer,
)
from manabot.training.models import StageRecord, TrainingRun


@dataclass
class UpdateSnapshot:
    format_version: Literal[2]
    iteration: int
    completed_stages: list[StageRecord]
    collector_before: CollectorStats
    record: StageRecord
    learner: dict[str, torch.Tensor]
    ema: dict[str, torch.Tensor] | None
    # torch/NumPy optimizer and RNG serialization boundaries own nested schemas.
    optimizer: dict[str, Any]
    minibatch_rng: dict[str, Any]
    python_rng: tuple[Any, ...]
    numpy_rng: tuple[Any, ...]
    torch_rng: torch.Tensor
    collector: CollectorSnapshot
    regime_digest: str
    seed: int
    identities: dict[str, Any]


def save_update(
    path: Path,
    trainer: NetOpponentTrainer,
    ema: Agent | None,
    rng: np.random.Generator,
    iteration: int,
    record: StageRecord,
    run: TrainingRun,
    collector_before: CollectorStats,
) -> None:
    """Publish immutable bytes before the store advertises their digest."""
    state = UpdateSnapshot(
        format_version=2,
        iteration=iteration,
        completed_stages=deepcopy(run.stages[:-1]),
        collector_before=deepcopy(collector_before),
        record=record.model_copy(deep=True),
        learner=deepcopy(trainer.agent.state_dict()),
        ema=deepcopy(ema.state_dict()) if ema is not None else None,
        optimizer=deepcopy(trainer.optimizer.state_dict()),
        minibatch_rng=deepcopy(rng.bit_generator.state),
        python_rng=random.getstate(),
        numpy_rng=np.random.get_state(),
        torch_rng=torch.get_rng_state(),
        collector=trainer.collector.snapshot(),
        regime_digest=run.regime_digest,
        seed=run.seed,
        identities=deepcopy(run.identities),
    )
    temporary = path.with_suffix(".tmp")
    with temporary.open("xb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    # Hard link publishes without ever replacing a previous checkpoint.
    os.link(temporary, path)
    temporary.unlink()


def load_update(parent: TrainingRun) -> UpdateSnapshot:
    """Load only the latest artifact registered in the canonical local store."""
    artifact = parent.recovery_artifact
    if artifact is None:
        raise ValueError("attempt has no committed recovery boundary")
    path = Path(artifact["path"])
    if file_sha256(path) != artifact["sha256"]:
        raise ValueError("recovery artifact digest mismatch")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(state, UpdateSnapshot)
        or getattr(state, "format_version", None) != 2
    ):
        raise ValueError("invalid recovery snapshot type")
    if (
        state.regime_digest != parent.regime_digest
        or state.seed != parent.seed
        or state.identities != parent.identities
    ):
        raise ValueError("recovery snapshot identity mismatch")
    records = [*state.completed_stages, state.record]
    expected = [stage.id for stage in parent.regime.stages[: len(records)]]
    if [record.id for record in records] != expected or any(
        record.status != "completed" for record in state.completed_stages
    ):
        raise ValueError("recovery stage prefix mismatch")
    for record in records:
        for item in (*record.artifacts.values(), *record.inputs.values()):
            if file_sha256(item["path"]) != item["sha256"]:
                raise ValueError("recovery stage artifact digest mismatch")
    return state


def restore_learning(
    state: UpdateSnapshot,
    trainer: NetOpponentTrainer,
    ema: Agent | None,
    rng: np.random.Generator,
) -> None:
    """Restore RNG last: construction and replay must not perturb training."""
    trainer.agent.load_state_dict(state.learner)
    trainer.optimizer.load_state_dict(state.optimizer)
    if (ema is None) != (state.ema is None):
        raise ValueError("recovery EMA contract mismatch")
    if ema is not None:
        ema.load_state_dict(state.ema)
    rng.bit_generator.state = state.minibatch_rng
    random.setstate(state.python_rng)
    np.random.set_state(state.numpy_rng)
    torch.set_rng_state(state.torch_rng)


@contextmanager
def attempt_lock(path: Path, *, existing: bool = False) -> Iterator[None]:
    """Hold a local POSIX lease through execution; process death releases it.

    Never unlink a lock file: replacing its inode could admit two writers.
    Missing old leases fail closed instead of treating old runs as recoverable.
    """
    with path.open("r+b" if existing else "a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("training attempt still has a live writer") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def settle_orphan(parent: TrainingRun) -> TrainingRun:
    """Called only while holding its saved local lease; retain export evidence.

    Calendar elapsed time conservatively includes unobserved work and downtime.
    A backwards clock cannot support that accounting and is rejected.
    """
    if parent.recovery_host != socket.gethostname():
        raise ValueError("recovery lease belongs to a different host")
    if parent.status != "running":
        return parent
    if parent.last_recorded_wall_seconds is None:
        raise ValueError("attempt has no crash accounting timestamp")
    gap = time.time() - parent.last_recorded_wall_seconds
    if gap < 0:
        raise ValueError("calendar clock moved backwards; crash cost is unknown")
    settled = parent.model_copy(deep=True)
    settled.status = "interrupted"
    settled.seconds += gap
    settled.watchdog_seconds += gap
    settled.unobserved_seconds += gap
    if settled.stages and settled.stages[-1].status != "completed":
        settled.stages[-1].watchdog_seconds += gap
    settled.error = "Writer lease released; unobserved interval charged conservatively"
    return settled
