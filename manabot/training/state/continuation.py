"""Host state at a safe self-play update and the backend's staged tensor writer."""

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from manabot.training.rollout.collection import (
    Collector,
    CollectorSnapshot,
    capture_collector,
)
from manabot.training.rollout.numerics import CollectionSession
from manabot.training.rollout.snapshots import RandomState

HOST_FILE = "continuation.json"


class HostContinuation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[2]
    collector: CollectorSnapshot
    minibatch_rng: dict[str, Any]
    python_rng: RandomState
    numpy_global_rng: tuple[Literal["MT19937"], list[int], int, int, float]

    @model_validator(mode="after")
    def validate_generators(self) -> "HostContinuation":
        # Check admission on isolated generators before restoring any live state.
        random.Random(0).setstate(self.python_rng)
        _, keys, position, gaussian, _ = self.numpy_global_rng
        if len(keys) != 624 or not 0 <= position <= 624 or gaussian not in (0, 1):
            raise ValueError("invalid NumPy global RNG state")
        if any(not 0 <= key < 2**32 for key in keys):
            raise ValueError("invalid NumPy global RNG key")
        probe = np.random.default_rng(0)
        probe.bit_generator.state = self.minibatch_rng
        if self.collector.opponent_rng is not None:
            probe.bit_generator.state = self.collector.opponent_rng
        return self


def host_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_host(folder: Path, host: HostContinuation) -> str:
    """Stage the host state and return the digest of the exact bytes written."""
    payload = host.model_dump_json().encode()
    (folder / HOST_FILE).write_bytes(payload)
    return host_sha256(payload)


def read_host(folder: Path) -> bytes | None:
    """The staged host bytes. The backend binds their digest before decoding them."""
    path = folder / HOST_FILE
    return path.read_bytes() if path.is_file() else None


def capture_host(collector: Collector, rng: np.random.Generator) -> HostContinuation:
    name, keys, position, gaussian, cached = np.random.get_state()
    return HostContinuation(
        schema_version=2,
        collector=capture_collector(collector),
        minibatch_rng=rng.bit_generator.state,
        python_rng=random.getstate(),
        numpy_global_rng=(name, keys.tolist(), position, gaussian, cached),
    )


def restore_host_rng(host: HostContinuation, rng: np.random.Generator) -> None:
    rng.bit_generator.state = host.minibatch_rng
    random.setstate(host.python_rng)
    name, keys, position, gaussian, cached = host.numpy_global_rng
    np.random.set_state(
        (name, np.asarray(keys, dtype=np.uint32), position, gaussian, cached)
    )


@dataclass(frozen=True)
class StagedContinuation:
    """Admitted host state and backend operations requiring a live collector."""

    host: HostContinuation | None
    restore_sampling: Callable[[CollectionSession], None]
    export: Callable[[Path, str, CollectionSession], None]
