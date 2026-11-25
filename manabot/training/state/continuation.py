"""Portable self-play recovery at a completed update.

Etude stages the files together. The learner header binds the counts, lookup
buffers, live native games and private sampling state to one optimizer update.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import struct
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from manabot.infra.identity import canonical_json, canonical_sha256, file_sha256
from manabot.policy.checkpoint import read_safetensors, write_safetensors
from manabot.training.rollout.collection import (
    Collector,
    CollectorSnapshot,
    capture_collector,
)
from manabot.training.rollout.snapshots import RandomState
from manabot.training.state.optimizer import AdamState, admit_adam, admit_policy_arrays

if TYPE_CHECKING:
    from manabot.policy.protocol import BackendPolicy
    from manabot.training.loops.self_play import Counts, SelfPlay
    from manabot.training.state.backend import Continuation, SelfPlayLearner

LEARNER_FILE = "learner.safetensors"
SAMPLING_FILE = "sampling.safetensors"
HOST_FILE = "continuation.json"
# The learner header binds the save's other files by these digests.
DIGESTED_FILES = {
    "counts_sha256": "state.json counts",
    "continuation_sha256": HOST_FILE,
    "sampling_sha256": SAMPLING_FILE,
}


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


def read_host(folder: Path) -> bytes:
    """Read the exact host bytes for digest verification before decoding."""
    path = folder / HOST_FILE
    if not path.is_file():
        raise ValueError(f"missing {HOST_FILE}")
    return path.read_bytes()


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
class RestoredContinuation:
    host: HostContinuation
    sampling: dict[str, np.ndarray] | None = None
    sampling_reseeded: bool = False
    adam_steps_aligned: int = 0


def buffer_digests(buffers: Mapping[str, np.ndarray]) -> dict[str, list[object]]:
    return {
        name: [
            str(value.dtype),
            list(value.shape),
            hashlib.sha256(value.tobytes()).hexdigest(),
        ]
        for name, value in buffers.items()
    }


def continuation_header(
    settings: SelfPlay,
    *,
    counts_digest: str,
    has_ema: bool,
    host_digest: str,
    sampling_path: Path,
) -> dict[str, str]:
    return {
        "schema": "canonical-v1",
        "precision": settings.precision,
        "has_ema": json.dumps(has_ema),
        "observation": canonical_json(
            settings.observation.model_dump(mode="json")
        ).decode(),
        "counts_sha256": counts_digest,
        "continuation_sha256": host_digest,
        "sampling_sha256": file_sha256(sampling_path),
    }


def export_continuation(
    folder: Path,
    *,
    state: Continuation,
    learner: SelfPlayLearner,
    ema: BackendPolicy | None,
    counts: Counts,
    settings: SelfPlay,
    collector: Collector,
    rng: np.random.Generator,
) -> None:
    host_digest = write_host(folder, capture_host(collector, rng))
    sampling = folder / SAMPLING_FILE
    write_safetensors(sampling, state.sampling_state(collector.numerics), {})
    adam = state.adam_state(learner, step=counts.optimizer_updates)
    arrays = {"adam/step": np.array(adam.step, dtype=np.int64)}
    for prefix, values in (
        ("raw", state.policy_arrays(learner.agent)),
        ("ema", state.policy_arrays(ema) if ema is not None else {}),
        ("adam/exp_avg", adam.exp_avg),
        ("adam/exp_avg_sq", adam.exp_avg_sq),
    ):
        arrays.update({f"{prefix}/{name}": value for name, value in values.items()})
    write_safetensors(
        folder / LEARNER_FILE,
        arrays,
        continuation_header(
            settings,
            counts_digest=canonical_sha256(counts.model_dump(mode="json")),
            has_ema=ema is not None,
            host_digest=host_digest,
            sampling_path=sampling,
        )
        | {
            "param_steps": canonical_json(adam.param_steps).decode(),
            "buffers": canonical_json(
                buffer_digests(state.policy_buffers(learner.agent))
            ).decode(),
            "sampling_domain": state.sampling_domain(collector.numerics.device_name),
        },
    )


def read_tensors(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    try:
        arrays, header = read_safetensors(path)
        if not isinstance(header, dict) or any(
            not isinstance(v, str) for v in header.values()
        ):
            raise ValueError("invalid safetensors metadata")
        return arrays, header
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        struct.error,
    ) as error:
        raise ValueError(f"{path.name}: {error}") from error


def admit_save_format(folder: Path) -> dict[str, Any]:
    """Read Etude's record and refuse pre-portable saves before model admission."""
    if (folder / "training.safetensors").exists():
        raise ValueError("training.safetensors: unsupported pre-portable MLX save")
    record = json.loads((folder / "state.json").read_bytes())
    if "learner" in record["files"] or (folder / "learner.pt").exists():
        raise ValueError(
            "state.json: this save holds learner.pt, written before portable saves; "
            "resume it with manabot 0.2.5, or start a new run warm-started from "
            "the run's published policy"
        )
    return record


def restore_continuation(
    folder: Path,
    *,
    state: Continuation,
    learner: SelfPlayLearner,
    ema: BackendPolicy | None,
    counts: Counts,
    settings: SelfPlay,
    record: dict[str, Any] | None = None,
) -> RestoredContinuation:
    """Validate the entire snapshot before copying into the live policies and Adam."""
    if record is None:
        record = admit_save_format(folder)
    if record["at"].get("updates") != counts.updates:
        raise ValueError("state.json: updates differs from the saved counts")
    files = record["files"]
    # Etude decodes extra into Counts before the callback. Check the recorded
    # payload as well, so defaults or ignored fields cannot mask a damaged file.
    counts_digest = canonical_sha256(record["extra"])
    if counts_digest != canonical_sha256(counts.model_dump(mode="json")):
        raise ValueError("state.json: counts_sha256 differs from the decoded counts")
    expected_files = {name: name for name in (LEARNER_FILE, SAMPLING_FILE, HOST_FILE)}
    if files != expected_files:
        raise ValueError(f"state.json: expected files {expected_files}, got {files}")
    arrays, header = read_tensors(folder / LEARNER_FILE)
    host_bytes = read_host(folder)
    sampling_path = folder / SAMPLING_FILE
    if not sampling_path.is_file():
        raise ValueError(f"missing {SAMPLING_FILE}")
    domain = header.get("sampling_domain", "")
    if domain not in {"mlx", "torch:cpu", "torch:mps", "torch:cuda"}:
        raise ValueError(f"{LEARNER_FILE}: invalid sampling_domain {domain!r}")
    expected_header = continuation_header(
        settings,
        counts_digest=counts_digest,
        has_ema=ema is not None,
        host_digest=host_sha256(host_bytes),
        sampling_path=sampling_path,
    )
    if differing := header.keys() ^ (
        expected_header.keys() | {"sampling_domain", "buffers", "param_steps"}
    ):
        raise ValueError(
            f"{LEARNER_FILE}: incompatible header keys {sorted(differing)}"
        )
    for key, value in expected_header.items():
        if header[key] == value:
            continue
        if key in DIGESTED_FILES:
            raise ValueError(
                f"{DIGESTED_FILES[key]} differs from the {key} that "
                f"{LEARNER_FILE} recorded; the save is damaged or mixed"
            )
        if key == "observation":
            raise ValueError(f"{LEARNER_FILE}: saved observation config differs")
        raise ValueError(
            f"{LEARNER_FILE}: saved {key} {header[key]!r}, this run {value!r}"
        )
    try:
        if counts.model is None or counts.readings is None:
            raise ValueError("state.json: missing saved model or readings")
        buffers = json.loads(header["buffers"])
        steps = json.loads(header["param_steps"])
        if not isinstance(buffers, dict) or not isinstance(steps, dict):
            raise ValueError("buffers and param_steps must be maps")
        expected_buffers = buffer_digests(state.policy_buffers(learner.agent))
        for key in sorted(buffers.keys() | expected_buffers.keys()):
            if key not in expected_buffers or buffers.get(key) != expected_buffers[key]:
                raise ValueError(f"incompatible buffer {key}")
        step = arrays.pop("adam/step")
        if step.shape != () or step.dtype != np.int64:
            raise ValueError("adam/step: incompatible shape or dtype")

        def take(prefix: str) -> dict[str, np.ndarray]:
            return {
                key.removeprefix(prefix): arrays.pop(key)
                for key in list(arrays)
                if key.startswith(prefix)
            }

        raw, averaged = take("raw/"), take("ema/")
        adam = AdamState(
            int(step), steps, take("adam/exp_avg/"), take("adam/exp_avg_sq/")
        )
        if arrays:
            raise ValueError(f"unexpected tensor keys {sorted(arrays)}")
        shapes = state.parameter_shapes(learner.agent)
        admit_policy_arrays(raw, shapes)
        admit_policy_arrays(
            averaged, state.parameter_shapes(ema) if ema is not None else {}
        )
        admit_adam(adam, shapes, step=counts.optimizer_updates)
    except (KeyError, ValueError) as error:
        raise ValueError(f"{LEARNER_FILE}: {error}") from error
    try:
        host = HostContinuation.model_validate_json(host_bytes)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{HOST_FILE}: {error}") from error
    reseeded = domain != state.sampling_domain(settings.device)
    sampling = None
    if not reseeded:
        sampling, _ = read_tensors(sampling_path)
        state.validate_sampling(settings.device, sampling)
    state.load_policy_arrays(learner.agent, raw)
    if ema is not None:
        state.load_policy_arrays(ema, averaged)
    aligned = state.load_adam(learner, adam)
    return RestoredContinuation(host, sampling, reseeded, aligned)
