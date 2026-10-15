"""Pair ordinary collector trajectories with reusable CUDA input staging.

CollectorStaging owns transient input buffers for one collector. The collector
continues to own rules, sampling, pending copies and final rollout arrays. This
research adapter is deliberately separate from production collection/learning.
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import median
import sys
import time
from types import ModuleType
from unittest.mock import patch

import numpy as np
from numpy.typing import NDArray
import torch

from manabot.env import ObservationSpace
from manabot.model.agent import Agent
from manabot.sim.net_opponent import RolloutBatch, SeatRoutedCollector
from manabot.sim.rollout import OBS_KEYS
from manabot.training.execution import atomic_json
from manabot.training.models import TrainingRegime


@dataclass
class Identity:
    batch: str
    buffers: str
    journal: str
    game_seeds: str
    sampling_rng: str


@dataclass
class Sample:
    staged: bool
    seconds: float
    construction_seconds: float
    allocation_seconds: float
    startup_inclusive_seconds: float
    identity: Identity
    exact: bool
    rollout_storage_independent: bool
    decisions: int
    learner_transitions: int
    games: int
    truncations: int
    pinned_bytes: int
    device_bytes: int
    peak_allocated_bytes: int


def validation_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "collector_validation", "scratch/gpu-validation.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def array_digest(arrays: dict[str, NDArray[np.generic]]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        digest.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


class CollectorStaging:
    """Gather current CPU rows; return CUDA views valid until the next call.

    All calls and subsequent forwards belong to one CUDA stream. Synchronizing
    before the next gather protects pinned host and device reuse. Nothing here
    may be retained by autograd or replace the collector's independent CPU copies.
    """

    def __init__(self, collector: SeatRoutedCollector) -> None:
        self.collector = collector
        self.stream = torch.cuda.current_stream()
        self.sources = {
            key: torch.from_numpy(collector._buffers[key]) for key in OBS_KEYS
        }
        self.native_host = {
            key: torch.empty_like(value, pin_memory=True)
            for key, value in self.sources.items()
        }
        self.float_focus = torch.empty_like(
            self.sources["action_focus"], dtype=torch.float32, pin_memory=True
        )
        self.device = {
            key: torch.empty_like(
                value,
                device="cuda",
                dtype=torch.float32 if key == "action_focus" else value.dtype,
            )
            for key, value in self.sources.items()
        }
        self.pinned_bytes = (
            sum(t.numel() * t.element_size() for t in self.native_host.values())
            + self.float_focus.numel() * self.float_focus.element_size()
        )
        self.device_bytes = sum(
            t.numel() * t.element_size() for t in self.device.values()
        )

    def slice(self, rows: NDArray[np.int64]) -> dict[str, torch.Tensor]:
        if (
            rows.ndim != 1
            or rows.dtype != np.int64
            or len(rows) == 0
            or len(rows) > self.collector.num_envs
            or rows.min() < 0
            or rows.max() >= self.collector.num_envs
        ):
            raise ValueError("invalid staging rows")
        if torch.cuda.current_stream() != self.stream:
            raise ValueError("collector staging changed CUDA stream")
        self.stream.synchronize()
        indices = torch.from_numpy(rows)
        count = len(rows)
        result = {}
        for key, source in self.sources.items():
            host = self.native_host[key][:count]
            torch.index_select(source, 0, indices, out=host)
            if key == "action_focus":
                self.float_focus[:count].copy_(host)
                host = self.float_focus[:count]
            device = self.device[key][:count]
            device.copy_(host, non_blocking=True)
            result[key] = device
        return result

    def verify(self) -> dict[str, object]:
        host_pointers = {k: v.data_ptr() for k, v in self.native_host.items()}
        device_pointers = {k: v.data_ptr() for k, v in self.device.items()}
        original = self.collector._slice_obs_tensors
        sizes = []
        for rows in (
            np.arange(self.collector.num_envs, dtype=np.int64),
            np.array([3], dtype=np.int64),
            np.array([17, 0, 3], dtype=np.int64),
            np.arange(self.collector.num_envs - 1, -1, -1, dtype=np.int64),
        ):
            expected = original(rows)
            actual = self.slice(rows)
            torch.cuda.synchronize()
            assert all(
                actual[k].dtype == v.dtype and torch.equal(actual[k], v)
                for k, v in expected.items()
            )
            assert actual["action_focus"].dtype == torch.float32
            assert all(
                v.is_pinned() and v.data_ptr() == host_pointers[k]
                for k, v in self.native_host.items()
            )
            assert self.float_focus.is_pinned()
            assert all(v.data_ptr() == device_pointers[k] for k, v in actual.items())
            sizes.append(len(rows))
        for rows in (
            np.array([], dtype=np.int64),
            np.array([-1], dtype=np.int64),
            np.array([self.collector.num_envs], dtype=np.int64),
        ):
            try:
                self.slice(rows)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid row accepted")
        return {
            "sizes": sizes,
            "all_fields_exact": True,
            "stable_pointers": True,
            "invalid_rows_rejected": True,
            "action_focus_dtype": "float32",
        }

    def verify_ownership(self, batch: RolloutBatch) -> bool:
        storage = (
            [v.numpy() for v in self.native_host.values()]
            + [self.float_focus.numpy()]
            + list(self.collector._buffers.values())
        )
        for field in fields(batch):
            value = getattr(batch, field.name)
            arrays = value.values() if isinstance(value, dict) else [value]
            for array in arrays:
                if array is not None:
                    assert isinstance(array, np.ndarray)
                    if any(np.shares_memory(array, source) for source in storage):
                        return False
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--streams", type=int, choices=(64, 512, 4096), required=True)
    parser.add_argument("--seed", type=int, choices=(10349, 10350), required=True)
    args = parser.parse_args()
    assert not (args.out / "result.json").exists()
    args.out.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert (
        torch.cuda.is_available()
        and torch.cuda.get_device_name() == "NVIDIA H100 80GB HBM3"
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    recipe = TrainingRegime.model_validate_json(
        (args.root / "baseline.json").read_text()
    )
    module = validation_module()
    torch.manual_seed(args.seed)
    agent = Agent(ObservationSpace(recipe.observation), recipe.agent).to("cuda").eval()
    for staged in (False, True):
        env = module.collector(recipe, args.streams, args.seed)
        backend = CollectorStaging(env)
        if staged:
            atomic_json(args.out / "fixed-witnesses.json", backend.verify())
        with (
            patch.object(env, "_slice_obs_tensors", backend.slice)
            if staged
            else nullcontext()
        ):
            env.collect(agent, 2, deadline_monotonic=time.perf_counter() + 25)
        del backend, env
    reference: Identity | None = None
    samples: list[Sample] = []
    order = (False, True, True, False) if args.seed % 2 else (True, False, False, True)
    for staged in order:
        started = time.perf_counter()
        env = module.collector(recipe, args.streams, args.seed)
        construction_seconds = time.perf_counter() - started
        started = time.perf_counter()
        backend = CollectorStaging(env) if staged else None
        torch.cuda.synchronize()
        allocation_seconds = time.perf_counter() - started
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with (
            patch.object(env, "_slice_obs_tensors", backend.slice)
            if backend
            else nullcontext()
        ):
            batch = env.collect(agent, 8, deadline_monotonic=time.perf_counter() + 25)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        identity = Identity(
            module.digest_batch(batch),
            array_digest(env._buffers),
            hashlib.sha256(json.dumps(env._journal).encode()).hexdigest(),
            hashlib.sha256(
                json.dumps(env._env.current_game_seeds()).encode()
            ).hexdigest(),
            hashlib.sha256(
                env._self_rng.get_state().cpu().numpy().tobytes()
            ).hexdigest(),
        )
        independent = backend.verify_ownership(batch) if backend else True
        exact = reference is None or identity == reference
        reference = identity if reference is None else reference
        samples.append(
            Sample(
                staged,
                seconds,
                construction_seconds,
                allocation_seconds,
                construction_seconds + allocation_seconds + seconds,
                identity,
                exact,
                independent,
                env.stats.micro_steps,
                env.stats.learner_transitions,
                env.stats.games,
                env.stats.truncations,
                backend.pinned_bytes if backend else 0,
                backend.device_bytes if backend else 0,
                torch.cuda.max_memory_allocated(),
            )
        )
        atomic_json(
            args.out / "result.json",
            {
                "streams": args.streams,
                "seed": args.seed,
                "order": "ABBA" if args.seed % 2 else "BAAB",
                "samples": [asdict(s) for s in samples],
            },
        )
        if not exact or not independent:
            raise ValueError("ordinary collector trajectory parity failed")
        assert (
            env.stats.learner_transitions == args.streams * 8
            and env.stats.truncations == 0
        )
        del batch, backend, env
    gain = median(s.seconds for s in samples if not s.staged) / median(
        s.seconds for s in samples if s.staged
    )
    atomic_json(
        args.out / "summary.json",
        {
            "median_speedup": gain,
            "target_passed": 1 / gain <= 1.05 if args.streams == 64 else gain >= 1.10,
            "all_exact": True,
            "samples": len(samples),
            "limits": "Untrained fixed policy, actual ordinary collection; no optimizer or distributed training measurement.",
        },
    )
    print(f"streams={args.streams} seed={args.seed} gain={gain:.6f}", flush=True)


if __name__ == "__main__":
    main()
