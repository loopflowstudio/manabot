"""Two bounded C1 diagnostics over one retained 256-stream observation (.npz).

Run on the admission rental after CUDA proof, never as a speed cell:
  uv run python scripts/diagnose_training_transfers.py recording --fixture batch.npz --out recording.json
  uv run python scripts/diagnose_training_transfers.py transfers --fixture batch.npz --out transfers.json

A supplied fixture is kept unchanged. If absent, retain one initial configured-match
observation at the control's shapes, seed 26410; this is not speed-run exposure.
No games, optimizer steps, RNG draws, or extra workers create contention. Each
probe has a 120-second process deadline. These small fixed-input probes separate
cold page faults/copies and device waits; they cannot explain a whole-update stall
by themselves. They change no production path and claim no optimization gain.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Literal

import numpy as np
from numpy.typing import NDArray
import torch

from experiments.elemental_variants import elemental_recipe
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import RewardConfig
from manabot.training.learning.learner import observation_tensors
from manabot.training.rollout.actions import ACTION_FIELDS, PackedActions
from manabot.training.rollout.collection import SeatRoutedCollector, _ObservationRows


@contextmanager
def interval(result: dict[str, object], name: str) -> Iterator[None]:
    before = resource.getrusage(resource.RUSAGE_SELF)
    start = time.perf_counter()
    yield
    after = resource.getrusage(resource.RUSAGE_SELF)
    result[name] = {
        "seconds": time.perf_counter() - start,
        "minor_faults": after.ru_minflt - before.ru_minflt,
        "major_faults": after.ru_majflt - before.ru_majflt,
    }


def diagnose(
    kind: Literal["recording", "transfers"], obs: dict[str, NDArray[np.generic]]
) -> dict[str, object]:
    if not obs or any(value.shape[0] != 256 for value in obs.values()):
        raise ValueError("diagnostic requires one retained 256-stream batch")
    result: dict[str, object] = {
        "kind": kind,
        "rows": 256,
        "bytes": sum(a.nbytes for a in obs.values()),
    }
    rows = np.arange(256)
    if kind == "recording":
        keys = [key for key in obs if key not in ACTION_FIELDS]
        with interval(result, "allocate_virtual"):
            store = _ObservationRows(obs, keys, 256)
        with interval(result, "cold_copy"):
            store.append(rows)
        # Same arrays/pages, same source/selection: isolate reuse from first touch.
        store.rows = 0
        with interval(result, "resident_copy"):
            store.append(rows)
        assert all(np.array_equal(store.arrays()[key], obs[key]) for key in keys)
        valid = obs["actions_valid"].astype(bool)
        if not valid.any(axis=1).all():
            raise ValueError("retained batch contains an empty legal action row")
        probs = valid.astype(np.float32) / valid.sum(
            axis=1, keepdims=True, dtype=np.float32
        )
        logs = np.full_like(probs, -np.inf)
        np.log(probs, out=logs, where=valid)
        with interval(result, "pack_actions"):
            packed = PackedActions.pack(obs, probs, logs)
        result["legal_actions"] = len(packed.slots)
    else:
        if not torch.cuda.is_available():
            raise ValueError("transfer diagnostic requires real CUDA")
        device = torch.device("cuda")
        # Fixed, disclosed mask; no random draw or selection intervention.
        selected = torch.arange(256, device=device).remainder(2) == 0
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            with interval(result, "device_selection_and_host_indices"):
                indexes = torch.arange(256, device=device)[selected].cpu().numpy()
            with interval(result, "host_gather"):
                gathered = {key: value[indexes] for key, value in obs.items()}
            with interval(result, "host_to_device_and_wait"):
                moved = observation_tensors(gathered, device)
                torch.cuda.synchronize()
        for key, value in moved.items():
            np.testing.assert_array_equal(value.cpu().numpy(), gathered[key])
        result["profile"] = profile.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=25
        )
        result["selected_indices"] = indexes.tolist()
        result["gpu"] = torch.cuda.get_device_name()
    return result


def capture_fixture(path: Path) -> None:
    """Retain one initial native batch at the control's shapes, without training.

    This is a fixed configured-match observation, not sampled exposure from a
    speed run. No games complete; collection is closed even if retention fails.
    """
    regime = elemental_recipe("throughput-c1-diagnostic-fixture", 1)
    if (
        regime.observation.policy_input_version != 6
        or regime.observation.max_facts != 512
    ):
        raise ValueError("diagnostic fixture requires frozen v6 / 512-fact shapes")
    collector = SeatRoutedCollector(
        ObservationSpace(regime.observation),
        Match(regime.match),
        Reward(RewardConfig()),
        num_envs=256,
        seed=26410,
        opponent_mode="self",
        recovery_max_microsteps=10000,
    )
    try:
        tensors = collector._slice_obs_tensors(np.arange(256))
        # Exclusive creation preserves a supplied or already retained fixture.
        with path.open("xb") as output:
            np.savez(output, **{key: value.numpy() for key, value in tensors.items()})
    finally:
        collector.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("recording", "transfers"))
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.child:
        # One child, reaped on deadline; no workload remains after the probe.
        subprocess.run(
            [sys.executable, __file__, *sys.argv[1:], "--child"],
            check=True,
            timeout=120,
        )
        return
    torch.set_num_threads(1)
    created = not args.fixture.exists()
    if created:
        capture_fixture(args.fixture)
    with np.load(args.fixture, allow_pickle=False) as fixture:
        obs = {key: fixture[key] for key in fixture.files}
    result = diagnose(args.kind, obs)
    result["fixture_created"] = created
    result["fixture_seed_if_created"] = 26410 if created else None
    result["fixture_sha256"] = hashlib.sha256(args.fixture.read_bytes()).hexdigest()
    result["shapes"] = {key: list(value.shape) for key, value in obs.items()}
    args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
