"""Paired reference/reusable-buffer serving with exact fixed-batch witnesses.

All scheduling and forward execution come from physical-serving-probe. This
orchestrator changes buffer ownership only, retaining virtual host delays and
explicit separation between fixed-batch parity and live batching differences.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import median
import sys
import time
from types import ModuleType

import torch

from experiments.runners.cuda_performance import load_observations
from experiments.runners.pipeline_benchmark import Replay, Tape
from manabot.env import ObservationSpace
from manabot.model.agent import Agent
from manabot.training.execution import atomic_json
from manabot.training.models import TrainingRegime


def physical_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "staging_physical", "scratch/physical-serving-probe.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--actors", type=int, choices=(16, 128), required=True)
    parser.add_argument("--max-batch", type=int, choices=(256, 1024), required=True)
    args = parser.parse_args()
    if (args.out / "result.json").exists():
        raise ValueError("fresh output required")
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
    torch.manual_seed(10349)
    agent = Agent(ObservationSpace(recipe.observation), recipe.agent).to("cuda").eval()
    observations = load_observations(args.root / "observations.pt")
    tape_path = Path("scratch/accelerator-probes/cohort-t8-s64/tape.json")
    replay_path = tape_path.parent / "replay.json"
    tape = Tape.model_validate_json(tape_path.read_text())
    replay_data = json.loads(replay_path.read_text())
    assert (
        replay_data["tape_sha256"] == hashlib.sha256(tape_path.read_bytes()).hexdigest()
    )
    replay = Replay.model_validate(replay_data["samples"][1])
    assert (
        replay.request_host_seconds is not None and replay.tail_host_seconds is not None
    )
    assert replay.batch_sha256 == tape.batch_sha256
    module = physical_module()
    queue = module.queue_module()
    order = (
        "ABBA" if (args.actors, args.max_batch) in ((16, 256), (128, 1024)) else "BAAB"
    )
    trials = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        # Dynamic scratch-module boundary: actual shared backends/receipts remain
        # owned by the physical instrument; no second scheduler or model path.
        reference = executor.submit(module.CudaBackend, agent, observations).result()
        started = time.perf_counter()
        staged = executor.submit(
            module.StagedCudaBackend, agent, observations, args.max_batch
        ).result()
        allocation_seconds = time.perf_counter() - started
        pinned_bytes = sum(
            v.numel() * v.element_size() for v in staged.host_buffers.values()
        )
        device_bytes = sum(
            v.numel() * v.element_size() for v in staged.device_buffers.values()
        )

        def witnesses() -> None:
            host_pointers = {k: v.data_ptr() for k, v in staged.host_buffers.items()}
            device_pointers = {
                k: v.data_ptr() for k, v in staged.device_buffers.items()
            }
            for indices in (
                torch.arange(args.max_batch) % staged.available,
                torch.tensor([17, 3, 0]),
                torch.tensor([0, 19, 2]),
                torch.arange(args.max_batch - 1) % staged.available,
            ):
                host = staged.materialize(indices)
                assert all(
                    v.is_pinned() and v.data_ptr() == host_pointers[k]
                    for k, v in host.items()
                )
                expected = reference.materialize(indices)
                assert all(torch.equal(host[k], v) for k, v in expected.items()), (
                    "CPU staging differs"
                )
                device = staged.upload(host)
                torch.cuda.synchronize()
                assert all(
                    v.data_ptr() == device_pointers[k] for k, v in device.items()
                )
                assert all(
                    torch.equal(device[k].cpu(), v) for k, v in expected.items()
                ), "CUDA staging differs"
            try:
                staged.materialize(torch.arange(args.max_batch + 1))
            except ValueError:
                pass
            else:
                raise AssertionError("oversized staging request accepted")
            fixed = []
            for padded, rows in (
                (1, [1]),
                (4, [3]),
                (16, [3, 5]),
                (64, [5] * 8),
                (256, [13] * 16),
                (1024, [7] * 128),
            ):
                if padded > args.max_batch:
                    continue
                for position in (0, 11):
                    requests = tuple(
                        module.LiveRequest(a, position, n, 0)
                        for a, n in enumerate(rows)
                    )
                    receipts = [
                        backend(requests, padded)
                        for backend in (reference, staged, staged, reference)
                    ]
                    match = all(
                        r.deliveries == receipts[0].deliveries for r in receipts
                    )
                    fixed.append(
                        {
                            "padded": padded,
                            "requests": len(rows),
                            "position": position,
                            "exact": match,
                            "receipts": [asdict(r) for r in receipts],
                        }
                    )
                    atomic_json(
                        args.out / "fixed-witnesses.json",
                        {
                            "storage_identity_passed": True,
                            "oversize_rejected": True,
                            "fixed_batches": fixed,
                        },
                    )
                    if not match:
                        raise ValueError("fixed-batch output parity failed")

        executor.submit(witnesses).result()
        for backend in (reference, staged):
            for _ in range(2):
                executor.submit(
                    backend,
                    (module.LiveRequest(0, 0, args.max_batch, 0),),
                    args.max_batch,
                ).result()
        for variant in order:
            torch.cuda.reset_peak_memory_stats()
            outcome = module.live(
                [f.rows for f in tape.frames],
                replay.request_host_seconds,
                replay.tail_host_seconds,
                args.actors,
                min(args.actors, 8),
                args.max_batch,
                [s for s in (1, 4, 16, 64, 256, 1024) if s <= args.max_batch],
                0.00025,
                reference if variant == "A" else staged,
                queue.pop_batch,
                executor,
            )
            assert (
                outcome.requests == args.actors * 59
                and outcome.decisions == args.actors * 1099
            )
            trials.append(
                {
                    "variant": "reference" if variant == "A" else "staged",
                    "outcome": asdict(outcome),
                    "coexisting_backends_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                }
            )
            atomic_json(args.out / "trials.json", trials)
    reference_first = next(t for t in trials if t["variant"] == "reference")["outcome"]
    inputs = {
        (d["actor"], d["position"]): d["input_index_sha256"]
        for d in reference_first["deliveries"]
    }
    outputs = {
        (d["actor"], d["position"]): d["output_sha256"]
        for d in reference_first["deliveries"]
    }
    for trial in trials:
        deliveries = trial["outcome"]["deliveries"]
        assert {
            (d["actor"], d["position"]): d["input_index_sha256"] for d in deliveries
        } == inputs
        trial["output_hash_differences_vs_first_reference"] = sum(
            d["output_sha256"] != outputs[(d["actor"], d["position"])]
            for d in deliveries
        )
    reference_seconds = median(
        t["outcome"]["seconds"] for t in trials if t["variant"] == "reference"
    )
    staged_seconds = median(
        t["outcome"]["seconds"] for t in trials if t["variant"] == "staged"
    )
    paths = (
        Path(__file__),
        Path("scratch/physical-serving-probe.py"),
        Path("scratch/accelerator-queue-probe.py"),
        tape_path,
        replay_path,
        args.root / "observations.pt",
    )
    report = {
        "actors": args.actors,
        "max_batch": args.max_batch,
        "order": order,
        "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "allocation_seconds": allocation_seconds,
        "persistent_pinned_bytes": pinned_bytes,
        "persistent_device_bytes": device_bytes,
        "trials": trials,
        "reference_median_seconds": reference_seconds,
        "staged_median_seconds": staged_seconds,
        "median_speedup": reference_seconds / staged_seconds,
        "limits": [
            "Real one-worker serving, virtual host delays, no network/IPC/learning or trajectory claim.",
            "Every call materializes and uploads current CPU observations; immutable input pool is never cached on GPU.",
            "Both backends coexist; peak allocation is a combined footprint, not a per-variant memory comparison.",
            "Fixed-batch output parity is exact; live grouping differences and output hashes remain explicit.",
        ],
    }
    atomic_json(args.out / "result.json", report)
    print(
        "Paired staging completed",
        args.actors,
        args.max_batch,
        report["median_speedup"],
    )


if __name__ == "__main__":
    main()
