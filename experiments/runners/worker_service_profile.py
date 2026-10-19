"""Attribute the shared policy boundary separately from throughput measurements.

Explicit synchronization separates transfer and forward wall time, so these are
attribution samples, not the concurrent serving curve. The CUDA trace also labels
semantic catalog encoding inside forward. Inputs are real initial observations;
this short profile does not represent a mature policy's state distribution.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from experiments.runners.pipeline_benchmark import PreparedReplay
from experiments.runners.worker_inference import CudaService
from experiments.runners.worker_scaling import Json, write_json


@contextmanager
def _catalog_span(service: CudaService) -> Iterator[None]:
    encoder = service.agent.semantic_cards
    if encoder is None:
        yield
        return
    original = encoder.definitions

    def definitions() -> torch.Tensor:
        with torch.profiler.record_function("service/catalog"):
            return original()

    with patch.object(encoder, "definitions", definitions):
        yield


def profile_service(service: CudaService, out: Path) -> None:
    prepared = PreparedReplay(service.tape_dir, verify_inputs=True)
    samples: list[Json] = []
    with (
        _catalog_span(service),
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler,
    ):
        for rows in (1, 16, 32, 64):
            obs = prepared.collector._slice_obs_tensors(np.arange(rows))
            expected = service.forward(obs)
            service.forward(obs)
            for repeat in range(4):
                timings: dict[str, Json] = {"rows": rows, "repeat": repeat}
                torch.cuda.synchronize()
                with torch.inference_mode():
                    started = time.perf_counter()
                    with torch.profiler.record_function("service/h2d"):
                        device_obs = service.inference.prepare(obs)
                        torch.cuda.synchronize()
                    timings["h2d_seconds"] = time.perf_counter() - started
                    started = time.perf_counter()
                    with torch.profiler.record_function("service/forward"):
                        logits, value = service.agent.forward_distribution(device_obs)
                        torch.cuda.synchronize()
                    timings["forward_seconds"] = time.perf_counter() - started
                    started = time.perf_counter()
                    with torch.profiler.record_function("service/d2h"):
                        actual = logits.cpu(), value.cpu()
                        torch.cuda.synchronize()
                    timings["d2h_seconds"] = time.perf_counter() - started
                if not all(
                    torch.equal(a, b) for a, b in zip(actual, expected, strict=True)
                ):
                    raise ValueError("profiled policy differs from ordinary serving")
                samples.append(timings)
    profiler.export_chrome_trace(str(out / "service-trace.json"))
    write_json(
        out / "service-profile.json",
        {
            "samples": samples,
            "parity": "exact logits and values at each original input shape",
            "boundary": "synchronized CPU admission+H2D/forward/D2H, profiler enabled; excludes worker IPC and observation slicing; scope catalog already warm",
            "inputs": "first real initial observations, not stationary training distribution",
        },
    )
