"""Bounded optimizer-only overhead probe; not RL throughput or strength.

Run with uv run python scripts/benchmark_numerical_health.py --device cpu.
CUDA must be requested explicitly on a capable host; no silent CPU fallback.
"""

import argparse
import json
from statistics import median
import time

import torch

from manabot.training.health import NumericalHealth, OptimizerHealth


def measure(device: str, steps: int, guarded: bool) -> float:
    model = torch.nn.Module()
    model.register_parameter(
        "weights", torch.nn.Parameter(torch.full((188418,), 0.25, device=device))
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for iteration in range(1, steps + 1):
        loss = model.weights.square().mean()
        if guarded:
            health = NumericalHealth(
                optimizer_steps=0, rejected_steps=0, skipped_steps=0, sampled_steps=0
            )
            OptimizerHealth(model, optimizer, health, iteration).step(loss, 0.5)
        else:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 0.5, error_if_nonfinite=True
            )
            optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    measure(args.device, 20, True)
    baseline: list[float] = []
    guarded: list[float] = []
    for repeat in range(5):
        # Alternate order to reduce warmup/order bias; no statistical hardware claim.
        for enabled in (False, True) if repeat % 2 == 0 else (True, False):
            (guarded if enabled else baseline).append(
                measure(args.device, 100, enabled)
            )
    print(
        json.dumps(
            {
                "device": args.device,
                "torch": torch.__version__,
                "parameters": 188418,
                "steps": 100,
                "threads": 1,
                "baseline_seconds": baseline,
                "guarded_seconds": guarded,
                "median_baseline": median(baseline),
                "median_guarded": median(guarded),
                "ratio": median(guarded) / median(baseline),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
