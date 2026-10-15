"""Compare scoped preparation with ordinary inputs through the same TrainingRun.

The reference changes only preparation to the original CPU-to-device dictionary
transfer. Checkpoint witnesses are computed after training timing completes and
include raw/EMA parameters and Adam state, not serialization-specific file bytes.
"""

import argparse
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import resource
import threading
import time
from unittest.mock import patch

import torch

from experiments.runners.cuda_performance import Cell, child
from experiments.runners.pipeline_benchmark import (
    _batch_tensors,
    _buffer_digest,
    _TimedEnvironment,
)
from manabot.model.agent import Agent
from manabot.model.inference import PolicyInference
from manabot.sim.net_opponent import RolloutBatch, SeatRoutedCollector


def _ordinary_inputs(
    scope: PolicyInference, obs: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return {key: tensor.to(scope.device) for key, tensor in obs.items()}


@contextmanager
def reference_inference(enabled: bool) -> Iterator[None]:
    if enabled:
        with patch.object(PolicyInference, "prepare", _ordinary_inputs):
            yield
    else:
        yield


def state_digest(value: object) -> str:
    """Canonical tensor/optimizer payload hash, independent of torch.save IDs."""
    digest = hashlib.sha256()

    def visit(item: object) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(
                json.dumps(["tensor", str(tensor.dtype), list(tensor.shape)]).encode()
            )
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                visit(key)
                visit(item[key])
            digest.update(b"end")
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for element in item:
                visit(element)
            digest.update(b"end")
        elif item is None or isinstance(item, (str, int, float, bool)):
            digest.update(json.dumps([type(item).__name__, item]).encode())
        else:
            raise TypeError(
                f"unsupported checkpoint witness type: {type(item).__name__}"
            )

    visit(value)
    return digest.hexdigest()


def checkpoint_witnesses(directory: Path) -> dict[str, str]:
    """Hash exports, leaving separately typed recovery envelopes to recovery tests."""
    witnesses: dict[str, str] = {}
    for path in sorted(directory.glob("*.pt")):
        if not path.name.endswith(("-raw.pt", "-ema.pt", "-optimizer.pt")):
            continue
        # These are the experiment's own ordinary TrainingRun exports.
        payload: object = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError("training export must be a dictionary")
        if "model_state_dict" in payload:
            payload = payload["model_state_dict"]
        witnesses[path.name] = state_digest(payload)
    for suffix in ("-raw.pt", "-ema.pt", "-optimizer.pt"):
        if not any(name.endswith(suffix) for name in witnesses):
            raise ValueError(f"training produced no {suffix} witness")
    return witnesses


class _WitnessEnvironment(_TimedEnvironment):
    """Hash the global action/active journal without retaining a second copy."""

    def __init__(self, native: object) -> None:
        super().__init__(native)
        self.journal = hashlib.sha256()

    def step_into_buffers(self, actions: list[int], active: list[bool]) -> None:
        self.journal.update(json.dumps([actions, active]).encode())
        super().step_into_buffers(actions, active)


@contextmanager
def collection_witnesses(root: Path) -> Iterator[None]:
    """Diagnostic only: hash each full rollout and its final collection state.

    The environment proxy is restored even on collection failure. Hashing adds
    CPU work and synchronization, so this scope is excluded from timing cells.
    """
    original = SeatRoutedCollector.collect
    count = 0

    @wraps(original)
    def collect(
        collector: SeatRoutedCollector,
        agent: Agent,
        num_steps: int,
        *,
        deadline_monotonic: float | None = None,
        check: Callable[[], None] | None = None,
    ) -> RolloutBatch:
        nonlocal count
        native = collector._env
        witness = _WitnessEnvironment(native)
        collector._env = witness
        try:
            batch = original(
                collector,
                agent,
                num_steps,
                deadline_monotonic=deadline_monotonic,
                check=check,
            )
            payload = {
                "rollout": state_digest(_batch_tensors(batch)),
                "buffers": _buffer_digest(collector),
                "sampling_rng": state_digest(collector._self_rng.get_state()),
                "game_seeds": state_digest(witness.current_game_seeds()),
                "actions_and_active": witness.journal.hexdigest(),
                "microsteps": witness.calls,
            }
            (root / f"collection-witness-{count}.json").write_text(
                json.dumps(payload, indent=2) + "\n"
            )
            count += 1
            return batch
        finally:
            collector._env = native

    with patch.object(SeatRoutedCollector, "collect", collect):
        yield


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cell", type=Cell.model_validate_json, required=True)
    parser.add_argument("--reference-policy", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    if args.cell.kind != "loop":
        parser.error("preparation comparison requires an ordinary training loop")
    if args.deterministic:
        # Diagnostic only: isolate producer equivalence from CUDA reduction order.
        # See PyTorch 2.10 reproducibility and CUDA RNN documentation.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    before = resource.getrusage(resource.RUSAGE_SELF)
    out = args.root / args.cell.name
    proof = collection_witnesses(out) if args.deterministic else nullcontext()
    with reference_inference(args.reference_policy), proof:
        child(args.root, args.cell)
    after = resource.getrusage(resource.RUSAGE_SELF)
    remaining = [
        t.name for t in threading.enumerate() if t.name.startswith("manabot-producer")
    ]
    (out / "execution-witness.json").write_text(
        json.dumps(
            {
                "workers": args.cell.workers,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "seconds": time.perf_counter() - started,
                "cpu_seconds": after.ru_utime
                + after.ru_stime
                - before.ru_utime
                - before.ru_stime,
                "remaining_producer_threads": remaining,
            },
            indent=2,
        )
        + "\n"
    )
    if remaining:
        raise RuntimeError("training left producer threads alive")
    witnesses = checkpoint_witnesses(out / "run")
    (out / "policy-witness.json").write_text(json.dumps(witnesses, indent=2) + "\n")


if __name__ == "__main__":
    main()
