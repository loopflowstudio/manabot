"""Frozen CUDA policies and single-flight transport to a shared policy process.

Each worker owns one input/output slot. It publishes a request only after writing
inputs, and cannot reuse its slot until the controller acknowledges CPU outputs.
The controller serves complete requests at their original batch sizes. This
isolates actual collector concurrency without introducing cross-request batching
or changing the neural computation's batch shape between worker-count cases.
CudaPolicy can instead live in each worker, preserving the same forward path
while removing the shared dispatcher and its transport from collection.
"""

from collections.abc import Callable
import json
from pathlib import Path
import sys
import time
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
import torch

from experiments.runners.pipeline_benchmark import (
    PolicyForward,
    PreparedReplay,
    Tape,
    _load_outputs,
    capture,
)
from manabot.env import ObservationSpace
from manabot.model.agent import Agent

# Array dtype differs by observation field; validate once against frozen schemas.
Array: TypeAlias = NDArray[np.generic]


class Exchange:
    """Memory-mapped arrays, private to one worker/controller pair."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        schema: object = json.loads((directory / "schema.json").read_text())
        if not isinstance(schema, dict) or set(schema) != {"inputs", "outputs"}:
            raise ValueError("invalid exchange schema")
        self.inputs = self._open(schema["inputs"])
        self.outputs = self._open(schema["outputs"])
        self.sequence = 0

    def _open(self, names: object) -> dict[str, Array]:
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise ValueError("invalid exchange names")
        if any(not n.replace("_", "").isalnum() for n in names):
            raise ValueError("invalid exchange array name")
        return {n: np.load(self.directory / f"{n}.npy", mmap_mode="r+") for n in names}

    @classmethod
    def create(
        cls,
        directory: Path,
        inputs: dict[str, torch.Tensor],
        outputs: tuple[torch.Tensor, torch.Tensor],
    ) -> "Exchange":
        directory.mkdir(parents=True, exist_ok=False)
        arrays = dict(inputs, response_logits=outputs[0], response_value=outputs[1])
        for key, tensor in arrays.items():
            value = tensor.detach().cpu().numpy()
            np.lib.format.open_memmap(
                directory / f"{key}.npy",
                mode="w+",
                dtype=value.dtype,
                shape=(64, *value.shape[1:]),
            )
        (directory / "schema.json").write_text(
            json.dumps(
                {
                    "inputs": list(inputs),
                    "outputs": ["response_logits", "response_value"],
                }
            )
        )
        return cls(directory)

    def client(self, worker: int) -> PolicyForward:
        def forward(obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            if set(obs) != set(self.inputs):
                raise ValueError("policy input keys differ from exchange")
            rows = len(obs["actions_valid"])
            if not 0 < rows <= 64:
                raise ValueError("policy request exceeds exchange capacity")
            for key, tensor in obs.items():
                array = tensor.detach().cpu().numpy()
                target = self.inputs[key][:rows]
                if array.shape != target.shape or array.dtype != target.dtype:
                    raise ValueError("policy input schema differs from exchange")
                target[...] = array
            self.sequence += 1
            print(
                json.dumps(
                    {
                        "kind": "inference",
                        "worker": worker,
                        "sequence": self.sequence,
                        "rows": rows,
                    }
                ),
                flush=True,
            )
            response: object = json.loads(sys.stdin.readline())
            if response != {"kind": "inference_done", "sequence": self.sequence}:
                raise ValueError("inference acknowledgment mismatch")
            return (
                torch.from_numpy(self.outputs["response_logits"][:rows].copy()),
                torch.from_numpy(self.outputs["response_value"][:rows].copy()),
            )

        return forward

    def serve(self, rows: int, sequence: int, forward: PolicyForward) -> None:
        if sequence != self.sequence + 1 or not 0 < rows <= 64:
            raise ValueError("inference request sequence/capacity mismatch")
        obs = {
            key: torch.from_numpy(value[:rows]) for key, value in self.inputs.items()
        }
        logits, value = forward(obs)
        for key, tensor in [("response_logits", logits), ("response_value", value)]:
            array = tensor.detach().cpu().numpy()
            target = self.outputs[key][:rows]
            if array.shape != target.shape or array.dtype != target.dtype:
                raise ValueError("policy response schema differs from exchange")
            target[...] = array
        self.sequence = sequence


RequestHandler: TypeAlias = Callable[[int, int, int], None]


class CudaPolicy:
    """A frozen model owned by the calling process, with synchronous CPU outputs."""

    def __init__(self, tape: Tape) -> None:
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(tape.seed)
        self.agent = (
            Agent(ObservationSpace(tape.recipe.observation), tape.recipe.agent)
            .eval()
            .to("cuda")
        )
        self.calls = 0
        self.seconds = 0.0

    def forward(
        self, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        started = time.perf_counter()
        with torch.inference_mode():
            logits, value = self.agent.forward_distribution(
                {k: v.to("cuda") for k, v in obs.items()}
            )
            result = logits.cpu(), value.cpu()
        self.calls += 1
        self.seconds += time.perf_counter() - started
        return result


class CudaService(CudaPolicy):
    """One frozen CUDA model and four worker slots, owned by the controller."""

    def __init__(self, source_tape: Path, out: Path) -> None:
        tape = Tape.model_validate_json((source_tape / "tape.json").read_text())
        if tape.streams > 64:
            raise ValueError("worker service supports at most64 streams per actor")
        super().__init__(tape)
        self.tape_dir = out / "physical-tape"
        self.tape_dir.mkdir()
        # Freeze outputs on this GPU. CPU/CUDA numerical differences must not
        # be confused with differences between worker counts on the same GPU.
        tape = capture(
            tape.recipe,
            self.tape_dir,
            tape.streams,
            tape.transitions,
            tape.seed,
            forward=self.forward,
        )
        tape.initialization = (
            "seeded untrained policy; CUDA float32/TF32off; CPU sampling"
        )
        (self.tape_dir / "tape.json").write_text(tape.model_dump_json(indent=2) + "\n")
        prepared = PreparedReplay(self.tape_dir, verify_inputs=True)
        template = prepared.collector._slice_obs_tensors(np.arange(tape.streams))
        output = _load_outputs(self.tape_dir / "outputs.pt", tape)[0]
        self.exchange_dir = out / "exchange"
        self.exchanges = [
            Exchange.create(self.exchange_dir / str(i), template, output)
            for i in range(4)
        ]
        self.hardware: dict[str, str | int] = {
            "mode": "shared-cuda",
            "device": torch.cuda.get_device_name(),
            "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "batching": "one complete request at a time, original batch shape",
        }

    def handle(self, worker: int, rows: int, sequence: int) -> None:
        self.exchanges[worker].serve(rows, sequence, self.forward)
