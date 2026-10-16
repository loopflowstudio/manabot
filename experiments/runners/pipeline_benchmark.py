"""Replay ordinary collection with policy outputs supplied by a verified tape.

Capture pays for a real frozen CPU policy once. Replay preserves the collector,
sampling, rules, observations and rollout storage, but replaces neural forward
computation with recorded outputs. This measures host demand when inference is
fast; it does not run CUDA or train a policy. Project combines that demand with
retained CUDA model timings and explicit scale/transfer assumptions.

Run all commands through uv. Outputs are immutable attempt directories. The CLI
uses one bounded child process and retains failures, including timeouts and OOMs.
"""

import argparse
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
from typing import Any, Literal
from unittest.mock import patch

import numpy as np
import psutil
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter
import torch

from manabot.arena.models import file_sha256
from manabot.env import Match, ObservationSpace, Reward
from manabot.infra.accelerator_model import DeviceService, HostDemand, Scenario, project
from manabot.infra.hypers import RewardHypers
from manabot.model.agent import Agent
from manabot.model.inference import PolicyInference
from manabot.model.world import checkpoint_world
from manabot.sim.net_opponent import RolloutBatch, SeatRoutedCollector
from manabot.training.execution import atomic_json
from manabot.training.models import TrainingRegime
import managym

JSON_OBJECT = TypeAdapter(dict[str, JsonValue])
PolicyForward = Callable[[dict[str, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]]


class Frame(BaseModel):
    input_sha256: str
    rows: int = Field(gt=0)
    input_bytes: int = Field(gt=0)
    output_bytes: int = Field(gt=0)


class Tape(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    recipe: TrainingRegime
    seed: int
    streams: int = Field(gt=0)
    transitions: int = Field(gt=0)
    frames: list[Frame]
    outputs_sha256: str
    batch_sha256: str
    buffers_sha256: str
    journal_sha256: str
    game_seeds: list[int]
    world_binding: dict[str, JsonValue]
    native_sha256: str
    source: dict[str, str]
    capture_seconds: float
    initialization: str = "seeded untrained policy, eval mode, CPU float32"


class Replay(BaseModel):
    seconds: float
    native_seconds: float
    demand: HostDemand
    batch_sha256: str
    buffers_sha256: str
    journal_sha256: str
    game_seeds: list[int]
    forward_batches: dict[int, int]
    native_calls: int
    games: int
    truncated_games: int
    rss_bytes: int
    # Entry-to-entry host intervals exclude modeled GPU service. The last
    # response's materialization/final assembly belongs to a separate tail.
    request_host_seconds: list[float] | None = None
    tail_host_seconds: float | None = None


def _digest_tensors(values: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(values.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(json.dumps([key, str(array.dtype), list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _batch_tensors(batch: RolloutBatch) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for field in fields(batch):
        value = getattr(batch, field.name)
        if isinstance(value, np.ndarray):
            result[field.name] = torch.from_numpy(value)
        elif isinstance(value, dict):
            for key, array in value.items():
                if not isinstance(key, str) or not isinstance(array, np.ndarray):
                    raise TypeError("unexpected rollout field")
                result[f"{field.name}/{key}"] = torch.from_numpy(array)
        elif value is not None:
            raise TypeError(f"unexpected rollout field {field.name}")
    return result


def _source() -> dict[str, str]:
    paths = sorted(Path("manabot").rglob("*.py")) + [Path(__file__)]
    return {str(p): file_sha256(p) for p in paths}


def _world(recipe: TrainingRegime) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_python(
        checkpoint_world(
            Match(recipe.match).to_rust(), ObservationSpace(recipe.observation)
        )
    )


def _collector(recipe: TrainingRegime, streams: int, seed: int) -> SeatRoutedCollector:
    return SeatRoutedCollector(
        ObservationSpace(recipe.observation),
        Match(recipe.match),
        Reward(RewardHypers()),
        num_envs=streams,
        seed=seed,
        opponent_mode="self",
        recovery_max_microsteps=1_000_000,
    )


def _buffer_digest(collector: SeatRoutedCollector) -> str:
    return _digest_tensors(
        {k: torch.from_numpy(v) for k, v in collector._buffers.items()}
    )


def _journal_digest(collector: SeatRoutedCollector) -> str:
    return hashlib.sha256(
        json.dumps(collector._journal, separators=(",", ":")).encode()
    ).hexdigest()


def _tensor_bytes(tensors: dict[str, torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in tensors.values())


@contextmanager
def _policy_override(
    agent: Agent, forward: PolicyForward, *, external: bool
) -> Iterator[None]:
    # A recorded/external policy owns admission and encoding. The placeholder
    # CPU Agent must not repeat that work before requesting the actual policy.
    original = PolicyInference.prepare

    def prepare(
        scope: PolicyInference, obs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if external and scope.agent is agent:
            return {key: value.to(scope.device) for key, value in obs.items()}
        return original(scope, obs)

    with (
        patch.object(agent, "forward_distribution", forward),
        patch.object(PolicyInference, "prepare", prepare),
    ):
        yield


def capture(
    recipe: TrainingRegime,
    out: Path,
    streams: int,
    transitions: int,
    seed: int,
    *,
    forward: PolicyForward | None = None,
) -> Tape:
    """Capture ordinary network outputs; no model update or synthetic action choice."""
    torch.manual_seed(seed)
    agent = Agent(ObservationSpace(recipe.observation), recipe.agent).eval()
    collector = _collector(recipe, streams, seed)
    original = agent.forward_distribution if forward is None else forward
    frames: list[Frame] = []
    outputs: list[dict[str, torch.Tensor]] = []

    def record(obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        logits, value = original(obs)
        saved = {
            "logits": logits.detach().cpu().clone(),
            "value": value.detach().cpu().clone(),
        }
        outputs.append(saved)
        frames.append(
            Frame(
                input_sha256=_digest_tensors(obs),
                rows=len(logits),
                input_bytes=_tensor_bytes(obs),
                output_bytes=_tensor_bytes(saved),
            )
        )
        return logits, value

    started = time.perf_counter()
    with _policy_override(agent, record, external=forward is not None):
        batch = collector.collect(agent, transitions)
    elapsed = time.perf_counter() - started
    torch.save(outputs, out / "outputs.pt")
    tape = Tape(
        recipe=recipe,
        seed=seed,
        streams=streams,
        transitions=transitions,
        frames=frames,
        outputs_sha256=file_sha256(out / "outputs.pt"),
        batch_sha256=_digest_tensors(_batch_tensors(batch)),
        buffers_sha256=_buffer_digest(collector),
        journal_sha256=_journal_digest(collector),
        game_seeds=collector._env.current_game_seeds(),
        world_binding=_world(recipe),
        native_sha256=file_sha256(Path(managym._managym.__file__)),
        source=_source(),
        capture_seconds=elapsed,
    )
    atomic_json(out / "tape.json", tape.model_dump(mode="json"))
    return tape


class _TimedEnvironment:
    """Only the native methods used after collector construction.

    PyO3 exposes no static Python type stubs here. Keep that untyped boundary
    local; consumers receive typed timings and the original arrays unchanged.
    """

    def __init__(self, native: Any) -> None:
        self.native = native
        self.seconds = 0.0
        self.calls = 0

    def current_agent_indices(self) -> list[int]:
        return self.native.current_agent_indices()

    def current_game_seeds(self) -> list[int]:
        return self.native.current_game_seeds()

    def get_last_info(self) -> list[dict[str, Any]]:
        return self.native.get_last_info()

    def step_into_buffers(self, actions: list[int], active: list[bool]) -> None:
        started = time.perf_counter()
        self.native.step_into_buffers(actions, active)
        self.seconds += time.perf_counter() - started
        self.calls += 1


def _load_outputs(path: Path, tape: Tape) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if file_sha256(path) != tape.outputs_sha256:
        raise ValueError("policy output artifact hash mismatch")
    value: object = torch.load(path, weights_only=True, map_location="cpu")
    if not isinstance(value, list) or len(value) != len(tape.frames):
        raise ValueError("output/frame count mismatch")
    outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for row, frame in zip(value, tape.frames, strict=True):
        if not isinstance(row, dict) or set(row) != {"logits", "value"}:
            raise ValueError("invalid output row")
        logits, critic = row["logits"], row["value"]
        if not isinstance(logits, torch.Tensor) or not isinstance(critic, torch.Tensor):
            raise ValueError("non-tensor policy output")
        if len(logits) != frame.rows or len(critic) != frame.rows:
            raise ValueError("policy output row count mismatch")
        outputs.append((logits, critic))
    return outputs


class PreparedReplay:
    """One resident actor with separate collection and verification phases.

    Collect once, then verify while retaining the rollout. Multiple instances
    may coexist in one process; each restores its captured CPU sampling state
    before collection. They must be collected sequentially within that process
    because Torch's CPU generator is process-global. Use separate processes for
    concurrent collection. No neural forward or simulated device wait runs here.
    """

    def __init__(
        self, tape_dir: Path, *, verify_inputs: bool, request_timings: bool = False
    ) -> None:
        self.tape_dir = tape_dir
        self.verify_inputs = verify_inputs
        self.request_timings = request_timings
        tape = Tape.model_validate_json((self.tape_dir / "tape.json").read_text())
        if _world(tape.recipe) != tape.world_binding:
            raise ValueError("world/schema/setup differs from capture")
        outputs = _load_outputs(self.tape_dir / "outputs.pt", tape)
        torch.manual_seed(tape.seed)
        agent = Agent(
            ObservationSpace(tape.recipe.observation), tape.recipe.agent
        ).eval()
        collector = _collector(tape.recipe, tape.streams, tape.seed)
        timer = _TimedEnvironment(collector._env)
        collector._env = timer
        self.tape = tape
        self.outputs = outputs
        self.agent = agent
        self.collector = collector
        self.timer = timer
        self.sampling_state = torch.get_rng_state().clone()
        self.batch: RolloutBatch | None = None
        self._attempted = False
        self._cursor = 0
        self._elapsed = 0.0
        self._intervals: list[float] | None = None
        self._tail: float | None = None

    def collect(self, forward: PolicyForward | None = None) -> None:
        """Collect once and retain arrays; a failed session cannot be retried."""
        if self._attempted:
            raise RuntimeError("replay collection already attempted")
        self._attempted = True
        tape, outputs = self.tape, self.outputs
        agent, collector = self.agent, self.collector
        verify_inputs, request_timings = self.verify_inputs, self.request_timings
        torch.set_rng_state(self.sampling_state)
        cursor = 0
        request_entries: list[float] = []

        def playback(obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal cursor
            if request_timings:
                request_entries.append(time.perf_counter())
            if cursor >= len(outputs):
                raise ValueError("replay requested an extra policy call")
            frame = tape.frames[cursor]
            if len(obs["actions_valid"]) != frame.rows:
                raise ValueError("policy batch shape diverged")
            if verify_inputs and _digest_tensors(obs) != frame.input_sha256:
                raise ValueError("acting-view policy input diverged")
            logits, critic = outputs[cursor]
            cursor += 1
            if forward is not None:
                actual = forward(obs)
                if verify_inputs and not (
                    torch.equal(actual[0], logits) and torch.equal(actual[1], critic)
                ):
                    raise ValueError("live policy outputs diverged from tape")
                return actual
            # Include output materialization on the host. Device transfer latency
            # itself is a separate modeled demand, not a sleep in this measurement.
            return logits.clone(), critic.clone()

        started = time.perf_counter()
        with _policy_override(agent, playback, external=True):
            batch = collector.collect(agent, tape.transitions)
        finished = time.perf_counter()
        elapsed = finished - started
        intervals = (
            [
                entry - previous
                for previous, entry in zip(
                    [started, *request_entries[:-1]], request_entries, strict=True
                )
            ]
            if request_entries
            else None
        )
        self.batch = batch
        self._cursor = cursor
        self._elapsed = elapsed
        self._intervals = intervals
        self._tail = finished - request_entries[-1] if request_entries else None

    def verify(self) -> Replay:
        """Check every retained rollout against its tape, outside timed work."""
        if self.batch is None:
            raise RuntimeError("replay has no completed collection")
        batch, tape = self.batch, self.tape
        collector, timer = self.collector, self.timer
        cursor, elapsed = self._cursor, self._elapsed
        intervals = self._intervals
        tensors = _batch_tensors(batch)
        batch_hash = _digest_tensors(tensors)
        buffers_hash = _buffer_digest(collector)
        journal_hash = _journal_digest(collector)
        seeds = timer.current_game_seeds()
        checks = {
            "forward_calls": (cursor, len(tape.frames)),
            "batch_sha256": (batch_hash, tape.batch_sha256),
            "buffers_sha256": (buffers_hash, tape.buffers_sha256),
            "journal_sha256": (journal_hash, tape.journal_sha256),
            "game_seeds": (seeds, tape.game_seeds),
        }
        mismatches = {
            name: {"actual": actual, "expected": expected}
            for name, (actual, expected) in checks.items()
            if actual != expected
        }
        if mismatches:
            raise ValueError(
                "collector replay diverged from captured policy: "
                + json.dumps(mismatches, sort_keys=True)
            )
        buffer_bytes = sum(v.nbytes for v in collector._buffers.values())
        return Replay(
            seconds=elapsed,
            native_seconds=timer.seconds,
            demand=HostDemand(
                decisions=collector.stats.micro_steps,
                learner_transitions=collector.stats.learner_transitions,
                seconds=elapsed,
                native_seconds=timer.seconds,
                input_bytes=sum(f.input_bytes for f in tape.frames),
                output_bytes=sum(f.output_bytes for f in tape.frames),
                forward_calls=len(tape.frames),
                streams=tape.streams,
                rollout_bytes=_tensor_bytes(tensors),
                buffer_bytes=buffer_bytes,
            ),
            batch_sha256=batch_hash,
            buffers_sha256=buffers_hash,
            journal_sha256=journal_hash,
            game_seeds=seeds,
            forward_batches=dict(Counter(f.rows for f in tape.frames)),
            native_calls=timer.calls,
            games=collector.stats.games,
            truncated_games=collector.stats.truncations,
            rss_bytes=psutil.Process().memory_info().rss,
            request_host_seconds=intervals,
            tail_host_seconds=self._tail,
        )


def replay(
    tape_dir: Path, *, verify_inputs: bool, request_timings: bool = False
) -> Replay:
    """Replay exact outputs; omit input hashing only after an audited pass.

    Every call checks action journals, rollout arrays, final observation buffers
    and reset seeds. These witness the collector boundary, not the complete
    hidden authority of each native environment.
    """
    prepared = PreparedReplay(
        tape_dir, verify_inputs=verify_inputs, request_timings=request_timings
    )
    prepared.collect()
    return prepared.verify()


class HardwareRow(BaseModel):
    """Read only the existing capacity report fields needed for service demand."""

    card: str
    cell: str
    capacity: str
    kind: str
    status: str
    batch: int
    inference_observations_per_second: float | None = None
    optimizer_observations_per_second: float | None = None
    peak_allocated_bytes: int | None = None


class HardwareReport(BaseModel):
    hardware: list[HardwareRow]


def load_services(path: Path) -> list[DeviceService]:
    """No interpolation, missing-cell substitution or out-of-memory extrapolation."""
    report = HardwareReport.model_validate_json(path.read_text())
    digest = file_sha256(path)
    return [
        DeviceService(
            card=row.card,
            capacity=row.capacity,
            batch=row.batch,
            inference_seconds=row.batch / row.inference_observations_per_second,
            optimizer_seconds=row.batch / row.optimizer_observations_per_second,
            peak_allocated_bytes=row.peak_allocated_bytes,
            evidence_sha256=digest,
            cell=row.cell,
        )
        for row in report.hardware
        if row.kind == "model"
        and row.status == "completed"
        and row.inference_observations_per_second
        and row.optimizer_observations_per_second
        and row.peak_allocated_bytes
    ]


def _child(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.mode == "capture":
        if args.recipe is None:
            raise ValueError("capture requires --recipe")
        recipe = TrainingRegime.model_validate_json(args.recipe.read_text())
        capture(recipe, args.out, args.streams, args.transitions, args.seed)
    elif args.mode == "replay":
        if args.tape is None:
            raise ValueError("replay requires --tape")
        # Audited replay precedes timing; inputs must match every policy call.
        audited = replay(args.tape, verify_inputs=True)
        samples = [
            replay(args.tape, verify_inputs=False, request_timings=args.request_timings)
            for _ in range(args.repeats)
        ]
        atomic_json(
            args.out / "replay.json",
            {
                "tape_sha256": file_sha256(args.tape / "tape.json"),
                "audited": audited.model_dump(mode="json"),
                "samples": [s.model_dump(mode="json") for s in samples],
                "source": _source(),
                "native_sha256": file_sha256(Path(managym._managym.__file__)),
                "limits": "Host replay with recorded policy outputs; input audit excluded from timed repeats. No CUDA, gradients, checkpoint export or learning progress measured.",
            },
        )
    else:
        if args.replay is None or args.hardware is None:
            raise ValueError("project requires --replay and --hardware")
        data = json.loads(args.replay.read_text())
        samples = TypeAdapter(list[Replay]).validate_python(data["samples"])
        if not samples:
            raise ValueError("empty replay measurements")
        # Preserve every repeat through projections instead of hiding variation
        # behind the fastest run or treating the median as an error-free fact.
        services = load_services(args.hardware)
        scenarios = [
            Scenario(
                cpu_workers=workers,
                gpus=gpus,
                actors=actors,
                optimizer_observations_per_decision=learning,
                transfer_gb_per_second=bandwidth,
                transfer_latency_us=latency,
            )
            for workers, gpus, actors in [
                (1, 1, 1),
                (16, 1, 64),
                (128, 8, 512),
                (512, 64, 4096),
            ]
            for learning in (0.0, 0.25, 1.0)
            for bandwidth, latency in [(12.0, 20.0), (24.0, 5.0)]
        ]
        rows = [
            dict(
                repeat=i,
                **project(sample.demand, service, scenario).model_dump(mode="json"),
            )
            for i, sample in enumerate(samples)
            for service in services
            for scenario in scenarios
        ]
        atomic_json(
            args.out / "projections.json",
            {
                "schema_version": 1,
                "replay_sha256": file_sha256(args.replay),
                "hardware_sha256": file_sha256(args.hardware),
                "rows": rows,
                "unavailable_services": "Failed/missing hardware cells are omitted, never extrapolated",
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "replay", "project"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--tape", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--hardware", type=Path)
    parser.add_argument("--streams", type=int, default=16)
    parser.add_argument("--transitions", type=int, default=32)
    parser.add_argument("--seed", type=int, default=10831)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--request-timings", action="store_true")
    parser.add_argument("--seconds", type=float, default=90)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (
        1 <= args.streams <= 64
        and 1 <= args.transitions <= 64
        and 1 <= args.repeats <= 5
        and 0 < args.seconds <= 90
    ):
        parser.error("limits: 1–64 streams/transitions, 1–5 repeats, (0,90] seconds")
    if args.child:
        _child(args)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        "-m",
        "experiments.runners.pipeline_benchmark",
        *sys.argv[1:],
        "--child",
    ]
    receipt: dict[str, JsonValue] = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "platform": platform.platform(),
        "load_start": list(os.getloadavg()),
        "seconds_limit": args.seconds,
        "rss_limit_bytes": 6 * 1024**3,
        "threads": 1,
        "paid_resource_dollars": 0,
    }
    atomic_json(args.out / "attempt.json", receipt)
    started = time.monotonic()
    rss: list[dict[str, float | int]] = []
    with (
        (args.out / "stdout.log").open("w") as stdout,
        (args.out / "stderr.log").open("w") as stderr,
    ):
        process = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            env=dict(
                os.environ,
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                VECLIB_MAXIMUM_THREADS="1",
                WANDB_MODE="disabled",
            ),
        )
        status = "completed"
        try:
            while process.poll() is None:
                try:
                    used = psutil.Process(process.pid).memory_info().rss
                except psutil.NoSuchProcess:
                    used = 0
                elapsed = time.monotonic() - started
                rss.append({"seconds": elapsed, "bytes": used})
                if elapsed >= args.seconds or used > 6 * 1024**3:
                    status = "timeout" if elapsed >= args.seconds else "memory_limit"
                    break
                time.sleep(0.05)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
            receipt.update(
                status=status if code == 0 or status != "completed" else "failed",
                exit_code=code,
                seconds=time.monotonic() - started,
                load_end=list(os.getloadavg()),
                rss_samples=rss,
            )
            atomic_json(args.out / "attempt.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "rss_samples"}))
    if code != 0 or status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
