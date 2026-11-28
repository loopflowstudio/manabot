"""Capture TLA evaluation batches and measure checkpoint forwards on identical inputs.

Capture keeps the arena's original batches and independently prepared single
rows. Replay times transfer, forward and readback before profiling a bounded
sample. The offline report joins the resulting receipts.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace
import gzip
import heapq
from itertools import count, islice
import json
import os
from pathlib import Path
import pickle
from random import Random
from statistics import mean
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Literal, TypedDict, cast

import numpy as np
import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function

from experiments.eval_throughput import (
    DEALS,
    checkpoint_ref,
    random_checkpoint,
    run_vector,
    trophy_deals,
)
from manabot.backends.torch.arrays import on_device
from manabot.backends.torch.learner import forward_precision
from manabot.backends.torch.policy import ActorCritic
from manabot.eval.agent_loader import (
    device_available,
    load_torch_policy,
    load_with_precision,
    torch_forward_context,
)
from manabot.eval.agents import build, checked_logits, local_forward, prepare_batch
from manabot.eval.vector import LocalMatches
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.infra.files import atomic_json
from manabot.policy.admission import Spec, binding
from manabot.policy.batching import concatenate
from manabot.policy.prepare import Prepared
from manabot.policy.readings import Readings
from manabot.policy.spans import ObservationSelection
from manabot.sim.registration import EvalPrecision

ARENA = "eval-precision-profile"
VARIANTS = ("float32", "tf32", "bf16", "bf16-linear-once", "bf16-cached-autocast")
GRID_ROWS = (1, 4, 16, 64)
FORWARD = "policy.forward"
ATTENTION = "policy.attention"
GRU = "policy.packed_gru"

type HostBatch = Prepared[np.ndarray]
type DeviceName = Literal["cpu", "mps", "cuda"]


@dataclass(frozen=True)
class BatchContract:
    checkpoint_sha256: str
    model: ModelConfig
    observation: ObservationConfig
    readings: Readings

    def bind(self, policy: ActorCritic, *, grid: bool) -> tuple[str, str]:
        model = policy.config
        reference = self.model
        if grid:
            reference = reference.model_copy(
                update={
                    "hidden_dim": model.hidden_dim,
                    "num_attention_heads": model.num_attention_heads,
                }
            )
        space = policy.observation_space
        if (reference, self.observation, self.readings) != (
            model,
            space.config,
            policy.readings,
        ):
            raise ValueError("captured batches have a different policy input contract")
        return binding(self.model, space, self.readings), binding(
            model, space, policy.readings
        )


@dataclass(frozen=True)
class CapturedCall:
    batch: HostBatch
    singles: tuple[HostBatch, ...]
    logits: np.ndarray


def capture(
    checkpoint: Path,
    batches: Path,
    out: Path,
    *,
    device: DeviceName,
    deals: int,
    slots: int,
) -> None:
    ref = checkpoint_ref(checkpoint, ARENA)
    heuristic, specs, matches = trophy_deals(ref, deals, "float32")
    out.mkdir(parents=True, exist_ok=False)
    batches.parent.mkdir(parents=True, exist_ok=True)
    os.environ["MANABOT_EVAL_DEVICE"] = device
    os.environ["OMP_NUM_THREADS"] = "1"
    forward = load_with_precision(ref, "float32")
    pending: tuple[HostBatch, ObservationSelection, Spec] | None = None
    count = 0

    def observe(batch: HostBatch, selection: ObservationSelection, spec: Spec) -> None:
        nonlocal pending
        assert pending is None, "capture expects one checkpoint per round"
        pending = batch, selection, spec

    with NamedTemporaryFile(dir=batches.parent, suffix=".partial") as partial:
        with gzip.GzipFile(fileobj=partial, mode="wb", compresslevel=1) as stream:

            def record(inputs: list[object]) -> list[object]:
                nonlocal pending, count
                assert (
                    pending is not None and len(inputs) == 1 and inputs[0] is pending[0]
                )
                batch, selection, spec = pending
                pending = None
                if count == 0:
                    pickle.dump(
                        (
                            ref.artifact.sha256,
                            spec.model,
                            spec.space.config,
                            dict(spec.tables.layout.readings),
                        ),
                        stream,
                        protocol=5,
                    )
                outputs = forward(inputs)
                singles = (
                    (batch,)
                    if batch.batch == 1
                    else tuple(
                        prepare_batch(
                            ObservationSelection(selection.space, (coordinate,)), spec
                        )
                        for coordinate in selection.coordinates
                    )
                )
                pickle.dump(
                    (batch, singles, checked_logits(outputs, batch)),
                    stream,
                    protocol=5,
                )
                count += 1
                return outputs

            agents = (local_forward(record), build(heuristic, {}))
            source = LocalMatches([(match, agents) for match in matches])
            games = asyncio.run(
                run_vector(source, specs, slots, out, observe_prepared=observe)
            )
            if games != deals * 4 or not count:
                raise ValueError(
                    f"capture completed {games} games and {count} forwards"
                )
            # A terminator distinguishes a complete capture from a truncated one.
            pickle.dump(None, stream, protocol=5)
        partial.flush()
        # Link only complete bytes and fail if another capture owns the destination.
        os.link(partial.name, batches)
    print(f"Captured {count} calls from {games} games in {batches}")


@contextmanager
def open_capture(path: Path) -> Iterator[tuple[BatchContract, Iterator[CapturedCall]]]:
    """Read a trusted local capture from this checkout."""
    with gzip.open(path, "rb") as stream:
        contract = BatchContract(*pickle.load(stream))

        def calls() -> Iterator[CapturedCall]:
            while (call := pickle.load(stream)) is not None:
                yield CapturedCall(*call)

        yield contract, calls()


def profile_inputs(
    calls: Iterator[CapturedCall],
    grid_rows: int | None,
    limit: int | None,
    skip_rows: int,
) -> Iterator[HostBatch]:
    if grid_rows is None:
        yield from (call.batch for call in islice(calls, limit))
        return
    singles = (row for call in calls for row in call.singles)
    singles = islice(singles, skip_rows, None)
    for _ in islice(count(), limit):
        group = list(islice(singles, grid_rows))
        if len(group) != grid_rows:
            break
        yield group[0] if grid_rows == 1 else concatenate(group)


def linear_weights_bf16(policy: ActorCritic) -> None:
    """Round only weights that autocast rounds for Linear and MHA projections."""
    for module in policy.modules():
        if isinstance(module, (nn.Linear, nn.MultiheadAttention)):
            for parameter in module.parameters(recurse=False):
                parameter.data = parameter.data.to(torch.bfloat16)


def device_sync(device: torch.device) -> Callable[[], None]:
    if device.type == "cuda":
        return lambda: torch.cuda.synchronize(device)
    if device.type == "mps":
        return torch.mps.synchronize
    return lambda: None


@dataclass(frozen=True)
class CallTiming:
    rows: int
    packed_gru_rows: int
    repeat: int
    transfer_seconds: float
    forward_seconds: float
    readback_seconds: float


def run_call(
    policy: ActorCritic,
    batch: HostBatch,
    precision: EvalPrecision,
    device: torch.device,
    sync: Callable[[], None],
    repeat: int,
    *,
    profiling: bool = False,
) -> tuple[np.ndarray, CallTiming]:
    scope = record_function if profiling else lambda _: nullcontext()
    with torch_forward_context(precision, device):
        start = perf_counter()
        with scope("policy.transfer"):
            moved = on_device(batch, device)
            sync()
        transferred = perf_counter()
        with scope(FORWARD):
            scores, _ = policy(moved)
            sync()
        forwarded = perf_counter()
    with scope("policy.readback"):
        logits = scores.float().cpu().numpy()
        sync()
    read = perf_counter()
    logits[~batch.legal] = -np.inf
    schedules = (batch.facts.programs, batch.effects, batch.memory, batch.programs)
    return logits, CallTiming(
        batch.batch,
        sum(sum(rows.schedule.sizes) for rows in schedules if rows is not None),
        repeat,
        transferred - start,
        forwarded - transferred,
        read - forwarded,
    )


@contextmanager
def profile_modules(policy: ActorCritic) -> Iterator[None]:
    """Label attention projections and packed GRU operators in the CPU trace."""
    with ExitStack() as hooks, ExitStack() as scopes:
        for module in policy.modules():
            if isinstance(module, nn.MultiheadAttention):
                label = ATTENTION
            elif isinstance(module, nn.GRU):
                label = GRU
            else:
                continue

            def enter(
                _module: nn.Module, _inputs: tuple[object, ...], label: str = label
            ) -> None:
                scopes.enter_context(record_function(label))

            def leave(
                _module: nn.Module, _inputs: tuple[object, ...], _output: object
            ) -> None:
                scopes.close()

            hooks.callback(module.register_forward_pre_hook(enter).remove)
            hooks.callback(module.register_forward_hook(leave, always_call=True).remove)
        yield


class TraceEvent(TypedDict):
    cat: str
    name: str
    pid: int
    tid: int
    ts: float
    dur: float
    args: dict[str, int]


class ForwardProfile(TypedDict):
    calls: int
    profiled_forward_seconds: float
    wall_buckets_seconds: dict[str, float]
    cuda_buckets_seconds: dict[str, float] | None
    sdpa_dispatches: dict[str, int]
    native_mha_dispatches: int
    cuda_attention_kernels: dict[str, int] | None


def op_bucket(name: str, parent: str) -> str:
    if name in {"aten::to", "aten::_to_copy", "aten::copy_"}:
        return "casts_and_copies"
    if name in {ATTENTION, GRU}:
        return name.removeprefix("policy.")
    if parent != "other":
        return parent
    if name in {
        "aten::linear",
        "aten::addmm",
        "aten::mm",
        "aten::matmul",
        "aten::relu",
        "aten::gelu",
        "aten::silu",
    }:
        return "linear_and_mlp"
    if "norm" in name:
        return "normalization"
    if name.startswith(
        ("aten::index", "aten::gather", "aten::embedding", "aten::scatter")
    ):
        return "gather_and_index"
    return "other"


def forward_scopes(events: list[TraceEvent]) -> list[TraceEvent]:
    # Kineto also mirrors these annotations onto the GPU timeline.
    return [
        event
        for event in events
        if event["cat"] == "user_annotation" and event["name"] == FORWARD
    ]


def bucket_trace(
    events: list[TraceEvent],
) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    """Partition profiled forward wall time. A running GPU kernel owns overlap."""
    host: list[tuple[float, float, str]] = []
    device: list[tuple[float, float, str]] = []
    by_launch: dict[int, str] = {}
    by_op: dict[int, str] = {}
    attention_kernels: defaultdict[str, int] = defaultdict(int)
    for root in forward_scopes(events):
        start, end = root["ts"], root["ts"] + root["dur"]
        cpu = sorted(
            (
                event
                for event in events
                if (event["pid"], event["tid"]) == (root["pid"], root["tid"])
                and start <= event["ts"] < end
            ),
            key=lambda event: (event["ts"], -event["dur"]),
        )
        stack: list[tuple[float, str]] = []
        for event in cpu:
            start_at = event["ts"]
            while stack and stack[-1][0] <= start_at:
                stack.pop()
            bucket = op_bucket(event["name"], stack[-1][1] if stack else "other")
            end_at = min(start_at + event["dur"], end)
            stack.append((end_at, bucket))
            host.append((start_at, end_at, bucket))
            correlation = event.get("args", {}).get("correlation")
            if correlation is not None:
                by_launch[correlation] = bucket
            external = event.get("args", {}).get("External id")
            if external is not None and event["cat"] == "cpu_op":
                by_op[external] = bucket
    for event in events:
        if event["cat"] not in {"kernel", "gpu_memcpy", "gpu_memset"}:
            continue
        args = event.get("args", {})
        correlation, external = args.get("correlation"), args.get("External id")
        bucket = by_launch.get(correlation) if correlation is not None else None
        if bucket is None and external is not None:
            bucket = by_op.get(external)
        if bucket is not None:
            device.append((event["ts"], event["ts"] + event["dur"], bucket))
            if bucket == "attention" and event["cat"] == "kernel":
                attention_kernels[event["name"]] += 1

    buckets: defaultdict[str, float] = defaultdict(float)
    gpu_buckets: defaultdict[str, float] = defaultdict(float)
    # The deepest CPU interval wins. The latest-started CUDA kernel wins when
    # kernels overlap, so no elapsed time is charged twice in either case.
    spans = [(*span, False) for span in host] + [(*span, True) for span in device]
    boundaries = sorted(
        (time, opening, i)
        for i, (start, end, _, _) in enumerate(spans)
        if start < end
        for time, opening in ((start, True), (end, False))
    )
    active: list[tuple[bool, float, float, int]] = []
    previous = 0.0
    for time, opening, i in boundaries:
        while active and spans[active[0][3]][1] <= previous:
            heapq.heappop(active)
        if active and time > previous:
            _, _, bucket, is_gpu = spans[active[0][3]]
            buckets[bucket] += (time - previous) / 1e6
            if is_gpu:
                gpu_buckets[bucket] += (time - previous) / 1e6
        if opening:
            start, end, _, is_gpu = spans[i]
            heapq.heappush(active, (not is_gpu, -start, end, i))
        previous = time
    return dict(buckets), dict(gpu_buckets), dict(attention_kernels)


def profile_forward(
    policy: ActorCritic,
    batches: Sequence[HostBatch],
    precision: EvalPrecision,
    device: torch.device,
    sync: Callable[[], None],
    trace: Path,
) -> ForwardProfile:
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile_modules(policy), profile(activities=activities) as profiler:
        for batch in batches:
            run_call(policy, batch, precision, device, sync, 0, profiling=True)
    profiler.export_chrome_trace(str(trace))
    events = [
        cast(TraceEvent, event)
        for event in json.loads(trace.read_bytes())["traceEvents"]
        if event.get("ph") == "X" and event["dur"] > 0
    ]
    buckets, device_buckets, attention_kernels = bucket_trace(events)
    elapsed = sum(event["dur"] / 1e6 for event in forward_scopes(events))
    if not elapsed or abs(sum(buckets.values()) - elapsed) > elapsed * 0.05:
        raise ValueError("profile buckets do not cover the recorded forward scopes")
    if device.type == "cuda" and not device_buckets:
        raise ValueError("CUDA trace has no kernels linked to the recorded forwards")
    dispatches: defaultdict[str, int] = defaultdict(int)
    native_mha_dispatches = 0
    for event in events:
        if event["cat"] != "cpu_op":
            continue
        if "scaled_dot_product" in event["name"]:
            dispatches[event["name"]] += 1
        elif event["name"] == "aten::_native_multi_head_attention":
            native_mha_dispatches += 1
    return {
        "calls": len(batches),
        "profiled_forward_seconds": elapsed,
        "wall_buckets_seconds": buckets,
        "cuda_buckets_seconds": device_buckets if device.type == "cuda" else None,
        "sdpa_dispatches": dict(dispatches),
        "native_mha_dispatches": native_mha_dispatches,
        "cuda_attention_kernels": dict(attention_kernels)
        if device.type == "cuda"
        else None,
    }


def bound_inputs(
    batches: Path,
    policy: ActorCritic,
    checkpoint_sha256: str,
    grid_rows: int | None,
    limit: int | None,
    skip_rows: int,
) -> Iterator[HostBatch]:
    with open_capture(batches) as (contract, calls):
        grid = grid_rows is not None
        if not grid and contract.checkpoint_sha256 != checkpoint_sha256:
            raise ValueError(
                "arena batches require the capturing checkpoint's exact bytes"
            )
        source, target = contract.bind(policy, grid=grid)
        for batch in profile_inputs(calls, grid_rows, limit, skip_rows):
            if batch.binding != source:
                raise ValueError(
                    "captured batch does not match its policy input contract"
                )
            yield replace(batch, binding=target) if source != target else batch


def logit_difference(
    actual: np.ndarray, expected: np.ndarray, legal: np.ndarray
) -> tuple[bool, float]:
    actual, expected = actual[legal], expected[legal]
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("non-finite legal logits in replay")
    return bool(np.array_equal(actual, expected)), float(
        np.abs(actual - expected).max()
    )


def replay(
    checkpoint: Path,
    batches: Path,
    out: Path,
    *,
    device_name: DeviceName,
    variant: str,
    grid_rows: int | None = None,
    limit: int | None = None,
    skip_rows: int = 0,
    repeats: int = 2,
    profile_calls: int = 4,
) -> dict[str, object]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown replay variant {variant!r}")
    precision = cast(EvalPrecision, "bf16" if variant.startswith("bf16") else variant)
    device = torch.device(device_name)
    sync = device_sync(device)
    digest = checkpoint_ref(checkpoint, ARENA).artifact.sha256
    assert digest is not None
    out.mkdir(parents=True, exist_ok=False)

    def inputs(policy: ActorCritic) -> Iterator[HostBatch]:
        return bound_inputs(batches, policy, digest, grid_rows, limit, skip_rows)

    policy = load_torch_policy(checkpoint, precision, device)
    if variant == "bf16-linear-once":
        linear_weights_bf16(policy)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def cached_context() -> AbstractContextManager[object]:
        return (
            forward_precision(precision, device)
            if variant == "bf16-cached-autocast"
            else nullcontext()
        )

    calls: list[CallTiming] = []
    sample: list[tuple[int, HostBatch]] = []
    compare_bf16 = variant != "bf16" and device.type != "cpu"
    first_logits: list[np.ndarray] = []
    exact = True
    difference = 0.0
    rng = Random(1)
    with (out / "calls.jsonl").open("w") as receipt, cached_context():
        for repeat in range(repeats + 1):
            for index, batch in enumerate(inputs(policy)):
                logits, timing = run_call(
                    policy, batch, precision, device, sync, repeat
                )
                calls.append(timing)
                receipt.write(json.dumps({"call": index, **asdict(timing)}) + "\n")
                if repeat:
                    continue
                if compare_bf16:
                    first_logits.append(logits)
                if index < profile_calls:
                    sample.append((index, batch))
                else:
                    selected = rng.randrange(index + 1)
                    if selected < profile_calls:
                        sample[selected] = index, batch
            if not calls:
                raise ValueError("no captured calls fit the requested replay")
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    warm = [call for call in calls if call.repeat]
    stages = ("transfer_seconds", "forward_seconds", "readback_seconds")
    stage_means = {
        stage: mean(getattr(call, stage) for call in warm) for stage in stages
    }
    first_means = {
        stage: mean(getattr(call, stage) for call in calls if not call.repeat)
        for stage in stages
    }
    sample.sort()
    with cached_context():
        # Prime this separate profiling context without recording a cold cast.
        run_call(policy, sample[0][1], precision, device, sync, 0)
        breakdown = profile_forward(
            policy,
            [batch for _, batch in sample],
            precision,
            device,
            sync,
            out / "profile.json",
        )
    selected_indices = {index for index, _ in sample}
    calls_per_repeat = len(calls) // (repeats + 1)
    sample_mean = mean(
        call.forward_seconds
        for offset, call in enumerate(calls)
        if call.repeat and offset % calls_per_repeat in selected_indices
    )
    profiled_total = breakdown["profiled_forward_seconds"]
    raw_buckets = breakdown["wall_buckets_seconds"]
    model = policy.config
    if compare_bf16:
        del policy
        reference = load_torch_policy(checkpoint, "bf16", device)
        for batch, logits in zip(inputs(reference), first_logits, strict=True):
            baseline, _ = run_call(reference, batch, "bf16", device, sync, 0)
            same, delta = logit_difference(logits, baseline, batch.legal)
            exact &= same
            difference = max(difference, delta)
    summary = {
        "checkpoint_sha256": digest,
        "batches": str(batches),
        "width": model.hidden_dim,
        "heads": model.num_attention_heads,
        "head_dim": model.hidden_dim // model.num_attention_heads,
        "device": device_name,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "variant": variant,
        "grid_rows": grid_rows,
        "grid_skip_rows": skip_rows,
        "mean_rows": mean(call.rows for call in warm),
        "mean_packed_gru_rows": mean(call.packed_gru_rows for call in warm),
        "calls_per_repeat": calls_per_repeat,
        "first_touch_stage_means": first_means,
        "repeat_stage_means": stage_means,
        "peak_cuda_bytes": peak,
        "exact_bf16_logits": exact if compare_bf16 or variant == "bf16" else None,
        "max_abs_difference_from_bf16": difference
        if compare_bf16 or variant == "bf16"
        else None,
        "profile_synced_sample_mean_seconds": sample_mean,
        "profile_elapsed_over_synced": profiled_total / len(sample) / sample_mean,
        "profile_estimated_bucket_mean_seconds": {
            bucket: seconds / profiled_total * sample_mean
            for bucket, seconds in raw_buckets.items()
        },
        **breakdown,
    }
    atomic_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def render_report(receipts: Path, output: Path) -> None:
    profiles: list[tuple[str, dict[str, object]]] = []
    arena: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for path in sorted(receipts.rglob("summary.json")):
        summary = json.loads(path.read_bytes())
        if "variant" in summary:
            profiles.append((str(path.parent.relative_to(receipts)), summary))
        elif summary.get("device") == "cuda" and summary.get("runner") == "vector":
            arena[(summary["model"], summary["precision"])].append(
                summary["stage_seconds"]["forward"]
            )
    if not profiles:
        raise ValueError(f"no replay summaries under {receipts}")
    lines = [
        "# Evaluation precision receipts",
        "",
        "Stage times are uninstrumented repeat means in milliseconds per call.",
        "Bucket estimates apply the profiled wall-time shares to the same calls' uninstrumented mean.",
        "GPU work owns intervals that overlap with host dispatch.",
        "",
        "| Replay arm | Head width | Rows/call | Packed GRU rows/call | Transfer (ms) | Forward (ms) | Readback (ms) | CUDA peak (MiB) | Attention operators | Exact bf16 logits |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for arm, row in profiles:
        stages = cast(dict[str, float], row["repeat_stage_means"])
        dispatches = cast(dict[str, int], row["sdpa_dispatches"])
        operators = [name.removeprefix("aten::") for name in dispatches]
        if row.get("native_mha_dispatches"):
            operators.append("_native_multi_head_attention")
        peak = cast(int | None, row["peak_cuda_bytes"])
        peak_mib = f"{peak / 2**20:.1f}" if peak is not None else "—"
        transfer, forward, readback = (
            1e3 * stages[f"{stage}_seconds"]
            for stage in ("transfer", "forward", "readback")
        )
        exact = row["exact_bf16_logits"]
        lines.append(
            f"| {arm} | {row['head_dim']} | {row['mean_rows']:.2f} | {row['mean_packed_gru_rows']:.1f} | "
            f"{transfer:.3f} | {forward:.3f} | {readback:.3f} | "
            f"{peak_mib} | {', '.join(operators) or '—'} | {exact if exact is not None else '—'} |"
        )
    columns = sorted(
        {
            key
            for _, row in profiles
            for key in cast(
                dict[str, float], row["profile_estimated_bucket_mean_seconds"]
            )
        }
    )
    lines.extend(
        [
            "",
            "## Estimated forward time by bucket (ms/call)",
            "",
            "| Replay arm | "
            + " | ".join(name.replace("_", " ") for name in columns)
            + " |",
            "| --- | " + " | ".join("---:" for _ in columns) + " |",
        ]
    )
    for arm, row in profiles:
        values = cast(dict[str, float], row["profile_estimated_bucket_mean_seconds"])
        lines.append(
            f"| {arm} | "
            + " | ".join(f"{1e3 * values.get(name, 0):.3f}" for name in columns)
            + " |"
        )
    if arena:
        lines.extend(
            [
                "",
                "## Vector arena repeats",
                "",
                "| Checkpoint | Precision | Repeats | Forward (s), min / mean / max |",
                "| --- | --- | ---: | --- |",
            ]
        )
        for (model, precision), values in sorted(arena.items()):
            lines.append(
                f"| {model} | {precision} | {len(values)} | "
                f"{min(values):.2f} / {mean(values):.2f} / {max(values):.2f} |"
            )
    output.write_text("\n".join(lines) + "\n")


def positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture_cli = commands.add_parser("capture")
    weights = capture_cli.add_mutually_exclusive_group(required=True)
    weights.add_argument("--checkpoint", type=Path)
    weights.add_argument("--random-width", type=positive)
    capture_cli.add_argument(
        "--template", type=Path, help="saved input contract for seeded weights"
    )
    capture_cli.add_argument("--heads", type=positive, default=8)
    capture_cli.add_argument(
        "--deals", type=int, choices=range(1, DEALS + 1), default=DEALS
    )
    capture_cli.add_argument("--slots", type=positive, default=64)
    replay_cli = commands.add_parser("replay")
    replay_cli.add_argument("--checkpoint", required=True, type=Path)
    replay_cli.add_argument("--variant", choices=VARIANTS, required=True)
    replay_cli.add_argument("--grid-rows", type=int, choices=GRID_ROWS)
    replay_cli.add_argument("--skip-rows", type=int, default=0)
    replay_cli.add_argument("--limit", type=positive)
    replay_cli.add_argument("--repeats", type=positive, default=2)
    replay_cli.add_argument("--profile-calls", type=positive, default=4)
    for cmd in (capture_cli, replay_cli):
        cmd.add_argument("--device", choices=("cpu", "mps", "cuda"), required=True)
        cmd.add_argument("--batches", type=Path, required=True)
        cmd.add_argument("--output-dir", type=Path, required=True)
    report_cli = commands.add_parser("render_report")
    report_cli.add_argument("--receipts", type=Path, required=True)
    report_cli.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "render_report":
        render_report(args.receipts, args.output)
        return
    if not device_available(args.device):
        parser.error(f"--device {args.device} is unavailable on this host")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error(f"--checkpoint {args.checkpoint} is not a file")
    if args.output_dir.exists():
        parser.error(f"--output-dir {args.output_dir} exists")
    if args.command == "capture":
        if args.template is not None and (
            args.checkpoint is not None or not args.template.is_file()
        ):
            parser.error(
                "--template requires --random-width and an existing checkpoint"
            )
        if args.batches.exists():
            parser.error(f"--batches {args.batches} exists")
        checkpoint = args.checkpoint
        if checkpoint is None:
            if args.random_width % args.heads:
                parser.error("--random-width must be divisible by --heads")
            args.batches.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = args.batches.with_suffix(".safetensors")
            if checkpoint.exists():
                parser.error(f"generated checkpoint {checkpoint} exists")
            torch.set_num_threads(1)
            random_checkpoint(
                checkpoint, args.random_width, args.heads, template=args.template
            )
        capture(
            checkpoint,
            args.batches,
            args.output_dir,
            device=args.device,
            deals=args.deals,
            slots=args.slots,
        )
    else:
        if not args.batches.is_file():
            parser.error(f"--batches {args.batches} is not a capture file")
        if args.skip_rows < 0 or (args.skip_rows and args.grid_rows is None):
            parser.error("--skip-rows must be nonnegative and requires --grid-rows")
        replay(
            args.checkpoint,
            args.batches,
            args.output_dir,
            device_name=args.device,
            variant=args.variant,
            grid_rows=args.grid_rows,
            limit=args.limit,
            skip_rows=args.skip_rows,
            repeats=args.repeats,
            profile_calls=args.profile_calls,
        )


if __name__ == "__main__":
    main()
