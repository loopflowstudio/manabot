"""MTG-318: can CUDA graphs remove the host-bound small-call tail of collection?

Pre-registered in `collection_tail_graphs.md`. One rented L4 runs, in order:

1. `insitu`: the frozen MTG-267 w160l8 regime (`experiments.tla_frontier.frozen()`)
   on the current world with `Execution(bucketed_shapes=True)`, for `UPDATES`
   real updates from its own initialization. Hooks leave training unchanged:
   every policy call records its rows, active streams, bucket signature and
   phase times; rollout recording is split into its observation-store append
   (time, rows, store growth, minor page faults), action packing, and the
   rest, with Python garbage-collection pauses attributed to the phase they
   interrupt. A spread of real observations is kept per row band, weighted to
   the small-call tail.
2. `probe`: on one small retained input, the remaining synchronizing CUDA calls
   of a bucketed collection call by site, `torch._dynamo.explain` graph breaks
   by reason and site, and one attempt to capture the whole forward as a CUDA
   graph (its own process: a failed capture can poison the context).
3. `replay`: the retained inputs through collection's preparation, forward and
   action selection, one process per arm and size:
   `eager` (default shapes), `bucketed`, `compiled` (bucketed under
   `torch.compile(mode="reduce-overhead")`), and `trunk-graph` (bucketed, with
   the attention trunk, the only host-read-free stretch of the forward,
   replayed from one hand-captured CUDA graph per shape). Each records per-input
   median latency, first-call (compile or capture) cost, outputs against
   `bucketed`, compile counters and one profiled pass for GPU busy and kernels.
4. `analyze`: `summary.json` with per-band latencies, decision-rule inputs, the
   whole-collection projection over the in-situ call mix and the recording
   decomposition.

Collection makes one policy call per environment step over every active stream
(`SeatRoutedCollector._collect`), so there are no concurrent small requests to
coalesce; the in-situ receipt checks this (calls == steps) and measures the
tail as active-stream counts instead.

`job` runs every stage under Etude and leaves `results.tar.gz` in
ETUDE_OUTPUT_DIR. `job --smoke` runs them on CPU at toy scale.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
import gc
import gzip
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from typing import Any

import numpy as np
import torch

from experiments.l4_step_profile import (
    SEED,
    SIZES,
    SyncCounter,
    _host,
    _profiler,
    _state_digest,
    _sync,
    _write,
    trace_windows,
)

ROOT = Path(__file__).resolve().parents[1]
UPDATES = 3
ARMS = ("eager", "bucketed", "compiled", "trunk-graph")
REFERENCE = "bucketed"
# Row bands (inclusive upper bounds) and how many in-situ inputs to keep in
# each, taking every `stride`-th call of the band during update 1.
BANDS: tuple[tuple[str, int, int, int], ...] = (
    # name, upper rows, keep, stride
    ("1", 1, 5, 60),
    ("2", 2, 4, 25),
    ("3-4", 4, 4, 12),
    ("5-8", 8, 4, 8),
    ("9-16", 16, 3, 6),
    ("17-32", 32, 3, 6),
    ("33-128", 128, 2, 20),
    ("129+", 1 << 30, 2, 40),
)
TIMED_REPEATS = 15
WARM_REPEATS = 2
SMOKE_REPEATS = 2


def band(rows: int) -> str:
    return next(name for name, upper, _, _ in BANDS if rows <= upper)


# --------------------------------------------------------------------------
# Stage 1: in-situ updates


@dataclass
class _Clock:
    """Phase attribution for garbage-collection pauses."""

    phase: str = "other"
    started: float = 0.0


def insitu(out: Path, captures: Path, smoke: bool = False) -> None:
    from experiments.tla_frontier import frozen
    import manabot.model.inference as inference_module
    import manabot.training.execution as execution_module
    from manabot.training.regime import TrainingRegime
    import manabot.training.rollout.collection as collection_module
    from manabot.training.rollout.collection import SeatRoutedCollector
    from manabot.verify.store import VerifyStore

    out.mkdir(parents=True, exist_ok=True)
    (captures / "inference").mkdir(parents=True, exist_ok=True)
    regimes, protocol = frozen()
    data = regimes["w160l8"].model_dump()
    stage = data["stages"][0]
    stage["updates"] = 2 if smoke else UPDATES
    stage["execution"].update(wall_seconds=7200.0, bucketed_shapes=True)
    if smoke:
        stage.update(streams=4, transitions=2)
        stage["execution"].update(device="cpu", workers=1)
    data["wall_seconds"] = 7200.0
    regime = TrainingRegime.model_validate(data)
    _write(captures / "regime.json", regime.model_dump(mode="json"))
    receipt: dict[str, Any] = {
        "host": _host(),
        "seed": protocol.training_seed,
        "regime": regime.id,
        "execution": stage["execution"],
        "updates": [],
    }
    clock = _Clock()
    gc_pauses: dict[str, float] = defaultdict(float)
    gc_counts: Counter[str] = Counter()

    def on_gc(phase: str, info: dict[str, int]) -> None:
        if phase == "start":
            clock.started = time.perf_counter()
            return
        name = f"{clock.phase}/gen{info['generation']}"
        gc_pauses[name] += time.perf_counter() - clock.started
        gc_counts[name] += 1

    calls: list[dict[str, Any]] = []
    kept: Counter[str] = Counter()
    seen: Counter[str] = Counter()
    current: dict[str, Any] = {}
    state: dict[str, Any] = {"update": 0, "steps": 0}

    original_actor = execution_module.ActorCritic
    original_prepare = inference_module.PolicyInference.prepare
    original_sample = SeatRoutedCollector._sample_learning_rows
    original_slice = SeatRoutedCollector._slice_obs_tensors
    original_append = collection_module._ObservationRows.append
    original_pack = collection_module.PackedActions.pack
    original_collect = SeatRoutedCollector.collect
    original_step = SeatRoutedCollector._settle_step

    def make_actor(space: Any, config: Any, **kwargs: Any) -> Any:
        agent = original_actor(space, config, **kwargs)
        if not (captures / "initial.pt").exists():
            torch.save(
                {k: v.detach().cpu().clone() for k, v in agent.state_dict().items()},
                captures / "initial.pt",
            )
        return agent

    def prepare(self: Any, obs: dict[str, torch.Tensor]) -> Any:
        prepared = original_prepare(self, obs)
        if current:
            buckets = prepared.buckets
            current["buckets"] = None if buckets is None else dict(buckets.sizes)
            if current.pop("keep", False):
                name = f"{len(calls):05d}"
                torch.save(
                    {k: v.clone() for k, v in obs.items()},
                    captures / "inference" / f"{name}.pt",
                )
                current["kept"] = name
        return prepared

    def slice_obs(self: Any, rows: np.ndarray, inference: Any) -> Any:
        tick = time.perf_counter()
        clock.phase = "preparation"
        result = original_slice(self, rows, inference)
        current["preparation_s"] = time.perf_counter() - tick
        clock.phase = "inference"
        current["inference_started"] = time.perf_counter()
        return result

    def append(self: Any, rows: np.ndarray) -> int:
        if "inference_started" in current:
            current.setdefault(
                "inference_s", time.perf_counter() - current["inference_started"]
            )
        clock.phase = "recording"
        capacity = self.capacity
        faults = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
        tick = time.perf_counter()
        first = original_append(self, rows)
        current["append_s"] = time.perf_counter() - tick
        current["append_faults"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_minflt - faults
        )
        current["append_bytes"] = int(
            sum(a[0].nbytes for a in self._arrays.values()) * len(rows)
        )
        if self.capacity != capacity:
            current["grew"] = [capacity, self.capacity]
        return first

    def pack(cls: Any, *args: Any) -> Any:
        tick = time.perf_counter()
        result = original_pack.__func__(cls, *args)
        if current:
            current["pack_s"] = time.perf_counter() - tick
        return result

    def sample(self: Any, agent: Any, rows: np.ndarray, *args: Any) -> Any:
        count = len(rows)
        name = band(count)
        current.clear()
        current.update(
            update=state["update"],
            rows=count,
            gc_before=sum(gc_pauses.values()),
        )
        if state["update"] == 0:
            seen[name] += 1
            quota, stride = next((k, s) for b, _, k, s in BANDS if b == name)
            stride = 1 if smoke else stride
            if kept[name] < quota and (seen[name] - 1) % stride == 0:
                kept[name] += 1
                current["keep"] = True
        tick = time.perf_counter()
        result = original_sample(self, agent, rows, *args)
        current["seconds"] = time.perf_counter() - tick
        current["recording_s"] = (
            current["seconds"]
            - current.get("preparation_s", 0.0)
            - current.get("inference_s", 0.0)
        )
        current["gc_s"] = sum(gc_pauses.values()) - current.pop("gc_before")
        current.pop("inference_started", None)
        current.pop("keep", None)
        calls.append(dict(current))
        current.clear()
        clock.phase = "other"
        return result

    def settle(self: Any, *args: Any, **kwargs: Any) -> Any:
        state["steps"] += 1
        return original_step(self, *args, **kwargs)

    def collect(self: SeatRoutedCollector, *args: Any, **kwargs: Any) -> Any:
        first_call = len(calls)
        steps = state["steps"]
        gc_before = dict(gc_pauses)
        faults = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
        tick = time.perf_counter()
        batch = original_collect(self, *args, **kwargs)
        seconds = time.perf_counter() - tick
        receipt["updates"].append(
            {
                "update": state["update"],
                "collection_seconds": seconds,
                "rows": int(batch.actions.size),
                "calls": len(calls) - first_call,
                "steps": state["steps"] - steps,
                "minor_faults": resource.getrusage(resource.RUSAGE_SELF).ru_minflt
                - faults,
                "max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "phases": self.profiler.get_stats(),
                "gc_s": {k: v - gc_before.get(k, 0.0) for k, v in gc_pauses.items()},
                "idle_stream_steps": int(self.stats.idle_stream_steps),
            }
        )
        state["update"] += 1
        return batch

    # Etude's unpacked bundle has no .git; the launch names the frozen commit.
    if revision := os.environ.get("MTG318_REVISION"):
        execution_module.source_commit = lambda: revision
        receipt["source_commit"] = revision
    execution_module.ActorCritic = make_actor
    inference_module.PolicyInference.prepare = prepare  # type: ignore[method-assign]
    SeatRoutedCollector._sample_learning_rows = sample  # type: ignore[method-assign]
    SeatRoutedCollector._slice_obs_tensors = slice_obs  # type: ignore[method-assign]
    SeatRoutedCollector._settle_step = settle  # type: ignore[method-assign]
    SeatRoutedCollector.collect = collect  # type: ignore[method-assign]
    collection_module._ObservationRows.append = append  # type: ignore[method-assign]
    collection_module.PackedActions.pack = classmethod(pack)  # type: ignore[assignment]
    gc.callbacks.append(on_gc)
    started = time.perf_counter()
    try:
        with VerifyStore(captures / "store.sqlite") as store:
            result = execution_module.execute_regime(
                regime,
                protocol.training_seed,
                captures / "execution",
                store,
                experiment="collection_tail_graphs",
                training_exposure=False,
            )
        receipt["status"] = str(result.status)
    finally:
        gc.callbacks.remove(on_gc)
        receipt["wall_seconds"] = time.perf_counter() - started
        receipt["calls"] = calls
        receipt["kept"] = dict(kept)
        receipt["gc_s"] = dict(gc_pauses)
        receipt["gc_counts"] = dict(gc_counts)
        _write(out / "receipt.json", receipt)


# --------------------------------------------------------------------------
# Shared replay pieces


def _inputs(captures: Path) -> list[tuple[str, dict[str, torch.Tensor]]]:
    paths = sorted((captures / "inference").glob("*.pt"))
    return [(p.stem, torch.load(p, weights_only=False)) for p in paths]


def _agent(captures: Path, size: str, device: str, *, bucketed: bool) -> Any:
    from manabot.backends.torch.policy import ActorCritic
    from manabot.env.observation import ObservationSpace
    from manabot.training.regime import TrainingRegime

    regime = TrainingRegime.model_validate_json((captures / "regime.json").read_text())
    config = regime.agent.model_copy(update=SIZES[size])
    torch.manual_seed(SEED)
    agent = ActorCritic(
        ObservationSpace(regime.observation), config, bucketed_shapes=bucketed
    )
    if not SIZES[size]:
        agent.load_state_dict(torch.load(captures / "initial.pt"))
    return agent.to(device).eval()


Forward = Callable[[Any], tuple[torch.Tensor, torch.Tensor]]


def call(
    agent: Any,
    forward: Forward,
    inference: Any,
    obs: dict[str, torch.Tensor],
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Collection's preparation and `Policy inference and action selection` block
    (`SeatRoutedCollector._sample_learning_rows`), with `forward` in place of the
    policy's own `forward_distribution`."""
    from manabot.model.policy_distribution import legal_mask, policy_logs, require

    obs_t = inference.prepare(obs)
    with torch.no_grad():
        logits, raw = forward(obs_t)
        logits, raw = logits.float(), raw.float()
        if agent.config.value_kind == "categorical_wdl":
            outcome = raw.softmax(dim=-1)
            value_t = outcome[:, 2] - outcome[:, 0]
            outcome.cpu().numpy()
        else:
            value_t = raw.squeeze(-1)
        logs = policy_logs(logits, legal_mask(obs_t["actions_valid"]))
        require(torch.isfinite(value_t), "nonfinite_collection_values", values=value_t)
        weights = logs.exp()
        action_t = torch.multinomial(weights, 1, generator=generator).squeeze(-1)
        logprob_t = logs.gather(-1, action_t[:, None]).squeeze(-1)
        probabilities = weights.cpu()
        log_probabilities = logs.cpu()
        action_t.cpu().numpy()
        logprob_t.cpu().numpy()
        values = value_t.view(-1).cpu()
    return {
        "logits": logits.detach().cpu(),
        "values": values,
        "logs": log_probabilities,
        "probabilities": probabilities,
    }


def _signature(obs: Any) -> str:
    buckets = getattr(obs, "buckets", None)
    sizes = (
        ""
        if buckets is None
        else ",".join(f"{k}{v}" for k, v in sorted(buckets.sizes.items()))
    )
    return f"B{len(obs['actions_valid'])}:{sizes}"


class TrunkGraphs:
    """The attention trunk replayed from one CUDA graph per input shape.

    The trunk (`attention` and `extra_attention` over objects, ownership and
    padding mask) reads nothing on the host on CUDA, so it can be captured as
    is; the rest of the forward cannot (see the probe). Graphs share one memory
    pool; replay copies the inputs into the captured buffers and returns the
    captured output buffer, which the forward consumes before the next call.
    """

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs: dict[tuple[Any, ...], tuple[Any, ...]] = {}
        self.capture_seconds: dict[str, float] = {}

    def _trunk(
        self, objects: torch.Tensor, is_agent: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        agent = self.agent
        objects = agent.attention(objects, is_agent, key_padding_mask=mask)
        for layer in agent.extra_attention:
            objects = layer(objects, is_agent, mask)
        return objects

    def __call__(
        self, objects: torch.Tensor, is_agent: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        key = (tuple(objects.shape), tuple(is_agent.shape), tuple(mask.shape))
        entry = self.graphs.get(key)
        if entry is None:
            tick = time.perf_counter()
            static = (objects.clone(), is_agent.clone(), mask.clone())
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._trunk(*static)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                output = self._trunk(*static)
            torch.cuda.synchronize()
            entry = (graph, static, output)
            self.graphs[key] = entry
            self.capture_seconds[str(key[0])] = time.perf_counter() - tick
        graph, static, output = entry
        for target, value in zip(static, (objects, is_agent, mask), strict=True):
            target.copy_(value)
        graph.replay()
        return output


def _install_trunk_graphs(agent: Any) -> TrunkGraphs:
    """Route `_attend_objects` (device inference path) through `TrunkGraphs`."""
    graphs = TrunkGraphs(agent)
    original = agent._attend_objects

    def attend(
        objects: torch.Tensor, is_agent: torch.Tensor, validity: torch.Tensor
    ) -> torch.Tensor:
        if objects.device.type != "cuda" or not agent.config.attention_on:
            return original(objects, is_agent, validity)
        mask = validity == 0
        if agent.value_token is not None:
            objects = torch.cat(
                (objects, agent.value_token.expand(objects.shape[0], -1, -1)), dim=1
            )
            mask = torch.cat((mask, torch.zeros_like(mask[:, :1])), dim=1)
        return graphs(objects, is_agent, mask)

    agent._attend_objects = attend
    return graphs


def _forward(agent: Any, arm: str, device: str) -> tuple[Forward, Any]:
    if arm in {"eager", "bucketed"}:
        return agent.forward_distribution, None
    if arm == "compiled":
        import torch._dynamo

        torch._dynamo.config.cache_size_limit = 256
        if device != "cuda":
            # Smoke: exercise the path without paying for inductor on a laptop.
            return torch.compile(agent.forward_distribution, backend="eager"), None
        compiled = torch.compile(agent.forward_distribution, mode="reduce-overhead")

        def forward(obs: Any) -> tuple[torch.Tensor, torch.Tensor]:
            torch.compiler.cudagraph_mark_step_begin()
            return compiled(obs)

        return forward, None
    if arm == "trunk-graph":
        if device != "cuda":
            return agent.forward_distribution, None
        graphs = _install_trunk_graphs(agent)
        return agent.forward_distribution, graphs
    raise ValueError(f"unknown arm {arm!r}")


# --------------------------------------------------------------------------
# Stage 2: probe


def probe(captures: Path, out: Path, device: str) -> None:
    """Remaining host reads, compile graph breaks and a whole-forward capture."""
    import torch._dynamo

    from manabot.model.inference import PolicyInference

    out.mkdir(parents=True, exist_ok=True)
    inputs = sorted(_inputs(captures), key=lambda item: len(item[1]["actions_valid"]))
    name, obs = inputs[0]
    agent = _agent(captures, "w160l8", device, bucketed=True)
    receipt: dict[str, Any] = {"input": name, "rows": len(obs["actions_valid"])}
    syncs = SyncCounter(device)
    with PolicyInference(agent) as inference:
        generator = torch.Generator(device=device).manual_seed(SEED)
        call(agent, agent.forward_distribution, inference, obs, generator)
        with syncs.window("prepare"):
            prepared = inference.prepare(obs)
        with syncs.window("forward"), torch.no_grad():
            agent.forward_distribution(prepared)
        receipt["syncs"] = syncs.summary() | {"records": syncs.records}
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        try:
            with torch.no_grad():
                explained = torch._dynamo.explain(agent.forward_distribution)(prepared)
            receipt["explain"] = {
                "graphs": explained.graph_count,
                "breaks": explained.graph_break_count,
                "ops": explained.op_count,
                "reasons": [
                    {
                        "reason": b.reason.splitlines()[0][:300],
                        "stack": [
                            f"{_short(f.filename)}:{f.lineno} {(f.line or '').strip()}"
                            for f in b.user_stack[-4:]
                        ],
                    }
                    for b in explained.break_reasons
                ],
            }
        except Exception as error:  # noqa: BLE001 - recorded evidence
            receipt["explain"] = {"error": f"{type(error).__name__}: {error}"[:2000]}
        torch._dynamo.reset()
        _write(out / "receipt.json", receipt)
        if device == "cuda":
            graph = torch.cuda.CUDAGraph()
            try:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.no_grad(), torch.cuda.stream(stream):
                    agent.forward_distribution(prepared)
                torch.cuda.current_stream().wait_stream(stream)
                with torch.no_grad(), torch.cuda.graph(graph):
                    agent.forward_distribution(prepared)
                receipt["whole_capture"] = {"status": "captured"}
            except Exception as error:  # noqa: BLE001 - recorded evidence
                frames = [
                    f
                    for f in traceback.extract_tb(error.__traceback__)
                    if "/manabot/" in f.filename
                ]
                receipt["whole_capture"] = {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}"[:600],
                    "site": [
                        f"{_short(f.filename)}:{f.lineno} {f.line}" for f in frames[-3:]
                    ],
                }
    _write(out / "receipt.json", receipt)


def _short(path: str) -> str:
    for marker in ("/manabot/", "/experiments/", "/torch/"):
        if marker in path:
            return marker.strip("/") + "/" + path.split(marker, 1)[1]
    return path


# --------------------------------------------------------------------------
# Stage 3: replays


def replay(
    captures: Path, out: Path, arm: str, size: str, device: str, repeats: int
) -> None:
    from manabot.model.inference import PolicyInference

    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    receipt: dict[str, Any] = {"arm": arm, "size": size, "host": _host()}
    inputs = _inputs(captures)
    agent = _agent(captures, size, device, bucketed=arm != "eager")
    receipt["parameters"] = sum(p.numel() for p in agent.parameters())
    receipt["weights"] = _state_digest(agent)
    forward, graphs = _forward(agent, arm, device)
    per_input: dict[str, dict[str, Any]] = {}
    outputs: dict[str, dict[str, torch.Tensor]] = {}
    try:
        with PolicyInference(agent) as inference:
            generator = torch.Generator(device=device).manual_seed(SEED)
            # The catalog is encoded once per scope, as in collection.
            inference.catalog()
            for name, obs in inputs:
                signature = _signature(inference.prepare(obs))
                _sync(device)
                tick = time.perf_counter()
                first = call(agent, forward, inference, obs, generator)
                _sync(device)
                first_s = time.perf_counter() - tick
                for _ in range(WARM_REPEATS):
                    call(agent, forward, inference, obs, generator)
                samples = []
                for _ in range(repeats):
                    _sync(device)
                    tick = time.perf_counter()
                    call(agent, forward, inference, obs, generator)
                    _sync(device)
                    samples.append(time.perf_counter() - tick)
                outputs[name] = first
                per_input[name] = {
                    "rows": len(obs["actions_valid"]),
                    "band": band(len(obs["actions_valid"])),
                    "signature": signature,
                    "first_s": first_s,
                    "samples_s": samples,
                    "median_s": float(np.median(samples)),
                }
            receipt["per_input"] = per_input
            # One profiled pass over every input for GPU busy and kernels.
            with _profiler(None, out / "unused") as prof:
                for name, obs in inputs:
                    with torch.profiler.record_function(f"mtg301/call/{name}"):
                        call(agent, forward, inference, obs, generator)
                    _sync(device)
            prof.export_chrome_trace(str(out / "trace.json"))
        receipt["peaks"] = (
            {
                "allocated": torch.cuda.max_memory_allocated(),
                "reserved": torch.cuda.max_memory_reserved(),
            }
            if device == "cuda"
            else {}
        )
        if graphs is not None:
            receipt["trunk_graphs"] = {
                "count": len(graphs.graphs),
                "capture_seconds": graphs.capture_seconds,
            }
        if arm == "compiled":
            from torch._dynamo.utils import counters

            receipt["dynamo_counters"] = {
                group: {str(k): v for k, v in values.items()}
                for group, values in counters.items()
            }
        receipt["status"] = "ok"
    except torch.cuda.OutOfMemoryError as error:
        receipt["status"] = "oom"
        receipt["error"] = str(error)[:2000]
    except Exception as error:  # noqa: BLE001 - recorded evidence
        receipt["status"] = "error"
        receipt["error"] = traceback.format_exc()[-4000:]
        print(receipt["error"], flush=True)
        del error
    finally:
        torch.save(outputs, out / "outputs.pt")
        _write(out / "receipt.json", receipt)


def compare_outputs(reference: Path, other: Path) -> dict[str, Any]:
    """Max absolute differences per input on logits (legal entries), values and
    log-probabilities (legal entries); illegal entries must agree exactly."""
    a = torch.load(reference, weights_only=False)
    b = torch.load(other, weights_only=False)
    rows: dict[str, dict[str, float | bool]] = {}
    for name in sorted(set(a) & set(b)):
        x, y = a[name], b[name]
        legal = torch.isfinite(x["logs"]) & torch.isfinite(y["logs"])
        same_support = bool(
            torch.equal(torch.isfinite(x["logs"]), torch.isfinite(y["logs"]))
        )
        rows[name] = {
            "logits": float((x["logits"] - y["logits"])[legal].abs().max()),
            "values": float((x["values"].double() - y["values"].double()).abs().max()),
            "logs": float((x["logs"] - y["logs"])[legal].abs().max()),
            "same_support": same_support,
        }
    worst = {
        key: max((float(r[key]) for r in rows.values()), default=0.0)
        for key in ("logits", "values", "logs")
    }
    return {
        "inputs": rows,
        "worst": worst,
        "within_1e-5": all(v <= 1e-5 for v in worst.values())
        and all(bool(r["same_support"]) for r in rows.values()),
    }


# --------------------------------------------------------------------------
# Stage 4: analysis


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _cudagraph_skips(log: Path) -> dict[str, int]:
    if not log.exists():
        return {}
    reasons: Counter[str] = Counter()
    for line in log.read_text(errors="replace").splitlines():
        match = re.search(r"skipping cudagraphs due to (.*)", line)
        if match:
            reasons[match.group(1)[:200]] += 1
    return dict(reasons)


def analyze(out: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    receipt_path = out / "insitu" / "receipt.json"
    insitu_calls: list[dict[str, Any]] = []
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        insitu_calls = receipt.get("calls", [])
        updates = []
        for update in receipt.get("updates", []):
            index = update["update"]
            calls = [c for c in insitu_calls if c["update"] == index]
            recording = sum(c.get("recording_s", 0.0) for c in calls)
            append = sum(c.get("append_s", 0.0) for c in calls)
            pack = sum(c.get("pack_s", 0.0) for c in calls)
            gc_recording = sum(
                v for k, v in update["gc_s"].items() if k.startswith("recording/")
            )
            bands: dict[str, dict[str, float]] = {}
            for name, *_ in BANDS:
                inside = [c for c in calls if band(c["rows"]) == name]
                bands[name] = {
                    "calls": len(inside),
                    "seconds": sum(c["seconds"] for c in inside),
                    "inference_s": sum(c.get("inference_s", 0.0) for c in inside),
                    "median_s": _median([c["seconds"] for c in inside]) or 0.0,
                }
            updates.append(
                {
                    "update": index,
                    "collection_seconds": update["collection_seconds"],
                    "rows": update["rows"],
                    "calls": update["calls"],
                    "steps": update["steps"],
                    "calls_equal_steps": update["calls"] == update["steps"],
                    "idle_stream_steps": update["idle_stream_steps"],
                    "minor_faults": update["minor_faults"],
                    "max_rss_kb": update["max_rss_kb"],
                    "phase_inference_s": sum(c.get("inference_s", 0.0) for c in calls),
                    "phase_preparation_s": sum(
                        c.get("preparation_s", 0.0) for c in calls
                    ),
                    "recording": {
                        "total_s": recording,
                        "append_s": append,
                        "pack_s": pack,
                        "rest_s": recording - append - pack,
                        "gc_s": gc_recording,
                        "append_faults": sum(c.get("append_faults", 0) for c in calls),
                        "append_bytes": sum(c.get("append_bytes", 0) for c in calls),
                        "growths": [c["grew"] for c in calls if "grew" in c],
                        "slowest_calls": sorted(
                            (
                                {
                                    k: c.get(k)
                                    for k in (
                                        "rows",
                                        "recording_s",
                                        "append_s",
                                        "pack_s",
                                        "gc_s",
                                        "append_faults",
                                        "grew",
                                    )
                                }
                                for c in calls
                            ),
                            key=lambda c: -float(c["recording_s"] or 0.0),
                        )[:10],
                    },
                    "gc_s": update["gc_s"],
                    "bands": bands,
                    "signatures": len(
                        {
                            (c["rows"], json.dumps(c.get("buckets"), sort_keys=True))
                            for c in calls
                        }
                    ),
                }
            )
        summary["insitu"] = {
            "host": receipt.get("host"),
            "status": receipt.get("status"),
            "wall_seconds": receipt.get("wall_seconds"),
            "kept": receipt.get("kept"),
            "updates": updates,
        }
    replays: dict[str, Any] = {}
    for directory in sorted((out / "replay").glob("*")):
        path = directory / "receipt.json"
        if not path.exists():
            replays[directory.name] = {"status": "missing"}
            continue
        receipt = json.loads(path.read_text())
        entry: dict[str, Any] = {
            k: receipt.get(k)
            for k in (
                "status",
                "error",
                "parameters",
                "weights",
                "peaks",
                "trunk_graphs",
                "dynamo_counters",
            )
        }
        per_input = receipt.get("per_input", {})
        entry["per_input"] = {
            name: {k: v for k, v in row.items() if k != "samples_s"}
            for name, row in per_input.items()
        }
        entry["bands"] = {
            name: {
                "median_s": _median(
                    [r["median_s"] for r in per_input.values() if r["band"] == name]
                ),
                "inputs": sum(1 for r in per_input.values() if r["band"] == name),
            }
            for name, *_ in BANDS
        }
        small = [r["median_s"] for r in per_input.values() if r["rows"] <= 8]
        middle = [r["median_s"] for r in per_input.values() if 8 < r["rows"] <= 32]
        entry["median_le8_s"] = _median(small)
        entry["median_9_32_s"] = _median(middle)
        entry["first_call_extra_s"] = sum(
            r["first_s"] - r["median_s"] for r in per_input.values()
        )
        entry["cudagraph_skips"] = _cudagraph_skips(
            out / "logs" / f"replay-{directory.name}.log"
        )
        trace = directory / "trace.json"
        if trace.exists():
            windows = trace_windows(trace, "mtg301/call/")
            entry["windows"] = [
                {
                    k: w[k]
                    for k in (
                        "window",
                        "wall_us",
                        "gpu_busy_us",
                        "kernels",
                        "launches",
                        "syncs",
                    )
                }
                for w in windows
            ]
        replays[directory.name] = entry
    summary["replays"] = replays
    comparisons: dict[str, Any] = {}
    for directory in sorted((out / "replay").glob("*")):
        arm, size = directory.name.rsplit("-", 1)
        reference = out / "replay" / f"{REFERENCE}-{size}" / "outputs.pt"
        if arm == REFERENCE or not reference.exists():
            continue
        if (directory / "outputs.pt").exists():
            comparisons[directory.name] = compare_outputs(
                reference, directory / "outputs.pt"
            )
    summary["numerics"] = comparisons
    summary["decision"] = _decision(replays, comparisons, insitu_calls)
    probe_path = out / "probe" / "receipt.json"
    if probe_path.exists():
        probe_receipt = json.loads(probe_path.read_text())
        summary["probe"] = {
            "input": probe_receipt.get("input"),
            "rows": probe_receipt.get("rows"),
            "syncs": {
                k: probe_receipt.get("syncs", {}).get(k) for k in ("per_label", "sites")
            },
            "explain": probe_receipt.get("explain"),
            "whole_capture": probe_receipt.get("whole_capture"),
        }
    _write(out / "summary.json", summary)
    return summary


def _decision(
    replays: dict[str, Any],
    comparisons: dict[str, Any],
    calls: list[dict[str, Any]],
) -> dict[str, Any]:
    """The pre-registered rule applied per candidate arm; projection over the
    in-situ (update 1) call mix by band, with first-call cost spread over 100
    updates."""
    mix = Counter(band(c["rows"]) for c in calls if c["update"] == 0)
    result: dict[str, Any] = {"call_mix": dict(mix)}

    def projected(entry: dict[str, Any]) -> float | None:
        total = 0.0
        for name, count in mix.items():
            median = entry["bands"].get(name, {}).get("median_s")
            if median is None:
                return None
            total += count * median
        return total

    for arm in ("compiled", "trunk-graph"):
        verdict: dict[str, Any] = {}
        ok = True
        for size in ("w160l8", "w384l8"):
            candidate = replays.get(f"{arm}-{size}")
            reference = replays.get(f"{REFERENCE}-{size}")
            if not candidate or not reference or candidate.get("status") != "ok":
                verdict[size] = {"status": "missing"}
                ok = False
                continue
            small = (
                reference["median_le8_s"] / candidate["median_le8_s"]
                if candidate.get("median_le8_s") and reference.get("median_le8_s")
                else None
            )
            middle = (
                candidate["median_9_32_s"] / reference["median_9_32_s"]
                if candidate.get("median_9_32_s") and reference.get("median_9_32_s")
                else None
            )
            numerics = comparisons.get(f"{arm}-{size}", {})
            row = {
                "speedup_le8": small,
                "slowdown_9_32": middle,
                "numerics_worst": numerics.get("worst"),
                "numerics_ok": bool(numerics.get("within_1e-5")),
                "rule1": small is not None and small >= 1.5,
                "rule2": middle is not None and middle <= 1.05,
                "rule3": bool(numerics.get("within_1e-5")),
            }
            if size == "w160l8":
                base = projected(reference)
                new = projected(candidate)
                if base is not None and new is not None:
                    new += candidate["first_call_extra_s"] / 100
                row["projected_update_s"] = {"reference": base, "candidate": new}
                row["rule4"] = base is not None and new is not None and new < base
            ok = ok and all(v for k, v in row.items() if k.startswith("rule"))
            verdict[size] = row
        verdict["accepted"] = ok
        result[arm] = verdict
    return result


# --------------------------------------------------------------------------
# Job orchestration


def _clean_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("ETUDE_")}
    env["PYTHONPATH"] = str(ROOT)
    return env


def _stage(args: list[str], log: Path, timeout: float) -> dict[str, Any]:
    tick = time.perf_counter()
    log.parent.mkdir(parents=True, exist_ok=True)
    env = _clean_env()
    if args[0] == "replay" and args[3] == "compiled":
        env["TORCH_LOGS"] = "perf_hints,recompiles,graph_breaks"
    with log.open("w") as handle:
        try:
            code = subprocess.run(
                [sys.executable, "-m", "experiments.collection_tail_graphs", *args],
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            ).returncode
        except subprocess.TimeoutExpired:
            code = -9
    tail = log.read_text(errors="replace").splitlines()[-40 if code else -3 :]
    print(f"== {args[0]} {log.name}: exit {code}", *tail, sep="\n", flush=True)
    return {"args": args, "returncode": code, "seconds": time.perf_counter() - tick}


def job(smoke: bool = False) -> None:
    out = Path(os.environ["ETUDE_OUTPUT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    work = Path(os.environ.get("MTG318_WORK") or tempfile.mkdtemp(prefix="mtg318-"))
    captures = work / "captures"
    device = "cpu" if smoke else "cuda"
    flags = ["--smoke"] if smoke else []
    repeats = str(SMOKE_REPEATS if smoke else TIMED_REPEATS)
    sizes = ("w160l8",) if smoke else ("w160l8", "w384l8")
    log: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        log.append(
            _stage(
                ["insitu", str(out / "insitu"), str(captures), *flags],
                out / "logs" / "insitu.log",
                5400,
            )
        )
        if not any((captures / "inference").glob("*.pt")):
            raise RuntimeError("in-situ stage kept no inputs; nothing to replay")
        log.append(
            _stage(
                ["probe", str(captures), str(out / "probe"), "--device", device],
                out / "logs" / "probe.log",
                900,
            )
        )

        def replay_stage(arm: str, size: str, name: str) -> None:
            log.append(
                _stage(
                    [
                        "replay",
                        str(captures),
                        str(out / "replay" / name),
                        arm,
                        size,
                        "--device",
                        device,
                        "--repeats",
                        repeats,
                    ],
                    out / "logs" / f"replay-{name}.log",
                    2400,
                )
            )

        # Interleaved, with a reference repeat at the end to read host drift.
        for size in sizes:
            for arm in ARMS:
                replay_stage(arm, size, f"{arm}-{size}")
        replay_stage(REFERENCE, "w160l8", "repeat-w160l8")
    finally:
        _write(
            out / "job.json", {"stages": log, "seconds": time.perf_counter() - started}
        )
        _finish(out, captures)


def _finish(out: Path, captures: Path) -> None:
    """Analyze what exists, keep the replayed inputs, compress traces, pack."""
    try:
        analyze(out)
    except Exception:  # noqa: BLE001 - partial evidence is still delivered
        (out / "analyze-error.txt").write_text(traceback.format_exc())
        print(traceback.format_exc(), flush=True)
    keep = out / "inputs"
    if (captures / "inference").exists():
        shutil.copytree(captures / "inference", keep / "inference", dirs_exist_ok=True)
    for name in ("regime.json", "initial.pt"):
        if (captures / name).exists():
            keep.mkdir(parents=True, exist_ok=True)
            shutil.copy(captures / name, keep / name)
    for trace in out.rglob("*.json"):
        if trace.name == "trace.json":
            with trace.open("rb") as src, gzip.open(f"{trace}.gz", "wb") as dst:
                shutil.copyfileobj(src, dst)
            trace.unlink()
    packed = captures.parent / "results.tar.gz"
    with tarfile.open(packed, "w:gz") as archive:
        for path in sorted(out.iterdir()):
            archive.add(path, arcname=path.name)
    shutil.move(packed, out / "results.tar.gz")


def runs(revision: str, hours: str = "1.5") -> Any:
    from etude.launch.records import Job, Variable

    from manabot.experiment.job import TOOLCHAIN_ENV, machine_setup

    return Job(
        name="mtg318-collection-tail-graphs",
        environment=(Variable(name="MTG318_REVISION", value=revision),),
        setup=machine_setup(""),
        command=(
            'mkdir -p "$ETUDE_OUTPUT_DIR"\n'
            "nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used "
            '--format=csv,noheader,nounits -l 1 > "$ETUDE_OUTPUT_DIR/gpu-samples.csv" &\n'
            "sampler=$!\n"
            "trap 'kill \"$sampler\"' EXIT\n"
            f"{TOOLCHAIN_ENV}\n"
            "uv run --locked python -m experiments.collection_tail_graphs job"
        ),
        repo=str(ROOT),
        revision=revision,
        max_hours=float(hours),
        max_usd=Decimal("2.50"),
        max_attempts=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("job")
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("insitu")
    p.add_argument("out", type=Path)
    p.add_argument("captures", type=Path)
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("probe")
    p.add_argument("captures", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("replay")
    p.add_argument("captures", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("arm", choices=ARMS)
    p.add_argument("size", choices=sorted(SIZES))
    p.add_argument("--device", default="cuda")
    p.add_argument("--repeats", type=int, default=TIMED_REPEATS)
    p = sub.add_parser("analyze")
    p.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.command == "job":
        job(args.smoke)
    elif args.command == "insitu":
        insitu(args.out, args.captures, args.smoke)
    elif args.command == "probe":
        probe(args.captures, args.out, args.device)
    elif args.command == "replay":
        replay(args.captures, args.out, args.arm, args.size, args.device, args.repeats)
    else:
        analyze(args.out)


if __name__ == "__main__":
    main()
