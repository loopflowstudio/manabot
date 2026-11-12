"""MTG-301: where an L4 learner step and a collection forward spend their time.

Exploratory. One rented L4 runs, in order:

1. `insitu`: the frozen MTG-267 w160l8 regime (`experiments.tla_frontier.frozen()`)
   for one real update on CUDA from its own initialization, four producers.
   Hooks leave the training code unchanged. They profile a window of collection
   inference calls and learner minibatches with `torch.profiler` (CPU + CUDA),
   count synchronizing CUDA calls in separate windows with
   `torch.cuda.set_sync_debug_mode("warn")` and the Python stack, and keep a spread
   of collection observations and learner minibatches. The update then unwinds
   before any export.
2. `replay`: those inputs through the inference and learner-step code in one
   subprocess per arm (`baseline` at the job revision, `option3` with
   `l4_step_profile_option3.patch` applied to a copy of `manabot/`) and size
   (w160l8 with the in-situ initial weights; a larger random initialization). The
   learner step reproduces `update_damped_iteration`'s minibatch body, including
   its index round trip, `OptimizerHealth` check/step and diagnostic reads.
   Labelled top-level modules and `SemanticCardEncoder.definitions` attribute
   kernels. Each arm also writes fixed-weight numerics for cross-arm comparison.
3. `analyze`: reads every trace for wall time, GPU busy (union of kernel/memcpy/memset
   intervals), host time blocked in CUDA waits, gaps, launches and per-label
   kernel time. Results go to `summary.json`.

`job` runs all three under Etude and leaves `results.tar.gz` in ETUDE_OUTPUT_DIR.
`numerics` and `compare` also run locally on retained CPU inputs: the option-3
arm must agree with baseline within float tolerance before it is timed.

    etude launch experiments/l4_step_profile.py --rent l4-etude --max-usd 3 \\
        --revision COMMIT
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
import gzip
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from typing import Any, TypedDict
import warnings

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PATCH = Path(__file__).with_name("l4_step_profile_option3.patch")
SEED = 30101
# Larger sizes tried in order until one fits the L4's memory.
SIZES: dict[str, dict[str, int]] = {
    "w160l8": {},
    "w384l8": {"hidden_dim": 384, "attention_layers": 8, "num_attention_heads": 8},
    "w256l8": {"hidden_dim": 256, "attention_layers": 8, "num_attention_heads": 8},
}


@dataclass(frozen=True)
class Windows:
    """In-situ windows over collection inference calls and learner optimizer steps.
    Profiles are (wait, warmup, active) schedules; captures are retained inputs."""

    collect_profile: tuple[int, int, int] = (120, 2, 16)
    collect_sync: range = range(200, 216)
    collect_capture_every: int = 12
    collect_captures: int = 48
    learn_profile: tuple[int, int, int] = (30, 2, 8)
    learn_sync: range = range(50, 60)
    learn_capture: range = range(70, 90)


# `--smoke`: a tiny CPU update that only exercises the hooks and analysis.
SMOKE = Windows((2, 1, 3), range(6, 8), 3, 4, (1, 1, 2), range(4, 5), range(5, 7))
# Replay repetitions.
TIMED_PASSES = 2
NUMERIC_LEARNER = 4
NUMERIC_INFERENCE = 8
SYNC_MESSAGE = "synchronizing CUDA operation"
WAITS = {
    "cudaStreamSynchronize",
    "cudaDeviceSynchronize",
    "cudaEventSynchronize",
    "cudaMemcpy",
    "cudaMemcpyAsync",
}
SYNCS = {"cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize"}
GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


class Captured(BaseException):
    """Unwind the executor after the update's learning phase."""


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True, default=str))


def _host() -> dict[str, object]:
    facts: dict[str, object] = {"torch": torch.__version__, "cpus": os.cpu_count()}
    if torch.cuda.is_available():
        facts["gpu"] = torch.cuda.get_device_name()
        facts["cuda"] = torch.version.cuda
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                facts["cpu_model"] = line.split(":", 1)[1].strip()
                break
        facts["affinity"] = len(os.sched_getaffinity(0))
    except (OSError, AttributeError):
        pass
    return facts


# --------------------------------------------------------------------------
# Synchronization attribution


class SyncCounter:
    """Record each sync-debug warning with the innermost repository frame."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.label: str | None = None
        self.records: list[dict[str, object]] = []
        self._previous: Callable[..., None] | None = None

    def _show(
        self,
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: object = None,
        line: str | None = None,
    ) -> None:
        if SYNC_MESSAGE not in str(message) or self.label is None:
            if self._previous is not None:
                self._previous(message, category, filename, lineno, file, line)
            return
        frames = [
            f
            for f in traceback.extract_stack()[:-1]
            if ("/manabot/" in f.filename or "/experiments/" in f.filename)
            and "l4_step_profile" not in f.filename
        ]
        inner = frames[-1] if frames else None
        self.records.append(
            {
                "label": self.label,
                "site": f"{_short(inner.filename)}:{inner.lineno}" if inner else "?",
                "code": (inner.line or "").strip() if inner else "",
                "caller": [f"{_short(f.filename)}:{f.lineno}" for f in frames[-4:-1]],
            }
        )

    @contextmanager
    def window(self, label: str) -> Iterator[None]:
        if self.device != "cuda":
            yield
            return
        self.label = label
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            self._previous = warnings.showwarning
            warnings.showwarning = self._show
            torch.cuda.set_sync_debug_mode("warn")
            try:
                yield
            finally:
                torch.cuda.set_sync_debug_mode("default")
                warnings.showwarning = self._previous
                self.label = None

    def summary(self) -> dict[str, object]:
        labels = Counter(str(r["label"]) for r in self.records)
        sites = Counter((str(r["site"]), str(r["code"])) for r in self.records)
        return {
            "per_label": dict(labels),
            "sites": [
                {"site": site, "code": code, "count": count}
                for (site, code), count in sites.most_common()
            ],
        }


def _short(path: str) -> str:
    for marker in ("/manabot/", "/experiments/"):
        if marker in path:
            return marker.strip("/") + "/" + path.split(marker, 1)[1]
    return path


# --------------------------------------------------------------------------
# Kernel attribution labels


def label_modules(agent: torch.nn.Module) -> None:
    """Wrap each top-level child (and ModuleList elements) in a profiler label,
    and SemanticCardEncoder.definitions in its own label."""
    from manabot.model.semantic_cards import SemanticCardEncoder

    def attach(name: str, module: torch.nn.Module) -> None:
        stack: list[Any] = []

        def pre(_m: torch.nn.Module, _a: object) -> None:
            ctx = torch.profiler.record_function(f"mtg301/module/{name}")
            ctx.__enter__()
            stack.append(ctx)

        def post(_m: torch.nn.Module, _a: object, _o: object) -> None:
            stack.pop().__exit__(None, None, None)

        module.register_forward_pre_hook(pre)
        module.register_forward_hook(post)

    for name, child in agent.named_children():
        if isinstance(child, torch.nn.ModuleList):
            for index, element in enumerate(child):
                attach(f"{name}.{index}", element)
        else:
            attach(name, child)
    if not getattr(SemanticCardEncoder.definitions, "_mtg301", False):
        original = SemanticCardEncoder.definitions

        def definitions(
            self: SemanticCardEncoder, rows: torch.Tensor | None = None
        ) -> torch.Tensor:
            with torch.profiler.record_function("mtg301/catalog_definitions"):
                return original(self, rows)

        definitions._mtg301 = True  # type: ignore[attr-defined]
        SemanticCardEncoder.definitions = definitions  # type: ignore[method-assign]


def _profiler(
    schedule: tuple[int, int, int] | None, trace: Path
) -> torch.profiler.profile:
    def ready(prof: torch.profiler.profile) -> None:
        prof.export_chrome_trace(str(trace))

    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=schedule[0], warmup=schedule[1], active=schedule[2], repeat=1
        )
        if schedule
        else None,
        on_trace_ready=ready if schedule else None,
    )


# --------------------------------------------------------------------------
# Stage 1: in-situ update


def insitu(out: Path, captures: Path, smoke: bool = False) -> None:
    from experiments.tla_frontier import frozen
    import manabot.model.inference as inference_module
    import manabot.training.execution as execution_module
    from manabot.training.regime import TrainingRegime
    from manabot.training.rollout.collection import SeatRoutedCollector
    from manabot.training.updates.health import OptimizerHealth
    from manabot.verify.store import VerifyStore

    out.mkdir(parents=True, exist_ok=True)
    (captures / "inference").mkdir(parents=True, exist_ok=True)
    (captures / "learner").mkdir(parents=True, exist_ok=True)
    regimes, protocol = frozen()
    data = regimes["w160l8"].model_dump()
    stage = data["stages"][0]
    stage["updates"] = 1
    stage["execution"].update(wall_seconds=3600.0)
    windows = SMOKE if smoke else Windows()
    if smoke:
        stage.update(streams=4, transitions=2)
        stage["execution"].update(device="cpu", workers=1)
    data["wall_seconds"] = 3600.0
    regime = TrainingRegime.model_validate(data)
    _write(captures / "regime.json", regime.model_dump(mode="json"))
    receipt: dict[str, Any] = {
        "host": _host(),
        "seed": protocol.training_seed,
        "regime": regime.id,
        "execution": stage["execution"],
    }
    device = stage["execution"]["device"]
    syncs = SyncCounter(device)
    calls: list[dict[str, float | int]] = []
    steps: list[dict[str, float | int]] = []
    state: dict[str, Any] = {
        "call": 0,
        "step": 0,
        "capture_obs": False,
        "last_obs": None,
        "collect_prof": None,
        "learn_prof": None,
        "last_step": None,
    }
    original_actor = execution_module.ActorCritic
    original_prepare = inference_module.PolicyInference.prepare
    original_sample = SeatRoutedCollector._sample_learning_rows
    original_check = OptimizerHealth.check
    original_step = OptimizerHealth.step
    original_update = execution_module.update_iteration

    def make_actor(space: Any, config: Any) -> Any:
        agent = original_actor(space, config)
        if (captures / "initial.pt").exists():
            raise RuntimeError("the update constructed a second policy")
        torch.save(
            {k: v.detach().cpu().clone() for k, v in agent.state_dict().items()},
            captures / "initial.pt",
        )
        label_modules(agent)
        return agent

    def prepare(self: Any, obs: dict[str, torch.Tensor]) -> Any:
        if state["capture_obs"]:
            state["last_obs"] = {k: v.clone() for k, v in obs.items()}
        return original_prepare(self, obs)

    def sample(self: Any, agent: Any, rows: np.ndarray, *args: Any) -> Any:
        index = state["call"]
        if index == 0:
            prof = _profiler(windows.collect_profile, out / "collection-trace.json")
            prof.__enter__()
            state["collect_prof"] = prof
        state["capture_obs"] = (
            index % windows.collect_capture_every == 0
            and index // windows.collect_capture_every < windows.collect_captures
        )
        tick = time.perf_counter()
        with (
            syncs.window(f"collect/{index}")
            if index in windows.collect_sync
            else _nothing(),
            torch.profiler.record_function("mtg301/sample_learning_rows"),
        ):
            result = original_sample(self, agent, rows, *args)
        calls.append(
            {"call": index, "rows": len(rows), "seconds": time.perf_counter() - tick}
        )
        if state["capture_obs"] and state["last_obs"] is not None:
            torch.save(state["last_obs"], captures / "inference" / f"{index:05d}.pt")
            state["last_obs"] = None
        state["capture_obs"] = False
        prof = state["collect_prof"]
        if prof is not None:
            prof.step()
            if index + 1 >= sum(windows.collect_profile):
                prof.__exit__(None, None, None)
                state["collect_prof"] = None
        state["call"] = index + 1
        return result

    def check(self: OptimizerHealth, loss: torch.Tensor) -> None:
        original_check(self, loss)
        if state["step"] not in windows.learn_capture:
            return
        frame = inspect.currentframe()
        assert frame is not None and frame.f_back is not None
        loc = frame.f_back.f_locals
        assert frame.f_back.f_code.co_name == "update_damped_iteration"
        indexes, targets = loc["indexes"], loc["targets"]
        payload = {
            "obs": {k: v.detach().cpu().clone() for k, v in loc["obs"].items()},
            "selected": loc["selected"].detach().cpu().clone(),
            "actions": loc["actions"][indexes].detach().cpu().clone(),
            "advantages": targets.advantages[indexes].detach().cpu().clone(),
            "value_targets": targets.values[indexes].detach().cpu().clone(),
            "actor_mask": loc["actor_mask"].detach().cpu().clone(),
            "behavior_logs": loc["behavior_logs"].detach().cpu().clone(),
            "learning": loc["learning"].model_dump(mode="json"),
            "tau": loc["tau"],
            "loss": float(loss.detach()),
        }
        del loc, frame
        torch.save(payload, captures / "learner" / f"{state['step']:03d}.pt")

    def step(self: OptimizerHealth, *args: Any, **kwargs: Any) -> torch.Tensor:
        index = state["step"]
        label = f"learn/{index}"
        with syncs.window(label) if index in windows.learn_sync else _nothing():
            norm = original_step(self, *args, **kwargs)
        now = time.perf_counter()
        if state["last_step"] is not None:
            steps.append({"step": index, "seconds": now - state["last_step"]})
        state["last_step"] = now
        prof = state["learn_prof"]
        if prof is not None:
            prof.step()
            if index + 1 >= sum(windows.learn_profile):
                prof.__exit__(None, None, None)
                state["learn_prof"] = None
        state["step"] = index + 1
        return norm

    def update(learner: Any, *args: Any, **kwargs: Any) -> Any:
        prof = _profiler(windows.learn_profile, out / "learner-trace.json")
        prof.__enter__()
        state["learn_prof"] = prof
        state["last_step"] = time.perf_counter()
        tick = time.perf_counter()
        _reset_memory(device)
        diagnostics = original_update(learner, *args, **kwargs)
        _sync(device)
        receipt.update(
            learning_seconds=time.perf_counter() - tick,
            learner_phases=learner.profiler.get_stats(),
            optimizer_steps=state["step"],
            rows=diagnostics.get("rows"),
            **_peaks(device),
        )
        if state["learn_prof"] is not None:
            state["learn_prof"].__exit__(None, None, None)
        raise Captured()

    def collect(self: SeatRoutedCollector, *args: Any, **kwargs: Any) -> Any:
        tick = time.perf_counter()
        batch = original_collect(self, *args, **kwargs)
        receipt["collection_seconds"] = time.perf_counter() - tick
        receipt["collection_rows"] = int(batch.actions.size)
        receipt["collection_phases"] = self.profiler.get_stats()
        receipt["idle_stream_steps"] = int(self.stats.idle_stream_steps)
        return batch

    original_collect = SeatRoutedCollector.collect
    # Etude's unpacked bundle has no .git; the launch names the frozen commit.
    if revision := os.environ.get("MTG301_REVISION"):
        execution_module.source_commit = lambda: revision
        receipt["source_commit"] = revision
    execution_module.ActorCritic = make_actor
    inference_module.PolicyInference.prepare = prepare  # type: ignore[method-assign]
    SeatRoutedCollector._sample_learning_rows = sample  # type: ignore[method-assign]
    SeatRoutedCollector.collect = collect  # type: ignore[method-assign]
    OptimizerHealth.check = check  # type: ignore[method-assign]
    OptimizerHealth.step = step  # type: ignore[method-assign]
    execution_module.update_iteration = update
    started = time.perf_counter()
    try:
        with VerifyStore(captures / "store.sqlite") as store:
            result = execution_module.execute_regime(
                regime,
                protocol.training_seed,
                captures / "execution",
                store,
                experiment="l4_step_profile",
                training_exposure=False,
            )
        raise RuntimeError(f"update ended unexpectedly: {result.status}")
    except Captured:
        receipt["status"] = "captured"
    finally:
        receipt["wall_seconds"] = time.perf_counter() - started
        receipt["inference_calls"] = calls
        receipt["learner_steps"] = steps
        _write(out / "receipt.json", receipt)
        _write(out / "syncs.json", syncs.summary() | {"records": syncs.records})


@contextmanager
def _nothing() -> Iterator[None]:
    yield


# --------------------------------------------------------------------------
# Stage 2: replays


class Inputs(TypedDict):
    inference: list[dict[str, torch.Tensor]]
    learner: list[dict[str, Any]]


def _inputs(captures: Path) -> Inputs:
    def load(path: Path) -> Any:
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as arrays:
                return {k: torch.from_numpy(arrays[k]) for k in arrays.files}
        return torch.load(path, weights_only=False)

    inference = sorted(
        p for p in (captures / "inference").glob("*") if p.suffix in {".pt", ".npz"}
    )
    return {
        "inference": [load(p) for p in inference],
        "learner": [load(p) for p in sorted((captures / "learner").glob("*.pt"))],
    }


def _agent(captures: Path, size: str, device: str) -> Any:
    from manabot.env.observation import ObservationSpace
    from manabot.model.actor_critic import ActorCritic
    from manabot.training.regime import TrainingRegime

    regime = TrainingRegime.model_validate_json((captures / "regime.json").read_text())
    config = regime.agent.model_copy(update=SIZES[size])
    torch.manual_seed(SEED)
    agent = ActorCritic(ObservationSpace(regime.observation), config)
    if not SIZES[size]:
        agent.load_state_dict(torch.load(captures / "initial.pt"))
    return agent.to(device)


def _state_digest(agent: torch.nn.Module) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name, value in sorted(agent.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


def infer_once(
    agent: Any, inference: Any, obs: dict[str, torch.Tensor], generator: torch.Generator
) -> dict[str, Any]:
    """Collection's preparation and `Policy inference and action selection` block."""
    from manabot.model.policy_distribution import legal_mask, policy_logs
    from manabot.rl.numerics import require

    with torch.profiler.record_function("mtg301/prepare"):
        obs_t = inference.prepare(obs)
    with torch.profiler.record_function("mtg301/inference"), torch.no_grad():
        if agent.config.value_kind == "categorical_wdl":
            logits, value_t = agent.forward_distribution(obs_t)
        else:
            logits, value_t = agent(obs_t)
        logits, value_t = logits.float(), value_t.float()
        if agent.config.value_kind == "categorical_wdl":
            outcome = value_t.softmax(dim=-1)
            value_t = outcome[:, 2] - outcome[:, 0]
            outcome.cpu().numpy()
        logs = policy_logs(logits, legal_mask(obs_t["actions_valid"]))
        require(torch.isfinite(value_t), "nonfinite_collection_values", values=value_t)
        weights = logs.exp()
        action_t = torch.multinomial(weights, 1, generator=generator).squeeze(-1)
        logprob_t = logs.gather(-1, action_t[:, None]).squeeze(-1)
        weights.cpu().numpy()
        logs.cpu().numpy()
        action_t.cpu().numpy()
        logprob_t.cpu().numpy()
        values = value_t.view(-1).cpu().numpy()
    return {"logits": logits, "values": values}


def learn_once(
    learner: Any, observer: Any, payload: dict[str, Any], *, step: bool = True
) -> dict[str, Any]:
    """`update_damped_iteration`'s minibatch body on one retained minibatch."""
    from manabot.model.policy_distribution import legal_mask, policy_logs
    from manabot.rl.losses import reverse_kl
    from manabot.rl.numerics import float64_operand
    from manabot.rl.selection import selected_mean
    from manabot.training.regime import DampedPolicyLearning
    from manabot.training.updates.damped_policy import damped_policy_loss
    from manabot.training.updates.learner import observation_tensors
    from manabot.training.updates.references import reference_distribution

    device = learner.device
    learning = DampedPolicyLearning.model_validate(payload["learning"])
    tau = payload["tau"]
    with torch.profiler.record_function("mtg301/forward_and_loss"):
        rows = len(payload["actions"])
        selected_host = payload.get("selected")
        if selected_host is None:
            selected_host = torch.ones(rows, dtype=torch.bool)
        selected = selected_host.to(device)
        if not selected.any():
            raise ValueError("retained minibatch has no selected row")
        indexes = torch.arange(len(selected), device=device)[selected]
        indexes.cpu().numpy()
        local = torch.arange(rows, device=device)
        obs = observation_tensors(
            {k: v.numpy() for k, v in payload["obs"].items()}, device
        )
        valid = legal_mask(obs["actions_valid"])
        behavior_logs = payload["behavior_logs"].to(device)
        actions = payload["actions"].to(device)[local]
        advantages = payload["advantages"].to(device)[local]
        value_targets = payload["value_targets"].to(device)[local]
        with learner.forward_precision():
            logits, value_logits = learner.agent.forward_distribution(obs)
        logits, value_logits = logits.float(), value_logits.float()
        reference = reference_distribution(obs, learning.reference)
        policy_loss = damped_policy_loss(
            logits,
            actions,
            advantages,
            behavior_logs,
            reference,
            valid,
            clip=learning.clip,
            collection_kl=learning.collection_kl,
            tau=tau,
        )
        actor_mask = payload["actor_mask"].to(device)
        policy_loss = selected_mean(policy_loss, actor_mask)
        if learner.agent.config.value_kind == "categorical_wdl":
            value_loss = -(value_targets * value_logits.log_softmax(-1)).sum(-1).mean()
        else:
            value_loss = (value_logits.squeeze(-1) - value_targets).square().mean()
        loss = policy_loss + float64_operand(value_loss)
        observer.check(loss)
    if not step:
        learner.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        return {"loss": loss.detach(), "logits": logits.detach()}
    with torch.profiler.record_function("mtg301/backward_and_step"):
        norm = observer.step(loss, learning.max_grad_norm, checked=True)
    with torch.profiler.record_function("mtg301/diagnostics"):
        out: dict[str, Any] = {
            "exposures": int(selected.sum()),
            "actor": int(actor_mask.sum()),
            "critic": int(selected.sum()),
        }
        with torch.no_grad():
            logs = policy_logs(logits, valid)
            probabilities = logs.exp()
            out["entropy"] = float(
                -(probabilities * logs.masked_fill(~valid, 0)).sum(-1).mean()
            )
            out["collection_kl"] = float(reverse_kl(logs, behavior_logs, valid).mean())
            out["reference_kl"] = float(
                reverse_kl(logs, reference.masked_fill(~valid, 1).log(), valid).mean()
            )
            ratio_logs = (logs - behavior_logs).gather(-1, actions[:, None]).squeeze(-1)
            out["clip_fraction"] = float(
                ((float64_operand(ratio_logs).double().exp() - 1).abs() > learning.clip)
                .float()
                .mean()
            )
        out.update(
            loss=float(loss.detach()),
            value_loss=float(value_loss.detach()),
            policy_loss=float(policy_loss.detach()),
            gradient_norm=float(norm),
        )
    return out


def _learner(agent: Any, device: str) -> tuple[Any, Any]:
    from manabot.training.updates.health import NumericalHealth, OptimizerHealth
    from manabot.training.updates.learner import Learner

    learner = Learner(agent, device, minibatch_rows=256, precision="float32")
    health = NumericalHealth(
        optimizer_steps=0, skipped_steps=0, rejected_steps=0, sampled_steps=1
    )
    return learner, OptimizerHealth(learner.agent, learner.optimizer, health, 2)


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _reset_memory(device: str) -> None:
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _peaks(device: str) -> dict[str, int]:
    if device != "cuda":
        return {}
    return {
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
    }


def numerics(captures: Path, out: Path, size: str, device: str) -> None:
    """Fixed-weight inference outputs, losses and gradients for cross-arm checks."""
    import manabot

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    inputs = _inputs(captures)
    agent = _agent(captures, size, device).eval()
    from manabot.model.inference import PolicyInference

    result: dict[str, Any] = {
        "manabot": manabot.__file__,
        "weights": _state_digest(agent),
        "inference": [],
        "learner": [],
    }
    with PolicyInference(agent) as inference:
        for obs in inputs["inference"][:NUMERIC_INFERENCE]:
            generator = torch.Generator(device=device).manual_seed(SEED)
            got = infer_once(agent, inference, obs, generator)
            result["inference"].append(
                {
                    "logits": got["logits"].cpu(),
                    "values": torch.from_numpy(got["values"]),
                }
            )
    learner, observer = _learner(agent, device)
    for payload in inputs["learner"][:NUMERIC_LEARNER]:
        got = learn_once(learner, observer, payload, step=False)
        grads = torch.cat(
            [
                (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                for p in learner.agent.parameters()
            ]
        )
        result["learner"].append(
            {
                "loss": got["loss"].cpu(),
                "logits": got["logits"].cpu(),
                "grad": grads.cpu(),
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out)


def compare(first: Path, second: Path) -> dict[str, Any]:
    """Max differences; float tolerance covers summation-order changes."""
    a, b = torch.load(first, weights_only=False), torch.load(second, weights_only=False)

    def diff(
        x: torch.Tensor, y: torch.Tensor, rtol: float, atol: float
    ) -> dict[str, Any]:
        x, y = x.double(), y.double()
        finite = torch.isfinite(x) & torch.isfinite(y)
        same_finite = bool(torch.equal(torch.isfinite(x), torch.isfinite(y)))
        error = (x[finite] - y[finite]).abs()
        return {
            "max_abs": float(error.max()) if error.numel() else 0.0,
            "max_rel": float((error / y[finite].abs().clamp_min(1e-12)).max())
            if error.numel()
            else 0.0,
            "ok": same_finite
            and bool(torch.allclose(x[finite], y[finite], rtol=rtol, atol=atol)),
        }

    report: dict[str, Any] = {
        "weights_equal": a["weights"] == b["weights"],
        "inference": [
            {
                "logits": diff(x["logits"], y["logits"], 1e-4, 1e-4),
                "values": diff(x["values"], y["values"], 1e-4, 1e-4),
            }
            for x, y in zip(a["inference"], b["inference"], strict=True)
        ],
        "learner": [],
    }
    for x, y in zip(a["learner"], b["learner"], strict=True):
        norm_x, norm_y = float(x["grad"].norm()), float(y["grad"].norm())
        report["learner"].append(
            {
                "loss": diff(x["loss"].reshape(1), y["loss"].reshape(1), 1e-4, 1e-5),
                "logits": diff(x["logits"], y["logits"], 1e-4, 1e-4),
                "grad": diff(x["grad"], y["grad"], 1e-3, 1e-6),
                "grad_norm": [norm_x, norm_y],
                "grad_cosine": float(
                    torch.nn.functional.cosine_similarity(
                        x["grad"].double(), y["grad"].double(), dim=0
                    )
                ),
            }
        )
    report["ok"] = (
        report["weights_equal"]
        and all(all(part["ok"] for part in row.values()) for row in report["inference"])
        and all(
            row["loss"]["ok"] and row["logits"]["ok"] and row["grad"]["ok"]
            for row in report["learner"]
        )
    )
    return report


def replay(
    captures: Path, out: Path, arm: str, size: str, device: str = "cuda"
) -> None:
    import manabot
    from manabot.model.inference import PolicyInference

    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    receipt: dict[str, Any] = {
        "arm": arm,
        "size": size,
        "manabot": manabot.__file__,
        "host": _host(),
    }
    inputs = _inputs(captures)
    agent = _agent(captures, size, device)
    receipt["bucketed_shapes"] = getattr(agent, "bucketed_shapes", None)
    receipt["parameters"] = sum(p.numel() for p in agent.parameters())
    receipt["weights"] = _state_digest(agent)
    receipt["inference_rows"] = [len(o["actions_valid"]) for o in inputs["inference"]]
    receipt["learner_rows"] = [len(p["actions"]) for p in inputs["learner"]]
    label_modules(agent)
    syncs = SyncCounter(device)
    try:
        # Inference: eval mode, one behavior scope, as collection holds it.
        agent.eval()
        _reset_memory(device)
        with PolicyInference(agent) as inference:
            generator = torch.Generator(device=device).manual_seed(SEED)
            for obs in inputs["inference"]:
                infer_once(agent, inference, obs, generator)
            timed: list[list[float]] = []
            for _ in range(TIMED_PASSES):
                row: list[float] = []
                for obs in inputs["inference"]:
                    tick = time.perf_counter()
                    infer_once(agent, inference, obs, generator)
                    _sync(device)
                    row.append(time.perf_counter() - tick)
                timed.append(row)
            receipt["inference_seconds"] = timed
            with _profiler(None, out / "unused") as prof:
                for obs in inputs["inference"]:
                    with torch.profiler.record_function("mtg301/inference_call"):
                        infer_once(agent, inference, obs, generator)
                    _sync(device)
            prof.export_chrome_trace(str(out / "inference-trace.json"))
            for index, obs in enumerate(inputs["inference"]):
                with syncs.window(f"inference/{index}"):
                    infer_once(agent, inference, obs, generator)
        receipt["inference_peaks"] = _peaks(device)
        # Learner: train mode, Adam steps continue across passes.
        _reset_memory(device)
        learner, observer = _learner(agent, device)
        for payload in inputs["learner"]:
            learn_once(learner, observer, payload)
        timed = []
        for _ in range(TIMED_PASSES):
            row = []
            for payload in inputs["learner"]:
                tick = time.perf_counter()
                learn_once(learner, observer, payload)
                _sync(device)
                row.append(time.perf_counter() - tick)
            timed.append(row)
        receipt["learner_seconds"] = timed
        with _profiler(None, out / "unused") as prof:
            for payload in inputs["learner"]:
                with torch.profiler.record_function("mtg301/learner_step"):
                    learn_once(learner, observer, payload)
                _sync(device)
        prof.export_chrome_trace(str(out / "learner-trace.json"))
        for index, payload in enumerate(inputs["learner"]):
            with syncs.window(f"learner/{index}"):
                learn_once(learner, observer, payload)
        receipt["learner_peaks"] = _peaks(device)
        receipt["catalog"] = _catalog_bench(learner.agent, device)
        receipt["status"] = "ok"
    except torch.cuda.OutOfMemoryError as error:
        receipt["status"] = "oom"
        receipt["error"] = str(error)[:2000]
    finally:
        _write(out / "receipt.json", receipt)
        _write(out / "syncs.json", syncs.summary() | {"records": syncs.records})


def _catalog_bench(agent: Any, device: str, repeats: int = 10) -> dict[str, Any]:
    """Full-catalog program encoding alone: forward, and forward plus backward."""
    encoder = agent.semantic_cards
    if encoder is None:
        return {}
    result: dict[str, Any] = {"definitions": int(len(encoder.definition_rows))}
    for mode in ("forward", "forward_backward"):
        samples = []
        for _ in range(repeats + 2):
            agent.zero_grad(set_to_none=True)
            _sync(device)
            tick = time.perf_counter()
            if mode == "forward":
                with torch.no_grad():
                    encoder.definitions()
            else:
                encoder.definitions().square().sum().backward()
            _sync(device)
            samples.append(time.perf_counter() - tick)
        result[mode] = sorted(samples[2:])
    agent.zero_grad(set_to_none=True)
    return result


# --------------------------------------------------------------------------
# Stage 3: trace analysis


def _union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _read_trace(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        data = json.load(handle)
    events = data["traceEvents"] if isinstance(data, dict) else data
    return [e for e in events if e.get("ph") == "X" and "dur" in e]


def trace_windows(path: Path, span: str) -> list[dict[str, Any]]:
    """Per window named `span`: wall, GPU busy, host waits, gaps and labels (µs)."""
    events = _read_trace(path)
    gpu = sorted(
        (float(e["ts"]), float(e["ts"]) + float(e["dur"]), e)
        for e in events
        if e.get("cat") in GPU_CATS
    )
    runtime = [e for e in events if e.get("cat") == "cuda_runtime"]
    annotations = [
        e
        for e in events
        if e.get("cat") == "user_annotation"
        and str(e.get("name", "")).startswith("mtg301/")
    ]
    windows = [e for e in events if str(e.get("name", "")).startswith(span)]
    # Kernel → launching runtime event → innermost enclosing label on its thread.
    by_correlation = {
        e.get("args", {}).get("correlation"): e for e in runtime if "args" in e
    }
    by_thread: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for a in annotations:
        by_thread[(a.get("pid"), a.get("tid"))].append(a)

    def label(kernel: dict[str, Any]) -> str:
        launch = by_correlation.get(kernel.get("args", {}).get("correlation"))
        if launch is None:
            return "unlaunched"
        ts = float(launch["ts"])
        enclosing = [
            a
            for a in by_thread.get((launch.get("pid"), launch.get("tid")), [])
            if float(a["ts"]) <= ts <= float(a["ts"]) + float(a["dur"])
        ]
        if not enclosing:
            return "other-thread-or-unlabelled"
        inner = min(enclosing, key=lambda a: float(a["dur"]))
        return str(inner["name"]).removeprefix("mtg301/")

    found = []
    for window in windows:
        start = float(window["ts"])
        end = start + float(window["dur"])
        inside = [
            (max(s, start), min(e, end), k) for s, e, k in gpu if e > start and s < end
        ]
        merged = _union([(s, e) for s, e, _ in inside])
        busy = sum(e - s for s, e in merged)
        gaps = [b[0] - a[1] for a, b in zip(merged, merged[1:])]
        kernels = [k for _, _, k in inside if k.get("cat") == "kernel"]
        waits = [
            r
            for r in runtime
            if start <= float(r["ts"]) < end and r.get("name") in WAITS
        ]
        labels: dict[str, float] = defaultdict(float)
        label_counts: Counter[str] = Counter()
        for s, e, k in inside:
            name = label(k)
            labels[name] += e - s
            label_counts[name] += 1
        host_labels: dict[str, float] = defaultdict(float)
        for a in annotations:
            ts = float(a["ts"])
            if start <= ts < end:
                host_labels[str(a["name"]).removeprefix("mtg301/")] += float(a["dur"])
        durations = sorted(float(k["dur"]) for k in kernels)
        found.append(
            {
                "window": window["name"],
                "wall_us": end - start,
                "gpu_busy_us": busy,
                "kernel_us": sum(durations),
                "kernels": len(durations),
                "kernel_median_us": durations[len(durations) // 2]
                if durations
                else 0.0,
                "kernels_under_10us": sum(d < 10 for d in durations),
                "memcpy_dtoh": sum(
                    1 for _, _, k in inside if "DtoH" in str(k.get("name"))
                ),
                "memcpy_htod": sum(
                    1 for _, _, k in inside if "HtoD" in str(k.get("name"))
                ),
                "launches": sum(
                    1
                    for r in runtime
                    if start <= float(r["ts"]) < end and "Launch" in str(r.get("name"))
                ),
                "syncs": sum(1 for w in waits if w.get("name") in SYNCS),
                "wait_us": sum(float(w["dur"]) for w in waits),
                "waits": len(waits),
                "gaps": len(gaps),
                "gap_us": sum(gaps),
                "gaps_over_50us_us": sum(g for g in gaps if g > 50),
                "gpu_us_by_label": dict(labels),
                "gpu_events_by_label": dict(label_counts),
                "host_us_by_label": dict(host_labels),
            }
        )
    return found


def _bounds(window: dict[str, Any]) -> dict[str, float]:
    """Decompose one window. Without host waits the host and GPU overlap fully, so
    the wall cannot fall below max(GPU busy, host work excluding waits)."""
    wall, busy, wait = window["wall_us"], window["gpu_busy_us"], window["wait_us"]
    floor = max(busy, wall - wait)
    return {
        "wall_us": wall,
        "gpu_busy_us": busy,
        "gpu_idle_us": wall - busy,
        "host_wait_us": wait,
        "host_work_us": wall - wait,
        "sync_removal_upper_bound_us": wall - floor,
    }


def analyze(out: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    insitu_dir = out / "insitu"
    if (insitu_dir / "receipt.json").exists():
        receipt = json.loads((insitu_dir / "receipt.json").read_text())
        summary["insitu"] = {
            k: receipt.get(k)
            for k in (
                "host",
                "collection_seconds",
                "learning_seconds",
                "collection_rows",
                "optimizer_steps",
                "peak_allocated",
                "peak_reserved",
                "idle_stream_steps",
                "rows",
                "collection_phases",
                "learner_phases",
                "wall_seconds",
                "status",
            )
        }
        calls = receipt.get("inference_calls", [])
        steps = receipt.get("learner_steps", [])
        summary["insitu"]["inference_calls"] = len(calls)
        summary["insitu"]["inference_rows_mean"] = (
            float(np.mean([c["rows"] for c in calls])) if calls else None
        )
        summary["insitu"]["inference_call_median_s"] = (
            float(np.median([c["seconds"] for c in calls])) if calls else None
        )
        summary["insitu"]["learner_step_median_s"] = (
            float(np.median([s["seconds"] for s in steps])) if steps else None
        )
        for name, span in (
            ("collection", "mtg301/sample_learning_rows"),
            ("collection_loop", "ProfilerStep#"),
            ("learner", "ProfilerStep#"),
        ):
            trace = insitu_dir / f"{name.split('_')[0]}-trace.json"
            if trace.exists():
                rows = trace_windows(trace, span)
                summary["insitu"][f"{name}_windows"] = rows
                summary["insitu"][f"{name}_bounds"] = [_bounds(r) for r in rows]
        syncs = json.loads((insitu_dir / "syncs.json").read_text())
        summary["insitu"]["syncs"] = {k: syncs[k] for k in ("per_label", "sites")}
    summary["replays"] = {}
    for replay_dir in sorted((out / "replay").glob("*")):
        if not (replay_dir / "receipt.json").exists():
            summary["replays"][replay_dir.name] = {"status": "missing"}
            continue
        receipt = json.loads((replay_dir / "receipt.json").read_text())
        entry: dict[str, Any] = {
            k: receipt.get(k)
            for k in (
                "status",
                "bucketed_shapes",
                "parameters",
                "weights",
                "inference_peaks",
                "learner_peaks",
                "catalog",
                "error",
            )
        }
        for kind in ("inference", "learner"):
            seconds = receipt.get(f"{kind}_seconds")
            if seconds:
                entry[f"{kind}_total_s_per_pass"] = [float(sum(p)) for p in seconds]
            trace = replay_dir / f"{kind}-trace.json"
            if trace.exists():
                span = (
                    "mtg301/inference_call"
                    if kind == "inference"
                    else "mtg301/learner_step"
                )
                rows = trace_windows(trace, span)
                entry[f"{kind}_windows"] = rows
                entry[f"{kind}_bounds"] = [_bounds(r) for r in rows]
        if (replay_dir / "syncs.json").exists():
            syncs = json.loads((replay_dir / "syncs.json").read_text())
            entry["syncs"] = {k: syncs[k] for k in ("per_label", "sites")}
        summary["replays"][replay_dir.name] = entry
    comparisons = {}
    for path in sorted((out / "numerics").glob("*-*.pt")):
        if path.name.startswith("baseline-"):
            continue
        base = path.with_name("baseline-" + path.name.split("-", 1)[1])
        if base.exists():
            comparisons[path.stem] = compare(base, path)
    summary["numerics"] = comparisons
    _write(out / "summary.json", summary)
    return summary


# --------------------------------------------------------------------------
# Job orchestration


def _option3_tree(target: Path) -> Path:
    """A copy of `manabot/` with the option-3 patch applied (Python only)."""
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.copytree(ROOT / "manabot", target / "manabot")
    # Package data resolves against the repository root (`content/`, ...).
    for entry in ROOT.iterdir():
        if entry.name not in {"manabot", ".git", ".venv"}:
            (target / entry.name).symlink_to(entry)
    subprocess.run(
        ["patch", "-p1", "--forward", "-i", str(PATCH)], cwd=target, check=True
    )
    return target


@dataclass(frozen=True)
class Arm:
    """Source tree (None: the job revision) and switches for one arm's processes."""

    tree: Path | None = None
    environment: tuple[tuple[str, str], ...] = ()


def _arms(candidate: str, work: Path) -> dict[str, Arm]:
    """Baseline plus one candidate: the option-3 patch, or MTG-302's switch."""
    if candidate == "option3":
        return {"baseline": Arm(), "option3": Arm(_option3_tree(work / "option3"))}
    if candidate == "bucketed":
        if not (ROOT / "manabot" / "model" / "buckets.py").exists():
            raise RuntimeError("this revision has no bucketed-shapes switch")
        return {
            "baseline": Arm(),
            "bucketed": Arm(environment=(("MANABOT_BUCKETED_SHAPES", "1"),)),
        }
    raise ValueError(f"unknown candidate {candidate!r}")


def _clean_env(arm: Arm) -> dict[str, str]:
    # Stage processes track locally, never as the launcher's run; only the
    # arm's own switches reach them.
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("ETUDE_") and k != "MANABOT_BUCKETED_SHAPES"
    }
    # A patched tree shadows the editable install's `manabot`.
    env["PYTHONPATH"] = str(arm.tree or ROOT)
    env.update(arm.environment)
    return env


def _stage(args: list[str], log: Path, arm: Arm, timeout: float) -> dict[str, Any]:
    tick = time.perf_counter()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        try:
            code = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), *args],
                cwd=arm.tree or ROOT,
                env=_clean_env(arm),
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            ).returncode
        except subprocess.TimeoutExpired:
            code = -9
    # Etude retains stdout; a failed stage's tail stays readable without outputs.
    tail = log.read_text(errors="replace").splitlines()[-40 if code else -3 :]
    print(
        f"== {' '.join(args[:1])} {log.name}: exit {code}", *tail, sep="\n", flush=True
    )
    return {"args": args, "returncode": code, "seconds": time.perf_counter() - tick}


def job(smoke: bool = False) -> None:
    """All stages in order; `smoke` runs them on CPU at toy scale."""
    out = Path(os.environ["ETUDE_OUTPUT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    # Attempts on one rented machine share /tmp; each takes its own directory.
    work = Path(os.environ.get("MTG301_WORK") or tempfile.mkdtemp(prefix="mtg301-"))
    captures = work / "captures"
    device = "cpu" if smoke else "cuda"
    flags = ["--smoke"] if smoke else []
    larger = ("w256l8",) if smoke else ("w384l8", "w256l8")
    log: list[dict[str, Any]] = []
    started = time.perf_counter()

    candidate = os.environ.get("MTG301_CANDIDATE", "option3")
    arms: dict[str, Arm] = {"baseline": Arm()}

    def replay_stage(arm: str, size: str, name: str) -> bool:
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
                ],
                out / "logs" / f"replay-{name}.log",
                arms[arm],
                1200,
            )
        )
        receipt = out / "replay" / name / "receipt.json"
        return receipt.exists() and json.loads(receipt.read_text())["status"] == "ok"

    try:
        log.append(
            _stage(
                ["insitu", str(out / "insitu"), str(captures), *flags],
                out / "logs" / "insitu.log",
                arms["baseline"],
                1800,
            )
        )
        if log[-1]["returncode"] != 0:
            raise RuntimeError("in-situ stage failed; nothing to replay")
        arms = _arms(candidate, work)
        for arm in arms:
            log.append(
                _stage(
                    [
                        "numerics",
                        str(captures),
                        str(out / "numerics" / f"{arm}-w160l8.pt"),
                        "w160l8",
                        device,
                    ],
                    out / "logs" / f"numerics-{arm}.log",
                    arms[arm],
                    600,
                )
            )
        # Interleaved so that host drift shows as a baseline repeat difference.
        large = next(
            (s for s in larger if replay_stage("baseline", s, f"baseline-{s}")), None
        )
        replay_stage("baseline", "w160l8", "baseline-w160l8")
        replay_stage(candidate, "w160l8", f"{candidate}-w160l8")
        if large:
            replay_stage(candidate, large, f"{candidate}-{large}")
        replay_stage("baseline", "w160l8", "baseline-w160l8-repeat")
    finally:
        _write(
            out / "job.json", {"stages": log, "seconds": time.perf_counter() - started}
        )
        _finish(out, captures)


def _finish(out: Path, captures: Path) -> None:
    """Analyze what exists, keep a few inputs, compress traces and pack outputs."""
    try:
        analyze(out)
    except Exception:  # noqa: BLE001 - partial evidence is still delivered
        (out / "analyze-error.txt").write_text(traceback.format_exc())
        print(traceback.format_exc(), flush=True)
    keep = out / "inputs"
    for kind, count in (("learner", 4), ("inference", 8)):
        for path in sorted((captures / kind).glob("*"))[:count]:
            (keep / kind).mkdir(parents=True, exist_ok=True)
            shutil.copy(path, keep / kind / path.name)
    for name in ("regime.json", "initial.pt"):
        if (captures / name).exists():
            keep.mkdir(parents=True, exist_ok=True)
            shutil.copy(captures / name, keep / name)
    for trace in out.rglob("*-trace.json"):
        with trace.open("rb") as src, gzip.open(f"{trace}.gz", "wb") as dst:
            shutil.copyfileobj(src, dst)
        trace.unlink()
    packed = captures.parent / "results.tar.gz"
    with tarfile.open(packed, "w:gz") as archive:
        for path in sorted(out.iterdir()):
            archive.add(path, arcname=path.name)
    shutil.move(packed, out / "results.tar.gz")


def runs(revision: str, candidate: str = "option3", hours: str = "1.25") -> Any:
    """`candidate`: option3, or bucketed at a revision carrying MTG-302's switch."""
    from etude.launch.records import Job, Variable

    from manabot.experiment.job import TOOLCHAIN_ENV, machine_setup

    return Job(
        name=f"mtg301-l4-step-profile-{candidate}",
        environment=(
            Variable(name="MTG301_CANDIDATE", value=candidate),
            Variable(name="MTG301_REVISION", value=revision),
        ),
        setup=machine_setup(""),
        command=(
            'mkdir -p "$ETUDE_OUTPUT_DIR"\n'
            "nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used "
            '--format=csv,noheader,nounits -l 1 > "$ETUDE_OUTPUT_DIR/gpu-samples.csv" &\n'
            "sampler=$!\n"
            "trap 'kill \"$sampler\"' EXIT\n"
            f"{TOOLCHAIN_ENV}\n"
            "uv run --locked python -m experiments.l4_step_profile job"
        ),
        repo=str(ROOT),
        revision=revision,
        max_hours=float(hours),
        max_usd=Decimal("1.50"),
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
    p = sub.add_parser("replay")
    p.add_argument("captures", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("arm")
    p.add_argument("size", choices=sorted(SIZES))
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("numerics")
    p.add_argument("captures", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("size", choices=sorted(SIZES))
    p.add_argument("device")
    p = sub.add_parser("compare")
    p.add_argument("first", type=Path)
    p.add_argument("second", type=Path)
    p = sub.add_parser("analyze")
    p.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.command == "job":
        job(args.smoke)
    elif args.command == "insitu":
        insitu(args.out, args.captures, args.smoke)
    elif args.command == "replay":
        replay(args.captures, args.out, args.arm, args.size, args.device)
    elif args.command == "numerics":
        numerics(args.captures, args.out, args.size, args.device)
    elif args.command == "compare":
        print(json.dumps(compare(args.first, args.second), indent=1))
    else:
        analyze(args.out)


if __name__ == "__main__":
    main()
