"""MLX pilot on an Apple M2: capture real w160l8 inputs, then time MLX against CPU8.

Report: experiments/mlx_pilot.md. Three commands, all run on the measurement host:

- `capture OUT`: one real CPU update of the frozen TLA frontier w160l8 regime
  (`experiments.tla_frontier.frozen`), recording every collection policy input,
  every learner minibatch and the time collection spends outside the policy.
  The run itself is unchanged: hooks only copy inputs and read clocks.
- `bench OUT --phase P`: one phase per process. `parity`
  (`manabot.backends.parity`) gates the rest; `collection` times per-call
  latency at B = 1, 8, 64, 256; `learner` and `learner-mps` time the learner
  step at `--width` 160 or 384, CPU8 alternating with MLX or with torch MPS;
  `widths` finds the largest width whose MLX step fits 24 GiB.
- `soak OUT`: the whole update replayed in each arm (the MPS study's X3
  shape): every captured collection call, then every learner minibatch with
  weights evolving from the initial checkpoint. Update wall time is the
  measured policy and learner time plus the capture's engine time, which both
  arms share because the engine is the same CPU code.
- `report BENCH OUT`: `summary.json` (medians, paired ratios, the decision)
  and one chart per question.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass
import gc
import inspect
import json
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BATCH_SIZES = (1, 8, 64, 256)
CPU_THREADS = 8


def _uptime() -> str:
    return subprocess.check_output(["/usr/bin/uptime"], text=True).strip()


def _emit(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(row, default=str) + "\n")


def _host() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "uptime": _uptime(),
    }


# -- capture -------------------------------------------------------------------


def capture(out: Path) -> None:
    """Run one CPU8 update of the w160l8 regime, keeping its inputs as npz."""
    import torch

    from experiments.tla_frontier import frozen
    from manabot.model.actor_critic import ActorCritic
    from manabot.model.inference import PolicyInference
    from manabot.training.execution import execute_regime
    from manabot.training.learning.health import OptimizerHealth
    from manabot.training.regime import TrainingRegime
    from manabot.training.rollout.collection import SeatRoutedCollector
    from manabot.verify.store import VerifyStore

    out.mkdir(parents=True, exist_ok=False)
    (out / "collection").mkdir()
    (out / "learner").mkdir()
    regimes, protocol = frozen()
    data = regimes["w160l8"].model_dump()
    stage = data["stages"][0]
    stage["updates"] = 1
    # Regimes allow at most four threads; the capture keeps only inputs and the
    # engine's share of collection time, which policy threads do not change.
    stage["execution"].update(device="cpu", workers=1, threads=4, wall_seconds=7200.0)
    data["wall_seconds"] = 7200.0
    regime = TrainingRegime.model_validate(data)
    (out / "regime.json").write_text(regime.model_dump_json(indent=2))

    calls: list[dict[str, Any]] = []
    minibatches: list[dict[str, Any]] = []
    clock: dict[str, float] = {}
    in_collection = False
    original_prepare = PolicyInference.prepare
    original_forward = ActorCritic.forward
    original_collect = SeatRoutedCollector.collect
    original_check = OptimizerHealth.check

    def prepare(self: PolicyInference, obs: dict[str, torch.Tensor]) -> Any:
        if not in_collection:
            return original_prepare(self, obs)
        tick = time.perf_counter()
        prepared = original_prepare(self, obs)
        calls.append(
            {
                "index": len(calls),
                "B": int(obs["actions_valid"].shape[0]),
                "prepare": time.perf_counter() - tick,
                "forward": 0.0,
            }
        )
        np.savez(
            out / "collection" / f"{len(calls) - 1:05d}.npz",
            **{key: value.numpy() for key, value in obs.items()},
        )
        return prepared

    def forward(self: ActorCritic, obs: Any) -> Any:
        tick = time.perf_counter()
        result = original_forward(self, obs)
        if in_collection and calls:
            calls[-1]["forward"] += time.perf_counter() - tick
        return result

    def collect(self: SeatRoutedCollector, *args: Any, **kwargs: Any) -> Any:
        nonlocal in_collection
        in_collection = True
        tick = time.perf_counter()
        try:
            batch = original_collect(self, *args, **kwargs)
        finally:
            in_collection = False
            clock["collection"] = time.perf_counter() - tick
        clock["learning_start"] = time.perf_counter()
        return batch

    def check(self: OptimizerHealth, loss: torch.Tensor) -> None:
        original_check(self, loss)
        frame = inspect.currentframe()
        assert frame is not None and frame.f_back is not None
        local = frame.f_back.f_locals
        assert frame.f_back.f_code.co_name == "update_damped_iteration"
        indexes = local["indexes"]
        targets = local["targets"]
        learning = local["learning"]
        arrays = {f"obs/{k}": v.detach().numpy() for k, v in local["obs"].items()}
        arrays.update(
            actions=local["actions"][indexes].numpy(),
            advantages=targets.advantages[indexes].numpy(),
            value_targets=targets.values[indexes].numpy(),
            actor_mask=local["actor_mask"].numpy(),
            behavior_logs=local["behavior_logs"].numpy(),
        )
        np.savez(out / "learner" / f"{len(minibatches):03d}.npz", **arrays)
        minibatches.append(
            {
                "index": len(minibatches),
                "B": int(len(indexes)),
                "loss": float(loss.detach()),
                "iteration": int(local["iteration"]),
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                "tau": float(local["tau"]),
                "clip": learning.clip,
                "collection_kl": learning.collection_kl,
                "reference": learning.reference,
                "max_grad_norm": learning.max_grad_norm,
                "time": time.perf_counter(),
            }
        )
        del local, frame

    PolicyInference.prepare = prepare  # type: ignore[method-assign]
    ActorCritic.forward = forward  # type: ignore[method-assign]
    SeatRoutedCollector.collect = collect  # type: ignore[method-assign]
    OptimizerHealth.check = check  # type: ignore[method-assign]
    started = time.perf_counter()
    meta: dict[str, Any] = {"host_before": _host(), "torch": torch.__version__}
    try:
        with VerifyStore(out / "capture.sqlite") as store:
            result = execute_regime(
                regime,
                protocol.training_seed,
                out / "execution",
                store,
                experiment="mlx-pilot-capture",
                training_exposure=False,
            )
    finally:
        PolicyInference.prepare = original_prepare  # type: ignore[method-assign]
        ActorCritic.forward = original_forward  # type: ignore[method-assign]
        SeatRoutedCollector.collect = original_collect  # type: ignore[method-assign]
        OptimizerHealth.check = original_check  # type: ignore[method-assign]
    initial = sorted((out / "execution").rglob("*-initial-raw.pt"))
    if len(initial) != 1:
        raise RuntimeError(f"expected one initial checkpoint, found {initial}")
    shutil.copyfile(initial[0], out / "initial.pt")
    policy = sum(c["prepare"] + c["forward"] for c in calls)
    meta.update(
        status=str(result.status),
        wall=time.perf_counter() - started,
        collection_wall=clock.get("collection"),
        collection_policy=policy,
        engine=clock["collection"] - policy,
        calls=len(calls),
        minibatches=len(minibatches),
        host_after=_host(),
    )
    (out / "calls.json").write_text(json.dumps(calls, indent=1))
    (out / "minibatches.json").write_text(json.dumps(minibatches, indent=1))
    (out / "capture.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps(meta, indent=2, default=str))


# -- shared bench helpers ------------------------------------------------------


# -- arms ------------------------------------------------------------------------------


def _quiet() -> float:
    """The one-minute load average, read before and after each measured block."""
    return float(_uptime().split("load averages:")[1].split()[0])


class TorchLearner:
    """The production learner step: damped loss, `OptimizerHealth` backward,
    clipping and Adam, on CPU (8 threads) or MPS."""

    def __init__(self, agent: Any, device: str) -> None:
        import torch

        from manabot.training.learning.health import OptimizerHealth

        self.torch = torch
        self.device = torch.device(device)
        self.agent = agent.to(self.device).train()
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=2.5e-4, eps=1e-5)
        self.health = {"sampled_steps": 0, "rejected_steps": 0, "optimizer_steps": 0}
        self.observer = OptimizerHealth(self.agent, self.optimizer, self.health, 2)

    def sync(self) -> None:
        if self.device.type == "mps":
            self.torch.mps.synchronize()

    def step(
        self, obs: dict[str, np.ndarray], targets: Any, settings: Any
    ) -> tuple[float, float]:
        torch = self.torch
        from manabot.model.admission import bind_admitted
        from manabot.model.inference import transfer_admitted
        from manabot.model.numerics import float64_operand
        from manabot.model.policy_distribution import legal_mask
        from manabot.training.learning.damped_policy import damped_policy_loss
        from manabot.training.learning.references import reference_distribution
        from manabot.training.learning.selection import selected_mean

        device = self.device
        admitted, _ = transfer_admitted(
            bind_admitted(self.agent, {k: torch.from_numpy(v) for k, v in obs.items()}),
            device,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = settings.learning_rate
        valid = legal_mask(admitted["actions_valid"])
        logits, values = self.agent.forward_distribution(admitted)
        logits, values = logits.float(), values.float()
        reference = reference_distribution(admitted, settings.reference)
        policy = damped_policy_loss(
            logits,
            torch.from_numpy(targets.actions).to(device),
            torch.from_numpy(targets.advantages).to(device),
            torch.from_numpy(targets.behavior_logs).to(device),
            reference,
            valid,
            clip=settings.clip,
            collection_kl=settings.collection_kl,
            tau=settings.tau,
        )
        policy = selected_mean(policy, torch.from_numpy(targets.actor_mask).to(device))
        value = (
            (values.squeeze(-1) - torch.from_numpy(targets.value_targets).to(device))
            .square()
            .mean()
        )
        loss = policy + float64_operand(value)
        self.observer.check(loss)
        norm = self.observer.step(loss, settings.max_grad_norm, checked=True)
        self.sync()
        return float(loss.detach()), float(norm)


@dataclass(frozen=True)
class StepSettings:
    learning_rate: float
    clip: float
    collection_kl: float
    tau: float
    max_grad_norm: float
    reference: str


def _minibatches(
    capture_dir: Path, count: int | None = None
) -> Iterator[tuple[dict[str, np.ndarray], Any, StepSettings]]:
    from manabot.backends.parity import minibatches

    for obs, targets, settings, meta in minibatches(capture_dir, count):
        yield (
            obs,
            targets,
            StepSettings(
                settings.learning_rate,
                settings.clip,
                settings.collection_kl,
                settings.tau,
                settings.max_grad_norm,
                meta["reference"],
            ),
        )


def _agents(capture_dir: Path, width: int) -> tuple[Any, Any]:
    """The capture's initial checkpoint at w160l8, or a fresh seeded model of
    the same architecture at another width; and its MLX twin."""
    import torch

    from manabot.backends.mlx.checkpoint import build
    from manabot.model.actor_critic import ActorCritic
    from manabot.model.checkpoint import load_checkpoint

    agent, space = load_checkpoint(str(capture_dir / "initial.pt"))
    if width != agent.config.hidden_dim:
        config = agent.config.model_copy(update={"hidden_dim": width})
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(21501)
            agent = ActorCritic(space, config)
    state = {k: v.detach().numpy().copy() for k, v in agent.state_dict().items()}
    policy, _ = build(agent.config, space.encoder.config, state)
    return agent, policy


def _mlx_step(
    learner: Any,
    policy: Any,
    obs: dict[str, np.ndarray],
    targets: Any,
    settings: StepSettings,
) -> tuple[float, float]:
    from manabot.policy.protocol import Settings

    return learner.step(
        policy.prepare(obs),
        targets,
        Settings(
            settings.learning_rate,
            settings.clip,
            settings.collection_kl,
            settings.tau,
            settings.max_grad_norm,
        ),
    )


# -- bench ---------------------------------------------------------------------------


def _collection_batches(
    capture_dir: Path, size: int, count: int
) -> list[dict[str, np.ndarray]]:
    """`count` batches of exactly `size` real rows, cut from consecutive
    captured collection calls (spread over the update)."""
    calls = sorted((capture_dir / "collection").glob("*.npz"))
    batches: list[dict[str, np.ndarray]] = []
    starts = np.linspace(0, len(calls) - 1, count + 1).astype(int)[:-1]
    for start in starts:
        rows: list[dict[str, np.ndarray]] = []
        total = 0
        index = int(start)
        while total < size:
            call = _load_npz(calls[index % len(calls)])
            take = min(size - total, len(call["actions_valid"]))
            rows.append({k: v[:take] for k, v in call.items()})
            total += take
            index += 1
        batches.append({k: np.concatenate([r[k] for r in rows]) for k in rows[0]})
    return batches


def _torch_inference(agent: Any, inference: Any, obs: dict[str, np.ndarray]) -> float:
    """One collection policy call inside an open `PolicyInference` scope."""
    import torch

    tick = time.perf_counter()
    with torch.no_grad():
        logits, _ = agent(
            inference.prepare({k: torch.from_numpy(v) for k, v in obs.items()})
        )
        logits.numpy()
    return time.perf_counter() - tick


def _torch_scope(agent: Any) -> tuple[Any, float]:
    """Open the collection scope and encode its catalog once, timed."""
    import torch

    from manabot.model.inference import PolicyInference

    tick = time.perf_counter()
    inference = PolicyInference(agent).__enter__()
    with torch.no_grad():
        inference.catalog()
    return inference, time.perf_counter() - tick


def _mlx_inference(policy: Any, obs: dict[str, np.ndarray]) -> float:
    import mlx.core as mx

    tick = time.perf_counter()
    logits, values = policy(policy.prepare(obs, inference=True))
    mx.eval(logits, values)
    np.array(logits)
    return time.perf_counter() - tick


PHASES = ("parity", "collection", "learner", "learner-mps", "widths")


def bench(out: Path, capture_dir: Path, phase: str, width: int = 160) -> None:
    """One measurement phase per process, so arms never share GPU memory.

    `parity` gates the rest: later phases refuse to run without a passing
    parity record in the same ledger.
    """
    import mlx.core as mx
    import torch

    out.mkdir(parents=True, exist_ok=True)
    ledger = out / "bench.jsonl"
    torch.set_num_threads(CPU_THREADS)
    _emit(
        ledger,
        {
            "event": "phase",
            "phase": phase,
            "width": width,
            **_host(),
            "torch": torch.__version__,
            "mlx": mx.__version__,
        },
    )
    if phase == "parity":
        from manabot.backends.parity import check

        report = check(
            capture_dir / "initial.pt", capture_dir, 8, 16, backend_name="mlx"
        )
        failures = report.failures()
        _emit(ledger, {"event": "parity", "failures": failures, **vars(report)})
        if failures:
            raise SystemExit(f"parity failed ({failures}); no timing")
        return
    passed = [
        r
        for r in map(json.loads, ledger.read_text().splitlines())
        if r.get("event") == "parity" and not r["failures"]
    ]
    if not passed:
        raise SystemExit("no passing parity record; run the parity phase first")
    {
        "collection": _bench_collection,
        "learner": _bench_learner,
        "learner-mps": _bench_learner,
        "widths": _bench_widths,
    }[phase](ledger, capture_dir, phase, width)


def _bench_collection(ledger: Path, capture_dir: Path, phase: str, width: int) -> None:
    """Per-call latency: one PyTorch inference scope for the whole block, as
    collection opens one per collection; MLX encodes its catalog once."""
    agent, policy = _agents(capture_dir, width)
    agent.eval()
    policy.cache_programs()
    inference, _ = _torch_scope(agent)
    for size in BATCH_SIZES:
        batches = _collection_batches(capture_dir, size, 8)
        _torch_inference(agent, inference, batches[0])
        _mlx_inference(policy, batches[0])
        load = _quiet()
        for rep in range(3):
            for index, obs in enumerate(batches):
                _emit(
                    ledger,
                    {
                        "event": "collection",
                        "B": size,
                        "batch": index,
                        "rep": rep,
                        "cpu8": _torch_inference(agent, inference, obs),
                        "mlx": _mlx_inference(policy, obs),
                    },
                )
        _emit(
            ledger,
            {
                "event": "load",
                "block": f"collection B={size}",
                "before": load,
                "after": _quiet(),
            },
        )
    inference.__exit__(None, None, None)


def _bench_learner(ledger: Path, capture_dir: Path, phase: str, width: int) -> None:
    """Learner steps with weights evolving; CPU8 alternates per minibatch with
    MLX (`learner`) or with torch MPS (`learner-mps`)."""
    import mlx.core as mx
    import torch

    from manabot.backends.mlx.learner import MLXLearner

    label = f"w{width}l8"
    count = 24 if phase == "learner" else 12
    agent, policy = _agents(capture_dir, width)
    state = {k: v.detach().clone() for k, v in agent.state_dict().items()}
    cpu = TorchLearner(agent, "cpu")
    other: Any
    if phase == "learner":
        other = MLXLearner(policy)
        arm = "mlx"
        mx.reset_peak_memory()
    else:
        del policy
        mps_agent, _ = _agents(capture_dir, width)
        mps_agent.load_state_dict(state)
        other = TorchLearner(mps_agent, "mps")
        arm = "mps"
    peak = 0
    load = _quiet()
    for index, (obs, targets, settings) in enumerate(_minibatches(capture_dir, count)):
        row: dict[str, Any] = {
            "event": "learner",
            "width": label,
            "minibatch": index,
            "B": len(targets.actions),
            "pair": arm,
        }
        tick = time.perf_counter()
        row["cpu8_loss"], row["cpu8_norm"] = cpu.step(obs, targets, settings)
        row["cpu8"] = time.perf_counter() - tick
        tick = time.perf_counter()
        if arm == "mlx":
            row["mlx_loss"], row["mlx_norm"] = _mlx_step(
                other, policy, obs, targets, settings
            )
        else:
            row["mps_loss"], row["mps_norm"] = other.step(obs, targets, settings)
            peak = max(peak, torch.mps.driver_allocated_memory())
        row[arm] = time.perf_counter() - tick
        _emit(ledger, row)
        print(json.dumps(row), flush=True)
    memory = (
        {"mlx_peak": mx.get_peak_memory()}
        if arm == "mlx"
        else {"mps_driver_peak": peak}
    )
    _emit(ledger, {"event": "memory", "width": label, "pair": arm, **memory})
    _emit(
        ledger,
        {
            "event": "load",
            "block": f"{phase} {label}",
            "before": load,
            "after": _quiet(),
        },
    )


def _bench_widths(ledger: Path, capture_dir: Path, phase: str, width: int) -> None:
    """The largest width whose MLX learner step fits 24 GiB, on the biggest of
    the first 24 captured minibatches; stops before a predicted overflow."""
    import mlx.core as mx

    from manabot.backends.mlx.learner import MLXLearner

    obs, targets, settings = max(
        _minibatches(capture_dir, 24), key=lambda s: len(s[1].actions)
    )
    limit = 24 * 2**30
    measured: list[tuple[int, int]] = []
    for candidate in (160, 256, 320, 384, 448, 512, 640):
        if measured:
            # Activations dominate and grow linearly in width at a fixed
            # head count; extrapolate the last two points (or proportionally
            # from the first) and stop before a predicted overflow.
            last_width, last_peak = measured[-1]
            if len(measured) == 1:
                predicted = last_peak * candidate / last_width
            else:
                width_before, peak_before = measured[-2]
                slope = (last_peak - peak_before) / (last_width - width_before)
                predicted = last_peak + slope * (candidate - last_width)
            if predicted > 0.95 * limit:
                _emit(
                    ledger,
                    {
                        "event": "width_limit",
                        "stopped_before": candidate,
                        "predicted_peak": predicted,
                        "last_width": last_width,
                        "last_peak": last_peak,
                    },
                )
                break
        _, policy = _agents(capture_dir, candidate)
        learner = MLXLearner(policy)
        mx.reset_peak_memory()
        tick = time.perf_counter()
        _mlx_step(learner, policy, obs, targets, settings)
        _mlx_step(learner, policy, obs, targets, settings)
        peak = mx.get_peak_memory()
        _emit(
            ledger,
            {
                "event": "width",
                "width": candidate,
                "B": len(targets.actions),
                "mlx_peak": peak,
                "two_steps": time.perf_counter() - tick,
            },
        )
        measured.append((candidate, peak))
        del policy, learner
        gc.collect()
        mx.clear_cache()


# -- soak ----------------------------------------------------------------------------


def soak(out: Path, capture_dir: Path) -> None:
    """Replay the whole captured update in one arm per invocation's order:
    MLX first, then CPU8, each from the initial checkpoint."""
    import mlx.core as mx
    import torch

    from manabot.backends.mlx.learner import MLXLearner

    out.mkdir(parents=True, exist_ok=True)
    ledger = out / "soak.jsonl"
    torch.set_num_threads(CPU_THREADS)
    meta = json.loads((capture_dir / "capture.json").read_text())
    calls = sorted((capture_dir / "collection").glob("*.npz"))
    minibatch_count = len(json.loads((capture_dir / "minibatches.json").read_text()))
    _emit(
        ledger,
        {
            "event": "start",
            **_host(),
            "calls": len(calls),
            "minibatches": minibatch_count,
            "engine": meta["engine"],
        },
    )
    for arm in ("mlx", "cpu8"):
        agent, policy = _agents(capture_dir, 160)
        load = _quiet()
        policy_time = 0.0
        agent.eval()
        if arm == "mlx":
            mx.reset_peak_memory()
            tick = time.perf_counter()
            policy.cache_programs()
            policy_time += time.perf_counter() - tick
        else:
            inference, policy_time = _torch_scope(agent)
        for path in calls:
            obs = _load_npz(path)
            policy_time += (
                _mlx_inference(policy, obs)
                if arm == "mlx"
                else _torch_inference(agent, inference, obs)
            )
        if arm == "mlx":
            policy.clear_cache()
        else:
            inference.__exit__(None, None, None)
        learner: Any = (
            MLXLearner(policy) if arm == "mlx" else TorchLearner(agent, "cpu")
        )
        learn_time = 0.0
        losses: list[float] = []
        for obs, targets, settings in _minibatches(capture_dir):
            tick = time.perf_counter()
            if arm == "mlx":
                loss, _ = _mlx_step(learner, policy, obs, targets, settings)
            else:
                loss, _ = learner.step(obs, targets, settings)
            learn_time += time.perf_counter() - tick
            losses.append(loss)
        row = {
            "event": "arm",
            "arm": arm,
            "collection_policy": policy_time,
            "learning": learn_time,
            "engine": meta["engine"],
            "update": meta["engine"] + policy_time + learn_time,
            "losses": losses,
            "load_before": load,
            "load_after": _quiet(),
        }
        if arm == "mlx":
            row["mlx_peak"] = mx.get_peak_memory()
        _emit(ledger, row)
        print(json.dumps({k: v for k, v in row.items() if k != "losses"}), flush=True)
        del agent, policy, learner
        gc.collect()
        mx.clear_cache()


# -- report --------------------------------------------------------------------------


def _rows(path: Path) -> list[dict[str, Any]]:
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def summarize(bench_dir: Path) -> dict[str, Any]:
    """Medians, paired ratios and the decision from the bench and soak ledgers.

    Learner ratios are the median of per-minibatch CPU8/arm ratios from the
    same alternating pair, excluding each phase's first (warm-up) step.
    """
    import statistics

    rows = _rows(bench_dir / "bench.jsonl")
    soak_rows = [r for r in _rows(bench_dir / "soak.jsonl") if r["event"] == "arm"]
    summary: dict[str, Any] = {
        "collection": {},
        "learner": {},
        "memory": {},
        "widths": [],
    }
    for size in BATCH_SIZES:
        calls = [r for r in rows if r["event"] == "collection" and r["B"] == size]
        if calls:
            summary["collection"][size] = {
                "cpu8": statistics.median(r["cpu8"] for r in calls),
                "mlx": statistics.median(r["mlx"] for r in calls),
                "speedup": statistics.median(r["cpu8"] / r["mlx"] for r in calls),
            }
    for row in rows:
        if row["event"] == "learner" and row["minibatch"] > 0:
            entry = (
                summary["learner"]
                .setdefault(row["width"], {})
                .setdefault(row["pair"], [])
            )
            entry.append(row)
        elif row["event"] == "memory":
            summary["memory"].setdefault(row["width"], {}).update(
                {k: v for k, v in row.items() if k.endswith("peak")}
            )
        elif row["event"] in ("width", "width_limit"):
            summary["widths"].append(row)
    for width, pairs in summary["learner"].items():
        for arm, steps in list(pairs.items()):
            pairs[arm] = {
                "steps": len(steps),
                "cpu8": statistics.median(r["cpu8"] for r in steps),
                arm: statistics.median(r[arm] for r in steps),
                "speedup": statistics.median(r["cpu8"] / r[arm] for r in steps),
                "max_loss_relative": max(
                    abs(r[f"{arm}_loss"] - r["cpu8_loss"]) / abs(r["cpu8_loss"])
                    for r in steps
                ),
            }
    arms = {r["arm"]: r for r in soak_rows}
    if {"mlx", "cpu8"} <= set(arms):
        summary["soak"] = {
            arm: {
                k: arms[arm][k]
                for k in ("engine", "collection_policy", "learning", "update")
            }
            for arm in ("cpu8", "mlx")
        }
        summary["soak"]["speedup"] = arms["cpu8"]["update"] / arms["mlx"]["update"]
        summary["soak"]["mlx_peak"] = arms["mlx"].get("mlx_peak")
        summary["soak"]["final_loss"] = {
            a: arms[a]["losses"][-1] for a in ("cpu8", "mlx")
        }
    learner384 = summary["learner"].get("w384l8", {}).get("mlx", {}).get("speedup")
    whole = summary.get("soak", {}).get("speedup")
    if learner384 is not None and whole is not None:
        summary["accepted"] = whole >= 1.5 and learner384 >= 1.5
    return summary


def report(bench_dir: Path, out: Path) -> None:
    """Write `summary.json` and one chart per question into `out`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(bench_dir)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    colors = {"cpu8": "#4c72b0", "mlx": "#dd8452", "mps": "#55a868"}
    names = {"cpu8": "PyTorch CPU, 8 threads", "mlx": "MLX (GPU)", "mps": "PyTorch MPS"}

    figure, axis = plt.subplots(figsize=(6, 4))
    sizes = sorted(summary["collection"], key=int)
    for arm in ("cpu8", "mlx"):
        axis.plot(
            [int(b) for b in sizes],
            [1e3 * summary["collection"][b][arm] for b in sizes],
            marker="o",
            color=colors[arm],
            label=names[arm],
        )
    axis.set(
        xscale="log",
        yscale="log",
        xlabel="rows per policy call (B)",
        ylabel="median latency per call (ms)",
        title="Collection policy call, w160l8",
    )
    axis.set_xticks([int(b) for b in sizes], [str(b) for b in sizes])
    axis.legend()
    figure.tight_layout()
    figure.savefig(out / "collection_latency.png", dpi=150)

    figure, axis = plt.subplots(figsize=(6, 4))
    widths = [w for w in ("w160l8", "w384l8") if w in summary["learner"]]
    for offset, arm in enumerate(("cpu8", "mlx", "mps")):
        values = []
        for width in widths:
            pairs = summary["learner"][width]
            values.append(
                pairs["mlx"]["cpu8"]
                if arm == "cpu8"
                else pairs.get(arm, {}).get(arm, np.nan)
            )
        axis.bar(
            np.arange(len(widths)) + (offset - 1) * 0.27,
            values,
            0.27,
            color=colors[arm],
            label=names[arm],
        )
    axis.set_xticks(np.arange(len(widths)), widths)
    axis.set(
        ylabel="median seconds per learner step",
        title="Learner step (minibatch of one update)",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(out / "learner_step.png", dpi=150)

    if "soak" in summary:
        figure, axis = plt.subplots(figsize=(6, 3.6))
        parts = (
            ("engine", "engine (shared)", "#999999"),
            ("collection_policy", "collection policy calls", "#8172b3"),
            ("learning", "learner steps", "#c44e52"),
        )
        for row, arm in enumerate(("cpu8", "mlx")):
            left = 0.0
            for key, label, color in parts:
                value = summary["soak"][arm][key]
                axis.barh(
                    row,
                    value,
                    left=left,
                    color=color,
                    label=label if row == 0 else None,
                )
                left += value
        axis.set_yticks([0, 1], [names["cpu8"], names["mlx"]])
        axis.set(xlabel="seconds", title="One whole w160l8 update (replayed)")
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize=8)
        figure.tight_layout()
        figure.savefig(out / "whole_update.png", dpi=150)

    measured = [r for r in summary["widths"] if r["event"] == "width"]
    if measured:
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.plot(
            [r["width"] for r in measured],
            [r["mlx_peak"] / 2**30 for r in measured],
            marker="o",
            color=colors["mlx"],
            label="width sweep: two steps on that minibatch",
        )
        phase = summary["memory"].get("w384l8", {}).get("mlx_peak")
        if phase is not None:
            axis.plot(
                384,
                phase / 2**30,
                marker="s",
                linestyle="none",
                color=colors["mlx"],
                label="w384l8 learner phase: 24 steps, B up to 108",
            )
        axis.axhline(
            24,
            color="black",
            linestyle="--",
            linewidth=1,
            label="24 GiB unified memory",
        )
        axis.set(
            xlabel="hidden width (8 attention layers)",
            ylabel="peak memory (GiB)",
            title=f"MLX learner memory, largest minibatch (B={measured[0]['B']})",
        )
        axis.legend(loc="lower right", fontsize=8)
        figure.tight_layout()
        figure.savefig(out / "memory.png", dpi=150)
    print(json.dumps(summary, indent=2, default=str))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


def _peak_rss() -> int:
    # macOS reports bytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    report_command = commands.add_parser("report")
    report_command.add_argument("bench", type=Path)
    report_command.add_argument("out", type=Path)
    for name in ("capture", "bench", "soak"):
        command = commands.add_parser(name)
        command.add_argument("out", type=Path)
        if name != "capture":
            command.add_argument("--capture", type=Path, required=True)
        if name == "bench":
            command.add_argument("--phase", choices=PHASES, required=True)
            command.add_argument("--width", type=int, default=160)
    args = parser.parse_args()
    if args.command == "report":
        report(args.bench, args.out)
    elif args.command == "capture":
        capture(args.out)
    elif args.command == "bench":
        bench(args.out, args.capture, args.phase, args.width)
    else:
        soak(args.out, args.capture)


if __name__ == "__main__":
    _main()
