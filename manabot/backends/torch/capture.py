"""Capture real Torch policy calls and learner inputs for backend comparisons.

`capture` runs a supplied one-update regime, retaining the initial checkpoint,
full-capacity observation arrays, selected targets and per-minibatch settings.
The instrumentation observes the production damped update at its checked-loss
boundary. Hook frame access is confined here; readers use `backends.parity`.
Disk serialization is subtracted from the shared engine time, never counted as
engine work. Captures are execution evidence, not resumable training state.
"""

import inspect
import json
from pathlib import Path
import platform
import shutil
import time
from typing import Any

import numpy as np
import torch

from manabot.backends.torch.policy import ActorCritic, PolicyInference
from manabot.env.observation import ObservationBuffers
from manabot.policy.prepare import Prepared
from manabot.training.execution import execute_regime
from manabot.training.regime import TrainingRegime
from manabot.training.rollout.collection import RolloutBatch, SeatRoutedCollector
from manabot.training.updates.health import OptimizerHealth
from manabot.verify.store import VerifyStore


def capture(out: Path, regime: TrainingRegime, seed: int, *, experiment: str) -> None:
    """Record one real damped update; restore hooks even when execution fails."""
    if len(regime.stages) != 1 or getattr(regime.stages[0], "updates", None) != 1:
        raise ValueError("backend capture requires exactly one self-play update")
    out.mkdir(parents=True, exist_ok=False)
    (out / "collection").mkdir()
    (out / "learner").mkdir()
    (out / "regime.json").write_text(regime.model_dump_json(indent=2))

    calls: list[dict[str, Any]] = []
    minibatches: list[dict[str, Any]] = []
    collection_seconds = 0.0
    learner_decisions = 0
    recording_seconds = 0.0
    in_collection = False
    original_prepare = PolicyInference.prepare
    original_forward = ActorCritic.forward
    original_collect = SeatRoutedCollector.collect
    original_check = OptimizerHealth.check

    def prepare(
        self: PolicyInference, obs: ObservationBuffers
    ) -> Prepared[torch.Tensor]:
        nonlocal recording_seconds
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
        recording_start = time.perf_counter()
        np.savez(
            out / "collection" / f"{len(calls) - 1:05d}.npz",
            **obs,
        )
        recording_seconds += time.perf_counter() - recording_start
        return prepared

    def forward(
        self: ActorCritic, obs: Prepared[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tick = time.perf_counter()
        result = original_forward(self, obs)
        if in_collection and calls:
            calls[-1]["forward"] += time.perf_counter() - tick
        return result

    def collect(self: SeatRoutedCollector, *args: Any, **kwargs: Any) -> RolloutBatch:
        nonlocal in_collection, collection_seconds, learner_decisions
        in_collection = True
        tick = time.perf_counter()
        try:
            batch = original_collect(self, *args, **kwargs)
            learner_decisions = int(batch.actions.size)
        finally:
            in_collection = False
            collection_seconds = time.perf_counter() - tick
        return batch

    def check(self: OptimizerHealth, loss: torch.Tensor) -> None:
        original_check(self, loss)
        frame = inspect.currentframe()
        assert frame is not None and frame.f_back is not None
        local = frame.f_back.f_locals
        assert frame.f_back.f_code.co_name == "update_damped_iteration"
        indexes = local["indexes"]
        learning = local["learning"]
        arrays = {f"obs/{k}": v for k, v in local["rows"].observation.items()}
        arrays.update(
            actions=local["minibatch_actions"].detach().cpu().numpy(),
            advantages=local["advantages"][indexes].detach().cpu().numpy(),
            value_targets=local["value_targets"][indexes].detach().cpu().numpy(),
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
                "learning_rate": float(local["rate"]),
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
    meta: dict[str, Any] = {
        "host_before": platform.platform(),
        "torch": torch.__version__,
    }
    try:
        with VerifyStore(out / "capture.sqlite") as store:
            result = execute_regime(
                regime,
                seed,
                out / "execution",
                store,
                experiment=experiment,
                training_exposure=False,
            )
    finally:
        PolicyInference.prepare = original_prepare  # type: ignore[method-assign]
        ActorCritic.forward = original_forward  # type: ignore[method-assign]
        SeatRoutedCollector.collect = original_collect  # type: ignore[method-assign]
        OptimizerHealth.check = original_check  # type: ignore[method-assign]
    initial = sorted((out / "execution").rglob("*-initial-raw.safetensors"))
    if result.status != "completed" or not calls or not minibatches:
        raise RuntimeError(f"capture did not complete one update: {result.status}")
    if len(initial) != 1:
        raise RuntimeError(f"expected one initial checkpoint, found {initial}")
    shutil.copyfile(initial[0], out / "initial.pt")
    policy = sum(c["prepare"] + c["forward"] for c in calls)
    meta.update(
        status=str(result.status),
        wall=time.perf_counter() - started,
        collection_wall=collection_seconds,
        collection_policy=policy,
        engine=collection_seconds - policy - recording_seconds,
        recording_seconds=recording_seconds,
        learner_decisions=learner_decisions,
        calls=len(calls),
        minibatches=len(minibatches),
        host_after=platform.platform(),
    )
    (out / "calls.json").write_text(json.dumps(calls, indent=1))
    (out / "minibatches.json").write_text(json.dumps(minibatches, indent=1))
    (out / "capture.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps(meta, indent=2, default=str))
