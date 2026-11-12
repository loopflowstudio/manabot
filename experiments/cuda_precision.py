"""MTG-317: CUDA training under TF32 and BF16 against float32 on one L4.

Report and the acceptance rule fixed before any scored run:
`experiments/cuda_precision.md`. One rented L4 runs `measure` (`etude.call`):

1. **Capture.** The frozen MTG-267 w160l8 regime (`experiments.tla_frontier.frozen()`)
   plays one real float32 update on CUDA from its own initialization, as MTG-301's
   in-situ stage did. Hooks keep the initial weights, every `collect_every`-th
   collection forward's observation (up to `collect_keep`) and the complete loss
   inputs of `learn_keep` learner minibatches from optimizer step `learn_from`
   (observations, actions, advantages, value targets, behavior logs, actor mask,
   learning rule, tau and learning rate).
2. **Arms.** For each size (w160l8 with the captured initial weights; w384l8, a
   seeded random initialization) and each `bucketed_shapes` setting, five arms
   in order: `float32`, `tf32`, `bf16`, `float32-order` (another minibatch order)
   and `float32-repeat`. Each arm starts from the same weights and
   - runs every kept collection observation through collection's preparation,
     forward and action selection (`_infer`), keeping legal log-probabilities and
     values, then times `inference_passes` more passes per call;
   - takes loss parts, logits, value logits and the full gradient at the initial
     weights on the first `numeric_minibatches` minibatches (no optimizer step);
   - trains `epochs` passes over the kept minibatches from a fresh Adam with
     `update_damped_iteration`'s minibatch body (`_learn`), recording per step
     the policy loss, value loss, clip fraction, gradient norm and wall time.
   The float32 arm is each group's reference; later arms are compared to it in
   process, so full gradients never leave the machine.
3. **Verdict.** `analyze` applies the pre-registered rule (`RULE`) to every
   group and writes `summary.json`; `trajectories.json` keeps every step.

    uv run python -m experiments.cuda_precision smoke   # CPU, toy scale
    uv run --with "etude-sdk[s3,eval]==0.2.15" etude launch \\
        experiments/cuda_precision.py --rent l4 --revision COMMIT
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import time
from typing import Any, Literal, TypedDict

from etude.launch.records import Job, Run
from pydantic import BaseModel, ConfigDict
import torch

from manabot.experiment.job import TOOLCHAIN_ENV, machine_setup
from manabot.infra.source import SourceBundle, bind_etude_source, bundle_at
from manabot.training.regime import DampedPolicyLearning, Precision

ROOT = Path(__file__).resolve().parents[1]
type ArmName = Literal["float32", "tf32", "bf16", "float32-order", "float32-repeat"]
# Run order inside a group; the float32 repeat last shows host drift.
ARMS: tuple[ArmName, ...] = (
    "float32",
    "tf32",
    "bf16",
    "float32-order",
    "float32-repeat",
)
CANDIDATES: tuple[Precision, ...] = ("tf32", "bf16")
SIZES: dict[str, dict[str, int]] = {
    "w160l8": {},
    "w256l8": {"hidden_dim": 256, "attention_layers": 8, "num_attention_heads": 8},
    "w384l8": {"hidden_dim": 384, "attention_layers": 8, "num_attention_heads": 8},
}
# Trajectory metrics the rule compares, as recorded per optimizer step.
METRICS = ("policy_loss", "value_loss", "clip_fraction")
# Collection calls of at least this many rows are "large" (MTG-301's bands).
LARGE_CALL_ROWS = 64


@dataclass(frozen=True)
class Rule:
    """The acceptance rule, fixed in the report before the scored run.

    Per group (size × bucketed shapes), a trajectory metric's deviation is the
    largest per-epoch mean difference from the float32 arm; its reference is the
    larger of the `float32-order` and `float32-repeat` deviations. A candidate
    is *accepted* when every group passes `accept_*`, *rejected* when any group
    fails a `reject_*` bound, and otherwise *unresolved*, which keeps float32.
    """

    accept_deviation_ratio: float = 1.0
    reject_deviation_ratio: float = 2.0
    accept_grad_cosine: float = 0.99
    reject_grad_cosine: float = 0.95
    accept_loss_relative: float = 1e-2
    reject_loss_relative: float = 5e-2
    # A recommended default must be at least this much faster than float32 in
    # both the learner step and large collection calls, at both sizes, with
    # bucketed shapes on (the current recipes' setting).
    default_speedup: float = 1.10


RULE = Rule()


class Probe(BaseModel):
    """The job's configuration."""

    model_config = ConfigDict(extra="forbid")
    source_bundle: SourceBundle | None = None
    seed: int = 31701
    collect_every: int = 12
    collect_keep: int = 48
    learn_from: int = 70
    learn_keep: int = 20
    numeric_minibatches: int = 8
    inference_passes: int = 2
    epochs: int = 10
    sizes: tuple[str, ...] = ("w160l8", "w384l8")
    bucketed: tuple[bool, ...] = (False, True)
    # `smoke` only: every stage on CPU at toy scale (bf16 autocasts on CPU).
    device: Literal["cuda", "cpu"] = "cuda"
    streams: int | None = None
    transitions: int | None = None


SMOKE = Probe(
    device="cpu",
    streams=4,
    transitions=2,
    collect_every=1,
    collect_keep=3,
    learn_from=0,
    learn_keep=2,
    numeric_minibatches=2,
    inference_passes=1,
    epochs=2,
    sizes=("w160l8",),
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True, default=str))


# --------------------------------------------------------------------------- capture


class _Captured(TypedDict):
    obs: dict[str, torch.Tensor]
    actions: torch.Tensor
    advantages: torch.Tensor
    value_targets: torch.Tensor
    behavior_logs: torch.Tensor
    actor_mask: torch.Tensor
    learning: dict[str, Any]
    tau: float
    lr: float


@contextmanager
def _capturing(folder: Path, probe: Probe) -> Iterator[None]:
    """Keep initial weights, collection observations and learner loss inputs."""
    from manabot.model.actor_critic import ActorCritic
    from manabot.model.inference import PreparedObservation
    from manabot.training.learning.health import OptimizerHealth

    (folder / "collection").mkdir(parents=True, exist_ok=True)
    (folder / "learner").mkdir(parents=True, exist_ok=True)
    forward = ActorCritic.forward_distribution
    check = OptimizerHealth.check
    counts = {"collect": 0, "kept": 0, "learn": 0}

    def hooked_forward(
        self: ActorCritic, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not (folder / "initial.pt").exists():
            torch.save(
                {k: v.detach().cpu().clone() for k, v in self.state_dict().items()},
                folder / "initial.pt",
            )
        if not torch.is_grad_enabled() and isinstance(obs, PreparedObservation):
            if (
                counts["collect"] % probe.collect_every == 0
                and counts["kept"] < probe.collect_keep
            ):
                torch.save(
                    {k: v.detach().cpu().clone() for k, v in obs.items()},
                    folder / "collection" / f"{counts['kept']:03d}.pt",
                )
                counts["kept"] += 1
            counts["collect"] += 1
        return forward(self, obs)

    def hooked_check(self: OptimizerHealth, loss: torch.Tensor) -> None:
        check(self, loss)
        step = counts["learn"]
        counts["learn"] += 1
        if not probe.learn_from <= step < probe.learn_from + probe.learn_keep:
            return
        frame = inspect.currentframe()
        assert frame is not None and frame.f_back is not None
        local = frame.f_back.f_locals
        if frame.f_back.f_code.co_name != "update_damped_iteration":
            raise RuntimeError("the captured minibatch is not the damped learner's")
        indexes, targets = local["indexes"], local["targets"]
        payload: _Captured = {
            "obs": {k: v.detach().cpu().clone() for k, v in local["obs"].items()},
            "actions": local["actions"][indexes].detach().cpu().clone(),
            "advantages": targets.advantages[indexes].detach().cpu().clone(),
            "value_targets": targets.values[indexes].detach().cpu().clone(),
            "behavior_logs": local["behavior_logs"].detach().cpu().clone(),
            "actor_mask": local["actor_mask"].detach().cpu().clone(),
            "learning": local["learning"].model_dump(mode="json"),
            "tau": float(local["tau"]),
            "lr": float(local["learner"].optimizer.param_groups[0]["lr"]),
        }
        del local, frame
        torch.save(payload, folder / "learner" / f"{step:03d}.pt")

    ActorCritic.forward_distribution = hooked_forward  # type: ignore[method-assign]
    OptimizerHealth.check = hooked_check  # type: ignore[method-assign]
    try:
        yield
    finally:
        ActorCritic.forward_distribution = forward  # type: ignore[method-assign]
        OptimizerHealth.check = check  # type: ignore[method-assign]


@contextmanager
def _untracked() -> Iterator[None]:
    """The capture update is a local run, never this job's Etude run."""
    saved = {k: v for k, v in os.environ.items() if k.startswith("ETUDE_")}
    for key in saved:
        del os.environ[key]
    try:
        yield
    finally:
        os.environ.update(saved)


def capture(folder: Path, probe: Probe) -> dict[str, object]:
    """One float32 update of the frozen w160l8 regime with the capture hooks."""
    from experiments.tla_frontier import frozen
    from manabot.training.execution import execute_regime
    from manabot.training.regime import TrainingRegime
    from manabot.verify.store import VerifyStore

    regimes, protocol = frozen()
    data = regimes["w160l8"].model_dump(mode="json")
    stage = data["stages"][0]
    stage["updates"] = 1
    stage["execution"].update(wall_seconds=3600.0, device=probe.device)
    if probe.streams is not None:
        stage["streams"] = probe.streams
        stage["execution"]["workers"] = 1
    if probe.transitions is not None:
        stage["transitions"] = probe.transitions
    data["wall_seconds"] = 3600.0
    regime = TrainingRegime.model_validate(data)
    if regime.stages[0].execution.precision != "float32":
        raise RuntimeError("the capture update must train in float32")
    _write(folder / "regime.json", regime.model_dump(mode="json"))
    started = time.perf_counter()
    with (
        _untracked(),
        VerifyStore(folder / "runs.sqlite") as store,
        _capturing(folder, probe),
    ):
        run = execute_regime(
            regime,
            protocol.training_seed,
            folder / "run",
            store,
            training_exposure=False,
        )
    if run.status != "completed":
        raise RuntimeError(f"capture update {run.status}: {run.error}")
    record = run.stages[0]
    return {
        "regime": regime.id,
        "seed": protocol.training_seed,
        "process_seconds": time.perf_counter() - started,
        "collection_seconds": record.collection_seconds,
        "learning_seconds": record.learning_seconds,
        "collection_kept": len(list((folder / "collection").glob("*.pt"))),
        "learner_kept": len(list((folder / "learner").glob("*.pt"))),
    }


# --------------------------------------------------------------------------- arms


@dataclass(frozen=True)
class Minibatch:
    """One kept learner minibatch, its observation admitted for one agent (CPU)."""

    obs: Any  # AdmittedObservation
    actions: torch.Tensor
    advantages: torch.Tensor
    value_targets: torch.Tensor
    behavior_logs: torch.Tensor
    actor_mask: torch.Tensor
    learning: DampedPolicyLearning
    tau: float
    lr: float


@dataclass
class Outputs:
    """Fixed-weight outputs of one arm, compared to its group's float32 arm."""

    inference_logs: list[torch.Tensor] = field(default_factory=list)
    inference_values: list[torch.Tensor] = field(default_factory=list)
    legal: list[torch.Tensor] = field(default_factory=list)
    losses: list[dict[str, float]] = field(default_factory=list)
    learner_logs: list[torch.Tensor] = field(default_factory=list)
    value_logits: list[torch.Tensor] = field(default_factory=list)
    grads: list[torch.Tensor] = field(default_factory=list)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _timed(call: Callable[[], object], device: torch.device) -> float:
    """Milliseconds of one call: CUDA events on CUDA, else the host clock."""
    if device.type != "cuda":
        started = time.perf_counter()
        call()
        return (time.perf_counter() - started) * 1000
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    call()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def _peak(device: torch.device) -> int | None:
    return torch.cuda.max_memory_allocated() if device.type == "cuda" else None


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _agent(
    folder: Path, size: str, bucketed: bool, seed: int, device: torch.device
) -> Any:
    """w160l8 from the captured initial weights; larger sizes from `seed`."""
    from manabot.env.observation import ObservationSpace
    from manabot.model.actor_critic import ActorCritic
    from manabot.training.regime import TrainingRegime

    regime = TrainingRegime.model_validate_json((folder / "regime.json").read_text())
    config = regime.agent.model_copy(update=SIZES[size])
    torch.manual_seed(seed)
    agent = ActorCritic(
        ObservationSpace(regime.observation), config, bucketed_shapes=bucketed
    )
    if not SIZES[size]:
        agent.load_state_dict(torch.load(folder / "initial.pt"))
    return agent.to(device)


def _minibatches(folder: Path, agent: Any) -> list[Minibatch]:
    from manabot.model.admission import admit_observation

    found = []
    for path in sorted((folder / "learner").glob("*.pt")):
        payload: _Captured = torch.load(path, weights_only=False)
        found.append(
            Minibatch(
                obs=admit_observation(agent, payload["obs"]),
                actions=payload["actions"],
                advantages=payload["advantages"],
                value_targets=payload["value_targets"],
                behavior_logs=payload["behavior_logs"],
                actor_mask=payload["actor_mask"],
                learning=DampedPolicyLearning.model_validate(payload["learning"]),
                tau=payload["tau"],
                lr=payload["lr"],
            )
        )
    return found


def _infer(
    agent: Any,
    scope: Any,
    obs: dict[str, torch.Tensor],
    precision: Precision,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collection's preparation, forward and action selection
    (`SeatRoutedCollector._sample_learning_rows` up to rollout recording)."""
    from manabot.model.policy_distribution import legal_mask, policy_logs, require
    from manabot.training.precision import forward_precision

    prepared = scope.prepare(obs)
    with torch.no_grad():
        with forward_precision(precision, scope.device):
            logits, value = agent.forward_distribution(prepared)
        logits, value = logits.float(), value.float()
        if agent.config.value_kind == "categorical_wdl":
            outcome = value.softmax(dim=-1)
            value = outcome[:, 2] - outcome[:, 0]
            outcome.cpu().numpy()
        legal = legal_mask(prepared["actions_valid"])
        logs = policy_logs(logits, legal)
        require(torch.isfinite(value), "nonfinite_collection_values", values=value)
        weights = logs.exp()
        action = torch.multinomial(weights, 1, generator=generator).squeeze(-1)
        logprob = logs.gather(-1, action[:, None]).squeeze(-1)
        weights.cpu().numpy()
        logs.cpu().numpy()
        action.cpu().numpy()
        logprob.cpu().numpy()
        value = value.view(-1)
        value.cpu().numpy()
    return logs, value, legal


@dataclass
class _Loss:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    logits: torch.Tensor
    value_logits: torch.Tensor
    valid: torch.Tensor
    reference: torch.Tensor
    behavior_logs: torch.Tensor
    actions: torch.Tensor
    actor_mask: torch.Tensor


def _loss(learner: Any, batch: Minibatch) -> _Loss:
    """`update_damped_iteration`'s forward and loss on one kept minibatch."""
    from manabot.model.inference import transfer_admitted
    from manabot.model.numerics import float64_operand
    from manabot.model.policy_distribution import legal_mask
    from manabot.training.learning.damped_policy import damped_policy_loss
    from manabot.training.learning.references import reference_distribution
    from manabot.training.learning.selection import selected_mean

    device = learner.device
    obs = transfer_admitted(batch.obs, device)[0]
    valid = legal_mask(obs["actions_valid"])
    behavior_logs = batch.behavior_logs.to(device)
    actions = batch.actions.to(device)
    advantages = batch.advantages.to(device)
    value_targets = batch.value_targets.to(device)
    actor_mask = batch.actor_mask.to(device)
    with learner.forward_precision():
        logits, value_logits = learner.agent.forward_distribution(obs)
    logits, value_logits = logits.float(), value_logits.float()
    reference = reference_distribution(obs, batch.learning.reference)
    policy_loss = damped_policy_loss(
        logits,
        actions,
        advantages,
        behavior_logs,
        reference,
        valid,
        clip=batch.learning.clip,
        collection_kl=batch.learning.collection_kl,
        tau=batch.tau,
    )
    policy_loss = selected_mean(policy_loss, actor_mask)
    if learner.agent.config.value_kind == "categorical_wdl":
        value_loss = -(value_targets * value_logits.log_softmax(-1)).sum(-1).mean()
    else:
        value_loss = (value_logits.squeeze(-1) - value_targets).square().mean()
    loss = policy_loss + float64_operand(value_loss)
    return _Loss(
        loss,
        policy_loss,
        value_loss,
        logits,
        value_logits,
        valid,
        reference,
        behavior_logs,
        actions,
        actor_mask,
    )


def _learn(learner: Any, observer: Any, batch: Minibatch) -> dict[str, float]:
    """One optimizer step with the learner's checks and diagnostics."""
    from manabot.model.numerics import float64_operand
    from manabot.model.policy_distribution import policy_logs, reverse_kl

    for group in learner.optimizer.param_groups:
        group["lr"] = batch.lr
    parts = _loss(learner, batch)
    observer.check(parts.loss)
    norm = observer.step(parts.loss, batch.learning.max_grad_norm, checked=True)
    with torch.no_grad():
        logs = policy_logs(parts.logits, parts.valid)
        probabilities = logs.exp()
        ratio_logs = (logs - parts.behavior_logs).gather(-1, parts.actions[:, None])
        return {
            "loss": float(parts.loss.detach()),
            "policy_loss": float(parts.policy_loss.detach()),
            "value_loss": float(parts.value_loss.detach()),
            "gradient_norm": float(norm),
            "entropy": float(
                -(probabilities * logs.masked_fill(~parts.valid, 0)).sum(-1).mean()
            ),
            "collection_kl": float(
                reverse_kl(logs, parts.behavior_logs, parts.valid).mean()
            ),
            "clip_fraction": float(
                (
                    (float64_operand(ratio_logs.squeeze(-1)).double().exp() - 1).abs()
                    > batch.learning.clip
                )
                .float()
                .mean()
            ),
        }


def _order(arm: ArmName, epoch: int, count: int) -> list[int]:
    """The minibatch order of one epoch: shared by every arm but `float32-order`."""
    generator = torch.Generator().manual_seed(1000 * (arm == "float32-order") + epoch)
    return [int(i) for i in torch.randperm(count, generator=generator)]


def _difference(
    x: torch.Tensor, y: torch.Tensor, where: torch.Tensor | None = None
) -> float:
    if where is not None:
        x, y = x[where], y[where]
    return float((x.double() - y.double()).abs().max()) if x.numel() else 0.0


def _compare(outputs: Outputs, reference: Outputs) -> dict[str, float]:
    """Largest fixed-weight differences of one arm from its group's float32 arm."""
    stats: dict[str, float] = {
        "inference_log_abs": 0.0,
        "inference_tv": 0.0,
        "inference_value_abs": 0.0,
        "learner_log_abs": 0.0,
        "value_logit_abs": 0.0,
        "grad_cosine_min": 1.0,
        "grad_norm_relative": 0.0,
        "grad_abs": 0.0,
    }
    for logs, ref, values, ref_values, legal in zip(
        outputs.inference_logs,
        reference.inference_logs,
        outputs.inference_values,
        reference.inference_values,
        reference.legal,
        strict=True,
    ):
        stats["inference_log_abs"] = max(
            stats["inference_log_abs"], _difference(logs, ref, legal)
        )
        tv = 0.5 * (logs.double().exp() - ref.double().exp()).abs().sum(-1).max()
        stats["inference_tv"] = max(stats["inference_tv"], float(tv))
        stats["inference_value_abs"] = max(
            stats["inference_value_abs"], _difference(values, ref_values)
        )
    for name in ("loss", "policy_loss", "value_loss"):
        stats[f"{name}_abs"] = 0.0
        stats[f"{name}_relative"] = 0.0
    for got, want in zip(outputs.losses, reference.losses, strict=True):
        for name in ("loss", "policy_loss", "value_loss"):
            error = abs(got[name] - want[name])
            stats[f"{name}_abs"] = max(stats[f"{name}_abs"], error)
            stats[f"{name}_relative"] = max(
                stats[f"{name}_relative"], error / max(abs(want[name]), 1e-12)
            )
    for logs, ref, values, ref_values, grad, ref_grad in zip(
        outputs.learner_logs,
        reference.learner_logs,
        outputs.value_logits,
        reference.value_logits,
        outputs.grads,
        reference.grads,
        strict=True,
    ):
        stats["learner_log_abs"] = max(
            stats["learner_log_abs"], _difference(logs, ref, torch.isfinite(ref))
        )
        stats["value_logit_abs"] = max(
            stats["value_logit_abs"], _difference(values, ref_values)
        )
        cosine = float(
            torch.nn.functional.cosine_similarity(
                grad.double(), ref_grad.double(), dim=0
            )
        )
        stats["grad_cosine_min"] = min(stats["grad_cosine_min"], cosine)
        norm, ref_norm = float(grad.norm()), float(ref_grad.norm())
        stats["grad_norm_relative"] = max(
            stats["grad_norm_relative"], abs(norm - ref_norm) / max(ref_norm, 1e-12)
        )
        stats["grad_abs"] = max(stats["grad_abs"], _difference(grad, ref_grad))
    return stats


def run_arm(
    folder: Path,
    probe: Probe,
    size: str,
    bucketed: bool,
    arm: ArmName,
    reference: Outputs | None,
) -> tuple[dict[str, Any], Outputs]:
    """One arm of one group; returns its receipt and fixed-weight outputs."""
    from manabot.model.inference import PolicyInference
    from manabot.model.policy_distribution import NumericalError, policy_logs
    from manabot.training.learning.health import NumericalHealth, OptimizerHealth
    from manabot.training.learning.learner import Learner
    from manabot.training.precision import apply_matmul_precision

    device = torch.device(probe.device)
    precision: Precision = "float32" if arm.startswith("float32") else arm  # type: ignore[assignment]
    apply_matmul_precision(precision)
    agent = _agent(folder, size, bucketed, probe.seed, device)
    collection = [
        torch.load(path) for path in sorted((folder / "collection").glob("*.pt"))
    ]
    minibatches = _minibatches(folder, agent)
    outputs = Outputs()
    receipt: dict[str, Any] = {
        "size": size,
        "bucketed_shapes": bucketed,
        "arm": arm,
        "precision": precision,
        "parameters": sum(p.numel() for p in agent.parameters()),
        "tf32": [
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        ],
        "collection_rows": [int(o["actions_valid"].shape[0]) for o in collection],
        "learner_rows": [int(b.actions.shape[0]) for b in minibatches],
    }
    try:
        # Collection: eval mode in one behavior scope, as an update holds it.
        agent.eval()
        _reset_peak(device)
        generator = torch.Generator(device=device).manual_seed(probe.seed)
        with PolicyInference(agent) as scope:
            for obs in collection:
                logs, values, legal = _infer(agent, scope, obs, precision, generator)
                outputs.inference_logs.append(logs.cpu())
                outputs.inference_values.append(values.cpu())
                outputs.legal.append(legal.cpu())
            receipt["collection_ms"] = [
                [
                    _timed(
                        lambda obs=obs: _infer(agent, scope, obs, precision, generator),
                        device,
                    )
                    for obs in collection
                ]
                for _ in range(probe.inference_passes)
            ]
        receipt["collection_peak_bytes"] = _peak(device)
        # Fixed weights: loss parts, logits and gradient without a step.
        learner = Learner(agent, device, minibatch_rows=256, precision=precision)
        for batch in minibatches[: probe.numeric_minibatches]:
            learner.optimizer.zero_grad(set_to_none=True)
            parts = _loss(learner, batch)
            parts.loss.backward()
            outputs.losses.append(
                {
                    "loss": float(parts.loss.detach()),
                    "policy_loss": float(parts.policy_loss.detach()),
                    "value_loss": float(parts.value_loss.detach()),
                }
            )
            with torch.no_grad():
                outputs.learner_logs.append(
                    policy_logs(parts.logits, parts.valid).cpu()
                )
            outputs.value_logits.append(parts.value_logits.detach().cpu())
            outputs.grads.append(
                torch.cat(
                    [
                        (
                            p.grad if p.grad is not None else torch.zeros_like(p)
                        ).flatten()
                        for p in learner.agent.parameters()
                    ]
                ).cpu()
            )
        learner.optimizer.zero_grad(set_to_none=True)
        # Learning: `epochs` passes from the same weights and a fresh Adam.
        _reset_peak(device)
        health = NumericalHealth(
            optimizer_steps=0, skipped_steps=0, rejected_steps=0, sampled_steps=1
        )
        observer = OptimizerHealth(learner.agent, learner.optimizer, health, 2)
        steps: list[dict[str, float]] = []
        for epoch in range(probe.epochs):
            for index in _order(arm, epoch, len(minibatches)):
                row: dict[str, float] = {}

                def step(index: int = index, row: dict[str, float] = row) -> None:
                    row.update(_learn(learner, observer, minibatches[index]))

                row["ms"] = _timed(step, device)
                row.update(epoch=epoch, minibatch=index)
                steps.append(row)
        receipt["steps"] = steps
        receipt["learner_peak_bytes"] = _peak(device)
        receipt["health"] = dict(health)
        receipt["status"] = "ok"
    except NumericalError as error:
        receipt.update(status="numerical", error=str(error)[:2000])
    except torch.cuda.OutOfMemoryError as error:
        receipt.update(status="oom", error=str(error)[:2000])
    finally:
        apply_matmul_precision("float32")
    if receipt["status"] == "ok" and reference is not None:
        receipt["fixed_weights"] = _compare(outputs, reference)
    _reset_peak(device)
    return receipt, outputs


# --------------------------------------------------------------------------- verdict


def _epoch_means(receipt: dict[str, Any], metric: str) -> list[float]:
    by_epoch: dict[int, list[float]] = {}
    for row in receipt["steps"]:
        by_epoch.setdefault(int(row["epoch"]), []).append(float(row[metric]))
    return [statistics.fmean(by_epoch[e]) for e in sorted(by_epoch)]


def _deviation(receipt: dict[str, Any], base: dict[str, Any], metric: str) -> float:
    return max(
        abs(a - b)
        for a, b in zip(
            _epoch_means(receipt, metric), _epoch_means(base, metric), strict=True
        )
    )


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else math.nan


def _timing(receipt: dict[str, Any]) -> dict[str, float]:
    """Median collection call by size band and median learner step, ms.

    The first epoch's steps warm caches and allocators and are excluded."""
    rows = receipt["collection_rows"]
    passes = receipt["collection_ms"]
    large = [
        ms
        for p in passes
        for ms, r in zip(p, rows, strict=True)
        if r >= LARGE_CALL_ROWS
    ]
    small = [ms for p in passes for ms, r in zip(p, rows, strict=True) if r <= 8]
    steps = [row["ms"] for row in receipt["steps"] if row["epoch"] > 0] or [
        row["ms"] for row in receipt["steps"]
    ]
    return {
        "collection_large_ms": _median(large),
        "collection_small_ms": _median(small),
        "collection_pass_ms": _median([sum(p) for p in passes]),
        "learner_step_ms": _median(steps),
    }


def group_verdict(arms: dict[str, dict[str, Any]], rule: Rule = RULE) -> dict[str, Any]:
    """Each candidate's standing in one group under `rule`."""
    base = arms["float32"]
    reference = {
        metric: max(
            _deviation(arms["float32-order"], base, metric),
            _deviation(arms["float32-repeat"], base, metric),
        )
        for metric in METRICS
    }
    found: dict[str, Any] = {"reference": reference, "candidates": {}}
    for candidate in CANDIDATES:
        receipt = arms[candidate]
        if receipt["status"] != "ok":
            found["candidates"][candidate] = {
                "verdict": "rejected",
                "why": receipt["status"],
            }
            continue
        health = receipt["health"]
        fixed = receipt["fixed_weights"]
        ratios = {
            metric: _deviation(receipt, base, metric) / max(reference[metric], 1e-12)
            for metric in METRICS
        }
        loss_relative = max(
            fixed[f"{name}_relative"] for name in ("loss", "policy_loss", "value_loss")
        )
        rejected = (
            health.get("rejected_steps", 0) > 0
            or health.get("skipped_steps", 0) > 0
            or any(r > rule.reject_deviation_ratio for r in ratios.values())
            or fixed["grad_cosine_min"] < rule.reject_grad_cosine
            or loss_relative > rule.reject_loss_relative
        )
        accepted = (
            not rejected
            and all(r <= rule.accept_deviation_ratio for r in ratios.values())
            and fixed["grad_cosine_min"] >= rule.accept_grad_cosine
            and loss_relative <= rule.accept_loss_relative
        )
        found["candidates"][candidate] = {
            "verdict": "accepted"
            if accepted
            else "rejected"
            if rejected
            else "unresolved",
            "deviation_ratio": ratios,
            "loss_relative": loss_relative,
            "grad_cosine_min": fixed["grad_cosine_min"],
        }
    return found


def analyze(
    receipts: dict[str, dict[str, dict[str, Any]]], rule: Rule = RULE
) -> dict[str, Any]:
    """Per-group verdicts and timings, each candidate's overall verdict, and the
    recommended CUDA precision."""
    groups: dict[str, Any] = {}
    for name, arms in receipts.items():
        groups[name] = group_verdict(arms, rule)
        groups[name]["timing"] = {
            arm: _timing(r)
            | {
                "collection_peak_bytes": r.get("collection_peak_bytes"),
                "learner_peak_bytes": r.get("learner_peak_bytes"),
            }
            for arm, r in arms.items()
            if r["status"] == "ok"
        }
        groups[name]["fixed_weights"] = {
            arm: r["fixed_weights"] for arm, r in arms.items() if "fixed_weights" in r
        }
    overall: dict[str, str] = {}
    for candidate in CANDIDATES:
        verdicts = {g["candidates"][candidate]["verdict"] for g in groups.values()}
        overall[candidate] = (
            "rejected"
            if "rejected" in verdicts
            else "accepted"
            if verdicts == {"accepted"}
            else "unresolved"
        )
    speedups: dict[str, float] = {}
    for candidate in CANDIDATES:
        ratios = []
        for name, group in groups.items():
            if not name.endswith("/bucketed"):
                continue
            timing = group["timing"]
            if candidate not in timing:
                ratios.append(0.0)
                continue
            for key in ("learner_step_ms", "collection_large_ms"):
                ratios.append(timing["float32"][key] / timing[candidate][key])
        speedups[candidate] = min(ratios) if ratios else math.nan
    eligible = [
        c
        for c in CANDIDATES
        if overall[c] == "accepted" and speedups[c] >= rule.default_speedup
    ]
    recommendation = max(eligible, key=lambda c: speedups[c]) if eligible else "float32"
    return {
        "groups": groups,
        "overall": overall,
        "bucketed_min_speedup": speedups,
        "recommendation": recommendation,
        "rule": rule.__dict__,
    }


# --------------------------------------------------------------------------- job


def measure(probe: Probe) -> None:
    """The job (`etude.call`): capture, every group's arms, then the verdict."""
    output = Path(os.environ["ETUDE_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    if probe.source_bundle is not None:
        bind_etude_source()
    if probe.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback measurement")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    summary: dict[str, Any] = {
        "host": {
            "device": probe.device,
            "gpu": torch.cuda.get_device_name() if probe.device == "cuda" else None,
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "cpus": os.cpu_count(),
        },
        "probe": probe.model_dump(mode="json", exclude={"source_bundle"}),
    }
    # Captures stay off the output: only receipts and the kept regime are evidence.
    folder = Path(tempfile.mkdtemp(prefix="mtg317-"))
    summary["capture"] = capture(folder, probe)
    _write(output / "regime.json", json.loads((folder / "regime.json").read_text()))
    _write(output / "summary.json", summary)
    receipts: dict[str, dict[str, dict[str, Any]]] = {}
    for size in probe.sizes:
        for bucketed in probe.bucketed:
            name = f"{size}/{'bucketed' if bucketed else 'padded'}"
            receipts[name] = {}
            reference: Outputs | None = None
            for arm in ARMS:
                started = time.perf_counter()
                receipt, outputs = run_arm(
                    folder, probe, size, bucketed, arm, reference
                )
                receipt["seconds"] = time.perf_counter() - started
                if arm == "float32":
                    if receipt["status"] != "ok":
                        raise RuntimeError(f"{name} float32 arm {receipt['status']}")
                    reference = outputs
                receipts[name][arm] = receipt
                _write(output / "trajectories.json", receipts)
                print(
                    f"ARM {name} {arm}: {receipt['status']} {receipt['seconds']:.0f}s "
                    + json.dumps(_timing(receipt) if receipt["status"] == "ok" else {}),
                    flush=True,
                )
            del reference
    summary["verdict"] = analyze(receipts)
    _write(output / "summary.json", summary)
    print(
        "VERDICT "
        + json.dumps(
            {
                k: summary["verdict"][k]
                for k in ("overall", "bucketed_min_speedup", "recommendation")
            }
        ),
        flush=True,
    )


def runs(
    revision: str, hours: float = 1.5, usd: float = 1.5, sizes: str = ""
) -> list[Run]:
    """The one L4 job at `revision`; `sizes` (comma-separated `SIZES` keys)
    replaces the default pair, as MTG-324's w256l8 step run does."""
    chosen = tuple(s for s in sizes.split(",") if s) or Probe().sizes
    unknown = set(chosen) - SIZES.keys()
    if unknown:
        raise ValueError(f"unknown sizes {sorted(unknown)}; known {sorted(SIZES)}")
    probe = Probe(source_bundle=bundle_at(ROOT, revision), sizes=chosen)
    return Job(
        name="mtg317-cuda-precision" + ("" if not sizes else "-" + "-".join(chosen)),
        setup=machine_setup(""),
        command=f"{TOOLCHAIN_ENV}\nuv run --locked python -m etude.call "
        "--function experiments.cuda_precision:measure",
        config=probe.model_dump_json(),
        repo=str(ROOT),
        revision=revision,
        max_hours=hours,
        max_usd=Decimal(str(usd)),
        max_attempts=1,
    ).runs(probe.seed)


# --------------------------------------------------------------------------- charts

_GROUPS = ("w160l8/padded", "w160l8/bucketed", "w384l8/padded", "w384l8/bucketed")
_COLORS: dict[str, str] = {"tf32": "#2a78d6", "bf16": "#eb6834"}
_INK, _MUTED, _GRID, _SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def charts(folder: Path) -> None:
    """The report's two figures from a run's `summary.json` (needs matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    verdict = json.loads((folder / "summary.json").read_text())["verdict"]
    groups = verdict["groups"]
    plt.rcParams.update(
        {
            "font.size": 9,
            "text.color": _INK,
            "axes.edgecolor": _MUTED,
            "axes.labelcolor": _INK,
            "xtick.color": _MUTED,
            "ytick.color": _INK,
            "svg.fonttype": "none",
        }
    )

    def frame(axis: Any) -> None:
        axis.set_facecolor(_SURFACE)
        axis.grid(axis="x", color=_GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.tick_params(axis="y", length=0)

    # Speed: float32 time / candidate time per group, two panels.
    figure, axes = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True)
    figure.patch.set_facecolor(_SURFACE)
    rows = list(range(len(_GROUPS)))
    for axis, (key, title) in zip(
        axes,
        (
            ("learner_step_ms", "Learner step"),
            ("collection_large_ms", "Collection call, 64+ rows"),
        ),
        strict=True,
    ):
        frame(axis)
        axis.axvline(1.0, color=_MUTED, linewidth=1)
        for offset, candidate in ((-0.18, "tf32"), (0.18, "bf16")):
            values = [
                groups[g]["timing"]["float32"][key]
                / groups[g]["timing"][candidate][key]
                for g in _GROUPS
            ]
            axis.barh(
                [r + offset for r in rows],
                [v - 1 for v in values],
                left=1,
                height=0.34,
                color=_COLORS[candidate],
                label=candidate.upper(),
            )
            for r, v in zip(rows, values, strict=True):
                axis.text(
                    max(v, 1) + 0.04, r + offset, f"{v:.2f}×", va="center", fontsize=8
                )
        axis.set_title(title, loc="left", fontsize=9)
        axis.set_xlim(0.5, 3.1)
        axis.set_xlabel("speed relative to float32 (×)")
    axes[0].set_yticks(rows, [g.replace("/", " ") for g in _GROUPS])
    axes[0].invert_yaxis()
    axes[1].legend(frameon=False, loc="lower right")
    figure.tight_layout()
    figure.savefig(folder / "speed.svg")
    plt.close(figure)

    # Equivalence: deviation / float32 reference per group and metric.
    labels = [
        f"{g.replace('/', ' ')} · {m.replace('_', ' ')}"
        for g in _GROUPS
        for m in METRICS
    ]
    figure, axis = plt.subplots(figsize=(7, 4.2))
    figure.patch.set_facecolor(_SURFACE)
    frame(axis)
    axis.axvspan(0, RULE.accept_deviation_ratio, color=_GRID, alpha=0.6, linewidth=0)
    axis.axvline(RULE.reject_deviation_ratio, color=_MUTED, linewidth=1, linestyle="--")
    for offset, candidate in ((-0.15, "tf32"), (0.15, "bf16")):
        ratios = [
            groups[g]["candidates"][candidate]["deviation_ratio"][m]
            for g in _GROUPS
            for m in METRICS
        ]
        axis.scatter(
            ratios,
            [i + offset for i in range(len(labels))],
            s=36,
            color=_COLORS[candidate],
            edgecolors=_SURFACE,
            linewidths=1.5,
            label=candidate.upper(),
            zorder=3,
        )
    axis.set_yticks(range(len(labels)), labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 3.4)
    axis.set_xlabel("largest epoch-mean deviation from float32 ÷ float32's own spread")
    axis.text(0.5, -0.9, "accept ≤ 1", ha="center", color=_MUTED, fontsize=8)
    axis.text(2.05, -0.9, "reject > 2", ha="left", color=_MUTED, fontsize=8)
    axis.legend(frameon=False, loc="lower right")
    figure.tight_layout()
    figure.savefig(folder / "equivalence.svg")
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke", help="every stage on CPU at toy scale")
    smoke.add_argument("--out", type=Path, default=None)
    figures = sub.add_parser("charts", help="the report's figures from summary.json")
    figures.add_argument("folder", type=Path)
    args = parser.parse_args(argv)
    if args.command == "charts":
        charts(args.folder)
        return
    out = args.out or Path(tempfile.mkdtemp(prefix="mtg317-smoke-"))
    os.environ["ETUDE_OUTPUT_DIR"] = str(out)
    measure(SMOKE)
    print(out)


if __name__ == "__main__":
    main()
