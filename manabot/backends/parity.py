"""Check that every installed backend agrees with the PyTorch policy on captured
real inputs.

    uv run --extra mlx python -m manabot.backends.parity CHECKPOINT.pt --capture DIR

`DIR` is a capture from `manabot.backends.torch.capture` containing learner
minibatches (`learner/NNN.npz` with `minibatches.json`) and collection calls
(`collection/NNNNN.npz`). The reference is the PyTorch model in float32 on CPU.
Both sides run the production damped objective with their own learner. The
report compares legal logits, critic values, loss, gradients and the weights
after one clipped Adam step. Each minibatch starts at the checkpoint weights.

`TOLERANCES` sets the gates for each precision. Float32 checks individual
gradient cosines above the rounding-noise floor. Bfloat16 checks the full
gradient cosine and records the per-tensor cosines and Adam difference as
diagnostics. Use `--precision bfloat16` with MLX or Torch MPS/CUDA.

`--json` writes one report per backend. The CLI exits 0 on agreement, 1 on
a failed check and 2 for an invalid input or unavailable execution target.
"""

import argparse
from collections.abc import Iterator, Mapping
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import numpy as np
import torch

from manabot.backends import resolve
import manabot.backends.torch as torch_backend
from manabot.backends.torch import runtime as torch_runtime
from manabot.backends.torch.arrays import on_device
from manabot.backends.torch.checkpoint import read_policy, restore_policy
from manabot.infra.identity import file_sha256
from manabot.policy.checkpoint import (
    ParameterArrays,
    PolicyCheckpoint,
    canonical_parameters,
)
from manabot.policy.prepare import Prepared, reference_distribution
from manabot.policy.protocol import (
    BackendLearner,
    BackendPolicy,
    Precision,
    Settings,
    Targets,
)
from manabot.training.updates.backend import Arrays, Learner, Policy, UpdateBackend
from manabot.training.updates.objective import minibatch_objective

if TYPE_CHECKING:
    from manabot.backends.mlx.policy import MLXActorCritic


class ParityPolicy(BackendPolicy, Policy, Protocol):
    """One policy for both host inference and differentiable parity checks."""

    bucketed_shapes: bool


class ParityLearner(Learner, BackendLearner, Protocol):
    """An update learner whose parameters and gradients can be compared on the host."""


class ParityBackend(UpdateBackend, Protocol):
    """Fresh learner and layout conversions used by numerical comparisons."""

    def load(
        self,
        checkpoint: PolicyCheckpoint,
        *,
        device: str | None = None,
        precision: Precision | None = None,
        bucketed_shapes: bool = False,
        minibatch_rows: int = 1,
        execution_rows: int | None = None,
        bootstrap_rows: int | None = None,
    ) -> tuple[ParityPolicy, ParityLearner]:
        """A policy holding the checkpoint's weights, and a fresh learner on it."""
        ...

    def parameter_layout(self, canonical: ParameterArrays) -> ParameterArrays:
        """Canonical parameters (or deltas: the map is linear) in the
        backend's names and layout, as `BackendLearner.parameters` returns them."""
        ...

    def gradient_layout(self, canonical: ParameterArrays) -> ParameterArrays:
        """Gradients of canonical parameters as gradients of the backend's,
        as `BackendLearner.gradients` returns them."""
        ...


LOGIT_TOLERANCE = 1e-4
LOSS_TOLERANCE = 1e-3
COSINE_TOLERANCE = 0.999
STEP_PARAMETER_TOLERANCE = 1e-6
ZERO_GRADIENT = 1e-6  # of the global gradient norm


@dataclass(frozen=True)
class Tolerances:
    """Numerical acceptance for a compute dtype, not a learning-quality claim."""

    logits: float
    values: float
    loss: float
    cosine: float
    cosine_scope: Literal["per_tensor", "full_gradient"]
    step: float | None


# Half bars cover the retained w384/w768 real-input checks at 140eec28d.
# Where half has no gate (tensor cosines, Adam deltas) the value is still reported.
TOLERANCES: dict[Precision, Tolerances] = {
    "float32": Tolerances(
        LOGIT_TOLERANCE,
        LOGIT_TOLERANCE,
        LOSS_TOLERANCE,
        COSINE_TOLERANCE,
        "per_tensor",
        STEP_PARAMETER_TOLERANCE,
    ),
    "bfloat16": Tolerances(1e-3, 1e-2, 1e-2, 0.99, "full_gradient", None),
}


# Backend name -> the framework it needs.
BACKENDS = {"torch": "torch", "mlx": "mlx"}


def installed_backends() -> list[str]:
    return [
        name
        for name, framework in BACKENDS.items()
        if importlib.util.find_spec(framework) is not None
    ]


def minibatch_loss(
    module: UpdateBackend,
    policy: ParityPolicy,
    learner: ParityLearner,
    batch: Prepared,
    targets: Targets,
    settings: Settings,
) -> Any:
    """The damped update's loss on one captured minibatch, as `module`'s
    array, ready for `learner.step` or `learner.gradients`.

    The same `minibatch_objective` the update trains, over the backend's own
    `rl` functions, with the checkpoint's configured critic.
    """
    rl, to_device = module.rl, module.arrays.from_numpy
    device = learner.device
    valid = rl.legal_mask(to_device(batch.legal, device))
    constants = {
        "actions": to_device(targets.actions.astype(np.int64), device),
        "advantages": to_device(targets.advantages, device),
        "value_targets": to_device(targets.value_targets, device),
        "behavior_logs": to_device(targets.behavior_logs, device),
        "reference": to_device(targets.reference, device),
        "actor_mask": to_device(targets.actor_mask.astype(bool), device),
    }

    if isinstance(policy, torch_backend.policy.ActorCritic):
        policy.train()
        device_batch = on_device(batch, device)
    else:
        device_batch = batch

    def objective() -> tuple[Any, ...]:
        with learner.forward_precision():
            logits, values = policy.forward_distribution(device_batch)
        return minibatch_objective(
            rl,
            module.arrays.float32(logits),
            module.arrays.float32(values),
            valid,
            **constants,
            categorical=policy.config.value_kind == "categorical_wdl",
            clip=settings.clip,
            collection_kl=settings.collection_kl,
            tau=settings.tau,
            reference_forward_kl=settings.reference_forward_kl,
        )

    return learner.loss(objective)[0]


@dataclass
class Report:
    backend: str = ""
    requested_device: str = ""
    requested_bucketed_shapes: bool = False
    actual_device: str | None = None
    observed_forward_dtype: str | None = None
    bucketed_shapes: bool | None = None
    threads: int = 0
    capture: str = ""
    checkpoint_sha256: str = ""
    collection_indices: list[int] = field(default_factory=list)
    execution_error: str | None = None
    precision: Precision = "float32"
    gradient_cosine: float = 1.0
    forward_logits: float = 0.0
    forward_logits_relative: float = 0.0
    forward_values: float = 0.0
    loss_relative: float = 0.0
    gradient_min_cosine: float = 1.0
    gradient_worst: str = ""
    gradient_tensors: int = 0
    gradient_both_zero: int = 0
    collection_logits: float = 0.0
    collection_values: float = 0.0
    step_parameter_max_abs: float = 0.0
    step_update_min_cosine: float = 1.0
    step_checked: bool = False
    step_reference_norm: float = 0.0
    step_backend_norm: float = 0.0
    step_max_grad_norm: float = 0.0
    minibatches: list[dict[str, Any]] = field(default_factory=list)

    @property
    def placement_matches(self) -> bool:
        device = "mps" if self.requested_device == "gpu" else self.requested_device
        return (
            self.actual_device == device
            and self.bucketed_shapes == self.requested_bucketed_shapes
        )

    @property
    def minibatch_indices(self) -> list[int]:
        return [row["index"] for row in self.minibatches]

    @property
    def minibatches_checked(self) -> int:
        return len(self.minibatches)

    @property
    def collection_checked(self) -> int:
        return len(self.collection_indices)

    def checks(self) -> list[tuple[str, str, bool]]:
        """(component, measured against its dtype's bar, passed)."""
        tolerance = TOLERANCES[self.precision]
        finite_step = self.step_checked and bool(
            np.isfinite(
                [
                    self.step_parameter_max_abs,
                    self.step_reference_norm,
                    self.step_backend_norm,
                ]
            ).all()
        )
        checks = [
            (
                "execution",
                self.execution_error
                or f"{self.actual_device} {self.observed_forward_dtype}",
                self.execution_error is None
                and self.placement_matches
                and self.observed_forward_dtype == self.precision,
            ),
            (
                "learner samples",
                str(self.minibatches_checked),
                self.minibatches_checked > 0,
            ),
            (
                "collection samples",
                str(self.collection_checked),
                self.collection_checked > 0,
            ),
            (
                "clipped Adam weights"
                if tolerance.step is not None
                else "finite clipped Adam step",
                f"max |Δ| {self.step_parameter_max_abs:.1e} "
                + (
                    f"(tol {tolerance.step:g})"
                    if tolerance.step is not None
                    else "(diagnostic; no float32 step bound)"
                ),
                finite_step
                and (
                    tolerance.step is None
                    or self.step_parameter_max_abs <= tolerance.step
                ),
            ),
        ]
        for name, value, limit in (
            ("forward logits", self.forward_logits, tolerance.logits),
            ("forward values", self.forward_values, tolerance.values),
            ("collection logits", self.collection_logits, tolerance.logits),
            ("collection values", self.collection_values, tolerance.values),
            ("learner loss", self.loss_relative, tolerance.loss),
        ):
            checks.append((name, f"Δ {value:.3g} (tol {limit:g})", value <= limit))
        if tolerance.cosine_scope == "per_tensor":
            checks.append(
                (
                    f"gradient {self.gradient_worst}",
                    f"min cosine {self.gradient_min_cosine:.6f} over "
                    f"{self.gradient_tensors} tensors (tol {tolerance.cosine:g})",
                    self.gradient_min_cosine >= tolerance.cosine,
                )
            )
        else:
            checks.append(
                (
                    "full gradient",
                    f"cosine {self.gradient_cosine:.6f} "
                    f"(tol {tolerance.cosine:g}); tensor minimum "
                    f"{self.gradient_min_cosine:.6f} ({self.gradient_worst}, diagnostic)",
                    self.gradient_cosine >= tolerance.cosine,
                )
            )
        return checks

    def failures(self) -> list[str]:
        return [name for name, _, passed in self.checks() if not passed]


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


def minibatches(
    capture: Path, count: int | None
) -> Iterator[tuple[dict[str, np.ndarray], Targets, Settings, dict[str, Any]]]:
    """Captured learner minibatches as (observation, targets, settings, meta)."""
    rows = json.loads((capture / "minibatches.json").read_text())
    for row in rows[:count]:
        arrays = read_npz(capture / "learner" / f"{row['index']:03d}.npz")
        obs = {k[4:]: v for k, v in arrays.items() if k.startswith("obs/")}
        targets = Targets(
            actions=arrays["actions"],
            advantages=arrays["advantages"],
            value_targets=arrays["value_targets"],
            actor_mask=arrays["actor_mask"],
            behavior_logs=arrays["behavior_logs"],
            reference=reference_distribution(obs, row["reference"]),
        )
        settings = Settings(
            learning_rate=row["learning_rate"],
            clip=row["clip"],
            collection_kl=row["collection_kl"],
            tau=row["tau"],
            reference_forward_kl=row.get("reference_forward_kl", 0.0),
            max_grad_norm=row["max_grad_norm"],
        )
        yield obs, targets, settings, row


def _cosine(a: np.ndarray, b: np.ndarray, floor: float = 0.0) -> float | None:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= floor and nb <= floor:
        return None
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


@dataclass(frozen=True)
class GradientAgreement:
    """Cosines between a backend's gradients and the reference's."""

    cosine: float  # all parameters as one vector
    tensors: dict[str, float]  # each tensor above the zero-gradient floor
    both_zero: int  # tensors below the floor on both sides, never compared

    @property
    def worst(self) -> str:
        return min(self.tensors, key=self.tensors.__getitem__) if self.tensors else ""

    @property
    def min_cosine(self) -> float:
        return self.tensors.get(self.worst, 1.0)


def gradient_agreement(
    actual: Mapping[str, np.ndarray], expected: Mapping[str, np.ndarray]
) -> GradientAgreement:
    """Compare gradients held in one parameter layout.

    A tensor whose norm is below `ZERO_GRADIENT` of the reference's global
    norm on both sides holds rounding noise and has no per-tensor cosine; the
    full-vector cosine still includes it.
    """
    if actual.keys() != expected.keys():
        raise ValueError("backend gradient parameter sets differ")
    flat = {
        name: (
            actual[name].astype(np.float64).ravel(),
            wanted.astype(np.float64).ravel(),
        )
        for name, wanted in expected.items()
    }
    for name, (a, b) in flat.items():
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ParityMismatch(f"invalid gradient: {name}")
    dot = sum(float(a @ b) for a, b in flat.values())
    actual_square = sum(float(a @ a) for a, _ in flat.values())
    expected_square = sum(float(b @ b) for _, b in flat.values())
    floor = ZERO_GRADIENT * float(np.sqrt(expected_square))
    cosines = {name: _cosine(a, b, floor) for name, (a, b) in flat.items()}
    return GradientAgreement(
        float(dot / np.sqrt(actual_square * expected_square))
        if actual_square * expected_square
        else float(actual_square == expected_square == 0),
        {name: cosine for name, cosine in cosines.items() if cosine is not None},
        sum(cosine is None for cosine in cosines.values()),
    )


def _legal_difference(
    logits: np.ndarray, torch_logits: np.ndarray, legal: np.ndarray
) -> tuple[float, float]:
    if not np.isfinite(logits).all() or not np.isfinite(torch_logits).all():
        raise ParityMismatch("nonfinite logits")
    difference = np.abs(logits - torch_logits)[legal]
    scale = np.abs(torch_logits[legal]).max()
    return float(difference.max()), float(difference.max() / scale) if scale else 0.0


class ParityMismatch(ValueError):
    """A measured candidate cannot supply the requested numerical comparison."""


def evaluate(
    arrays: Arrays,
    policy: ParityPolicy,
    learner: ParityLearner,
    batch: Prepared,
    report: Report,
) -> tuple[np.ndarray, np.ndarray]:
    """Read the actual critic dtype before the host-facing float32 conversion."""
    with torch.no_grad(), learner.forward_precision():
        if isinstance(policy, torch_backend.policy.ActorCritic):
            logits, values = policy.forward_distribution(
                on_device(batch, learner.device)
            )
        else:
            mlx_policy = cast("MLXActorCritic", policy)
            logits, values = mlx_policy.forward_heads(batch)
    dtype = str(values.dtype).rsplit(".", 1)[-1]
    report.observed_forward_dtype = dtype
    if dtype != report.precision:
        raise ParityMismatch(f"forward dtype {dtype} differs from {report.precision}")
    return arrays.to_numpy(logits), arrays.to_numpy(values).reshape(-1)


def check(
    checkpoint: Path,
    capture: Path,
    minibatch_count: int | None = 8,
    collection_count: int = 8,
    *,
    backend_name: str,
    precision: Precision = "float32",
    device: str | None = None,
    bucketed_shapes: bool = False,
    threads: int | None = None,
) -> Report:
    """Compare this execution target with the padded CPU float32 reference."""
    port = resolve(backend_name)
    if device is None:
        device = "cpu" if backend_name == "torch" else "mps"
    if backend_name == "torch" and device not in {"cpu", "mps", "cuda"}:
        raise ValueError(f"unsupported Torch parity device: {device!r}")
    if (
        backend_name == "torch"
        and device == "mps"
        and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
    ):
        raise ValueError("Torch MPS parity requires operator fallback disabled")
    if precision not in TOLERANCES:
        raise ValueError(f"unsupported parity precision: {precision!r}")
    if threads is None:
        threads = torch.get_num_threads()
    if (
        threads < 1
        or collection_count < 0
        or (minibatch_count is not None and minibatch_count < 0)
    ):
        raise ValueError("threads must be positive and sample counts nonnegative")
    report = Report(
        backend=backend_name,
        requested_device=device,
        requested_bucketed_shapes=bucketed_shapes,
        precision=precision,
        capture=str(capture),
        checkpoint_sha256=file_sha256(checkpoint),
        threads=threads,
    )
    with ExitStack() as scope:
        if backend_name != "torch":
            scope.enter_context(torch_runtime.scoped(threads, "float32", "cpu"))
        scope.enter_context(
            port.runtime.scoped(
                threads, "bf16" if precision == "bfloat16" else precision, device
            )
        )
        weights = read_policy(str(checkpoint))
        try:
            compare(port, weights, capture, minibatch_count, collection_count, report)
        except (
            ParityMismatch,
            torch_backend.rl.NumericalError,
            port.rl.NumericalError,
        ) as error:
            report.execution_error = str(error)
    return report


def compare(
    port: ParityBackend,
    weights: PolicyCheckpoint,
    capture: Path,
    minibatch_count: int | None,
    collection_count: int,
    report: Report,
) -> None:
    def load_pair() -> tuple[
        torch_backend.learner.Learner, ParityPolicy, ParityLearner
    ]:
        reference = torch_backend.learner.Learner(
            restore_policy(weights), "cpu", minibatch_rows=1
        )
        policy, learner = port.load(
            weights,
            device=report.requested_device,
            precision=report.precision,
            bucketed_shapes=report.requested_bucketed_shapes,
        )
        report.actual_device = policy.device_type
        report.bucketed_shapes = policy.bucketed_shapes
        if not report.placement_matches:
            raise ParityMismatch(
                f"loaded {policy.device_type}, bucketed={policy.bucketed_shapes}; "
                f"requested {report.requested_device}, bucketed={report.requested_bucketed_shapes}"
            )
        return reference, policy, learner

    reference_learner, policy, learner = load_pair()
    for index, (obs, targets, settings, meta) in enumerate(
        minibatches(capture, minibatch_count)
    ):
        reference_policy = reference_learner.agent
        reference_batch = reference_policy.prepare(obs)
        valid = reference_batch.legal
        with torch.no_grad():
            logits, values = reference_policy(on_device(reference_batch, "cpu"))
        loss = minibatch_loss(
            torch_backend,
            reference_policy,
            reference_learner,
            reference_batch,
            targets,
            settings,
        )
        reference_loss = float(loss.detach())
        expected = port.gradient_layout(
            canonical_parameters(reference_learner.gradients(loss))
        )

        batch = policy.prepare(obs)
        port_logits, port_values = evaluate(port.arrays, policy, learner, batch, report)
        port_loss = minibatch_loss(port, policy, learner, batch, targets, settings)
        loss_value = float(port.arrays.to_numpy(port_loss))
        grads = learner.gradients(port_loss)
        absolute, relative = _legal_difference(port_logits, logits.numpy(), valid)
        value_difference = float(np.abs(port_values - values.numpy().reshape(-1)).max())
        loss_relative = abs(loss_value - reference_loss) / max(
            abs(reference_loss), 1e-12
        )
        agreement = gradient_agreement(grads, expected)
        if not np.isfinite(
            [
                absolute,
                relative,
                value_difference,
                loss_relative,
                agreement.cosine,
                agreement.min_cosine,
            ]
        ).all():
            raise ParityMismatch(
                f"nonfinite learner comparison on minibatch {meta['index']}"
            )
        row = {
            "index": meta["index"],
            "B": int(len(targets.actions)),
            "logits": absolute,
            "logits_relative": relative,
            "values": value_difference,
            "loss_torch": reference_loss,
            "loss_backend": loss_value,
            "loss_relative": loss_relative,
            "gradient_cosine": agreement.cosine,
            "gradient_min_cosine": agreement.min_cosine,
            "gradient_worst": agreement.worst,
        }
        report.minibatches.append(row)
        report.forward_logits = max(report.forward_logits, absolute)
        report.forward_logits_relative = max(report.forward_logits_relative, relative)
        report.forward_values = max(report.forward_values, value_difference)
        report.loss_relative = max(report.loss_relative, loss_relative)
        report.gradient_tensors = len(grads)
        report.gradient_both_zero = max(report.gradient_both_zero, agreement.both_zero)
        report.gradient_cosine = min(report.gradient_cosine, agreement.cosine)
        if agreement.min_cosine < report.gradient_min_cosine:
            report.gradient_min_cosine = agreement.min_cosine
            report.gradient_worst = agreement.worst
        if index == 0:
            reference_loss_again = minibatch_loss(
                torch_backend,
                reference_policy,
                reference_learner,
                reference_batch,
                targets,
                settings,
            )
            _compare_step(
                reference_learner,
                reference_loss_again,
                port,
                learner,
                minibatch_loss(port, policy, learner, batch, targets, settings),
                settings,
                report,
            )
            # Release the stepped optimizers before restoring both checkpoint models.
            del reference_learner, reference_policy, learner, policy
            reference_learner, policy, learner = load_pair()
        print(
            f"{report.backend} minibatch {meta['index']:3d}  B {row['B']:3d}  logits {absolute:.1e}  "
            f"values {value_difference:.1e}  loss {loss_relative:.1e}  "
            f"min cosine {agreement.min_cosine:.6f} ({agreement.worst})",
            flush=True,
        )
    _collection(
        port.arrays,
        reference_learner.agent,
        policy,
        learner,
        capture,
        collection_count,
        report,
    )


def _compare_step(
    reference: BackendLearner,
    reference_loss: Any,
    port: ParityBackend,
    learner: BackendLearner,
    loss: Any,
    settings: Settings,
    report: Report,
) -> None:
    """One clipped Adam step in both frameworks from the same weights.

    All parameters count, including tensors with near-zero gradients. Cosine
    is diagnostic only; float32 gates the absolute difference, while half
    reports it beside its dtype-specific forward, loss and gradient gates.
    """
    before = reference.parameters()
    reference.set_learning_rate(settings.learning_rate)
    report.step_reference_norm = float(
        reference.step(reference_loss, settings.max_grad_norm)
    )
    now = reference.parameters()
    initial = learner.parameters()
    learner.set_learning_rate(settings.learning_rate)
    report.step_backend_norm = float(learner.step(loss, settings.max_grad_norm))
    report.step_max_grad_norm = settings.max_grad_norm
    after = learner.parameters()
    expected_after = port.parameter_layout(canonical_parameters(now))
    expected_delta = port.parameter_layout(
        canonical_parameters({k: v - before[k] for k, v in now.items()})
    )
    worst = 1.0
    largest = 0.0
    for name, value in after.items():
        delta = float(np.abs(value - expected_after[name]).max())
        if not np.isfinite(delta):
            raise ParityMismatch(f"nonfinite post-step parameter: {name}")
        largest = max(largest, delta)
        cosine = _cosine(value - initial[name], expected_delta[name])
        if cosine is not None:
            if not np.isfinite(cosine):
                raise ParityMismatch(f"nonfinite step cosine: {name}")
            worst = min(worst, cosine)
    report.step_parameter_max_abs = largest
    report.step_update_min_cosine = worst
    report.step_checked = True


def _collection(
    arrays: Arrays,
    agent: torch_backend.policy.ActorCritic,
    policy: ParityPolicy,
    learner: ParityLearner,
    capture: Path,
    count: int,
    report: Report,
) -> None:
    """Inference parity on distinct spread-out captured collection calls."""
    calls = sorted((capture / "collection").glob("*.npz"))
    if not calls or not count:
        return
    agent.eval()
    policy.eval()
    with torch.no_grad(), learner.forward_precision():
        try:
            policy.cache_programs()
            for index in np.linspace(
                0, len(calls) - 1, min(count, len(calls)), dtype=int
            ):
                path = calls[index]
                obs = read_npz(path)
                logits, values = agent.evaluate(agent.prepare(obs, inference=True))
                port_logits, port_values = evaluate(
                    arrays, policy, learner, policy.prepare(obs, inference=True), report
                )
                absolute, _ = _legal_difference(
                    port_logits, logits, obs["actions_valid"] > 0
                )
                delta = float(np.abs(port_values - values.reshape(-1)).max())
                if not np.isfinite(delta):
                    raise ParityMismatch(
                        f"nonfinite collection value on call {path.stem}"
                    )
                report.collection_logits = max(report.collection_logits, absolute)
                report.collection_values = max(report.collection_values, delta)
                report.collection_indices.append(int(path.stem))
        finally:
            policy.clear_cache()


def _print(name: str, report: Report) -> None:
    for component, measured, passed in report.checks():
        print(f"{name:<6} {component:<40} {measured}  {'ok' if passed else 'FAIL'}")
    print(
        f"{name:<6} {'one step':<40} params max |Δ| "
        f"{report.step_parameter_max_abs:.1e}, update min cosine "
        f"{report.step_update_min_cosine:.6f} (cosine diagnostic only)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--minibatches", type=int, default=8)
    parser.add_argument("--collection", type=int, default=8)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--backend", choices=tuple(BACKENDS))
    parser.add_argument("--precision", choices=tuple(TOLERANCES), default="float32")
    parser.add_argument("--device", choices=("cpu", "mps", "gpu", "cuda"))
    parser.add_argument("--bucketed", action="store_true")
    parser.add_argument("--threads", type=int)
    args = parser.parse_args(argv)
    installed = installed_backends()
    names = [args.backend] if args.backend else installed
    if args.backend and args.backend not in installed:
        parser.error("the selected backend's framework is not installed")
    if not names:
        parser.error("no framework backend is installed (" + ", ".join(BACKENDS) + ")")
    reports: dict[str, Report] = {}
    try:
        for name in names:
            reports[name] = check(
                args.checkpoint,
                args.capture,
                args.minibatches,
                args.collection,
                backend_name=name,
                precision=args.precision,
                device=args.device,
                bucketed_shapes=args.bucketed,
                threads=args.threads,
            )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))  # exits 2; 1 means the models disagree
    failures = {name: report.failures() for name, report in reports.items()}
    for name, report in reports.items():
        _print(name, report)
    if args.json is not None:
        args.json.write_text(
            json.dumps(
                {
                    name: asdict(report)
                    | {
                        "minibatch_indices": report.minibatch_indices,
                        "minibatches_checked": report.minibatches_checked,
                        "collection_checked": report.collection_checked,
                        "tolerances": asdict(TOLERANCES[report.precision]),
                        "failures": failures[name],
                    }
                    for name, report in reports.items()
                },
                indent=2,
            )
        )
    failing = [f"{name}: " + ", ".join(f) for name, f in failures.items() if f]
    if failing:
        print("parity: FAIL (" + "; ".join(failing) + ")")
        return 1
    print("parity: pass (" + ", ".join(names) + ")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
