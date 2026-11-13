"""Check that every installed backend agrees with the PyTorch policy on captured
real inputs.

    uv run --extra mlx python -m manabot.backends.parity CHECKPOINT.pt --capture DIR

`DIR` is a capture written by `experiments/mlx_pilot.py capture`: learner
minibatches (`learner/NNN.npz` with `minibatches.json`) and collection calls
(`collection/NNNNN.npz`). The reference is the PyTorch model in float32 on CPU.
Each backend whose framework is installed (`BACKENDS`) loads the same weights
through the shared container's canonical names and is compared in its own
parameter layout (`manabot.policy.protocol.Backend`). Both sides run the one
loss the damped update trains (`minibatch_objective`), each over its own
backend's `rl` functions and learner (`minibatch_loss`). For each minibatch,
at the checkpoint's weights:

- forward logits on legal actions and values: max absolute difference <= 1e-4;
- damped policy loss: relative difference <= 0.1%;
- every parameter's gradient: cosine similarity >= 0.999. A tensor whose
  gradient norm is below 1e-6 of the global gradient norm in both frameworks
  counts as zero: the policy-score bias and every attention key bias have
  analytically zero gradients (softmax is invariant to a shift shared by all
  its inputs), so both frameworks return rounding noise whose cosine is
  meaningless.

Collection calls compare inference logits and values with the same 1e-4 bar.
The first minibatch also takes one clipped Adam step in both frameworks:
every parameter must agree within 1e-6 absolute, including near-zero-gradient
tensors. This is 0.4% of the default learning rate (2.5e-4), tight enough to
reject a missing or doubled update. The command prints one line per check and
backend, writes
`--json` (one report per backend) when given, and exits 1 naming each failing
backend's worst components if any check fails; it exits 2 when the checkpoint
or capture is missing, the checkpoint is refused or no backend is installed.

Use `--backend mlx --precision bfloat16` (or `float16`) for explicit half
precision. Half uses its measured logit/value, loss and full-gradient cosine
bars in `TOLERANCES`; the minimum per-tensor cosine and one-step difference
remain diagnostics. Float32 retains all the stricter checks above. Passing half's
numerical gates does not establish equivalent optimization or playing strength.
"""

import argparse
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Literal, cast

import numpy as np
import torch

import manabot.backends.torch as torch_backend
from manabot.backends.torch.checkpoint import load_checkpoint
from manabot.model.admission import admit_observation
from manabot.policy.checkpoint import (
    canonical_parameters,
    from_torch_state,
)
from manabot.policy.prepare import Prepared, reference_distribution
from manabot.policy.protocol import (
    Backend,
    BackendLearner,
    BackendPolicy,
    Precision,
    Settings,
    Targets,
)
from manabot.training.updates.backend import Backend as UpdateBackend
from manabot.training.updates.damped_policy import minibatch_objective

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
    "float16": Tolerances(1e-4, 2e-3, 2e-3, 0.999, "full_gradient", None),
}


# Backend name -> the framework it needs; each provides
# `manabot.backends.<name>.checkpoint.BACKEND`.
BACKENDS = {"torch": "torch", "mlx": "mlx"}


def installed_backends() -> list[str]:
    return [
        name
        for name, framework in BACKENDS.items()
        if importlib.util.find_spec(framework) is not None
    ]


def backend_module(name: str) -> ModuleType:
    """`manabot.backends.<name>`, the module an update takes as its backend."""
    return importlib.import_module(f"manabot.backends.{name}")


def backend(name: str, *, precision: Precision = "float32") -> Backend:
    """Select explicit MLX compute precision; Torch remains the float32 reference.

    Parity always names its precision, so MLX's width default never selects
    the tolerances a report is held to."""
    module = backend_module(name)
    if name == "mlx":
        return cast(Backend, module.checkpoint.MLXBackend(precision=precision))
    if precision != "float32":
        raise ValueError("half-precision parity is supported only for the MLX backend")
    return cast(Backend, module.checkpoint.BACKEND)


def minibatch_loss(
    module: UpdateBackend,
    policy: BackendPolicy,
    learner: BackendLearner,
    batch: Prepared,
    targets: Targets,
    settings: Settings,
) -> Any:
    """The damped update's loss on one captured minibatch, as `module`'s
    array, ready for `learner.step` or `learner.gradients`.

    The same `minibatch_objective` the update trains, over the backend's own
    `rl` functions, with the scalar critic every backend supports.
    """
    rl, to_device = module.rl, module.arrays.from_numpy
    device = getattr(learner, "device", "cpu")
    valid = rl.legal_mask(to_device(batch.legal, device))
    constants = {
        "actions": to_device(targets.actions.astype(np.int64), device),
        "advantages": to_device(targets.advantages, device),
        "value_targets": to_device(targets.value_targets, device),
        "behavior_logs": to_device(targets.behavior_logs, device),
        "reference": to_device(targets.reference, device),
        "actor_mask": to_device(targets.actor_mask.astype(bool), device),
    }

    def objective() -> tuple[Any, ...]:
        logits, values = policy.forward_distribution(batch)
        return minibatch_objective(
            rl,
            logits,
            values,
            valid,
            **constants,
            categorical=False,
            clip=settings.clip,
            collection_kl=settings.collection_kl,
            tau=settings.tau,
        )

    return learner.loss(objective)[0]


@dataclass
class Report:
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
            raise ValueError(f"invalid gradient: {name}")
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
    difference = np.abs(logits - torch_logits)[legal]
    scale = np.abs(torch_logits[legal]).max()
    return float(difference.max()), float(difference.max() / scale) if scale else 0.0


def check(
    checkpoint: Path,
    capture: Path,
    minibatch_count: int | None = 8,
    collection_count: int = 8,
    *,
    backend_name: str,
    precision: Precision = "float32",
) -> Report:
    """Compare one backend against the PyTorch reference."""
    torch.set_num_threads(8)
    agent, space = load_checkpoint(str(checkpoint))
    state = {k: v.detach().numpy().copy() for k, v in agent.state_dict().items()}
    weights = from_torch_state(agent.config, space.encoder.config, state)
    module = backend_module(backend_name)
    port = backend(backend_name, precision=precision)
    policy, learner = port.load(weights)
    # The reference: the same loss over the torch backend, on the loaded agent.
    reference_policy = torch_backend.checkpoint.TorchPolicy(agent, weights)
    reference_learner = torch_backend.learner.Learner(agent, "cpu", minibatch_rows=1)
    report = Report(precision=precision)
    for index, (obs, targets, settings, meta) in enumerate(
        minibatches(capture, minibatch_count)
    ):
        reference_batch = reference_policy.prepare(obs)
        valid = reference_batch.legal
        with torch.no_grad():
            logits, values = reference_policy.forward_distribution(reference_batch)
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
        port_logits, port_values = policy.evaluate(batch)
        port_loss = minibatch_loss(module, policy, learner, batch, targets, settings)
        loss_value = float(module.arrays.to_numpy(port_loss))
        grads = learner.gradients(port_loss)
        absolute, relative = _legal_difference(port_logits, logits.numpy(), valid)
        value_difference = float(np.abs(port_values - values.numpy().reshape(-1)).max())
        loss_relative = abs(loss_value - reference_loss) / max(
            abs(reference_loss), 1e-12
        )
        agreement = gradient_agreement(grads, expected)
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
                minibatch_loss(module, policy, learner, batch, targets, settings),
                settings,
                report,
            )
            policy, learner = port.load(weights)
            agent.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})
            reference_learner = torch_backend.learner.Learner(
                agent, "cpu", minibatch_rows=1
            )
        print(
            f"{backend_name} minibatch {meta['index']:3d}  B {row['B']:3d}  logits {absolute:.1e}  "
            f"values {value_difference:.1e}  loss {loss_relative:.1e}  "
            f"min cosine {agreement.min_cosine:.6f} ({agreement.worst})",
            flush=True,
        )
    _collection(agent, policy, capture, collection_count, report)
    return report


def _compare_step(
    reference: BackendLearner,
    reference_loss: Any,
    port: Backend,
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
        largest = max(largest, float(np.abs(value - expected_after[name]).max()))
        cosine = _cosine(value - initial[name], expected_delta[name])
        if cosine is not None:
            worst = min(worst, cosine)
    report.step_parameter_max_abs = largest
    report.step_update_min_cosine = worst
    report.step_checked = True


def _collection(
    agent: torch.nn.Module,
    policy: BackendPolicy,
    capture: Path,
    count: int,
    report: Report,
) -> None:
    """Inference parity on spread-out captured collection calls."""
    agent.eval()
    calls = sorted((capture / "collection").glob("*.npz"))
    if not calls:
        return
    policy.cache_programs()
    for path in [calls[i] for i in np.linspace(0, len(calls) - 1, count).astype(int)]:
        obs = read_npz(path)
        with torch.no_grad():
            logits, values = agent(
                admit_observation(
                    agent, {k: torch.from_numpy(v) for k, v in obs.items()}
                )
            )
        port_logits, port_values = policy.evaluate(policy.prepare(obs, inference=True))
        absolute, _ = _legal_difference(
            port_logits, logits.numpy(), obs["actions_valid"] > 0
        )
        report.collection_logits = max(report.collection_logits, absolute)
        report.collection_values = max(
            report.collection_values,
            float(np.abs(port_values.reshape(-1) - values.numpy().reshape(-1)).max()),
        )
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
    args = parser.parse_args(argv)
    if args.precision != "float32" and args.backend != "mlx":
        parser.error("half precision requires --backend mlx; Torch stays float32")
    names = [args.backend] if args.backend else installed_backends()
    if any(name not in installed_backends() for name in names):
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
