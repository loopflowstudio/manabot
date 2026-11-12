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
- PPO loss: relative difference <= 0.1%;
- every parameter's gradient: cosine similarity >= 0.999. A tensor whose
  gradient norm is below 1e-6 of the global gradient norm in both frameworks
  counts as zero: the policy-score bias and every attention key bias have
  analytically zero gradients (softmax is invariant to a shift shared by all
  its inputs), so both frameworks return rounding noise whose cosine is
  meaningless.

Collection calls compare inference logits and values with the same 1e-4 bar.
The first minibatch also takes one clipped Adam step in both frameworks and
reports how far the parameters then differ; that comparison is informative and
not part of the bar. The command prints one line per check and backend, writes
`--json` (one report per backend) when given, and exits 1 naming each failing
backend's worst components if any check fails; it exits 2 when the checkpoint
or capture is missing, the checkpoint is refused or no backend is installed.
"""

import argparse
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, cast

import numpy as np
import torch

import manabot.backends.torch as torch_backend
from manabot.backends.torch.checkpoint import load_checkpoint
from manabot.model.admission import admit_observation
from manabot.policy.checkpoint import (
    UnsupportedPolicy,
    canonical_parameters,
    from_torch_state,
)
from manabot.policy.prepare import Prepared, reference_distribution
from manabot.policy.protocol import (
    Backend,
    BackendLearner,
    BackendPolicy,
    Settings,
    Targets,
)
from manabot.training.updates.damped_policy import minibatch_objective

LOGIT_TOLERANCE = 1e-4
LOSS_TOLERANCE = 1e-3
COSINE_TOLERANCE = 0.999
ZERO_GRADIENT = 1e-6  # of the global gradient norm

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


def backend(name: str) -> Backend:
    return cast(Backend, backend_module(name).checkpoint.BACKEND)


def minibatch_loss(
    module: ModuleType,
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
    minibatches: list[dict[str, Any]] = field(default_factory=list)

    def checks(self) -> list[tuple[str, str, bool]]:
        """(component, measured against its bar, passed) for every gated check."""
        return [
            (
                "forward logits",
                f"max |Δ| {self.forward_logits:.1e} (tol 1e-4)",
                self.forward_logits <= LOGIT_TOLERANCE,
            ),
            (
                "forward values",
                f"max |Δ| {self.forward_values:.1e} (tol 1e-4)",
                self.forward_values <= LOGIT_TOLERANCE,
            ),
            (
                "collection logits",
                f"max |Δ| {self.collection_logits:.1e} (tol 1e-4)",
                self.collection_logits <= LOGIT_TOLERANCE,
            ),
            (
                "collection values",
                f"max |Δ| {self.collection_values:.1e} (tol 1e-4)",
                self.collection_values <= LOGIT_TOLERANCE,
            ),
            (
                "learner loss",
                f"rel Δ {100 * self.loss_relative:.4f}% (tol 0.1%)",
                self.loss_relative <= LOSS_TOLERANCE,
            ),
            (
                f"gradient {self.gradient_worst}",
                f"min cosine {self.gradient_min_cosine:.6f} over "
                f"{self.gradient_tensors} tensors (tol 0.999)",
                self.gradient_min_cosine >= COSINE_TOLERANCE,
            ),
        ]

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
) -> Report:
    """Compare one backend against the PyTorch reference."""
    torch.set_num_threads(8)
    agent, space = load_checkpoint(str(checkpoint))
    state = {k: v.detach().numpy().copy() for k, v in agent.state_dict().items()}
    weights = from_torch_state(agent.config, space.encoder.config, state)
    module = backend_module(backend_name)
    port = backend(backend_name)
    policy, learner = port.load(weights)
    # The reference: the same loss over the torch backend, on the loaded agent.
    reference_policy = torch_backend.checkpoint.TorchPolicy(agent, weights)
    reference_learner = torch_backend.learner.Learner(agent, "cpu", minibatch_rows=1)
    report = Report()
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
        loss_value = float(port_loss)
        grads = learner.gradients(port_loss)
        absolute, relative = _legal_difference(port_logits, logits.numpy(), valid)
        value_difference = float(np.abs(port_values - values.numpy().reshape(-1)).max())
        loss_relative = abs(loss_value - reference_loss) / max(
            abs(reference_loss), 1e-12
        )
        cosines: dict[str, float] = {}
        zero: set[str] = set()
        both_zero = 0
        floor = ZERO_GRADIENT * float(
            np.sqrt(
                sum(np.square(g.astype(np.float64)).sum() for g in expected.values())
            )
        )
        for name, grad in grads.items():
            cosine = _cosine(grad, expected[name], floor)
            if cosine is None:
                both_zero += 1
                zero.add(name)
            else:
                cosines[name] = cosine
        worst = min(cosines, key=cosines.__getitem__)
        row = {
            "index": meta["index"],
            "B": int(len(targets.actions)),
            "logits": absolute,
            "logits_relative": relative,
            "values": value_difference,
            "loss_torch": reference_loss,
            "loss_backend": loss_value,
            "loss_relative": loss_relative,
            "gradient_min_cosine": cosines[worst],
            "gradient_worst": worst,
        }
        report.minibatches.append(row)
        report.forward_logits = max(report.forward_logits, absolute)
        report.forward_logits_relative = max(report.forward_logits_relative, relative)
        report.forward_values = max(report.forward_values, value_difference)
        report.loss_relative = max(report.loss_relative, loss_relative)
        report.gradient_tensors = len(cosines) + both_zero
        report.gradient_both_zero = max(report.gradient_both_zero, both_zero)
        if cosines[worst] < report.gradient_min_cosine:
            report.gradient_min_cosine = cosines[worst]
            report.gradient_worst = worst
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
                zero,
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
            f"min cosine {cosines[worst]:.6f} ({worst})",
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
    zero: set[str],
    report: Report,
) -> None:
    """One clipped Adam step in both frameworks from the same weights.

    Tensors with structurally zero gradients (`zero`) are skipped: Adam's
    first step moves their rounding noise by +-lr in either direction.
    """
    before = reference.parameters()
    reference.set_learning_rate(settings.learning_rate)
    reference.step(reference_loss, settings.max_grad_norm)
    now = reference.parameters()
    initial = learner.parameters()
    learner.set_learning_rate(settings.learning_rate)
    learner.step(loss, settings.max_grad_norm)
    after = learner.parameters()
    expected_after = port.parameter_layout(canonical_parameters(now))
    expected_delta = port.parameter_layout(
        canonical_parameters({k: v - before[k] for k, v in now.items()})
    )
    worst = 1.0
    largest = 0.0
    for name, value in after.items():
        if name in zero:
            continue
        largest = max(largest, float(np.abs(value - expected_after[name]).max()))
        cosine = _cosine(value - initial[name], expected_delta[name])
        if cosine is not None:
            worst = min(worst, cosine)
    report.step_parameter_max_abs = largest
    report.step_update_min_cosine = worst


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
            float(np.abs(port_values - values.numpy()).max()),
        )
    policy.clear_cache()


def _print(name: str, report: Report) -> None:
    for component, measured, passed in report.checks():
        print(f"{name:<6} {component:<40} {measured}  {'ok' if passed else 'FAIL'}")
    print(
        f"{name:<6} {'one step':<40} params max |Δ| "
        f"{report.step_parameter_max_abs:.1e}, update min cosine "
        f"{report.step_update_min_cosine:.6f} (informative)"
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
    args = parser.parse_args(argv)
    names = installed_backends()
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
            )
    except (FileNotFoundError, UnsupportedPolicy) as error:
        parser.error(str(error))  # exits 2; 1 means the models disagree
    failures = {name: report.failures() for name, report in reports.items()}
    for name, report in reports.items():
        _print(name, report)
    if args.json is not None:
        args.json.write_text(
            json.dumps(
                {
                    name: asdict(report) | {"failures": failures[name]}
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
