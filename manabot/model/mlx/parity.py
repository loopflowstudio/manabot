"""Check that the MLX port agrees with the PyTorch policy on captured real inputs.

    uv run --extra mlx python -m manabot.model.mlx.parity CHECKPOINT.pt --capture DIR

`DIR` is a capture written by `experiments/mlx_pilot.py capture`: learner
minibatches (`learner/NNN.npz` with `minibatches.json`) and collection calls
(`collection/NNNNN.npz`). The reference is the PyTorch model in float32 on CPU,
running the training code's own loss (`damped_policy_loss`, `selected_mean`).
For each minibatch, at the checkpoint's weights:

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
not part of the bar. The command prints one line per check, writes `--json`
when given, and exits 1 naming the worst component if any check fails.
"""

import argparse
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sys
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np
import torch

from manabot.model.admission import admit_observation
from manabot.model.checkpoint import load_checkpoint
from manabot.model.mlx.checkpoint import GRUS, build, mlx_parameters, torch_to_mlx
from manabot.model.mlx.learner import MLXLearner, Settings, Targets, damped_loss
from manabot.model.mlx.prepare import reference_distribution
from manabot.model.policy_distribution import legal_mask
from manabot.training.learning.damped_policy import damped_policy_loss
from manabot.training.learning.references import (
    reference_distribution as torch_reference,
)
from manabot.training.learning.selection import selected_mean

LOGIT_TOLERANCE = 1e-4
LOSS_TOLERANCE = 1e-3
COSINE_TOLERANCE = 0.999
ZERO_GRADIENT = 1e-6  # of the global gradient norm


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


def torch_gradients(agent: torch.nn.Module, hidden: int) -> dict[str, np.ndarray]:
    """PyTorch gradients under MLX parameter names.

    A folded GRU input bias `b` receives exactly `bias_ih`'s gradient (the
    reset/update parts of `bias_hh` have the same one), and `bhn` the new-gate
    slice of `bias_hh`'s.
    """
    grads: dict[str, np.ndarray] = {}
    for name, parameter in agent.named_parameters():
        grad = (
            np.zeros(tuple(parameter.shape), np.float32)
            if parameter.grad is None
            else parameter.grad.detach().numpy()
        )
        prefix = next((p for p in GRUS if name.startswith(p + ".")), None)
        if prefix is not None and name.endswith("bias_ih_l0"):
            grads[f"{prefix}.b"] = grad
        grads.update(torch_to_mlx(name, grad, hidden))
    return grads


def _cosine(a: np.ndarray, b: np.ndarray, floor: float = 0.0) -> float | None:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= floor and nb <= floor:
        return None
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def _legal_difference(
    mlx_logits: np.ndarray, torch_logits: np.ndarray, legal: np.ndarray
) -> tuple[float, float]:
    difference = np.abs(mlx_logits - torch_logits)[legal]
    scale = np.abs(torch_logits[legal]).max()
    return float(difference.max()), float(difference.max() / scale) if scale else 0.0


def check(
    checkpoint: Path,
    capture: Path,
    minibatch_count: int | None = 8,
    collection_count: int = 8,
) -> Report:
    torch.set_num_threads(8)
    agent, space = load_checkpoint(str(checkpoint))
    state = {k: v.detach().numpy().copy() for k, v in agent.state_dict().items()}
    policy, _ = build(agent.config, space.encoder.config, state)
    hidden = agent.config.hidden_dim
    learner = MLXLearner(policy)
    report = Report()
    agent.train()
    for index, (obs, targets, settings, meta) in enumerate(
        minibatches(capture, minibatch_count)
    ):
        tensors = {k: torch.from_numpy(v) for k, v in obs.items()}
        admitted = admit_observation(agent, tensors)
        valid = legal_mask(admitted["actions_valid"])
        agent.zero_grad(set_to_none=True)
        logits, values = agent.forward_distribution(admitted)
        reference = torch_reference(admitted, meta["reference"])
        if not np.allclose(reference.numpy(), targets.reference, atol=1e-7):
            raise AssertionError("numpy and torch reference distributions differ")
        policy_loss = damped_policy_loss(
            logits,
            torch.from_numpy(targets.actions),
            torch.from_numpy(targets.advantages),
            torch.from_numpy(targets.behavior_logs),
            reference,
            valid,
            clip=settings.clip,
            collection_kl=settings.collection_kl,
            tau=settings.tau,
        )
        policy_loss = selected_mean(policy_loss, torch.from_numpy(targets.actor_mask))
        value_loss = (
            (values.squeeze(-1) - torch.from_numpy(targets.value_targets))
            .square()
            .mean()
        )
        loss = policy_loss + value_loss
        loss.backward()
        expected = torch_gradients(agent, hidden)

        batch = policy.prepare(obs)
        mlx_loss, (mlx_logits, mlx_values) = damped_loss(
            policy, batch, targets, settings
        )
        loss_value, grads = learner.gradients(batch, targets, settings)
        mx.eval(mlx_logits, mlx_values, mlx_loss)
        absolute, relative = _legal_difference(
            np.array(mlx_logits), logits.detach().numpy(), valid.numpy()
        )
        value_difference = float(
            np.abs(np.array(mlx_values) - values.detach().numpy()).max()
        )
        reference_loss = float(loss.detach())
        loss_relative = abs(float(loss_value) - reference_loss) / max(
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
        for name, grad in tree_flatten(grads):
            cosine = _cosine(np.array(grad), expected[name], floor)
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
            "loss_mlx": float(loss_value),
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
            _compare_step(
                agent, learner, batch, targets, settings, zero, hidden, report
            )
            policy.update(
                build(agent.config, space.encoder.config, state)[0].parameters()
            )
            learner = MLXLearner(policy)
            agent.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})
        print(
            f"minibatch {meta['index']:3d}  B {row['B']:3d}  logits {absolute:.1e}  "
            f"values {value_difference:.1e}  loss {loss_relative:.1e}  "
            f"min cosine {cosines[worst]:.6f} ({worst})",
            flush=True,
        )
    _collection(agent, policy, capture, collection_count, report)
    return report


def _compare_step(
    agent: torch.nn.Module,
    learner: MLXLearner,
    batch: Any,
    targets: Targets,
    settings: Settings,
    zero: set[str],
    hidden: int,
    report: Report,
) -> None:
    """One clipped Adam step in both frameworks from the same weights.

    Tensors with structurally zero gradients (`zero`) are skipped: Adam's
    first step moves their rounding noise by +-lr in either direction.
    """
    before = {k: v.detach().clone() for k, v in agent.named_parameters()}
    optimizer = torch.optim.Adam(
        agent.parameters(), lr=settings.learning_rate, eps=1e-5
    )
    torch.nn.utils.clip_grad_norm_(list(agent.parameters()), settings.max_grad_norm)
    optimizer.step()
    initial = {k: np.array(v) for k, v in tree_flatten(learner.policy.parameters())}
    learner.step(batch, targets, settings)
    after = {k: np.array(v) for k, v in tree_flatten(learner.policy.parameters())}
    now = {k: v.detach().numpy() for k, v in agent.named_parameters()}
    expected_after = mlx_parameters(now, hidden)
    expected_delta = mlx_parameters(
        {k: v - before[k].numpy() for k, v in now.items()}, hidden
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
    policy: Any,
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
        mlx_logits, mlx_values = policy(policy.prepare(obs, inference=True))
        mx.eval(mlx_logits, mlx_values)
        absolute, _ = _legal_difference(
            np.array(mlx_logits), logits.numpy(), obs["actions_valid"] > 0
        )
        report.collection_logits = max(report.collection_logits, absolute)
        report.collection_values = max(
            report.collection_values,
            float(np.abs(np.array(mlx_values)[:, 0] - values.numpy()).max()),
        )
    policy.clear_cache()


def _print(report: Report) -> None:
    for name, measured, passed in report.checks():
        print(f"{name:<40} {measured}  {'ok' if passed else 'FAIL'}")
    print(
        f"{'one step':<40} params max |Δ| {report.step_parameter_max_abs:.1e}, "
        f"update min cosine {report.step_update_min_cosine:.6f} (informative)"
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
    report = check(args.checkpoint, args.capture, args.minibatches, args.collection)
    _print(report)
    failures = report.failures()
    if args.json is not None:
        args.json.write_text(
            json.dumps(asdict(report) | {"failures": failures}, indent=2)
        )
    if failures:
        print("parity: FAIL (" + ", ".join(failures) + ")")
        return 1
    print("parity: pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
