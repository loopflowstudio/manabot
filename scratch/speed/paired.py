"""Alternating kept/candidate PPO on one real batch and one locked build.

The kept lookup is recovered verbatim from 734f2cd5c. Both implementations
are compiled in this process; only this method differs. Timing has no gradient
hooks. Two separate audit updates per arm retain first-step raw gradients,
first-step/final weights and policy probes, with same-arm drift comparators.
"""

import argparse
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import statistics
import time

from baseline_lookup import lookup as _baseline
import numpy as np
import torch
from torch import Tensor
from torch.optim.optimizer import register_optimizer_step_post_hook

import manabot.backends.torch as backend
from manabot.backends.torch.learner import Learner, StepRecorder, apply_matmul_precision
from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import RewardConfig
from manabot.infra.runtime import seed_random
from manabot.model.admission import admit_observation
from manabot.model.inference import PolicyInference
from manabot.model.semantic_cards import SemanticCardEncoder
from manabot.training.execution import validate_regime
from manabot.training.regime import Learning, TrainingRegime, TrainSelfPlay
from manabot.training.rollout.collection import RolloutBatch, SeatRoutedCollector
from manabot.training.updates.ppo import update_iteration
from manabot.training.updates.selection import UpdateDiagnostics
import managym

_CANDIDATE = SemanticCardEncoder.lookup
_RECIPE = (
    Path(__file__).resolve().parents[2]
    / "experiments/regimes/elemental-long-183-v1.json"
)


def select_arm(arm: str) -> None:
    assert arm in ("baseline", "candidate")
    SemanticCardEncoder.lookup = staticmethod(
        _baseline if arm == "baseline" else _CANDIDATE
    )


def _snapshot(model: ActorCritic) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def _digest(values: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _write(path: Path, value: object) -> None:
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=1, allow_nan=False))
    temporary.replace(path)


class AuditLearner(Learner):
    """Capture raw gradients once, outside all reported timing samples."""

    def __init__(
        self, agent: ActorCritic, device: torch.device, rows: int, precision: str
    ) -> None:
        super().__init__(agent, device, minibatch_rows=rows, precision=precision)
        self.capture: dict[str, Tensor] = {}
        self.calls = 0

    def step(
        self,
        loss: Tensor,
        max_grad_norm: float,
        *,
        recorder: StepRecorder | None = None,
    ) -> Tensor:
        self.calls += 1
        if self.calls != 1:
            return super().step(loss, max_grad_norm, recorder=recorder)
        self.capture["first/loss"] = loss.detach().cpu()

        def save(name: str) -> Callable[[Tensor], None]:
            def hook(gradient: Tensor) -> None:
                self.capture[f"gradient/{name}"] = gradient.detach().cpu().clone()

            return hook

        handles = [
            p.register_hook(save(n))
            for n, p in self.agent.named_parameters()
            if p.requires_grad
        ]
        try:
            norm = super().step(loss, max_grad_norm, recorder=recorder)
        finally:
            for handle in handles:
                handle.remove()
        self.capture.update(
            {f"first_weight/{n}": p for n, p in _snapshot(self.agent).items()}
        )
        return norm


@dataclass
class Workload:
    agent: ActorCritic
    batch: RolloutBatch
    probe: dict[str, Tensor]
    initial: dict[str, Tensor]
    collection_seconds: float
    threads: int


def _collect(streams: int, transitions: int, device: torch.device) -> Workload:
    regime = validate_regime(
        TrainingRegime.model_validate_json(_RECIPE.read_bytes()).model_copy(
            update={"id": "mtg338-paired", "world": managym.WORLD_VERSION}
        )
    )
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    torch.set_num_threads(stage.execution.threads)
    seed_random(25201)
    space = ObservationSpace(regime.observation)
    agent = ActorCritic(space, regime.agent, bucketed_shapes=True, live_tokens=True).to(
        device
    )
    initial = deepcopy(agent.state_dict())
    collector = SeatRoutedCollector(
        space,
        Match(regime.root_match),
        Reward(RewardConfig()),
        num_envs=streams,
        seed=25201,
        opponent_mode="self",
        learning_seats="both",
        batch_closure=stage.batch_closure,
        games_per_stream=stage.games_per_stream,
        recovery_max_microsteps=regime.recovery_max_microsteps,
        matchup_curriculum="elemental-uniform",
        device=str(device),
        precision="tf32",
    )
    try:
        tick = time.perf_counter()
        batch = collector.collect(agent, transitions)
        elapsed = time.perf_counter() - tick
        # Same actual rows for policy probes before and after each update.
        with PolicyInference(agent) as scope:
            probe = {
                k: v.clone()
                for k, v in collector._slice_obs_tensors(
                    np.arange(min(24, streams)), scope
                ).items()
            }
    finally:
        collector.close()
    return Workload(agent, batch, probe, initial, elapsed, stage.execution.threads)


def _probe(work: Workload) -> dict[str, Tensor]:
    with torch.no_grad():
        logits, values = work.agent(admit_observation(work.agent, work.probe))
        # Policy support is checked independently; invalid logits may be -inf.
        valid = work.probe["actions_valid"].bool()
        probabilities = logits.masked_fill(~valid, -torch.inf).softmax(-1)
        return {
            "policy": probabilities.cpu(),
            "value": values.cpu(),
            "valid": valid.cpu(),
        }


def _run(
    work: Workload, arm: str, streams: int, audit: bool = False
) -> tuple[dict[str, object], dict[str, Tensor]]:
    select_arm(arm)
    work.agent.load_state_dict(work.initial)
    work.agent.train()
    learner = (
        AuditLearner(work.agent, torch.device("cuda"), streams, "tf32")
        if audit
        else Learner(work.agent, "cuda", minibatch_rows=streams, precision="tf32")
    )
    batch = deepcopy(work.batch)
    before = _probe(work) if audit else {}
    steps = 0

    def counted(*_: object) -> None:
        nonlocal steps
        steps += 1

    handle = register_optimizer_step_post_hook(counted)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    tick = time.perf_counter()
    try:
        diagnostics: UpdateDiagnostics = update_iteration(
            backend,
            learner,
            batch,
            Learning(reference="action_type_uniform"),
            0.3,
            np.random.default_rng(3),
            iteration=1,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - tick
        peak = torch.cuda.max_memory_allocated()
    finally:
        handle.remove()
    health = diagnostics["numerical"]
    assert steps == health["optimizer_steps"] == 16
    assert health["rejected_steps"] == health["skipped_steps"] == 0
    weights = _snapshot(work.agent)
    assert all(
        torch.isfinite(p).all() for p in weights.values() if p.is_floating_point()
    )
    receipt: dict[str, object] = {
        "arm": arm,
        "seconds": elapsed,
        "steps_per_second": steps / elapsed,
        "steps": steps,
        "errors": 0,
        "peak_allocated_bytes": peak,
        "diagnostics": diagnostics,
        "sections": learner.profiler.get_stats(),
        "weights_sha256": _digest(weights),
    }
    capture = learner.capture if isinstance(learner, AuditLearner) else {}
    if audit:
        capture.update({f"initial/{n}": p for n, p in before.items()})
        capture.update({f"final_weight/{n}": p for n, p in weights.items()})
        capture.update({f"final/{n}": p for n, p in _probe(work).items()})
    return receipt, capture


def _differences(
    left: dict[str, Tensor], right: dict[str, Tensor]
) -> dict[str, object]:
    assert left.keys() == right.keys()
    result: dict[str, object] = {}
    for name, a in left.items():
        b = right[name]
        assert a.shape == b.shape and a.dtype == b.dtype
        assert torch.isfinite(a).all() and torch.isfinite(b).all(), name
        a, b = a.double(), b.double()
        delta = (a - b).abs()
        norm = float(a.norm())
        result[name] = {
            "max_abs": float(delta.max()) if delta.numel() else 0,
            "relative_l2": float(delta.norm()) / max(norm, 1e-30),
            "reference_l2": norm,
            "different": int((a != b).sum()),
            "left_nonzero": int((a != 0).sum()),
            "right_nonzero": int((b != 0).sum()),
        }
    return result


def _lookup_reference() -> dict[str, object]:
    """Non-dyadic repeated sums measured against a small float64 oracle."""
    encoder = SemanticCardEncoder("ur-lessons-vs-gw-allies", 4).cuda()
    admitted = (encoder.definition_rows > 0).nonzero().flatten()[:2]
    choices = torch.cat((admitted.new_zeros(1), admitted))
    ids = choices[torch.arange(8193, device="cuda") % 3]
    rows = encoder._rows(ids).cpu()
    generator = torch.Generator().manual_seed(31)
    weights = torch.randn(8193, 4, generator=generator)
    count = encoder.catalog_shape.definitions + 1
    expected = torch.zeros(count, 4, dtype=torch.float64).index_add_(
        0, rows, weights.double()
    )
    values = torch.randn(count, 4, generator=generator).cuda()
    results: dict[str, object] = {}
    for arm in ("baseline", "candidate"):
        select_arm(arm)
        table = values.clone().requires_grad_()
        outputs = encoder.embed_definitions(ids, table)
        assert torch.equal(outputs, values[rows.cuda()])
        outputs.backward(weights.cuda())
        actual = table.grad.cpu().double()
        delta = (actual - expected).abs()
        results[arm] = {
            "max_abs": float(delta.max()),
            "relative_l2": float(delta.norm() / expected.norm()),
            "row_zero_max_abs": float(delta[0].max()),
            "outputs_exact": True,
            "nonzero_rows": int((actual != 0).any(-1).sum()),
        }
        assert torch.isfinite(actual).all()
        assert torch.equal((actual != 0).any(-1), (expected != 0).any(-1))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--streams", type=int, default=256)
    parser.add_argument("--transitions", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    apply_matmul_precision("tf32")
    work = _collect(args.streams, args.transitions, torch.device("cuda"))
    result: dict[str, object] = {
        "baseline_revision": "734f2cd5c",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "threads": work.threads,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "collection_seconds": work.collection_seconds,
        "learner_rows": work.batch.rewards.size,
        "initial_sha256": _digest({k: v.cpu() for k, v in work.initial.items()}),
        "samples": [],
        "audits": [],
        "lookup_reference": _lookup_reference(),
    }
    print(
        f"collection {work.collection_seconds:.3f}s rows={work.batch.rewards.size}",
        flush=True,
    )
    # One equal full-update warmup per arm. Audits are separate and follow timing.
    for arm in ("baseline", "candidate"):
        receipt, _ = _run(work, arm, args.streams)
        print(f"warmup {arm}: {receipt['seconds']:.3f}s", flush=True)
    for repeat in range(args.repeats):
        for arm in ("baseline", "candidate"):
            receipt, _ = _run(work, arm, args.streams)
            receipt["repeat"] = repeat + 1
            result["samples"].append(receipt)
            _write(args.out / "paired.json", result)
            print(
                f"{arm} {repeat + 1}: {receipt['seconds']:.3f}s steps={receipt['steps']}",
                flush=True,
            )
    captures: dict[str, dict[str, Tensor]] = {}
    for repeat in range(2):
        for arm in ("baseline", "candidate"):
            receipt, capture = _run(work, arm, args.streams, audit=True)
            name = f"{arm}-{repeat}"
            captures[name] = capture
            torch.save(capture, args.out / f"{name}.pt")
            result["audits"].append(receipt)
            _write(args.out / "paired.json", result)
            print(f"audit {name} complete", flush=True)
    comparisons = {}
    for a, b in [
        ("baseline-0", "baseline-1"),
        ("candidate-0", "candidate-1"),
        ("baseline-0", "candidate-0"),
        ("baseline-1", "candidate-1"),
    ]:
        comparisons[f"{a}:{b}"] = _differences(captures[a], captures[b])
    _write(args.out / "differences.json", comparisons)
    for arm in ("baseline", "candidate"):
        times = [r["seconds"] for r in result["samples"] if r["arm"] == arm]
        print(
            f"{arm}: median={statistics.median(times):.6f} range={min(times):.6f}..{max(times):.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
