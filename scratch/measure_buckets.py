"""MTG-302 replay: bucketed shapes against the default path on captured inputs.

Usage: uv run python scratch/measure_buckets.py <e1 dir> <out.json>
CPU, one thread, sequential. Counts only; no timing.
"""

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

from manabot.model.buckets import bucket_sizes, trim_observation
from manabot.model.checkpoint import load_checkpoint
from manabot.model.inference import PolicyInference

torch.set_num_threads(1)
E = Path(sys.argv[1])


def masked(index: Any) -> bool:
    if isinstance(index, torch.Tensor):
        return index.dtype == torch.bool
    return isinstance(index, tuple) and any(masked(x) for x in index)


class Reads:
    """E1's counter: host API crossings plus data-dependent output shapes."""

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.shapes: list[tuple[str, tuple[int, ...]]] = []
        self.active = False
        self.saved: list[tuple[Any, str, Any]] = []
        for name in ("item", "__bool__", "cpu", "tolist", "nonzero", "__getitem__"):
            original = getattr(torch.Tensor, name)
            self.saved.append((torch.Tensor, name, original))

            def wrapper(t, *a, _n=name, _o=original, **k):  # noqa: ANN001, ANN002, ANN003, ANN202
                selected = _n == "__getitem__" and bool(a) and masked(a[0])
                if self.active and (_n != "__getitem__" or selected):
                    self.counts[_n] += 1
                result = _o(t, *a, **k)
                if self.active and (selected or _n == "nonzero"):
                    if isinstance(result, torch.Tensor):
                        self.shapes.append((_n, tuple(result.shape)))
                return result

            setattr(torch.Tensor, name, wrapper)
        for name in ("equal", "nonzero"):
            original = getattr(torch, name)
            self.saved.append((torch, name, original))

            def function(*a, _n=name, _o=original, **k):  # noqa: ANN002, ANN003, ANN202
                if self.active:
                    self.counts[f"torch.{_n}"] += 1
                result = _o(*a, **k)
                if self.active and _n == "nonzero" and isinstance(result, torch.Tensor):
                    self.shapes.append((_n, tuple(result.shape)))
                return result

            setattr(torch, name, function)

    def run(self, inputs: dict[str, torch.Tensor], call: Any) -> tuple[Any, int, str]:
        self.counts.clear()
        self.shapes.clear()
        self.active = True
        try:
            result = call()
        finally:
            self.active = False
        shapes = [(k, tuple(v.shape)) for k, v in sorted(inputs.items())]
        digest = hashlib.sha256(json.dumps([shapes, self.shapes]).encode()).hexdigest()
        return result, sum(self.counts.values()), digest[:16]


def load_obs(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as arrays:
        return {key: torch.from_numpy(arrays[key]) for key in arrays.files}


def agents(size: str) -> tuple[Any, Any]:
    os.environ.pop("MANABOT_BUCKETED_SHAPES", None)
    default, _ = load_checkpoint(E / f"captures/{size}/initial.pt")
    os.environ["MANABOT_BUCKETED_SHAPES"] = "1"
    bucketed, _ = load_checkpoint(E / f"captures/{size}/initial.pt")
    os.environ.pop("MANABOT_BUCKETED_SHAPES")
    assert bucketed.bucketed_shapes and not default.bucketed_shapes
    # Captured update-zero weights have zero fact/link output layers; perturb
    # every parameter identically so those paths carry signal.
    generator = torch.Generator().manual_seed(302)
    with torch.no_grad():
        for a, b in zip(default.parameters(), bucketed.parameters(), strict=True):
            noise = torch.randn(a.shape, generator=generator) * 0.02
            a.add_(noise)
            b.add_(noise)
    return default.eval(), bucketed.eval()


def model_rows(obs: dict[str, torch.Tensor], agent: Any) -> dict[str, int]:
    """Rows per axis the bucketed forward processes, per observation batch."""
    buckets = bucket_sizes(
        obs, agent.observation_space.encoder, objects=agent.object_trimming()
    )
    assert buckets is not None
    return dict(buckets.sizes)


def cpu_compacted(obs: dict[str, torch.Tensor]) -> dict[str, int]:
    """Current CPU-inference compaction: union of live fact/object columns."""
    validity = torch.cat(
        [
            obs[k]
            for k in (
                "agent_player_valid",
                "opponent_player_valid",
                "agent_cards_valid",
                "opponent_cards_valid",
                "agent_permanents_valid",
                "opponent_permanents_valid",
            )
        ],
        1,
    )
    return {
        "facts": int(obs["facts"][..., 11].bool().any(0).sum()),
        "objects": int((validity != 0).any(0).sum()),
    }


def compare(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).abs().max()) if a.numel() else 0.0


def main() -> None:
    reads = Reads()
    out: dict[str, Any] = {}
    for size in sys.argv[3:] or ("w64", "w160l8"):
        default, bucketed = agents(size)
        capacities = bucket_sizes(
            load_obs(E / f"captures/{size}/collection/000.npz"),
            default.observation_space.encoder,
            objects=True,
        ).capacities
        rows: dict[str, list[int]] = {}
        reads_before: list[int] = []
        reads_after: list[int] = []
        signatures_before: set[str] = set()
        signatures_after: set[str] = set()
        compacted: list[dict[str, int]] = []
        worst = {"logits": 0.0, "values": 0.0}
        elements: dict[str, int] = {}
        elements_total = 0
        batches = 0
        for path in sorted((E / f"captures/{size}/collection").glob("*.npz")):
            obs = load_obs(path)
            sizes = model_rows(obs, bucketed)
            batch = obs["facts"].shape[0]
            elements_total += batch
            for axis, count in sizes.items():
                rows.setdefault(axis, []).append(count)
                elements.setdefault(axis, 0)
                elements[axis] += count * batch
            compacted.append(cpu_compacted(obs))
            with torch.no_grad():
                with PolicyInference(default) as scope:
                    prepared = scope.prepare(obs)
                    (lb, vb), n, sig = reads.run(prepared, lambda: default(prepared))
                reads_before.append(n)
                signatures_before.add(sig)
                with PolicyInference(bucketed) as scope:
                    prepared = scope.prepare(obs)
                    trimmed = trim_observation(prepared, prepared.buckets)
                    (la, va), n, sig = reads.run(trimmed, lambda: bucketed(prepared))
                reads_after.append(n)
                signatures_after.add(sig)
            legal = obs["actions_valid"].bool()
            worst["logits"] = max(worst["logits"], compare(lb[legal], la[legal]))
            worst["values"] = max(worst["values"], compare(vb, va))
            assert torch.equal(lb[~legal], la[~legal])
            batches += 1
            if batches % 50 == 0:
                print(size, "collection", batches, worst, flush=True)
        out[f"{size}/collection"] = {
            "batches": batches,
            "max_abs_diff": worst,
            "signatures": [len(signatures_before), len(signatures_after)],
            "host_reads_median": [
                float(np.median(reads_before)),
                float(np.median(reads_after)),
            ],
            "capacity": capacities,
            "bucket_rows_median_per_batch": {
                axis: int(np.median(values))
                for axis, values in rows.items()
            },
            "bucket_rows_distinct": {
                axis: sorted(set(values)) for axis, values in rows.items()
            },
            "padded_elements_ratio_bucketed_over_capacity": {
                axis: count / (capacities[axis] * elements_total)
                for axis, count in elements.items()
            },
            "cpu_compacted_median": {
                key: float(np.median([c[key] for c in compacted]))
                for key in ("facts", "objects")
            },
        }
        # Learner minibatches: autograd through a surrogate loss on every output.
        learner: dict[str, Any] = {
            "cases": 0,
            "loss": 0.0,
            "grad": 0.0,
            "grad_rel": 0.0,
            "none_to_zero": [],
            "zero_to_none": [],
            "signatures": [set(), set()],
            "host_reads": [[], []],
        }
        for path in sorted((E / f"captures/{size}/learner").glob("*.pt")):
            capture = torch.load(path, weights_only=False)
            obs, actions = capture["obs"], capture["actions"].long()
            losses = []
            for index, agent in enumerate((default, bucketed)):
                agent.zero_grad(set_to_none=True)

                def step() -> torch.Tensor:
                    logits, values = agent.forward(obs)  # noqa: B023
                    logs = torch.log_softmax(logits, -1)
                    chosen = logs.gather(1, actions[:, None]).squeeze(1)  # noqa: B023
                    loss = -(chosen * capture["advantages"]).mean() + (  # noqa: B023
                        (values - capture["value_targets"]) ** 2  # noqa: B023
                    ).mean()
                    loss.backward()
                    return loss.detach()

                inputs = obs
                if index == 1:
                    buckets = bucket_sizes(
                        obs, agent.observation_space.encoder, objects=True
                    )
                    inputs = trim_observation(obs, buckets)
                loss, n, sig = reads.run(inputs, step)
                learner["host_reads"][index].append(n)
                learner["signatures"][index].add(sig)
                losses.append(loss)
            learner["cases"] += 1
            learner["loss"] = max(learner["loss"], compare(losses[0], losses[1]))
            for (name, a), b in zip(
                default.named_parameters(), bucketed.parameters(), strict=True
            ):
                if a.grad is None and b.grad is not None:
                    learner["none_to_zero"].append(
                        [name, path.name, float(b.grad.abs().max())]
                    )
                elif a.grad is not None and b.grad is None:
                    learner["zero_to_none"].append([name, path.name])
                elif a.grad is not None:
                    diff = compare(a.grad, b.grad)
                    learner["grad"] = max(learner["grad"], diff)
                    scale = float(a.grad.abs().max())
                    if scale > 0:
                        learner["grad_rel"] = max(learner["grad_rel"], diff / scale)
        learner["signatures"] = [len(s) for s in learner["signatures"]]
        learner["host_reads_median"] = [
            float(np.median(r)) for r in learner.pop("host_reads")
        ]
        out[f"{size}/learner"] = learner
        print(json.dumps({k: v for k, v in out.items() if k.startswith(size)}, indent=1))
    Path(sys.argv[2]).write_text(json.dumps(out, indent=2) + "\n")


main()
