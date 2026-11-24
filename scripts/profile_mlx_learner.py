"""Attribute MLX learner time on retained, real minibatches from torch.capture.

Two controls keep the forward loss fixed. The catalog control reuses the program
table at the current weights. The attention control also replays each object
attention block's output with an identity derivative on its object input, so
the rest of the encoder still has a backward pass. Their timings overlap.
Only the ordinary gradient is passed to the production clipping and Adam step.

    uv run --extra mlx python -m scripts.profile_mlx_learner CAPTURE --bucketed
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict, dataclass, fields
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from manabot.backends import resolve
from manabot.backends.mlx.learner import Learner
from manabot.backends.mlx.policy import ObjectAttention
from manabot.backends.parity import TOLERANCES, minibatch_loss, minibatches
from manabot.env.observation import ObservationBuffers
from manabot.policy.protocol import Precision, Settings, Targets
from manabot.training.updates.backend import UpdateBackend


@dataclass(frozen=True)
class BatchProfile:
    rows: int
    attention_shapes: tuple[tuple[int, ...], ...]
    live_object_columns: int
    live_objects_per_row: float
    catalog_definitions: int
    catalog_tokens: int
    catalog_steps: int
    load: float
    loss: float
    control_loss_max_abs: float
    norm: float
    host_prepare_ms: float
    catalog_forward_ms: float
    full_ms: float
    fixed_catalog_ms: float
    fixed_catalog_attention_ms: float
    clip_adam_ms: float


def profile_step(
    backend: UpdateBackend,
    learner: Learner,
    obs: ObservationBuffers,
    targets: Targets,
    settings: Settings,
    index: int,
) -> BatchProfile:
    """Measure one batch and advance the learner by its ordinary damped update."""
    policy = learner.agent
    learner.set_learning_rate(settings.learning_rate)
    load = os.getloadavg()[0]
    mx.synchronize()
    start = perf_counter()
    prepared = policy.prepare(obs)
    prep = perf_counter() - start
    programs = prepared.programs
    assert programs is not None
    start = perf_counter()
    fixed = policy.programs(programs)
    mx.eval(fixed)
    catalog_forward = perf_counter() - start

    outputs: list[mx.array] = []
    shapes: list[tuple[int, ...]] = []
    attention = ObjectAttention.__call__

    def record(layer: ObjectAttention, *args: mx.array) -> mx.array:
        value = attention(layer, *args)
        outputs.append(value)
        if layer is policy.attention:
            shapes.append(value.shape)
        return value

    def replay(layer: ObjectAttention, objects: mx.array, *args: mx.array) -> mx.array:
        return next(recorded) + (objects - mx.stop_gradient(objects))

    with (
        patch.object(type(policy), "programs", return_value=fixed),
        patch.object(ObjectAttention, "__call__", record),
    ):
        mx.eval(policy(prepared), outputs)
    recorded = iter(outputs)

    controls = ["fixed_catalog", "fixed_catalog_attention"]
    if index % 2:
        controls.reverse()
    times: dict[str, float] = {}
    losses: list[float] = []
    # learner.loss retains the last gradient. Restore the full objective last.
    for mode in [*controls, "full"]:
        with ExitStack() as stack:
            if mode != "full":
                stack.enter_context(
                    patch.object(type(policy), "programs", return_value=fixed)
                )
            if mode == "fixed_catalog_attention":
                stack.enter_context(patch.object(ObjectAttention, "__call__", replay))
            mx.synchronize()
            start = perf_counter()
            loss = minibatch_loss(backend, policy, learner, prepared, targets, settings)
            mx.synchronize()
            times[mode] = (perf_counter() - start) * 1000
            losses.append(float(loss))
    # Repeated ordinary bfloat16 losses can differ at identical weights.
    rtol = TOLERANCES["bfloat16"].loss if policy.precision == "bfloat16" else 1e-6
    np.testing.assert_allclose(
        losses, losses[-1], rtol=rtol, atol=1e-7, equal_nan=False
    )
    start = perf_counter()
    norm = learner.step(loss, settings.max_grad_norm)
    mx.synchronize()
    step = perf_counter() - start
    return BatchProfile(
        rows=len(targets.actions),
        attention_shapes=tuple(shapes),
        live_object_columns=int(prepared.validity.any(axis=0).sum()),
        live_objects_per_row=float(prepared.validity.sum(axis=1).mean()),
        catalog_definitions=len(programs.rows),
        catalog_tokens=programs.tokens.size,
        catalog_steps=len(programs.schedule.sizes),
        load=load,
        loss=losses[-1],
        control_loss_max_abs=max(abs(value - losses[-1]) for value in losses[:-1]),
        norm=float(norm),
        host_prepare_ms=1000 * prep,
        catalog_forward_ms=1000 * catalog_forward,
        full_ms=times["full"],
        fixed_catalog_ms=times["fixed_catalog"],
        fixed_catalog_attention_ms=times["fixed_catalog_attention"],
        clip_adam_ms=1000 * step,
    )


def profile(
    source: Path, bucketed: bool, precision: Precision, count: int
) -> dict[str, object]:
    if count < 2:
        raise ValueError("profiling requires a warmup and a measured batch")
    backend = resolve("mlx", precision=precision)
    recipe_precision = "bf16" if precision == "bfloat16" else "float32"
    with backend.runtime.scoped(4, recipe_precision, "mps"):
        policy, _ = backend.load_policy(source / "initial.pt", bucketed_shapes=bucketed)
        policy.to("mps")
        learner = backend.runtime.create_learner(
            policy,
            "mps",
            minibatch_rows=1,
            precision=recipe_precision,
            execution_rows=None,
            bootstrap_rows=None,
        )
        samples = []
        for i, (obs, targets, settings, _) in enumerate(minibatches(source, count)):
            sample = profile_step(backend, learner, obs, targets, settings, i)
            samples.append(sample)
            print(json.dumps(asdict(sample)), flush=True)
    if len(samples) < 2:
        raise ValueError("the capture needs at least two minibatches")
    return {
        "source": str(source),
        "width": policy.config.hidden_dim,
        "bucketed_shapes": bucketed,
        "precision": policy.precision,
        "samples": [asdict(sample) for sample in samples],
        "warmup": 1,
        "median_ms": {
            field.name: median(getattr(sample, field.name) for sample in samples[1:])
            for field in fields(BatchProfile)
            if field.name.endswith("_ms")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--bucketed", action="store_true")
    parser.add_argument(
        "--precision", choices=("float32", "bfloat16"), default="float32"
    )
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = profile(args.source, args.bucketed, args.precision, args.batches)
    print(json.dumps(result["median_ms"]))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
