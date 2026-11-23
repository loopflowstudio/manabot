"""MLX device setup and fresh policy construction."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import mlx.core as mx
from mlx.utils import tree_flatten

from manabot.backends.mlx.arrays import all_finite, mlx_device
from manabot.backends.mlx.checkpoint import require_supported
from manabot.backends.mlx.learner import Learner
from manabot.backends.mlx.policy import MLXActorCritic
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.policy.catalog import fresh_policy_tables
from manabot.policy.checkpoint import Dimensions
from manabot.policy.prepare import Layout
from manabot.policy.protocol import Precision as MLXPrecision
from manabot.policy.readings import fresh_readings
from manabot.training.learning import Precision


def configure(threads: int, precision: Precision, device: str) -> None:
    """Set the MLX device. MLX manages its own CPU thread pool."""
    if precision == "tf32":
        raise ValueError("precision tf32 requires CUDA")
    mx.set_default_device(mlx_device(device))


@contextmanager
def scoped(threads: int, precision: Precision, device: str) -> Iterator[None]:
    previous_device = mx.default_device()
    try:
        configure(threads, precision, device)
        yield
    finally:
        mx.set_default_device(previous_device)


@contextmanager
def preserve_rng(device: str) -> Iterator[None]:
    # MLX exposes the current key as two uint32 words and seeds from a uint64.
    high, low = mx.random.state[0].tolist()
    try:
        yield
    finally:
        mx.random.seed((high << 32) | low)


def synchronize(device: str) -> None:
    mx.synchronize(mlx_device(device))


def memory_metrics(device: str) -> dict[str, int]:
    return (
        {"memory/active_mlx_bytes": mx.get_active_memory()} if device == "mps" else {}
    )


def create_learner(
    agent: MLXActorCritic,
    device: str,
    *,
    minibatch_rows: int,
    precision: Precision,
    execution_rows: int | None,
    bootstrap_rows: int | None,
) -> Learner:
    expected = "bfloat16" if precision == "bf16" else precision
    if expected != agent.precision:
        raise ValueError(
            f"self-play precision {precision} differs from MLX policy precision {agent.precision}"
        )
    return Learner(
        agent,
        device,
        minibatch_rows=minibatch_rows,
        execution_rows=execution_rows,
        bootstrap_rows=bootstrap_rows,
    )


@dataclass(frozen=True)
class Runtime:
    """The same configured precision constructs fresh and restored policies."""

    precision: MLXPrecision | None
    configure = staticmethod(configure)
    scoped = staticmethod(scoped)
    preserve_rng = staticmethod(preserve_rng)
    synchronize = staticmethod(synchronize)
    memory_metrics = staticmethod(memory_metrics)
    create_learner = staticmethod(create_learner)

    def create_policy(
        self,
        space: ObservationSpace,
        config: ModelConfig,
        *,
        pretrained: MLXActorCritic | None = None,
        bucketed_shapes: bool = False,
        live_tokens: bool = False,
    ) -> MLXActorCritic:
        if live_tokens:
            raise ValueError("MLX does not support live-token inference")
        require_supported(config, space.config)
        if pretrained is not None:
            if pretrained.config != config:
                raise ValueError("pretrained policy has another model config")
            if not all_finite(
                value for _, value in tree_flatten(pretrained.parameters())
            ):
                raise ValueError("pretrained policy contains nonfinite tensors")
            return pretrained
        layout = Layout(fresh_readings(space))
        tables = fresh_policy_tables(space, config)
        dims = Dimensions.from_layout(
            config,
            space,
            layout,
            token_rows=tables.token_rows,
            name_rows=len(tables.definitions),
        )
        policy = MLXActorCritic(
            dims,
            tables.preparation_tables(layout),
            config,
            space,
            precision=self.precision,
            bucketed_shapes=bucketed_shapes,
        )
        mx.eval(policy.parameters())
        return policy
