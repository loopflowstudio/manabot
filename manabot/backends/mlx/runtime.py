"""MLX device setup and fresh policy construction."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import mlx.core as mx
from mlx.utils import tree_flatten

from manabot.backends.mlx.arrays import all_finite, mlx_device, training_precision
from manabot.backends.mlx.learner import Learner
from manabot.backends.mlx.policy import MLXActorCritic
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.policy.catalog import fresh_policy_tables
from manabot.policy.checkpoint import (
    Dimensions,
    memory_parameter,
    pretrained_memory_branch,
)
from manabot.policy.prepare import Layout
from manabot.policy.protocol import Precision as MLXPrecision
from manabot.policy.readings import fresh_readings
from manabot.training.learning import Precision


def configure(threads: int, precision: Precision, device: str) -> None:
    """Set the MLX device. MLX manages its own CPU thread pool."""
    training_precision(precision)
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
def preserve_rng(device: str | None = None) -> Iterator[None]:
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
    if training_precision(precision) != agent.precision:
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

    @staticmethod
    def admit_execution(device: str, *, live_tokens: bool) -> None:
        mlx_device(device)

    def create_policy(
        self,
        space: ObservationSpace,
        config: ModelConfig,
        *,
        pretrained: MLXActorCritic | None = None,
        precision: Precision | None = None,
        bucketed_shapes: bool = False,
        live_tokens: bool = False,
    ) -> MLXActorCritic:
        requested = training_precision(precision) if precision is not None else None
        if (
            self.precision is not None
            and requested is not None
            and requested != self.precision
        ):
            raise ValueError(
                f"self-play precision {precision} differs from MLX policy precision {self.precision}"
            )
        effective = requested or self.precision
        if pretrained is None:
            layout = Layout(fresh_readings(space))
            catalog = fresh_policy_tables(space, config)
            dims = Dimensions.from_layout(
                config,
                space,
                layout,
                token_rows=catalog.token_rows,
                name_rows=len(catalog.definitions),
            )
            tables = catalog.preparation_tables(layout)
        else:
            branch = pretrained_memory_branch(pretrained.config, config, space.config)
            weights = list(tree_flatten(pretrained.parameters()))
            if not all_finite(value for _, value in weights):
                raise ValueError("pretrained policy contains nonfinite tensors")
            if not branch:
                if effective is not None:
                    pretrained.set_precision(effective)
                pretrained.live_tokens = live_tokens
                pretrained.bucketed_shapes = bucketed_shapes
                return pretrained
            # The published table binds the writer's reading widths and vocabulary.
            dims, tables = pretrained.dims, pretrained._static.tables
        policy = MLXActorCritic(
            dims,
            tables,
            config,
            space,
            precision=effective,
            bucketed_shapes=bucketed_shapes,
            live_tokens=live_tokens,
        )
        if pretrained is not None:
            common = {
                name: array.shape
                for name, array in tree_flatten(policy.parameters())
                if not memory_parameter(name)
            }
            if common != {name: array.shape for name, array in weights}:
                raise ValueError(
                    "pretrained copy would leave non-memory parameters unused or uninitialized"
                )
            policy.load_weights(weights, strict=False)
        mx.eval(policy.parameters())
        return policy
