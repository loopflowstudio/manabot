"""MLX device setup and fresh policy construction."""

from __future__ import annotations

import mlx.core as mx

from manabot.backends.mlx.arrays import mlx_device
from manabot.backends.mlx.checkpoint import require_supported
from manabot.backends.mlx.policy import MLXActorCritic
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.policy.catalog import fresh_policy_tables
from manabot.policy.checkpoint import Dimensions
from manabot.policy.protocol import Precision as MLXPrecision
from manabot.policy.readings import Readings, fresh_readings
from manabot.training.learning import Precision

_THREADS = 1


def configure(threads: int, precision: Precision, device: str) -> None:
    global _THREADS
    target = mlx_device(device)
    if precision == "tf32":
        raise ValueError("precision tf32 requires CUDA")
    _THREADS = threads
    mx.set_default_device(target)


def threads() -> int:
    return _THREADS


def create_policy(
    space: ObservationSpace,
    config: ModelConfig,
    *,
    readings: Readings | None = None,
    bucketed_shapes: bool = False,
    live_tokens: bool = False,
    precision: MLXPrecision | None = None,
) -> MLXActorCritic:
    if live_tokens:
        raise ValueError("MLX does not support live-token inference")
    require_supported(config, space.config)
    chosen = fresh_readings(space) if readings is None else readings
    tables, token_rows, name_rows = fresh_policy_tables(space, config, chosen)
    dims = Dimensions.from_layout(
        config, space, tables.layout, token_rows=token_rows, name_rows=name_rows
    )
    policy = MLXActorCritic(
        dims,
        tables,
        config,
        space,
        precision=precision,
        bucketed_shapes=bucketed_shapes,
    )
    mx.eval(policy.parameters())
    return policy
