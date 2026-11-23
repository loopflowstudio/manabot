"""MLX device setup and fresh policy construction."""

from __future__ import annotations

import mlx.core as mx

from manabot.backends.mlx.checkpoint import require_supported
from manabot.backends.mlx.policy import MLXActorCritic
from manabot.env.observation import FACT_KINDS, ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.policy.catalog import fresh_policy_tables
from manabot.policy.checkpoint import Dimensions
from manabot.policy.prepare import Layout
from manabot.policy.protocol import Precision as MLXPrecision
from manabot.policy.readings import Readings, fresh_readings
from manabot.training.learning import Precision

_THREADS = 1


def configure(threads: int, precision: Precision, device: str) -> None:
    global _THREADS
    if device == "cuda":
        raise ValueError("MLX does not support CUDA")
    if device not in {"cpu", "mps", "gpu"}:
        raise ValueError(f"MLX does not support device {device!r}")
    if precision == "tf32":
        raise ValueError("precision tf32 requires CUDA")
    _THREADS = threads
    mx.set_default_device(mx.cpu if device == "cpu" else mx.gpu)


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
    columns = Layout(chosen)
    player = columns.numeric_width("agent_player")
    card = columns.numeric_width("agent_cards")
    if space.turn_state:
        player += columns.numeric_width("player_turn_state")
        card += 1
    dims = Dimensions(
        hidden=config.hidden_dim,
        heads=config.num_attention_heads,
        layers=config.attention_layers,
        feedforward=config.attention_feedforward_dim
        or config.num_attention_heads * config.hidden_dim,
        player=player,
        card=card,
        permanent=columns.numeric_width("agent_permanents"),
        action=columns.numeric_width("actions"),
        focus=space.config.max_focus_objects,
        relation=columns.numeric_width("relations"),
        effect=columns.numeric_width("effects"),
        fact=columns.fact_width(catalog_ids=True),
        fact_kinds=FACT_KINDS,
        token_rows=token_rows,
        name_rows=name_rows,
        links=len(columns.link_columns),
        stack=columns.numeric_width("stack") if space.one_token else 0,
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
