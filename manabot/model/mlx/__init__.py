"""MLX port of the policy and its PPO learner step (Apple Silicon); see mlx.md.

Requires the optional `mlx` extra. No other manabot module imports this package.
"""

try:
    import mlx.core  # noqa: F401
except ImportError as error:  # pragma: no cover - depends on the platform
    raise ImportError(
        "manabot.model.mlx requires the optional 'mlx' extra "
        "(uv sync --extra mlx, Apple Silicon only)"
    ) from error
