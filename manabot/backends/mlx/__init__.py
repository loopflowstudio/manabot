"""The MLX backend: the policy and its PPO learner step on Apple Silicon; see
../mlx.md.

Requires the optional `mlx` extra. It is the only manabot package that imports
MLX, and no other manabot module imports it.
"""

try:
    import mlx.core  # noqa: F401
except ImportError as error:  # pragma: no cover - depends on the platform
    raise ImportError(
        "manabot.backends.mlx requires the optional 'mlx' extra "
        "(uv sync --extra mlx, Apple Silicon only)"
    ) from error
