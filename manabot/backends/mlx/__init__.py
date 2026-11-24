"""The MLX backend: policy inference, learning and belief sampling on Apple Silicon.

Requires the optional `mlx` extra. It is the only manabot package that imports
MLX, and no other manabot module imports it. The updates take this module as
their `backend` argument and reach its files as attributes (`backend.rl`,
`backend.learner`, `backend.arrays`, `backend.rollout`), which load on first
use.
"""

import importlib
from types import ModuleType

try:
    import mlx.core  # noqa: F401
except ImportError as error:  # pragma: no cover - depends on the platform
    raise ImportError(
        "manabot.backends.mlx requires the optional 'mlx' extra "
        "(uv sync --extra mlx, Apple Silicon only)"
    ) from error

_FILES = frozenset(
    {
        "arrays",
        "belief",
        "belief_fit",
        "checkpoint",
        "collection",
        "learner",
        "policy",
        "rl",
        "rollout",
        "runtime",
        "state",
    }
)


def __getattr__(name: str) -> ModuleType:
    if name in _FILES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
