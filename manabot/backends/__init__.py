"""Framework backends; only composition roots select one through `resolve`."""

import importlib
from types import ModuleType


def resolve(name: str) -> ModuleType:
    """Load a backend lazily, with a useful error for an absent optional extra."""
    if name not in {"torch", "mlx"}:
        raise ValueError(f"unknown backend {name!r}; available backends: torch, mlx")
    try:
        importlib.import_module("torch" if name == "torch" else "mlx.core")
    except ImportError as error:
        raise ImportError(
            f"backend {name!r} requires the {name!r} extra (uv sync --extra {name})"
        ) from error
    return importlib.import_module(f"manabot.backends.{name}")
