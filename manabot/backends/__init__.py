"""Framework backends; only composition roots select one through `resolve`."""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from manabot.policy.protocol import Backend, Precision


def resolve(name: str, *, precision: "Precision | None" = None) -> "Backend":
    """Select a configured backend, loading its optional framework on first use.

    MLX's omitted precision follows each policy's width. Torch's resolver
    supports float32; reduced-precision training is configured on its learner.
    """
    if name not in {"torch", "mlx"}:
        raise ValueError(f"unknown backend {name!r}; available backends: torch, mlx")
    if name == "torch" and precision not in (None, "float32"):
        raise ValueError("half-precision parity is supported only for the MLX backend")
    try:
        importlib.import_module("torch" if name == "torch" else "mlx.core")
    except ImportError as error:
        raise ImportError(
            f"backend {name!r} requires the {name!r} extra (uv sync --extra {name})"
        ) from error
    checkpoint = importlib.import_module(f"manabot.backends.{name}.checkpoint")
    if name == "mlx":
        return checkpoint.MLXBackend(precision=precision)
    return checkpoint.BACKEND
