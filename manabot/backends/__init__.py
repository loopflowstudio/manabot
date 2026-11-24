"""Framework backends; only composition roots select one through `resolve`."""

import importlib
from typing import TYPE_CHECKING, Literal, overload

if TYPE_CHECKING:
    from manabot.backends.mlx.checkpoint import MLXBackend
    from manabot.backends.torch.checkpoint import TorchBackend
    from manabot.policy.protocol import BackendName, Precision


@overload
def resolve(
    name: Literal["torch"], *, precision: "Precision | None" = None
) -> "TorchBackend": ...


@overload
def resolve(
    name: Literal["mlx"], *, precision: "Precision | None" = None
) -> "MLXBackend": ...


@overload
def resolve(
    name: str, *, precision: "Precision | None" = None
) -> "TorchBackend | MLXBackend": ...


def resolve(
    name: str, *, precision: "Precision | None" = None
) -> "TorchBackend | MLXBackend":
    """Select a configured backend, loading its optional framework on first use.

    With precision omitted, MLX uses the self-play settings for training
    and the policy's width for standalone inference. Torch's resolver
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


def local_cpu_backend(name: "BackendName") -> "TorchBackend | MLXBackend":
    """Configure a standalone evaluation child before it loads any policies."""
    backend = resolve(name, precision="float32")
    backend.configure_cpu_inference()
    return backend
