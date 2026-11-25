"""Deprecated imports; use managym.card_capture (removed in 0.3.0)."""

from typing import Any
import warnings

from managym import card_capture


def __getattr__(name: str) -> Any:
    if name.startswith("_"):
        raise AttributeError(name)
    warnings.warn(
        f"{__name__}.{name} moved to managym.card_capture; removal in 0.3.0",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(card_capture, name)
