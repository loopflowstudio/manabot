"""Deprecated imports; use managym.journal (removed in 0.3.0)."""

from typing import Any
import warnings

from managym import journal


def __getattr__(name: str) -> Any:
    if name.startswith("_"):
        raise AttributeError(name)
    warnings.warn(
        f"{__name__}.{name} moved to managym.journal; removal in 0.3.0",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(journal, name)
