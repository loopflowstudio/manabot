"""Deprecated aliases for names that moved to another module.

A module that published a name keeps resolving it after the move through a
module-level `__getattr__`, warning with the new import path and the release
that removes the alias. The target imports lazily, so an alias never adds an
import edge to the module that keeps it.
"""

from collections.abc import Callable, Mapping
import importlib
import warnings


def moved(
    module: str, targets: Mapping[str, str], *, removal: str = "0.3.0"
) -> Callable[[str], object]:
    """A module `__getattr__` resolving `targets` (name -> new module) with a warning."""

    def __getattr__(name: str) -> object:
        target = targets.get(name)
        if target is None:
            raise AttributeError(f"module {module!r} has no attribute {name!r}")
        warnings.warn(
            f"{module}.{name} is deprecated; import {target}.{name}. "
            f"It will be removed in {removal}.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(importlib.import_module(target), name)

    return __getattr__
