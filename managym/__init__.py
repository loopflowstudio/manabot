"""The rules engine and native public API.

The engine types are implemented by PyO3 and described in ``__init__.pyi``;
shared decision/protocol adapters live beside this module.
"""

from collections.abc import Callable
import sys
from typing import ParamSpec, TypeVar
import warnings

from . import _managym
from ._managym import *  # noqa: F403 — expose the native API described by the stub.
from ._managym import format

# PyO3 submodules are attributes, not importable modules; registering the
# native ``format`` lets ``import managym.format`` and ``from managym.format
# import Limited`` resolve to the same object as ``managym.format``.
sys.modules[f"{__name__}.format"] = format

# ---------------------------------------------------------------------------
# Deprecated spellings, removed in 0.3.0. Engine names describe what managym
# produces; these older names described how manabot's policy consumes it.
# ---------------------------------------------------------------------------

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _deprecated_alias(old: str, new: Callable[_P, _R]) -> Callable[_P, _R]:
    def alias(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        warnings.warn(
            f"managym.{old} is deprecated and removed in 0.3.0; "
            f"use managym.{new.__name__}",
            DeprecationWarning,
            stacklevel=2,
        )
        return new(*args, **kwargs)

    alias.__name__ = old
    alias.__qualname__ = old
    alias.__doc__ = f"Deprecated alias of :func:`{new.__name__}`, removed in 0.3.0."
    return alias


learning_catalog = _deprecated_alias("learning_catalog", _managym.catalog_tokens)
learning_token_value = _deprecated_alias(
    "learning_token_value", _managym.catalog_token_value
)
learning_record_names = _deprecated_alias(
    "learning_record_names", _managym.record_vocabulary
)
