"""Manabot's bundled rules engine and native public API.

The engine types are implemented by PyO3 and described in ``__init__.pyi``;
shared decision/protocol adapters live beside this module.
"""

from ._managym import *  # noqa: F403 — expose the native API described by the stub.
