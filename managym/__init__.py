"""The rules engine and native public API.

The engine types are implemented by PyO3 and described in ``__init__.pyi``;
shared decision/protocol adapters live beside this module.
"""

import sys

from ._managym import *  # noqa: F403 — expose the native API described by the stub.
from ._managym import format

# PyO3 submodules are attributes, not importable modules; registering the
# native ``format`` lets ``import managym.format`` and ``from managym.format
# import Limited`` resolve to the same object as ``managym.format``.
sys.modules[f"{__name__}.format"] = format
