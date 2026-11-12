"""The PyTorch backend: the reference every other backend is compared with.

The updates take this module as their `backend` argument and reach its files
as attributes (`backend.rl`, `backend.learner`, `backend.arrays`,
`backend.rollout`), which load on first use.
"""

import importlib
from types import ModuleType

_FILES = frozenset({"arrays", "checkpoint", "learner", "policy", "rl", "rollout"})


def __getattr__(name: str) -> ModuleType:
    if name in _FILES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
