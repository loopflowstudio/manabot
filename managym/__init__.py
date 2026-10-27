"""Compatibility import for the engine bundled as ``manabot.managym``."""

from importlib import import_module
import sys

_engine = import_module("manabot.managym")
sys.modules[__name__] = _engine
for _name in (
    "_managym",
    "decision",
    "decision_capture",
    "enums",
    "experience",
    "experience_projection",
    "possible_worlds",
):
    sys.modules[f"{__name__}.{_name}"] = import_module(f"manabot.managym.{_name}")
