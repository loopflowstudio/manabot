"""Compatibility alias for the separately distributed ``managym`` engine."""

from importlib import import_module
import sys

_engine = import_module("managym")
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
    sys.modules[f"{__name__}.{_name}"] = import_module(f"managym.{_name}")
