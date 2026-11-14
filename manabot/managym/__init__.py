"""Compatibility alias for the separately distributed ``managym`` engine."""

from importlib import import_module
from importlib.util import find_spec
import sys

# manabot <= 0.2.1 owned ``managym/__init__.py``. Upgrading can uninstall
# that file after the separate engine wheel is installed, leaving a namespace
# package. Diagnose this only when the engine is requested: offline report
# imports must not even probe the native runtime.
_spec = find_spec("managym")
if _spec is not None and _spec.origin in (None, "namespace"):
    raise ImportError(
        "managym is missing its package files, usually because upgrading from "
        "manabot 0.2.0 or 0.2.1 removed them. Reinstall the engine at manabot's version: "
        "`uv pip install --reinstall-package managym 'managym==<version>'`."
    )

_engine = import_module("managym")
sys.modules[__name__] = _engine
for _name in (
    "_managym",
    "decision",
    "decision_capture",
    "enums",
    "experience",
    "experience_projection",
    "format",
    "possible_worlds",
):
    sys.modules[f"{__name__}.{_name}"] = import_module(f"managym.{_name}")
