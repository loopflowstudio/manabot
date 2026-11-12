"""Manabot: training and evaluation for agents playing in the managym engine."""

from importlib.util import find_spec

# manabot <= 0.2.0 bundled the engine and its wheel RECORD owns
# ``managym/__init__.py``. Uninstalling it after the separate ``managym`` wheel
# is installed (pip's order when upgrading) deletes that file and leaves an
# unusable namespace package. Name the repair instead of failing deep inside an
# engine submodule; this checks the import spec without loading the engine.
_engine = find_spec("managym")
if _engine is not None and _engine.origin in (None, "namespace"):
    raise ImportError(
        "managym is missing its package files, usually because upgrading from "
        "manabot 0.2.0 removed them. Reinstall the engine at manabot's version: "
        "`pip install --force-reinstall 'managym==<version>'` "
        "(uv: `uv pip install --reinstall-package managym 'managym==<version>'`)."
    )
del _engine, find_spec
