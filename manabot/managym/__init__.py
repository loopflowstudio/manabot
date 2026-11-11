"""Manabot's bundled rules engine and native public API.

The engine types are implemented by PyO3 and described in ``__init__.pyi``;
shared decision/protocol adapters live beside this module.
"""

import importlib.util as _importlib_util
import os as _os
from pathlib import Path as _Path
import sys as _sys

# Process-start selection keeps experimental builds separate from frozen runs.
if _extension := _os.environ.get("MANABOT_MANAGYM_EXTENSION"):
    _path = _Path(_extension).resolve(strict=True)
    _name = f"{__name__}._managym"
    _spec = _importlib_util.spec_from_file_location(_name, _path)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"cannot load selected managym extension: {_path}")
    _module = _importlib_util.module_from_spec(_spec)
    _sys.modules[_name] = _module
    _spec.loader.exec_module(_module)
    _managym = _module

from ._managym import *  # noqa: F403 — expose the native API described by the stub.
