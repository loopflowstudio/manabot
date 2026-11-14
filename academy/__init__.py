"""Deprecated import name for managym.academy; removed in 0.4.0."""

from collections.abc import Sequence
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from importlib.util import spec_from_loader
import sys
from types import ModuleType
import warnings

warnings.warn(
    "academy is deprecated; use managym.academy. Removed in 0.4.0.",
    DeprecationWarning,
    stacklevel=2,
)


class _Alias(MetaPathFinder, Loader):
    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if fullname.startswith("academy."):
            return spec_from_loader(fullname, self)
        return None

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        return import_module("managym." + spec.name)

    def exec_module(self, module: ModuleType) -> None:
        pass


sys.meta_path.insert(0, _Alias())
