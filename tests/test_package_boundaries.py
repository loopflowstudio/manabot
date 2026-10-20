"""Package dependencies flow academy -> manabot -> managym, never the reverse.

Academy is a client of manabot: it may import manabot and managym, while
manabot imports only managym. The static check walks every manabot module,
including imports nested inside functions; the runtime check imports manabot
entry points with every academy import refused.
"""

import ast
from collections.abc import Sequence
import importlib.abc
import importlib.machinery
from pathlib import Path
import subprocess
import sys
import textwrap
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]


def _imported_modules(tree: ast.Module) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.lineno, node.module))
    return found


def test_manabot_never_imports_academy() -> None:
    violations = [
        f"{path.relative_to(REPO_ROOT)}:{line}: {module}"
        for path in sorted((REPO_ROOT / "manabot").rglob("*.py"))
        for line, module in _imported_modules(ast.parse(path.read_text(), str(path)))
        if module == "academy" or module.startswith("academy.")
    ]
    assert violations == []


class _RefuseAcademy(importlib.abc.MetaPathFinder):
    """Fail any attempt to import academy, as if it were not installed."""

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname == "academy" or fullname.startswith("academy."):
            raise ImportError(f"manabot must not import {fullname}")
        return None


_INDEPENDENCE_PROBE = textwrap.dedent(
    """
    import sys

    from tests.test_package_boundaries import _RefuseAcademy

    sys.meta_path.insert(0, _RefuseAcademy())
    import manabot.arena.players
    import manabot.arena.replay
    import manabot.semantic.policy
    import manabot.sim.teacher1_evidence

    assert not any(name.split(".")[0] == "academy" for name in sys.modules)
    """
)


def test_manabot_entry_points_import_without_academy() -> None:
    # A fresh interpreter, so modules other tests already imported cannot mask
    # an academy import.
    subprocess.run(
        [sys.executable, "-c", _INDEPENDENCE_PROBE],
        cwd=REPO_ROOT,
        check=True,
    )
