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
import importlib.util
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
    import manabot.sim.players
    import manabot.sim.replay
    import manabot.semantic.policy
    import manabot.sim.search.teacher1_evidence

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


def test_managym_legacy_import_is_the_bundled_engine() -> None:
    from managym.decision import Command as LegacyCommand

    from manabot import managym as engine
    from manabot.managym.decision import Command
    import managym as legacy

    assert legacy is engine
    assert LegacyCommand is Command
    assert Path(engine._managym.__file__).parent == REPO_ROOT / "manabot/managym"


class _RefuseTraining(importlib.abc.MetaPathFinder):
    """Simulation must remain usable without learning or run orchestration."""

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if any(
            fullname == root or fullname.startswith(root + ".")
            for root in (
                "manabot.training",
                "manabot.experiment",
                "manabot.reports",
                "manabot.eval",
            )
        ):
            raise ImportError(f"simulation must not import {fullname}")
        return None


def test_simulation_never_imports_training() -> None:
    violations: list[str] = []
    for path in sorted((REPO_ROOT / "manabot/sim").rglob("*.py")):
        package = ".".join(path.relative_to(REPO_ROOT).parent.parts)
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    module = importlib.util.resolve_name(
                        "." * node.level + module, package
                    )
                modules = [module, *(module + "." + alias.name for alias in node.names)]
            for module in modules:
                if any(
                    module == root or module.startswith(root + ".")
                    for root in (
                        "manabot.training",
                        "manabot.experiment",
                        "manabot.reports",
                        "manabot.eval",
                    )
                ):
                    violations.append(
                        f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {module}"
                    )
    assert violations == []


def test_simulation_imports_and_plays_without_training() -> None:
    # Exercise both eager imports and the deferred checkpoint-player factory in
    # a clean interpreter. Already-imported modules cannot hide a reverse edge.
    probe = textwrap.dedent(
        """
        import importlib
        from pathlib import Path
        import pkgutil
        import sys
        import tempfile
        from tests.test_package_boundaries import _RefuseTraining
        sys.meta_path.insert(0, _RefuseTraining())
        import manabot.sim
        for module in pkgutil.walk_packages(manabot.sim.__path__, 'manabot.sim.'):
            importlib.import_module(module.name)
        from manabot.env.env import Env
        from manabot.env.match import Match, Reward
        from manabot.env.observation import ObservationSpace
        from manabot.infra.config import ModelConfig, RewardConfig
        from manabot.model.actor_critic import ActorCritic
        from manabot.model.checkpoint import save_checkpoint
        from manabot.sim.search.flat_mc import make_player
        match, space = Match(), ObservationSpace()
        model = ActorCritic(space, ModelConfig())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'policy.pt'
            save_checkpoint(model, space, checkpoint, player_configs=match.to_rust())
            player, _ = make_player({'kind': 'checkpoint', 'path': str(checkpoint)}, 17)
            env = Env(match, space, Reward(RewardConfig()), seed=17)
            observation, _ = env.reset(seed=17)
            action = player.act(env, observation)
            assert observation['actions_valid'][action] > 0
        assert not any(name.startswith('manabot.training') for name in sys.modules)
        """
    )
    subprocess.run([sys.executable, "-c", probe], cwd=REPO_ROOT, check=True)


# Game, replay and supervisor processes under shared inference. They must not
# load Torch: a served game child would otherwise hold ~200 MB for nothing.
_TORCH_FREE_MODULES = (
    "manabot.sim.child",
    "manabot.sim.frames",
    "manabot.sim.match",
    "manabot.sim.native_game",
    "manabot.sim.replay",
    "manabot.sim.serving_protocol",
    "manabot.sim.supervision",
    "manabot.sim.threads",
    "manabot.verify.games",
    "manabot.verify.scenarios",
)


def test_evaluation_child_modules_import_without_torch() -> None:
    probe = textwrap.dedent(
        f"""
        import importlib
        import sys
        for name in {_TORCH_FREE_MODULES!r}:
            importlib.import_module(name)
        heavy = sorted(m for m in ("torch", "etude.train") if m in sys.modules)
        assert not heavy, heavy
        """
    )
    subprocess.run([sys.executable, "-c", probe], cwd=REPO_ROOT, check=True)


def test_retired_arena_package_is_absent() -> None:
    assert importlib.util.find_spec("manabot.arena") is None
    assert not (REPO_ROOT / "manabot" / "arena").exists()
