"""Training and Academy depend on the engine without importing each other."""

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

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_shared_manabot_imports_use_public_names() -> None:
    """A helper used by another production module is a public contract."""
    violations: list[str] = []
    for directory in ("manabot", "scripts", "experiments"):
        for path in sorted((REPO_ROOT / directory).rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(), str(path))):
                if not isinstance(node, ast.ImportFrom):
                    continue
                module = node.module or ""
                if not module.startswith("manabot.") and not (
                    node.level and path.is_relative_to(REPO_ROOT / "manabot")
                ):
                    continue
                for alias in node.names:
                    if alias.name.startswith("_"):
                        violations.append(
                            f"{path.relative_to(REPO_ROOT)}:{node.lineno}: "
                            f"{'.' * node.level}{module}.{alias.name}"
                        )
    assert violations == []


def _imported_modules(tree: ast.Module) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.lineno, node.module))
            if node.module == "manabot" and any(
                alias.name == "backends" for alias in node.names
            ):
                found.append((node.lineno, "manabot.backends"))
    return found


def test_manabot_never_imports_academy() -> None:
    violations = [
        f"{path.relative_to(REPO_ROOT)}:{line}: {module}"
        for path in sorted((REPO_ROOT / "manabot").rglob("*.py"))
        for line, module in _imported_modules(ast.parse(path.read_text(), str(path)))
        if module == "academy" or module.startswith(("academy.", "managym.academy"))
    ]
    assert violations == []


class _RefuseAcademy(importlib.abc.MetaPathFinder):
    """Fail any attempt to import managym.academy, as if it were not installed."""

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname == "academy" or fullname.startswith(
            ("academy.", "managym.academy")
        ):
            raise ImportError(f"manabot must not import {fullname}")
        return None


_INDEPENDENCE_PROBE = textwrap.dedent(
    """
    import sys

    from tests.test_package_boundaries import _RefuseAcademy

    sys.meta_path.insert(0, _RefuseAcademy())
    import manabot.sim.players
    import manabot.sim.replay

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


def test_manabot_compatibility_import_is_the_separate_engine() -> None:
    from manabot import managym as compatibility
    from manabot.managym.decision import Command as CompatibilityCommand
    import managym
    from managym.decision import Command

    assert compatibility is managym
    assert CompatibilityCommand is Command
    assert Path(managym._managym.__file__).parent == REPO_ROOT / "managym"


def test_engine_compatibility_import_diagnoses_incomplete_upgrade(
    tmp_path: Path,
) -> None:
    # pip can remove the old wheel's shared __init__.py after installing the
    # new engine. Reproduce that namespace package without installed packages.
    for relative in ("manabot/__init__.py", "manabot/managym/__init__.py"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((REPO_ROOT / relative).read_text())
    (tmp_path / "managym").mkdir()
    probe = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        import manabot
        assert 'managym' not in sys.modules
        try:
            from manabot import managym
        except ImportError as error:
            assert 'managym is missing its package files' in str(error), error
            assert 'uv pip install --reinstall-package managym' in str(error), error
        else:
            raise AssertionError('Incomplete engine installation was accepted')
        """
    )
    subprocess.run(
        [sys.executable, "-I", "-S", "-c", probe],
        cwd=tmp_path,
        check=True,
        timeout=30,
    )


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
        from manabot.backends.torch.policy import ActorCritic
        from manabot.backends import resolve
        from manabot.backends.torch.checkpoint import save_checkpoint
        from manabot.sim.search.flat_mc import make_player
        match, space = Match(), ObservationSpace()
        model = ActorCritic(space, ModelConfig())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'policy.pt'
            save_checkpoint(model, space, checkpoint, player_configs=match.to_rust())
            player, _ = make_player({'kind': 'checkpoint', 'path': str(checkpoint)}, 17, backend=resolve("torch"))
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
    "manabot.sim.table_bots",
    "managym.academy.launch",
    "manabot.sim.supervision",
    "manabot.sim.threads",
    "manabot.eval.game",
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
        # Etude's Agents declaration imports its own training record package;
        # game-side builders must still load no array framework or Manabot trainer.
        import manabot.eval.agents
        assert "torch" not in sys.modules
        assert "manabot.training" not in sys.modules
        """
    )
    subprocess.run([sys.executable, "-c", probe], cwd=REPO_ROOT, check=True)


def test_retired_arena_package_is_absent() -> None:
    assert importlib.util.find_spec("manabot.arena") is None
    assert not (REPO_ROOT / "manabot" / "arena").exists()


class _RefuseEngineClients(_RefuseAcademy):
    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname.split(".")[0] in {"manabot", "torch", "gymnasium", "etude"}:
            raise ImportError(fullname)
        return super().find_spec(fullname, path, target)


def test_engine_imports_without_training_or_academy() -> None:
    probe = textwrap.dedent(
        """
        import importlib
        import pkgutil
        import sys
        from tests.test_package_boundaries import _RefuseEngineClients
        sys.meta_path.insert(0, _RefuseEngineClients())
        import managym
        for module in pkgutil.iter_modules(managym.__path__, 'managym.'):
            if module.name != 'managym.academy':
                importlib.import_module(module.name)
        assert 'manabot' not in sys.modules
        """
    )
    subprocess.run([sys.executable, "-c", probe], cwd=REPO_ROOT, check=True)


_FRAMEWORKS = ("torch", "mlx", "jax", "jaxlib", "flax")
_BACKENDS = REPO_ROOT / "manabot/backends"


def _manabot_imports() -> list[tuple[Path, int, str]]:
    return [
        (path, line, module)
        for path in sorted((REPO_ROOT / "manabot").rglob("*.py"))
        for line, module in _imported_modules(ast.parse(path.read_text(), str(path)))
    ]


def test_policy_contract_imports_no_array_framework() -> None:
    policy = REPO_ROOT / "manabot/policy"
    violations = [
        f"{path.relative_to(REPO_ROOT)}:{line}: {module}"
        for path, line, module in _manabot_imports()
        if policy in path.parents
        and (
            module.split(".")[0] in _FRAMEWORKS
            or module == "manabot.backends"
            or module.startswith("manabot.backends.")
        )
    ]
    assert violations == []
    # Transitively too: the shared preparation and container load no framework.
    probe = textwrap.dedent(
        f"""
        import sys
        import manabot.policy.admission, manabot.policy.checkpoint, manabot.policy.prepare, manabot.policy.protocol
        loaded = sorted(m for m in sys.modules if m.split(".")[0] in {_FRAMEWORKS!r})
        assert not loaded, loaded
        """
    )
    subprocess.run([sys.executable, "-c", probe], check=True, cwd=REPO_ROOT)


def test_backends_never_import_each_other() -> None:
    violations = [
        f"{path.relative_to(REPO_ROOT)}:{line}: {module}"
        for path, line, module in _manabot_imports()
        if _BACKENDS in path.parents
        and path.parent != _BACKENDS
        and module.startswith("manabot.backends.")
        and module.split(".")[2] != path.relative_to(_BACKENDS).parts[0]
    ]
    assert violations == []


def test_only_the_mlx_backend_imports_mlx() -> None:
    # MLX is an optional Apple Silicon extra: the PyTorch path, training and
    # every other module import neither it nor its backend, eagerly or lazily.
    # Outside `manabot/backends/`, only the torch backend is imported.
    mlx = _BACKENDS / "mlx"
    violations = [
        f"{path.relative_to(REPO_ROOT)}:{line}: {module}"
        for path, line, module in _manabot_imports()
        if (module.split(".")[0] == "mlx" and mlx not in path.parents)
        or (
            module.startswith("manabot.backends.")
            and module.split(".")[2] != "torch"
            and _BACKENDS not in path.parents
        )
    ]
    assert violations == []


# The Etude self-play loops are still written against Torch; they choose it
# themselves until they take an injected backend.
_TORCH_LOOPS = frozenset(
    ("manabot/training/loops/minimal.py", "manabot/training/loops/self_play.py")
)

# Modules outside the torch backend that still import torch. Training,
# evaluation, search and the model's components move behind a backend in later
# changes; the list only shrinks, and a module that stops importing torch
# leaves it.
_TORCH_NOT_YET_MOVED = frozenset(
    (
        "manabot/backends/parity.py",
        "manabot/belief/population.py",
        "manabot/env/vector_env.py",
        "manabot/eval/evaluate.py",
        "manabot/eval/local.py",
        "manabot/eval/agent_loader.py",
        "manabot/experiment/numerics.py",
        "manabot/infra/torch.py",
        "manabot/llm/transformers_runtime.py",
        "manabot/sim/qwen_provider.py",
        "manabot/reports/belief.py",
        "manabot/sim/native_game.py",
        *_TORCH_LOOPS,
        "manabot/training/mps.py",
        "manabot/verify/action_cases.py",
        "manabot/verify/choice_corpus.py",
    )
)


def test_only_listed_modules_import_torch_outside_its_backend() -> None:
    torch_backend = _BACKENDS / "torch"
    importing = {
        str(path.relative_to(REPO_ROOT))
        for path, _, module in _manabot_imports()
        if module.split(".")[0] == "torch" and torch_backend not in path.parents
    }
    assert sorted(importing - _TORCH_NOT_YET_MOVED) == []
    assert sorted(_TORCH_NOT_YET_MOVED - importing) == []


# The standard RL functions: a deck builder or any other role reuses them
# unchanged, so they never reach the game, its observations, the player model,
# collection or the engine.
_GENERIC = (
    "manabot/backends/torch/rl.py",
    "manabot/backends/torch/learner.py",
    "manabot/backends/torch/arrays.py",
)
_GENERIC_ALLOWED = (
    "manabot.backends.torch.rl",
    "manabot.backends.torch.arrays",
    "manabot.infra.profiler",
    "manabot.policy.returns",
)


def test_generic_learning_pieces_import_only_themselves() -> None:
    violations = [
        f"{relative}: {module}"
        for relative in _GENERIC
        for _, module in _imported_modules(
            ast.parse((REPO_ROOT / relative).read_text(), relative)
        )
        if module.split(".")[0] in {"managym", "academy", "etude"}
        or (module.startswith("manabot") and module not in _GENERIC_ALLOWED)
    ]
    # Role-specific types appear only in annotations (under TYPE_CHECKING);
    # the runtime probe below holds both the learner and array transfer to that.
    assert violations == [
        "manabot/backends/torch/learner.py: manabot.backends.torch.policy",
        "manabot/backends/torch/learner.py: manabot.policy.checkpoint",
        "manabot/backends/torch/learner.py: manabot.training.learning",
        "manabot/backends/torch/arrays.py: manabot.policy.prepare",
    ]


class _RefuseRoles(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        root = fullname.split(".")[0]
        allowed = {
            "manabot",
            "manabot.backends",
            "manabot.backends.torch",
            "manabot.infra",
            "manabot.policy",
            *_GENERIC_ALLOWED,
        }
        if root in {"managym", "academy", "etude"} or (
            root == "manabot"
            and fullname not in allowed
            and fullname.replace(".", "/") + ".py" not in _GENERIC
        ):
            raise ImportError(f"generic learning pieces must not import {fullname}")
        return None


def test_generic_learning_pieces_import_without_the_player() -> None:
    probe = textwrap.dedent(
        """
        import importlib
        import sys
        from tests.test_package_boundaries import _GENERIC, _RefuseRoles
        sys.meta_path.insert(0, _RefuseRoles())
        for path in _GENERIC:
            importlib.import_module(path.removesuffix(".py").replace("/", "."))
        """
    )
    subprocess.run([sys.executable, "-c", probe], cwd=REPO_ROOT, check=True)


# The updates are written once and take the backend module as an argument,
# typed by the Protocols in `manabot.training.updates.backend`, so they import
# neither a framework nor a backend, not even for annotations.
_WRITTEN_ONCE = (
    "manabot/training/updates/backend.py",
    "manabot/training/updates/damped_policy.py",
    "manabot/training/updates/ppo.py",
)


def test_updates_take_the_backend_as_an_argument() -> None:
    violations = [
        f"{relative}: {module}"
        for relative in _WRITTEN_ONCE
        for _, module in _imported_modules(
            ast.parse((REPO_ROOT / relative).read_text(), relative)
        )
        if module.split(".")[0] in _FRAMEWORKS or module.startswith("manabot.backends")
    ]
    assert violations == []


def test_backend_resolver_selects_configured_backends_and_names_missing_extras(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from manabot.backends import resolve

    assert resolve("torch").name == "torch"
    assert resolve("torch", precision="float32").name == "torch"
    with pytest.raises(ValueError, match="only for the MLX backend"):
        resolve("torch", precision="bfloat16")
    with pytest.raises(ValueError, match="available backends: torch, mlx"):
        resolve("unknown")
    original = importlib.import_module

    def without_framework(name: str, package: str | None = None) -> ModuleType:
        if name == "mlx.core":
            raise ImportError("fixture: unavailable framework")
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", without_framework)
    with pytest.raises(ImportError, match="'mlx' extra"):
        resolve("mlx")


# CLI and child-process entry points choose the backend before passing its
# capabilities to library code. Mixed modules permit only named entry points.
_BACKEND_ROOTS = frozenset(
    (
        "manabot/sim/table_bots.py",
        "manabot/sim/policy_provider.py",
        "manabot/cli.py",
        "manabot/eval/cards.py",
        "manabot/eval/agent_loader.py",
        "manabot/eval/evaluate.py",
        "manabot/eval/local.py",
        "manabot/sim/native_game.py",
        "manabot/sim/sim.py",
    )
)
# Only these actual CLI/child entry-point scopes choose a backend in mixed modules.
_BACKEND_ENTRYPOINTS = {
    "manabot/belief/demo.py": {"main"},
    "manabot/recipes/baseline.py": {"evaluate"},
    "manabot/recipes/belief_sampler.py": {"train"},
    "manabot/reports/belief.py": {"main"},
    "manabot/verify/checkpoint_scenarios.py": {"main"},
    "manabot/verify/competency.py": {"_scenario_worker", "_micro_worker", "main"},
    "manabot/verify/decision_profile.py": {"main"},
    "manabot/verify/learning_regression.py": {"main"},
    "manabot/verify/llm_game.py": {"main"},
    "manabot/verify/record_coverage.py": {"main"},
    "manabot/verify/decision_memory.py": {"main"},
}


def _backend_violations(path: str, source: str) -> list[int]:
    violations: list[int] = []

    def visit(node: ast.AST, scope: str) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope = f"{scope}.{node.name}" if scope else node.name
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports = _imported_modules(ast.Module(body=[node], type_ignores=[]))
            for line, module in imports:
                if module == "manabot.backends" or module.startswith(
                    "manabot.backends."
                ):
                    if (
                        path not in _BACKEND_ROOTS | _TORCH_LOOPS
                        and scope not in _BACKEND_ENTRYPOINTS.get(path, set())
                    ):
                        violations.append(line)
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    visit(ast.parse(source), "")
    return violations


def test_only_composition_roots_choose_a_backend() -> None:
    violations = [
        f"{path.relative_to(REPO_ROOT)}:{line}"
        for package in ("manabot", "academy")
        for path in (REPO_ROOT / package).rglob("*.py")
        if _BACKENDS not in path.parents
        for line in _backend_violations(
            str(path.relative_to(REPO_ROOT)), path.read_text()
        )
    ]
    assert violations == []


def test_mixed_cli_does_not_make_library_functions_backend_roots() -> None:
    path = "manabot/verify/competency.py"
    assert (
        _backend_violations(
            path, "def main():\n    from manabot.backends import resolve\n"
        )
        == []
    )
    assert _backend_violations(
        path, "def helper():\n    from manabot.backends import resolve\n"
    ) == [2]
    assert _backend_violations(
        path,
        "def main():\n    def helper():\n        from manabot.backends import resolve\n",
    ) == [3]


def test_importing_the_resolver_loads_no_framework() -> None:
    """Roots import `resolve` at module top; only calling it loads a framework."""
    probe = textwrap.dedent(
        f"""
        import sys
        from manabot.backends import resolve
        loaded = sorted(m for m in sys.modules if m.split(".")[0] in {_FRAMEWORKS!r})
        assert not loaded, loaded
        """
    )
    subprocess.run([sys.executable, "-c", probe], check=True, cwd=REPO_ROOT)


def test_engine_has_no_training_imports() -> None:
    violations: list[str] = []
    for path in (REPO_ROOT / "managym").rglob("*.py"):
        if "target" in path.parts or "tests" in path.parts:
            continue
        for line, module in _imported_modules(ast.parse(path.read_text())):
            root = module.split(".")[0]
            if root in {"manabot", "torch", "etude", "gymnasium"} or (
                root in {"fastapi", "starlette", "uvicorn"}
                and "academy" not in path.parts
                and path.name != "cli.py"
            ):
                violations.append(f"{path}:{line}: {module}")
    assert not violations


def test_academy_without_training_provider() -> None:
    probe = """
import importlib.abc
import sys
from unittest.mock import patch
class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'manabot', 'torch', 'etude', 'gymnasium'}:
            raise ImportError(fullname)
sys.meta_path.insert(0, Refuse())
with patch('managym.academy.extensions.entry_points', return_value=[]):
    from managym.academy.extensions import bot_extension
    from managym.academy.server import _parse_game_config
    from managym.academy.villain import build_villain_policy
    import managym
    assert bot_extension() is None
    config = _parse_game_config({'hero_deck': 'interactive', 'villain_deck': 'interactive', 'villain_type': 'search', 'villain_sims': 1, 'seed': 4})
    engine = managym.Env(seed=4)
    observation, _ = engine.reset(config.to_rust())
    policy = build_villain_policy(config)
    observation, *_ = engine.step(policy(engine, observation))
    assert observation.action_space.actions or observation.game_over
"""
    subprocess.run([sys.executable, "-c", probe], cwd=REPO_ROOT, check=True)


def test_deprecated_academy_alias_keeps_module_identity() -> None:
    probe = """
import warnings
with warnings.catch_warnings(record=True) as seen:
    warnings.simplefilter('always')
    import academy.server as old
    import managym.academy.server as current
assert old is current
assert any('0.4.0' in str(row.message) for row in seen)
"""
    subprocess.run([sys.executable, "-c", probe], cwd=REPO_ROOT, check=True)


_ROLLOUT_IMPORTS = tuple(
    ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)
    for path in sorted((REPO_ROOT / "manabot/training/rollout").rglob("*.py"))
)
_UPDATE_IMPORTS = (
    "manabot.sim.search.sampling",
    "manabot.training.updates.ppo",
    "manabot.training.updates.damped_policy",
)


@pytest.mark.parametrize(
    "modules",
    [(module,) for module in (*_ROLLOUT_IMPORTS, *_UPDATE_IMPORTS)]
    + [(*_ROLLOUT_IMPORTS, *_UPDATE_IMPORTS)],
)
def test_rollout_and_update_imports_without_torch(modules: tuple[str, ...]) -> None:
    probe = """
import importlib
import importlib.abc
import sys

class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'torch':
            raise AssertionError('Torch import: ' + fullname)

sys.meta_path.insert(0, BlockTorch())
for module in sys.argv[1:]:
    importlib.import_module(module)
assert 'torch' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, *modules],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
