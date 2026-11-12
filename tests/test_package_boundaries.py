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
        from manabot.backends.torch.checkpoint import save_checkpoint
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
        import manabot.policy.checkpoint, manabot.policy.prepare, manabot.policy.protocol
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


# Modules outside the torch backend that still import torch. Training,
# evaluation, search and the model's components move behind a backend in later
# changes; the list only shrinks, and a module that stops importing torch
# leaves it.
_TORCH_NOT_YET_MOVED = frozenset(
    (
        "manabot/belief/agent.py",
        "manabot/belief/checkpoint.py",
        "manabot/belief/demo.py",
        "manabot/belief/encoding.py",
        "manabot/belief/likelihood.py",
        "manabot/belief/population.py",
        "manabot/belief/runtime.py",
        "manabot/belief/sampling.py",
        "manabot/belief/sampling_data.py",
        "manabot/belief/sampling_fit.py",
        "manabot/env/vector_env.py",
        "manabot/eval/evaluate.py",
        "manabot/eval/local.py",
        "manabot/eval/serving_benchmark.py",
        "manabot/experiment/attack_execution.py",
        "manabot/experiment/evaluation_queue.py",
        "manabot/experiment/numerics.py",
        "manabot/experiment/runner.py",
        "manabot/experiment/study.py",
        "manabot/infra/runtime.py",
        "manabot/infra/torch.py",
        "manabot/llm/transformers_runtime.py",
        "manabot/model/admission.py",
        "manabot/model/buckets.py",
        "manabot/model/catalog.py",
        "manabot/model/continuation.py",
        "manabot/model/facts.py",
        "manabot/model/inference.py",
        "manabot/model/name_candidates.py",
        "manabot/model/numerics.py",
        "manabot/model/policy_distribution.py",
        "manabot/model/recent_events.py",
        "manabot/model/records.py",
        "manabot/model/semantic_cards.py",
        "manabot/model/shared_effects.py",
        "manabot/model/token_stream.py",
        "manabot/recipes/device.py",
        "manabot/reports/belief.py",
        "manabot/reports/selection.py",
        "manabot/sim/guidance.py",
        "manabot/sim/native_game.py",
        "manabot/sim/player.py",
        "manabot/sim/rollout.py",
        "manabot/sim/search/local_update.py",
        "manabot/sim/search/puct.py",
        "manabot/sim/search/sampling.py",
        "manabot/sim/search/value.py",
        "manabot/sim/serving.py",
        "manabot/training/execution.py",
        "manabot/training/loops/minimal.py",
        "manabot/training/mps.py",
        "manabot/training/rollout/actions.py",
        "manabot/training/rollout/collection.py",
        "manabot/training/rollout/selection_data.py",
        "manabot/training/state/admission.py",
        "manabot/training/state/incidents.py",
        "manabot/training/state/learning_state.py",
        "manabot/training/state/recovery.py",
        "manabot/training/updates/damped_policy.py",
        "manabot/training/updates/health.py",
        "manabot/training/updates/learner.py",
        "manabot/training/updates/ppo.py",
        "manabot/training/updates/references.py",
        "manabot/training/updates/selection.py",
        "manabot/verify/action_cases.py",
        "manabot/verify/checkpoint_scenarios.py",
        "manabot/verify/choice_corpus.py",
        "manabot/verify/competency.py",
        "manabot/verify/decision_profile.py",
        "manabot/verify/learning_regression.py",
        "manabot/verify/llm_game.py",
        "manabot/verify/llm_opponent.py",
        "manabot/verify/record_coverage.py",
        "manabot/verify/util.py",
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
_GENERIC_ALLOWED = ("manabot.backends.torch.rl", "manabot.infra.profiler")


def test_generic_learning_pieces_import_only_themselves() -> None:
    violations = [
        f"{relative}:{line}: {module}"
        for relative in _GENERIC
        for line, module in _imported_modules(
            ast.parse((REPO_ROOT / relative).read_text(), relative)
        )
        if module.split(".")[0] in {"managym", "academy", "etude"}
        or (module.startswith("manabot") and module not in _GENERIC_ALLOWED)
    ]
    # The learner names the player's network and its precision in annotations
    # only (under TYPE_CHECKING); the runtime probe below holds it to that.
    assert violations == [
        "manabot/backends/torch/learner.py:25: manabot.backends.torch.policy",
        "manabot/backends/torch/learner.py:26: manabot.training.regime",
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
