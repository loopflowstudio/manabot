"""Release projects must retain runtime data and rebuild without the checkout."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tomllib
from types import ModuleType
from urllib.error import HTTPError

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"release_{name}", ROOT / "scripts/release" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_projects_preserve_native_paths_and_runtime_data(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    _load("prepare").prepare(stage)
    bot = stage
    config = tomllib.loads((bot / "pyproject.toml").read_text())
    assert "torch==2.10.0" in config["project"]["dependencies"]
    assert not any(
        dep.startswith("managym") for dep in config["project"]["dependencies"]
    )
    assert config["tool"]["maturin"]["module-name"] == "manabot.managym._managym"
    assert (bot / "managym/src/lib.rs").is_file()
    assert (bot / "manabot/managym/decision.py").is_file()
    assert (bot / "manabot/sim/match.py").is_file()
    assert (bot / "manabot/eval/cards.py").is_file()
    assert (bot / "manabot/experiment/checkpoint_evaluation.py").is_file()
    assert not (bot / "manabot/arena").exists()
    assert not (bot / "managym/pyproject.toml").exists()
    assert not (bot / "managym/libmanagym.dylib.dSYM").exists()
    assert (bot / "content/semantic/v1/generated/two_deck.ir.json").is_file()
    assert (bot / "academy/ui/src/lib/packs/elemental-world/v2/manifest.json").is_file()
    assert (bot / "protocol/experience-v1.schema.json").read_bytes() == (
        ROOT / "protocol/experience-v1.schema.json"
    ).read_bytes()
    assert (bot / "manabot/reports/time_cohort.html").is_file()
    assert (bot / "manabot/remote/guardian.sh").is_file()
    assert not (bot / "academy/ui/node_modules").exists()
    assert not (bot / "experiments").exists()
    source = json.loads((bot / ".manabot-source.json").read_text())
    assert len(source["commit"]) == 40
    assert "manabot/training/execution.py" in source["files"]
    assert "protocol/experience-v1.schema.json" in source["files"]
    for name, digest in source["files"].items():
        assert hashlib.sha256((bot / name).read_bytes()).hexdigest() == digest

    with pytest.raises(FileExistsError):
        _load("prepare").prepare(stage)


def test_release_versions_and_tags_agree() -> None:
    module = _load("check")
    module.check("v0.2.0", False)
    with pytest.raises(ValueError, match="expected tag"):
        module.check("v9.9.9", False)


@pytest.mark.parametrize("status", [404, 403, 503])
def test_registry_failures_only_allow_absent_projects(
    status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load("check")

    def fail(url: str, timeout: int) -> None:
        raise HTTPError(url, status, "registry response", None, None)

    monkeypatch.setattr(module, "urlopen", fail)
    if status == 404:
        module.check(None, True)
    else:
        with pytest.raises(HTTPError):
            module.check(None, True)
