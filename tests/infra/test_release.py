"""Release projects must retain runtime data and rebuild without the checkout."""

import importlib.util
from pathlib import Path
import subprocess
import tomllib
from types import ModuleType
from urllib.error import HTTPError
import zipfile

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
    native = stage / "managym"
    native_config = tomllib.loads((native / "pyproject.toml").read_text())
    assert native_config["tool"]["maturin"]["manifest-path"] == "managym/Cargo.toml"
    assert native_config["tool"]["maturin"]["bindings"] == "pyo3"
    assert (native / "managym/src/lib.rs").is_file()
    assert (native / "content/semantic/w11/generated/elemental_world.ir.json").is_file()
    assert not (native / "managym/libmanagym.dylib.dSYM").exists()
    bot = stage / "manabot"
    config = tomllib.loads((bot / "pyproject.toml").read_text())
    assert "torch==2.10.0" in config["project"]["dependencies"]
    assert "managym==0.2.0" in config["project"]["dependencies"]
    assert (bot / "content/semantic/v1/generated/two_deck.ir.json").is_file()
    assert (bot / "academy/ui/src/lib/packs/elemental-world/v2/manifest.json").is_file()
    assert (bot / "manabot/training/time_cohort.html").is_file()
    assert (bot / "manabot/remote/guardian.sh").is_file()
    assert not (bot / "academy/ui/node_modules").exists()
    assert not (bot / "experiments").exists()
    with pytest.raises(FileExistsError):
        _load("prepare").prepare(stage)


def test_release_versions_and_tags_agree() -> None:
    module = _load("check")
    module.check("v0.2.0", False)
    with pytest.raises(ValueError, match="expected tag"):
        module.check("v9.9.9", False)


def test_prepared_manabot_sdist_builds_complete_wheel(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    _load("prepare").prepare(stage)
    subprocess.run(
        [
            "uv",
            "build",
            "--no-sources",
            str(stage / "manabot"),
            "--out-dir",
            str(tmp_path / "dist"),
        ],
        check=True,
    )
    # uv build creates the wheel from its sdist; the staging checkout cannot
    # mask missing forced-includes in that source archive.
    wheel = next((tmp_path / "dist").glob("manabot-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        paths = archive.namelist()
        assert "content/semantic/w11/generated/elemental_world.ir.json" in paths
        assert "academy/ui/src/lib/packs/elemental-world/v2/manifest.json" in paths
        assert "manabot/remote/guardian.sh" in paths
        metadata = archive.read("manabot-0.2.0.dist-info/METADATA").decode()
        assert "Requires-Dist: torch==2.10.0" in metadata
        assert "Requires-Dist: managym==0.2.0" in metadata
        assert "Requires-Python: <3.13,>=3.12" in metadata
        assert "Editable" not in metadata


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
