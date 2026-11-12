"""Release projects must retain runtime data and rebuild without the checkout."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tomllib
from types import ModuleType
from urllib.error import HTTPError
from urllib.request import Request
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
    config = tomllib.loads((stage / "pyproject.toml").read_text())
    for name in ("LICENSE", "protocol/experience-v1.schema.json"):
        assert (stage / name).read_bytes() == (ROOT / name).read_bytes(), name
    assert (
        config["project"]
        == tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    )
    assert (
        f"managym=={config['project']['version']}" in config["project"]["dependencies"]
    )
    assert config["build-system"]["build-backend"] == "hatchling.build"
    engine = stage / "managym"
    engine_config = tomllib.loads((engine / "pyproject.toml").read_text())
    assert engine_config["tool"]["maturin"]["module-name"] == "managym._managym"
    assert engine_config["tool"]["maturin"]["python-source"] == "."
    assert (engine / "managym/src/lib.rs").is_file()
    for crate in ("state", "cards"):
        assert (engine / "managym" / crate / "Cargo.toml").is_file()
        assert (engine / "managym" / crate / "src/lib.rs").is_file()
    assert (engine / "managym/decision.py").is_file()
    assert (engine / "content/semantic/current/setups.json").is_file()
    assert not (stage / "manabot/managym/decision.py").exists()
    assert (stage / "manabot/sim/match.py").is_file()
    assert (stage / "manabot/eval/cards.py").is_file()
    assert (stage / "manabot/experiment/checkpoint_evaluation.py").is_file()
    assert not (stage / "manabot/arena").exists()
    assert not (engine / "managym/libmanagym.dylib.dSYM").exists()
    assert (stage / "content/semantic/current/setups.json").is_file()
    assert (
        stage / "academy/ui/src/lib/packs/elemental-world/v2/manifest.json"
    ).is_file()
    assert (stage / "manabot/reports/time_cohort.html").is_file()
    assert not (stage / "academy/ui/node_modules").exists()
    assert not (stage / "experiments").exists()
    source = json.loads((stage / ".manabot-source.json").read_text())
    assert len(source["commit"]) == 40
    assert "manabot/training/execution.py" in source["files"]
    assert "managym/decision.py" in source["files"]
    assert "protocol/experience-v1.schema.json" in source["files"]
    for name, digest in source["files"].items():
        path = stage / "managym" / name if name.startswith("managym/") else stage / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest

    with pytest.raises(FileExistsError):
        _load("prepare").prepare(stage)


def test_current_release_metadata_is_consistent() -> None:
    _load("check").check(None)


def test_release_requires_matching_tag_and_notes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load("check")
    (tmp_path / "managym").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nversion = "9.9.9"\ndependencies = ["managym==9.9.9"]\n'
    )
    (tmp_path / "managym/Cargo.toml").write_text('[package]\nversion = "9.9.9"\n')
    (tmp_path / "managym/pyproject.toml").write_text('[project]\nversion = "9.9.9"\n')
    for crate in ("state", "cards"):
        path = tmp_path / "managym" / crate / "Cargo.toml"
        path.parent.mkdir()
        path.write_text('[package]\nversion = "9.9.9"\n')
    monkeypatch.setattr(module, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="release/v9.9.9/NOTES.md"):
        module.check(None)
    notes = tmp_path / "release/v9.9.9/NOTES.md"
    notes.parent.mkdir(parents=True)
    notes.write_text("\n")
    with pytest.raises(ValueError, match="missing or empty"):
        module.check(None)
    notes.write_text("# v9.9.9\n")
    module.check("v9.9.9")
    with pytest.raises(ValueError, match="expected tag v9.9.9"):
        module.check("v0.0.0")


def _artifacts(directory: Path) -> None:
    """Small archives exercise paired inventories without compiling the engine."""
    for project in ("managym", "manabot"):
        metadata = f"Name: {project}\nVersion: 0.2.0\nRequires-Python: >=3.12, <3.14\n"
        if project == "managym":
            metadata += (
                "Requires-Dist: numpy>=1.24.0\nRequires-Dist: pydantic>=2.12.0\n"
            )
            tags = [
                f"cp{python}-cp{python}-{target}"
                for python in ("312", "313")
                for target in ("manylinux_2_28_x86_64", "macosx_11_0_arm64")
            ]
        else:
            metadata += (
                "Requires-Dist: managym==0.2.0\n"
                "Requires-Dist: torch>=2.15.0 ; sys_platform == 'linux'\n"
                "Requires-Dist: torch>=2.15.0 ; sys_platform != 'linux'\n"
                "Requires-Dist: etude-sdk>=0.2.16\n"
            )
            tags = ["py3-none-any"]
        for tag in tags:
            with zipfile.ZipFile(
                directory / f"{project}-0.2.0-{tag}.whl", "w"
            ) as archive:
                archive.writestr(f"{project}-0.2.0.dist-info/METADATA", metadata)
                if project == "managym":
                    archive.writestr("managym/_managym.cpython.so", b"native fixture")
                    archive.writestr("managym/Cargo.lock", "# lockfile fixture\n")
                else:
                    archive.writestr(
                        "content/semantic/current/learning_schema.json", "{}"
                    )
        with tarfile.open(directory / f"{project}-0.2.0.tar.gz", "w:gz") as archive:
            for name, data in (
                ("PKG-INFO", metadata),
                ("content/semantic/current/learning_schema.json", "{}"),
            ):
                info = tarfile.TarInfo(f"{project}-0.2.0/{name}")
                info.size = len(data.encode())
                archive.addfile(info, io.BytesIO(data.encode()))


def test_release_requires_both_platforms_and_interpreters(tmp_path: Path) -> None:
    _artifacts(tmp_path)
    module = _load("validate")
    module.validate(tmp_path, True)
    next(tmp_path.glob("*cp313*whl")).unlink()
    with pytest.raises(ValueError, match="four wheels"):
        module.validate(tmp_path, True)


def test_engine_wheel_ships_the_lockfile_training_provenance_hashes(
    tmp_path: Path,
) -> None:
    _artifacts(tmp_path)
    wheel = next(tmp_path.glob("managym-*cp313*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    del files["managym/Cargo.lock"]
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    with pytest.raises(ValueError, match="provenance lockfile"):
        _load("validate").validate(tmp_path, True)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (b"Name: manabot", b"Name: managym"),
        (b"Version: 0.2.0", b"Version: 9.9.9"),
        (b">=3.12, <3.14", b">=3.12, <3.13"),
        (b"etude-sdk>=0.2.16", b"etude-sdk==0.2.16"),
    ],
)
def test_source_metadata_must_match_wheels(
    tmp_path: Path, before: bytes, after: bytes
) -> None:
    _artifacts(tmp_path)
    wheel = next(tmp_path.glob("manabot-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    name = "manabot-0.2.0.dist-info/METADATA"
    files[name] = files[name].replace(before, after)
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    with pytest.raises(ValueError, match="source and wheel metadata differ"):
        _load("validate").validate(tmp_path, True)


@pytest.mark.parametrize("status", [404, 403, 503])
def test_registry_failures_only_allow_absent_projects(
    status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load("complete")

    def fail(request: Request, timeout: int) -> None:
        raise HTTPError(request.full_url, status, "registry response", None, None)

    monkeypatch.setattr(module, "urlopen", fail)
    if status == 404:
        assert module.published_files("managym", "0.2.0") == {}
    else:
        with pytest.raises(HTTPError):
            module.published_files("managym", "0.2.0")


@pytest.mark.parametrize(
    "state", ["absent", "complete", "partial", "different", "extra"]
)
@pytest.mark.parametrize("project", ["managym", "manabot"])
def test_pypi_retry_requires_complete_identical_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str, project: str
) -> None:
    _artifacts(tmp_path)
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in tmp_path.glob(f"{project}-*")
    }
    if state == "absent":
        files = {}
    elif state == "partial":
        files.pop(next(iter(files)))
    elif state == "different":
        files[next(iter(files))] = "0" * 64
    elif state == "extra":
        files["unexpected.whl"] = "0" * 64
    module = _load("complete")
    monkeypatch.setattr(module, "published_files", lambda name, version: files)
    if state in {"absent", "complete"}:
        assert module.needs_upload(tmp_path, project, "0.2.0") is (state == "absent")
    else:
        with pytest.raises(ValueError, match="inspect"):
            module.needs_upload(tmp_path, project, "0.2.0")


@pytest.mark.parametrize("existing", [False, True])
def test_github_release_is_created_or_verified_without_republishing(
    monkeypatch: pytest.MonkeyPatch, existing: bool
) -> None:
    module = _load("complete")
    monkeypatch.setenv("GH_TOKEN", "fixture-token")
    notes = "Compatibility changes"
    body = (
        notes
        + "\n\nInstall [managym](https://pypi.org/project/managym/0.2.0/) or [manabot](https://pypi.org/project/manabot/0.2.0/) from PyPI.\n"
    )
    saved: dict[str, object] = (
        {
            "tag_name": "v0.2.0",
            "draft": False,
            "prerelease": False,
            "body": body,
            "html_url": "https://github.com/example/repo/releases/tag/v0.2.0",
        }
        if existing
        else {}
    )

    def service(request: Request) -> dict[str, object]:
        if request.get_method() == "POST":
            assert not saved
            saved.update(json.loads(request.data or b"{}"))
            saved["html_url"] = "https://github.com/example/repo/releases/tag/v0.2.0"
        if not saved:
            raise HTTPError(request.full_url, 404, "absent", None, None)
        return saved

    monkeypatch.setattr(module, "_json", service)
    assert (
        module.complete_github("example/repo", "v0.2.0", "abc", notes)
        == saved["html_url"]
    )
    assert saved["body"] == body
    assert (
        module.complete_github("example/repo", "v0.2.0", "abc", notes)
        == saved["html_url"]
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"draft": True},
        {"body": "no PyPI link"},
        {"body": "https://pypi.org/project/manabot/0.2.0/"},
        {"tag_name": "v9.9.9"},
        {"prerelease": True},
    ],
)
def test_incomplete_github_release_fails(
    monkeypatch: pytest.MonkeyPatch, bad: dict[str, object]
) -> None:
    module = _load("complete")
    monkeypatch.setenv("GH_TOKEN", "fixture-token")
    notes = "Compatibility changes"
    body = (
        notes
        + "\n\nInstall [managym](https://pypi.org/project/managym/0.2.0/) or [manabot](https://pypi.org/project/manabot/0.2.0/) from PyPI.\n"
    )
    release = {
        "tag_name": "v0.2.0",
        "draft": False,
        "prerelease": False,
        "body": body,
        "html_url": "url",
        **bad,
    }
    monkeypatch.setattr(module, "_json", lambda request: release)
    with pytest.raises(ValueError, match="GitHub Release"):
        module.complete_github("example/repo", "v0.2.0", "abc", notes)


@pytest.mark.parametrize(
    "path",
    [
        "managym/pyproject.toml",
        "managym/Cargo.toml",
        "managym/state/Cargo.toml",
        "managym/cards/Cargo.toml",
    ],
)
def test_each_package_version_must_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    for name in (
        "pyproject.toml",
        "managym/pyproject.toml",
        "managym/Cargo.toml",
        "managym/state/Cargo.toml",
        "managym/cards/Cargo.toml",
    ):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((ROOT / name).read_text())
    target = tmp_path / path
    version = tomllib.loads((tmp_path / "pyproject.toml").read_text())["project"][
        "version"
    ]
    target.write_text(
        target.read_text().replace(f'version = "{version}"', 'version = "9.9.9"')
    )
    module = _load("check")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="versions must match"):
        module.check(None)


def test_second_project_failure_prevents_all_upload_instructions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _artifacts(tmp_path)
    module = _load("complete")

    def registry(project: str, version: str) -> dict[str, str]:
        return {} if project == "managym" else {"partial.whl": "0" * 64}

    monkeypatch.setattr(module, "published_files", registry)
    output = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr("sys.argv", ["complete.py", str(tmp_path), "--tag", "v0.2.0"])
    with pytest.raises(ValueError, match="inspect"):
        module.main()
    assert not output.exists()


def test_completed_engine_can_resume_absent_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _artifacts(tmp_path)
    module = _load("complete")

    def registry(project: str, version: str) -> dict[str, str]:
        return (
            {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in tmp_path.glob("managym-*")
            }
            if project == "managym"
            else {}
        )

    monkeypatch.setattr(module, "published_files", registry)
    output = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr("sys.argv", ["complete.py", str(tmp_path), "--tag", "v0.2.0"])
    module.main()
    assert output.read_text() == "managym=false\nmanabot=true\n"


def test_post_upload_checks_wait_for_pypi_to_list_the_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _artifacts(tmp_path)
    module = _load("complete")
    engine = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in tmp_path.glob("managym-*")
    }
    calls: list[str] = []

    def registry(project: str, version: str) -> dict[str, str]:
        calls.append(project)
        # PyPI's JSON reports the just-uploaded engine as absent once.
        return engine if len(calls) > 1 else {}

    monkeypatch.setattr(module, "published_files", registry)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "complete.py",
            str(tmp_path),
            "--tag",
            "v0.2.0",
            "--require-engine",
            "--wait",
            "60",
        ],
    )
    module.main()
    assert calls == ["managym", "managym"]


def test_post_upload_checks_fail_once_the_wait_expires(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _artifacts(tmp_path)
    module = _load("complete")
    monkeypatch.setattr(module, "published_files", lambda project, version: {})
    monkeypatch.setattr(
        "sys.argv",
        ["complete.py", str(tmp_path), "--tag", "v0.2.0", "--require-engine"],
    )
    with pytest.raises(ValueError, match="engine publication is missing"):
        module.main()
