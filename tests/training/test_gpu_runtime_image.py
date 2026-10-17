"""A runtime image must fail closed before source installation or training."""

import hashlib
import json
from pathlib import Path

import pytest

from experiments.runners import client_install_canary, gpu_runtime_image
from experiments.runners.gpu_job import Deadlines
from scripts.benchmarks import prebuilt_gpu_setup as runtime
from tests.training.test_locked_gpu_setup import fixture


def prepared(tmp_path: Path) -> tuple[Path, Path, str]:
    case = fixture(tmp_path / "bundle")
    (case.source / "pyproject.toml").write_text('[project]\nname="example"\n')
    context = tmp_path / "context"
    context.mkdir()
    identity = gpu_runtime_image.prepare(
        case.bundle, case.archive, case.source, context
    )
    return case.source, context / "runtime.json", identity


def compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(runtime.sys, "version_info", (3, 12, 0))
    monkeypatch.setattr(runtime, "installed_packages", lambda: {"example": "1.0"})

    def version(command: list[str], *, text: bool) -> str:
        return {"uv": "uv 0.8.22 (test)", "rustc": "rustc 1.98.1 (test)"}[command[0]]

    monkeypatch.setattr(runtime.subprocess, "check_output", version)


def test_image_admits_exact_source_and_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, manifest, identity = prepared(tmp_path)
    compatible(monkeypatch)
    admitted = runtime.admit(source, manifest, identity)
    assert admitted["packages"][0]["name"] == "example"
    assert admitted["pyproject_sha256"] == runtime.digest(source / "pyproject.toml")
    assert (manifest.parent / "verified/wheels/example-1.0-py3-none-any.whl").exists()


@pytest.mark.parametrize(
    "fault",
    [
        "manifest",
        "lock",
        "project",
        "missing",
        "version",
        "arch",
        "python",
        "toolchain",
    ],
)
def test_incompatible_image_stops_before_environment_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    source, manifest, identity = prepared(tmp_path)
    compatible(monkeypatch)
    if fault == "manifest":
        manifest.write_text(manifest.read_text() + " ")
    elif fault in ("lock", "project"):
        path = source / ("uv.lock" if fault == "lock" else "pyproject.toml")
        path.write_text(path.read_text() + "# changed\n")
    elif fault in ("missing", "version"):
        monkeypatch.setattr(
            runtime,
            "installed_packages",
            lambda: {} if fault == "missing" else {"example": "2.0"},
        )
    elif fault == "arch":
        monkeypatch.setattr(runtime.platform, "machine", lambda: "aarch64")
    elif fault == "python":
        monkeypatch.setattr(runtime.sys, "version_info", (3, 13, 0))
    else:
        monkeypatch.setattr(
            runtime.subprocess, "check_output", lambda *a, **k: "uv 0.9.0"
        )
    monkeypatch.setattr(runtime, "ROOT", manifest.parent)
    monkeypatch.chdir(source)
    monkeypatch.setattr(runtime.sys, "argv", ["setup", "--identity", identity])

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("Admission failure reached an installer")

    monkeypatch.setattr(runtime.subprocess, "run", unexpected)
    with pytest.raises(ValueError):
        runtime.main()
    assert not (source / ".venv").exists()


def test_build_identity_changes_when_recipe_or_project_changes(tmp_path: Path) -> None:
    source, manifest, identity = prepared(tmp_path)
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == identity
    value = json.loads(manifest.read_text())
    assert len(value["recipe_sha256"]) == 64
    case = fixture(tmp_path / "second")
    (case.source / "pyproject.toml").write_text('[project]\nname="different"\n')
    context = tmp_path / "different"
    context.mkdir()
    other = gpu_runtime_image.prepare(case.bundle, case.archive, case.source, context)
    assert other != identity
    assert (
        json.loads((context / "runtime.json").read_text())["recipe_sha256"]
        != value["recipe_sha256"]
    )


def test_candidate_reuses_canary_deadline_and_failure_barrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[tuple[str, list[str], int]] = []

    class Worker:
        def __init__(self, output: Path, deadlines: Deadlines) -> None:
            self.stopping = False

        def stop(self, signum: int, frame: object) -> None:
            self.stopping = True

        def phase(self, name: str, **details: object) -> None:
            pass

        def command(self, name: str, argv: list[str], seconds: int) -> int:
            commands.append((name, argv, seconds))
            return 124 if name == "setup" else 0

    monkeypatch.setattr(client_install_canary, "Worker", Worker)
    monkeypatch.setattr(client_install_canary.signal, "signal", lambda *_: None)
    monkeypatch.setenv("ETUDE_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("ETUDE_DEADLINE", "2026-10-17T20:00:00+00:00")
    monkeypatch.setattr(
        client_install_canary.sys, "argv", ["canary", "--prebuilt-runtime", "a" * 64]
    )
    with pytest.raises(SystemExit) as caught:
        client_install_canary.main()
    assert caught.value.code == 124
    assert [name for name, _, _ in commands] == ["nvidia-smi", "setup"]
    assert commands[-1] == (
        "setup",
        [
            "/opt/manabot/runtime/bin/python",
            "/opt/manabot/prebuilt_gpu_setup.py",
            "--identity",
            "a" * 64,
        ],
        1200,
    )
