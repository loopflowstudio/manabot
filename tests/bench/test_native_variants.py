"""Native experiment selection preserves loaded bytes and restores failed builds."""

import json
import os
from pathlib import Path
import subprocess
import zipfile

import pytest

from experiments.runners.native_variants import (
    NativeBuild,
    _source_commit,
    file_digest,
    install,
    prepare,
)


def test_job_environment_exposes_cargo_to_later_commands(tmp_path: Path) -> None:
    cargo_home = tmp_path / "cargo"
    binary = cargo_home / "bin/cargo"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    script = (
        Path(__file__).resolve().parents[2] / "scripts/benchmarks/worker-gpu-env.sh"
    )
    environment = dict(os.environ, PATH="/usr/bin:/bin", CARGO_HOME=str(cargo_home))
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; command -v cargo; bash -c "command -v cargo"',
            "native-build-test",
            str(script),
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.splitlines() == [str(binary), str(binary)]


def test_archive_build_identity_requires_no_git_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"source": {"commit": "a" * 40}}))
    monkeypatch.setenv("ETUDE_CONFIG", str(config))
    assert _source_commit() == "a" * 40
    config.write_text(json.dumps({"source": {"commit": "main"}}))
    with pytest.raises(ValueError, match="exact source commit"):
        _source_commit()


def test_selection_is_atomic_and_rejects_modified_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "variant"
    directory.mkdir()
    native = tmp_path / "managym"
    native.mkdir()
    name = "_managym.cpython-312-test.so"
    source, target = directory / name, native / name
    source.write_bytes(b"selected library")
    target.write_bytes(b"previous loaded library")
    build = NativeBuild(
        variant="borrowed",
        library=name,
        library_sha256=file_digest(source),
        wheel_sha256="unused",
        sources={},
        command=[],
    )
    with target.open("rb") as loaded:
        install(directory, build)
        assert loaded.read() == b"previous loaded library"
        assert target.read_bytes() == b"selected library"
    source.write_bytes(b"modified library")
    with pytest.raises(ValueError, match="differs from build receipt"):
        install(directory, build)
    assert target.read_bytes() == b"selected library"


def test_second_build_failure_restores_candidate_source_and_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = Path("managym/src/agent/vector_env.rs")
    source.parent.mkdir(parents=True)
    source.write_bytes(b"candidate source")
    for name in ("Cargo.toml", "Cargo.lock", "pyproject.toml"):
        Path("managym", name).write_text("fixture")
    library = Path("managym/_managym.cpython-312-test.so")
    library.write_bytes(b"installed library")
    reference = Path("experiments/data/worker-scaling/vector-env-e41edb26.rs")
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b"historical source")
    reference.with_suffix(".json").write_text(
        json.dumps({"sha256": file_digest(reference)})
    )
    calls = 0

    def build(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        if calls == 2:
            assert source.read_bytes() == b"historical source"
            raise subprocess.CalledProcessError(1, command)
        output = Path(command[-1])
        output.mkdir()
        with zipfile.ZipFile(output / "fixture.whl", "w") as wheel:
            wheel.writestr("managym/" + library.name, b"candidate build")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", build)
    with pytest.raises(subprocess.CalledProcessError):
        prepare(tmp_path / "builds")
    assert calls == 2
    assert source.read_bytes() == b"candidate source"
    assert library.read_bytes() == b"installed library"
