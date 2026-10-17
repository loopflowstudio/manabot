"""Cached GPU setup must reject altered dependencies before invoking installers."""

from dataclasses import dataclass, replace
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from typing import Literal
from urllib.request import Request

import pytest

from experiments.runners import (
    client_install_canary,
    grouped_grid,
    locked_gpu_setup as setup,
)
from experiments.runners.gpu_job import Deadlines


@dataclass(frozen=True)
class Fixture:
    bundle: setup.Bundle
    archive: Path
    source: Path


def fixture(
    root: Path,
    change: Literal["none", "wheel", "extra", "requirements", "unlocked"] = "none",
) -> Fixture:
    root.mkdir(parents=True, exist_ok=True)
    source = root / "source"
    source.mkdir()
    wheel = b"exact locked distribution bytes"
    digest = hashlib.sha256(wheel).hexdigest()
    filename = "example-1.0-py3-none-any.whl"
    lock = (
        'version = 1\n[[package]]\nname = "example"\nversion = "1.0"\n'
        f'wheels = [{{url = "https://example.test/{filename}", '
        f'hash = "sha256:{digest}"}}]\n'
    ).encode()
    (source / "uv.lock").write_bytes(lock)
    manifest = json.dumps(
        {
            "format": 1,
            "python": "3.12",
            "platform": "linux-x86_64",
            "allowed_lock_sha256": [hashlib.sha256(lock).hexdigest()],
            "packages": [
                {
                    "name": "different" if change == "unlocked" else "example",
                    "version": "1.0",
                    "filename": filename,
                    "sha256": digest,
                    "bytes": len(wheel),
                }
            ],
        }
    ).encode()
    entries = {
        "manifest.json": manifest,
        "requirements.txt": f"example==1.0 --hash=sha256:{digest}\n".encode(),
        "wheels/" + filename: wheel,
    }
    if change == "wheel":
        entries["wheels/" + filename] = b"substituted distribution"
    elif change == "extra":
        entries["../outside"] = b"not admitted"
    elif change == "requirements":
        entries["requirements.txt"] = b"example==2.0\n"
    archive = root / "bundle.tar"
    with tarfile.open(archive, "w") as output:
        for name, data in entries.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            output.addfile(member, io.BytesIO(data))
    bundle = setup.Bundle(
        "https://example.test/bundle.tar",
        hashlib.sha256(archive.read_bytes()).hexdigest(),
        archive.stat().st_size,
        hashlib.sha256(manifest).hexdigest(),
    )
    return Fixture(bundle, archive, source)


@pytest.mark.parametrize("change", ["wheel", "extra", "requirements", "unlocked"])
def test_rejects_changed_installable_content(
    tmp_path: Path,
    change: Literal["wheel", "extra", "requirements", "unlocked"],
) -> None:
    case = fixture(tmp_path, change)
    with pytest.raises(ValueError):
        setup.unpack_verified(
            case.bundle, case.archive, tmp_path / "verified", case.source
        )
    assert not (tmp_path / "outside").exists()


def test_source_lock_and_archive_are_both_binding(tmp_path: Path) -> None:
    case = fixture(tmp_path)
    with pytest.raises(ValueError, match="archive identity"):
        setup.unpack_verified(
            replace(case.bundle, sha256="0" * 64),
            case.archive,
            tmp_path / "verified",
            case.source,
        )
    with (case.source / "uv.lock").open("a") as stream:
        stream.write("# another source lock\n")
    with pytest.raises(ValueError, match="source lock"):
        setup.unpack_verified(
            case.bundle, case.archive, tmp_path / "verified", case.source
        )


class Response(io.BytesIO):
    def __init__(self, body: bytes, content_range: str) -> None:
        super().__init__(body)
        self.status = 206
        self.headers = {"Content-Range": content_range}


@pytest.mark.parametrize("fault", ["none", "range", "truncated", "content"])
def test_download_admits_only_exact_range_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    body = b"x" * (8 * 1024**2) + b"last-range"
    bundle = setup.Bundle(
        "https://example.test/object?secret=never-log",
        hashlib.sha256(body).hexdigest(),
        len(body),
        "0" * 64,
    )

    def open_range(request: Request, timeout: int) -> Response:
        assert timeout == 30
        value = request.get_header("Range")
        assert isinstance(value, str)
        start, end = (int(n) for n in value.removeprefix("bytes=").split("-"))
        data = body[start : end + 1]
        if fault == "truncated":
            data = data[:-1]
        elif fault == "content":
            data = b"z" + data[1:]
        return Response(
            data,
            "invalid" if fault == "range" else f"bytes {start}-{end}/{len(body)}",
        )

    monkeypatch.setattr(setup, "urlopen", open_range)
    target = tmp_path / "download.tar"
    if fault == "none":
        setup.download(bundle, target, tmp_path)
        assert target.read_bytes() == body
        progress = (tmp_path / "etude-result.json").read_text()
        assert json.loads(progress)["clocks"]["dependency_download_bytes"] == len(body)
        assert "secret" not in progress
    else:
        with pytest.raises(ValueError):
            setup.download(bundle, target, tmp_path)


@pytest.mark.parametrize("altered", [False, True])
def test_installs_offline_only_after_verification_then_runs_original_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, altered: bool
) -> None:
    case = fixture(tmp_path, "wheel" if altered else "none")
    output = tmp_path / "output"
    output.mkdir()
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"locked_dependencies": case.bundle.__dict__}))
    monkeypatch.chdir(case.source)
    monkeypatch.setenv("ETUDE_CONFIG", str(config))
    monkeypatch.setenv("ETUDE_OUTPUT_DIR", str(output))
    monkeypatch.setattr(setup.platform, "system", lambda: "Linux")
    monkeypatch.setattr(setup.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(setup.sys, "version_info", (3, 12, 0))
    scratch = tmp_path / "download"
    scratch.mkdir()
    monkeypatch.setattr(setup.tempfile, "mkdtemp", lambda **_: str(scratch))

    def download(bundle: setup.Bundle, target: Path, output: Path) -> None:
        assert bundle == case.bundle
        shutil.copyfile(case.archive, target)

    calls: list[list[str]] = []

    def run(argv: list[str], *, check: bool) -> subprocess.CompletedProcess[bytes]:
        assert check
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    def execvp(file: str, args: list[str]) -> None:
        assert file == "bash"
        assert not scratch.exists()
        calls.append(args)

    monkeypatch.setattr(setup, "download", download)
    monkeypatch.setattr(setup.subprocess, "run", run)
    monkeypatch.setattr(setup.os, "execvp", execvp)
    if altered:
        with pytest.raises(ValueError, match="wheel bytes"):
            setup.main()
        assert calls == []
    else:
        setup.main()
        assert len(calls) == 3
        assert calls[0][:2] == ["uv", "venv"]
        assert calls[1][:3] == ["uv", "pip", "install"]
        for option in ("--offline", "--no-index", "--no-deps", "--require-hashes"):
            assert option in calls[1]
        assert calls[2] == ["bash", "scripts/benchmarks/setup-worker-gpu.sh"]
        assert json.loads((output / "dependency-setup.json").read_text())["wheels"] == 1


@pytest.mark.parametrize("grouped", [False, True])
def test_cached_setup_stays_inside_existing_deadline_and_failure_stops_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, grouped: bool
) -> None:
    runner = grouped_grid if grouped else client_install_canary
    helper = tmp_path / "helper.py"
    commands: list[tuple[str, list[str], int]] = []

    class Worker:
        def __init__(self, output: Path, deadlines: Deadlines) -> None:
            self.output = output
            self.stopping = False

        def stop(self, signum: int, frame: object) -> None:
            self.stopping = True

        def phase(self, name: str, **details: object) -> None:
            pass

        def command(self, name: str, argv: list[str], seconds: int) -> int:
            commands.append((name, argv, seconds))
            return 124 if name == "setup" else 0

    monkeypatch.setattr(runner, "Worker", Worker)
    monkeypatch.setattr(runner.signal, "signal", lambda *_: None)
    monkeypatch.setattr(runner, "project_interpreter", lambda _: "/unused/python")
    monkeypatch.setenv("ETUDE_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("ETUDE_DEADLINE", "2026-10-17T20:00:00+00:00")
    argv = ["runner", "--locked-setup", str(helper)]
    if grouped:
        argv.extend(["--cell", "small-deep:256"])
    monkeypatch.setattr(runner.sys, "argv", argv)
    with pytest.raises(SystemExit) as caught:
        runner.main()
    assert caught.value.code == 124
    assert [name for name, _, _ in commands] == ["nvidia-smi", "setup"]
    assert commands[-1][1][1] == str(helper)
    assert commands[-1][2] == 1200
