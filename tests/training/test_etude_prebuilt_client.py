"""Prebuilt startup preserves client identity and the original failure boundary."""

import io
from pathlib import Path
import subprocess
import zipfile

from etude.launch.providers.package import program_archive
import pytest

from experiments.runners import (
    etude_client_install as installer,
    etude_prebuilt_client as adapter,
)


def client() -> installer.ClientBundle:
    source = program_archive()
    wheel = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(source)) as original,
        zipfile.ZipFile(wheel, "w") as built,
    ):
        for name in original.namelist():
            if name.endswith(".py"):
                built.writestr(name, original.read(name))
        for name in ("pyproject.toml", "uv.lock"):
            built.writestr("etude/_package/" + name, original.read(name))
        info = f"etude_sdk-{installer.SDK_VERSION}.dist-info/"
        built.writestr(
            info + "METADATA", f"Name: etude-sdk\nVersion: {installer.SDK_VERSION}\n"
        )
        built.writestr(info + "WHEEL", "Tag: py3-none-any\n")
    return installer.ClientBundle(source, wheel.getvalue())


def test_bundle_rejects_changed_client_source() -> None:
    original = client()
    changed = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(original.wheel)) as before,
        zipfile.ZipFile(changed, "w") as after,
    ):
        for name in before.namelist():
            after.writestr(
                name, b"changed" if name == "etude/__init__.py" else before.read(name)
            )
    with pytest.raises(ValueError, match="Python source differs"):
        adapter.prepare(original.source, changed.getvalue())


def test_bundle_round_trip_and_substitution_rejection() -> None:
    original = client()
    code = adapter.prepare(original.source, original.wheel)
    assert installer.read_bundle(code) == original
    changed = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(code)) as before,
        zipfile.ZipFile(changed, "w") as after,
    ):
        for name in before.namelist():
            after.writestr(
                name, b"substituted" if name == "client.whl" else before.read(name)
            )
    with pytest.raises(ValueError, match="identity differs"):
        installer.read_bundle(changed.getvalue())


def test_adapter_preserves_guardian_and_replaces_only_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from etude.launch.providers.runpod import machines

    monkeypatch.setattr(adapter, "_INSTALLED", None)
    monkeypatch.setattr(machines, "program_archive", machines.program_archive)
    monkeypatch.setattr(machines, "_BOOT", machines._BOOT)
    guardian = machines.program_archive(guardian=True)
    original = client()
    path = tmp_path / "client.zip"
    path.write_bytes(adapter.prepare(original.source, original.wheel))
    digest = adapter.enable(path)
    assert adapter.enable(path) == digest
    assert machines.program_archive() == path.read_bytes()
    with (
        zipfile.ZipFile(io.BytesIO(guardian)) as before,
        zipfile.ZipFile(io.BytesIO(machines.program_archive(guardian=True))) as after,
    ):
        assert all(before.read(n) == after.read(n) for n in before.namelist())
        assert set(after.namelist()) - set(before.namelist()) == {
            "etude/launch/prebuilt_install.py"
        }
        # The installer logs through the SDK's own retained bootstrap log.
        assert "etude/launch/logging.py" in after.namelist()
    compile(machines._BOOT, "bootstrap", "exec")
    assert machines._BOOT.index("_prebuilt_pod.install =") < machines._BOOT.index(
        "from etude.launch.providers.runpod.pod import main"
    )


def test_expired_total_install_budget_keeps_failed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = client()
    code = adapter.prepare(original.source, original.wheel)
    monkeypatch.setenv("ETUDE_HOME", str(tmp_path))
    log = tmp_path / "bootstrap.log"
    monkeypatch.setenv("ETUDE_LOG_PATH", str(log))
    monkeypatch.setattr(installer.shutil, "which", lambda _: "/usr/bin/uv")
    clock = [0.0]
    monkeypatch.setattr(installer.time, "monotonic", lambda: clock[0])
    calls: list[list[str]] = []

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(argv)
        assert kwargs["timeout"] == 120
        assert "--no-install-project" in argv
        clock[0] = 121
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(installer.subprocess, "run", run)
    with pytest.raises(subprocess.TimeoutExpired):
        installer.install(code)
    assert len(calls) == 1
    assert not list(tmp_path.rglob("ready"))
    assert list(tmp_path.rglob(f"etude_sdk-{installer.SDK_VERSION}-py3-none-any.whl"))
    retained = log.read_text()
    assert "client install: starting client_runtime_dependencies" in retained
    assert "client install: TimeoutExpired: timed out after 120 seconds" in retained
