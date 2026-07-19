"""Contract tests for the one-command clean-machine play launcher."""

from __future__ import annotations

from pathlib import Path
import re
import socket
import subprocess
import tomllib
from types import SimpleNamespace

import pytest

from scripts import play


def test_gui_wire_enum_mirrors_match_the_native_engine():
    from etude.enums import (
        ActionEnum,
        ActionSpaceEnum,
        EventTypeEnum,
        PhaseEnum,
        StepEnum,
        ZoneEnum,
    )
    import managym

    for mirror, native in (
        (ActionEnum, managym.ActionEnum),
        (ActionSpaceEnum, managym.ActionSpaceEnum),
        (EventTypeEnum, managym.EventTypeEnum),
        (PhaseEnum, managym.PhaseEnum),
        (StepEnum, managym.StepEnum),
        (ZoneEnum, managym.ZoneEnum),
    ):
        assert {member.name: int(member) for member in mirror} == {
            member.name: int(getattr(native, member.name)) for member in mirror
        }


def completed(
    argv: list[str],
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(argv, returncode, stdout, stderr)


def test_python_version_is_exactly_cp312():
    assert (
        play.validate_python_version(SimpleNamespace(major=3, minor=12, micro=11))
        == "3.12.11"
    )
    with pytest.raises(play.PlayError, match="CPython 3.12") as raised:
        play.validate_python_version(SimpleNamespace(major=3, minor=14, micro=0))
    assert raised.value.code == "python.version"


@pytest.mark.parametrize(
    ("major", "minor", "supported"),
    [
        (20, 18, False),
        (20, 19, True),
        (21, 9, False),
        (22, 11, False),
        (22, 12, True),
        (23, 9, False),
        (24, 0, True),
        (25, 8, True),
    ],
)
def test_node_version_support_matches_locked_vite(major, minor, supported):
    assert play.node_version_is_supported(major, minor) is supported


def test_invalid_node_version_has_stable_diagnostic():
    def runner(argv, **_kwargs):
        return completed(argv, stdout="v22.11.0\n")

    with pytest.raises(play.PlayError) as raised:
        play.validate_node_version("node", runner=runner)
    assert raised.value.code == "prerequisite.node"


def test_missing_and_invalid_pack_fail_before_start(monkeypatch, tmp_path):
    missing = tmp_path / "manifest.json"
    notice = tmp_path / "NOTICE.md"
    notice.write_text("notice", encoding="utf-8")
    monkeypatch.setattr(play, "PACK_MANIFEST", missing)
    monkeypatch.setattr(play, "PACK_NOTICE", notice)

    with pytest.raises(play.PlayError) as raised:
        play.validate_pack()
    assert raised.value.code == "pack.missing"

    missing.write_text("{}", encoding="utf-8")
    with pytest.raises(play.PlayError) as raised:
        play.validate_pack()
    assert raised.value.code == "pack.invalid"


def test_missing_pack_notice_has_stable_diagnostic(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(play, "PACK_MANIFEST", manifest)
    monkeypatch.setattr(play, "PACK_NOTICE", tmp_path / "NOTICE.md")

    with pytest.raises(play.PlayError) as raised:
        play.validate_pack()
    assert raised.value.code == "pack.notice"


def test_frontend_install_marker_is_bound_to_lock(monkeypatch, tmp_path):
    lock = tmp_path / "package-lock.json"
    marker = tmp_path / "node_modules" / ".etude-package-lock.sha256"
    required = tmp_path / "node_modules" / ".bin" / "vite"
    lock.write_text('{"lockfileVersion": 3}', encoding="utf-8")
    monkeypatch.setattr(play, "FRONTEND_LOCK", lock)
    monkeypatch.setattr(play, "FRONTEND_INSTALL_MARKER", marker)
    monkeypatch.setattr(play, "FRONTEND_REQUIRED_PATHS", (required,))

    assert play.frontend_needs_install()
    marker.parent.mkdir(parents=True)
    required.parent.mkdir()
    required.write_text("installed", encoding="utf-8")
    marker.write_text(play.lock_sha256() + "\n", encoding="utf-8")
    assert not play.frontend_needs_install()
    required.unlink()
    assert play.frontend_needs_install()
    required.write_text("installed", encoding="utf-8")
    lock.write_text('{"lockfileVersion": 4}', encoding="utf-8")
    assert play.frontend_needs_install()


def test_failed_npm_ci_has_stable_diagnostic(monkeypatch, tmp_path):
    lock = tmp_path / "package-lock.json"
    lock.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(play, "FRONTEND", tmp_path)
    monkeypatch.setattr(play, "FRONTEND_LOCK", lock)
    monkeypatch.setattr(
        play,
        "FRONTEND_INSTALL_MARKER",
        tmp_path / "node_modules" / ".etude-package-lock.sha256",
    )

    def runner(argv, **_kwargs):
        return completed(argv, returncode=1, stderr="registry unavailable")

    with pytest.raises(play.PlayError) as raised:
        play.ensure_frontend("npm", runner=runner)
    assert raised.value.code == "frontend.install"
    assert "registry unavailable" in (raised.value.detail or "")


def test_failed_svelte_sync_has_stable_diagnostic(monkeypatch, tmp_path):
    lock = tmp_path / "package-lock.json"
    marker = tmp_path / "node_modules" / ".etude-package-lock.sha256"
    lock.write_text("{}", encoding="utf-8")
    marker.parent.mkdir()
    monkeypatch.setattr(play, "FRONTEND", tmp_path)
    monkeypatch.setattr(play, "FRONTEND_LOCK", lock)
    monkeypatch.setattr(play, "FRONTEND_INSTALL_MARKER", marker)

    def runner(argv, **_kwargs):
        if "svelte-kit" in argv:
            return completed(argv, returncode=1, stderr="invalid Svelte config")
        return completed(argv)

    with pytest.raises(play.PlayError) as raised:
        play.ensure_frontend("npm", runner=runner)
    assert raised.value.code == "frontend.install"
    assert "invalid Svelte config" in (raised.value.detail or "")
    assert not marker.exists()


def test_failed_native_build_and_import_are_distinct():
    def failed_build(argv, **_kwargs):
        if "maturin" in argv:
            return completed(argv, returncode=1, stderr="cargo failed")
        return completed(argv, returncode=1, stderr="missing module")

    with pytest.raises(play.PlayError) as raised:
        play.ensure_native(runner=failed_build)
    assert raised.value.code == "native.build"

    def missing_after_build(argv, **_kwargs):
        if "maturin" in argv:
            return completed(argv)
        return completed(argv, returncode=1, stderr="wrong ABI")

    with pytest.raises(play.PlayError) as raised:
        play.ensure_native(runner=missing_after_build)
    assert raised.value.code == "native.import"


def test_native_import_reports_the_extension_abi_and_path():
    def runner(argv, **_kwargs):
        assert argv[: len(play.UV_RUNTIME)] == play.UV_RUNTIME
        assert "import managym._managym as native" in argv[-1]
        return completed(
            argv,
            stdout='{"abi": "cp312", "module": "/tmp/_managym.cpython-312.so"}\n',
        )

    assert play.native_import(runner=runner) == {
        "abi": "cp312",
        "module": "/tmp/_managym.cpython-312.so",
    }


def test_locked_play_runtime_covers_and_imports_live_advice_dependencies():
    project = tomllib.loads((play.ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    group = project["dependency-groups"]["play-runtime"]
    names = {re.split(r"[<>=!~; ]", requirement, maxsplit=1)[0] for requirement in group}
    assert {"numpy", "torch"} <= names

    uv = project["tool"]["uv"]
    assert uv["default-groups"] == ["training-runtime"]
    assert uv["conflicts"] == [
        [{"group": "training-runtime"}, {"group": "play-runtime"}]
    ]
    torch_sources = uv["sources"]["torch"]
    assert {
        (source["group"], source["index"], source.get("marker"))
        for source in torch_sources
    } == {
        ("play-runtime", "pytorch-cpu", None),
        ("training-runtime", "pytorch-cuda", "sys_platform == 'linux'"),
        ("training-runtime", "pytorch-training", "sys_platform != 'linux'"),
    }

    observed: list[str] = []

    def runner(argv, **_kwargs):
        assert argv[: len(play.UV_RUNTIME)] == play.UV_RUNTIME
        observed.extend(argv[-1].split("; "))
        return completed(argv)

    play.ensure_runtime_imports(runner=runner)
    assert observed == [f"import {module}" for module in play.RUNTIME_IMPORTS]


def _exported_requirements(group: str) -> tuple[str, set[str]]:
    result = subprocess.run(
        [
            "uv",
            "export",
            "--locked",
            "--only-group",
            group,
            "--no-hashes",
            "--no-header",
        ],
        cwd=play.ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    names = {
        line.split("==", 1)[0].lower().replace("_", "-")
        for line in result.stdout.splitlines()
        if line and not line.startswith((" ", "#", "-e ")) and "==" in line
    }
    return result.stdout, names


def test_locked_play_closure_is_cpu_only_and_training_retains_cuda():
    play_export, play_names = _exported_requirements("play-runtime")
    forbidden = {
        name
        for name in play_names
        if name.startswith(("cuda-", "nvidia-")) or name == "triton"
    }
    assert not forbidden
    assert any(
        line.startswith("torch==")
        and "+cpu" in line
        and (
            "sys_platform == 'linux'" in line
            or "sys_platform != 'darwin'" in line
        )
        for line in play_export.splitlines()
    )

    training_export, training_names = _exported_requirements("training-runtime")
    assert any(
        line.startswith("torch==")
        and "+cu128" in line
        and "sys_platform == 'linux'" in line
        for line in training_export.splitlines()
    )
    assert "triton" in training_names
    assert any(name.startswith("nvidia-") for name in training_names)


def test_occupied_port_fails_deterministically():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with pytest.raises(play.PlayError) as raised:
            play.assert_port_available(port, "backend")
    assert raised.value.code == "port.in_use"
    assert str(port) in (raised.value.detail or "")


class FakeProcess:
    def __init__(self, returncode=None):
        self.returncode = returncode
        self.pid = 999_999
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        del timeout
        return self.returncode


def test_frontend_spawn_failure_cleans_up_backend(monkeypatch):
    backend = FakeProcess()
    spawns = 0
    cleaned = []

    def popen(*_args, **_kwargs):
        nonlocal spawns
        spawns += 1
        if spawns == 1:
            return backend
        raise OSError("frontend unavailable")

    monkeypatch.setattr(play.subprocess, "Popen", popen)
    monkeypatch.setattr(
        play,
        "terminate_processes",
        lambda processes: cleaned.append(processes),
    )

    with pytest.raises(play.PlayError) as raised:
        play.start_processes(8000, 5173, "npm")
    assert raised.value.code == "frontend.start"
    assert cleaned == [[("backend", backend)]]


def test_child_exit_and_readiness_timeout_are_distinct(monkeypatch):
    exited = FakeProcess(returncode=7)
    with pytest.raises(play.PlayError) as raised:
        play.wait_for_readiness(
            [("backend", exited)],
            {"backend": "http://127.0.0.1:1"},
            0.1,
        )
    assert raised.value.code == "backend.start"

    running = FakeProcess()
    monkeypatch.setattr(play, "endpoint_ready", lambda _url: False)
    with pytest.raises(play.PlayError) as raised:
        play.wait_for_readiness(
            [("backend", running)],
            {"backend": "http://127.0.0.1:1"},
            0,
        )
    assert raised.value.code == "ready.timeout"


def test_ready_record_pins_pack_and_local_urls():
    payload = play.build_ready_payload(
        started_at=10.0,
        now=10.25,
        backend_port=8011,
        frontend_port=5183,
        python_version="3.12.11",
        node_version="22.12.0",
        npm_version="10.9.0",
        native={"abi": "cp312", "module": "/tmp/managym.so"},
        asset_pack={"id": "pack", "version": "1", "manifest_sha256": "abc"},
    )
    assert payload["schema_version"] == 1
    assert payload["url"] == "http://127.0.0.1:5183"
    assert payload["backend_url"] == "http://127.0.0.1:8011"
    assert payload["elapsed_ms"] == 250.0
    assert payload["asset_pack"]["manifest_sha256"] == "abc"


def test_shutdown_terminates_the_process_group(monkeypatch):
    process = FakeProcess()
    signals: list[tuple[int, int]] = []

    def killpg(pid, signal_number):
        signals.append((pid, signal_number))
        process.returncode = -signal_number

    monkeypatch.setattr(play.os, "killpg", killpg)
    play.terminate_processes([("backend", process)])
    assert signals == [(process.pid, play.signal.SIGTERM)]


def test_wrapper_reports_missing_uv_before_python():
    wrapper = Path(__file__).resolve().parents[2] / "scripts" / "play"
    result = subprocess.run(
        ["/bin/sh", str(wrapper)],
        cwd=wrapper.parents[1],
        env={"PATH": "/usr/bin:/bin"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert '"code":"prerequisite.uv"' in result.stderr


def test_wrapper_no_frontend_does_not_require_node_or_npm(tmp_path):
    wrapper = Path(__file__).resolve().parents[2] / "scripts" / "play"
    for command in ("uv", "rustc", "cargo"):
        executable = tmp_path / command
        body = "#!/bin/sh\nexit 0\n"
        if command == "uv":
            body = (
                "#!/bin/sh\n"
                "if [ \"$1\" = export ]; then\n"
                "  echo \"torch==2.10.0+cpu ; sys_platform != 'darwin'\"\n"
                "fi\n"
                "exit 0\n"
            )
        executable.write_text(body, encoding="utf-8")
        executable.chmod(0o755)
    result = subprocess.run(
        ["/bin/sh", str(wrapper), "--no-frontend"],
        cwd=wrapper.parents[1],
        env={"PATH": f"{tmp_path}:/usr/bin:/bin"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "prerequisite.node" not in result.stderr


def test_wrapper_attributes_pre_python_uv_failure_to_the_lock(tmp_path):
    wrapper = Path(__file__).resolve().parents[2] / "scripts" / "play"
    for command in ("rustc", "cargo"):
        executable = tmp_path / command
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
    uv = tmp_path / "uv"
    uv.write_text("#!/bin/sh\nexit 2\n", encoding="utf-8")
    uv.chmod(0o755)

    result = subprocess.run(
        ["/bin/sh", str(wrapper), "--no-frontend"],
        cwd=wrapper.parents[1],
        env={"PATH": f"{tmp_path}:/usr/bin:/bin"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert '"code":"lock.python"' in result.stderr
