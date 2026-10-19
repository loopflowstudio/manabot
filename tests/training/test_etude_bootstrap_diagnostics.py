"""Startup observation preserves boot behavior and keeps credentials out of logs."""

import gzip
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import ModuleType, SimpleNamespace

import pytest

from experiments.runners.etude_bootstrap_diagnostics import instrument_bootstrap


@pytest.mark.parametrize("install_fails", [False, True])
def test_boot_events_preserve_success_and_failure(
    install_fails: bool,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []
    pod = ModuleType("etude.launch.providers.runpod.pod")

    def ensure_uv() -> None:
        calls.append("uv")

    def install(code: bytes) -> Path:
        assert code == b"verified-package"
        calls.append("install")
        if install_fails:
            raise RuntimeError("installation failed")
        return Path("/installed")

    def main(argv: list[str]) -> int:
        assert argv == ["pod", "boot"]
        pod.ensure_uv()
        assert pod.install(b"verified-package") == Path("/installed")
        calls.append("worker")
        return 0

    pod.ensure_uv, pod.install, pod.main, pod.Path = ensure_uv, install, main, Path
    parent = ModuleType("etude.launch.providers.runpod")
    parent.pod = pod
    monkeypatch.setitem(sys.modules, parent.__name__, parent)
    monkeypatch.setitem(sys.modules, pod.__name__, pod)
    monkeypatch.setenv("ETUDE_BOOTSTRAP_URL", "secret-not-for-logs")
    source = (
        "from etude.launch.providers.runpod.pod import main\n"
        "raise SystemExit(main(['pod','boot']))\n"
    )
    with pytest.raises(RuntimeError if install_fails else SystemExit) as raised:
        exec(instrument_bootstrap(source), {})
    if not install_fails:
        assert raised.value.code == 0
    assert calls == (
        ["uv", "install"] if install_fails else ["uv", "install", "worker"]
    )
    output = capsys.readouterr().out
    assert "secret-not-for-logs" not in output
    phases = [
        json.loads(line.removeprefix("ETUDE_BOOT_PHASE "))["phase"]
        for line in output.splitlines()
        if line.startswith("ETUDE_BOOT_PHASE ")
    ]
    assert phases == [
        "entrypoint",
        "guardian_imported",
        "uv_start",
        "uv_ready",
        "client_install_start",
    ] + (["client_install_failed"] if install_fails else ["client_install_ready"])


def test_install_timeout_retains_sanitized_stderr_and_original_exception(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    failure = subprocess.TimeoutExpired(
        ["uv", "sync"],
        120,
        stderr=b"x"
        * 5000
        + b" dependency download stalled https://example.test/private?token=secret "
        b"provider-sensitive-token request timed out",
    )
    pod = ModuleType("etude.launch.providers.runpod.pod")

    def install(code: bytes) -> Path:
        raise failure

    def ensure_uv() -> None:
        pass

    def main(argv: list[str]) -> int:
        pod.install(b"package")
        raise AssertionError("installation unexpectedly succeeded")

    pod.ensure_uv, pod.install, pod.main, pod.Path = ensure_uv, install, main, Path
    parent = ModuleType("etude.launch.providers.runpod")
    parent.pod = pod
    monkeypatch.setitem(sys.modules, parent.__name__, parent)
    monkeypatch.setitem(sys.modules, pod.__name__, pod)
    monkeypatch.setenv("RUNPOD_API_KEY", "provider-sensitive-token")
    source = "from etude.launch.providers.runpod.pod import main\nraise SystemExit(main(['pod','boot']))"
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        exec(instrument_bootstrap(source), {})
    assert raised.value is failure
    output = capsys.readouterr().out
    assert "provider-sensitive-token" not in output and "example.test" not in output
    event = next(
        json.loads(line.removeprefix("ETUDE_BOOT_FAILURE "))
        for line in output.splitlines()
        if line.startswith("ETUDE_BOOT_FAILURE ")
    )
    assert event["error_type"] == "TimeoutExpired"
    assert event["timeout_seconds"] == 120
    assert len(event["stderr_tail"]) == 4096
    assert "<redacted-url>" in event["stderr_tail"]
    assert "<redacted>" in event["stderr_tail"]
    assert event["stderr_tail"].endswith("request timed out")


def test_changed_sdk_bootstrap_requires_review() -> None:
    with pytest.raises(ValueError, match="Unsupported Etude bootstrap"):
        instrument_bootstrap("print('different SDK')")


def test_optional_worker_observer_preserves_child_arguments_and_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pod = ModuleType("etude.launch.providers.runpod.pod")
    commands: list[list[str]] = []
    environment = {"UNCHANGED": "private-value"}
    worker_source = "# external observer, never SDK source\n"

    def call(argv: list[str], *, env: dict[str, str]) -> int:
        assert env is environment
        commands.append(argv)
        return 17

    def install(code: bytes) -> Path:
        assert code == b"original-package"
        return tmp_path

    def ensure_uv() -> None:
        pass

    def main(argv: list[str]) -> int:
        assert argv == ["pod", "boot"]
        pod.install(b"original-package")
        result = pod.subprocess.call(
            [
                ".venv/bin/python",
                "-m",
                "etude",
                "worker",
                "--bootstrap",
                "private-bootstrap",
            ],
            env=environment,
        )
        assert pod.subprocess.call(["unrelated-command"], env=environment) == 17
        return result

    pod.ensure_uv, pod.install, pod.main, pod.Path = ensure_uv, install, main, Path
    pod.subprocess = SimpleNamespace(call=call)
    parent = ModuleType("etude.launch.providers.runpod")
    parent.pod = pod
    monkeypatch.setitem(sys.modules, parent.__name__, parent)
    monkeypatch.setitem(sys.modules, pod.__name__, pod)
    source = "from etude.launch.providers.runpod.pod import main\nraise SystemExit(main(['pod','boot']))"
    with pytest.raises(SystemExit) as raised:
        exec(instrument_bootstrap(source, worker_source=worker_source), {})
    assert raised.value.code == 17
    helper = tmp_path / "worker-startup-observer.py"
    assert helper.read_text() == worker_source
    assert commands == [
        [".venv/bin/python", str(helper), "worker", "--bootstrap", "private-bootstrap"],
        ["unrelated-command"],
    ]


def test_worker_accepts_lossless_gzip_source(tmp_path: Path) -> None:
    from etude.launch.schemas import Work
    from etude.launch.worker.attempts import _prepare

    source = io.BytesIO()
    contents = b"the exact frozen source bytes\n"
    with tarfile.open(fileobj=source, mode="w") as archive:
        entry = tarfile.TarInfo("source.txt")
        entry.size = len(contents)
        archive.addfile(entry, io.BytesIO(contents))
    compressed = gzip.compress(source.getvalue(), mtime=0)
    assert gzip.decompress(compressed) == source.getvalue()
    session = tmp_path / "sessions" / "probe"
    home = session / "attempts" / "attempt"
    home.mkdir(parents=True)
    (session / "sources").mkdir()
    # Etude names source objects .tar; its real worker detects compression.
    (session / "sources" / "revision.tar").write_bytes(compressed)
    work = Work(
        attempt="attempt",
        command="true",
        limit_seconds=60,
        limit_kind="out_of_time",
        seed=None,
        until=None,
        start=None,
        config=None,
        repo="frozen-source",
        commit="revision",
        checkpoint=None,
        run="probe",
    )
    environment = _prepare(session, home, work)
    assert (home / "src/source.txt").read_bytes() == contents
    assert Path(environment["ETUDE_OUTPUT_DIR"]) == home / "outputs"
