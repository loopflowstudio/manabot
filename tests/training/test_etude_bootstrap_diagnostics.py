"""Startup observation preserves boot behavior and keeps credentials out of logs."""

import json
from pathlib import Path
import sys
from types import ModuleType

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
    ]
    assert phases == [
        "entrypoint",
        "guardian_imported",
        "uv_start",
        "uv_ready",
        "client_install_start",
    ] + ([] if install_fails else ["client_install_ready"])


def test_changed_sdk_bootstrap_requires_review() -> None:
    with pytest.raises(ValueError, match="Unsupported Etude bootstrap"):
        instrument_bootstrap("print('different SDK')")
