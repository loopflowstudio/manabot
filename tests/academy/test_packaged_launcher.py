"""Installed Academy reports actionable launch errors without training imports."""

import json
from pathlib import Path
import socket
import subprocess
import sys

from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest

from managym.academy import extensions, opponent, server
from managym.academy.watch import WatchRequest, create_watch, import_arena
from managym.cli import academy


def test_launcher_help_lists_options_and_guide() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "managym.academy", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--traces-dir" in result.stdout
    assert "http://127.0.0.1:8000" in result.stdout
    assert "docs/academy.md" in result.stdout


def test_missing_site_names_checkout_launcher(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ACADEMY_FRONTEND_BUILD", str(tmp_path))
    assert academy(["--no-open"]) == 2
    record = json.loads(capsys.readouterr().err.removeprefix("ACADEMY_PLAY_ERROR "))
    assert record["code"] == "site.missing"
    assert "./scripts/play" in record["message"]


def test_busy_port_has_documented_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "index.html").write_text("Academy")
    monkeypatch.setenv("ACADEMY_FRONTEND_BUILD", str(tmp_path))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        assert academy(["--no-open", "--port", str(listener.getsockname()[1])]) == 2
    record = json.loads(capsys.readouterr().err.removeprefix("ACADEMY_PLAY_ERROR "))
    assert record["code"] == "port.in_use"


def test_play_only_capabilities_are_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(extensions, "entry_points", lambda **kwargs: [])
    extensions.bot_extension.cache_clear()
    monkeypatch.setattr(server.trace_store, "TRACES_DIR", tmp_path)
    try:
        catalog = opponent.opponent_catalog()
        assert catalog["extension"] is None
        assert catalog["opponents"] == []
        assert catalog["watch_players"] == ["random", "search"]
        with pytest.raises(HTTPException) as refused:
            create_watch(WatchRequest(players=["greedy", "random"]))
        assert refused.value.status_code == 400
        assert refused.value.detail == "watch_player_unavailable"
        with pytest.raises(HTTPException) as refused:
            import_arena({})
        assert refused.value.detail == "arena_import_unavailable"
        from managym.academy import advice

        monkeypatch.setattr(advice, "FIXTURE_PATH", tmp_path / "absent.json")
        advice.load_advice_fixture.cache_clear()
        with TestClient(server.app) as client:
            response = client.get("/api/advice")
        assert response.status_code == 404
        assert response.json()["detail"] == "advice_unavailable"
    finally:
        extensions.bot_extension.cache_clear()


def test_launcher_without_extra_names_the_install_command() -> None:
    probe = """
import importlib.abc
import sys
class RefuseServer(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'fastapi', 'starlette', 'uvicorn'}:
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, RefuseServer())
from managym.cli import academy
raise SystemExit(academy(['--no-open']))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )
    assert result.returncode == 2
    record = json.loads(result.stderr.removeprefix("ACADEMY_PLAY_ERROR "))
    assert record["code"] == "extra.missing"
    assert "managym[academy]" in record["message"]
