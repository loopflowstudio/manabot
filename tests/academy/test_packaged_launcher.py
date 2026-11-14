"""Installed Academy reports actionable launch errors without training imports."""

import json
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest

from managym.academy import extensions, opponent, server
from managym.academy.watch import WatchRequest, create_watch, import_arena
from managym.bots import BotPolicy, passive_policy
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


@pytest.mark.parametrize("held", ["127.0.0.1", "0.0.0.0"])
def test_busy_port_has_documented_error(
    held: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A wildcard listener matters on macOS, where SO_REUSEADDR would otherwise
    # let Academy bind 127.0.0.1 beside it and take its local traffic.
    (tmp_path / "index.html").write_text("Academy")
    monkeypatch.setenv("ACADEMY_FRONTEND_BUILD", str(tmp_path))
    with socket.socket() as listener:
        listener.bind((held, 0))
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


def test_extension_declared_watch_player_plays_and_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from manabot.sim.table_bots import TrainingBots
    from managym.academy.extensions import CheckpointSummary, UnavailableEntry
    from managym.academy.watch import inspect_watch, step_watch

    class ExtraBots(TrainingBots):
        watch_players = ("extra",)

        def discover(
            self, paths: list[Path], runs_dir: Path
        ) -> tuple[list[CheckpointSummary], list[UnavailableEntry]]:
            return [], []

        def watch_policy(self, kind: str) -> BotPolicy:
            return passive_policy

    provider = ExtraBots()
    monkeypatch.setattr(
        extensions,
        "entry_points",
        lambda **kwargs: [SimpleNamespace(name="extra", load=lambda: provider)],
    )
    monkeypatch.setattr(server.trace_store, "TRACES_DIR", tmp_path)
    extensions.bot_extension.cache_clear()
    created = None
    try:
        with TestClient(server.app) as client:
            catalog = client.get("/api/opponents").json()
            assert catalog["watch_players"] == ["random", "search", "extra"]
            response = client.post(
                "/api/watch", json={"players": ["extra", "extra"], "seed": 42}
            )
            assert response.status_code == 200, response.text
            created = response.json()
            result = step_watch(created["id"], created["token"], 0)
            assert result["revision"] == 1
            row = inspect_watch(created["id"], 0, created["token"])
            assert row["players"] == ["extra", "extra"]
            assert [player["name"] for player in row["identities"]] == [
                "Extra",
                "Extra",
            ]
            assert row["outputs"]["probabilities"][row["outputs"]["action"]] == 1
            for kind in ("greedy", "unknown", "passive"):
                refused = client.post("/api/watch", json={"players": ["extra", kind]})
                assert refused.status_code == 400
                assert refused.json()["detail"] == "watch_player_unavailable"
    finally:
        if created is not None:
            server._drop_session(created["id"], "test")
        extensions.bot_extension.cache_clear()


@pytest.mark.parametrize("kind", ["random", "search", "passive", "unknown"])
def test_training_extension_rejects_undeclared_watch_players(kind: str) -> None:
    from manabot.sim.table_bots import TrainingBots

    provider = TrainingBots()
    assert provider.watch_players == ("greedy", "heuristic")
    with pytest.raises(ValueError, match="Unsupported watch player"):
        provider.watch_policy(kind)
