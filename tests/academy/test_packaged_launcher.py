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
from managym.academy.launch import Academy
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


@pytest.mark.parametrize("collision", ["file", "directory"])
def test_cli_bundled_decks_win_over_paths(
    collision: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("Academy")
    monkeypatch.setenv("ACADEMY_FRONTEND_BUILD", str(site))
    monkeypatch.chdir(tmp_path)
    for name in ("ur_lessons", "gw_allies"):
        path = tmp_path / name
        if collision == "directory":
            path.mkdir()
        else:
            path.write_text("not a decklist")
    configs: list[server.GameConfig] = []

    def inspect_launch(running: Academy) -> None:
        configs.extend(
            server._parse_game_config(c) for c in server.LAUNCH_CONFIGS.values()
        )

    monkeypatch.setattr(Academy, "wait", inspect_launch)
    previous = server.trace_store.TRACES_DIR
    try:
        assert (
            academy(
                [
                    "play",
                    "--deck",
                    "ur_lessons",
                    "--opponent-deck",
                    "gw_allies",
                    "--port",
                    "0",
                    "--no-open",
                ]
            )
            == 0
        )
        assert configs[0].hero_deck_name == "ur_lessons"
        assert configs[0].villain_deck_name == "gw_allies"
        assert "ACADEMY_PLAY_READY" in capsys.readouterr().out
        # Explicit paths still select files (or report an unreadable directory).
        assert (
            academy(
                [
                    "play",
                    "--deck",
                    "./ur_lessons",
                    "--port",
                    "0",
                    "--no-open",
                ]
            )
            == 2
        )
        error_line = next(
            line
            for line in capsys.readouterr().err.splitlines()
            if line.startswith("ACADEMY_PLAY_ERROR ")
        )
        error = json.loads(error_line.removeprefix("ACADEMY_PLAY_ERROR "))
        assert error["code"] == (
            "deck.unreadable" if collision == "directory" else "deck.rejected"
        )
    finally:
        server.trace_store.TRACES_DIR = previous


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
    from managym.policy import PolicyRegistry

    monkeypatch.setattr(opponent, "_registry", PolicyRegistry(providers={}))
    monkeypatch.setattr(server.trace_store, "TRACES_DIR", tmp_path)
    try:
        catalog = opponent.opponent_catalog()
        assert catalog["extension"] is None
        assert catalog["opponents"] == []
        assert catalog["watch_players"] == ["random", "search"]
        with pytest.raises(HTTPException) as refused:
            create_watch(WatchRequest(players=["greedy", "random"]))
        assert refused.value.status_code == 400
        assert "no longer available" in refused.value.detail
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


@pytest.mark.parametrize(
    "refused", [("fastapi", "starlette", "uvicorn"), ("typer", "click")]
)
def test_launcher_without_extra_names_the_install_command(
    refused: tuple[str, ...],
) -> None:
    probe = f"""
import importlib.abc
import sys
class RefuseExtra(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {set(refused)!r}:
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, RefuseExtra())
from managym.cli import main
raise SystemExit(main(['academy', '--no-open']))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )
    assert result.returncode == 2
    record = json.loads(result.stderr.removeprefix("ACADEMY_PLAY_ERROR "))
    assert record["code"] == "extra.missing"
    assert "managym[academy]" in record["message"]


def test_shared_options_go_before_or_after_the_verb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "index.html").write_text("Academy")
    monkeypatch.setenv("ACADEMY_FRONTEND_BUILD", str(tmp_path))
    monkeypatch.setattr(Academy, "wait", lambda self: None)
    previous = server.trace_store.TRACES_DIR
    try:
        for argv in (
            ["--port", "0", "--traces-dir", str(tmp_path / "a"), "play", "--no-open"],
            ["play", "--port", "0", "--traces-dir", str(tmp_path / "a"), "--no-open"],
            [
                "--traces-dir",
                str(tmp_path / "b"),
                "play",
                "--traces-dir",
                str(tmp_path / "a"),
                "--port",
                "0",
                "--no-open",
            ],
        ):
            assert academy(argv) == 0
            ready = json.loads(
                capsys.readouterr()
                .out.splitlines()[0]
                .removeprefix("ACADEMY_PLAY_READY ")
            )
            assert ready["traces_dir"] == str((tmp_path / "a").resolve())
            assert "launch=" in ready["open"]
    finally:
        server.trace_store.TRACES_DIR = previous


def test_policy_cli_uses_spec_alias_and_reports_missing_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("Academy")
    monkeypatch.setenv("ACADEMY_FRONTEND_BUILD", str(site))
    monkeypatch.setattr(Academy, "wait", lambda self: None)
    specification = tmp_path / "challenger.json"
    specification.write_text(
        json.dumps({"provider": "manabot", "config": {"kind": "heuristic"}})
    )
    previous = server.trace_store.TRACES_DIR
    try:
        assert (
            academy(
                [
                    "watch",
                    "--policy",
                    str(specification),
                    "--player",
                    "challenger",
                    "--player",
                    "random",
                    "--port",
                    "0",
                    "--no-open",
                ]
            )
            == 0
        )
        ready = json.loads(
            capsys.readouterr().out.splitlines()[0].removeprefix("ACADEMY_PLAY_READY ")
        )
        assert "seat1=policy%3A" in ready["open"]
        assert "seat2=random" in ready["open"]
        specification.write_text(json.dumps({"provider": "absent-provider"}))
        assert (
            academy(["--policy", str(specification), "--port", "0", "--no-open"]) == 2
        )
        error = json.loads(capsys.readouterr().err.removeprefix("ACADEMY_PLAY_ERROR "))
        assert error["code"] == "policy.unavailable"
        assert "absent-provider" in error["message"]
    finally:
        server.trace_store.TRACES_DIR = previous
