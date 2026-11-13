"""Configured launch reaches the existing game and replay authorities."""

from collections.abc import Iterator
import copy
import gzip
import json
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any
from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient
import pytest
from websockets.sync.client import connect

from managym.academy import extensions, server
from managym.academy.launch import Academy, AcademyLaunchError, read_games, serve


@pytest.fixture
def academy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Academy]:
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("Academy")
    monkeypatch.setenv("ACADEMY_FRONTEND_BUILD", str(site))
    previous = server.trace_store.TRACES_DIR
    with serve(port=0, traces_dir=tmp_path / "traces", open_browser=False) as running:
        yield running
    server.trace_store.TRACES_DIR = previous


def _launch_id(url: str) -> str:
    return parse_qs(urlparse(url).query)["launch"][0]


def test_game_launch_preserves_config_pending_decision_and_identity(
    academy: Academy,
) -> None:
    launched = academy.new_game(
        hero="gw_allies", villain="ur_lessons", opponent="passive", seed=7
    )
    launch_id = _launch_id(launched.url)
    with connect(academy.url.replace("http:", "ws:") + "/ws/play") as ws:
        ws.send(json.dumps({"type": "new_game", "config": {"launch": launch_id}}))
        message = json.loads(ws.recv(timeout=20))
        assert message["type"] == "observation", message
        frame = (
            message["recovery"]["frame"] if "recovery" in message else message["frame"]
        )
        assert frame["prompt"] and frame["offers"]
        assert message["launch_setup"]["hero_deck"] == "gw_allies"
        assert "seed" not in message["launch_setup"]
        session = server.SESSION_REGISTRY[message["session_id"]].game
        assert session.trace is not None
        assert session.trace.config.seed == 7
        assert session.trace.config.villain_type == "passive"
        assert session.trace.config.hero_deck_name == "gw_allies"
        ws.send(json.dumps({"type": "new_game", "config": {"launch": launch_id}}))
        again = json.loads(ws.recv(timeout=20))
        assert again["frame"]["match_id"] == frame["match_id"]


def test_mixed_file_and_bundled_decks_admit_sideboards(
    academy: Academy, tmp_path: Path
) -> None:
    deck = tmp_path / "deck.txt"
    deck.write_text("20 Mountain\n20 Gray Ogre\nSideboard\n2 Lightning Bolt\n")
    launched = academy.new_game(hero=deck, opponent="random", seed=11)
    config = server._parse_game_config(server.LAUNCH_CONFIGS[_launch_id(launched.url)])
    assert config.hero_deck_name == config.villain_deck_name == "imported"
    assert config.hero_sideboard == {"Lightning Bolt": 2}
    assert config.villain_sideboard
    assert config.asset_pack is None


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"hero": "absent"}, "deck.unknown"),
        ({"hero": Path("/absent/deck.txt")}, "deck.unreadable"),
        ({"seed": -1}, "setup.invalid"),
        ({"sims": 0}, "setup.invalid"),
        ({"decklists": "closed"}, "setup.invalid"),
        ({"opponent": "a" * 64}, "opponent.unavailable"),
    ],
)
def test_setup_errors_before_open(
    academy: Academy, kwargs: dict[str, Any], code: str
) -> None:
    with pytest.raises(AcademyLaunchError) as error:
        academy.new_game(**kwargs)
    assert error.value.code == code


def test_native_deck_diagnostics_and_unknown_launch(
    academy: Academy, tmp_path: Path
) -> None:
    deck = tmp_path / "bad.txt"
    deck.write_text("20 Mountain\n2 Not A Card\n")
    with pytest.raises(AcademyLaunchError) as error:
        academy.new_game(hero=deck)
    assert error.value.code == "deck.rejected"
    assert "hero" in error.value.message
    assert "2" in error.value.message
    with TestClient(server.app).websocket_connect("/ws/play") as ws:
        ws.send_json({"type": "new_game", "config": {"launch": "expired"}})
        assert "launch_unknown" in ws.receive_json()["message"]


def test_lifecycle_exclusive_and_releases_port(academy: Academy) -> None:
    with pytest.raises(AcademyLaunchError, match="already running") as error:
        serve(port=0, open_browser=False)
    assert error.value.code == "server.running"
    port = urlparse(academy.url).port
    academy.close()
    academy.close()
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", port))
    with serve(port=0, traces_dir=academy.traces_dir, open_browser=False):
        pass


def _record(index: int) -> dict[str, Any]:
    return {
        "deal_seed": index,
        "leg": 0,
        "seat_players": ["random", "random"],
        "seat_decks": ["one", "two"],
        "game_trace_sha256": str(index) * 64,
        "decisions": [],
    }


def test_trace_read_selection_and_errors(academy: Academy, tmp_path: Path) -> None:
    path = tmp_path / "games.jsonl.gz"
    path.write_bytes(
        gzip.compress(b"\n".join(json.dumps(_record(i)).encode() for i in (1, 2)))
    )
    games = read_games(path)
    assert [game.index for game in games] == [0, 1]
    assert games[1].players == ("random", "random")
    with pytest.raises(AcademyLaunchError) as error:
        academy.replay(path)
    assert error.value.code == "replay.ambiguous"
    assert len(error.value.games) == 2
    for game in (4, "ab"):
        with pytest.raises(AcademyLaunchError) as error:
            academy.replay(path, game=game)
        assert error.value.code == "replay.game_unknown"
    path.write_text("not a trace")
    with pytest.raises(AcademyLaunchError) as error:
        read_games(path)
    assert error.value.code == "replay.unreadable"
    for source in ("absent", tmp_path / "absent.json"):
        with pytest.raises(AcademyLaunchError) as error:
            academy.replay(source)
        assert error.value.code == "replay.missing"


def test_replay_needs_optional_manabot(
    academy: Academy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(extensions, "entry_points", lambda **kwargs: [])
    extensions.bot_extension.cache_clear()
    path = tmp_path / "game.json"
    path.write_text(json.dumps(_record(1)))
    try:
        with pytest.raises(AcademyLaunchError) as error:
            academy.replay(path)
        assert error.value.code == "replay.needs_manabot"
        assert "uv pip install" in error.value.message
    finally:
        extensions.bot_extension.cache_clear()


def test_evaluation_replay_reuses_recording_and_rejects_tampering(
    academy: Academy, tmp_path: Path
) -> None:
    from manabot.eval.match import play_cell
    from manabot.sim.replay import read_trace
    from managym.academy.watch import inspect_watch
    from tests.eval.arena_fixture import arena_fixture

    contract = arena_fixture()
    _, trace, receipt = play_cell(
        key=contract.key,
        player_a=contract.anchors[0],
        player_b=contract.anchors[1],
        deal_seeds=(777,),
        legs=(0,),
        out_dir=tmp_path / "arena",
    )
    assert receipt["passed"]
    source = Path(trace["path"])
    original = source.read_bytes()
    records = read_games(source)
    launched = academy.replay(source, game=0)
    assert source.read_bytes() == original
    assert academy.replay(source, game=records[0].game_trace_sha256[:12]) == launched
    trace_id = parse_qs(urlparse(launched.url).query)["trace"][0]
    inspected = inspect_watch(trace_id, 0, None)
    assert inspected["count"] == len(read_trace(source)[0]["decisions"])
    assert academy.replay(trace_id).url.endswith(f"trace={trace_id}")
    broken = copy.deepcopy(read_trace(source)[0])
    broken["decisions"][0]["command"]["offer_id"] += 1
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(broken))
    with pytest.raises(AcademyLaunchError) as error:
        academy.replay(bad)
    assert error.value.code == "replay.rejected"
    assert "mismatches" in error.value.message


def test_replay_verification_does_not_import_torch() -> None:
    probe = """
import importlib.abc
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'torch':
            raise ImportError('Replay imported Torch: ' + fullname)
sys.meta_path.insert(0, NoTorch())
from scripts.release.academy_replay_smoke import write_example_trace
from manabot.sim.replay import read_trace
from manabot.sim.table_bots import TrainingBots
with TemporaryDirectory() as directory:
    path = Path(directory) / 'game.jsonl.gz'
    write_example_trace(path)
    arena = TrainingBots().verify_arena(read_trace(path)[0])
    assert arena.decisions
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", probe], check=True, timeout=30)


def test_launch_setup_stays_with_initiator_and_shutdown_keeps_history(
    academy: Academy,
) -> None:
    from managym.academy.attempts import AttemptStore

    first = _launch_id(
        academy.new_game(opponent="passive", seed=7, decklists="closed").url
    )
    endpoint = academy.url.replace("http:", "ws:") + "/ws/play"
    with connect(endpoint) as pilot, connect(endpoint) as watcher:
        pilot.send(json.dumps({"type": "new_game", "config": {"launch": first}}))
        initial = json.loads(pilot.recv(timeout=20))
        invite = parse_qs(urlparse(initial["table"]["watcher_invite"]).fragment)
        watcher.send(
            json.dumps(
                {
                    "type": "join_table",
                    "table_id": invite["table"][0],
                    "invite_token": invite["watch"][0],
                }
            )
        )
        watched = json.loads(watcher.recv(timeout=20))
        assert "launch" not in watched and "launch_setup" not in watched
        second = _launch_id(
            academy.new_game(opponent="passive", seed=8, decklists="closed").url
        )
        pilot.send(
            json.dumps(
                {
                    "type": "new_game",
                    "grant_revision": initial["table"]["access"]["grant_revision"],
                    "config": {"launch": second},
                }
            )
        )
        changed = json.loads(watcher.recv(timeout=20))
        assert changed["type"] == "observation", changed
        assert "launch" not in changed and "launch_setup" not in changed
        trace_id = changed["table"]["attempt_id"]
    academy.close()
    store = AttemptStore(academy.traces_dir / "play.sqlite")
    retained = store.load(trace_id)
    assert retained is not None
    assert retained["end_reason"] == "server_restart"
    assert retained["winner"] is None
