"""Exercise the deployment wrapper independently of the game engine."""

from pathlib import Path
import runpy
import sys
from types import ModuleType

from fastapi import FastAPI, WebSocket
import pytest
from starlette.testclient import TestClient


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    (tmp_path / "index.html").write_text("<html>Academy</html>")
    monkeypatch.setenv("ACADEMY_FRONTEND_BUILD", str(tmp_path))
    server = ModuleType("academy.server")
    app = FastAPI()
    server.app = app  # type: ignore[attr-defined]  # Substitute only the engine boundary.
    monkeypatch.setitem(sys.modules, "academy.server", server)

    @app.websocket("/ws/play")
    async def play(socket: WebSocket) -> None:
        await socket.accept()
        await socket.send_text("connected")
        await socket.close()

    runpy.run_path(str(Path(__file__).parents[1] / "asgi.py"))
    return TestClient(app, base_url="https://play.manabot.gg")


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_legacy_http_redirect_preserves_url(client: TestClient, method: str) -> None:
    path = "/replay/a%2Fb?tag=a&tag=b&next=%2Fgames%3Fpage%3D2"
    response = client.request(
        method, path, headers={"host": "play.etude.gg"}, follow_redirects=False
    )
    assert response.status_code == 308
    assert response.headers["location"] == f"https://play.manabot.gg{path}"


@pytest.mark.parametrize(
    "host", ["play.manabot.gg", "academy.manabot.gg", "academy-play.fly.dev", "localhost"]
)
def test_current_hosts_serve_spa(client: TestClient, host: str) -> None:
    response = client.get("/replay", headers={"host": host}, follow_redirects=False)
    assert response.status_code == 200
    assert response.text == "<html>Academy</html>"


@pytest.mark.parametrize("host", ["play.etude.gg", "play.manabot.gg", "academy.manabot.gg"])
def test_websocket_hosts_remain_supported(client: TestClient, host: str) -> None:
    with client.websocket_connect(f"wss://{host}/ws/play") as socket:
        assert socket.receive_text() == "connected"
