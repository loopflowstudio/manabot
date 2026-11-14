"""Exercise Academy from an installed engine wheel, outside the checkout.

The release installer supplies the extra. This proof needs only that runtime:
no test framework, browser, training distribution or build tool is imported.
"""

from importlib.metadata import distributions
import json
import os
from pathlib import Path
from queue import Queue
import re
import signal
import socket
import subprocess
import sys
import tempfile
from threading import Thread
from urllib.request import urlopen

from websockets.sync.client import connect

import managym

INSTALL_COMMAND = (
    "uv pip install --python .venv/bin/python --only-binary :all: 'managym[academy]'"
)
LAUNCH_COMMAND = "uv run --no-project managym academy"
DEFAULT_URL = "http://127.0.0.1:8000"


def main() -> None:
    prefix = Path(sys.prefix).resolve()
    assert Path(managym.__file__).resolve().is_relative_to(prefix)
    assert Path(managym._managym.__file__).resolve().is_relative_to(prefix)
    installed = {d.metadata["Name"].lower() for d in distributions()}
    assert not installed.intersection({"manabot", "torch", "etude-sdk"})
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="academy-wheel-") as directory:
        ready: Queue[str] = Queue()
        log: list[str] = []
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "managym.academy",
                "--no-open",
                "--port",
                str(port),
                "--traces-dir",
                directory,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            # Ignore checkout overrides: this proof must serve installed bytes.
            env={**os.environ, "PYTHONPATH": "", "ACADEMY_FRONTEND_BUILD": ""},
        )

        def read_output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                log.append(line)
                if line.startswith("ACADEMY_PLAY_READY "):
                    ready.put(line)

        reader = Thread(target=read_output, daemon=True)
        reader.start()
        try:
            record = json.loads(
                ready.get(timeout=30).removeprefix("ACADEMY_PLAY_READY ")
            )
            url = f"http://127.0.0.1:{port}"
            assert record == {"traces_dir": str(Path(directory).resolve()), "url": url}

            def get(path: str) -> bytes:
                with urlopen(url + path, timeout=10) as response:
                    return response.read()

            shell = get("/").decode()
            assert get("/replay") == shell.encode()
            assets = re.findall(r'[^"\s<>]*_app/immutable/[^"\s<>]+', shell)
            assert assets, shell
            assert get("/" + assets[0].removeprefix("./").lstrip("/"))
            catalog = json.loads(get("/api/opponents"))
            assert catalog["extension"] is None and catalog["opponents"] == []
            with connect(
                url.replace("http:", "ws:") + "/ws/play", open_timeout=10
            ) as ws:
                ws.send(
                    json.dumps(
                        {
                            "type": "new_game",
                            "config": {
                                "hero_deck": {"Mountain": 1, "Gray Ogre": 1},
                                "villain_deck": {"Forest": 1, "Llanowar Elves": 1},
                                "villain_type": "search",
                                "villain_sims": 1,
                                "seed": 7,
                            },
                        }
                    )
                )
                payload = json.loads(ws.recv(timeout=30))
                frame = payload["frame"]
                for step in range(300):
                    if frame["projection"]["game_over"]:
                        break
                    offer = frame["offers"][0]
                    ws.send(
                        json.dumps(
                            {
                                "type": "command",
                                "command": {
                                    "command_id": f"wheel-{step}",
                                    "match_id": frame["match_id"],
                                    "expected_revision": frame["revision"],
                                    "prompt_id": frame["prompt"]["id"],
                                    "offer_id": offer["id"],
                                    "answers": [],
                                },
                            }
                        )
                    )
                    payload = json.loads(ws.recv(timeout=30))
                    assert payload["status"] == "accepted", payload
                    frame = payload["update"]["frame"]
                else:
                    raise AssertionError("Search game exceeded decision bound")
            traces = json.loads(get("/api/traces"))
            assert isinstance(traces, list) and traces
            trace_id = traces[0]["id"]
            replay = json.loads(get(f"/api/traces/{trace_id}/decisions"))
            assert replay
        finally:
            process.send_signal(signal.SIGINT)
            try:
                status = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise
            reader.join(timeout=5)
            if status != 0:
                raise AssertionError(f"Academy exited {status}: {''.join(log)}")
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", port))
    print(
        "ACADEMY_WHEEL_PROOF "
        + json.dumps(
            {
                "manabot": False,
                "torch": False,
                "site": True,
                "played": True,
                "replay": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
