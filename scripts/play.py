#!/usr/bin/env python3
"""Launch the human-vs-bot table: backend + frontend, one command.

    uv run scripts/play.py [--port 8000] [--no-frontend]

Starts the FastAPI/uvicorn backend and the Vite dev server, prints the play
URL, and shuts both down together on Ctrl-C. Installs frontend deps on first
run if node_modules is missing.
"""

import argparse
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000, help="backend port")
    parser.add_argument(
        "--no-frontend", action="store_true", help="backend only (e.g. built frontend)"
    )
    args = parser.parse_args()

    procs: list[subprocess.Popen] = []

    def shutdown(*_):
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    backend = subprocess.Popen(
        [
            "uv", "run", "uvicorn", "gui.server:app",
            "--port", str(args.port),
        ],
        cwd=ROOT,
    )
    procs.append(backend)

    if not args.no_frontend:
        npm = shutil.which("npm")
        if npm is None:
            print("npm not found on PATH — install Node, or use --no-frontend", file=sys.stderr)
            shutdown()
        if not (FRONTEND / "node_modules").exists():
            print("frontend deps missing — running npm install (first run only)")
            subprocess.run([npm, "install"], cwd=FRONTEND, check=True)
        procs.append(subprocess.Popen([npm, "run", "dev"], cwd=FRONTEND))
        time.sleep(1.5)
        print(f"\n  play:   http://localhost:5173\n  replay: http://localhost:5173/replay\n  (backend on :{args.port}; Ctrl-C stops both)\n")

    # Exit when either child dies; take the other down with us.
    while True:
        for p in procs:
            if p.poll() is not None:
                print(f"process {p.args[0]} exited ({p.returncode}); shutting down")
                shutdown()
        time.sleep(1)


if __name__ == "__main__":
    raise SystemExit(main())
