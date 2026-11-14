"""Engine command line: launch the packaged Academy browser UI."""

import argparse
import errno
import json
import os
from pathlib import Path
import socket
import sys
import webbrowser

DEFAULT_URL = "http://127.0.0.1:8000"
GUIDE_URL = "https://github.com/loopflowstudio/manabot/blob/main/docs/academy.md#running-academy"


def _error(code: str, message: str) -> int:
    print(
        "ACADEMY_PLAY_ERROR "
        + json.dumps({"code": code, "message": message}, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )
    return 2


def _accepting(host: str, port: int) -> bool:
    """Whether something already answers at the address Academy would print.

    SO_REUSEADDR (kept so a restart can rebind over TIME_WAIT) lets macOS bind
    127.0.0.1 beside another program's wildcard listener, which would then lose
    its local traffic to Academy; a connection probe catches that on every
    platform.
    """
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def academy(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="managym academy",
        description=f"Open Academy at {DEFAULT_URL}. Launch guide: {GUIDE_URL}",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--traces-dir",
        type=Path,
        default=Path(os.environ.get("ACADEMY_TRACES_DIR", "academy/traces")),
    )
    parser.add_argument(
        "--no-open", action="store_true", help="Do not open the browser"
    )
    args = parser.parse_args(argv)
    try:
        import uvicorn

        from managym.academy.site import SpaStaticFiles, frontend_directory
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] in {"fastapi", "uvicorn", "starlette"}:
            return _error(
                "extra.missing",
                "Install the server dependencies: uv pip install 'managym[academy]'",
            )
        raise
    directory = frontend_directory()
    if not (directory / "index.html").is_file():
        return _error(
            "site.missing",
            "Packaged frontend missing. From a checkout use ./scripts/play, or set ACADEMY_FRONTEND_BUILD to a built site.",
        )
    traces = args.traces_dir.expanduser().resolve()
    os.environ["ACADEMY_TRACES_DIR"] = str(traces)
    from managym.academy.extensions import bot_extension
    from managym.academy.server import app

    bot_extension()
    app.mount("/", SpaStaticFiles(directory=directory, html=True), name="frontend")
    host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    url = f"http://{'[' + host + ']' if ':' in host else host}:{args.port}"

    class AcademyServer(uvicorn.Server):
        async def startup(self, sockets: list[socket.socket] | None = None) -> None:
            await super().startup(sockets=sockets)
            if self.started:
                print(
                    "ACADEMY_PLAY_READY "
                    + json.dumps(
                        {"traces_dir": str(traces), "url": url}, sort_keys=True
                    ),
                    flush=True,
                )
                print(f"  Academy: {url}\n  Ctrl-C stops the server", flush=True)
                if not args.no_open:
                    webbrowser.open(url)

    if _accepting(host, args.port):
        return _error(
            "port.in_use", f"Port {args.port} is in use; choose --port <free-port>."
        )
    try:
        # Own the listener so bind failures produce the documented error record.
        family = socket.AF_INET6 if ":" in args.host else socket.AF_INET
        with socket.socket(family) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((args.host, args.port))
            listener.listen(128)
            AcademyServer(uvicorn.Config(app, host=args.host, port=args.port)).run(
                sockets=[listener]
            )
    except KeyboardInterrupt:
        pass
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            return _error(
                "port.in_use", f"Port {args.port} is in use; choose --port <free-port>."
            )
        raise
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments[:1] == ["academy"]:
        return academy(arguments[1:])
    parser = argparse.ArgumentParser(description="managym engine tools")
    parser.add_argument("command", choices=["academy"])
    parser.parse_args(arguments)  # exits with help or a usage error
    return 2
