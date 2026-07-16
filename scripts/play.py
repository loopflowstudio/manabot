#!/usr/bin/env -S uv run --python 3.12 --locked --only-group play-runtime
"""Install and launch the human-vs-bot table through the supported wrapper.

    ./scripts/play [--port 8000] [--frontend-port 5173] [--no-frontend]

The wrapper pins CPython 3.12 and locked Python dependencies. This module
validates the installed curated pack, builds the native extension when needed,
installs locked frontend dependencies, starts uvicorn and Vite, waits for both
to answer on loopback, and shuts the process group down together.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
FRONTEND_LOCK = FRONTEND / "package-lock.json"
FRONTEND_INSTALL_MARKER = FRONTEND / "node_modules" / ".etude-package-lock.sha256"
FRONTEND_REQUIRED_PATHS = (
    FRONTEND / "node_modules" / ".bin" / "svelte-kit",
    FRONTEND / "node_modules" / ".bin" / "vite",
    FRONTEND / "node_modules" / "@playwright" / "test" / "package.json",
)
PACK_DIR = FRONTEND / "src" / "lib" / "packs" / "tla-ur-lessons-vs-gw-allies" / "v1"
PACK_MANIFEST = PACK_DIR / "manifest.json"
PACK_NOTICE = PACK_DIR / "NOTICE.md"
ERROR_MARKER_ENV = "ETUDE_PLAY_ERROR_MARKER"
READY_PREFIX = "ETUDE_PLAY_READY "
ERROR_PREFIX = "ETUDE_PLAY_ERROR "
PYTHON_VERSION = (3, 12)
DEFAULT_READY_TIMEOUT = 30.0
NODE_VERSION_PATTERN = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)")
UV_RUNTIME = [
    "uv",
    "run",
    "--python",
    "3.12",
    "--locked",
    "--only-group",
    "play-runtime",
]


@dataclass(frozen=True)
class PlayError(RuntimeError):
    code: str
    message: str
    detail: str | None = None

    def payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "code": self.code,
            "message": self.message,
        }
        if self.detail:
            payload["detail"] = self.detail[-2000:]
        return payload


Runner = Callable[..., subprocess.CompletedProcess[str]]


def emit_error(error: PlayError) -> None:
    marker = os.getenv(ERROR_MARKER_ENV)
    if marker:
        try:
            Path(marker).touch()
        except OSError:
            pass
    print(
        f"{ERROR_PREFIX}{json.dumps(error.payload(), sort_keys=True)}",
        file=sys.stderr,
        flush=True,
    )


def run_text(
    argv: list[str],
    *,
    cwd: Path = ROOT,
    runner: Runner = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    return runner(
        argv,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def validate_python_version(version_info: Any = sys.version_info) -> str:
    actual = (int(version_info.major), int(version_info.minor))
    if actual != PYTHON_VERSION:
        raise PlayError(
            "python.version",
            "etude play requires CPython 3.12",
            f"running {actual[0]}.{actual[1]}",
        )
    return f"{actual[0]}.{actual[1]}.{int(version_info.micro)}"


def node_version_is_supported(major: int, minor: int) -> bool:
    return (major == 20 and minor >= 19) or (major == 22 and minor >= 12) or major >= 24


def validate_node_version(node: str, *, runner: Runner = subprocess.run) -> str:
    result = run_text([node, "--version"], runner=runner)
    if result.returncode != 0:
        raise PlayError(
            "prerequisite.node",
            "Node could not report its version",
            result.stderr.strip(),
        )
    raw = result.stdout.strip()
    match = NODE_VERSION_PATTERN.match(raw)
    if match is None:
        raise PlayError("prerequisite.node", "Node reported an invalid version", raw)
    major, minor, _ = (int(part) for part in match.groups())
    if not node_version_is_supported(major, minor):
        raise PlayError(
            "prerequisite.node",
            "Node must be 20.19+, 22.12+, or 24+",
            raw,
        )
    return raw.removeprefix("v")


def command_version(
    executable: str,
    flag: str = "--version",
    *,
    runner: Runner = subprocess.run,
) -> str:
    result = run_text([executable, flag], runner=runner)
    if result.returncode != 0:
        return "unavailable"
    return (result.stdout or result.stderr).strip().splitlines()[0]


def validate_pack() -> Any:
    if not PACK_MANIFEST.is_file():
        raise PlayError(
            "pack.missing", "the installed curated-pack manifest is missing"
        )
    if not PACK_NOTICE.is_file():
        raise PlayError("pack.notice", "the installed curated-pack notice is missing")
    try:
        from gui.curated_pack import load_curated_pack

        return load_curated_pack(PACK_MANIFEST)
    except (OSError, RuntimeError, ValueError) as error:
        raise PlayError(
            "pack.invalid",
            "the installed curated pack is invalid",
            str(error),
        ) from error


def lock_sha256(path: Path | None = None) -> str:
    target = path or FRONTEND_LOCK
    try:
        return hashlib.sha256(target.read_bytes()).hexdigest()
    except OSError as error:
        raise PlayError(
            "lock.frontend", "frontend/package-lock.json is unavailable", str(error)
        ) from error


def frontend_needs_install() -> bool:
    expected = lock_sha256()
    if any(not path.is_file() for path in FRONTEND_REQUIRED_PATHS):
        return True
    try:
        return FRONTEND_INSTALL_MARKER.read_text(encoding="utf-8").strip() != expected
    except OSError:
        return True


def ensure_frontend(npm: str, *, runner: Runner = subprocess.run) -> None:
    if not frontend_needs_install():
        return
    print("etude play: installing locked frontend dependencies", flush=True)
    result = run_text(
        [npm, "ci", "--no-audit", "--no-fund"],
        cwd=FRONTEND,
        runner=runner,
    )
    if result.returncode != 0:
        raise PlayError(
            "frontend.install",
            "npm ci failed",
            (result.stderr or result.stdout).strip(),
        )
    result = run_text(
        [npm, "exec", "--", "svelte-kit", "sync"],
        cwd=FRONTEND,
        runner=runner,
    )
    if result.returncode != 0:
        raise PlayError(
            "frontend.install",
            "the installed SvelteKit application could not be prepared",
            (result.stderr or result.stdout).strip(),
        )
    try:
        FRONTEND_INSTALL_MARKER.write_text(lock_sha256() + "\n", encoding="utf-8")
    except OSError as error:
        raise PlayError(
            "frontend.install",
            "could not record the locked frontend installation",
            str(error),
        ) from error


def native_import(*, runner: Runner = subprocess.run) -> dict[str, str] | None:
    snippet = (
        "import json, pathlib, sys; import managym._managym as native; "
        "print(json.dumps({'abi': f'cp{sys.version_info.major}{sys.version_info.minor}', "
        "'module': str(pathlib.Path(native.__file__).resolve())}))"
    )
    result = run_text(
        [*UV_RUNTIME, "python", "-c", snippet],
        runner=runner,
    )
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return None
    if payload.get("abi") != "cp312" or not payload.get("module"):
        return None
    return {"abi": payload["abi"], "module": payload["module"]}


def ensure_native(*, runner: Runner = subprocess.run) -> dict[str, str]:
    imported = native_import(runner=runner)
    if imported is not None:
        return imported

    print("etude play: building the CPython 3.12 managym extension", flush=True)
    result = run_text(
        [
            *UV_RUNTIME,
            "maturin",
            "develop",
            "--release",
            "--manifest-path",
            "managym/Cargo.toml",
            "--features",
            "python",
        ],
        runner=runner,
    )
    if result.returncode != 0:
        raise PlayError(
            "native.build",
            "the CPython 3.12 managym extension build failed",
            (result.stderr or result.stdout).strip(),
        )
    imported = native_import(runner=runner)
    if imported is None:
        raise PlayError(
            "native.import",
            "the built managym extension does not import under CPython 3.12",
        )
    return imported


def install_runtime(
    npm: str | None,
    *,
    runner: Runner = subprocess.run,
) -> dict[str, str]:
    if npm is None:
        return ensure_native(runner=runner)
    with ThreadPoolExecutor(max_workers=2) as executor:
        native_future = executor.submit(ensure_native, runner=runner)
        frontend_future = executor.submit(ensure_frontend, npm, runner=runner)
        native = native_future.result()
        frontend_future.result()
        return native


def assert_port_available(port: int, label: str) -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
            candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            candidate.bind(("127.0.0.1", port))
    except OSError as error:
        raise PlayError(
            "port.in_use",
            f"the requested {label} port is unavailable",
            f"127.0.0.1:{port}: {error}",
        ) from error


def endpoint_ready(url: str) -> bool:
    try:
        with urlopen(url, timeout=0.5) as response:
            return response.status < 500
    except HTTPError as error:
        return error.code < 500
    except (OSError, URLError):
        return False


def child_error(label: str, process: subprocess.Popen[Any]) -> PlayError:
    return PlayError(
        f"{label}.start",
        f"the {label} process exited before readiness",
        f"exit status {process.returncode}",
    )


def wait_for_readiness(
    processes: list[tuple[str, subprocess.Popen[Any]]],
    endpoints: dict[str, str],
    timeout: float,
) -> None:
    pending = dict(endpoints)
    deadline = time.monotonic() + timeout
    while pending and time.monotonic() < deadline:
        for label, process in processes:
            if process.poll() is not None:
                raise child_error(label, process)
        for label, url in list(pending.items()):
            if endpoint_ready(url):
                del pending[label]
        if pending:
            time.sleep(0.1)
    if pending:
        raise PlayError(
            "ready.timeout",
            "local services did not become ready before the deadline",
            ", ".join(f"{label}={url}" for label, url in sorted(pending.items())),
        )


def terminate_processes(processes: list[tuple[str, subprocess.Popen[Any]]]) -> None:
    for _, process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                process.terminate()

    deadline = time.monotonic() + 10
    for _, process in processes:
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                process.kill()
            process.wait()


def start_processes(
    backend_port: int,
    frontend_port: int,
    npm: str | None,
) -> list[tuple[str, subprocess.Popen[Any]]]:
    processes: list[tuple[str, subprocess.Popen[Any]]] = []
    try:
        backend = subprocess.Popen(
            [
                *UV_RUNTIME,
                "uvicorn",
                "gui.server:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(backend_port),
            ],
            cwd=ROOT,
            start_new_session=True,
        )
        processes.append(("backend", backend))
    except OSError as error:
        raise PlayError(
            "backend.start", "could not start the local backend", str(error)
        ) from error

    if npm is not None:
        try:
            environment = dict(os.environ)
            environment["ETUDE_API_PORT"] = str(backend_port)
            frontend = subprocess.Popen(
                [
                    npm,
                    "run",
                    "dev",
                    "--",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(frontend_port),
                    "--strictPort",
                ],
                cwd=FRONTEND,
                env=environment,
                start_new_session=True,
            )
            processes.append(("frontend", frontend))
        except OSError as error:
            terminate_processes(processes)
            raise PlayError(
                "frontend.start", "could not start the local frontend", str(error)
            ) from error
    return processes


def build_ready_payload(
    *,
    started_at: float,
    backend_port: int,
    frontend_port: int,
    python_version: str,
    node_version: str | None,
    npm_version: str | None,
    native: dict[str, str],
    asset_pack: dict[str, str],
    now: float | None = None,
) -> dict[str, Any]:
    has_frontend = node_version is not None
    play_url = (
        f"http://127.0.0.1:{frontend_port}"
        if has_frontend
        else f"http://127.0.0.1:{backend_port}"
    )
    return {
        "schema_version": 1,
        "url": play_url,
        "replay_url": f"{play_url}/replay" if has_frontend else None,
        "backend_url": f"http://127.0.0.1:{backend_port}",
        "elapsed_ms": round(
            ((now if now is not None else time.monotonic()) - started_at) * 1000, 1
        ),
        "python": python_version,
        "node": node_version,
        "npm": npm_version,
        "native": native,
        "asset_pack": asset_pack,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000, help="backend port")
    parser.add_argument("--frontend-port", type=int, default=5173, help="frontend port")
    parser.add_argument(
        "--no-frontend", action="store_true", help="backend only (e.g. built frontend)"
    )
    parser.add_argument(
        "--ready-timeout",
        type=float,
        default=DEFAULT_READY_TIMEOUT,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def run_launcher(args: argparse.Namespace) -> int:
    started_at = time.monotonic()
    python_version = validate_python_version()
    pack = validate_pack()

    npm: str | None = None
    node_version: str | None = None
    npm_version: str | None = None
    if not args.no_frontend:
        node = shutil.which("node")
        npm = shutil.which("npm")
        if node is None:
            raise PlayError("prerequisite.node", "Node is not available on PATH")
        if npm is None:
            raise PlayError("prerequisite.npm", "npm is not available on PATH")
        node_version = validate_node_version(node)
        npm_version = command_version(npm)

    native = install_runtime(npm)

    assert_port_available(args.port, "backend")
    if npm is not None:
        assert_port_available(args.frontend_port, "frontend")

    processes = start_processes(args.port, args.frontend_port, npm)
    stop_requested = False

    def request_stop(*_: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_handlers = {
        signal_number: signal.signal(signal_number, request_stop)
        for signal_number in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        endpoints = {"backend": f"http://127.0.0.1:{args.port}/api/traces"}
        if npm is not None:
            endpoints["frontend"] = f"http://127.0.0.1:{args.frontend_port}/"
        wait_for_readiness(processes, endpoints, args.ready_timeout)

        ready = build_ready_payload(
            started_at=started_at,
            backend_port=args.port,
            frontend_port=args.frontend_port,
            python_version=python_version,
            node_version=node_version,
            npm_version=npm_version,
            native=native,
            asset_pack=pack.reference,
        )
        play_url = ready["url"]
        print(f"{READY_PREFIX}{json.dumps(ready, sort_keys=True)}", flush=True)
        print(f"\n  play:   {play_url}", flush=True)
        if npm is not None:
            print(f"  replay: {play_url}/replay", flush=True)
        print("  Ctrl-C stops all local processes\n", flush=True)

        while not stop_requested:
            for label, process in processes:
                if process.poll() is not None:
                    raise child_error(label, process)
            time.sleep(0.25)
        return 0
    finally:
        terminate_processes(processes)
        for signal_number, handler in previous_handlers.items():
            signal.signal(signal_number, handler)


def main(argv: list[str] | None = None) -> int:
    try:
        return run_launcher(parse_args(argv))
    except PlayError as error:
        emit_error(error)
        return 2
    except KeyboardInterrupt:
        return 130
    except Exception as error:  # deterministic boundary for unexpected startup failures
        emit_error(
            PlayError("launcher.internal", "unexpected launcher failure", str(error))
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
