"""Start Academy and prepare browser addresses for games and retained replays.

`serve` owns the one process-local server. `Academy` prepares configurations
through the server's admission path and imports evaluation evidence through the
optional bot extension; neither operation chooses a game action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import errno
import gzip
import json
import os
from pathlib import Path
import secrets
import socket
import threading
import time
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlencode
import webbrowser

if TYPE_CHECKING:
    import uvicorn

_ACTIVE_LOCK = threading.Lock()


@dataclass(frozen=True)
class RecordedGame:
    index: int
    deal_seed: int
    leg: int
    players: tuple[str, str]
    decks: tuple[str, str]
    game_trace_sha256: str
    _record: dict[str, Any] = field(repr=False, compare=False)


class AcademyLaunchError(ValueError):
    def __init__(
        self, code: str, message: str, games: list[RecordedGame] | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.games = games or []


@dataclass(frozen=True)
class Launched:
    url: str


def read_games(path: Path) -> list[RecordedGame]:
    """Read JSON, JSONL or gzip evidence without altering its retained bytes."""
    try:
        data = path.expanduser().read_bytes()
    except OSError as exc:
        code = (
            "replay.missing"
            if isinstance(exc, FileNotFoundError)
            else "replay.unreadable"
        )
        raise AcademyLaunchError(code, f"{path}: {exc}") from exc
    try:
        if data.startswith(b"\x1f\x8b"):
            data = gzip.decompress(data)
        source = data.decode("utf-8")
        try:
            value = json.loads(source)
            records = value if isinstance(value, list) else [value]
        except json.JSONDecodeError:
            records = [json.loads(line) for line in source.splitlines() if line.strip()]
        games = []
        for index, record in enumerate(records):
            if not isinstance(record, dict) or not isinstance(
                record.get("decisions"), list
            ):
                raise ValueError("expected evaluation game records with decisions")
            digest = record["game_trace_sha256"]
            if not _is_sha256(digest):
                raise ValueError("invalid game_trace_sha256")
            if type(record["deal_seed"]) is not int or type(record["leg"]) is not int:
                raise ValueError("deal_seed and leg must be integers")
            games.append(
                RecordedGame(
                    index,
                    record["deal_seed"],
                    record["leg"],
                    _seat_pair(record, "seat_players"),
                    # Interactive evaluation traces omit seat_decks, as in the
                    # browser importer; their match_hypers still bind the real lists.
                    _seat_pair(
                        {"seat_decks": ["interactive"] * 2, **record}, "seat_decks"
                    ),
                    digest,
                    record,
                )
            )
        if not games:
            raise ValueError("no game records")
        return games
    except (ValueError, KeyError, TypeError, OSError, EOFError) as exc:
        raise AcademyLaunchError(
            "replay.unreadable", f"{path}: {exc}. Supply an evaluation command trace."
        ) from exc


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _seat_pair(record: dict[str, Any], key: str) -> tuple[str, str]:
    pair = record[key]
    if (
        not isinstance(pair, list)
        or len(pair) != 2
        or not all(isinstance(v, str) for v in pair)
    ):
        raise ValueError(f"{key} must name two seats")
    return pair[0], pair[1]


def _select_game(path: Path, game: int | str | None) -> RecordedGame:
    games = read_games(path)
    if game is None:
        if len(games) == 1:
            return games[0]
        raise AcademyLaunchError(
            "replay.ambiguous",
            f"{path} holds {len(games)} games; choose one with --game.",
            games,
        )
    matches = [
        row
        for row in games
        if (
            row.index == game
            if isinstance(game, int)
            else row.game_trace_sha256.startswith(game)
        )
    ]
    if len(matches) != 1:
        raise AcademyLaunchError(
            "replay.game_unknown", f"{game!r} must select exactly one game in {path}."
        )
    return matches[0]


def _accepting(host: str, port: int) -> bool:
    # macOS permits SO_REUSEADDR binds beside a wildcard listener. Probe first
    # so Academy never takes traffic away from that listener.
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


class Academy:
    def __init__(
        self,
        url: str,
        traces_dir: Path,
        server: uvicorn.Server,
        listener: socket.socket,
    ) -> None:
        self.url = url
        self.traces_dir = traces_dir
        self._server = server
        self._listener = listener
        self._closed = False
        self._thread = threading.Thread(
            target=server.run, kwargs={"sockets": [listener]}, name="academy"
        )

    def __enter__(self) -> Academy:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise AcademyLaunchError(
                "server.closed", "This Academy has closed; call serve again."
            )

    def open(self, url: str | None = None) -> None:
        self._require_open()
        webbrowser.open(url or self.url)

    def wait(self) -> None:
        """Wait until shutdown or Ctrl-C; context exit performs the cleanup."""
        while self._thread.is_alive():
            try:
                self._thread.join(timeout=0.2)
            except KeyboardInterrupt:
                # Supervisors can forward a group SIGINT as well as the original.
                # Every interrupt requests the same shutdown; finish joining it.
                self._server.should_exit = True
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._server.should_exit = True
        self._thread.join(timeout=15)
        if self._thread.is_alive():
            raise AcademyLaunchError(
                "server.shutdown", "Academy did not stop within 15 seconds."
            )
        from managym.academy import server

        for session_id in list(server.SESSION_REGISTRY):
            server._drop_session(session_id, "server_shutdown")
        server.LAUNCH_CONFIGS.clear()
        self._listener.close()
        self._closed = True
        _ACTIVE_LOCK.release()

    def new_game(
        self,
        *,
        hero: str | Path = "ur_lessons",
        villain: str | Path = "gw_allies",
        opponent: str = "search",
        sims: int = 64,
        seed: int | None = None,
        decklists: Literal["open", "closed"] = "open",
    ) -> Launched:
        self._require_open()
        import managym
        from managym.academy import server

        for deck in (hero, villain):
            if isinstance(deck, str) and deck not in server.NAMED_DECKS:
                raise AcademyLaunchError(
                    "deck.unknown",
                    f"Unknown deck {deck!r}; bundled names: {', '.join(sorted(server.NAMED_DECKS))}",
                )
        checkpoint = _is_sha256(opponent)
        imported = isinstance(hero, Path) or isinstance(villain, Path)
        config: dict[str, Any] = {
            "villain_type": "checkpoint" if checkpoint else opponent,
            "villain_sims": sims,
            "seed": seed,
            "decklists": decklists,
        }
        if checkpoint:
            config["opponent_sha256"] = opponent
            if imported:
                raise AcademyLaunchError(
                    "setup.invalid",
                    "Checkpoint opponents do not accept imported decklists.",
                )
        if imported:
            config["deck_import"] = {
                seat: _deck_file(deck, seat)
                for seat, deck in (("hero", hero), ("villain", villain))
            }
        else:
            config.update(hero_deck=hero, villain_deck=villain)
        try:
            server._parse_game_config(config)
        except managym.DeckImportError as exc:
            raise AcademyLaunchError("deck.rejected", str(exc)) from exc
        except ValueError as exc:
            code = (
                "opponent.unavailable"
                if checkpoint and "opponent" in str(exc).lower()
                else "setup.invalid"
            )
            raise AcademyLaunchError(code, str(exc)) from exc
        launch_id = secrets.token_hex(16)
        server.LAUNCH_CONFIGS[launch_id] = config
        return Launched(f"{self.url}/?{urlencode({'launch': launch_id})}")

    def replay(self, source: str | Path, game: int | str | None = None) -> Launched:
        self._require_open()
        from fastapi import HTTPException

        from managym.academy.attempts import AttemptStore
        from managym.academy.watch import import_arena

        if isinstance(source, Path):
            selected = _select_game(source, game)
            try:
                imported = import_arena(selected._record)
            except HTTPException as exc:
                if exc.detail == "arena_import_unavailable":
                    raise AcademyLaunchError(
                        "replay.needs_manabot",
                        "Install the matching manabot release: uv pip install --only-binary :all: --torch-backend cpu manabot",
                    ) from exc
                raise AcademyLaunchError("replay.rejected", str(exc.detail)) from exc
            except Exception as exc:
                raise AcademyLaunchError(
                    "replay.rejected",
                    f"{exc}; open the trace with the revision that wrote it.",
                ) from exc
            return Launched(
                f"{self.url}/replay?{urlencode({'watch': 1, 'trace': imported['id']})}"
            )
        store = AttemptStore(self.traces_dir / "play.sqlite")
        if store.owner(source) is None:
            raise AcademyLaunchError(
                "replay.missing", f"No recording {source!r} in {self.traces_dir}."
            )
        return Launched(f"{self.url}/replay?{urlencode({'trace': source})}")


def _deck_file(deck: str | Path, seat: str) -> dict[str, str]:
    if isinstance(deck, Path):
        try:
            return {
                "source": deck.expanduser().read_text(encoding="utf-8"),
                "filename": deck.name,
            }
        except (OSError, UnicodeError) as exc:
            raise AcademyLaunchError(
                "deck.unreadable", f"{seat}: {deck}: {exc}"
            ) from exc
    from managym.academy import server

    # Parse the named setup once to retain its native sideboard as well as main.
    setup = server._parse_game_config(
        {"hero_deck": deck, "villain_deck": deck, "villain_type": "passive"}
    )
    source = "\n".join(f"{count} {name}" for name, count in setup.hero_deck.items())
    if setup.hero_sideboard:
        source += "\nSideboard\n" + "\n".join(
            f"{count} {name}" for name, count in setup.hero_sideboard.items()
        )
    return {"source": source, "filename": f"{deck}.txt"}


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    traces_dir: Path | None = None,
    open_browser: bool = True,
) -> Academy:
    """Bind and start the packaged site; the caller must close the returned server."""
    if not 0 <= port <= 65535:
        raise AcademyLaunchError("setup.invalid", "port must be between 0 and 65535.")
    if not _ACTIVE_LOCK.acquire(blocking=False):
        raise AcademyLaunchError(
            "server.running",
            "An Academy is already running in this process; close it first.",
        )
    listener: socket.socket | None = None
    academy: Academy | None = None
    try:
        try:
            import uvicorn

            from managym.academy.site import SpaStaticFiles, frontend_directory
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.split(".")[0] in {
                "fastapi",
                "uvicorn",
                "starlette",
            }:
                raise AcademyLaunchError(
                    "extra.missing",
                    "Install the server dependencies: uv pip install 'managym[academy]'",
                ) from exc
            raise
        directory = frontend_directory()
        if not (directory / "index.html").is_file():
            raise AcademyLaunchError(
                "site.missing",
                "Packaged frontend missing. From a checkout use ./scripts/play, or set ACADEMY_FRONTEND_BUILD to a built site.",
            )
        traces = (
            (traces_dir or Path(os.environ.get("ACADEMY_TRACES_DIR", "academy/traces")))
            .expanduser()
            .resolve()
        )
        from managym.academy import server, trace
        from managym.academy.extensions import bot_extension

        trace.TRACES_DIR = traces
        bot_extension()
        # Keep the hosted app and its routes; replace only this launcher's mount.
        server.app.router.routes[:] = [
            route
            for route in server.app.router.routes
            if getattr(route, "name", None) != "frontend"
        ]
        server.app.mount(
            "/", SpaStaticFiles(directory=directory, html=True), name="frontend"
        )
        visible_host = (
            "127.0.0.1" if host == "0.0.0.0" else "::1" if host == "::" else host
        )
        if port and _accepting(visible_host, port):
            raise AcademyLaunchError(
                "port.in_use", f"Port {port} is in use; choose --port <free-port>."
            )
        listener = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(128)
        address = f"[{visible_host}]" if ":" in visible_host else visible_host
        url = f"http://{address}:{listener.getsockname()[1]}"
        academy = Academy(
            url,
            traces,
            uvicorn.Server(
                uvicorn.Config(
                    server.app, host=host, port=port, timeout_graceful_shutdown=5
                )
            ),
            listener,
        )
        academy._thread.start()
        deadline = time.monotonic() + 15
        while not academy._server.started:
            if not academy._thread.is_alive() or time.monotonic() >= deadline:
                raise AcademyLaunchError(
                    "server.startup", "Academy did not start within 15 seconds."
                )
            time.sleep(0.01)
        if open_browser:
            academy.open()
        return academy
    except BaseException as exc:
        if academy is not None:
            academy.close()
        else:
            if listener is not None:
                listener.close()
            _ACTIVE_LOCK.release()
        if isinstance(exc, OSError):
            if exc.errno == errno.EADDRINUSE:
                raise AcademyLaunchError(
                    "port.in_use", f"Port {port} is in use; choose --port <free-port>."
                ) from exc
            raise AcademyLaunchError("server.startup", str(exc)) from exc
        raise
