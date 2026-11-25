"""Start Academy and prepare browser addresses for games and retained replays.

`serve` owns the one process-local server. `Academy` prepares configurations
through the server's admission path and imports game files: managym game
records by re-execution, and retained Command traces through the optional bot
extension; neither operation chooses a game action.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
import errno
import gzip
import json
import os
from pathlib import Path
import random
import secrets
import socket
import threading
import time
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlencode
import warnings
import webbrowser

import managym
from managym.academy.deck_pools import (
    DeckPool,
    NamedDeck,
    deck_file_payload,
    is_pool_key,
    resolve_deck_pool,
    set_deck_pools,
)
from managym.disclosure import DecklistDisclosure
from managym.policy import (
    Policy,
    PolicyCallable,
    PolicySource,
    PolicySpec,
    is_python_policy_ref,
    load_python_policy,
)
from managym.search import SearchConfig, SearchError, search_descriptor

DeckInput = str | Path | managym.Deck | NamedDeck | tuple[str, managym.Deck]
PlayerInput = str | Path | Policy | PolicyCallable | PolicySpec
TurnOrder = Literal["play", "random", "draw"]
_GENERIC_DECK_STEMS = frozenset({"deck", "hero", "villain"})

if TYPE_CHECKING:
    import uvicorn

_ACTIVE_LOCK = threading.Lock()
_DECK_SUFFIXES = {".txt", ".dek"}
_TRACE_SUFFIXES = {".json", ".jsonl", ".gz"}


@dataclass(frozen=True)
class RecordedGame:
    index: int
    seed: int
    stream: int
    players: tuple[str, str]
    decisions: int
    ended: str
    digest: str
    _record: dict[str, Any] = field(repr=False, compare=False)

    @property
    def deal_seed(self) -> int:
        _warn_listing_field("deal_seed", "seed")
        return self.seed

    @property
    def leg(self) -> int:
        _warn_listing_field("leg", "stream")
        return self.stream

    @property
    def game_trace_sha256(self) -> str:
        _warn_listing_field("game_trace_sha256", "digest")
        return self.digest

    @property
    def decks(self) -> tuple[str, str]:
        _warn_listing_field("decks", "the original record's setups")
        if "seat_decks" in self._record:
            return _seat_pair(self._record, "seat_decks")
        games = self._record.get("games")
        if games:
            names = [player["name"] for player in games[0]["players"]]
            return names[0], names[1]
        return "interactive", "interactive"


def _warn_listing_field(name: str, replacement: str) -> None:
    warnings.warn(
        f"RecordedGame.{name} is deprecated; use {replacement}; removal in 0.3.0",
        DeprecationWarning,
        stacklevel=3,
    )


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


def _path_or_name(
    value: str, suffixes: set[str], names: Collection[str] = ()
) -> str | Path:
    """A known name is never a file; otherwise an existing or path-shaped value is one."""
    if value in names:
        return value
    path = Path(value).expanduser()
    return path if path.exists() or "/" in value or path.suffix in suffixes else value


def _file_hint(value: str, suffixes: set[str]) -> str:
    if isinstance(_path_or_name(value, suffixes), Path):
        return f" Pass pathlib.Path({value!r}) for a file."
    return ""


def read_games(path: Path) -> list[RecordedGame]:
    """Read a JSON, JSONL or gzip game file without altering its retained bytes."""
    if path.is_dir():
        games: list[RecordedGame] = []
        for child in sorted(path.glob("*.jsonl")):
            for game in read_games(child):
                games.append(replace(game, index=len(games)))
        if not games:
            raise AcademyLaunchError("replay.unreadable", f"{path}: no game records")
        return games
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
        from managym.recording import records_from_source

        games = []
        for record in records:
            native_records = records_from_source(record, journal_path=path)
            if native_records is not None:
                for native in native_records:
                    inputs = native.games[0]
                    names = tuple(p.name for p in inputs.players)
                    if len(names) != 2:
                        raise ValueError("record setup unavailable")
                    games.append(
                        RecordedGame(
                            index=len(games),
                            seed=inputs.seed,
                            stream=inputs.stream,
                            players=(names[0], names[1]),
                            decisions=len(inputs.actions),
                            ended=inputs.end.status if inputs.end else "interrupted",
                            digest=native.sha256,
                            _record=native.model_dump(mode="json", by_alias=True),
                        )
                    )
                # A journal's subsequent rows are changes, not separate games.
                if record.get("schema") == "training-game-checkpoint-v1":
                    break
                continue
            if not isinstance(record, dict) or not isinstance(
                record.get("decisions"), list
            ):
                raise ValueError("expected game records with decisions")
            digest = record["game_trace_sha256"]
            if not _is_sha256(digest):
                raise ValueError("invalid game_trace_sha256")
            if type(record["deal_seed"]) is not int or type(record["leg"]) is not int:
                raise ValueError("deal_seed and leg must be integers")
            games.append(
                RecordedGame(
                    index=len(games),
                    seed=record["deal_seed"],
                    stream=record["leg"],
                    players=_seat_pair(record, "seat_players"),
                    decisions=len(record["decisions"]),
                    ended="complete" if record.get("terminated") else "interrupted",
                    digest=digest,
                    _record=record,
                )
            )
        if not games:
            raise ValueError("no game records")
        return games
    except (ValueError, KeyError, TypeError, OSError, EOFError) as exc:
        raise AcademyLaunchError(
            "replay.unreadable",
            f"{path}: {exc}. Supply a managym game record or a retained evaluation command trace.",
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
            f"{path} holds {len(games)} games; select one by index or trace digest.",
            games,
        )
    # A decimal string is a position when the file has one, else a digest prefix.
    if isinstance(game, str) and game.isdecimal() and int(game) < len(games):
        game = int(game)
    matches = [
        row
        for row in games
        if (row.index == game if isinstance(game, int) else row.digest.startswith(game))
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
        from managym.academy.opponent import set_policy_registry

        set_policy_registry(None)
        set_deck_pools(None)
        _ACTIVE_LOCK.release()

    def _deck_config(
        self,
        deck: DeckInput,
        opponent_deck: DeckInput,
    ) -> dict[str, Any]:
        from managym.academy import server

        seats = (
            ("hero", deck, "your deck"),
            ("villain", opponent_deck, "opponent deck"),
        )
        normalized: list[DeckInput] = []
        for _, selected, label in seats:
            if isinstance(selected, str):
                if selected in server.NAMED_DECKS:
                    normalized.append(selected)
                elif (pool := resolve_deck_pool(selected)) is not None:
                    normalized.append(pool.key)
                elif is_pool_key(selected):
                    raise AcademyLaunchError(
                        "deck.unknown",
                        f"{label.capitalize()} {selected!r} is not a registered deck pool.",
                    )
                else:
                    raise AcademyLaunchError(
                        "deck.unknown",
                        f"{label.capitalize()} {selected!r} is not a bundled deck; "
                        f"bundled names: {', '.join(sorted(server.NAMED_DECKS))}."
                        + _file_hint(selected, _DECK_SUFFIXES),
                    )
            else:
                normalized.append(selected)
        if all(isinstance(item, str) for item in normalized):
            return {"hero_deck": normalized[0], "villain_deck": normalized[1]}
        return {
            "deck_import": {
                seat: _deck_file(selected, label)
                for (seat, _, label), selected in zip(seats, normalized, strict=True)
            }
        }

    def opponents(self) -> list[dict[str, Any]]:
        """Return the server's unified opponent catalog."""
        self._require_open()
        from managym.academy.opponent import opponent_catalog

        return list(opponent_catalog()["opponents"])

    def new_game(
        self,
        *,
        deck: DeckInput = "ur_lessons",
        opponent_deck: DeckInput = "gw_allies",
        opponent: PlayerInput = "search",
        sims: int = 64,
        seed: int | None = None,
        decklists: DecklistDisclosure = "open",
        search: SearchConfig | None = None,
        order: TurnOrder | None = None,
        human_seat: int | None = None,
    ) -> Launched:
        self._require_open()
        from managym.academy import server
        from managym.academy.villain import player_fields

        if order is not None and human_seat is not None:
            raise AcademyLaunchError(
                "setup.invalid", "Pass either order or human_seat, not both."
            )
        if human_seat is not None and human_seat not in (0, 1):
            raise AcademyLaunchError("setup.invalid", "human_seat must be 0 or 1.")
        if order is not None:
            if order == "play":
                resolved_seat = 0
            elif order == "draw":
                resolved_seat = 1
            elif order == "random":
                resolved_seat = random.SystemRandom().randrange(2)
            else:
                raise AcademyLaunchError(
                    "setup.invalid", "order must be 'play', 'random', or 'draw'."
                )
        else:
            resolved_seat = human_seat

        deck_fields = self._deck_config(deck, opponent_deck)
        opponent = self._player(opponent, "opponent.unavailable")
        checkpoint = opponent.startswith("policy:")
        config: dict[str, Any] = {
            "villain_sims": sims,
            "seed": seed,
            "decklists": decklists,
            "mirror": asdict(search) if search is not None else None,
            **player_fields(opponent),
            **deck_fields,
        }
        if resolved_seat is not None:
            config["human_seat"] = resolved_seat
        try:
            server._parse_game_config(config)
        except managym.DeckImportError as exc:
            raise AcademyLaunchError("deck.rejected", str(exc)) from exc
        except (ValueError, SearchError) as exc:
            code = (
                "opponent.unavailable"
                if checkpoint and "opponent" in str(exc).lower()
                else "setup.invalid"
            )
            raise AcademyLaunchError(code, str(exc)) from exc
        launch_id = secrets.token_hex(16)
        server.LAUNCH_CONFIGS[launch_id] = config
        return Launched(f"{self.url}/?{urlencode({'launch': launch_id})}")

    def _player(self, value: PlayerInput, code: str) -> str:
        from managym.academy.opponent import policy_registry

        try:
            registry = policy_registry()
            if isinstance(value, Path):
                if value.suffix == ".py":
                    return f"policy:{registry.register(load_python_policy(str(value)))}"
                return f"policy:{registry.register(PolicySpec.read(value))}"
            if isinstance(value, PolicySpec) or not isinstance(value, str):
                return f"policy:{registry.register(value)}"
            if is_python_policy_ref(value):
                return f"policy:{registry.register(load_python_policy(value))}"
            if value.endswith(".json"):
                return f"policy:{registry.register(PolicySpec.read(Path(value).expanduser()))}"
            key = value.removeprefix("policy:")
            if (
                key in registry.aliases
                or value.startswith("policy:")
                or _is_sha256(key)
            ):
                return f"policy:{registry.descriptor(key).id}"
            if value.startswith("search-"):
                if not 1 <= int(value[7:]) <= 4096:
                    raise ValueError("Search playout count must be between 1 and 4096.")
            elif value not in ("random", "search", "passive"):
                raise ValueError(
                    f"Player {value!r} is unavailable. Register a policy object, pass path/to/bot.py:func or policy.json, or choose a catalog id."
                )
            return value
        except (OSError, ValueError, ImportError) as error:
            raise AcademyLaunchError(code, str(error)) from error

    def watch(
        self,
        *,
        players: tuple[PlayerInput, PlayerInput],
        search: tuple[SearchConfig | None, SearchConfig | None] = (None, None),
        decks: tuple[DeckInput, DeckInput] | None = None,
        seed: int | None = None,
        decklists: DecklistDisclosure = "open",
    ) -> Launched:
        """Open Watch setup with exact seats selected, without starting a game."""
        self._require_open()
        seats = [self._player(player, "player.unavailable") for player in players]
        from managym.academy import server
        from managym.academy.opponent import policy_registry
        from managym.academy.villain import player_fields

        for player, spec in zip(seats, search, strict=True):
            if spec is not None:
                if not player.startswith("policy:"):
                    raise AcademyLaunchError(
                        "setup.invalid",
                        "Search requires a registered search-capable policy.",
                    )
                try:
                    search_descriptor(
                        policy_registry().descriptor(player.removeprefix("policy:")),
                        spec,
                    )
                except SearchError as exc:
                    raise AcademyLaunchError("setup.invalid", str(exc)) from exc
        query: dict[str, str] = {"watch": "1", "seat1": seats[0], "seat2": seats[1]}
        if any(search):
            query["search"] = json.dumps(
                [asdict(spec) if spec else None for spec in search]
            )
        if decks is not None or seed is not None or decklists != "open":
            deck_fields = (
                self._deck_config(decks[0], decks[1])
                if decks is not None
                else {"hero_deck": "ur_lessons", "villain_deck": "gw_allies"}
            )
            setup: dict[str, Any] = {
                "seed": seed,
                "decklists": decklists,
                **deck_fields,
            }
            try:
                for player, spec in zip(seats, search, strict=True):
                    server._parse_game_config(
                        {
                            **setup,
                            "mirror": asdict(spec) if spec is not None else None,
                            **player_fields(player),
                        }
                    )
            except managym.DeckImportError as exc:
                raise AcademyLaunchError("deck.rejected", str(exc)) from exc
            except (ValueError, SearchError) as exc:
                raise AcademyLaunchError("setup.invalid", str(exc)) from exc
            launch_id = secrets.token_hex(16)
            server.LAUNCH_CONFIGS[launch_id] = setup
            query["launch"] = launch_id
        return Launched(f"{self.url}/replay?{urlencode(query)}")

    def replay(self, source: str | Path, game: int | str | None = None) -> Launched:
        self._require_open()
        from fastapi import HTTPException

        from managym.academy.attempts import AttemptStore
        from managym.academy.watch import import_game

        if isinstance(source, Path):
            selected = _select_game(source, game)
            try:
                imported = import_game(selected._record)
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
                "replay.missing",
                f"No recording {source!r} in {self.traces_dir}."
                + _file_hint(source, _TRACE_SUFFIXES),
            )
        if game is not None:
            raise AcademyLaunchError(
                "replay.game_unknown",
                "A recording id names one game; a game selection applies only within a trace file.",
            )
        return Launched(f"{self.url}/replay?{urlencode({'trace': source})}")


def _deck_file(deck: DeckInput, seat: str) -> dict[str, str]:
    if isinstance(deck, NamedDeck):
        return deck_file_payload(deck.deck, name=deck.name)
    if (
        isinstance(deck, tuple)
        and len(deck) == 2
        and isinstance(deck[0], str)
        and isinstance(deck[1], managym.Deck)
    ):
        named = NamedDeck(deck[0], deck[1])
        return deck_file_payload(named.deck, name=named.name)
    if isinstance(deck, managym.Deck):
        return deck_file_payload(deck, name=f"Deck {deck.identity[:8]}")
    if isinstance(deck, Path):
        try:
            payload: dict[str, str] = {
                "source": deck.expanduser().read_text(encoding="utf-8"),
                "filename": deck.name,
            }
        except (OSError, UnicodeError) as exc:
            raise AcademyLaunchError(
                "deck.unreadable", f"{seat}: {deck}: {exc}"
            ) from exc
        if deck.stem.lower() not in _GENERIC_DECK_STEMS:
            payload["name"] = deck.stem
        return payload
    if isinstance(deck, str) and (pool := resolve_deck_pool(deck)) is not None:
        import random as _random

        sampled = pool.sample(_random.SystemRandom())
        return deck_file_payload(sampled.deck, name=sampled.name)
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
    policies: Mapping[str, Policy | PolicyCallable | PolicySpec] | None = None,
    sources: Sequence[PolicySource] = (),
    deck_pools: Sequence[DeckPool] | None = None,
    discover: bool = True,
) -> Academy:
    """Start the site without opening a browser; the caller must close the server."""
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
                "Packaged frontend missing (git/source installs omit the built UI). "
                "Install a published managym[academy] wheel, run ./scripts/play from "
                "a manabot checkout, or set ACADEMY_FRONTEND_BUILD to a built academy/ui directory.",
            )
        traces = (
            (traces_dir or Path(os.environ.get("ACADEMY_TRACES_DIR", "academy/traces")))
            .expanduser()
            .resolve()
        )
        from managym.academy import server, trace
        from managym.academy.opponent import policy_registry, set_policy_registry
        from managym.policy import PolicyRegistry

        trace.TRACES_DIR = traces
        if discover:
            set_policy_registry(None)
            registry = policy_registry()
        else:
            registry = PolicyRegistry(providers={})
            set_policy_registry(registry)
        try:
            set_deck_pools(deck_pools)
        except ValueError as exc:
            raise AcademyLaunchError("setup.invalid", str(exc)) from exc
        try:
            if discover or sources:
                registry.discover(sources)
            for name, policy in (policies or {}).items():
                registry.register(policy, name=name)
        except (ImportError, OSError, ValueError) as exc:
            raise AcademyLaunchError("policy.unavailable", str(exc)) from exc
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
        return academy
    except BaseException as exc:
        if academy is not None:
            academy.close()
        else:
            if listener is not None:
                listener.close()
            set_deck_pools(None)
            _ACTIVE_LOCK.release()
        if isinstance(exc, OSError):
            if exc.errno == errno.EADDRINUSE:
                raise AcademyLaunchError(
                    "port.in_use", f"Port {port} is in use; choose --port <free-port>."
                ) from exc
            raise AcademyLaunchError("server.startup", str(exc)) from exc
        raise


def play(
    *,
    opponent: PlayerInput | None = None,
    deck: DeckInput | None = None,
    opponent_deck: DeckInput | None = None,
    order: TurnOrder = "random",
    decklists: DecklistDisclosure | None = None,
    seed: int | None = None,
    sims: int = 64,
    search: SearchConfig | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    traces_dir: Path | None = None,
    policies: Mapping[str, Policy | PolicyCallable | PolicySpec] | None = None,
    deck_pools: Sequence[DeckPool] | None = None,
    discover: bool = True,
    open_browser: bool = True,
) -> None:
    """Serve Academy, launch a game (blocking until Ctrl-C), and open the browser.

    Accepts a plain ``(obs) -> int | probs`` function, a ``Policy`` object, a
    ``./bot.py:func`` string, or a catalog id as ``opponent``.
    """
    from managym.academy.deck_pools import ensure_default_deck_pools
    from managym.academy.opponent import default_opponent_id

    pools = list(deck_pools) if deck_pools is not None else ensure_default_deck_pools()
    with serve(
        host=host,
        port=port,
        traces_dir=traces_dir,
        policies=policies,
        deck_pools=pools or None,
        discover=discover,
    ) as running:
        has_trophies = resolve_deck_pool("pool:held-out") is not None
        chosen_opponent = opponent if opponent is not None else default_opponent_id()
        chosen_deck = (
            deck
            if deck is not None
            else ("pool:held-out" if has_trophies else "ur_lessons")
        )
        chosen_opp_deck = (
            opponent_deck
            if opponent_deck is not None
            else ("pool:held-out" if has_trophies else "gw_allies")
        )
        chosen_decklists: DecklistDisclosure = (
            decklists
            if decklists is not None
            else (
                "closed"
                if (is_pool_key(chosen_deck) or is_pool_key(chosen_opp_deck))
                and not (
                    isinstance(chosen_opponent, str)
                    and (
                        chosen_opponent == "search"
                        or chosen_opponent.startswith("search-")
                    )
                )
                else "open"
            )
        )
        target = running.new_game(
            deck=chosen_deck,
            opponent_deck=chosen_opp_deck,
            opponent=chosen_opponent,
            sims=sims,
            seed=seed,
            decklists=chosen_decklists,
            search=search,
            order=order,
        ).url
        if open_browser:
            running.open(target)
        running.wait()


def watch(
    player1: PlayerInput,
    player2: PlayerInput,
    *,
    decks: tuple[DeckInput, DeckInput] | None = None,
    decklists: DecklistDisclosure | None = None,
    seed: int | None = None,
    search: tuple[SearchConfig | None, SearchConfig | None] = (None, None),
    host: str = "127.0.0.1",
    port: int = 8000,
    traces_dir: Path | None = None,
    policies: Mapping[str, Policy | PolicyCallable | PolicySpec] | None = None,
    deck_pools: Sequence[DeckPool] | None = None,
    discover: bool = True,
    open_browser: bool = True,
) -> None:
    """Serve Academy and open a Bot-vs-Bot Watch setup (blocking until Ctrl-C)."""
    from managym.academy.deck_pools import ensure_default_deck_pools

    pools = list(deck_pools) if deck_pools is not None else ensure_default_deck_pools()
    with serve(
        host=host,
        port=port,
        traces_dir=traces_dir,
        policies=policies,
        deck_pools=pools or None,
        discover=discover,
    ) as running:
        has_trophies = resolve_deck_pool("pool:held-out") is not None
        chosen_decks = (
            decks
            if decks is not None
            else (("pool:held-out", "pool:held-out") if has_trophies else None)
        )
        chosen_decklists: DecklistDisclosure = (
            decklists
            if decklists is not None
            else (
                "closed"
                if chosen_decks is not None
                and (is_pool_key(chosen_decks[0]) or is_pool_key(chosen_decks[1]))
                else "open"
            )
        )
        target = running.watch(
            players=(player1, player2),
            search=search,
            decks=chosen_decks,
            seed=seed,
            decklists=chosen_decklists,
        ).url
        if open_browser:
            running.open(target)
        running.wait()
