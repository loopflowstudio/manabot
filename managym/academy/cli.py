"""`managym` command line: `managym play`, `managym watch`, `managym replay`, `managym opponents`.

Top-level commands and the `managym academy` alias share the exact same option
definitions and server launch lifecycle. `managym.academy.launch.serve` owns
the server, preparation and every error code. Typer belongs to the `academy`
extra, so `managym.cli` imports this module lazily.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path

from pydantic import TypeAdapter
import typer

from managym.academy.deck_pools import (
    ensure_default_deck_pools,
    is_pool_key,
    resolve_deck_pool,
)
from managym.academy.launch import (
    _DECK_SUFFIXES,
    _TRACE_SUFFIXES,
    Academy,
    AcademyLaunchError,
    TurnOrder,
    _path_or_name,
    serve,
)
from managym.academy.opponent import (
    default_opponent_id,
    opponent_catalog,
    policy_settings_fit,
)
from managym.disclosure import DecklistDisclosure
from managym.policy import PolicySpec, is_python_policy_ref
from managym.search import SearchConfig

DEFAULT_URL = "http://127.0.0.1:8000"
GUIDE_URL = "https://github.com/loopflowstudio/manabot/blob/main/docs/academy.md#running-academy"
SHOWN_GAMES = 20

_Prepare = Callable[[Academy], str]
"""Returns the address READY reports and the browser opens."""

app = typer.Typer(
    help="managym engine & Academy tools (`managym play`, `managym watch`, `managym replay`, `managym opponents`)",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
academy_app = typer.Typer(
    help=f"Open Academy at {DEFAULT_URL}. Launch guide: {GUIDE_URL}",
    invoke_without_command=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
app.add_typer(academy_app, name="academy")


@dataclass(frozen=True)
class _Shared:
    """Shared options as given; `None` means not given at this position."""

    host: str | None = None
    port: int | None = None
    traces_dir: Path | None = None
    no_open: bool | None = None
    policies: list[Path] | None = None
    discover: bool | None = None

    def over(self, earlier: "_Shared") -> "_Shared":
        return _Shared(
            host=self.host if self.host is not None else earlier.host,
            port=self.port if self.port is not None else earlier.port,
            traces_dir=self.traces_dir
            if self.traces_dir is not None
            else earlier.traces_dir,
            no_open=self.no_open if self.no_open is not None else earlier.no_open,
            policies=[*(earlier.policies or []), *(self.policies or [])],
            discover=self.discover if self.discover is not None else earlier.discover,
        )


_HOST = typer.Option(None, "--host", help="Address to listen on (default 127.0.0.1)")
_PORT = typer.Option(None, "--port", help="Port to listen on (default 8000)")
_TRACES_DIR = typer.Option(
    None,
    "--traces-dir",
    help="Where games are saved (default ACADEMY_TRACES_DIR or ./academy/traces)",
)
_POLICIES = typer.Option(
    None,
    "--policy",
    help="Saved policy specification JSON (repeatable; alias is its file stem)",
)
_NO_OPEN = typer.Option(None, "--no-open", help="Do not open a browser window")
_NO_DISCOVER = typer.Option(
    None,
    "--no-discover",
    help="Skip ambient policy provider discovery (use only explicit policies)",
)


def _earlier(ctx: typer.Context) -> _Shared:
    return ctx.obj if isinstance(ctx.obj, _Shared) else _Shared()


@academy_app.callback()
def _academy(
    ctx: typer.Context,
    host: str | None = _HOST,
    port: int | None = _PORT,
    traces_dir: Path | None = _TRACES_DIR,
    no_open: bool | None = _NO_OPEN,
    policies: list[Path] | None = _POLICIES,
    no_discover: bool | None = _NO_DISCOVER,
) -> None:
    shared = _Shared(
        host,
        port,
        traces_dir,
        no_open,
        policies,
        False if no_discover else None,
    )
    if ctx.invoked_subcommand is None:
        _run(shared, lambda running: running.url)
    ctx.obj = shared


def _resolve_cli_deck(raw: str | None, fallback_named: str) -> str | Path:
    from managym.academy import server

    if raw is None:
        return (
            "pool:held-out"
            if resolve_deck_pool("pool:held-out") is not None
            else fallback_named
        )
    if raw in ("trophies", "trophies:held-out"):
        return "pool:held-out"
    if raw == "trophies:train":
        return "pool:train"
    if is_pool_key(raw) or resolve_deck_pool(raw) is not None:
        return raw
    return _path_or_name(raw, _DECK_SUFFIXES, server.NAMED_DECKS)


def _play_cmd(
    ctx: typer.Context,
    deck: str | None = typer.Option(
        None,
        help="Your deck: bundled name, deck pool (pool:held-out), or decklist file (default: held-out TLA trophies if available, else ur_lessons)",
    ),
    opponent_deck: str | None = typer.Option(
        None,
        help="Opponent deck, named the same way (default: held-out TLA trophies if available, else gw_allies)",
    ),
    opponent: str | None = typer.Option(
        None,
        help="Catalog id, alias (manabot, heuristic, search-64), ./bot.py:func, or policy JSON (default: hosted policy -> heuristic -> search-64)",
    ),
    order: TurnOrder = typer.Option(
        "random",
        help="Your turn order: play, random, or draw",
    ),
    sims: int = typer.Option(64, help="Random playouts per legal action for search"),
    seed: int | None = typer.Option(
        None,
        help="Deal seed; the same seed and decks give the same deal (default random)",
    ),
    decklists: DecklistDisclosure | None = typer.Option(
        None,
        help="Whether each player knows the other's list (open or closed)",
    ),
    search_json: str | None = typer.Option(
        None,
        "--search",
        help="Search settings as a JSON object; requires a search-capable policy.",
    ),
    host: str | None = _HOST,
    port: int | None = _PORT,
    traces_dir: Path | None = _TRACES_DIR,
    no_open: bool | None = _NO_OPEN,
    policies: list[Path] | None = _POLICIES,
    no_discover: bool | None = _NO_DISCOVER,
) -> None:
    def prepare(running: Academy) -> str:
        try:
            search = (
                TypeAdapter(SearchConfig).validate_json(search_json)
                if search_json
                else None
            )
        except ValueError as exc:
            raise AcademyLaunchError(
                "setup.invalid", f"Invalid Search spec: {exc}"
            ) from exc
        explicit_named = deck is not None or opponent_deck is not None
        chosen_opp = opponent if opponent is not None else default_opponent_id()
        if chosen_opp == "search" and sims:
            opp_arg: str | Path = chosen_opp
        elif isinstance(chosen_opp, str) and is_python_policy_ref(chosen_opp):
            opp_arg = chosen_opp
        else:
            opp_arg = chosen_opp
        chosen_deck = _resolve_cli_deck(deck, "ur_lessons")
        chosen_opp_deck = _resolve_cli_deck(opponent_deck, "gw_allies")
        is_search_builtin = isinstance(opp_arg, str) and (
            opp_arg == "search" or opp_arg.startswith("search-")
        )
        if decklists is not None:
            chosen_disclosure: DecklistDisclosure = decklists
        elif (
            not explicit_named
            and (is_pool_key(chosen_deck) or is_pool_key(chosen_opp_deck))
            and not is_search_builtin
        ):
            chosen_disclosure = "closed"
        elif (
            is_pool_key(chosen_deck) or is_pool_key(chosen_opp_deck)
        ) and not is_search_builtin:
            chosen_disclosure = "closed"
        else:
            chosen_disclosure = "open"
        if isinstance(opp_arg, str):
            fit = policy_settings_fit(opp_arg, {"decklists": chosen_disclosure})
            if fit is not None and fit.warnings:
                typer.echo(f"NOTE {opp_arg}: {fit.summary()}", err=True)
        return running.new_game(
            deck=chosen_deck,
            opponent_deck=chosen_opp_deck,
            opponent=opp_arg,
            sims=sims,
            seed=seed,
            decklists=chosen_disclosure,
            search=search,
            order=order,
        ).url

    _run(
        _Shared(
            host,
            port,
            traces_dir,
            no_open,
            policies,
            False if no_discover else None,
        ).over(_earlier(ctx)),
        prepare,
    )


def _watch_cmd(
    ctx: typer.Context,
    player: list[str] | None = typer.Option(
        None,
        "--player",
        help="Seat player (repeatable twice): catalog id, ./bot.py:func, or search-N",
    ),
    bot1: str | None = typer.Option(
        None, "--bot1", help="Seat 1 bot (alternative to --player)"
    ),
    bot2: str | None = typer.Option(
        None, "--bot2", help="Seat 2 bot (alternative to --player)"
    ),
    deck: str | None = typer.Option(None, help="Seat 1 deck or deck pool"),
    opponent_deck: str | None = typer.Option(None, help="Seat 2 deck or deck pool"),
    seed: int | None = typer.Option(None, help="Deal seed (default random)"),
    decklists: DecklistDisclosure | None = typer.Option(
        None, help="Open or closed decklists"
    ),
    search_json: str | None = typer.Option(
        None,
        "--search",
        help='Two-seat JSON Search settings, e.g. [{"worlds":4},null].',
    ),
    host: str | None = _HOST,
    port: int | None = _PORT,
    traces_dir: Path | None = _TRACES_DIR,
    no_open: bool | None = _NO_OPEN,
    policies: list[Path] | None = _POLICIES,
    no_discover: bool | None = _NO_DISCOVER,
) -> None:
    seats: list[str] = []
    if bot1 is not None or bot2 is not None:
        if player:
            _report(
                AcademyLaunchError(
                    "setup.invalid", "Use either --bot1/--bot2 or --player, not both."
                )
            )
            raise typer.Exit(2)
        if bot1 is None or bot2 is None:
            _report(
                AcademyLaunchError(
                    "setup.invalid", "watch requires both --bot1 and --bot2."
                )
            )
            raise typer.Exit(2)
        seats = [bot1, bot2]
    elif player and len(player) == 2:
        seats = list(player)
    else:
        _report(
            AcademyLaunchError(
                "setup.invalid",
                "watch requires either --bot1 and --bot2 or exactly two --player values.",
            )
        )
        raise typer.Exit(2)

    def prepare(running: Academy) -> str:
        try:
            searches = (
                TypeAdapter(
                    tuple[SearchConfig | None, SearchConfig | None]
                ).validate_json(search_json)
                if search_json
                else (None, None)
            )
        except ValueError as exc:
            raise AcademyLaunchError(
                "setup.invalid", f"Invalid Search spec: {exc}"
            ) from exc
        decks_pair = None
        if deck is not None or opponent_deck is not None:
            decks_pair = (
                _resolve_cli_deck(deck, "ur_lessons"),
                _resolve_cli_deck(opponent_deck, "gw_allies"),
            )
        elif resolve_deck_pool("pool:held-out") is not None:
            decks_pair = ("pool:held-out", "pool:held-out")
        chosen_disclosure: DecklistDisclosure = (
            decklists
            if decklists is not None
            else (
                "closed"
                if decks_pair is not None
                and (is_pool_key(decks_pair[0]) or is_pool_key(decks_pair[1]))
                and not any(s == "search" or s.startswith("search-") for s in seats)
                else "open"
            )
        )
        return running.watch(
            players=(seats[0], seats[1]),
            search=searches,
            decks=decks_pair,
            seed=seed,
            decklists=chosen_disclosure,
        ).url

    _run(
        _Shared(
            host,
            port,
            traces_dir,
            no_open,
            policies,
            False if no_discover else None,
        ).over(_earlier(ctx)),
        prepare,
    )


def _replay_cmd(
    ctx: typer.Context,
    source: str = typer.Argument(
        help="A file or directory of recorded games, an evaluation trace, or a recording id from Games"
    ),
    game: str | None = typer.Option(
        None,
        help="A game within the file: zero-based position or a game_trace_sha256 prefix",
    ),
    host: str | None = _HOST,
    port: int | None = _PORT,
    traces_dir: Path | None = _TRACES_DIR,
    no_open: bool | None = _NO_OPEN,
    policies: list[Path] | None = _POLICIES,
) -> None:
    def prepare(running: Academy) -> str:
        return running.replay(_path_or_name(source, _TRACE_SUFFIXES), game=game).url

    _run(
        _Shared(host, port, traces_dir, no_open, policies).over(_earlier(ctx)),
        prepare,
    )


def _opponents_cmd(
    as_json: bool = typer.Option(False, "--json", help="Print catalog as JSON"),
) -> None:
    """List every available and refused opponent without starting a server."""
    catalog = opponent_catalog()
    if as_json:
        typer.echo(json.dumps(catalog, indent=2, sort_keys=True))
        raise typer.Exit(0)
    typer.echo(f"Default opponent: {catalog['default_opponent']}")
    rows: list[tuple[str, str, str, str]] = []
    for entry in catalog["opponents"]:
        ident = (entry.get("aliases") or [entry["sha256"][:12]])[0]
        status = "ready" if entry["available"] else f"unavailable: {entry['reason']}"
        rows.append((ident, entry["name"], entry["group"], status))
    for builtin in catalog["builtins"]:
        rows.append((builtin["id"], builtin["name"], builtin["group"], "ready"))
    id_w = max((len(r[0]) for r in rows), default=8)
    name_w = max((len(r[1]) for r in rows), default=8)
    grp_w = max((len(r[2]) for r in rows), default=8)
    typer.echo(f"{'id':<{id_w}}  {'name':<{name_w}}  {'group':<{grp_w}}  status")
    for ident, name, group, status in rows:
        typer.echo(f"{ident:<{id_w}}  {name:<{name_w}}  {group:<{grp_w}}  {status}")
    raise typer.Exit(0)


# Register directly on top-level `managym` AND under `managym academy` alias.
for _target in (app, academy_app):
    _target.command("play", help="Start in a chosen game")(_play_cmd)
    _target.command("watch", help="Choose two bots in Watch setup")(_watch_cmd)
    _target.command("replay", help="Open a game file or recording")(_replay_cmd)
    _target.command("opponents", help="List available and refused opponents")(
        _opponents_cmd
    )


def _run(shared: _Shared, prepare: _Prepare) -> None:
    """Serve, prepare the target, report READY, open it and wait for Ctrl-C."""
    try:
        pools = ensure_default_deck_pools()
        with serve(
            host=shared.host or "127.0.0.1",
            port=8000 if shared.port is None else shared.port,
            traces_dir=shared.traces_dir,
            policies={
                path.stem: PolicySpec.read(path) for path in shared.policies or []
            },
            deck_pools=pools or None,
            discover=True if shared.discover is None else shared.discover,
        ) as running:
            target = prepare(running)
            ready = {
                "traces_dir": str(running.traces_dir),
                "url": running.url,
                "open": target,
            }
            typer.echo("ACADEMY_PLAY_READY " + json.dumps(ready, sort_keys=True))
            typer.echo(f"  Academy: {target}\n  Ctrl-C stops the server")
            if not shared.no_open:
                running.open(target)
            running.wait()
    except (OSError, ImportError, ValueError) as exc:
        error = (
            exc
            if isinstance(exc, AcademyLaunchError)
            else AcademyLaunchError("policy.unavailable", str(exc))
        )
        _report(error)
        raise typer.Exit(2) from exc
    raise typer.Exit(0)


def _report(exc: AcademyLaunchError) -> None:
    record = {"code": exc.code, "message": exc.message}
    typer.echo("ACADEMY_PLAY_ERROR " + json.dumps(record, sort_keys=True), err=True)
    if not exc.games:
        return
    typer.echo("  Select a game with --game <index-or-digest>.", err=True)
    shown = exc.games[:SHOWN_GAMES]
    players = [" vs ".join(row.players) for row in shown]
    width = max(len("players"), *(len(names) for names in players))
    header = f"  game  seed        stream  {'players':<{width}}  decisions  ended        digest"
    typer.echo(header, err=True)
    for row, names in zip(shown, players, strict=True):
        typer.echo(
            f"  {row.index:4}  {row.seed:<10}  {row.stream:<6}  {names:<{width}}  "
            f"{row.decisions:<9}  {row.ended:<11}  {row.digest[:12]}",
            err=True,
        )
    if len(exc.games) > SHOWN_GAMES:
        typer.echo(f"  {SHOWN_GAMES} of {len(exc.games)} shown", err=True)


def main(argv: list[str] | None = None) -> int:
    """Run `managym` with `argv` (default: the process arguments)."""
    return _status(app, argv, "managym")


def academy(argv: list[str] | None = None) -> int:
    """Run `managym academy` with `argv` (default: the process arguments)."""
    return _status(academy_app, argv, "managym academy")


def _status(command: typer.Typer, argv: list[str] | None, prog_name: str) -> int:
    try:
        command(args=argv, prog_name=prog_name)
    except SystemExit as exit:
        if exit.code is None or isinstance(exit.code, int):
            return exit.code or 0
        return 1
    return 0
