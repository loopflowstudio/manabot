"""`managym academy` command line: prepare a game or replay, then serve Academy.

`app` is the `managym` Typer application; `main` and `academy` run it, or
its `academy` group, from an argument list and return the exit status. Commands parse and
report only; `managym.academy.launch.serve` owns the server, preparation and
every error code. Typer belongs to the `academy` extra, so `managym.cli`
imports this module lazily and reports a missing extra itself.

The shared options (`--host`, `--port`, `--traces-dir`, `--no-open`) are
accepted before or after `play`/`replay`: the group callback records the
values given before the verb and each command lets its own values win.
"""

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import typer

from managym.academy.launch import (
    _DECK_SUFFIXES,
    _TRACE_SUFFIXES,
    Academy,
    AcademyLaunchError,
    _path_or_name,
    serve,
)

DEFAULT_URL = "http://127.0.0.1:8000"
GUIDE_URL = "https://github.com/loopflowstudio/manabot/blob/main/docs/academy.md#running-academy"
SHOWN_GAMES = 20

_Prepare = Callable[[Academy], str]
"""Returns the address READY reports and the browser opens."""

app = typer.Typer(
    help="managym engine tools",
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

    def over(self, earlier: "_Shared") -> "_Shared":
        return _Shared(
            host=self.host if self.host is not None else earlier.host,
            port=self.port if self.port is not None else earlier.port,
            traces_dir=self.traces_dir
            if self.traces_dir is not None
            else earlier.traces_dir,
            no_open=self.no_open if self.no_open is not None else earlier.no_open,
        )


_HOST = typer.Option(None, "--host", help="Address to listen on (default 127.0.0.1)")
_PORT = typer.Option(None, "--port", help="Port to listen on (default 8000)")
_TRACES_DIR = typer.Option(
    None,
    "--traces-dir",
    help="Where games are saved (default ACADEMY_TRACES_DIR or ./academy/traces)",
)
_NO_OPEN = typer.Option(None, "--no-open", help="Do not open a browser window")


def _earlier(ctx: typer.Context) -> _Shared:
    return ctx.obj if isinstance(ctx.obj, _Shared) else _Shared()


@academy_app.callback()
def _academy(
    ctx: typer.Context,
    host: str | None = _HOST,
    port: int | None = _PORT,
    traces_dir: Path | None = _TRACES_DIR,
    no_open: bool | None = _NO_OPEN,
) -> None:
    shared = _Shared(host, port, traces_dir, no_open)
    if ctx.invoked_subcommand is None:
        _run(shared, lambda running: running.url)
    ctx.obj = shared


@academy_app.command(help="Start in a chosen game")
def play(
    ctx: typer.Context,
    deck: str = typer.Option(
        "ur_lessons", help="Your deck: a bundled deck name or a decklist file"
    ),
    opponent_deck: str = typer.Option(
        "gw_allies", help="The opponent's deck, named the same way"
    ),
    opponent: str = typer.Option(
        "search",
        help="search, random, passive, or a catalog checkpoint's 64-character digest",
    ),
    sims: int = typer.Option(64, help="Simulated playouts per decision for search"),
    seed: int | None = typer.Option(
        None,
        help="Deal seed; the same seed and decks give the same deal (default random)",
    ),
    decklists: Literal["open", "closed"] = typer.Option(
        "open", help="Whether each player knows the other's list"
    ),
    host: str | None = _HOST,
    port: int | None = _PORT,
    traces_dir: Path | None = _TRACES_DIR,
    no_open: bool | None = _NO_OPEN,
) -> None:
    from managym.academy import server

    def prepare(running: Academy) -> str:
        return running.new_game(
            deck=_path_or_name(deck, _DECK_SUFFIXES, server.NAMED_DECKS),
            opponent_deck=_path_or_name(
                opponent_deck, _DECK_SUFFIXES, server.NAMED_DECKS
            ),
            opponent=opponent,
            sims=sims,
            seed=seed,
            decklists=decklists,
        ).url

    _run(_Shared(host, port, traces_dir, no_open).over(_earlier(ctx)), prepare)


@academy_app.command(help="Open an evaluation trace or recording")
def replay(
    ctx: typer.Context,
    source: str = typer.Argument(
        help="A trace file (.json, .jsonl, gzipped) or a recording id from Games"
    ),
    game: str | None = typer.Option(
        None, help="A trace's game: zero-based position or a game_trace_sha256 prefix"
    ),
    host: str | None = _HOST,
    port: int | None = _PORT,
    traces_dir: Path | None = _TRACES_DIR,
    no_open: bool | None = _NO_OPEN,
) -> None:
    def prepare(running: Academy) -> str:
        return running.replay(_path_or_name(source, _TRACE_SUFFIXES), game=game).url

    _run(_Shared(host, port, traces_dir, no_open).over(_earlier(ctx)), prepare)


def _run(shared: _Shared, prepare: _Prepare) -> None:
    """Serve, prepare the target, report READY, open it and wait for Ctrl-C."""
    try:
        with serve(
            host=shared.host or "127.0.0.1",
            port=8000 if shared.port is None else shared.port,
            traces_dir=shared.traces_dir,
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
    except AcademyLaunchError as exc:
        _report(exc)
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
    header = f"  game  deal seed   leg  {'players':<{width}}  trace"
    typer.echo(header, err=True)
    for row, names in zip(shown, players, strict=True):
        typer.echo(
            f"  {row.index:4}  {row.deal_seed:<10}  {row.leg:<3}  {names:<{width}}  "
            f"{row.game_trace_sha256[:12]}",
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
