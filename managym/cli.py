"""Engine command line: prepare and launch the packaged Academy browser UI."""

import argparse
from collections.abc import Collection
import json
from pathlib import Path
import sys

DEFAULT_URL = "http://127.0.0.1:8000"
GUIDE_URL = "https://github.com/loopflowstudio/manabot/blob/main/docs/academy.md#running-academy"


def _path_or_name(
    value: str, suffixes: set[str], names: Collection[str] = ()
) -> str | Path:
    """A known name is never a file; otherwise an existing or path-shaped value is one."""
    if value in names:
        return value
    path = Path(value).expanduser()
    return path if path.exists() or "/" in value or path.suffix in suffixes else value


def academy(argv: list[str] | None = None) -> int:
    from managym.academy.launch import AcademyLaunchError, serve

    common = argparse.ArgumentParser(add_help=False)
    # Suppressed defaults let shared options occur before or after the verb.
    common.add_argument("--host", default=argparse.SUPPRESS)
    common.add_argument("--port", type=int, default=argparse.SUPPRESS)
    common.add_argument("--traces-dir", type=Path, default=argparse.SUPPRESS)
    common.add_argument("--no-open", action="store_true", default=argparse.SUPPRESS)
    parser = argparse.ArgumentParser(
        prog="managym academy",
        parents=[common],
        description=f"Open Academy at {DEFAULT_URL}. Launch guide: {GUIDE_URL}",
    )
    commands = parser.add_subparsers(dest="command")
    play = commands.add_parser("play", parents=[common], help="Start in a chosen game")
    play.add_argument("--deck", default="ur_lessons")
    play.add_argument("--opponent-deck", default="gw_allies")
    play.add_argument("--opponent", default="search")
    play.add_argument("--sims", type=int, default=64)
    play.add_argument("--seed", type=int)
    play.add_argument("--decklists", choices=["open", "closed"], default="open")
    replay = commands.add_parser(
        "replay", parents=[common], help="Open an evaluation trace or recording"
    )
    replay.add_argument("source")
    replay.add_argument("--game")
    args = parser.parse_args(argv)
    try:
        with serve(
            host=getattr(args, "host", "127.0.0.1"),
            port=getattr(args, "port", 8000),
            traces_dir=getattr(args, "traces_dir", None),
        ) as running:
            target = running.url
            if args.command == "play":
                from managym.academy import server

                decklist = {".txt", ".dek"}
                target = running.new_game(
                    deck=_path_or_name(args.deck, decklist, server.NAMED_DECKS),
                    opponent_deck=_path_or_name(
                        args.opponent_deck, decklist, server.NAMED_DECKS
                    ),
                    opponent=args.opponent,
                    sims=args.sims,
                    seed=args.seed,
                    decklists=args.decklists,
                ).url
            elif args.command == "replay":
                selection = (
                    int(args.game)
                    if args.game is not None and args.game.isdecimal()
                    else args.game
                )
                target = running.replay(
                    _path_or_name(args.source, {".json", ".jsonl", ".gz"}),
                    game=selection,
                ).url
            print(
                "ACADEMY_PLAY_READY "
                + json.dumps(
                    {
                        "traces_dir": str(running.traces_dir),
                        "url": running.url,
                        "open": target,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            print(f"  Academy: {target}\n  Ctrl-C stops the server", flush=True)
            if not getattr(args, "no_open", False):
                running.open(target)
            running.wait()
    except AcademyLaunchError as exc:
        print(
            "ACADEMY_PLAY_ERROR "
            + json.dumps({"code": exc.code, "message": exc.message}, sort_keys=True),
            file=sys.stderr,
            flush=True,
        )
        if exc.games:
            print("  game  deal seed  leg  players  trace", file=sys.stderr)
            for row in exc.games[:20]:
                print(
                    f"  {row.index:4}  {row.deal_seed}  {row.leg}  {' vs '.join(row.players)}  {row.game_trace_sha256[:12]}",
                    file=sys.stderr,
                )
            if len(exc.games) > 20:
                print(f"  20 of {len(exc.games)} shown", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments[:1] == ["academy"]:
        return academy(arguments[1:])
    parser = argparse.ArgumentParser(description="managym engine tools")
    parser.add_argument("command", choices=["academy"])
    parser.parse_args(arguments)  # exits with help or a usage error
    return 2
