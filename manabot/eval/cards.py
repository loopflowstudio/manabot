"""Card evaluation CLI: evaluate, data, audit, export-log and bounded verify.

Use the evaluation extra. Every evaluation names its experiment and run and has
admitted registrations; no command selects a checkpoint implicitly. `data` and
`audit` read the experiment's kept Etude database, caught up with `--source`
(by default, where runs are written on this computer)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time

from etude.run.storage import default_root

from manabot.eval.arena_log import CardMapping, export_arena_log
from manabot.eval.card_evaluation import (
    CardEvaluationSpec,
    evaluate_cards,
    verification_spec,
)
from manabot.eval.card_page import write_page
from manabot.eval.card_query import (
    CardData,
    CardSelection,
    audit,
    card_data,
    card_db,
    load_game,
)


def verify(games: int, out: Path) -> dict[str, int | float | str]:
    """Play bounded native games, then audit every retained game by replay.

    Card data is read for both participants, and again after the audit, so a
    corrupted row for either player fails here.
    """
    started = time.monotonic()
    spec = verification_spec(games)
    evaluate_cards(spec, out)
    selections = [
        CardSelection(
            experiment=spec.experiment, subjects=(p.player_id,), runs=(spec.run,)
        )
        for p in spec.players
    ]
    with card_db(spec.experiment, out) as db:
        before = [card_data(db, selection) for selection in selections]
        if (
            sum(data.games + sum(data.excluded.values()) for data in before)
            != 2 * games
        ):
            raise ValueError("verification did not retain every game for both players")
        result = audit(db, out, replay=True)
        if result.problems or result.games != games:
            raise ValueError(f"audit failed: {result.model_dump()}")
        if [card_data(db, selection) for selection in selections] != before:
            raise ValueError("card data changed when read again")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "games": games,
        "elapsed_seconds": time.monotonic() - started,
        "audited_games": result.games,
        "peak_parent_memory_bytes": int(
            usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        ),
        "peak_child_memory_bytes": int(
            children.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        ),
        "run": spec.run,
    }


def print_data(data: CardData) -> None:
    subjects = ", ".join(data.subjects.values()) or "every player"
    print(
        f"{data.selection.experiment}: {subjects}; {data.games} games, "
        f"{data.deals} deals, {len(data.runs)} runs"
    )
    print(f"opponents: {data.opponents}; excluded: {data.excluded}")
    print("card\t# GIH\tGIH WR\t95% deal interval\t# GNS\tGNS WR\tIWD")
    for card in data.cards:
        rate = "—" if card.gih.rate is None else f"{card.gih.rate:.1%}"
        gns = "—" if card.gns.rate is None else f"{card.gns.rate:.1%}"
        interval = (
            "—"
            if card.gih_interval is None
            else f"{card.gih_interval[0]:.1%}–{card.gih_interval[1]:.1%}"
        )
        iwd = "—" if card.iwd is None else f"{card.iwd:+.1f}pp"
        print(
            f"{card.name}\t{card.gih.count}\t{rate}\t{interval}\t{card.gns.count}\t{gns}\t{iwd}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--spec", type=Path, required=True)
    evaluation.add_argument("--out", type=Path, required=True)
    data = commands.add_parser("data", help="17Lands-style card columns")
    data.add_argument("--experiment", required=True)
    data.add_argument("--source", help="directory or bucket of eval runs")
    data.add_argument("--subject", action="append", default=[])
    data.add_argument("--opponent", action="append", default=[])
    data.add_argument("--run", action="append", default=[])
    data.add_argument("--commit", action="append", default=[])
    data.add_argument("--deck")
    data.add_argument("--opponent-deck")
    data.add_argument("--position", choices=["play", "draw"])
    data.add_argument("--rebuild", action="store_true", help="read every file again")
    shown = data.add_mutually_exclusive_group()
    shown.add_argument("--json", action="store_true")
    shown.add_argument("--html", type=Path, help="write a card data page")
    checking = commands.add_parser("audit", help="re-verify retained evidence")
    checking.add_argument("--experiment", required=True)
    checking.add_argument("--source")
    checking.add_argument("--replay", action="store_true")
    export = commands.add_parser("export-log")
    export.add_argument("--experiment", required=True)
    export.add_argument("--source")
    export.add_argument("--run", required=True)
    export.add_argument("--game", required=True)
    export.add_argument("--viewer", type=int, choices=[0, 1], required=True)
    export.add_argument("--mapping", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    verification = commands.add_parser("verify")
    verification.add_argument("--games", type=int, default=1024)
    verification.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "evaluate":
        result = evaluate_cards(
            CardEvaluationSpec.model_validate_json(args.spec.read_bytes()), args.out
        )
        print(result.model_dump_json(indent=2))
    elif args.command == "data":
        selection = CardSelection(
            experiment=args.experiment,
            subjects=tuple(args.subject),
            opponents=tuple(args.opponent),
            runs=tuple(args.run),
            commits=tuple(args.commit),
            deck=args.deck,
            opponent_deck=args.opponent_deck,
            on_play=None if args.position is None else args.position == "play",
        )
        with card_db(args.experiment, args.source, rebuild=args.rebuild) as db:
            if args.html:
                print(write_page(db, selection, args.html))
            elif args.json:
                print(card_data(db, selection).model_dump_json(indent=2))
            else:
                print_data(card_data(db, selection))
    elif args.command == "audit":
        source = args.source
        with card_db(args.experiment, source) as db:
            result = audit(db, source or default_root(), replay=args.replay)
        print(result.model_dump_json(indent=2))
        if result.problems:
            sys.exit(1)
    elif args.command == "export-log":
        source = args.source
        with card_db(args.experiment, source) as db:
            game = load_game(db, source or default_root(), args.run, args.game)
        log = export_arena_log(
            game,
            args.viewer,
            CardMapping.model_validate_json(args.mapping.read_bytes()),
        )
        args.out.write_text(log)
    else:
        print(json.dumps(verify(args.games, args.out), indent=2))


if __name__ == "__main__":
    main()
