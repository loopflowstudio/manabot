"""Launch Academy directly into random 17Lands TLA trophy matchups.

Loads the frozen ``pools-v2`` 17Lands trophy corpus (from the working tree or
the machine-wide ``manabot.data`` cache), registers held-out and training
``DeckPool`` populations with Academy, and opens a fresh trophy-vs-trophy game
(or Watch setup) with closed decklists by default. Click **Deal new decks**
inside Academy to roll a fresh pair of trophies without restarting.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

from pydantic import TypeAdapter

from manabot.data import cache_directory, fetch_data, load_manifest
from manabot.pools import Pool, PoolCorpus, PoolSplit
import managym
from managym.academy.deck_pools import DeckPool, NamedDeck
from managym.academy.launch import AcademyLaunchError, serve
from managym.disclosure import DecklistDisclosure
from managym.policy import PolicySpec
from managym.search import SearchConfig

ROOT = Path(__file__).resolve().parents[1]
POOLS_PATH = "experiments/data/mtg409/pools-v2/corpus.json"
_BASIC_COLORS = (
    ("Plains", "W"),
    ("Island", "U"),
    ("Swamp", "B"),
    ("Mountain", "R"),
    ("Forest", "G"),
)


def trophy_colors(deck: managym.Deck) -> str:
    """Short WUBRG code from basic lands in the trophy's main deck."""
    letters = "".join(
        code for land, code in _BASIC_COLORS if deck.main.get(land, 0) > 0
    )
    return letters or "Colorless"


def trophy_deck(pool: Pool) -> NamedDeck:
    """Convert one 17Lands ``Pool`` into a labeled ``NamedDeck``."""
    sideboard = {
        name: count - pool.human_build.get(name, 0)
        for name, count in pool.cards.items()
        if count > pool.human_build.get(name, 0)
    }
    deck = managym.Deck(pool.human_build, sideboard)
    label = f"TLA Trophy · {trophy_colors(deck)} · {pool.id[:8]}"
    return NamedDeck(name=label, deck=deck)


def corpus_file(path: Path | None = None) -> Path:
    """Locate ``pools-v2/corpus.json`` locally or in the machine-wide cache."""
    if path is not None:
        return path.expanduser().resolve()
    local = ROOT / POOLS_PATH
    if local.is_file():
        return local
    entry = load_manifest().files[POOLS_PATH]
    cached = cache_directory() / entry.sha256
    if cached.is_file():
        return cached
    fetch_data([POOLS_PATH])
    return local


def load_trophy_pools(path: Path | None = None) -> tuple[DeckPool, DeckPool]:
    """Build held-out and training 17Lands TLA trophy ``DeckPool`` instances."""
    corpus = PoolCorpus.read(corpus_file(path))
    by_split: dict[PoolSplit, list[NamedDeck]] = {"held_out": [], "train": []}
    for pool in corpus.pools:
        if managym.unsupported_cards([*pool.cards, *pool.human_build]):
            continue
        by_split[pool.split].append(trophy_deck(pool))
    return (
        DeckPool(
            id="tla_trophies_held_out",
            label="Random 17Lands TLA trophy (held-out)",
            decks=tuple(by_split["held_out"]),
            default_decklists="closed",
        ),
        DeckPool(
            id="tla_trophies_train",
            label="Random 17Lands TLA trophy (train)",
            decks=tuple(by_split["train"]),
            default_decklists="closed",
        ),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play a random 17Lands TLA trophy deck against another in Academy."
    )
    parser.add_argument(
        "--split",
        choices=("held_out", "train"),
        default="held_out",
        help="Trophy split to draw from (default: held_out).",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Optional path to a pools-v2 corpus.json file.",
    )
    parser.add_argument(
        "--opponent",
        default="heuristic",
        help="Opponent: heuristic, greedy, random, passive, search, registered alias or policy:<id>.",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        action="append",
        default=[],
        help="Saved PolicySpec JSON to register (repeatable; file stem is its alias).",
    )
    parser.add_argument(
        "--watch",
        nargs=2,
        metavar=("SEAT1", "SEAT2"),
        default=None,
        help="Open Watch setup for two bots on random TLA trophies instead of Play.",
    )
    parser.add_argument(
        "--decklists",
        choices=("closed", "open"),
        default="closed",
        help="Decklist disclosure (default: closed).",
    )
    parser.add_argument(
        "--search",
        default=None,
        help="Search JSON config for Play opponent (or two-element JSON list for --watch).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional uint64 deal seed (fixes both the sampled trophies and the deal).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--traces-dir", type=Path, default=None)
    parser.add_argument("--no-open", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    held_out, train = load_trophy_pools(args.corpus)
    selected = held_out if args.split == "held_out" else train
    decklists: DecklistDisclosure = args.decklists
    policies = {path.stem: PolicySpec.read(path) for path in args.policy}
    try:
        with serve(
            host=args.host,
            port=args.port,
            traces_dir=args.traces_dir,
            policies=policies,
            deck_pools=(held_out, train),
        ) as academy:
            if args.watch is not None:
                searches = (
                    TypeAdapter(
                        tuple[SearchConfig | None, SearchConfig | None]
                    ).validate_json(args.search)
                    if args.search
                    else (None, None)
                )
                target = academy.watch(
                    players=(args.watch[0], args.watch[1]),
                    search=searches,
                    decks=(selected.key, selected.key),
                    seed=args.seed,
                    decklists=decklists,
                ).url
            else:
                search = (
                    TypeAdapter(SearchConfig).validate_json(args.search)
                    if args.search
                    else None
                )
                target = academy.new_game(
                    deck=selected.key,
                    opponent_deck=selected.key,
                    opponent=args.opponent,
                    seed=args.seed,
                    decklists=decklists,
                    search=search,
                ).url
            ready = {
                "open": target,
                "pool": selected.id,
                "pool_decks": len(selected.decks),
                "traces_dir": str(academy.traces_dir),
                "url": academy.url,
            }
            print("ACADEMY_PLAY_READY " + json.dumps(ready, sort_keys=True), flush=True)
            print(
                f"  Academy ({selected.label}, {len(selected.decks)} decks): {target}\n"
                "  Ctrl-C stops the server",
                flush=True,
            )
            if not args.no_open:
                academy.open(target)
            academy.wait()
    except (AcademyLaunchError, OSError, ValueError) as exc:
        code = exc.code if isinstance(exc, AcademyLaunchError) else "setup.invalid"
        print(
            "ACADEMY_PLAY_ERROR "
            + json.dumps({"code": code, "message": str(exc)}, sort_keys=True),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
