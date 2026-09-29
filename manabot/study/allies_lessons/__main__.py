"""
__main__.py
Record Allies versus Lessons games and write the behavior report

    uv run python -m manabot.study.allies_lessons --out .runs/study

Pass --games to rebuild the report from an existing record without playing.
"""

from __future__ import annotations

# Standard library
import argparse
from pathlib import Path

# Local imports
from ..record import read_games, record_games, write_games
from .matchup import PLAYERS, schedule
from .report import build_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    parser.add_argument("--out", type=Path, default=Path(".runs/study/allies-lessons"))
    parser.add_argument("--games", type=Path, help="existing games.jsonl to report on")
    parser.add_argument("--subject", default="search", choices=sorted(PLAYERS))
    parser.add_argument("--baseline", default="random", choices=sorted(PLAYERS))
    parser.add_argument("--mirror-deals", type=int, default=60)
    parser.add_argument("--versus-deals", type=int, default=15)
    parser.add_argument("--baseline-deals", type=int, default=200)
    parser.add_argument("--seed", type=int, default=91_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fragment", action="store_true", help="omit the HTML wrapper")
    args = parser.parse_args()

    path = args.games
    if path is None:
        specs = (
            schedule(
                args.subject, args.subject, deals=args.mirror_deals, seed=args.seed
            )
            + schedule(
                args.subject,
                args.baseline,
                deals=args.versus_deals,
                seed=args.seed + 1_000,
            )
            + schedule(
                args.baseline,
                args.baseline,
                deals=args.baseline_deals,
                seed=args.seed + 2_000,
            )
        )
        path = args.out / "games.jsonl"

        def progress(done: int, total: int) -> None:
            if done % 20 == 0 or done == total:
                print(f"{done}/{total} games", flush=True)

        write_games(path, record_games(specs, workers=args.workers, progress=progress))

    report = args.out / "report.html"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        build_report(
            read_games(path),
            subject=args.subject,
            baseline=args.baseline,
            fragment=args.fragment,
        )
    )
    print(f"games:  {path}\nreport: {report}")


if __name__ == "__main__":
    main()
