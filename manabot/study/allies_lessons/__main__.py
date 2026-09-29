"""
__main__.py
Record Allies versus Lessons games and write the behavior report

    uv run python -m manabot.study.allies_lessons --out .runs/study/first
    uv run python -m manabot.study.allies_lessons --out .runs/study/second \\
        --config .runs/study/first/study.json --set mirror_deals=100

Each run writes study.json, games.jsonl.gz and report.html to --out. Pass
--games to rebuild the report from an existing record without playing.
"""

from __future__ import annotations

# Standard library
import argparse
import json
from pathlib import Path

# Local imports
from ..record import read_games, record_games, write_games
from .config import StudyConfig
from .report import build_report


def parse_set(pairs: list[str]) -> dict[str, object]:
    """['mirror_deals=100', 'ur_lessons=+2 Tiger-Seal'] as typed fields."""
    fields: dict[str, object] = {}
    for pair in pairs:
        key, separator, value = pair.partition("=")
        if not separator:
            raise SystemExit(f"--set needs key=value, got {pair!r}")
        try:
            fields[key] = json.loads(value)
        except json.JSONDecodeError:
            fields[key] = value
    return fields


def run(
    config: StudyConfig,
    out: Path,
    *,
    games: Path | None = None,
    fragment: bool = False,
) -> Path:
    """Record (unless `games` is given) and write the report. Returns its path."""
    out.mkdir(parents=True, exist_ok=True)
    config.save(out / "study.json")
    if games is None:
        games = out / "games.jsonl.gz"

        def progress(done: int, total: int) -> None:
            if done % 20 == 0 or done == total:
                print(f"{done}/{total} games", flush=True)

        write_games(
            games,
            record_games(config.games(), workers=config.workers, progress=progress),
        )
    report = out / "report.html"
    report.write_text(
        build_report(
            read_games(games),
            subject=config.subject,
            baseline=config.baseline,
            fragment=fragment,
        )
    )
    print(f"config: {out / 'study.json'}\ngames:  {games}\nreport: {report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    parser.add_argument("--out", type=Path, default=Path(".runs/study/allies-lessons"))
    parser.add_argument("--config", type=Path, help="study.json to start from")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one field; repeatable. Fields: "
        + ", ".join(StudyConfig.model_fields),
    )
    parser.add_argument("--games", type=Path, help="existing record to report on")
    parser.add_argument("--fragment", action="store_true", help="omit the HTML wrapper")
    args = parser.parse_args()

    config = StudyConfig.load(args.config) if args.config else StudyConfig()
    run(
        config.changed(**parse_set(args.set)),
        args.out,
        games=args.games,
        fragment=args.fragment,
    )


if __name__ == "__main__":
    main()
