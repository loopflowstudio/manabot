"""Compare engine benchmark output with the committed performance baseline.

The two release benchmarks under ``managym/examples`` print JSON lines:
``bench_search`` one line per seeded game (worst, median and total Search-64
reply time plus a digest of its search results) and a summary line;
``bench_steps`` one line for its headless random-play workload (steps per
second and a digest of every decision). ``managym/perf/baseline.json`` holds
the values recorded on the CI runner profile; ``compare`` fails when a timing
is more than the tolerance slower or a digest differs, and ``record`` rewrites
the baseline from fresh output with the commit, run and reason that move it.
Timings are compared only against the same runner profile; a laptop figure is
not a baseline. Stdlib only, so CI runs it without syncing the project.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import TypedDict

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "managym/perf/baseline.json"


class SearchSeed(TypedDict):
    worst_s: float
    total_s: float
    digest: int


class SearchBaseline(TypedDict):
    worst_s: float
    total_s: float
    seeds: dict[str, SearchSeed]


class StepsBaseline(TypedDict):
    steps_per_s: float
    steps: int
    games: int
    digest: int


class Recorded(TypedDict):
    commit: str
    date: str
    run: str
    reason: str


class Baseline(TypedDict):
    runner: str
    tolerance: float
    recorded: Recorded
    bench_search: SearchBaseline
    bench_steps: StepsBaseline


@dataclass(frozen=True)
class Check:
    """One budget line: ``fails`` is empty when the measurement is within it."""

    name: str
    baseline: str
    measured: str
    fails: str


def _lines(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        if line.startswith("{"):
            rows.append(json.loads(line))
    return rows


def read_search(path: Path) -> SearchBaseline:
    """Fold ``bench_search`` output into the baseline shape (summed totals)."""
    seeds: dict[str, SearchSeed] = {}
    for row in _lines(path):
        if "seed" not in row:
            continue
        seeds[str(row["seed"])] = SearchSeed(
            worst_s=float(str(row["worst_s"])),
            total_s=float(str(row["total_s"])),
            digest=int(str(row["digest"])),
        )
    if not seeds:
        raise SystemExit(f"{path}: no per-seed bench_search lines")
    return SearchBaseline(
        worst_s=max(seed["worst_s"] for seed in seeds.values()),
        total_s=round(sum(seed["total_s"] for seed in seeds.values()), 3),
        seeds=seeds,
    )


def read_steps(path: Path) -> StepsBaseline:
    rows = [row for row in _lines(path) if "steps_per_s" in row]
    if len(rows) != 1:
        raise SystemExit(f"{path}: expected one bench_steps summary line")
    row = rows[0]
    return StepsBaseline(
        steps_per_s=float(str(row["steps_per_s"])),
        steps=int(str(row["steps"])),
        games=int(str(row["games"])),
        digest=int(str(row["digest"])),
    )


def load_baseline(path: Path) -> Baseline:
    data = json.loads(path.read_text())
    return Baseline(
        runner=str(data["runner"]),
        tolerance=float(data["tolerance"]),
        recorded=Recorded(**data["recorded"]),
        bench_search=SearchBaseline(
            worst_s=float(data["bench_search"]["worst_s"]),
            total_s=float(data["bench_search"]["total_s"]),
            seeds={
                seed: SearchSeed(**values)
                for seed, values in data["bench_search"]["seeds"].items()
            },
        ),
        bench_steps=StepsBaseline(**data["bench_steps"]),
    )


def _slower(name: str, baseline: float, measured: float, tolerance: float) -> Check:
    limit = baseline * (1 + tolerance)
    return Check(
        name,
        f"{baseline:.3f} s",
        f"{measured:.3f} s",
        "" if measured <= limit else f"over {limit:.3f} s (+{tolerance:.0%})",
    )


def _faster(name: str, baseline: float, measured: float, tolerance: float) -> Check:
    limit = baseline / (1 + tolerance)
    return Check(
        name,
        f"{baseline:.1f}/s",
        f"{measured:.1f}/s",
        "" if measured >= limit else f"under {limit:.1f}/s (-{tolerance:.0%})",
    )


def _same(name: str, baseline: int, measured: int) -> Check:
    return Check(
        name, str(baseline), str(measured), "" if baseline == measured else "differs"
    )


def checks(baseline: Baseline, search: SearchBaseline, steps: StepsBaseline) -> list[Check]:
    """Every budget line, timings within tolerance and digests identical."""
    tolerance = baseline["tolerance"]
    base_search = baseline["bench_search"]
    base_steps = baseline["bench_steps"]
    result = [
        _slower("search worst reply", base_search["worst_s"], search["worst_s"], tolerance),
        _slower("search total", base_search["total_s"], search["total_s"], tolerance),
    ]
    for seed in sorted(set(base_search["seeds"]) | set(search["seeds"]), key=int):
        expected = base_search["seeds"].get(seed)
        actual = search["seeds"].get(seed)
        if expected is None or actual is None:
            result.append(
                Check(
                    f"search seed {seed} digest",
                    "absent" if expected is None else str(expected["digest"]),
                    "absent" if actual is None else str(actual["digest"]),
                    "seed set differs",
                )
            )
        else:
            result.append(
                _same(f"search seed {seed} digest", expected["digest"], actual["digest"])
            )
    result.extend(
        [
            _faster("steps throughput", base_steps["steps_per_s"], steps["steps_per_s"], tolerance),
            _same("steps games", base_steps["games"], steps["games"]),
            _same("steps count", base_steps["steps"], steps["steps"]),
            _same("steps digest", base_steps["digest"], steps["digest"]),
        ]
    )
    return result


def render(baseline: Baseline, rows: list[Check]) -> str:
    recorded = baseline["recorded"]
    lines = [
        f"Engine budget against {recorded['commit'][:9]} ({recorded['date']}, "
        f"{baseline['runner']}, tolerance {baseline['tolerance']:.0%})",
        "",
        "| Check | Baseline | Measured | Result |",
        "|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.name} | {row.baseline} | {row.measured} | "
            f"{row.fails or 'ok'} |"
        )
    failed = [row for row in rows if row.fails]
    lines.append("")
    if failed:
        lines.append(
            f"FAIL: {len(failed)} check(s) outside the budget. A deliberate change "
            "re-records managym/perf/baseline.json with `scripts/engine_budget.py "
            "record` and states why (docs/development.md, Engine performance budget)."
        )
    else:
        lines.append("PASS: within the recorded budget.")
    return "\n".join(lines)


def compare(args: argparse.Namespace) -> int:
    baseline = load_baseline(args.baseline)
    rows = checks(baseline, read_search(args.search), read_steps(args.steps))
    print(render(baseline, rows))
    return 1 if any(row.fails for row in rows) else 0


def record(args: argparse.Namespace) -> int:
    previous = load_baseline(args.baseline) if args.baseline.exists() else None
    baseline = Baseline(
        runner=args.runner or (previous["runner"] if previous else ""),
        tolerance=args.tolerance
        if args.tolerance is not None
        else (previous["tolerance"] if previous else 0.10),
        recorded=Recorded(commit=args.commit, date=args.date, run=args.run, reason=args.reason),
        bench_search=read_search(args.search),
        bench_steps=read_steps(args.steps),
    )
    if not baseline["runner"]:
        raise SystemExit("--runner is required for a first baseline")
    args.baseline.write_text(json.dumps(baseline, indent=2) + "\n")
    print(f"Recorded {args.baseline} at {args.commit[:9]}: {args.reason}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("compare", compare), ("record", record)):
        command = commands.add_parser(name)
        command.add_argument("--baseline", type=Path, default=BASELINE)
        command.add_argument("--search", type=Path, required=True, help="bench_search output")
        command.add_argument("--steps", type=Path, required=True, help="bench_steps output")
        command.set_defaults(handler=handler)
        if name == "record":
            command.add_argument("--commit", required=True, help="commit the output measured")
            command.add_argument("--date", required=True, help="ISO date of the measurement")
            command.add_argument("--run", required=True, help="CI run URL of the measurement")
            command.add_argument("--reason", required=True, help="why the baseline moves")
            command.add_argument("--runner", help="runner profile; kept from the previous baseline when omitted")
            command.add_argument("--tolerance", type=float, help="allowed slowdown fraction; kept when omitted")
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
