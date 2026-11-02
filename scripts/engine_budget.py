"""Compare engine benchmark output with the committed performance baseline.

The two release benchmarks under ``managym/examples`` print JSON lines:
``bench_search`` one line per seeded game (worst, median and total Search-64
reply time plus a digest of its search results) and a summary line;
``bench_steps`` one line for its headless random-play workload (steps per
second and a digest of every decision). ``managym/perf/baseline.json`` names a
reference commit, the digests both workloads must reproduce, and the figures
the reference measured when it was recorded.

GitHub-hosted runners land on different CPU generations (an identical engine
measured 6.8k and 10.3k steps/s on two of them), so wall time is never
compared across runs. CI rebuilds the reference commit with the current
benchmarks on the same runner and ``compare`` holds the candidate to the
reference's fresh timings within the tolerance, and both to the committed
digests. ``record`` rewrites the file from a reference measurement with the
commit, run and reason that move it. Stdlib only, so CI runs it without
syncing the project.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, TypedDict, cast

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "managym/perf/baseline.json"


class SearchSeed(TypedDict):
    worst_s: float
    total_s: float
    digest: int


class SearchResult(TypedDict):
    worst_s: float
    total_s: float
    seeds: dict[str, SearchSeed]


class StepsResult(TypedDict):
    steps_per_s: float
    steps: int
    games: int
    digest: int


class Reference(TypedDict):
    commit: str
    date: str
    run: str
    runner: str
    reason: str


class Baseline(TypedDict):
    tolerance: float
    reference: Reference
    bench_search: SearchResult
    bench_steps: StepsResult


@dataclass(frozen=True)
class Check:
    """One budget line: ``failure`` is empty when the measurement is within it."""

    name: str
    expected: str
    measured: str
    failure: str


def _lines(path: Path) -> list[dict[str, Any]]:
    """The JSON objects in benchmark output; cargo's own lines are skipped."""
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.startswith("{")
    ]


def read_search(path: Path) -> SearchResult:
    """Fold ``bench_search`` output into the baseline shape (summed totals)."""
    seeds = {
        str(row["seed"]): SearchSeed(
            worst_s=float(row["worst_s"]),
            total_s=float(row["total_s"]),
            digest=int(row["digest"]),
        )
        for row in _lines(path)
        if "seed" in row
    }
    if not seeds:
        raise SystemExit(f"{path}: no per-seed bench_search lines")
    return SearchResult(
        worst_s=max(seed["worst_s"] for seed in seeds.values()),
        total_s=round(sum(seed["total_s"] for seed in seeds.values()), 3),
        seeds=seeds,
    )


def read_steps(path: Path) -> StepsResult:
    rows = [row for row in _lines(path) if "steps_per_s" in row]
    if len(rows) != 1:
        raise SystemExit(f"{path}: expected one bench_steps summary line")
    row = rows[0]
    return StepsResult(
        steps_per_s=float(row["steps_per_s"]),
        steps=int(row["steps"]),
        games=int(row["games"]),
        digest=int(row["digest"]),
    )


def load_baseline(path: Path) -> Baseline:
    # The file is written by ``record`` below, so its shape is trusted.
    return cast(Baseline, json.loads(path.read_text()))


def _slower(name: str, reference: float, measured: float, tolerance: float) -> Check:
    limit = reference * (1 + tolerance)
    return Check(
        name,
        f"{reference:.3f} s",
        f"{measured:.3f} s",
        "" if measured <= limit else f"over {limit:.3f} s (+{tolerance:.0%})",
    )


def _faster(name: str, reference: float, measured: float, tolerance: float) -> Check:
    limit = reference / (1 + tolerance)
    return Check(
        name,
        f"{reference:.1f}/s",
        f"{measured:.1f}/s",
        "" if measured >= limit else f"under {limit:.1f}/s (-{tolerance:.0%})",
    )


def _same(name: str, expected: int | None, measured: int | None) -> Check:
    """A digest or count that must match; ``None`` is a seed one side lacks."""
    return Check(
        name,
        "absent" if expected is None else str(expected),
        "absent" if measured is None else str(measured),
        "" if expected == measured else "differs",
    )


def digest_checks(
    label: str, baseline: Baseline, search: SearchResult, steps: StepsResult
) -> list[Check]:
    """Both workloads reproduce the committed digests, seed set included."""
    expected_seeds = baseline["bench_search"]["seeds"]
    expected_steps = baseline["bench_steps"]
    checks = [
        _same(
            f"{label} search seed {seed} digest",
            expected_seeds[seed]["digest"] if seed in expected_seeds else None,
            search["seeds"][seed]["digest"] if seed in search["seeds"] else None,
        )
        for seed in sorted(set(expected_seeds) | set(search["seeds"]), key=int)
    ]
    checks += [
        _same(f"{label} steps games", expected_steps["games"], steps["games"]),
        _same(f"{label} steps count", expected_steps["steps"], steps["steps"]),
        _same(f"{label} steps digest", expected_steps["digest"], steps["digest"]),
    ]
    return checks


def timing_checks(
    tolerance: float,
    reference_search: SearchResult,
    reference_steps: StepsResult,
    search: SearchResult,
    steps: StepsResult,
) -> list[Check]:
    """Candidate timings against the reference measured on the same machine."""
    return [
        _slower(
            "search worst reply",
            reference_search["worst_s"],
            search["worst_s"],
            tolerance,
        ),
        _slower(
            "search total", reference_search["total_s"], search["total_s"], tolerance
        ),
        _faster(
            "steps throughput",
            reference_steps["steps_per_s"],
            steps["steps_per_s"],
            tolerance,
        ),
    ]


def render(baseline: Baseline, rows: list[Check]) -> str:
    reference = baseline["reference"]
    lines = [
        f"Engine budget: candidate against reference {reference['commit'][:9]} "
        f"rebuilt on this runner (recorded {reference['date']} on "
        f"{reference['runner']}; tolerance {baseline['tolerance']:.0%})",
        "",
        "| Check | Reference | Candidate | Result |",
        "|---|---:|---:|---|",
    ]
    lines += [
        f"| {row.name} | {row.expected} | {row.measured} | {row.failure or 'ok'} |"
        for row in rows
    ]
    failed = sum(1 for row in rows if row.failure)
    lines.append("")
    if failed:
        lines.append(
            f"FAIL: {failed} check(s) outside the budget. A deliberate change "
            "re-records managym/perf/baseline.json with `scripts/engine_budget.py "
            "record` and states why (docs/development.md, Engine performance budget)."
        )
    else:
        lines.append("PASS: within the recorded budget.")
    return "\n".join(lines)


def compare(args: argparse.Namespace) -> int:
    baseline = load_baseline(args.baseline)
    reference_search = read_search(args.reference_search)
    reference_steps = read_steps(args.reference_steps)
    search = read_search(args.search)
    steps = read_steps(args.steps)
    rows = digest_checks("reference", baseline, reference_search, reference_steps)
    if any(row.failure for row in rows):
        # The committed digests do not describe the reference commit, so the
        # file is wrong, not the candidate; say so before any timing.
        rows.insert(
            0,
            Check(
                "baseline file",
                baseline["reference"]["commit"][:9],
                "digests differ",
                "re-record from the reference commit",
            ),
        )
    rows += digest_checks("candidate", baseline, search, steps)
    rows += timing_checks(
        baseline["tolerance"], reference_search, reference_steps, search, steps
    )
    print(render(baseline, rows))
    return 1 if any(row.failure for row in rows) else 0


def record(args: argparse.Namespace) -> int:
    # The tolerance is a decision of its own, edited in the file; a new
    # recording keeps it.
    tolerance = (
        load_baseline(args.baseline)["tolerance"] if args.baseline.exists() else 0.10
    )
    baseline = Baseline(
        tolerance=tolerance,
        reference=Reference(
            commit=args.commit,
            date=args.date,
            run=args.run,
            runner=args.runner,
            reason=args.reason,
        ),
        bench_search=read_search(args.search),
        bench_steps=read_steps(args.steps),
    )
    args.baseline.write_text(json.dumps(baseline, indent=2) + "\n")
    print(f"Recorded {args.baseline} at {args.commit[:9]}: {args.reason}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    compare_command = commands.add_parser(
        "compare",
        help="candidate output against the reference commit's output from the same runner",
    )
    compare_command.add_argument("--baseline", type=Path, default=BASELINE)
    compare_command.add_argument("--reference-search", type=Path, required=True)
    compare_command.add_argument("--reference-steps", type=Path, required=True)
    compare_command.add_argument(
        "--search", type=Path, required=True, help="candidate bench_search output"
    )
    compare_command.add_argument(
        "--steps", type=Path, required=True, help="candidate bench_steps output"
    )
    compare_command.set_defaults(handler=compare)
    record_command = commands.add_parser(
        "record", help="rewrite the baseline from a reference measurement"
    )
    record_command.add_argument("--baseline", type=Path, default=BASELINE)
    record_command.add_argument(
        "--search", type=Path, required=True, help="reference bench_search output"
    )
    record_command.add_argument(
        "--steps", type=Path, required=True, help="reference bench_steps output"
    )
    record_command.add_argument(
        "--commit", required=True, help="reference commit, reachable from origin"
    )
    record_command.add_argument(
        "--date", required=True, help="ISO date of the measurement"
    )
    record_command.add_argument(
        "--run", required=True, help="CI run URL of the measurement"
    )
    record_command.add_argument(
        "--runner", required=True, help="runner and CPU the figures came from"
    )
    record_command.add_argument(
        "--reason", required=True, help="why the reference moves"
    )
    record_command.set_defaults(handler=record)
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
