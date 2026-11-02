"""Compare engine benchmark output with the committed performance baseline.

The two release benchmarks under ``managym/examples`` print JSON lines:
``bench_search`` one line per seeded game (worst, median and total Search-64
reply time plus a digest of its search results) and a summary line;
``bench_steps`` one line for its headless random-play workload (steps per
second and a digest of every decision). A *run directory* holds one run of
both as ``bench_search.jsonl`` and ``bench_steps.jsonl``; the Engine Budget
job writes the candidate's to ``.cache/engine-budget`` (with
``measurement.json`` naming the commit, date, run and runner) and the
reference's to ``.cache/engine-budget/reference``, and uploads the whole
directory as the ``engine-budget`` artifact.

``managym/perf/baseline.json`` names a reference commit, the digests both
workloads must reproduce, and the figures the reference measured when it was
recorded. GitHub-hosted runners land on different CPU generations (an
identical engine measured 6.8k and 10.3k steps/s on two of them), so wall time
is never compared across runs: CI rebuilds the reference commit with the
current benchmarks on the same runner and ``compare`` holds the candidate to
the reference's fresh timings within the tolerance, and both to the committed
digests. ``record`` rewrites the file from a downloaded artifact with the
reason that moves the reference. Stdlib only, so CI runs it without syncing
the project.
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
ARTIFACT = ROOT / ".cache/engine-budget"


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


class Measurement(TypedDict):
    """What the job writes beside its candidate output as ``measurement.json``."""

    commit: str
    date: str
    run: str
    runner: str


class Reference(Measurement):
    reason: str


class Baseline(TypedDict):
    tolerance: float
    reference: Reference
    bench_search: SearchResult
    bench_steps: StepsResult


@dataclass(frozen=True)
class Run:
    """One run of both benchmarks, read from a run directory."""

    search: SearchResult
    steps: StepsResult


@dataclass(frozen=True)
class Check:
    """One budget line: ``failure`` is empty when the measurement is within it."""

    name: str
    expected: str
    measured: str
    failure: str


# --- Reading -----------------------------------------------------------------


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


def read_run(directory: Path) -> Run:
    return Run(
        search=read_search(directory / "bench_search.jsonl"),
        steps=read_steps(directory / "bench_steps.jsonl"),
    )


def read_measurement(directory: Path) -> Measurement:
    """The job's ``measurement.json``; a hand-made directory fails by name."""
    path = directory / "measurement.json"
    if not path.is_file():
        raise SystemExit(
            f"{path}: not an Engine Budget artifact (download one with "
            f"`gh run download <run-id> --name engine-budget --dir {directory}`)"
        )
    row = json.loads(path.read_text())
    return Measurement(
        commit=str(row["commit"]),
        date=str(row["date"]),
        run=str(row["run"]),
        runner=str(row["runner"]),
    )


def load_baseline(path: Path) -> Baseline:
    # The file is written by ``record`` below, so its shape is trusted.
    return cast(Baseline, json.loads(path.read_text()))


# --- Checks ------------------------------------------------------------------


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


def digest_checks(label: str, baseline: Baseline, run: Run) -> list[Check]:
    """Both workloads reproduce the committed digests, seed set included."""
    expected_seeds = baseline["bench_search"]["seeds"]
    seeds = run.search["seeds"]
    expected_steps = baseline["bench_steps"]
    checks = [
        _same(
            f"{label} search seed {seed} digest",
            expected_seeds[seed]["digest"] if seed in expected_seeds else None,
            seeds[seed]["digest"] if seed in seeds else None,
        )
        for seed in sorted(set(expected_seeds) | set(seeds), key=int)
    ]
    checks += [
        _same(f"{label} steps games", expected_steps["games"], run.steps["games"]),
        _same(f"{label} steps count", expected_steps["steps"], run.steps["steps"]),
        _same(f"{label} steps digest", expected_steps["digest"], run.steps["digest"]),
    ]
    return checks


def timing_checks(tolerance: float, reference: Run, candidate: Run) -> list[Check]:
    """Candidate timings against the reference measured on the same machine."""
    return [
        _slower(
            "search worst reply",
            reference.search["worst_s"],
            candidate.search["worst_s"],
            tolerance,
        ),
        _slower(
            "search total",
            reference.search["total_s"],
            candidate.search["total_s"],
            tolerance,
        ),
        _faster(
            "steps throughput",
            reference.steps["steps_per_s"],
            candidate.steps["steps_per_s"],
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
            "adopts this run as the reference: `gh run download <run-id> --name "
            "engine-budget --dir .cache/engine-budget`, then `uv run "
            'scripts/engine_budget.py record --reason "<why>"` and commit '
            "managym/perf/baseline.json (docs/development.md, Engine performance "
            "budget)."
        )
    else:
        lines.append("PASS: within the recorded budget.")
    return "\n".join(lines)


# --- Commands ----------------------------------------------------------------


def compare(args: argparse.Namespace) -> int:
    baseline = load_baseline(args.baseline)
    reference = read_run(args.reference)
    candidate = read_run(args.artifact)
    rows = digest_checks("reference", baseline, reference)
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
    rows += digest_checks("candidate", baseline, candidate)
    rows += timing_checks(baseline["tolerance"], reference, candidate)
    print(render(baseline, rows))
    return 1 if any(row.failure for row in rows) else 0


def record(args: argparse.Namespace) -> int:
    """Adopt a downloaded artifact's candidate as the reference."""
    measurement = read_measurement(args.artifact)
    candidate = read_run(args.artifact)
    # The tolerance is a decision of its own, edited in the file; a new
    # recording keeps it.
    baseline = Baseline(
        tolerance=load_baseline(args.baseline)["tolerance"],
        reference=Reference(**measurement, reason=args.reason),
        bench_search=candidate.search,
        bench_steps=candidate.steps,
    )
    args.baseline.write_text(json.dumps(baseline, indent=2) + "\n")
    print(f"Recorded {args.baseline} at {measurement['commit'][:9]}: {args.reason}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    compare_command = commands.add_parser(
        "compare",
        help="the candidate against the reference commit's run from the same runner",
    )
    compare_command.add_argument(
        "--reference",
        type=Path,
        default=ARTIFACT / "reference",
        help="reference run directory",
    )
    compare_command.set_defaults(handler=compare)
    record_command = commands.add_parser(
        "record", help="rewrite the baseline from a downloaded Engine Budget artifact"
    )
    record_command.add_argument(
        "--reason", required=True, help="why the reference moves"
    )
    record_command.set_defaults(handler=record)
    for command in (compare_command, record_command):
        command.add_argument("--baseline", type=Path, default=BASELINE)
        command.add_argument(
            "--artifact",
            type=Path,
            default=ARTIFACT,
            help="candidate run directory: the job's output, or where "
            "`gh run download --name engine-budget` wrote it",
        )
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
