"""MTG-186 evaluation on rented CPU through Etude: the frozen grid, played remotely.

Jack Heart's direction (2026-10-25): no grid or arena evaluation on his laptop;
evaluations run remotely through Etude, inside the Task's $10 inclusive cap with
training. The games are MTG-181's frozen grid (`experiments.elemental_grid`),
played by its own `evaluate` with the same schedule, workers and bounds; only the
machine moved.

A `Point` is one checkpoint to evaluate: MTG-186's exports at updates 0, 200,
400, 600 and its endpoint on the full grid, MTG-181's update-200/400/600 exports
on the two UR/GW cells, and MTG-181's 18102/18103 endpoints on the full grid
(its addendum's grids against this heuristic, which its local run never
finished). Every checkpoint is an object in the `manabot` R2 bucket and reaches
the machine as a digest-checked Etude input.

`job` freezes one Etude job over the pending points at an explicit revision; on
the machine `grid` (run as `python -m experiments.mtg186_eval grid`) plays them
in order and writes, per point, `<name>.json` (`Result`), `<name>.games.tar.gz`
(each group's `games.json`, which carries play/draw) and `<name>.traces.tar.gz`
(the replay-checked traces) to `ETUDE_OUTPUT_DIR`. `collect` copies a finished
run's results and game rows into `experiments/data/mtg186/`; the traces stay in
the run's Etude outputs.

    uv run python -m experiments.mtg186_eval pending
    doppler run -p manabot -c prd -- doppler run -p etude -c prd -- \\
        uv run python -m experiments.mtg186_eval launch --revision <commit> --cohort <name> --max-usd <usd>
    doppler run -p manabot -c prd -- doppler run -p etude -c prd -- \\
        uv run python -m experiments.mtg186_eval collect --run <eval run>
"""

import argparse
from decimal import Decimal
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import time

from etude.launch.client import default, queue_cohort, quote, status
from etude.launch.records import Input, Job, Run
from etude.run.listing import runs as stored_runs
from etude.run.storage import storage_from_uri
from etude.train.read import read
from pydantic import BaseModel

from experiments.elemental_grid import (
    SCHEDULE_SHA256,
    GameRow,
    Summary,
    evaluate,
    schedule_sha256,
    summarize,
)
from experiments.mtg186 import CAP_USD, LAUNCHES, MILESTONES, ROOT, SEED, STORAGE
from manabot.experiment.job import _TOOLCHAIN, _setup
from manabot.infra.identity import file_sha256

RESULTS = ROOT / "experiments/data/mtg186/grids"
GAMES = ROOT / "experiments/data/mtg186/games"
BUCKET = "manabot"
RENT = "cpu32"
TRAINING_CELLS = (("elemental_ur", "elemental_gw"), ("elemental_gw", "elemental_ur"))
WANTED = (0, *MILESTONES)
# cpu3c 32 vCPUs: one bounded group worker per vCPU, less two for the session.
REMOTE_WORKERS = 30

# Training's share of the cap: run r3's job cap while it trains, then its charge.
TRAINING_USD = Decimal("5.50")
# Each evaluation cohort's cap, added before it is launched (cohorts are single-use).
EVALUATIONS: dict[str, Decimal] = {"mtg186-eval-1": Decimal("3.00")}


class Point(BaseModel):
    """One checkpoint to evaluate; `cells` is None for the full grid."""

    trial: str
    seed: int
    job: str
    updates: int
    sha256: str
    size_bytes: int
    # The checkpoint's object key in the `manabot` bucket.
    key: str
    cells: list[tuple[str, str]] | None = None

    @property
    def name(self) -> str:
        scope = "grid" if self.cells is None else "urgw"
        return f"{self.trial}-{self.seed}-{self.updates:06d}-{self.sha256[:12]}-{scope}"


class Result(BaseModel):
    point: Point
    schedule_sha256: str
    seconds: float
    summary: Summary


class GridJob(BaseModel):
    points: list[Point]
    workers: int = REMOTE_WORKERS


def _mtg181(seed: int, job: str, updates: int, sha256: str, size: int) -> Point:
    return Point(
        trial="mtg181",
        seed=seed,
        job=job,
        updates=updates,
        sha256=sha256,
        size_bytes=size,
        key=f"jobs/{job}/runtime/artifacts/sha256/{sha256}",
    )


_J1, _J2, _J3 = (
    "mtg181-elemental-two-seat-18101-r2",
    "mtg181-elemental-two-seat-18102",
    "mtg181-elemental-two-seat-18103",
)
# MTG-181's exports at updates 200/400/600 (all that exist), on the UR/GW cells.
MTG181_MATCHED = tuple(
    p.model_copy(update={"cells": list(TRAINING_CELLS)})
    for p in (
        _mtg181(18101, _J1, 200, "39f244d88396f1a313e46f4a1ca5f81063b9984db7a0b0a5587069ca3f761ab6", 1071367),
        _mtg181(18101, _J1, 400, "ee45113ff9e752ac6b14a8649559c7c5e6ad250edfbeb744b1b818471e05a8e9", 1071367),
        _mtg181(18101, _J1, 600, "a452cd255e1729f45af9e400756ea18b1b4d3aeed3581af137629069f6f0ccdb", 1071367),
        _mtg181(18102, _J2, 200, "8c401238ff9620fc1e2112c825d1f4de1652cd03bebeabdcd597ff891eb41b81", 1071367),
        _mtg181(18102, _J2, 400, "0a5535a1675b448a58e805e5dc1773da1e76d185c376d1066bffc6d76c87d11f", 1071367),
        _mtg181(18103, _J3, 200, "3fb4c1c12be7ded1e344ac8f6e9bfc8a2994b578991eabbe61503b322462e4c9", 1071367),
        _mtg181(18103, _J3, 400, "6b5766243b07dbf6cc49741d1b040881f7d178b66cd9a46883bec006cc7680be", 1071367),
    )
)  # fmt: skip
# MTG-181's endpoints (final raw policy) its addendum did not finish; 18101's is in its data.
MTG181_ENDPOINTS = (
    _mtg181(18102, _J2, 474, "1c203acb14e18bfbbf95facc368efcfa9588a3c6e17b37206dd83d0d769e13ee", 1071166),
    _mtg181(18103, _J3, 568, "feea8767b288916cbfdccf2296f54dd8d37dbc5c52cb083ac57d2a64db84e248", 1071166),
)  # fmt: skip


def mtg186_points(run: str) -> tuple[list[Point], bool]:
    """The run's stored exports at wanted updates, plus the endpoint once it is done."""
    storage = storage_from_uri(STORAGE)
    found = [(e, r) for e, r in stored_runs(storage) if r == run]
    records = [c for e, r in found for c in read(storage, e, r).checkpoints]
    done = status(run).done
    last = max((int(c.at.get("updates", -1)) for c in records), default=-1)
    points = []
    for record in records:
        updates = int(record.at.get("updates", -1))
        artifact = record.artifact
        if updates not in WANTED and not (done and updates == last):
            continue
        if artifact.sha256 is None or artifact.size_bytes is None:
            continue
        prefix = f"r2://{BUCKET}/"
        if not artifact.uri.startswith(prefix):
            raise ValueError(f"checkpoint outside the {BUCKET} bucket: {artifact.uri}")
        points.append(
            Point(
                trial="mtg186",
                seed=SEED,
                job=run,
                updates=updates,
                sha256=artifact.sha256,
                size_bytes=artifact.size_bytes,
                key=artifact.uri.removeprefix(prefix),
            )
        )
    return sorted(points, key=lambda p: p.updates), done


def pending(run: str) -> list[Point]:
    """Every wanted point without a collected result: MTG-186 first, oldest first."""
    points, _ = mtg186_points(run)
    wanted = [*points, *MTG181_MATCHED, *MTG181_ENDPOINTS]
    return [p for p in wanted if not (RESULTS / f"{p.name}.json").exists()]


# ---------------------------------------------------------------------------
# The Etude job and its machine side


def _input(point: Point) -> Input:
    return Input(
        name=f"{point.name}.pt",
        key=point.key,
        sha256=point.sha256,
        size_bytes=point.size_bytes,
    )


def job(
    points: list[Point], *, revision: str, max_hours: float, max_usd: Decimal
) -> Job:
    """One job playing `points` in order; construction launches nothing."""
    if schedule_sha256() != SCHEDULE_SHA256:
        raise ValueError("grid schedule differs from MTG-181's frozen digest")
    return Job(
        name="mtg186-grid",
        setup=_setup(""),
        command=f"{_TOOLCHAIN}\nuv run --locked python -m experiments.mtg186_eval grid",
        config=GridJob(points=points).model_dump_json(),
        repo=str(ROOT),
        revision=revision,
        max_hours=max_hours,
        max_usd=max_usd,
        max_attempts=1,
        inputs=tuple(_input(p) for p in points),
    )


def _tar(path: Path, directory: Path, names: str) -> None:
    """Pack the files under `directory` matching `names` (a glob) into `path`, atomically."""
    partial = path.with_suffix(".partial")
    with tarfile.open(partial, "w:gz") as archive:
        for file in sorted(directory.rglob(names)):
            archive.add(file, arcname=str(file.relative_to(directory)))
    partial.replace(path)


def grid(spec: GridJob) -> None:
    """On the machine: play each point, writing its result and evidence as it finishes."""
    if schedule_sha256() != SCHEDULE_SHA256:
        raise ValueError("grid schedule differs from MTG-181's frozen digest")
    inputs = Path(os.environ["ETUDE_INPUTS_DIR"])
    out = Path(os.environ["ETUDE_OUTPUT_DIR"])
    work = Path(tempfile.mkdtemp(prefix="mtg186-groups-"))
    for point in spec.points:
        checkpoint = inputs / _input(point).name
        if file_sha256(checkpoint) != point.sha256:
            raise ValueError(f"{checkpoint} differs from its recorded digest")
        started = time.time()
        directory = work / point.name
        rows = evaluate(
            checkpoint,
            point.sha256,
            point.seed,
            directory,
            workers=spec.workers,
            root=ROOT,
            cells=None if point.cells is None else frozenset(point.cells),
        )
        result = Result(
            point=point,
            schedule_sha256=SCHEDULE_SHA256,
            seconds=time.time() - started,
            summary=summarize(rows),
        )
        _tar(out / f"{point.name}.games.tar.gz", directory, "games.json")
        _tar(out / f"{point.name}.traces.tar.gz", directory, "attempt-*/*")
        partial = out / f"{point.name}.json.partial"
        partial.write_text(result.model_dump_json(indent=2) + "\n")
        partial.replace(out / f"{point.name}.json")
        overall = result.summary.overall
        print(
            f"{point.name}: {overall.score} [{overall.low}, {overall.high}],"
            f" {overall.invalid} invalid, {result.seconds:.0f}s",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Collecting results


def group_rows(name: str, x: str, y: str, seat: int) -> list[GameRow]:
    """One collected group's game rows from `<name>.games.tar.gz`; empty when absent."""
    path = GAMES / f"{name}.games.tar.gz"
    if not path.exists():
        return []
    with tarfile.open(path) as archive:
        try:
            member = archive.extractfile(f"{x}--{y}--seat{seat}/games.json")
        except KeyError:
            return []
        assert member is not None
        return [GameRow.model_validate(r) for r in json.loads(member.read())]


def collect(run: str) -> list[Result]:
    """Copy a run's results and game rows here; verifies each against its job's points."""
    scheduler = default()
    spec = GridJob.model_validate_json(scheduler.record(run).spec.config or "")
    attempt = scheduler.attempts(run)[-1]
    names = {output.name for output in attempt.outputs}
    collected = []
    with tempfile.TemporaryDirectory() as folder:
        for point in spec.points:
            if f"{point.name}.json" not in names:
                continue
            scratch = Path(folder)
            for suffix in (".json", ".games.tar.gz"):
                scheduler.fetch(run, f"{point.name}{suffix}", scratch)
            result = Result.model_validate_json(
                (scratch / f"{point.name}.json").read_text()
            )
            if result.point != point or result.schedule_sha256 != SCHEDULE_SHA256:
                raise ValueError(f"{run}: {point.name} differs from its job")
            RESULTS.mkdir(parents=True, exist_ok=True)
            GAMES.mkdir(parents=True, exist_ok=True)
            (scratch / f"{point.name}.games.tar.gz").replace(
                GAMES / f"{point.name}.games.tar.gz"
            )
            (scratch / f"{point.name}.json").replace(RESULTS / f"{point.name}.json")
            collected.append(result)
    return collected


def _commit(revision: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def remaining_usd() -> Decimal:
    """The cap less failed training launches, training's share and earlier evaluations."""
    return CAP_USD - sum(LAUNCHES.values()) - TRAINING_USD - sum(EVALUATIONS.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("pending", "grid", "quote", "launch", "collect")
    )
    parser.add_argument("--training", default="e92773d82a96", help="MTG-186's run")
    parser.add_argument("--revision")
    parser.add_argument("--cohort")
    parser.add_argument("--max-usd", type=Decimal)
    parser.add_argument("--max-hours", type=float, default=2.5)
    parser.add_argument("--run", help="the evaluation run to collect")
    args = parser.parse_args()
    if args.command == "grid":
        grid(GridJob.model_validate_json(Path(os.environ["ETUDE_CONFIG"]).read_text()))
        return
    if args.command == "collect":
        if not args.run:
            parser.error("collect needs --run")
        for result in collect(args.run):
            overall = result.summary.overall
            print(
                f"{result.point.name}: {overall.score} [{overall.low}, {overall.high}]"
            )
        return
    points = pending(args.training)
    for point in points:
        print(point.name)
    if args.command == "pending":
        return
    if not (args.revision and args.cohort and args.max_usd):
        parser.error("quote and launch need --revision, --cohort and --max-usd")
    if EVALUATIONS.get(args.cohort) != args.max_usd or remaining_usd() < 0:
        raise SystemExit(
            f"record {args.cohort} at ${args.max_usd} in EVALUATIONS within the cap;"
            f" it leaves ${remaining_usd() + EVALUATIONS.get(args.cohort, 0)}"
        )
    runs = [
        Run(
            job=job(
                points,
                revision=_commit(args.revision),
                max_hours=args.max_hours,
                max_usd=args.max_usd,
            )
        )
    ]
    priced = quote(runs, rent=RENT, cohort=args.cohort, max_usd=args.max_usd)
    print(priced.model_dump_json(indent=2))
    if args.command == "launch":
        print(queue_cohort(args.cohort, runs, rent=RENT, max_usd=args.max_usd))


if __name__ == "__main__":
    main()
