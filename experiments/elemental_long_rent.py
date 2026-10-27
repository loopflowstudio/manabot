"""MTG-183's evaluation on rented CPU: units the Mac mini has not reached, newest first.

Jack Heart's direction (2026-10-26): spend at most $10 on rented machines to play
as much of MTG-183's frozen evaluation design as possible, and measure how fast
shared batched inference plays it. The design, schedules and game rules are
`experiments.elemental_long_eval`'s; only where and how fast games run changes.

A rented machine gets prefix-scoped storage keys for its own run folder only,
so it neither reads training's tracking bucket nor writes the evaluation
bucket. The operator therefore admits checkpoints here (`admitted`), picks
pending units newest first (`plan`) and freezes each batch as one Etude job
(`job`): the `EvalState`, unit keys and every checkpoint the units seat, as
digest-checked inputs. On the machine, `machine` plays the batch with
`etude.eval.match.evaluate` into a local eval run and returns that run's folder
as `<run>.tar` plus `receipt.json` (wall time, games, workers, compute).
`collect` folds a finished batch's run folder into MTG-183's combined
database (`experiments.elemental_long_db`), which is published as one object.

Two eval runs keep two compute contracts apart: `mtg183-eval-rent` plays the
frozen scalar path (one Torch thread, one decision per forward pass), and
`mtg183-eval-batched` plays each group's games concurrently through one
batched inference service per group (`EvalCompute`), and `mtg183-eval-gpu`
plays every group on the machine through one CUDA inference service, so
concurrent games share one batch (`manabot.eval.endpoint.callback_allocation`).
The report treats them as separate sources until their agreement is checked.

    uv run --extra artifacts --extra evaluation python -m experiments.elemental_long_rent \\
        launch --batch-games 9000 --mode scalar --mode batched [--unit KEY]
    ... launch --rent l4 --mode gpu --unit KEY
    uv run --extra artifacts --extra evaluation python -m experiments.elemental_long_rent \\
        collect <etude run id>
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from contextlib import AbstractContextManager, nullcontext
from decimal import Decimal
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from typing import Literal

from etude.eval.match import evaluate
from etude.eval.records import Match, Model
from etude.launch.records import Input, Job, Run
from etude.run.storage import run_directory
from pydantic import BaseModel, ConfigDict

from experiments.elemental_grid import ENDPOINT, endpoint_environment
from experiments.elemental_long_db import MIRROR, BucketFolder, LocalFolder, ingest
from experiments.elemental_long_eval import (
    CARDS,
    COMPUTE,
    EVIDENCE,
    EXPERIMENT,
    GROUP_WORKERS,
    ROOT,
    Conditions,
    EvalState,
    Unit,
    admit,
    eval_environment,
    play,
    segment_source,
    units,
    window_checkpoints,
)
from manabot.eval.endpoint import InferenceEndpoint, callback_allocation
from manabot.experiment.job import TOOLCHAIN_ENV, machine_setup
from manabot.infra.identity import file_sha256
from manabot.infra.source import SourceBundle, bind_etude_source, bundle_at
from manabot.sim.execution import EvalCompute, EvalExecution

SCALAR_RUN = "mtg183-eval-rent"
BATCHED_RUN = "mtg183-eval-batched"
# The mini's eval run: units it has stored are not rented.
MINI_RUN = "mtg183-eval-cards"
BUCKET = "manabot"
EVALUATIONS = f"r2://{BUCKET}/evaluations"
TRACKING = f"r2://{BUCKET}/jobs"
WINDOWS = ROOT / "experiments/protocols/elemental-183-windows.json"
SEGMENTS = ("e7d47d8b220f", "f3748bcabac8")
# Each rental's $/hour ceiling, which a batch's cap must cover.
CEILINGS = {"cpu32": Decimal("2.0"), "l4": Decimal("0.60"), "a4500": Decimal("0.40")}
# cpu3c 32 vCPUs: one single-threaded group worker per vCPU, less two (MTG-186):
# ten matches at once (a unit has ten), three groups each.
WORKERS = 10
GROUP_WORKERS_SCALAR = 3
# Each batched group plays all its games at once; a few groups share the machine.
BATCHED = EvalCompute(max_active_games=12, max_batch_rows=16)
BATCHED_WORKERS = 6
GPU_RUN = "mtg183-eval-gpu"
# One machine-wide CUDA service: every concurrent game's decisions share a batch.
# Each game's native child imports Torch (~225 MB), so active games are sized
# to the A4500's 32 GB, not to the GPU.
GPU = EvalCompute(device="cuda", max_active_games=64, max_batch_rows=64)
# Ten matches, four groups each: enough pending games to keep 64 active.
GPU_WORKERS = 10
GROUP_WORKERS_GPU = 4
Mode = Literal["scalar", "batched", "gpu"]
# Each mode's eval run, concurrent matches, groups per match and compute contract.
MODES: dict[Mode, tuple[str, int, int, EvalCompute | None]] = {
    "scalar": (SCALAR_RUN, WORKERS, GROUP_WORKERS_SCALAR, None),
    "batched": (BATCHED_RUN, BATCHED_WORKERS, 1, BATCHED),
    "gpu": (GPU_RUN, GPU_WORKERS, GROUP_WORKERS_GPU, GPU),
}
# A batch starts no unit it would not finish this long before its hours end.
STOP_SECONDS = 600
CAP_USD = Decimal("10")
# A batch's cap must cover its hours and Etude's 15-minute setup allowance at
# its rental's ceiling; cpu3c is quoted at $0.96/hour, an L4 at $0.59 and an
# RTX A4500 (12 vCPUs) at $0.25.
SETUP_HOURS = Decimal("0.25")
# Launched batches, so a relaunch neither repeats nor forgets one.
LEDGER = ROOT / "experiments/protocols/elemental-183-rent.json"
EXTRAS = " --extra artifacts --extra evaluation"


class Batch(BaseModel):
    """One rented job's work: the job config `machine` reads."""

    model_config = ConfigDict(extra="forbid")
    run: str
    state: EvalState
    units: tuple[str, ...]
    workers: int
    compute: EvalCompute | None
    # Play every group through one machine-wide allocation under `compute`.
    shared: bool = False
    # Groups of one match playing at once.
    group_workers: int = 1
    # Play only each unit's first matches (calibration); None plays them all.
    match_limit: int | None = None
    # Demo (Etude blocks): play only each group's first deals, and start no
    # unit past this many seconds of play; None plays the design as frozen.
    deals_per_group: int | None = None
    play_seconds: float | None = None
    hours: float = 1.0
    # Read by `bind_etude_source` before anything reads source identity.
    source_bundle: SourceBundle


class Played(BaseModel):
    unit: str
    games: int
    seconds: float


class Receipt(BaseModel):
    """What a batch played and how fast: units in order, with each unit's wall time."""

    run: str
    units: tuple[str, ...]
    games: int
    seconds: float
    workers: int
    group_workers: int
    compute: EvalCompute | None
    played: list[Played] = []
    cpus: int | None
    error: str | None


class Ledger(BaseModel):
    """Launched Etude runs by id, with the units each holds and its cap."""

    launched: dict[str, Batch] = {}
    max_usd: dict[str, Decimal] = {}
    # A finished run's settled cost replaces its cap in the committed total.
    spent: dict[str, Decimal] = {}
    collected: list[str] = []

    def committed(self) -> Decimal:
        return sum(
            (self.spent.get(run, cap) for run, cap in self.max_usd.items()), Decimal(0)
        )


# ---------------------------------------------------------------------------
# On the machine


def _checkpoint_inputs(state: EvalState, chosen: Iterable[Unit]) -> dict[str, int]:
    """Each seated checkpoint's digest and size."""
    found: dict[str, int] = {}
    for unit in chosen:
        for match in unit.matches:
            for seated in match.players:
                if isinstance(seated, Model) and seated.checkpoint.artifact.sha256:
                    artifact = seated.checkpoint.artifact
                    found[artifact.sha256] = artifact.size_bytes or 0
    return found


def _chosen(batch: Batch) -> list[Unit]:
    wanted = set(batch.units)
    chosen = [unit for unit in units(batch.state) if unit.key in wanted]
    if {unit.key for unit in chosen} != wanted:
        raise ValueError("a batch unit is not in its state's design")
    return chosen


def machine(batch: Batch) -> None:
    """Play the batch into a local eval run; return its folder and a receipt."""
    bind_etude_source()
    inputs = Path(os.environ["ETUDE_INPUTS_DIR"])
    out = Path(os.environ["ETUDE_OUTPUT_DIR"])
    work = Path(tempfile.mkdtemp(prefix="mtg183-rent-"))
    chosen = _chosen(batch)
    # `play` finds checkpoints in the data cache by digest; the inputs are those bytes.
    cache = work / "cache"
    cache.mkdir()
    os.environ["MANABOT_DATA_CACHE"] = str(cache)
    for sha256 in _checkpoint_inputs(batch.state, chosen):
        source = inputs / f"{sha256}.pt"
        if file_sha256(source) != sha256:
            raise ValueError(f"input {source.name} differs from its digest")
        (cache / sha256).symlink_to(source)
    track = work / "track"
    root = run_directory(track, EXPERIMENT, batch.run)[1].resolve()
    os.environ[EVIDENCE] = str(work / "arena")
    # Workers write each game's evidence here; `play` returns it as the match's
    # blobs, so the run folder holds only Etude's sealed blocks.
    os.environ[CARDS] = str(work / "evidence")
    if batch.compute is not None and not batch.shared:
        os.environ[COMPUTE] = batch.compute.model_dump_json()
    allocation: AbstractContextManager[InferenceEndpoint | None] = nullcontext(None)
    if batch.shared:
        if batch.compute is None:
            raise ValueError("a shared allocation needs a compute contract")
        execution = EvalExecution(
            **batch.compute.model_dump(),
            deadline_unix=time.time() + batch.hours * 3600,
        )
        allocation = callback_allocation(execution, work / "allocation")
    if batch.group_workers > 1:
        os.environ[GROUP_WORKERS] = str(batch.group_workers)
    # Units play one at a time, each with every worker, and the run folder is
    # returned after each; no unit starts unless the slowest rate so far would
    # finish it before `STOP_SECONDS` short of the batch's hours.
    started = time.time()
    stop = started + batch.hours * 3600 - STOP_SECONDS
    playing = stop if batch.play_seconds is None else started + batch.play_seconds
    receipt = Receipt(
        run=batch.run,
        units=(),
        games=0,
        seconds=0.0,
        workers=batch.workers,
        group_workers=batch.group_workers,
        compute=batch.compute,
        cpus=os.cpu_count(),
        error=None,
    )

    def watch() -> None:
        """Report progress each minute; at `stop`, return what is stored and end.

        Etude kills a run at its deadline and then returns none of its outputs,
        so the batch ends itself first, mid-unit if need be.
        """
        while True:
            time.sleep(60)
            print(_progress(root, started), file=sys.stderr, flush=True)
            if time.time() > stop:
                receipt.seconds = time.time() - started
                receipt.error = "stopped at the batch's own deadline"
                _deliver(out, root, work, receipt)
                os._exit(0)

    threading.Thread(target=watch, daemon=True).start()
    try:
        with allocation as endpoint:
            if endpoint is not None:
                os.environ[ENDPOINT] = endpoint_environment(endpoint)
            for unit in chosen:
                matches = [
                    _trimmed(match, batch.deals_per_group)
                    for match in unit.matches[: batch.match_limit]
                ]
                games = sum(match.games for match in matches)
                rate = min((u.games / u.seconds for u in receipt.played), default=None)
                if rate is not None and time.time() + games / rate > playing:
                    break
                begun = time.time()
                evaluate(
                    experiment=EXPERIMENT,
                    environment=eval_environment(),
                    matches=matches,
                    play=play,
                    workers=batch.workers,
                    run=batch.run,
                    directory=track,
                )
                receipt.played.append(
                    Played(unit=unit.key, games=games, seconds=time.time() - begun)
                )
                receipt.units += (unit.key,)
                receipt.games += games
                receipt.seconds = time.time() - started
                _deliver(out, root, work, receipt)
    except Exception as exc:
        receipt.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        receipt.seconds = time.time() - started
        print(receipt.model_dump_json(), flush=True)
        _deliver(out, root, work, receipt)


def _trimmed(match: Match, deals: int | None) -> Match:
    """The match with each group cut to its first `deals` deals; itself for None."""
    if deals is None:
        return match
    conditions = Conditions.model_validate(match.conditions)
    cut = tuple(
        group.model_copy(update={"deals": group.deals[:deals]})
        for group in conditions.groups
    )
    return Match(
        players=match.players,
        conditions=conditions.model_copy(update={"groups": cut}).model_dump(
            mode="json"
        ),
        games=sum(len(group.deals) for group in cut),
    )


def _progress(root: Path, started: float) -> str:
    """Sealed blocks so far, load and memory, as JSON."""
    games = sum(1 for _ in root.glob("blocks/*/block.json"))
    processes = (
        [p for p in Path("/proc").glob("[0-9]*")] if Path("/proc").is_dir() else []
    )
    threads = 0
    for process in processes:
        try:
            status = (process / "status").read_text()
        except OSError:
            continue
        threads += int(status.split("Threads:")[1].split()[0])
    memory = {}
    if Path("/proc/meminfo").is_file():
        for line in Path("/proc/meminfo").read_text().splitlines():
            name, value = line.split(":")
            if name in ("MemTotal", "MemAvailable"):
                memory[name] = int(value.split()[0]) // 1024
    elapsed = time.time() - started
    return json.dumps(
        {
            "elapsed": round(elapsed),
            "blocks": games,
            "load": [round(x, 1) for x in os.getloadavg()],
            "processes": len(processes),
            "threads": threads,
            "memory_mib": memory,
        }
    )


def _deliver(out: Path, root: Path, work: Path, receipt: Receipt) -> None:
    """Replace the returned run folder, allocation receipts and receipt."""
    with tarfile.open(out / f"{receipt.run}.partial", "w") as archive:
        archive.add(root, arcname=receipt.run)
    (out / f"{receipt.run}.partial").replace(out / f"{receipt.run}.tar")
    # The shared allocation's serving and Perf receipts: where its time went.
    if (work / "allocation").is_dir():
        with tarfile.open(out / "allocation.partial", "w") as archive:
            archive.add(work / "allocation", arcname="allocation")
        (out / "allocation.partial").replace(out / "allocation.tar")
    (out / "receipt.json").write_text(receipt.model_dump_json(indent=1) + "\n")


# ---------------------------------------------------------------------------
# Operator: admission, planning, launching and collecting


def admitted() -> EvalState:
    """Every export the mini's follower would admit now, read from the buckets."""
    state = EvalState()
    admit(
        state,
        window_checkpoints(WINDOWS),
        [segment_source(TRACKING, run) for run in SEGMENTS],
    )
    if state.error is not None:
        raise RuntimeError(state.error)
    return state


def _match_identity(match: Match) -> tuple[str, str, str, str]:
    conditions = match.conditions
    assert isinstance(conditions, dict)
    return (
        match.players[0].name,
        match.players[1].name,
        str(conditions["kind"]),
        str(conditions["deck"]),
    )


def stored() -> set[tuple[str, str, str, str]]:
    """Matches stored by any MTG-183 eval run, from the combined database's mirror.

    The mini's run is folded in first, so its newest matches count. Runs at
    different source revisions key the same match differently, so matches are
    compared by what they play, not by key.
    """
    ingest(BucketFolder(MINI_RUN))
    return {
        _match_identity(Match.model_validate(json.loads(path.read_bytes())["match"]))
        for path in MIRROR.glob("*/*/matches/*.json")
        if not path.name.endswith(".failed.json")
    }


def plan(
    design: Sequence[Unit],
    done: set[tuple[str, str, str, str]],
    taken: set[str],
    games: int,
) -> list[Unit]:
    """Pending units newest first, up to about `games` games."""
    chosen: list[Unit] = []
    total = 0
    for unit in sorted(design, key=lambda u: (-u.newer, u.older is not None)):
        if unit.key in taken or all(_match_identity(m) in done for m in unit.matches):
            continue
        if chosen and total + sum(m.games for m in unit.matches) > games:
            break
        chosen.append(unit)
        total += sum(m.games for m in unit.matches)
    return chosen


def job(batch: Batch, *, revision: str, max_hours: float, max_usd: Decimal) -> Job:
    """One batch's job; construction launches nothing."""
    chosen = _chosen(batch)
    inputs = []
    for unit in chosen:
        for match in unit.matches:
            for seated in match.players:
                if not isinstance(seated, Model):
                    continue
                reference = seated.checkpoint.artifact
                if reference.sha256 is None:
                    continue
                name = f"{reference.sha256}.pt"
                if any(found.name == name for found in inputs):
                    continue
                prefix = f"r2://{BUCKET}/"
                if not reference.uri.startswith(prefix):
                    raise ValueError(f"{seated.name} is not in the {BUCKET} bucket")
                inputs.append(
                    Input(
                        name=name,
                        key=reference.uri.removeprefix(prefix),
                        sha256=reference.sha256,
                        size_bytes=reference.size_bytes,
                    )
                )
    return Job(
        name=f"mtg183-{batch.run.removeprefix('mtg183-')}",
        setup=machine_setup(EXTRAS),
        command=(
            f"{TOOLCHAIN_ENV}\nuv run --locked{EXTRAS}"
            " python -m experiments.elemental_long_rent machine"
        ),
        config=batch.model_dump_json(),
        repo=str(ROOT),
        revision=revision,
        max_hours=max_hours,
        max_usd=max_usd,
        max_attempts=1,
        stop_seconds=120.0,
        inputs=tuple(inputs),
        environment=(
            {"name": "OMP_NUM_THREADS", "value": "1"},
            {"name": "MKL_NUM_THREADS", "value": "1"},
        ),
    )


def _ledger() -> Ledger:
    return (
        Ledger.model_validate_json(LEDGER.read_text()) if LEDGER.exists() else Ledger()
    )


def launch(
    batch_games: int,
    modes: Sequence[Mode],
    hours: float,
    fixed: Sequence[str] = (),
    rent: str = "cpu32",
    match_limit: int | None = None,
) -> None:
    """Queue one batch per mode as one cohort on one `rent` rental.

    Batches take pending units newest first; `fixed` units instead go to every
    batch, which plays the same games under each compute contract.
    """
    from etude.launch.client import queue_cohort, quote
    from etude.launch.errors import LauncherError

    ledger = _ledger()
    usd = (CEILINGS[rent] * (Decimal(str(hours)) + SETUP_HOURS)).quantize(
        Decimal("0.01")
    )
    committed = ledger.committed()
    if committed + usd * len(modes) > CAP_USD:
        raise SystemExit(f"${committed} committed; {len(modes)} more exceed ${CAP_USD}")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    # The ledger itself changes with every launch and is not job source.
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", ".", f":!{LEDGER.relative_to(ROOT)}"],
        cwd=ROOT,
        text=True,
    )
    if dirty:
        raise SystemExit("commit first: the job runs an exact revision")
    state = admitted()
    design = units(state)
    done = stored()
    # Units of batches still live; a settled batch's played matches are `done`,
    # and the rest of its units are pending again.
    taken = {
        key
        for run, batch in ledger.launched.items()
        if run not in ledger.spent
        for key in batch.units
    }
    bundle = bundle_at(ROOT, revision)
    runs, frozen = [], []
    for mode in modes:
        if fixed:
            chosen = [unit for unit in design if unit.key in set(fixed)]
        else:
            chosen = plan(design, done, taken, batch_games)
            taken |= {unit.key for unit in chosen}
        if not chosen:
            break
        run, workers, group_workers, compute = MODES[mode]
        batch = Batch(
            run=run,
            state=state,
            units=tuple(unit.key for unit in chosen),
            workers=workers,
            group_workers=group_workers,
            compute=compute,
            shared=mode == "gpu",
            match_limit=match_limit,
            hours=hours,
            source_bundle=bundle,
        )
        frozen.append(batch)
        runs.append(
            Run(job=job(batch, revision=revision, max_hours=hours, max_usd=usd))
        )
        games = sum(m.games for u in chosen for m in u.matches)
        print(f"{batch.run}: {games} games, {batch.units[0]} … {batch.units[-1]}")
    if not runs:
        raise SystemExit("nothing pending")
    cohort = f"mtg183-rent-{int(time.time())}"
    try:
        print(quote(runs, rent=rent, cohort=cohort, max_usd=usd * len(runs)))
    except LauncherError as unavailable:
        # Queued runs wait for a machine; the cap still bounds what they spend.
        print(f"queueing to wait: {unavailable}")
    ids = queue_cohort(cohort, runs, rent=rent, max_usd=usd * len(runs))
    for run_id, batch in zip(ids, frozen, strict=True):
        ledger.launched[run_id] = batch
        ledger.max_usd[run_id] = usd
    LEDGER.write_text(ledger.model_dump_json(indent=1) + "\n")
    print("queued", *ids)


DEMO_RUN = "eb70f8-demo-gpu"


def demo(
    count: int, deals: int, play_seconds: float, hours: float, rent: str, usd: Decimal
) -> str:
    """Launch the Etude-blocks demo: `count` newest heuristic grids, trimmed, on one GPU.

    Its own eval run (`DEMO_RUN`), played into the machine's disk and returned
    as `<run>.tar`; it writes no shared bucket folder and no ledger entry.
    """
    from etude.launch.client import launch as launch_run

    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
        raise SystemExit("commit first: the job runs an exact revision")
    state = admitted()
    grids = sorted(
        (unit for unit in units(state) if unit.older is None), key=lambda u: -u.newer
    )[:count]
    batch = Batch(
        run=DEMO_RUN,
        state=state,
        units=tuple(unit.key for unit in grids),
        workers=GPU_WORKERS,
        group_workers=GROUP_WORKERS_GPU,
        compute=GPU,
        shared=True,
        deals_per_group=deals,
        play_seconds=play_seconds,
        hours=hours,
        source_bundle=bundle_at(ROOT, revision),
    )
    print(batch.units)
    run_id = launch_run(
        Run(job=job(batch, revision=revision, max_hours=hours, max_usd=usd)), rent=rent
    )
    print("launched", run_id)
    return run_id


def collect(run_id: str) -> Receipt:
    """Fold a finished batch's returned run folder into MTG-183's combined database.

    `experiments.elemental_long_db publish` then uploads that database as one object.
    """
    from etude.launch.client import default

    ledger = _ledger()
    batch = ledger.launched[run_id]
    scheduler = default()
    with tempfile.TemporaryDirectory() as folder:
        receipt = Receipt.model_validate_json(
            Path(scheduler.fetch(run_id, "receipt.json", Path(folder))).read_text()
        )
        archive = Path(scheduler.fetch(run_id, f"{batch.run}.tar", Path(folder)))
        with tarfile.open(archive) as opened:
            opened.extractall(folder, filter="data")
        # Folded into the combined database, never copied object by object.
        print(ingest(LocalFolder(batch.run, Path(folder) / batch.run, run_id)))
    if run_id not in ledger.collected:
        ledger.collected.append(run_id)
        LEDGER.write_text(ledger.model_dump_json(indent=1) + "\n")
    print(receipt.model_dump_json(indent=1))
    return receipt


def main() -> None:
    if sys.argv[1:] == ["machine"]:
        machine(Batch.model_validate_json(Path(os.environ["ETUDE_CONFIG"]).read_text()))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    launching = commands.add_parser("launch")
    launching.add_argument("--batch-games", type=int, required=True)
    # One batch per mode, in order.
    launching.add_argument("--mode", action="append", choices=tuple(MODES))
    launching.add_argument("--rent", choices=tuple(CEILINGS), default="cpu32")
    launching.add_argument("--hours", type=float, default=1.5)
    launching.add_argument("--unit", action="append", default=[])
    launching.add_argument("--match-limit", type=int)
    collecting = commands.add_parser("collect")
    collecting.add_argument("run")
    # A finished run's settled cost, as `etude runs` reports it.
    settling = commands.add_parser("settle")
    settling.add_argument("run")
    settling.add_argument("usd", type=Decimal)
    demoing = commands.add_parser("demo")
    demoing.add_argument("--units", type=int, default=6)
    demoing.add_argument("--deals", type=int, default=2)
    demoing.add_argument("--play-seconds", type=float, default=600)
    demoing.add_argument("--hours", type=float, default=0.75)
    demoing.add_argument("--rent", default="l4")
    demoing.add_argument("--max-usd", type=Decimal, default=Decimal("1.00"))
    args = parser.parse_args()
    if args.command == "demo":
        demo(
            args.units,
            args.deals,
            args.play_seconds,
            args.hours,
            args.rent,
            args.max_usd,
        )
    elif args.command == "launch":
        launch(
            args.batch_games,
            args.mode or ["scalar"],
            args.hours,
            args.unit,
            args.rent,
            args.match_limit,
        )
    elif args.command == "settle":
        ledger = _ledger()
        if args.run not in ledger.launched:
            raise SystemExit(f"{args.run} was not launched here")
        ledger.spent[args.run] = args.usd
        LEDGER.write_text(ledger.model_dump_json(indent=1) + "\n")
        print(f"committed ${ledger.committed()} of ${CAP_USD}")
    else:
        collect(args.run)


if __name__ == "__main__":
    main()
