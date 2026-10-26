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
`collect` copies a finished batch's run folder into
`r2://manabot/evaluations/<run>/`, beside the mini's `mtg183-eval-cards`.

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
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Literal

from etude.eval.match import evaluate
from etude.eval.records import Match
from etude.launch.records import Input, Job, Run
from etude.run.storage import run_directory, storage_from_uri
from pydantic import BaseModel, ConfigDict

from experiments.elemental_grid import ENDPOINT, endpoint_environment
from experiments.elemental_long_eval import (
    CARDS,
    COMPUTE,
    EVIDENCE,
    EXPERIMENT,
    ROOT,
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
CEILINGS = {"cpu32": Decimal("2.0"), "l4": Decimal("0.60")}
# cpu3c 32 vCPUs: one single-threaded group worker per vCPU, less two (MTG-186).
WORKERS = 30
# Each batched group plays all its games at once; a few groups share the machine.
BATCHED = EvalCompute(max_active_games=12, max_batch_rows=16)
BATCHED_WORKERS = 6
GPU_RUN = "mtg183-eval-gpu"
# One machine-wide CUDA service: every concurrent game's decisions share a batch.
GPU = EvalCompute(device="cuda", max_active_games=256, max_batch_rows=256)
# Concurrent matches; each plays its groups in turn, so this bounds pending games.
GPU_WORKERS = 40
Mode = Literal["scalar", "batched", "gpu"]
# Each mode's eval run, concurrent matches and compute contract.
MODES: dict[Mode, tuple[str, int, EvalCompute | None]] = {
    "scalar": (SCALAR_RUN, WORKERS, None),
    "batched": (BATCHED_RUN, BATCHED_WORKERS, BATCHED),
    "gpu": (GPU_RUN, GPU_WORKERS, GPU),
}
CAP_USD = Decimal("10")
# A batch's cap must cover its hours and Etude's 15-minute setup allowance at
# its rental's ceiling; cpu3c is quoted at $0.96/hour, an L4 at $0.59.
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
    hours: float = 1.0
    # Read by `bind_etude_source` before anything reads source identity.
    source_bundle: SourceBundle


class Receipt(BaseModel):
    run: str
    units: tuple[str, ...]
    games: int
    seconds: float
    workers: int
    compute: EvalCompute | None
    cpus: int | None
    error: str | None


class Ledger(BaseModel):
    """Launched Etude runs by id, with the units each holds and its cap."""

    launched: dict[str, Batch] = {}
    max_usd: dict[str, Decimal] = {}
    collected: list[str] = []


# ---------------------------------------------------------------------------
# On the machine


def _checkpoint_inputs(state: EvalState, chosen: Iterable[Unit]) -> dict[str, int]:
    """Each seated checkpoint's digest and size."""
    found: dict[str, int] = {}
    for unit in chosen:
        for match in unit.matches:
            for seated in match.agents:
                if seated.artifact is not None and seated.artifact.sha256:
                    found[seated.artifact.sha256] = seated.artifact.size_bytes or 0
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
    # Etude's stop signal unwinds through `finally`, so a stopped batch still
    # returns the matches it stored.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
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
    os.environ[CARDS] = str(root)
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
    matches: list[Match] = [match for unit in chosen for match in unit.matches]
    started, error = time.time(), None
    try:
        with allocation as endpoint:
            if endpoint is not None:
                os.environ[ENDPOINT] = endpoint_environment(endpoint)
            evaluate(
                experiment=EXPERIMENT,
                environment=eval_environment(),
                matches=matches,
                play=play,
                workers=batch.workers,
                run=batch.run,
                directory=track,
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        receipt = Receipt(
            run=batch.run,
            units=batch.units,
            games=sum(match.games for match in matches),
            seconds=time.time() - started,
            workers=batch.workers,
            compute=batch.compute,
            cpus=os.cpu_count(),
            error=error,
        )
        print(receipt.model_dump_json(), flush=True)
        with tarfile.open(out / f"{batch.run}.partial", "w") as archive:
            archive.add(root, arcname=batch.run)
        (out / f"{batch.run}.partial").replace(out / f"{batch.run}.tar")
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
        match.agents[0].name,
        match.agents[1].name,
        str(conditions["kind"]),
        str(conditions["deck"]),
    )


def stored(runs: Sequence[str]) -> set[tuple[str, str, str, str]]:
    """Matches stored by any of these eval runs, by seated names and conditions.

    Runs at different source revisions key the same match differently, so
    matches are compared by what they play, not by key.
    """
    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    found: set[tuple[str, str, str, str]] = set()
    for run in runs:
        pages = client.get_paginator("list_objects_v2").paginate(
            Bucket=BUCKET, Prefix=f"evaluations/{run}/matches/"
        )
        for page in pages:
            for entry in page.get("Contents", []):
                if entry["Key"].endswith(".failed.json"):
                    continue
                body = client.get_object(Bucket=BUCKET, Key=entry["Key"])["Body"]
                found.add(
                    _match_identity(Match.model_validate(json.load(body)["match"]))
                )
    return found


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
            for seated in match.agents:
                reference = seated.artifact
                if reference is None or reference.sha256 is None:
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
) -> None:
    """Queue one batch per mode as one cohort on one `rent` rental.

    Batches take pending units newest first; `fixed` units instead go to every
    batch, which plays the same games under each compute contract.
    """
    from etude.launch.client import queue_cohort, quote

    ledger = _ledger()
    usd = (CEILINGS[rent] * (Decimal(str(hours)) + SETUP_HOURS)).quantize(
        Decimal("0.01")
    )
    committed = sum(ledger.max_usd.values(), Decimal(0))
    if committed + usd * len(modes) > CAP_USD:
        raise SystemExit(f"${committed} committed; {len(modes)} more exceed ${CAP_USD}")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
        raise SystemExit("commit first: the job runs an exact revision")
    state = admitted()
    design = units(state)
    done = stored([MINI_RUN, SCALAR_RUN, BATCHED_RUN, GPU_RUN])
    taken = {key for batch in ledger.launched.values() for key in batch.units}
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
        run, workers, compute = MODES[mode]
        batch = Batch(
            run=run,
            state=state,
            units=tuple(unit.key for unit in chosen),
            workers=workers,
            compute=compute,
            shared=mode == "gpu",
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
    print(quote(runs, rent=rent, cohort=cohort, max_usd=usd * len(runs)))
    ids = queue_cohort(cohort, runs, rent=rent, max_usd=usd * len(runs))
    for run_id, batch in zip(ids, frozen, strict=True):
        ledger.launched[run_id] = batch
        ledger.max_usd[run_id] = usd
    LEDGER.write_text(ledger.model_dump_json(indent=1) + "\n")
    print("queued", *ids)


def collect(run_id: str) -> Receipt:
    """Copy a finished batch's eval run folder to the evaluations bucket."""
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
        root = Path(folder) / batch.run
        destination = storage_from_uri(f"{EVALUATIONS}/{batch.run}")
        for path in sorted(root.rglob("*")):
            if path.is_file():
                destination.put(str(path.relative_to(root)), path.read_bytes())
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
    collecting = commands.add_parser("collect")
    collecting.add_argument("run")
    args = parser.parse_args()
    if args.command == "launch":
        launch(
            args.batch_games, args.mode or ["scalar"], args.hours, args.unit, args.rent
        )
    else:
        collect(args.run)


if __name__ == "__main__":
    main()
