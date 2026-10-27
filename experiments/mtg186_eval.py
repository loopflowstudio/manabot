"""MTG-186 evaluation: head-to-head against MTG-181 seed 18101 at matched updates.

Jack Heart's direction (2026-10-25): the two models play each other directly,
matched on updates, remotely through Etude on rented CPU (never on his laptop),
inside the Task's $10 inclusive cap. No grids against the heuristic.

A `Pairing` is MTG-186 seed 18601 against MTG-181 seed 18101 at one update:
u200, u400, u600, then the endpoints (MTG-186's final export against 18101's
u616). Each pairing plays the same 2,000 games (`schedule`, pinned by
`SCHEDULE_SHA256`):

- **UR/GW**, 1,000 games: each model on UR against the other on GW, each on the
  play and on the draw, 250 games per combination. All four combinations use
  the same 250 deals.
- **Rest**, 1,000 games: 500 ordered deck pairs drawn with a fixed seed from the
  other 98 (mirrors included), each deal played once with MTG-186 on the play
  and once on the draw. The report pools them into one score.

Seat 0 takes the first turn. MTG-186 is arena player A. Rows reuse the grid's
`Group`/`GameRow`: `model_*` is MTG-186 and `heuristic_deck` is MTG-181's deck.
The score is MTG-186's (win 1, draw ½, loss 0). Only valid games count:
terminated, untruncated, no failure, trace replayed. A group that fails or
reaches its wall bound keeps its games as invalid and is never replayed.

`job` freezes one Etude job per pairing at an explicit commit, on the `cpu32`
on-demand. Both checkpoints come from the `manabot` R2 bucket as
digest-checked inputs. On the machine `pair` plays bounded worker processes
(`python -m experiments.mtg186_eval worker`), one per group of at most 25 deals.
It writes `<pairing>.json` (`PairResult`, every game row) and
`<pairing>.traces.tar.gz` to `ETUDE_OUTPUT_DIR`. `follow` runs on the operator's
machine and plays nothing. It launches each pairing as soon as both
checkpoints exist, drives the run with `etude wait`, and collects the result into
`experiments/data/mtg186/pairings/`.

    uv run python -m experiments.mtg186_eval schedule
    doppler run -p manabot -c prd -- doppler run -p etude -c prd -- \\
        uv run python -m experiments.mtg186_eval follow --revision <commit>
"""

import argparse
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from decimal import Decimal
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Literal

from etude.launch.client import default, queue_cohort, quote, status
from etude.launch.records import Input, Job, Run
from etude.run.listing import runs as stored_runs
from etude.run.storage import storage_from_uri
from etude.train.read import read
from pydantic import BaseModel, ConfigDict

from experiments.elemental_grid import (
    DECKS,
    GAME_SECONDS,
    GROUP_SECONDS,
    MAX_COMMANDS,
    PACK,
    GameRow,
    Group,
    _legs,
)
from experiments.mtg186 import CAP_USD, LAUNCHES, MILESTONES, ROOT, SEED, STORAGE
from manabot.eval.models import EvaluationKey
from manabot.experiment.job import TOOLCHAIN_ENV, machine_setup
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.sim.matchup import AuthoredMatchup
from manabot.sim.registration import PlayerRegistration
from manabot.sim.threads import child_environment
import managym

RESULTS = ROOT / "experiments/data/mtg186/pairings"
STATE = ROOT / ".runs/mtg186/pairings.json"
TRAINING_RUN = "e92773d82a96"
BUCKET = "manabot"
RENT = "cpu32"
UR, GW = "elemental_ur", "elemental_gw"
TRAINING_CELLS = ((UR, GW), (GW, UR))

PAIR_SEED = 18_690_001
DEAL_BASE = 1_869_000_000
URGW_DEALS = 250
REST_DRAWS = 500
GROUP_DEALS = 25
# cpu3c 32 vCPUs: one single-threaded group worker per vCPU, less two.
WORKERS = 30

# Spending against the $10 cap: failed training launches (`LAUNCHES`), training's
# share (r3's $5.50 job cap until its charge is known), the cancelled
# heuristic-grid evaluation, then each pairing's own cap.
TRAINING_USD = Decimal("5.50")
CANCELLED_USD = {"mtg186-eval-1": Decimal("0.30")}
PAIRING_USD = Decimal("1.00")


# ---------------------------------------------------------------------------
# The frozen schedule


@dataclass(frozen=True)
class Game:
    deck_186: str
    deck_181: str
    seat_186: Literal[0, 1]
    deal: int


def schedule() -> list[Game]:
    """The 2,000 games every pairing plays, in a fixed order."""
    games = [
        Game(x, y, seat, DEAL_BASE + index)
        for x, y in TRAINING_CELLS
        for seat in (0, 1)
        for index in range(URGW_DEALS)
    ]
    rng = random.Random(PAIR_SEED)
    rest = [(x, y) for x in DECKS for y in DECKS if (x, y) not in TRAINING_CELLS]
    for index in range(REST_DRAWS):
        x, y = rng.choice(rest)
        games += [Game(x, y, seat, DEAL_BASE + 100_000 + index) for seat in (0, 1)]
    return games


def schedule_sha256() -> str:
    return canonical_sha256([vars(game) for game in schedule()])


SCHEDULE_SHA256 = "c3b89f412f3903ad97ce89274422ea07bd2b6218136313ed467b7c05a05f84f1"


def groups(games: Iterable[Game]) -> list[Group]:
    """Games by (decks, seat) in schedule order, at most `GROUP_DEALS` per group."""
    found: dict[tuple[str, str, Literal[0, 1]], list[int]] = {}
    for game in games:
        key = (game.deck_186, game.deck_181, game.seat_186)
        found.setdefault(key, []).append(game.deal)
    return [
        Group(
            model_deck=x,
            heuristic_deck=y,
            model_seat=seat,
            deals=tuple(deals[i : i + GROUP_DEALS]),
        )
        for (x, y, seat), deals in found.items()
        for i in range(0, len(deals), GROUP_DEALS)
    ]


# ---------------------------------------------------------------------------
# Checkpoints and pairings


class Checkpoint(BaseModel):
    trial: str
    seed: int
    job: str
    updates: int
    sha256: str
    size_bytes: int
    # The checkpoint's object key in the `manabot` bucket.
    key: str


class Pairing(BaseModel):
    name: str
    mtg186: Checkpoint
    mtg181: Checkpoint


def _mtg181(updates: int, sha256: str, size: int) -> Checkpoint:
    job = "mtg181-elemental-two-seat-18101-r2"
    return Checkpoint(
        trial="mtg181",
        seed=18101,
        job=job,
        updates=updates,
        sha256=sha256,
        size_bytes=size,
        key=f"jobs/{job}/runtime/artifacts/sha256/{sha256}",
    )


# MTG-181 seed 18101's exports, and its final raw policy at u616.
MTG181 = {
    200: _mtg181(200, "39f244d88396f1a313e46f4a1ca5f81063b9984db7a0b0a5587069ca3f761ab6", 1071367),
    400: _mtg181(400, "ee45113ff9e752ac6b14a8649559c7c5e6ad250edfbeb744b1b818471e05a8e9", 1071367),
    600: _mtg181(600, "a452cd255e1729f45af9e400756ea18b1b4d3aeed3581af137629069f6f0ccdb", 1071367),
    616: _mtg181(616, "cc78fe1d44fcc65e93fe5673d8d78b6e44b85bf7433b5624a52a0be33410915e", 1071166),
}  # fmt: skip
ORDER = ("u200", "u400", "u600", "endpoint")


def pairings(run: str = TRAINING_RUN) -> list[Pairing]:
    """Pairings whose MTG-186 checkpoint exists now, in `ORDER`."""
    storage = storage_from_uri(STORAGE)
    found = [(e, r) for e, r in stored_runs(storage) if r == run]
    records = [c for e, r in found for c in read(storage, e, r).checkpoints]
    done = status(run).done
    last = max((int(c.at.get("updates", -1)) for c in records), default=-1)
    wanted: dict[str, tuple[int, int]] = {f"u{u}": (u, u) for u in MILESTONES}
    if done:
        wanted["endpoint"] = (last, 616)
    available = []
    for name in ORDER:
        if name not in wanted:
            continue
        updates, other = wanted[name]
        record = next(
            (c for c in records if int(c.at.get("updates", -1)) == updates), None
        )
        if record is None:
            continue
        artifact = record.artifact
        prefix = f"r2://{BUCKET}/"
        if artifact.sha256 is None or artifact.size_bytes is None:
            continue
        if not artifact.uri.startswith(prefix):
            raise ValueError(f"checkpoint outside the {BUCKET} bucket: {artifact.uri}")
        mine = Checkpoint(
            trial="mtg186",
            seed=SEED,
            job=run,
            updates=updates,
            sha256=artifact.sha256,
            size_bytes=artifact.size_bytes,
            key=artifact.uri.removeprefix(prefix),
        )
        available.append(Pairing(name=name, mtg186=mine, mtg181=MTG181[other]))
    return available


# ---------------------------------------------------------------------------
# Playing one group (worker process)


class Player(BaseModel):
    model_config = ConfigDict(extra="forbid")
    checkpoint: Checkpoint
    path: Path


class PairRequest(BaseModel):
    """One group for the worker: MTG-186 is arena player A."""

    model_config = ConfigDict(extra="forbid")
    a: Player
    b: Player
    group: Group
    output: Path


def arena(
    a: Player, b: Player, matchup: AuthoredMatchup, arena_version: str
) -> tuple[EvaluationKey, PlayerRegistration, PlayerRegistration]:
    """The arena key and byte-checked registrations: MTG-186 is A, MTG-181 is B."""
    from manabot.infra.provenance import runtime_fingerprints
    from manabot.model.checkpoint import load_checkpoint

    fingerprints = runtime_fingerprints(match_config=matchup.match())
    common = dict(
        world=managym.WORLD_VERSION,
        content_suite=matchup.content_suite,
        information_boundary="acting-viewer",
        observation_abi_sha256=fingerprints["observation_abi_sha256"],
        action_abi_sha256=fingerprints["action_abi_sha256"],
        matchup_sha256=canonical_sha256(matchup.match().model_dump()),
        player_seed_derivation_id="arena-pair-deal-player-v1",
    )

    def register(name: str, player: Player, role: str) -> PlayerRegistration:
        checkpoint = player.checkpoint
        if file_sha256(player.path) != checkpoint.sha256:
            raise ValueError("checkpoint byte identity differs")
        model, space = load_checkpoint(str(player.path))
        # The checkpoint's own observation space (its saved policy-input
        # version) names its ABI; served admission rejects any other.
        own = runtime_fingerprints(
            match_config=matchup.match(), observation_space=space
        )
        return PlayerRegistration(
            **{
                **common,
                "observation_abi_sha256": own["observation_abi_sha256"],
                "action_abi_sha256": own["action_abi_sha256"],
            },
            player_id=name,
            display_name=f"{checkpoint.trial} {checkpoint.seed} u{checkpoint.updates}",
            role=role,
            compute_class_id="policy-cpu-one-thread-one-pass",
            runner_kind="checkpoint",
            player_spec={
                "kind": "checkpoint",
                "device": "cpu",
                "batch_size": 1,
                "deterministic": False,
                "allow_deck_repetition": True,
            },
            checkpoint_sha256=checkpoint.sha256,
            checkpoint_bytes=player.path.stat().st_size,
            parameter_count=sum(p.numel() for p in model.parameters()),
            training_seed=checkpoint.seed,
            artifact_id=f"{checkpoint.trial}/{checkpoint.sha256}",
        )

    reg_a = register("mtg186", a, "challenger")
    reg_b = register("mtg181", b, "anchor")
    key = EvaluationKey(
        world=managym.WORLD_VERSION,
        content_suite=matchup.content_suite,
        viewer_boundary="acting-viewer",
        arena_version=arena_version,
        rating_model_version="unrated",
        rating_prior_sha256=canonical_sha256({}),
        anchor_cohort_sha256=canonical_sha256(reg_b.model_dump()),
        evaluation_compute_envelope_id="policy-cpu-one-thread-one-pass",
    )
    return key, reg_a, reg_b


def play_pair_group(request: PairRequest) -> list[GameRow]:
    """MTG-186 as player A through the grid's deck/seat legs; rows are A's."""
    import torch

    from manabot.eval.match import play_cell

    torch.set_num_threads(1)
    group = request.group
    role_a, role_b, leg = _legs(group)
    matchup = AuthoredMatchup(pack_key=PACK, role_a=role_a, role_b=role_b)
    key, a, b = arena(request.a, request.b, matchup, "mtg186-pair-v1")
    request.output.mkdir(parents=True, exist_ok=False)
    rows, trace, replay = play_cell(
        key=key,
        player_a=a,
        player_b=b,
        deal_seeds=group.deals,
        out_dir=request.output,
        checkpoint_paths={
            "mtg186": str(request.a.path),
            "mtg181": str(request.b.path),
        },
        game_seconds=GAME_SECONDS,
        max_commands=MAX_COMMANDS,
        authored_matchup=matchup,
        matchup_mode="cross" if leg < 4 else "mirrors",
        legs=(leg,),
    )
    for name, value in (("rows", rows), ("trace", trace), ("replay", replay)):
        (request.output / f"{name}.json").write_text(json.dumps(value, indent=2))
    result = []
    for row in rows:
        seat = row["player_a_seat"]
        decks = row["seat_decks"]
        if (
            decks[seat] != group.model_deck
            or decks[1 - seat] != group.heuristic_deck
            or seat != group.model_seat
        ):
            raise ValueError("arena leg did not assign the scheduled decks and seat")
        result.append(
            GameRow(
                model_deck=group.model_deck,
                heuristic_deck=group.heuristic_deck,
                model_seat=seat,
                deal=row["deal_seed"],
                score=row["score_a"],
                valid=bool(
                    row["terminated"]
                    and not row["truncated"]
                    and row["failure"] is None
                    and row["replay_passed"]
                ),
                terminated=row["terminated"],
                truncated=row["truncated"],
                failure=None if row["failure"] is None else str(row["failure"]),
                decisions=row["decisions"],
                model_decisions=row["latency"]["mtg186"]["count"],
                seconds=row["game_seconds"],
            )
        )
    return result


# ---------------------------------------------------------------------------
# One pairing on the machine


class PairJob(BaseModel):
    pairing: Pairing
    workers: int = WORKERS


class PairResult(BaseModel):
    pairing: Pairing
    schedule_sha256: str
    seconds: float
    rows: list[GameRow]


def _invalid(group: Group, failure: str) -> list[GameRow]:
    return [
        GameRow(
            model_deck=group.model_deck,
            heuristic_deck=group.heuristic_deck,
            model_seat=group.model_seat,
            deal=deal,
            score=None,
            valid=False,
            terminated=False,
            truncated=False,
            failure=f"group failed: {failure[-200:]}",
            decisions=0,
            model_decisions=0,
            seconds=0.0,
        )
        for deal in group.deals
    ]


def _input(role: str, checkpoint: Checkpoint) -> Input:
    return Input(
        name=f"{role}.pt",
        key=checkpoint.key,
        sha256=checkpoint.sha256,
        size_bytes=checkpoint.size_bytes,
    )


def pair(spec: PairJob) -> None:
    """On the machine: play every group once, then write the result and traces."""
    if schedule_sha256() != SCHEDULE_SHA256:
        raise ValueError("pair schedule differs from its pinned digest")
    inputs = Path(os.environ["ETUDE_INPUTS_DIR"])
    out = Path(os.environ["ETUDE_OUTPUT_DIR"])
    work = Path(tempfile.mkdtemp(prefix="mtg186-pair-"))
    a = Player(checkpoint=spec.pairing.mtg186, path=inputs / "mtg186.pt")
    b = Player(checkpoint=spec.pairing.mtg181, path=inputs / "mtg181.pt")

    def run(index: int, group: Group) -> list[GameRow]:
        output = work / f"group-{index:04d}"
        request = PairRequest(a=a, b=b, group=group, output=output)
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "experiments.mtg186_eval", "worker"],
                cwd=ROOT,
                input=request.model_dump_json(),
                text=True,
                capture_output=True,
                timeout=GROUP_SECONDS,
                env=child_environment(),
            )
        except subprocess.TimeoutExpired:
            return _invalid(group, "group wall-time bound reached")
        if completed.returncode != 0:
            print(completed.stderr[-2000:], file=sys.stderr, flush=True)
            return _invalid(group, completed.stderr[-4000:])
        rows = [GameRow.model_validate(r) for r in json.loads(completed.stdout)]
        if sorted(r.deal for r in rows) != sorted(group.deals):
            raise ValueError("worker rows differ from the scheduled deals")
        print(f"group {index}: {len(rows)} games", flush=True)
        return rows

    started = time.time()
    scheduled = groups(schedule())
    with ThreadPoolExecutor(max_workers=spec.workers) as pool:
        played = list(pool.map(run, range(len(scheduled)), scheduled))
    result = PairResult(
        pairing=spec.pairing,
        schedule_sha256=SCHEDULE_SHA256,
        seconds=time.time() - started,
        rows=[row for rows in played for row in rows],
    )
    name = spec.pairing.name
    traces = out / f"{name}.traces.tar.gz"
    with tarfile.open(out / f"{name}.traces.partial", "w:gz") as archive:
        archive.add(work, arcname=name)
    (out / f"{name}.traces.partial").replace(traces)
    (out / f"{name}.json.partial").write_text(result.model_dump_json() + "\n")
    (out / f"{name}.json.partial").replace(out / f"{name}.json")
    valid = [r for r in result.rows if r.valid]
    mean = sum(r.score or 0.0 for r in valid) / max(len(valid), 1)
    print(
        f"{name}: {len(valid)}/{len(result.rows)} valid,"
        f" MTG-186 scored {mean:.3f}, {result.seconds:.0f}s",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Launching and following (operator's machine; plays nothing)


def committed_usd(launched: int) -> Decimal:
    """Committed against the cap once `launched` pairings hold their own caps."""
    return (
        sum(LAUNCHES.values(), Decimal(0))
        + TRAINING_USD
        + sum(CANCELLED_USD.values(), Decimal(0))
        + PAIRING_USD * launched
    )


def job(pairing: Pairing, *, revision: str) -> Job:
    """One pairing's job; construction launches nothing."""
    if schedule_sha256() != SCHEDULE_SHA256:
        raise ValueError("pair schedule differs from its pinned digest")
    return Job(
        name=f"mtg186-pair-{pairing.name}",
        setup=machine_setup(""),
        command=f"{TOOLCHAIN_ENV}\nuv run --locked python -m experiments.mtg186_eval pair",
        config=PairJob(pairing=pairing).model_dump_json(),
        repo=str(ROOT),
        revision=revision,
        # At cpu32's $0.96/hour, with Etude's 15-minute setup allowance, within the cap.
        max_hours=0.75,
        max_usd=PAIRING_USD,
        max_attempts=1,
        inputs=(_input("mtg186", pairing.mtg186), _input("mtg181", pairing.mtg181)),
    )


class State(BaseModel):
    """Launched runs by pairing name: the local record `follow` resumes from."""

    runs: dict[str, str] = {}


def _state() -> State:
    return State.model_validate_json(STATE.read_text()) if STATE.exists() else State()


def collect(name: str, run: str) -> PairResult | None:
    """Copy a finished run's result here; None when it wrote none."""
    scheduler = default()
    spec = PairJob.model_validate_json(scheduler.record(run).spec.config or "")
    outputs = {o.name for o in scheduler.attempts(run)[-1].outputs}
    if f"{name}.json" not in outputs:
        return None
    with tempfile.TemporaryDirectory() as folder:
        path = scheduler.fetch(run, f"{name}.json", Path(folder))
        result = PairResult.model_validate_json(Path(path).read_text())
    if result.pairing != spec.pairing or result.schedule_sha256 != SCHEDULE_SHA256:
        raise ValueError(f"{run}: result differs from its job")
    played = sorted(
        (r.model_deck, r.heuristic_deck, r.model_seat, r.deal) for r in result.rows
    )
    wanted = sorted((g.deck_186, g.deck_181, g.seat_186, g.deal) for g in schedule())
    if played != wanted:
        raise ValueError(f"{run}: rows differ from the schedule")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(result.model_dump_json(indent=1) + "\n")
    return result


def _drive(run: str, every: float) -> None:
    """Keep `etude wait` driving `run` until its status is done.

    `etude wait` can outlive a finished run (observed with u200), so the run's
    own status decides when to stop it.
    """
    driver = subprocess.Popen(["etude", "wait", run], cwd=ROOT)
    try:
        while not status(run).done and driver.poll() is None:
            time.sleep(min(every, 60))
    finally:
        if driver.poll() is None:
            driver.terminate()
            driver.wait()


def follow(revision: str, every: float = 300) -> None:
    """Launch each pairing once both checkpoints exist; drive, collect; until all four.

    Runs serialize on cpu32 (one machine at a time), so each launched run is
    driven to its end before the next is launched.
    """
    commit = subprocess.run(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    while True:
        state = _state()
        for name, run in state.runs.items():
            if (RESULTS / f"{name}.json").exists():
                continue
            if not status(run).done:
                _drive(run, every)
            if status(run).done:
                outcome = "collected" if collect(name, run) else "no result"
                print(f"{name} ({run}): {outcome}", flush=True)
        available = [p for p in pairings() if p.name not in state.runs]
        if available:
            pairing = available[0]
            if committed_usd(len(state.runs) + 1) > CAP_USD:
                raise SystemExit(f"{pairing.name} would exceed the ${CAP_USD} cap")
            runs = [Run(job=job(pairing, revision=commit))]
            cohort = f"mtg186-pair-{pairing.name}"
            priced = quote(runs, rent=RENT, cohort=cohort, max_usd=PAIRING_USD)
            print(priced.model_dump_json(), flush=True)
            [run] = queue_cohort(cohort, runs, rent=RENT, max_usd=PAIRING_USD)
            state.runs[pairing.name] = run
            STATE.parent.mkdir(parents=True, exist_ok=True)
            STATE.write_text(state.model_dump_json(indent=2) + "\n")
            print(f"{pairing.name}: launched {run}", flush=True)
            continue
        if len(state.runs) == len(ORDER):
            return
        time.sleep(every)


def main() -> None:
    if sys.argv[1:] == ["worker"]:
        request = PairRequest.model_validate_json(sys.stdin.read())
        rows = play_pair_group(request)
        print(json.dumps([r.model_dump(mode="json") for r in rows]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("schedule", "pair", "follow"))
    parser.add_argument("--revision")
    args = parser.parse_args()
    if args.command == "schedule":
        print(schedule_sha256())
        return
    if args.command == "pair":
        config = Path(os.environ["ETUDE_CONFIG"]).read_text()
        pair(PairJob.model_validate_json(config))
        return
    if not args.revision:
        parser.error("follow needs --revision")
    follow(args.revision)


if __name__ == "__main__":
    main()
