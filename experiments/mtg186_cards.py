"""MTG-186 card data: UR vs GW between the two endpoints, one 17Lands table per model.

Jack Heart's direction (2026-10-25): the endpoint pairing's checkpoints (MTG-186
seed 18601's `learner_transitions-20488960.pt` against MTG-181 seed 18101's u616
`self-play-raw.pt`) play UR vs GW only, no mirrors, 1,000 games per side, 500 on
the play and 500 on the draw, on fresh deals. Card evidence is captured during
play, so `python -m manabot.eval.cards data --experiment mtg186-cards` reads it.

The schedule is 500 deals (`DEAL_BASE`…, disjoint from the pairings' deals),
each played on legs 0–3 of `AuthoredMatchup(elemental-world-v2, elemental_ur,
elemental_gw)` with MTG-186 as arena player A: legs 0 and 3 put MTG-186 on UR
in seats 0 and 1, legs 1 and 2 put it on GW. Mirror legs 4–7 never play.

One Etude eval run (`EXPERIMENT`/`RUN`) holds the games. Each match is
`MATCH_DEALS` deals between players `mtg186` (player 0) and `mtg181` (player 1),
played by one bounded worker process (`python -m experiments.mtg186_cards
worker`) through replay-checked `play_cell(card_evidence_dir=…)`. A match
returns table `games` (one row per game, player 0's: MTG-186's deck, seat and
score) and the card tables of `trace_card_tables`. Only valid games count:
terminated, untruncated, no failure, trace replayed. A worker that fails or
reaches its wall bound returns its games as invalid rows and no card rows; it
is never replayed.

`job` freezes the Etude job at an explicit commit on the `cpu32` on-demand,
with both checkpoints as digest-checked inputs and the commit's source bundle,
which the machine binds before playing (it has no `.git`). On the machine,
`cards` plays the eval run with `WORKERS` matches at a time and writes its run
folder to `ETUDE_OUTPUT_DIR` as `cards.tar`. `launch` and `collect` run on the
operator's machine and play nothing; `collect` unpacks the run under
`experiments/data/mtg186/cards/eval/` and `report` writes the two card tables
(text and `--html` pages) and the head-to-head scores beside it.

    doppler run -p manabot -c prd -- doppler run -p etude -c prd -- \\
        uv run --extra evaluation python -m experiments.mtg186_cards launch --revision <commit>
    uv run --extra evaluation python -m experiments.mtg186_cards collect --run <etude run>
    uv run --extra evaluation python -m experiments.mtg186_cards report
"""

import argparse
from contextlib import redirect_stdout
from decimal import Decimal
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import TYPE_CHECKING

from etude.eval.match import Row, Tables, environment as eval_environment, evaluate
from etude.eval.query import statsdb
from etude.eval.records import (
    CheckpointKey,
    CheckpointRef,
    Environment,
    Inference,
    Match,
    Model,
)
from etude.launch.client import default, queue_cohort, quote, status
from etude.launch.records import Job, Run
from etude.run.declaration import declare
from etude.run.records import ArtifactRef
from pydantic import BaseModel, ConfigDict

from experiments.elemental_grid import (
    GAME_SECONDS,
    MAX_COMMANDS,
    PACK,
    GameRow,
    Score,
    wilson,
)
from experiments.mtg186 import CAP_USD, ROOT
from experiments.mtg186_eval import (
    BUCKET,
    GW,
    RENT,
    RESULTS,
    UR,
    Checkpoint,
    Pairing,
    PairResult,
    Player,
    _input,
    arena,
)
from manabot.eval.card_evaluation import trace_card_tables
from manabot.experiment.job import TOOLCHAIN_ENV, machine_setup
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.infra.source import SourceBundle, bind_etude_source, bundle_at
from manabot.sim.matchup import AuthoredMatchup
from manabot.sim.threads import child_environment

if TYPE_CHECKING:
    import duckdb

OUT = ROOT / "experiments/data/mtg186/cards"
EXPERIMENT = "mtg186-cards"
RUN = "mtg186-cards-endpoint"
COHORT = "mtg186-cards-endpoint"

DEAL_BASE = 1_869_200_000
DEALS = 500
# Small matches keep 30 workers busy to the end: 100 matches of 20 games.
MATCH_DEALS = 5
# Legs 0–3 of the UR/GW authored matchup: every cross seating, no mirrors.
LEGS = (0, 1, 2, 3)
# cpu3c 32 vCPUs: one single-threaded worker per vCPU, less two.
WORKERS = 30
# A match is 20 games; the pairings' 25-game groups took under ten minutes.
MATCH_SECONDS = 3_600

# Settled Etude charges before this job (training, failed launches, the
# cancelled grid run and the four pairings), and this job's own cap.
SPENT_USD = Decimal("3.54")
CARDS_USD = Decimal("1.25")
INFERENCE_CODE = (
    ROOT / "manabot/model",
    ROOT / "manabot/env",
    ROOT / "manabot/sim/match.py",
    ROOT / "manabot/eval/match.py",
)


def matchup() -> AuthoredMatchup:
    return AuthoredMatchup(pack_key=PACK, role_a=UR, role_b=GW)


def deals() -> tuple[int, ...]:
    return tuple(range(DEAL_BASE, DEAL_BASE + DEALS))


def schedule_sha256() -> str:
    return canonical_sha256({"deals": list(deals()), "legs": list(LEGS)})


SCHEDULE_SHA256 = "a6e7b67ac718667007c852d59a57541564699771dc2d4f43611b5d87ef477486"


def endpoint() -> Pairing:
    """The endpoint pairing's checkpoints, as its collected result recorded them."""
    return PairResult.model_validate_json(
        (RESULTS / "endpoint.json").read_text()
    ).pairing


# ---------------------------------------------------------------------------
# Players and matches


def checkpoint_model(name: str, checkpoint: Checkpoint) -> Model:
    """A model keyed by its R2 object; `name` is what `--subject` takes."""
    uri = f"r2://{BUCKET}/{checkpoint.key}"
    return Model(
        checkpoint=CheckpointRef(
            key=CheckpointKey(address=uri),
            artifact=ArtifactRef(
                uri=uri, sha256=checkpoint.sha256, size_bytes=checkpoint.size_bytes
            ),
        ),
        name=name,
        inference=Inference(
            source=INFERENCE_CODE,
            config={"updates": checkpoint.updates, "seed": checkpoint.seed},
        ),
    )


class Conditions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    deals: tuple[int, ...]


def matches(pairing: Pairing) -> list[Match]:
    players = (
        checkpoint_model("mtg186", pairing.mtg186),
        checkpoint_model("mtg181", pairing.mtg181),
    )
    found = deals()
    return [
        Match(
            players=players,
            conditions=Conditions(deals=found[i : i + MATCH_DEALS]).model_dump(
                mode="json"
            ),
            games=len(LEGS) * len(found[i : i + MATCH_DEALS]),
        )
        for i in range(0, len(found), MATCH_DEALS)
    ]


# ---------------------------------------------------------------------------
# Playing one match (Etude worker, then a bounded worker process)


class CardRequest(BaseModel):
    """One match for the worker: MTG-186 is arena player A."""

    model_config = ConfigDict(extra="forbid")
    a: Player
    b: Player
    deals: tuple[int, ...]
    # The eval run's folder, where card evidence is retained.
    root: Path
    output: Path


class CardReply(BaseModel):
    rows: list[GameRow]
    trace: dict[str, object]


def play_card_group(request: CardRequest) -> CardReply:
    """Every deal on legs 0–3, MTG-186 as player A; rows are A's."""
    import torch

    from manabot.eval.match import play_cell

    torch.set_num_threads(1)
    found = matchup()
    key, a, b = arena(request.a, request.b, found, "mtg186-cards-v1")
    request.output.mkdir(parents=True, exist_ok=False)
    rows, trace, replay = play_cell(
        key=key,
        player_a=a,
        player_b=b,
        deal_seeds=request.deals,
        out_dir=request.output,
        checkpoint_paths={"mtg186": str(request.a.path), "mtg181": str(request.b.path)},
        game_seconds=GAME_SECONDS,
        max_commands=MAX_COMMANDS,
        authored_matchup=found,
        matchup_mode="cross",
        legs=LEGS,
        card_evidence_dir=request.root,
    )
    for name, value in (("rows", rows), ("trace", trace), ("replay", replay)):
        (request.output / f"{name}.json").write_text(json.dumps(value, indent=2))
    result = []
    for row in rows:
        seat = row["player_a_seat"]
        decks = row["seat_decks"]
        result.append(
            GameRow(
                model_deck=decks[seat],
                heuristic_deck=decks[1 - seat],
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
    seatings = sorted((r.deal, r.model_deck, r.model_seat) for r in result)
    wanted = sorted(
        (deal, deck, seat)
        for deal in request.deals
        for deck in (UR, GW)
        for seat in (0, 1)
    )
    if seatings != wanted:
        raise ValueError("arena legs did not assign every UR/GW seating once")
    return CardReply(rows=result, trace=trace)


def _invalid(deal_list: tuple[int, ...], failure: str) -> list[GameRow]:
    return [
        GameRow(
            model_deck=deck,
            heuristic_deck=GW if deck == UR else UR,
            model_seat=seat,
            deal=deal,
            score=None,
            valid=False,
            terminated=False,
            truncated=False,
            failure=f"match worker failed: {failure[-200:]}",
            decisions=0,
            model_decisions=0,
            seconds=0.0,
        )
        for deal in deal_list
        for deck in (UR, GW)
        for seat in (0, 1)
    ]


def game_rows(rows: list[GameRow]) -> list[Row]:
    """One row per game, player 0's (MTG-186's); `score` is null for an invalid game."""
    return [
        {
            "seat": 0,
            "deal": row.deal,
            "deck": row.model_deck,
            "opponent_deck": row.heuristic_deck,
            "game_seat": row.model_seat,
            "score": row.score if row.valid else None,
            "valid": row.valid,
            "terminated": row.terminated,
            "truncated": row.truncated,
            "failure": row.failure,
            "decisions": row.decisions,
            "model_decisions": row.model_decisions,
            "seconds": row.seconds,
        }
        for row in rows
    ]


class Machine(BaseModel):
    """What `play` needs on the machine, handed to Etude's spawned workers by env."""

    a: Player
    b: Player
    root: Path
    work: Path


def play(match: Match) -> Tables:
    """One match in a bounded worker process; invalid rows when it fails."""
    machine = Machine.model_validate_json(os.environ["MTG186_CARDS"])
    conditions = Conditions.model_validate(match.conditions)
    output = machine.work / match.key / f"attempt-{time.time_ns()}"
    request = CardRequest(
        a=machine.a,
        b=machine.b,
        deals=conditions.deals,
        root=machine.root,
        output=output,
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "experiments.mtg186_cards", "worker"],
            cwd=ROOT,
            input=request.model_dump_json(),
            text=True,
            capture_output=True,
            timeout=MATCH_SECONDS,
            env=child_environment(),
        )
        failure = None if completed.returncode == 0 else completed.stderr[-4000:]
    except subprocess.TimeoutExpired:
        completed, failure = None, "match wall-time bound reached"
    if failure is not None:
        print(failure, file=sys.stderr, flush=True)
        return {"games": game_rows(_invalid(conditions.deals, failure))}
    assert completed is not None
    reply = CardReply.model_validate_json(completed.stdout)
    return {
        "games": game_rows(reply.rows),
        **trace_card_tables(machine.root, reply.trace, ("mtg186", "mtg181")),
    }


# ---------------------------------------------------------------------------
# The job on the machine


class CardJob(BaseModel):
    pairing: Pairing
    schedule_sha256: str
    source_bundle: SourceBundle
    workers: int = WORKERS


def cards(spec: CardJob) -> None:
    """On the machine: bind the source, play the eval run, ship its folder."""
    if schedule_sha256() != SCHEDULE_SHA256 or spec.schedule_sha256 != SCHEDULE_SHA256:
        raise ValueError("card schedule differs from its pinned digest")
    bind_etude_source()
    inputs = Path(os.environ["ETUDE_INPUTS_DIR"])
    out = Path(os.environ["ETUDE_OUTPUT_DIR"])
    a = Player(checkpoint=spec.pairing.mtg186, path=inputs / "mtg186.pt")
    b = Player(checkpoint=spec.pairing.mtg181, path=inputs / "mtg181.pt")
    for player in (a, b):
        if file_sha256(player.path) != player.checkpoint.sha256:
            raise ValueError("input checkpoint byte identity differs")
    directory = Path(tempfile.mkdtemp(prefix="mtg186-cards-"))
    root = declare(
        experiment=EXPERIMENT,
        clocks=(),
        run=RUN,
        kind="evaluation",
        directory=directory / "eval",
    ).root
    machine = Machine(a=a, b=b, root=root.resolve(), work=directory / "work")
    os.environ["MTG186_CARDS"] = machine.model_dump_json()
    started = time.time()
    try:
        result = evaluate(
            experiment=EXPERIMENT,
            environment=environment(),
            matches=matches(spec.pairing),
            play=play,
            workers=spec.workers,
            run=RUN,
            directory=directory / "eval",
        )
        print(result.model_dump_json(), flush=True)
    finally:
        # Whatever the run stored ships, so a failed match's run is still evidence.
        partial = out / "cards.tar.partial"
        with tarfile.open(partial, "w") as archive:
            archive.add(root, arcname=RUN)
        partial.replace(out / "cards.tar")
    print(f"{RUN}: {time.time() - started:.0f}s", flush=True)


def environment() -> Environment:
    return eval_environment(
        "elemental-world-ur-gw",
        files=[
            ROOT / "experiments/mtg186_cards.py",
            ROOT / "experiments/elemental_grid.py",
        ],
        config={
            "pack": PACK,
            "decks": [UR, GW],
            "legs": list(LEGS),
            "match_deals": MATCH_DEALS,
            "game_seconds": GAME_SECONDS,
            "max_commands": MAX_COMMANDS,
            "match_seconds": MATCH_SECONDS,
            "schedule_sha256": SCHEDULE_SHA256,
        },
    )


# ---------------------------------------------------------------------------
# Launching and collecting (operator's machine; plays nothing)


def job(pairing: Pairing, *, revision: str) -> Job:
    """The card job; construction launches nothing."""
    if schedule_sha256() != SCHEDULE_SHA256:
        raise ValueError("card schedule differs from its pinned digest")
    spec = CardJob(
        pairing=pairing,
        schedule_sha256=SCHEDULE_SHA256,
        source_bundle=bundle_at(ROOT, revision),
    )
    return Job(
        name=COHORT,
        setup=machine_setup(" --extra evaluation"),
        command=(
            f"{TOOLCHAIN_ENV}\n"
            "uv run --locked --extra evaluation python -m experiments.mtg186_cards cards"
        ),
        config=spec.model_dump_json(),
        repo=str(ROOT),
        revision=spec.source_bundle.commit,
        # At cpu32's $0.96/hour, with Etude's setup allowance, within the cap.
        max_hours=1.0,
        max_usd=CARDS_USD,
        max_attempts=1,
        inputs=(_input("mtg186", pairing.mtg186), _input("mtg181", pairing.mtg181)),
    )


def launch(revision: str) -> str:
    if SPENT_USD + CARDS_USD > CAP_USD:
        raise SystemExit(f"the card job would exceed the ${CAP_USD} cap")
    runs = [Run(job=job(endpoint(), revision=revision))]
    priced = quote(runs, rent=RENT, cohort=COHORT, max_usd=CARDS_USD)
    print(priced.model_dump_json(), flush=True)
    [run] = queue_cohort(COHORT, runs, rent=RENT, max_usd=CARDS_USD)
    print(f"{COHORT}: launched {run}", flush=True)
    return run


def collect(run: str) -> Path:
    """Unpack a finished run's eval folder under `OUT/eval`; it plays nothing."""
    if not status(run).done:
        raise SystemExit(f"{run} is not done")
    scheduler = default()
    spec = CardJob.model_validate_json(scheduler.record(run).spec.config or "")
    if spec.schedule_sha256 != SCHEDULE_SHA256 or spec.pairing != endpoint():
        raise ValueError(f"{run}: job differs from the card schedule")
    target = OUT / "eval"
    with tempfile.TemporaryDirectory() as folder:
        path = scheduler.fetch(run, "cards.tar", Path(folder))
        with tarfile.open(path) as archive:
            archive.extractall(target, filter="data")
    return target / RUN


# ---------------------------------------------------------------------------
# Report


class Side(BaseModel):
    """MTG-186's score with one deck, overall and by play/draw."""

    all: Score
    play: Score
    draw: Score


class CardSummary(BaseModel):
    run: str
    schedule_sha256: str
    games: int
    valid: int
    mtg186_on_ur: Side
    mtg186_on_gw: Side
    overall: Score


def _rows(db: "duckdb.DuckDBPyConnection") -> list[GameRow]:
    """The `games` table as rows; Etude stores no column that is null throughout."""
    cursor = db.execute("select * from games")
    names = [column[0] for column in cursor.description]
    found = [dict(zip(names, values, strict=True)) for values in cursor.fetchall()]
    return [
        GameRow(
            deal=row["deal"],
            model_deck=row["deck"],
            heuristic_deck=row["opponent_deck"],
            model_seat=row["game_seat"],
            score=row["score"],
            valid=row["valid"],
            terminated=row["terminated"],
            truncated=row["truncated"],
            failure=row.get("failure"),
            decisions=row["decisions"],
            model_decisions=row["model_decisions"],
            seconds=row["seconds"],
        )
        for row in found
    ]


def summary(db: "duckdb.DuckDBPyConnection") -> CardSummary:
    rows = _rows(db)
    played = sorted((r.deal, r.model_deck, r.model_seat) for r in rows)
    wanted = sorted((d, x, s) for d in deals() for x in (UR, GW) for s in (0, 1))
    if played != wanted:
        raise ValueError("games differ from the card schedule")

    def side(deck: str) -> Side:
        mine = [r for r in rows if r.model_deck == deck]
        return Side(
            all=wilson(mine),
            play=wilson([r for r in mine if r.model_seat == 0]),
            draw=wilson([r for r in mine if r.model_seat == 1]),
        )

    return CardSummary(
        run=RUN,
        schedule_sha256=SCHEDULE_SHA256,
        games=len(rows),
        valid=sum(r.valid for r in rows),
        mtg186_on_ur=side(UR),
        mtg186_on_gw=side(GW),
        overall=wilson(rows),
    )


def write_tables(source: Path = OUT / "eval") -> None:
    """Both models' card tables (text and page) and the head-to-head summary."""
    # Card queries need duckdb (the `evaluation` extra); recipe discovery imports
    # every experiment module without it, so they load only here.
    from manabot.eval.card_page import write_page
    from manabot.eval.card_query import CardSelection, card_data
    from manabot.eval.cards import print_data

    with statsdb(EXPERIMENT, source, rebuild=True) as db:
        for subject in ("mtg186", "mtg181"):
            selection = CardSelection(experiment=EXPERIMENT, subjects=(subject,))
            text = io.StringIO()
            with redirect_stdout(text):
                print_data(card_data(db, selection))
            (OUT / f"{subject}.txt").write_text(text.getvalue())
            write_page(db, selection, OUT / f"{subject}.html")
        card_summary = summary(db)
    (OUT / "summary.json").write_text(card_summary.model_dump_json(indent=2) + "\n")


def main() -> None:
    if sys.argv[1:] == ["worker"]:
        request = CardRequest.model_validate_json(sys.stdin.read())
        print(play_card_group(request).model_dump_json())
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("schedule", "cards", "launch", "collect", "report")
    )
    parser.add_argument("--revision")
    parser.add_argument("--run")
    args = parser.parse_args()
    if args.command == "schedule":
        print(schedule_sha256())
    elif args.command == "cards":
        cards(CardJob.model_validate_json(Path(os.environ["ETUDE_CONFIG"]).read_text()))
    elif args.command == "launch":
        if not args.revision:
            parser.error("launch needs --revision")
        launch(args.revision)
    elif args.command == "collect":
        if not args.run:
            parser.error("collect needs --run")
        print(collect(args.run))
    else:
        write_tables()


if __name__ == "__main__":
    main()
