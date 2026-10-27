"""MTG-206 card smoke: a small real card evaluation on one rented L4, in Etude blocks.

Checks that a card evaluation played on a rented machine comes back as sealed
Etude blocks, with each game's card evidence a blob of its match and the run's
`CARD_ROLLUPS`, so `cards data` (the page) and `cards audit` pass on what it
returns. It plays MTG-186's card match (`experiments.mtg186_cards`: the
endpoint pairing's checkpoints, UR vs GW, legs 0–3, MTG-186 as player A) on
`DEALS` fresh deals, `MATCH_DEALS` per match, into its own experiment and run.
Nothing here is evidence about the models; the card data is too small to read.

    doppler run -p manabot -c prd -- doppler run -p etude -c prd -- \\
        uv run --extra evaluation python -m experiments.mtg206_cards launch --revision <commit>
    uv run --extra evaluation python -m experiments.mtg206_cards collect --run <etude run> --out <dir>
    uv run --extra evaluation python -m manabot.eval.cards data \\
        --experiment mtg206-cards --source <dir> --subject mtg186 --html <page>
    uv run --extra evaluation python -m manabot.eval.cards audit \\
        --experiment mtg206-cards --source <dir> --replay
"""

import argparse
from decimal import Decimal
import os
from pathlib import Path
import tarfile
import tempfile
import time

from etude.eval.match import environment as eval_environment, evaluate
from etude.eval.records import Match
from etude.launch.client import default, queue_cohort, quote, status
from etude.launch.records import Job, Run
from etude.run.declaration import declare
from pydantic import BaseModel

from experiments.elemental_grid import GAME_SECONDS, MAX_COMMANDS, PACK
from experiments.mtg186 import ROOT
from experiments.mtg186_cards import (
    LEGS,
    MATCH_SECONDS,
    Conditions,
    Machine,
    checkpoint_model,
    endpoint,
    play,
)
from experiments.mtg186_eval import GW, UR, Pairing, Player, _input
from manabot.eval.card_metrics import CARD_ROLLUPS
from manabot.experiment.job import TOOLCHAIN_ENV, machine_setup
from manabot.infra.identity import file_sha256
from manabot.infra.source import SourceBundle, bind_etude_source, bundle_at

EXPERIMENT = "mtg206-cards"
RUN = "mtg206-cards-l4"
RENT = "l4-etude"
MAX_USD = Decimal(1)

# Disjoint from MTG-186's pairing and card deals.
DEAL_BASE = 2_062_060_000
DEALS = 10
# Two matches of 20 games, one worker each.
MATCH_DEALS = 5
WORKERS = 2


def matches(pairing: Pairing) -> list[Match]:
    players = (
        checkpoint_model("mtg186", pairing.mtg186),
        checkpoint_model("mtg181", pairing.mtg181),
    )
    found = tuple(range(DEAL_BASE, DEAL_BASE + DEALS))
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


class SmokeJob(BaseModel):
    pairing: Pairing
    source_bundle: SourceBundle


def cards(spec: SmokeJob) -> None:
    """On the machine: bind the source, play the eval run, ship its folder as `cards.tar`."""
    bind_etude_source()
    inputs = Path(os.environ["ETUDE_INPUTS_DIR"])
    out = Path(os.environ["ETUDE_OUTPUT_DIR"])
    a = Player(checkpoint=spec.pairing.mtg186, path=inputs / "mtg186.pt")
    b = Player(checkpoint=spec.pairing.mtg181, path=inputs / "mtg181.pt")
    for player in (a, b):
        if file_sha256(player.path) != player.checkpoint.sha256:
            raise ValueError("input checkpoint byte identity differs")
    directory = Path(tempfile.mkdtemp(prefix="mtg206-cards-"))
    root = declare(
        experiment=EXPERIMENT,
        clocks=(),
        run=RUN,
        kind="evaluation",
        directory=directory / "eval",
    ).root
    machine = Machine(
        a=a, b=b, root=(directory / "evidence").resolve(), work=directory / "work"
    )
    # `mtg186_cards.play` reads its machine from here in Etude's workers.
    os.environ["MTG186_CARDS"] = machine.model_dump_json()
    started = time.time()
    try:
        result = evaluate(
            experiment=EXPERIMENT,
            environment=eval_environment(
                "elemental-world-ur-gw",
                files=[
                    ROOT / "experiments/mtg206_cards.py",
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
                },
            ),
            matches=matches(spec.pairing),
            play=play,
            workers=WORKERS,
            run=RUN,
            directory=directory / "eval",
            rollups=CARD_ROLLUPS,
        )
        print(result.model_dump_json(), flush=True)
    finally:
        partial = out / "cards.tar.partial"
        with tarfile.open(partial, "w") as archive:
            archive.add(root, arcname=RUN)
        partial.replace(out / "cards.tar")
    print(f"{RUN}: {time.time() - started:.0f}s", flush=True)


def job(pairing: Pairing, *, revision: str) -> Job:
    """The smoke job; construction launches nothing."""
    spec = SmokeJob(pairing=pairing, source_bundle=bundle_at(ROOT, revision))
    return Job(
        name=RUN,
        setup=machine_setup(" --extra evaluation"),
        command=(
            f"{TOOLCHAIN_ENV}\n"
            "uv run --locked --extra evaluation python -m experiments.mtg206_cards cards"
        ),
        config=spec.model_dump_json(),
        repo=str(ROOT),
        revision=spec.source_bundle.commit,
        max_hours=1.0,
        max_usd=MAX_USD,
        max_attempts=1,
        inputs=(_input("mtg186", pairing.mtg186), _input("mtg181", pairing.mtg181)),
    )


def launch(revision: str) -> str:
    runs = [Run(job=job(endpoint(), revision=revision))]
    priced = quote(runs, rent=RENT, cohort=RUN, max_usd=MAX_USD)
    print(priced.model_dump_json(), flush=True)
    [run] = queue_cohort(RUN, runs, rent=RENT, max_usd=MAX_USD)
    print(f"{RUN}: launched {run}", flush=True)
    return run


def collect(run: str, out: Path) -> Path:
    """Unpack a finished run's eval folder under `out`; it plays nothing."""
    if not status(run).done:
        raise SystemExit(f"{run} is not done")
    scheduler = default()
    with tempfile.TemporaryDirectory() as folder:
        path = scheduler.fetch(run, "cards.tar", Path(folder))
        with tarfile.open(path) as archive:
            archive.extractall(out, filter="data")
    return out / RUN


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("cards", "launch", "collect"))
    parser.add_argument("--revision")
    parser.add_argument("--run")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "cards":
        cards(
            SmokeJob.model_validate_json(Path(os.environ["ETUDE_CONFIG"]).read_text())
        )
    elif args.command == "launch":
        if not args.revision:
            parser.error("launch needs --revision")
        launch(args.revision)
    else:
        if not args.run or not args.out:
            parser.error("collect needs --run and --out")
        print(collect(args.run, args.out))


if __name__ == "__main__":
    main()
