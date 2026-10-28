"""MTG-206 card smoke: a small real card evaluation on one rented L4, in Etude blocks.

Checks that `evaluate_cards` played on a rented machine comes back as sealed
Etude blocks, with each game's card evidence a blob of its match and the run's
`CARD_ROLLUPS`, so `cards data` (the page) and `cards audit --replay` pass on
what it returns. It plays the `cards verify` schedule (two scripted players,
UG vs BG, `GAMES` games on fresh deals) into its own experiment and run, and
needs no checkpoint inputs. Nothing here is evidence about players.

    doppler run -p manabot -c prd -- doppler run -p etude -c prd -- \\
        uv run --extra evaluation python -m experiments.mtg206_cards launch --revision <commit>
    uv run --extra evaluation python -m experiments.mtg206_cards collect --run <etude run> --out <dir>
    uv run --extra evaluation python -m manabot.eval.cards data \\
        --experiment mtg206-cards --source <dir> --subject verify-one --html <page>
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

from etude.launch.client import default, queue_cohort, quote, status
from etude.launch.records import Job, Run
from pydantic import BaseModel

from manabot.eval.card_evaluation import (
    CardEvaluationSpec,
    evaluate_cards,
    verification_spec,
)
from manabot.experiment.job import TOOLCHAIN_ENV, machine_setup
from manabot.infra.source import SourceBundle, bind_etude_source, bundle_at

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "mtg206-cards"
RUN = "mtg206-cards-l4-verify"
RENT = "l4-etude"
MAX_USD = Decimal(1)

# Eight games per deal; one worker per vCPU of the L4 on-demand.
GAMES = 96
WORKERS = 6


def spec() -> CardEvaluationSpec:
    return verification_spec(GAMES).model_copy(
        update={"experiment": EXPERIMENT, "run": RUN, "workers": WORKERS}
    )


class SmokeJob(BaseModel):
    source_bundle: SourceBundle


def cards() -> None:
    """On the machine: bind the source, play the eval run, ship its runs folder as `cards.tar`."""
    bind_etude_source()
    out = Path(os.environ["ETUDE_OUTPUT_DIR"])
    directory = Path(tempfile.mkdtemp(prefix="mtg206-cards-")) / "runs"
    started = time.time()
    try:
        print(evaluate_cards(spec(), directory).model_dump_json(), flush=True)
    finally:
        partial = out / "cards.tar.partial"
        with tarfile.open(partial, "w") as archive:
            archive.add(directory, arcname="runs")
        partial.replace(out / "cards.tar")
    print(f"{RUN}: {time.time() - started:.0f}s", flush=True)


def job(revision: str) -> Job:
    """The smoke job; construction launches nothing."""
    bundle = bundle_at(ROOT, revision)
    return Job(
        name=RUN,
        setup=machine_setup(" --extra evaluation"),
        command=(
            f"{TOOLCHAIN_ENV}\n"
            "uv run --locked --extra evaluation python -m experiments.mtg206_cards cards"
        ),
        config=SmokeJob(source_bundle=bundle).model_dump_json(),
        repo=str(ROOT),
        revision=bundle.commit,
        max_hours=1.0,
        max_usd=MAX_USD,
        max_attempts=1,
    )


def launch(revision: str) -> str:
    runs = [Run(job=job(revision))]
    priced = quote(runs, rent=RENT, cohort=RUN, max_usd=MAX_USD)
    print(priced.model_dump_json(), flush=True)
    [run] = queue_cohort(RUN, runs, rent=RENT, max_usd=MAX_USD)
    print(f"{RUN}: launched {run}", flush=True)
    return run


def collect(run: str, out: Path) -> Path:
    """Unpack a finished run's runs folder as `out`; it plays nothing."""
    if not status(run).done:
        raise SystemExit(f"{run} is not done")
    with tempfile.TemporaryDirectory() as folder:
        path = default().fetch(run, "cards.tar", Path(folder))
        with tarfile.open(path) as archive:
            archive.extractall(out, filter="data")
    return out / "runs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("cards", "launch", "collect"))
    parser.add_argument("--revision")
    parser.add_argument("--run")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "cards":
        cards()
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
