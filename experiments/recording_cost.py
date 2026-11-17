"""Exploratory CUDA cost of capture options on matched Elemental training.

One job runs fresh training processes sequentially on one host, rotating seven
arms through five repeats. Each arm uses the same seed and four real updates;
measurements include the first update and all capture writes inside collection.
No artificial load, concurrent training or playing-strength claim is involved.
The driver retains receipts and streams cell progress to its launcher log.
"""

from __future__ import annotations

from decimal import Decimal
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from etude.launch.records import Job, Run
from pydantic import BaseModel
import torch

from experiments.elemental_variants import elemental_recipe, self_play_stage
from manabot.backends import resolve
from manabot.experiment.job import TOOLCHAIN_ENV, machine_setup
from manabot.infra.run_settings import RunSettings
from manabot.infra.source import SourceBundle, bind_etude_source, bundle_at
from manabot.training.execution import execute_regime
from manabot.verify.store import VerifyStore

ROOT = Path(__file__).resolve().parents[1]
SEED = 80161
UPDATES = 4
REPEATS = 5
EXPERIMENT = "recording_cost"
ARMS: dict[str, RunSettings] = {
    "off": RunSettings(diagnostics="off"),
    "statistics": RunSettings(diagnostics="off", statistics=True),
    "record-01": RunSettings(diagnostics="off", record_games=0.01),
    "record-10": RunSettings(diagnostics="off", record_games=0.1),
    "record-all": RunSettings(diagnostics="off", record_games=1),
    "journals": RunSettings(diagnostics="detailed"),
    "marker": RunSettings(diagnostics="standard"),
}


class Probe(BaseModel):
    source_bundle: SourceBundle


def _cell_environment(repeat: int, arm: str, destination: Path) -> dict[str, str]:
    """Keep each cell's tracking identity distinct within the one rental."""
    environment = dict(os.environ)
    name = f"{environment.get('ETUDE_RUN', 'recording-cost')}-r{repeat}-{arm}"
    environment.update(
        ETUDE_RUN=name,
        ETUDE_RUN_ADDRESS=f"{EXPERIMENT}/{name}",
        ETUDE_OUTPUT_DIR=str(destination),
    )
    return environment


def cell(arm: str, output: Path) -> None:
    """Run one ordinary training cell; refuse a CPU fallback or failed stage."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    regime = elemental_recipe("recording-cost-elemental", UPDATES)
    # Prospective input v11 uses the current shared representation. Preserve the
    # historical recipe bytes; all arms use this identical fresh-world regime.
    regime.observation.policy_input_version = 11
    stage = self_play_stage(regime)
    stage.execution.device = "cuda"
    regime.wall_seconds = stage.execution.wall_seconds = 1200
    output.mkdir(parents=True)
    (output / "regime.json").write_text(regime.model_dump_json(indent=2))
    with VerifyStore(output / "runs.sqlite") as store:
        result = execute_regime(
            regime,
            SEED,
            output / "training",
            store,
            backend=resolve("torch"),
            settings=ARMS[arm],
            experiment=EXPERIMENT,
        )
    (output / "receipt.json").write_text(result.model_dump_json(indent=2))
    if (
        result.status != "completed"
        or torch.device(result.stages[0].actual_device).type != "cuda"
    ):
        raise RuntimeError(f"cell {arm}: {result.status}: {result.error}")


def run(probe: Probe) -> None:
    bind_etude_source()
    output = Path(os.environ["ETUDE_OUTPUT_DIR"])
    names = list(ARMS)
    rows: list[dict[str, object]] = []
    for repeat in range(REPEATS):
        order = names[repeat:] + names[:repeat]
        for arm in order:
            destination = output / f"repeat-{repeat}" / arm
            print(f"START {repeat} {arm}", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.recording_cost",
                    arm,
                    str(destination),
                ],
                check=True,
                timeout=1500,
                env=_cell_environment(repeat, arm, destination),
            )
            receipt = json.loads((destination / "receipt.json").read_text())
            stage = receipt["stages"][0]
            if stage["updates"] != UPDATES:
                raise RuntimeError(f"cell {arm}: expected {UPDATES} updates")
            row = {
                "repeat": repeat,
                "arm": arm,
                "decisions": stage["environment_decisions"],
                "updates": stage["updates"],
                "collection_dps": stage["environment_decisions"]
                / stage["collection_seconds"],
                "seconds_per_update": (
                    stage["collection_seconds"] + stage["learning_seconds"]
                )
                / stage["updates"],
                "host": receipt["host"],
                "game_errors": receipt.get("game_errors"),
            }
            rows.append(row)
            (output / "measurements.json").write_text(json.dumps(rows, indent=2))
            print(json.dumps(row), flush=True)
    summaries = {}
    for arm in names:
        ratios = [
            float(row["collection_dps"])
            / float(
                next(
                    baseline["collection_dps"]
                    for baseline in rows
                    if baseline["repeat"] == row["repeat"] and baseline["arm"] == "off"
                )
            )
            for row in rows
            if row["arm"] == arm
        ]
        summaries[arm] = {
            "rate_ratio_median": statistics.median(ratios),
            "rate_ratio_range": [min(ratios), max(ratios)],
        }
    (output / "summary.json").write_text(json.dumps(summaries, indent=2))


def training_jobs(revision: str) -> list[Run]:
    job = Job(
        name="recording-cost",
        setup=machine_setup(" --extra artifacts --extra reports"),
        command=f"{TOOLCHAIN_ENV}\nuv run --locked python -m etude.call --function experiments.recording_cost:run",
        config=Probe(source_bundle=bundle_at(ROOT, revision)).model_dump_json(),
        repo=str(ROOT),
        revision=revision,
        experiment=EXPERIMENT,
        max_hours=3,
        max_usd=Decimal("5"),
        max_attempts=1,
    )
    return job.runs(SEED)


if __name__ == "__main__":
    cell(sys.argv[1], Path(sys.argv[2]))
