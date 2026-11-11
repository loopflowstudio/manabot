"""Frozen MTG-281 comparison contract shared by preparation and offline analysis.

These are prospective engineering thresholds, not confidence bounds. Changing
them after observing either arm requires a new explicitly exploratory analysis.
This module imports neither the native engine nor a training executor.
"""

from pathlib import Path
from typing import Literal

from etude.launch.records import OnDemand, Run
from pydantic import BaseModel, ConfigDict

from manabot.data import DataFile, file_entry

SOURCES = {
    "before": "b7c102f2c4266dab4be91b8b54901861745b9a3c",
    "after": "1659b0e290f55daa632ec8f869adff4a3059d79f",
}
RECIPE = "experiments/regimes/tla-frontier-c5fba4-w64.json"
RECIPE_SHA256 = "4fb74dda37a729e5c5ee5ee4065f1bcb604888a755889510233b938a1e90f526"
SEED = 21501
UPDATES = 500
# Provider deletion is requested five minutes before the 12-hour cost envelope.
# Its guardian and controller are best effort; billing must still be reconciled.
RENTAL_SECONDS = 42900
STARTUP_SECONDS = 900
DELIVERY_SECONDS = 600
TRAINING_SECONDS = 39600
FINALIZATION_SECONDS = 600


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Analysis(Record):
    first: Literal[51] = 51
    last: Literal[500] = 500
    early_last: Literal[100] = 100
    late_first: Literal[451] = 451
    # Practical flatness: fitted rise over the full window AND late/early p90
    # rise are <=10% of the early mean for collection (raw and per 1000 rows).
    time_growth_fraction: Literal[0.10] = 0.10
    # RSS: fitted rise and late-minus-early median are <=1 decimal GB.
    rss_growth_gb: Literal[1.0] = 1.0
    # All three primaries must first reproduce growth. Partial flattening then
    # requires >=50% reduction in every positive growth score.
    partial_reduction_fraction: Literal[0.50] = 0.50
    # Descriptive comparisons also require work/host comparability. Differing
    # host CPU models, quotas, device/runtime, >10% learner-row or game-length
    # differences, or >1 percentage point error-rate difference withhold verdict.
    work_difference_fraction: Literal[0.10] = 0.10
    error_rate_difference: Literal[0.01] = 0.01


class Preparation(Record):
    analysis: Analysis
    supply: OnDemand
    files: dict[str, DataFile]
    source_files_equal: list[str]
    status: Literal["prepared_not_launched"] = "prepared_not_launched"
    seed: Literal[21501] = SEED
    updates: Literal[500] = UPDATES
    total_usd: Literal["15.00"] = "15.00"
    rental_usd_per_arm: Literal["7.20"] = "7.20"
    ancillary_usd: Literal["0.60"] = "0.60"


def prepared_runs(directory: Path) -> tuple[Preparation, dict[str, Run]]:
    """Read digest-checked prepared inputs; never submit or acquire a resource."""
    prepared = Preparation.model_validate_json(
        (directory / "preparation.json").read_bytes()
    )
    if not {f"{arm}.json" for arm in SOURCES} <= prepared.files.keys():
        raise ValueError("prepared run files are not digest-bound")
    for name, expected in prepared.files.items():
        path = Path(name)
        if path.name != name or file_entry(directory / name) != expected:
            raise ValueError(f"prepared input differs: {name}")
    runs = {
        arm: Run.model_validate_json((directory / f"{arm}.json").read_bytes())
        for arm in SOURCES
    }
    for arm, run in runs.items():
        if run.job.revision != SOURCES[arm] or run.seed != SEED:
            raise ValueError(f"prepared run identity differs: {arm}")
    return prepared, runs
