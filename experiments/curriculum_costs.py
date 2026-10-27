"""Prospective cost observations for MTG-175's checkpoint selection.

This observer never schedules work. It joins genuine Etude exports to a later
launcher cost estimate and retains the first observation of each export.
Selection uses only coordinates and costs, never evaluation scores. The fixed
per-arm reserve is charged in full; these estimates are not invoices.
"""

import argparse
from datetime import UTC, datetime
from decimal import Decimal
import fcntl
import os
from pathlib import Path
import time
from typing import Literal

from etude.experiment import Checkpoint, ExperimentState, observe
from etude.launch.client import status
from etude.launch.records import Status
from etude.run.storage import storage_from_uri
from pydantic import BaseModel, ConfigDict, Field

from experiments.curriculum_uniform import EXPERIMENT, load_control
from manabot.infra.files import atomic_json


class CostObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    observed_at: datetime
    checkpoint: Checkpoint
    launcher: Status
    reserve_dollars: Decimal = Field(default=Decimal("0.50"), ge=0)

    @property
    def inclusive_dollars(self) -> Decimal:
        return self.launcher.usd + self.reserve_dollars


class CostLedger(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_: Literal["mtg175-cost-observations-v1"] = Field(
        default="mtg175-cost-observations-v1", alias="schema"
    )
    # Keys are launcher/training IDs, not display labels that can be renamed.
    arms: dict[str, Literal["M", "E", "ME"]]
    observations: list[CostObservation] = Field(default_factory=list)

    def add(
        self,
        state: ExperimentState,
        statuses: dict[str, Status],
        observed_at: datetime,
    ) -> None:
        """Keep immutable first-seen costs; reject changed bytes at a coordinate."""
        known = {
            (row.checkpoint.run, row.checkpoint.clocks["updates"]): row
            for row in self.observations
        }
        for run in state.runs:
            if run.name not in self.arms:
                continue
            current = statuses[run.name]
            if current.run != run.name:
                raise ValueError("launcher identity differs from the training run")
            for cp in run.checkpoints:
                # Storage can expose an export before the controller's next
                # progress refresh. Do not attach an older cost/clock snapshot.
                if cp.clocks["updates"] > current.clocks.get("updates", -1):
                    continue
                key = (run.name, cp.clocks["updates"])
                if prior := known.get(key):
                    if prior.checkpoint.artifact != cp.artifact:
                        raise ValueError(
                            "checkpoint bytes changed at an observed coordinate"
                        )
                    continue
                row = CostObservation(
                    observed_at=observed_at, checkpoint=cp, launcher=current
                )
                self.observations.append(row)
                known[key] = row

    def selections(self) -> dict[str, Checkpoint]:
        """Last noninitial export within U's retained cost, if one was observed.

        A run whose budget boundary was missed has no selected checkpoint. It
        cannot gain one by backdating observations or inspecting its scores.
        """
        bound = load_control().estimated_inclusive_dollars
        chosen: dict[str, Checkpoint] = {}
        for row in self.observations:
            cp = row.checkpoint
            update = cp.clocks["updates"]
            if update <= 0 or row.inclusive_dollars > bound:
                continue
            if cp.run not in chosen or update > chosen[cp.run].clocks["updates"]:
                chosen[cp.run] = cp
        return chosen


def selected_checkpoints() -> dict[str, Checkpoint]:
    """The evaluation planner may read a prospective ledger supplied by its operator."""
    path = os.environ.get("MTG175_COST_LEDGER")
    if path is None:
        return {}
    return CostLedger.model_validate_json(Path(path).read_bytes()).selections()


def sample(path: Path, arms: dict[str, Literal["M", "E", "ME"]], source: str) -> None:
    """Observe exports before asking for costs, so no earlier estimate is borrowed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = (
            CostLedger.model_validate_json(path.read_bytes())
            if path.exists()
            else CostLedger(arms=arms)
        )
        if ledger.arms != arms:
            raise ValueError("cost observer cannot change the retained arm identities")
        state = observe(storage_from_uri(source), EXPERIMENT)
        statuses = {run: status(run) for run in arms}
        ledger.add(state, statuses, datetime.now(UTC))
        atomic_json(path, ledger.model_dump(mode="json", by_alias=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--source", default="r2://manabot/jobs")
    parser.add_argument("--watch", type=float)
    args = parser.parse_args()
    arms = CostLedger.model_validate_json(args.arms.read_bytes()).arms
    if args.watch is not None and args.watch < 30:
        parser.error("cost observations must be at least 30 seconds apart")
    while True:
        sample(args.ledger, arms, args.source)
        if args.watch is None:
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
