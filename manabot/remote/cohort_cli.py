"""Cohort commands remain under manabot deploy and reuse its job lifecycle."""

from pathlib import Path
import time

import typer

from .cohort import (
    DEFAULT_COHORTS,
    Cohort,
    CohortState,
    cancel_cohort,
    load_cohort,
    prepare_cohort,
)
from .cohort_service import install_service, supervise_cohort
from .job_store import S3JobStore, cancellation_requested

app = typer.Typer(help="Run frozen Experiment jobs under an independent host service")


@app.command("start")
def start_command(
    plan: Path = typer.Option(...),
    state_dir: Path = typer.Option(...),
    doppler: bool = False,
) -> None:
    """Persist the cohort and install its service on this selected controller host."""
    cohort = Cohort.model_validate_json(plan.read_bytes())
    prepare_cohort(cohort)
    label = install_service(cohort, state_dir, doppler=doppler)
    typer.echo(f"Service {label} installed; status reports actual owner heartbeat.")


@app.command("supervise", hidden=True)
def supervise_command(
    plan: Path = typer.Option(...), state_dir: Path = typer.Option(...)
) -> None:
    """Service worker. A foreground shell is not an independently supervised owner."""
    supervise_cohort(plan, state_dir)


@app.command("status")
def status_command(
    cohort_id: str = typer.Option(...), destination: str = DEFAULT_COHORTS
) -> None:
    cohort = load_cohort(cohort_id, destination)
    store = S3JobStore(cohort.prefix)
    raw = store.read("state.json")
    if raw is None:
        typer.echo("missing: cohort has no supervisor state")
        return
    state = CohortState.model_validate_json(raw.data)
    typer.echo(state.model_dump_json(indent=2))
    typer.echo(f"Stale supervisor: {time.time() - state.heartbeat_at > 180}")
    typer.echo(
        f"Charged/reserved for admitted attempts: ${state.charged_dollars(cohort):.6f}"
    )
    typer.echo(f"Cancellation requested: {cancellation_requested(store) is not None}")


@app.command("cancel")
def cancel_command(
    cohort_id: str = typer.Option(...), destination: str = DEFAULT_COHORTS
) -> None:
    cancel_cohort(load_cohort(cohort_id, destination))
    typer.echo("Cohort cancellation recorded; service forwards it to the active job.")
