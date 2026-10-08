"""Cohort commands remain under manabot deploy and reuse its job lifecycle."""

from pathlib import Path
import shutil
import time

import typer

from .cohort import (
    DEFAULT_COHORTS,
    Cohort,
    CohortInsertion,
    CohortState,
    cancel_cohort,
    insert_cohort,
    load_cohort,
    prepare_cohort,
)
from .cohort_projection import (
    ProjectionConfig,
    follow_projection,
    project_before,
    project_once,
)
from .cohort_service import install_service, supervise_cohort
from .job_store import S3JobStore, cancellation_requested

app = typer.Typer(help="Run frozen Experiment jobs under an independent host service")


@app.command("insert")
def insert_command(
    cohort_id: str = typer.Option(...),
    plan: Path = typer.Option(..., help="Immutable insertion after an original job"),
    destination: str = DEFAULT_COHORTS,
) -> None:
    """Add bounded continuation allocations without changing original science jobs."""
    cohort = load_cohort(cohort_id, destination)
    insertion = CohortInsertion.model_validate_json(plan.read_bytes())
    state = insert_cohort(cohort, insertion)
    typer.echo(
        f"Verified durable order: {', '.join(e.job_id for e in state.entries(cohort))}"
    )


@app.command("start")
def start_command(
    plan: Path = typer.Option(...),
    state_dir: Path = typer.Option(...),
    doppler: bool = False,
    reports: bool = False,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    source_root: list[Path] = typer.Option(
        [], help="Retained exact worker source checkout"
    ),
    replace_service: bool = False,
) -> None:
    """Persist the cohort and install its service on this selected controller host."""
    cohort = Cohort.model_validate_json(plan.read_bytes())
    projection = ProjectionConfig(project=wandb_project, entity=wandb_entity)
    if wandb_project is not None:
        reports = True
    prepare_cohort(cohort)
    label = install_service(
        cohort,
        state_dir,
        doppler=doppler,
        source_roots=tuple(source_root),
        replace_service=replace_service,
    )
    typer.echo(f"Service {label} installed; status reports actual owner heartbeat.")
    if reports:
        companion = install_service(
            cohort,
            state_dir / "projection",
            doppler=doppler,
            projection=projection,
            source_roots=tuple(source_root),
            replace_service=replace_service,
        )
        typer.echo(f"Independent report service {companion} installed.")


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


@app.command("project")
def project_command(
    plan: Path = typer.Option(...),
    state_dir: Path = typer.Option(...),
    config: Path = typer.Option(...),
    follow: bool = False,
    deadline: float | None = None,
) -> None:
    """Refresh saved Dashboard exports; --follow is the bounded companion service."""
    cohort = Cohort.model_validate_json(plan.read_bytes())
    projection = ProjectionConfig.model_validate_json(config.read_bytes())
    if not follow:
        try:
            if deadline is None:
                project_once(cohort, state_dir, projection)
            else:
                project_before(cohort, state_dir, projection, deadline)
        except Exception as error:
            typer.echo(
                f"Projection unavailable ({type(error).__name__}); retained local evidence is unchanged."
            )
            raise typer.Exit(1) from None
        return
    uv = shutil.which("uv")
    if uv is None:
        raise ValueError("uv unavailable")
    command = [
        uv,
        "run",
        "--no-sync",
        "manabot",
        "deploy",
        "cohort",
        "project",
        "--plan",
        str(plan.resolve()),
        "--state-dir",
        str(state_dir.resolve()),
        "--config",
        str(config.resolve()),
    ]
    follow_projection(cohort, state_dir, projection, command)
