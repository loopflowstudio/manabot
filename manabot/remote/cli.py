"""Reviewable plans and one-command bounded RunPod deployment."""

from pathlib import Path
import time

import typer

from manabot.training.checkpoint_queue import MonitoringBudget

from .cohort_cli import app as cohort_app
from .deploy import cleanup, current_source, deploy
from .job_client import (
    cancel_job,
    fetch_job,
    fetch_job_file,
    job_status,
    load_job,
    prepare_job,
    submit_job,
)
from .jobs import DEFAULT_JOBS
from .plan import DeploymentPlan, JobSpec, compile_plan
from .provider import RunPod

app = typer.Typer(help="Submit bounded training and reconnect to durable jobs")
app.add_typer(cohort_app, name="cohort")


def _compile(regime: Path, spec: Path, seed: int) -> DeploymentPlan:
    return compile_plan(
        regime.read_text(),
        JobSpec.model_validate_json(spec.read_text()),
        current_source(Path.cwd()),
        seed,
    )


def _resolve_plan(
    plan: Path | None, regime: Path | None, spec: Path | None, seed: int
) -> DeploymentPlan:
    if plan is not None:
        if regime is not None or spec is not None:
            raise typer.BadParameter("use --plan or --regime and --spec")
        return DeploymentPlan.model_validate_json(plan.read_text())
    if regime is not None and spec is not None:
        return _compile(regime, spec, seed)
    raise typer.BadParameter("requires --plan or --regime and --spec")


@app.callback(invoke_without_command=True)
def deploy_command(
    ctx: typer.Context,
    plan: Path | None = None,
    regime: Path | None = None,
    spec: Path | None = None,
    seed: int = 197,
    job_id: str | None = None,
    monitoring: Path | None = None,
    checkpoint_seconds: float = 60,
    destination: str | None = None,
    validate_numerics: bool = False,
) -> None:
    """Submit directly, or select a job lifecycle command below."""
    submitting = any(
        value is not None for value in (plan, regime, spec, job_id, monitoring)
    )
    if ctx.invoked_subcommand is not None:
        if (
            submitting
            or seed != 197
            or checkpoint_seconds != 60
            or destination is not None
            or validate_numerics
        ):
            raise typer.BadParameter("put options after the selected lifecycle command")
        return
    if not submitting:
        typer.echo(ctx.get_help())
        return
    if job_id is None:
        raise typer.BadParameter("submission requires --job-id for safe retries")
    _submit(
        _resolve_plan(plan, regime, spec, seed),
        job_id,
        monitoring,
        checkpoint_seconds,
        destination,
        validate_numerics,
    )


@app.command("compile")
def compile_command(
    regime: Path = typer.Option(...),
    out: Path = typer.Option(...),
    spec: Path | None = None,
    seed: int = 197,
) -> None:
    """Resolve placement and projected cost without contacting RunPod."""
    plan = _resolve_plan(None, regime, spec, seed)
    if out.exists():
        raise typer.BadParameter("plan output already exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(plan.model_dump_json(indent=2) + "\n")
    typer.echo(
        f"Plan: {out}; projected ceiling ${plan.projected_dollars:.3f}, including reserves"
    )


@app.command("run")
def run_command(
    out: Path = typer.Option(...),
    plan: Path | None = None,
    regime: Path | None = None,
    spec: Path | None = None,
    seed: int = 197,
) -> None:
    """Prove the guardian, deploy training, retrieve evidence and confirm deletion."""
    value = _resolve_plan(plan, regime, spec, seed)
    typer.echo(value.model_dump_json(indent=2))
    result = deploy(
        value,
        out,
        Path.cwd(),
        destination=value.spec.access.destination,
    )
    typer.echo(
        f"Evidence: {out / 'evidence'}; deletion confirmed; estimated dollars: {result.estimated_dollars}"
    )
    for policy in result.policies:
        typer.echo(policy)


@app.command("status")
def status_command(job_id: str | None = None, destination: str = DEFAULT_JOBS) -> None:
    """Read current rentals without exposing provider/account identifiers."""
    if job_id is not None:
        status = job_status(load_job(job_id, destination))
        typer.echo(status.model_dump_json(indent=2))
        typer.echo(f"Stale/unavailable heartbeat: {status.stale}")
        return
    pods = RunPod().list()
    typer.echo(
        f"Current rentals: {len(pods)} pods; total compute ${sum(p.rate for p in pods):.3f}/hour (storage additional)"
    )
    typer.echo(
        f"Owned manabot rentals: {sum(p.name.startswith('manabot-') for p in pods)}"
    )


@app.command("cleanup")
def cleanup_command(deployment: Path = typer.Option(...)) -> None:
    """Retry deletion for exactly the attempts recorded in a private receipt."""
    cleanup(deployment)
    typer.echo("Recorded deployment pods confirmed absent.")


@app.command("submit")
def submit_command(
    plan: Path = typer.Option(...),
    job_id: str = typer.Option(
        ..., help="Stable ID; retry this exact ID after interruption"
    ),
    monitoring: Path | None = None,
    checkpoint_seconds: float = 60,
    destination: str | None = None,
    validate_numerics: bool = False,
) -> None:
    """Submit a job; returns only after remote acceptance (or an explicit uncertainty)."""
    _submit(
        DeploymentPlan.model_validate_json(plan.read_text()),
        job_id,
        monitoring,
        checkpoint_seconds,
        destination,
        validate_numerics,
    )


def _submit(
    plan: DeploymentPlan,
    job_id: str,
    monitoring: Path | None,
    checkpoint_seconds: float,
    destination: str | None,
    validate_numerics: bool = False,
) -> None:
    if destination is None:
        destination = plan.spec.access.destination
    spec = prepare_job(
        plan,
        job_id,
        destination=destination,
        checkpoint_seconds=checkpoint_seconds,
        validate_numerics=validate_numerics,
        monitoring=MonitoringBudget.model_validate_json(monitoring.read_text())
        if monitoring
        else None,
    )
    typer.echo(
        f"Job {spec.job_id}; reconnect: uv run manabot deploy status --job-id {spec.job_id}"
    )
    status = submit_job(spec)
    typer.echo(status.model_dump_json(indent=2))


@app.command("fetch")
def fetch_command(
    job_id: str = typer.Option(...),
    out: Path = typer.Option(...),
    destination: str = DEFAULT_JOBS,
) -> None:
    """Fetch the latest committed artifact generation, including after deletion."""
    typer.echo(str(fetch_job(load_job(job_id, destination), out)))


@app.command("cancel")
def cancel_command(
    job_id: str = typer.Option(...), destination: str = DEFAULT_JOBS
) -> None:
    """Request cancellation explicitly; the supervisor acknowledges before final upload."""
    cancel_job(load_job(job_id, destination))
    typer.echo(
        "Cancellation requested; status reports acknowledgement and confirmed deletion separately."
    )


@app.command("logs")
def logs_command(
    job_id: str = typer.Option(...),
    follow: bool = False,
    destination: str = DEFAULT_JOBS,
) -> None:
    """Read published log prefixes; Ctrl-C ends observation only."""
    from tempfile import TemporaryDirectory

    spec = load_job(job_id, destination)
    generation = -1
    offset = 0
    with TemporaryDirectory(prefix="manabot-logs-") as directory:
        while True:
            status = job_status(spec)
            record = status.record
            if (
                record is not None
                and record.manifest is not None
                and record.generation != generation
            ):
                log = fetch_job_file(spec, record, "training.log", Path(directory))
                if log is not None:
                    data = log.read_bytes()
                    typer.echo(data[offset:].decode(errors="replace"), nl=False)
                    offset = len(data)
                generation = record.generation
            if not follow or (record is not None and record.terminal):
                break
            time.sleep(5)


@app.command("attach")
def attach_command(
    job_id: str = typer.Option(...), destination: str = DEFAULT_JOBS
) -> None:
    """Follow the same persisted job; detaching never restarts or cancels training."""
    logs_command(job_id, True, destination)


@app.command("setup-worker")
def setup_worker_command(destination: str = DEFAULT_JOBS) -> None:
    """Create the S3-only worker role trusted by the current AWS identity."""
    from .job_store import configure_worker_role

    configure_worker_role(destination)
    typer.echo(
        "Private job-evidence worker role configured for the current AWS principal."
    )


@app.command("reconcile")
def reconcile_command(
    job_id: str = typer.Option(...),
    delete: bool = False,
    destination: str = DEFAULT_JOBS,
) -> None:
    """Reconcile uncertain creation/deletion; --delete may lose unpublished evidence."""
    from .job_client import reconcile_job

    typer.echo(
        reconcile_job(load_job(job_id, destination), delete=delete).model_dump_json(
            indent=2
        )
    )


@app.command("report")
def report_command(
    evidence: Path = typer.Option(...), out: Path = typer.Option(...)
) -> None:
    """Regenerate a notebook and HTML from a fetched generation, without a rental."""
    from manabot.training.comparison_notebook import write_comparison_notebook
    from manabot.training.experiment_report import (
        diagnostic_figures,
        load_evidence,
        strength_figures,
        write_dashboard,
    )

    from .bundle import Bundle

    Bundle.model_validate_json((evidence / "bundle.json").read_text()).verify(evidence)
    retained = load_evidence(evidence)
    out.mkdir(parents=True, exist_ok=True)
    write_comparison_notebook(evidence, out / "report.ipynb")
    path = write_dashboard(
        retained,
        out / "comparison.html",
        question="Remote job progress and checkpoint monitoring",
        docs="https://github.com/loopflowstudio/etude/blob/main/docs/experiment-metrics.md",
        sections=[
            (
                "Checkpoint monitoring",
                "comparisons",
                strength_figures(retained, "training_seconds"),
            ),
            ("Learning diagnostics", "sampling", diagnostic_figures(retained)),
        ],
        notes="Disconnected execution proof; monitoring does not establish scientific strength.",
    )
    typer.echo(str(path))
