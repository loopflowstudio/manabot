"""Reviewable plans and one-command bounded RunPod deployment."""

from pathlib import Path

import typer

from .deploy import cleanup, current_source, deploy
from .plan import DeploymentPlan, HardwareMix, compile_plan
from .provider import RunPod

app = typer.Typer(help="Compile, deploy and clean up bounded RunPod training")


def _compile(regime: Path, mix: Path, seed: int) -> DeploymentPlan:
    return compile_plan(
        regime.read_text(),
        HardwareMix.model_validate_json(mix.read_text()),
        current_source(Path.cwd()),
        seed,
    )


@app.command("compile")
def compile_command(
    regime: Path = typer.Option(...),
    mix: Path = typer.Option(...),
    out: Path = typer.Option(...),
    seed: int = 197,
) -> None:
    """Resolve placement and projected cost without contacting RunPod."""
    plan = _compile(regime, mix, seed)
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
    mix: Path | None = None,
    seed: int = 197,
) -> None:
    """Prove the guardian, deploy training, retrieve evidence and confirm deletion."""
    if plan is not None:
        if regime is not None or mix is not None:
            raise typer.BadParameter("use --plan or --regime and --mix")
        value = DeploymentPlan.model_validate_json(plan.read_text())
    elif regime is not None and mix is not None:
        value = _compile(regime, mix, seed)
    else:
        raise typer.BadParameter("requires --plan or --regime and --mix")
    typer.echo(value.model_dump_json(indent=2))
    result = deploy(value, out, Path.cwd())
    typer.echo(
        f"Evidence: {out / 'evidence'}; deletion confirmed; estimated dollars: {result.estimated_dollars}"
    )
    for policy in result.policies:
        typer.echo(policy)


@app.command("status")
def status_command() -> None:
    """Read current rentals without exposing provider/account identifiers."""
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
