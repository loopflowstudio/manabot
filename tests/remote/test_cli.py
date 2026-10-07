"""Public deployment names and direct submission use the shared job lifecycle."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from manabot.cli import app
from manabot.remote import cli
from manabot.remote.plan import DeploymentPlan, JobSpec, compile_plan
from tests.remote.test_compile import ROOT, SOURCE

runner = CliRunner()


def test_remote_namespace_is_rejected() -> None:
    result = runner.invoke(app, ["remote", "--help"])
    assert result.exit_code == 2
    assert "No such command" in result.output
    help_result = runner.invoke(app, ["--help"])
    assert "deploy" in help_result.output
    assert "remote" not in help_result.output


@pytest.mark.parametrize(
    "command",
    [
        "",
        "submit",
        "status",
        "logs",
        "attach",
        "fetch",
        "cancel",
        "reconcile",
        "report",
    ],
)
def test_deploy_help(command: str) -> None:
    result = runner.invoke(app, ["deploy", *([command] if command else []), "--help"])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("explicit", [False, True])
def test_submit_uses_same_plan_and_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    plan = compile_plan(
        (ROOT / "ops/examples/step-target.json").read_text(),
        JobSpec.model_validate_json((ROOT / "ops/jobs/runpod-small.json").read_text()),
        SOURCE,
        197,
    )
    path = tmp_path / "plan.json"
    path.write_text(plan.model_dump_json())
    called: list[tuple[DeploymentPlan, str]] = []

    def submit(
        value: DeploymentPlan,
        job_id: str,
        monitoring: Path | None,
        checkpoint_seconds: float,
        destination: str | None,
    ) -> None:
        assert monitoring is None and checkpoint_seconds == 60 and destination is None
        called.append((value, job_id))

    monkeypatch.setattr(cli, "_submit", submit)
    result = runner.invoke(
        app,
        [
            "deploy",
            *(["submit"] if explicit else []),
            "--plan",
            str(path),
            "--job-id",
            "same-job",
        ],
    )
    assert result.exit_code == 0, result.output
    assert called == [(plan, "same-job")]


@pytest.mark.parametrize(
    "args",
    [
        ["--plan", "missing.json"],
        ["--plan", "missing.json", "--regime", "also.json", "--job-id", "one"],
        ["--job-id", "one", "status"],
    ],
)
def test_invalid_submission_never_rents(
    monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    submit = Mock()
    monkeypatch.setattr(cli, "_submit", submit)
    result = runner.invoke(app, ["deploy", *args])
    assert result.exit_code == 2, result.output
    submit.assert_not_called()
