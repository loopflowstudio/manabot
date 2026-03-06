"""
test_bootstrap.py
Tests for user-data rendering used by sandbox/job bootstrapping.
"""

from __future__ import annotations

from ops.bootstrap import job_user_data, sandbox_user_data
from tests.ops.conftest import make_runtime_spec


def test_sandbox_user_data_includes_repo_ref():
    rendered = sandbox_user_data(
        make_runtime_spec(),
        repo_url="https://example.com/repo.git",
        repo_ref="feature/demo",
    )

    assert "REPO_REF=feature/demo" in rendered
    assert 'PYTHON_VERSION=3.13' in rendered
    assert 'checkout -B "$REPO_REF" "origin/$REPO_REF"' in rendered
    assert '--build-arg PYTHON_VERSION="$PYTHON_VERSION"' in rendered
    assert 'torch.cuda.is_available()' in rendered


def test_job_user_data_pins_repo_ref_to_main():
    rendered = job_user_data(
        make_runtime_spec(),
        config_name="simple",
        job_id="job-1",
        wandb_run_id="run-1",
        region="us-west-2",
        log_group="/manabot/job/tester",
    )

    assert "REPO_REF='main'" in rendered
