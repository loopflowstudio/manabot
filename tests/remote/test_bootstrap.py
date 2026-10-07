"""Bootstrap keeps dependency installation off the evidence volume."""

from pathlib import Path
import subprocess

from manabot.remote.plan import JobSpec, compile_plan
from manabot.remote.transport import REPO_DIR, UV_CACHE_DIR, bootstrap
from tests.remote.test_compile import ROOT, SOURCE


def test_local_install_and_timing_script() -> None:
    plan = compile_plan(
        (ROOT / "ops/examples/step-target.json").read_text(),
        JobSpec.model_validate_json((ROOT / "ops/jobs/runpod-small.json").read_text()),
        SOURCE,
        197,
    )
    script = bootstrap(plan)
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert Path(REPO_DIR).parent == Path(UV_CACHE_DIR).parent
    assert not REPO_DIR.startswith("/workspace")
    assert f"cd {REPO_DIR}" in script
    assert f"export UV_CACHE_DIR={UV_CACHE_DIR}" in script
    assert "uv sync --locked --python 3.12" in script
    assert "/workspace/evidence/bootstrap-timing.json" in script
    assert SOURCE.commit in script and SOURCE.lock_sha256 in script
