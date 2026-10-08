"""Validation gates use supervised processes; no CUDA acceptance is claimed here."""

import json
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
import torch

from manabot.remote import job_client, supervisor
from manabot.remote.jobs import Job
from manabot.remote.snapshots import snapshot_evidence
from manabot.remote.transport import job_startup
from manabot.remote.validation import ValidationReceipt, validation_command
from manabot.training.checkpoint_queue import MonitoringBudget
from manabot.training.experiments import Baseline, Experiment
from manabot.training.models import TrainingRegime
from tests.remote.job_fixtures import FileStore
from tests.remote.test_compile import SOURCE
from tests.remote.test_jobs import admitted, published, specification


@pytest.mark.parametrize(
    "mode", ["pass", "fail", "timeout", "no-cuda", "discovery-timeout"]
)
def test_gate_precedes_learner_and_retains_every_outcome(
    tmp_path: Path, mode: str
) -> None:
    # Each child owns a real process group. Stand-in commands exercise lifecycle,
    # not numerical correctness or actual CUDA device admission.
    script = """import pathlib,sys,time
from manabot.remote import validation as v
root=pathlib.Path(sys.argv[1]); mode=sys.argv[2]; source=v.Source.model_validate_json(sys.argv[3])
v.torch.cuda.is_available=lambda: (time.sleep(20) or True) if mode=='discovery-timeout' else mode != 'no-cuda'
v.torch.cuda.get_device_name=lambda index: 'fixture-only'
v.current_source=lambda root: source
code={'pass':'pass','fail':'raise SystemExit(3)','timeout':'import time; time.sleep(20)','no-cuda':'pass','discovery-timeout':'pass'}[mode]
v._checks=lambda: (('fixture-check',[sys.executable,'-c',code]),)
deadline=time.time()+(.3 if 'timeout' in mode else 10)
learner=[sys.executable,'-c',"from pathlib import Path; import sys; Path(sys.argv[1]).touch()",str(root/'learner-started')]
sys.argv=['validation','--root',str(root),'--deadline',str(deadline),'--',*learner]
v.main()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), mode, SOURCE.model_dump_json()],
        start_new_session=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    receipt = ValidationReceipt.model_validate_json(
        (tmp_path / "numerical-validation.json").read_bytes()
    )
    assert (tmp_path / "learner-started").exists() == (mode == "pass")
    assert (
        receipt.status
        == {
            "pass": "completed",
            "fail": "failed",
            "timeout": "deadline",
            "no-cuda": "failed",
            "discovery-timeout": "running",
        }[mode]
    )
    assert (
        result.returncode
        == {
            "pass": 0,
            "fail": 1,
            "timeout": -signal.SIGKILL,
            "no-cuda": 1,
            "discovery-timeout": -signal.SIGKILL,
        }[mode]
    )
    if mode in {"no-cuda", "discovery-timeout"}:
        assert receipt.device is None and not receipt.checks
    else:
        assert receipt.source == SOURCE and receipt.native_sha256
    if mode in {"pass", "fail"}:
        assert receipt.checks[0].exit_code == (0 if mode == "pass" else 3)
    snapshot = tmp_path / "snapshot"
    snapshot_evidence(tmp_path, snapshot)
    assert (snapshot / "numerical-validation.json").read_bytes() == (
        tmp_path / "numerical-validation.json"
    ).read_bytes()


def test_gate_identity_dependencies_and_absolute_deadline(tmp_path: Path) -> None:
    original = specification(FileStore(tmp_path / "control.sqlite"))
    assert "validate_numerics" not in json.loads(original.model_dump_json())
    assert (
        Job.model_validate_json(original.model_dump_json()).identity
        == original.identity
    )
    gated = original.model_copy(update={"validate_numerics": True})
    assert gated.identity != original.identity
    assert "--extra artifacts --extra dev" in job_startup(gated)
    assert "--extra artifacts --extra dev" not in job_startup(original)
    command = validation_command(gated, tmp_path, ["uv", "run", "manabot", "train"])
    deadline = float(command[command.index("--deadline") + 1])
    assert deadline <= gated.work_deadline
    assert deadline <= time.time() + 300
    assert command[0] == sys.executable


@pytest.mark.skipif(
    torch.cuda.is_available(), reason="explicit unavailable-CUDA fixture"
)
def test_supervisor_publishes_failed_gate_without_starting_learner(
    tmp_path: Path,
) -> None:
    store = FileStore(tmp_path / "control.sqlite")
    spec = specification(store).model_copy(update={"validate_numerics": True})
    resource = admitted(spec, store)
    root = tmp_path / "evidence"
    marker = root / "learner-started"
    result = supervisor.supervise(
        spec,
        store,
        root,
        resource.pod.id,
        command=[
            sys.executable,
            "-c",
            "import pathlib,sys; pathlib.Path(sys.argv[1]).touch()",
            str(marker),
        ],
        publish=published,
    )
    assert result.phase == "failed" and result.artifacts_complete
    assert result.error == "job process exited 1"
    assert not marker.exists()
    receipt = ValidationReceipt.model_validate_json(
        (root / "numerical-validation.json").read_bytes()
    )
    assert receipt.status == "failed" and not receipt.checks


def test_experiment_gate_is_bound_before_immutable_job_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileStore(tmp_path / "control.sqlite")
    original = specification(FileStore(tmp_path / "original.sqlite"))
    case = (
        Experiment(
            name="validation",
            baseline=Baseline.capture(
                "preserved",
                TrainingRegime.model_validate_json(original.plan.input_json),
            ),
        )
        .resolve()
        .cases[0]
    )
    monkeypatch.setattr(job_client, "S3JobStore", lambda prefix: store)
    spec = job_client.prepare_experiment_job(
        case,
        original.plan.spec,
        SOURCE,
        197,
        "validation-bound",
        monitoring=MonitoringBudget(seconds=60, attempt_seconds=30),
        validate_numerics=True,
    )
    assert spec.validate_numerics and spec.experiment_json is not None
    raw = store.read("spec.json")
    assert raw is not None and Job.model_validate_json(raw.data) == spec
    with pytest.raises(ValueError, match="different immutable content"):
        job_client.prepare_experiment_job(
            case,
            original.plan.spec,
            SOURCE,
            197,
            "validation-bound",
            monitoring=MonitoringBudget(seconds=60, attempt_seconds=30),
        )
