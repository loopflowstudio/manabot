"""Read-only admission of one failed history calibration, never campaign resume.

The caller pins every retained byte before launch. Only the original off-arm
reload failure is supported; continuation attempts cannot themselves be retried.
"""

from contextlib import closing
import json
from pathlib import Path
import sqlite3
from typing import Literal

import psutil
from pydantic import BaseModel, ConfigDict, Field

from experiments.runners.history_input import (
    CALIBRATION_SECONDS,
    CALIBRATION_SEED,
    ORDER,
    InputBinding,
    recipes,
    validate_bindings,
    validate_run,
)
from manabot.arena.models import canonical_sha256, file_sha256
from manabot.training.models import TrainingRun


class Predecessor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seconds: float = Field(gt=0, lt=CALIBRATION_SECONDS)
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")


class _Supervisor(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    pid: int = Field(gt=0)
    study: Literal["history-input"]
    status: Literal["failed"]
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    seconds: float = Field(gt=0, lt=CALIBRATION_SECONDS)
    started_unix: float = Field(gt=0)
    order: tuple[tuple[int, int], ...]
    error: str = Field(min_length=1)


class _Child(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    arguments: list[str]
    pid: int = Field(gt=0)
    status: Literal["completed", "failed"]
    seconds: float = Field(gt=0)
    started_unix: float = Field(gt=0)
    load: tuple[float, float, float]
    error: str | None = None


class _Runtime(BaseModel):
    model_config = ConfigDict(extra="forbid")
    common: dict[str, str]
    bindings: tuple[InputBinding, InputBinding]


def evidence_digest(path: Path) -> str:
    """Bind relative names, sizes and hashes, including logs, locks and SQLite."""
    files: dict[str, dict[str, str | int]] = {}
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise ValueError("predecessor evidence cannot contain symlinks")
        if item.is_file():
            files[str(item.relative_to(path))] = {
                "sha256": file_sha256(item),
                "bytes": item.stat().st_size,
            }
    return canonical_sha256(files)


def admit_predecessor(path: Path, sha256: str) -> Predecessor:
    """Fail closed on live, changed, admitted, scientific or ambiguous evidence."""
    path = path.resolve(strict=True)
    if evidence_digest(path) != sha256:
        raise ValueError("predecessor evidence hash changed")
    state = _Supervisor.model_validate_json((path / "supervisor.json").read_text())
    children = [
        _Child.model_validate_json(p.read_text())
        for p in sorted(path.glob("child-*.json"))
    ]
    if psutil.pid_exists(state.pid) or any(psutil.pid_exists(c.pid) for c in children):
        raise ValueError("predecessor supervisor or child is live or ambiguous")
    expected = [
        ["--preflight", "--out", str(path)],
        ["--calibrate-arm", "0", "--out", str(path)],
    ]
    if (
        state.order != ORDER
        or [c.arguments for c in children] != expected
        or [c.status for c in children] != ["completed", "failed"]
        or sum(c.seconds for c in children) > state.seconds
    ):
        raise ValueError("requires terminal failed pre-admission off calibration")
    runtime = _Runtime.model_validate_json((path / "runtime-bindings.json").read_text())
    validate_bindings(runtime.common, runtime.bindings)
    run_path = path / "calibration/history-off/run.json"
    run = TrainingRun.model_validate_json(run_path.read_text())
    if run.identities.get("source_commit") != state.source_commit:
        raise ValueError("predecessor source identity differs")
    validate_run(
        run,
        recipes(40, calibration=True)[0],
        CALIBRATION_SEED,
        runtime.bindings[0],
        runtime.common,
    )
    # Read the store without invoking VerifyStore's schema/write initialization.
    with closing(
        sqlite3.connect(
            f"{(path / 'calibration.sqlite').as_uri()}?mode=ro&immutable=1", uri=True
        )
    ) as database:
        rows = database.execute("SELECT payload FROM training_runs").fetchall()
        stages = database.execute(
            "SELECT run_id, payload FROM training_stages"
        ).fetchall()
        if (
            len(rows) != 1
            or json.loads(rows[0][0]) != run.model_dump(mode="json", exclude={"stages"})
            or len(stages) != len(run.stages)
            or any(run_id != run.id for run_id, _ in stages)
            or {json.loads(payload)["id"]: json.loads(payload) for _, payload in stages}
            != {s.id: s.model_dump(mode="json") for s in run.stages}
        ):
            raise ValueError("predecessor store differs from calibration receipt")
        for table in (
            "runs",
            "run_configs",
            "evaluations",
            "evaluation_choice_sets",
            "evaluation_actions",
            "reports",
        ):
            if database.execute(f"SELECT count(*) FROM {table}").fetchone()[0]:
                raise ValueError("predecessor store contains non-calibration evidence")
    if {str(p.relative_to(path)) for p in path.rglob("*") if p.is_dir()} - {
        "calibration",
        "calibration/history-off",
        "calibration/history-off/policy-0",
        "calibration/history-off/policy-1",
    }:
        raise ValueError(
            "predecessor has admitted, scientific or unrecognized directories"
        )
    allowed = {
        "supervisor.json",
        "runtime-bindings.json",
        "preflight.json",
        "children.log",
        "calibration.sqlite",
        "calibration/history-off.writer.lock",
        "calibration/history-off/run.json",
        *(p.name for p in path.glob("child-*.json")),
    }
    for stage in run.stages:
        if set(stage.artifacts) != {"raw", "ema", "optimizer"}:
            raise ValueError("predecessor calibration exports incomplete")
        for artifact in stage.artifacts.values():
            target = Path(artifact["path"]).resolve(strict=True)
            if (
                target.parent != run_path.parent
                or file_sha256(target) != artifact["sha256"]
            ):
                raise ValueError("predecessor artifact identity differs")
            allowed.add(str(target.relative_to(path)))
    actual = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()}
    if actual != allowed:
        raise ValueError(
            "predecessor has admitted, scientific or unrecognized evidence"
        )
    if evidence_digest(path) != sha256:
        raise ValueError("predecessor evidence changed during admission")
    return Predecessor(
        path=path,
        sha256=sha256,
        seconds=state.seconds,
        source_commit=state.source_commit,
    )
