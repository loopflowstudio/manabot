"""Admission fixtures retain failed bytes without training or checkpoint loading."""

import json
from pathlib import Path

import pytest

from experiments.runners import (
    history_input as history,
    history_predecessor as recovery,
    run_history_input as supervisor,
)
from manabot.arena.models import canonical_sha256, file_sha256
from manabot.training.execution import atomic_json
from manabot.training.models import StageRecord, TrainingRun
from manabot.verify.store import VerifyStore


@pytest.fixture
def predecessor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(recovery.psutil, "pid_exists", lambda pid: False)
    root = tmp_path / ".runs" / "failed"
    folder = root / "calibration/history-off"
    folder.mkdir(parents=True)
    common = {key: "a" * 64 for key in history.COMMON_KEYS}
    bindings = [
        history.InputBinding(
            recipe_id=name,
            observation_abi_sha256=str(i + 1) * 64,
            input_schema_sha256=str(i + 3) * 64,
            world_binding_sha256=str(i + 5) * 64,
            policy_history_version=i,
        )
        for i, name in enumerate(history.ARMS)
    ]
    recipe = history.recipes(40, calibration=True)[0]
    stages: list[StageRecord] = []
    for spec in recipe.stages:
        artifacts: dict[str, dict[str, str]] = {}
        for name in ("raw", "ema", "optimizer"):
            target = folder / f"{spec.id}-{name}.pt"
            target.write_bytes(b"fixture")
            artifacts[name] = {"path": str(target), "sha256": file_sha256(target)}
        stages.append(
            StageRecord(
                id=spec.id,
                status="completed",
                seconds=30,
                learner_transitions=5120,
                diagnostics=[{}] * 20,
                artifacts=artifacts,
            )
        )
    run = TrainingRun(
        id="fixture",
        regime=recipe,
        regime_digest=canonical_sha256(recipe.model_dump(mode="json")),
        seed=history.CALIBRATION_SEED,
        seed_streams={},
        status="completed",
        seconds=70,
        identities={
            **common,
            "source_commit": "b" * 40,
            "observation_abi_sha256": bindings[0].observation_abi_sha256,
        },
        stages=stages,
    )
    atomic_json(folder / "run.json", run.model_dump(mode="json"))
    with VerifyStore(root / "calibration.sqlite") as store:
        store.save_training_run(run)
    atomic_json(
        root / "runtime-bindings.json",
        {"common": common, "bindings": [b.model_dump(mode="json") for b in bindings]},
    )
    atomic_json(
        root / "supervisor.json",
        {
            "pid": 123,
            "study": "history-input",
            "status": "failed",
            "source_commit": "b" * 40,
            "seconds": 91.15643158298917,
            "started_unix": 100,
            "order": history.ORDER,
            "error": "reload failure",
        },
    )
    for i, (arguments, status, seconds) in enumerate(
        [
            (["--preflight", "--out", str(root)], "completed", 18),
            (["--calibrate-arm", "0", "--out", str(root)], "failed", 72),
        ]
    ):
        atomic_json(
            root / f"child-{i}.json",
            {
                "arguments": arguments,
                "status": status,
                "seconds": seconds,
                "pid": 124 + i,
                "started_unix": 101 + i,
                "load": [0, 0, 0],
            },
        )
    for name in (
        "preflight.json",
        "children.log",
        "calibration/history-off.writer.lock",
    ):
        (root / name).write_text("{}")
    return root


def test_failed_calibration_admitted_without_mutation(predecessor: Path) -> None:
    digest = recovery.evidence_digest(predecessor)
    receipt = recovery.admit_predecessor(predecessor, digest)
    assert receipt.seconds == 91.15643158298917
    assert recovery.evidence_digest(predecessor) == digest


def test_hash_mutation_rejected(predecessor: Path) -> None:
    digest = recovery.evidence_digest(predecessor)
    (predecessor / "children.log").write_text("mutation")
    with pytest.raises(ValueError, match="hash changed"):
        recovery.admit_predecessor(predecessor, digest)


@pytest.mark.parametrize("status", ["completed", "training", "preflight"])
def test_nonfailed_supervisor_rejected(predecessor: Path, status: str) -> None:
    path = predecessor / "supervisor.json"
    data = json.loads(path.read_text())
    data["status"] = status
    atomic_json(path, data)
    with pytest.raises(ValueError):
        recovery.admit_predecessor(predecessor, recovery.evidence_digest(predecessor))


def test_live_predecessor_rejected(
    predecessor: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recovery.psutil, "pid_exists", lambda pid: pid == 124)
    with pytest.raises(ValueError, match="live or ambiguous"):
        recovery.admit_predecessor(predecessor, recovery.evidence_digest(predecessor))


@pytest.mark.parametrize(
    "name", ["resolved-plan.json", "training/run.json", "study/study.json"]
)
def test_admitted_or_scientific_predecessor_rejected(
    predecessor: Path, name: str
) -> None:
    path = predecessor / name
    path.parent.mkdir(exist_ok=True)
    path.write_text("{}")
    with pytest.raises(ValueError, match="admitted, scientific"):
        recovery.admit_predecessor(predecessor, recovery.evidence_digest(predecessor))


def test_artifact_mutation_even_with_new_tree_hash_rejected(predecessor: Path) -> None:
    next((predecessor / "calibration/history-off").glob("*.pt")).write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact identity"):
        recovery.admit_predecessor(predecessor, recovery.evidence_digest(predecessor))


def test_only_selected_collision_is_exempt(
    predecessor: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = predecessor.parent
    monkeypatch.setattr(supervisor, "ROOT", root.parent)
    out = root / "fresh"
    out.mkdir()
    atomic_json(out / "supervisor.json", {"study": "history-input"})
    with pytest.raises(ValueError, match="collision"):
        supervisor._check_collisions(out)
    receipt = recovery.admit_predecessor(
        predecessor, recovery.evidence_digest(predecessor)
    )
    atomic_json(
        out / "supervisor.json",
        {"study": "history-input", "predecessor": receipt.model_dump(mode="json")},
    )
    supervisor._check_collisions(out)
    other = root / "unselected"
    other.mkdir()
    atomic_json(
        other / "supervisor.json", {"study": "history-input", "status": "failed"}
    )
    with pytest.raises(ValueError, match="collision"):
        supervisor._check_collisions(out)
