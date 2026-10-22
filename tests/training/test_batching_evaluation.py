"""Matched-experience evaluation cannot silently substitute another export."""

import json
from pathlib import Path
import shutil
import sys

import pytest
import torch

from experiments.runners import batching_evaluation
from experiments.runners.batching_evaluation import select_checkpoints
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.training import monitor_evaluation as monitor
from manabot.training.execution import execute_regime
from manabot.training.models import ArtifactReference, TrainingCoordinates, TrainingRun
from manabot.verify.store import VerifyStore
from tests.training.test_recovery import recipe


@pytest.fixture
def retained_run(tmp_path: Path) -> Path:
    torch.set_num_threads(1)
    source = tmp_path / "original"
    with VerifyStore(tmp_path / "training.sqlite") as store:
        run = execute_regime(recipe(), 13590, source, store, checkpoint_updates=(1, 2))
    assert run.status == "completed"
    relocated = tmp_path / "retained"
    shutil.move(str(source), relocated)
    return relocated


def test_selects_initial_milestone_and_endpoint_after_relocation(
    retained_run: Path,
) -> None:
    selected = select_checkpoints(retained_run, (0, 64, 128, 192))
    assert [point.coordinates.updates for point in selected.points] == [0, 1, 2, 3]
    assert [point.coordinates.learner_transitions for point in selected.points] == [
        0,
        64,
        128,
        192,
    ]
    for point in selected.points:
        assert Path(point.artifact["path"]).parent == retained_run
        agent, _ = load_checkpoint_agent(point.artifact["path"])
        assert all(bool(torch.isfinite(p).all()) for p in agent.parameters())
    # Selection does not rewrite the original producer paths or its run receipt.
    original = TrainingRun.model_validate_json((retained_run / "run.json").read_text())
    assert Path(original.stages[0].artifacts["raw"]["path"]).parent.name == "original"


def test_rejects_missing_experience_or_corrupt_export(retained_run: Path) -> None:
    with pytest.raises(ValueError, match="65 transitions"):
        select_checkpoints(retained_run, (0, 65, 192))
    selected = select_checkpoints(retained_run, (0, 192))
    Path(selected.points[-1].artifact["path"]).write_bytes(b"corrupt checkpoint")
    with pytest.raises(ValueError, match="checkpoint bytes differ"):
        select_checkpoints(retained_run, (0, 192))


def test_rejects_incomplete_trajectory(retained_run: Path) -> None:
    path = retained_run / "run.json"
    run = TrainingRun.model_validate_json(path.read_text())
    run.status = "interrupted"
    path.write_text(run.model_dump_json())
    with pytest.raises(ValueError, match="completed fresh single"):
        select_checkpoints(retained_run, (0, 64))


@pytest.mark.parametrize("incomplete", [False, True])
def test_cli_uses_evaluator_result_and_stops_incomplete_panel(
    retained_run: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    incomplete: bool,
) -> None:
    protocol = monitor.MonitorProtocol(deal_seeds=(1359001,))
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(protocol.model_dump_json())
    output = tmp_path / "evaluation"
    observed: list[int] = []

    def evaluate(
        run: TrainingRun,
        artifact: ArtifactReference,
        coordinates: TrainingCoordinates,
        output_dir: Path,
        *,
        protocol: monitor.MonitorProtocol,
        concurrent_activity: str,
    ) -> monitor.MonitorResult:
        # Exercise actual checkpoint admission and the public result type; only
        # the expensive games and their already-tested cohort scoring are stubbed.
        result = monitor.checkpoint_manifest(run, artifact, coordinates, protocol)
        observed.append(coordinates.learner_transitions)
        assert output_dir == output / f"transitions-{coordinates.learner_transitions}"
        assert concurrent_activity.startswith("post-training panel")
        result.status = "incomplete" if incomplete else "completed"
        result.error = "incomplete test cohort" if incomplete else None
        return result

    monkeypatch.setattr(batching_evaluation, "evaluate_checkpoint", evaluate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batching_evaluation",
            "--run",
            str(retained_run),
            "--transitions",
            "0",
            "192",
            "--protocol",
            str(protocol_path),
            "--out",
            str(output),
        ],
    )
    if incomplete:
        with pytest.raises(
            RuntimeError, match="evaluation incomplete at 0 transitions"
        ):
            batching_evaluation.main()
        assert observed == [0]
    else:
        batching_evaluation.main()
        assert observed == [0, 192]
    selection = json.loads((output / "selection.json").read_text())
    assert selection["protocol"] == protocol.model_dump(mode="json")
    assert [
        p["coordinates"]["learner_transitions"] for p in selection["checkpoints"]
    ] == [0, 192]
