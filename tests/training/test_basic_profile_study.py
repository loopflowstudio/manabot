"""The paid study's cells execute ordinary training with complete evidence."""

import json
from pathlib import Path

import torch

from experiments.runners.training_profile import Case, cases, child
from manabot.training.models import TrainingRun
from tests.training.test_recovery import recipe


def test_study_is_balanced_and_keeps_proofs_out_of_timing() -> None:
    planned = cases()
    assert len({c.name for c in planned}) == len(planned) == 28
    ordinary = [c for c in planned if not c.proof]
    for width, layers in ((64, 1), (384, 8)):
        for streams in (64, 512):
            for enabled in (False, True):
                selected = [
                    c
                    for c in ordinary
                    if (c.width, c.layers, c.streams, c.profile)
                    == (width, layers, streams, enabled)
                ]
                assert len(selected) == 3
                assert all(c.updates == 16 for c in selected)


def test_study_child_retains_real_training_and_profile(tmp_path: Path) -> None:
    (tmp_path / "baseline.json").write_text(recipe().model_dump_json())
    case = Case(
        name="proof", width=8, layers=1, streams=2, updates=2, profile=True, proof=True
    )
    deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        child(tmp_path, case, device="cpu")
    finally:
        torch.use_deterministic_algorithms(deterministic)
    root = tmp_path / case.name
    run = TrainingRun.model_validate_json((root / "run/run.json").read_text())
    assert run.status == "completed"
    assert run.stages[0].learner_transitions == 32
    assert run.stages[0].collection_timings is not None
    assert len(list(root.glob("collection-witness-*.json"))) == 2
    assert json.loads((root / "witnesses.json").read_text())
    assert json.loads((root / "result.json").read_text())["status"] == "completed"
