"""Scoped serving preserves an ordinary multi-update run's learned state."""

from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

from experiments.runners.collection_attribution import ProfileKind, attribute_collection
from experiments.runners.inference_preparation import (
    checkpoint_witnesses,
    reference_inference,
    state_digest,
)
from manabot.sim.net_opponent import SeatRoutedCollector
from manabot.training.execution import execute_regime
from manabot.training.models import TrainSelfPlay
from manabot.verify.store import VerifyStore
from tests.training.test_recovery import assert_learning_diagnostics_equal, recipe


def test_prepared_inference_preserves_complete_training(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    config = recipe()
    stage = config.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.transitions = 8
    runs = []
    witnesses: list[dict[str, str]] = []
    for reference in (True, False):
        root = tmp_path / ("reference" if reference else "prepared")
        root.mkdir()
        with (
            reference_inference(reference),
            VerifyStore(root / "verify.sqlite") as store,
        ):
            run = execute_regime(config, 10349, root / "run", store)
        assert run.status == "completed"
        runs.append(run)
        witnesses.append(checkpoint_witnesses(root / "run"))
    assert witnesses[0] == witnesses[1]
    assert_learning_diagnostics_equal(runs[0].stages[0], runs[1].stages[0])


def test_witness_detects_changed_optimizer_tensor_and_preserves_mapping_order() -> None:
    state = {"step": torch.tensor(3), "moment": torch.tensor([0.1, 0.2])}
    before = state_digest(state)
    assert before == state_digest(dict(reversed(list(state.items()))))
    state["moment"][1] += 0.01
    assert before != state_digest(state)


@pytest.mark.parametrize("kind", ["python", "torch"])
def test_collection_attribution_preserves_training_and_restores_patch(
    tmp_path: Path,
    kind: ProfileKind,
) -> None:
    torch.set_num_threads(1)
    config = recipe()
    original = SeatRoutedCollector.collect
    witnesses = []
    for instrumented in (False, True):
        root = tmp_path / str(instrumented)
        root.mkdir()
        context = attribute_collection(root, kind) if instrumented else nullcontext()
        with context, VerifyStore(root / "verify.sqlite") as store:
            run = execute_regime(config, 10349, root / "run", store)
        assert run.status == "completed"
        assert SeatRoutedCollector.collect is original
        witnesses.append(checkpoint_witnesses(root / "run"))
    assert witnesses[0] == witnesses[1]
    artifact = "collection.pstats" if kind == "python" else "trace.json"
    assert (tmp_path / "True" / artifact).stat().st_size > 0
