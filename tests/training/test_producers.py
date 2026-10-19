"""CPU producers preserve global collection, learning and recovery identities."""

from contextlib import ExitStack, closing
from dataclasses import asdict
import json
from pathlib import Path
import threading

import pytest
import torch

from experiments.runners.inference_preparation import (
    checkpoint_witnesses,
    collection_witnesses,
)
from manabot.env import Match, ObservationSpace, Reward
from manabot.infra.hypers import RewardHypers
from manabot.model.agent import Agent
from manabot.sim.net_opponent import SeatRoutedCollector
from manabot.training import execution
from manabot.training.matchups import Curriculum
from manabot.training.models import Execution, Stage, TrainingRegime, TrainSelfPlay
from manabot.training.recovery import load_update
from manabot.verify.store import VerifyStore
import managym
from tests.training.test_recovery import (
    assert_learning_diagnostics_equal,
    assert_state_equal,
    interrupt,
    recipe,
)


def _producers() -> set[threading.Thread]:
    return {t for t in threading.enumerate() if t.name.startswith("manabot-producer")}


def _collector(
    workers: int, streams: int, curriculum: Curriculum
) -> SeatRoutedCollector:
    value = recipe()
    return SeatRoutedCollector(
        ObservationSpace(value.observation),
        Match(value.match),
        Reward(RewardHypers()),
        num_envs=streams,
        seed=97,
        opponent_mode="self",
        workers=workers,
        recovery_max_game_microsteps=100_000,
        matchup_curriculum=curriculum,
    )


@pytest.mark.parametrize(
    "workers,streams,curriculum",
    [
        (2, 5, "fixed"),
        (4, 5, "fixed"),
        (4, 8, "cross-balanced"),
    ],
)
def test_producers_preserve_rollouts_and_cross_worker_restore(
    workers: int,
    streams: int,
    curriculum: Curriculum,
) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(197)
    value = recipe()
    agent = Agent(ObservationSpace(value.observation), value.agent).eval()
    before = _producers()
    with ExitStack() as owners:
        serial = owners.enter_context(closing(_collector(1, streams, curriculum)))
        parallel = owners.enter_context(
            closing(_collector(workers, streams, curriculum))
        )
        restored = owners.enter_context(
            closing(_collector(workers, streams, curriculum))
        )
        assert_state_equal(
            vars(serial.collect(agent, 32)), vars(parallel.collect(agent, 32))
        )
        left, right = serial.snapshot(), parallel.snapshot()
        left.stats.seconds = right.stats.seconds = 0
        assert_state_equal(asdict(left), asdict(right))
        restored.restore(left, lambda: None)
        expected = serial.collect(agent, 16)
        assert_state_equal(vars(expected), vars(restored.collect(agent, 16)))
        assert_state_equal(vars(expected), vars(parallel.collect(agent, 16)))
    assert _producers() == before


def test_producer_failure_closes_all_workers() -> None:
    before = _producers()
    with closing(_collector(2, 4, "fixed")) as collector:
        with pytest.raises(managym.AgentError):
            collector._env.step_into_buffers([-999, 0, 0, 0])
        with pytest.raises(RuntimeError, match="closed"):
            collector._env.current_agent_indices()
    assert _producers() == before


def test_constructor_failure_closes_producers(monkeypatch: pytest.MonkeyPatch) -> None:
    before = _producers()

    def fail_generator(*args: object, **kwargs: object) -> None:
        raise RuntimeError("sampling device unavailable")

    monkeypatch.setattr(torch, "Generator", fail_generator)
    with pytest.raises(RuntimeError, match="sampling device"):
        _collector(2, 4, "fixed")
    assert _producers() == before


def test_producer_admission_rejects_unsupported_execution() -> None:
    with pytest.raises(ValueError, match="only for train_self_play"):
        Stage(id="unsupported", execution=Execution(workers=2))
    value = recipe()
    value.stages[0].execution.workers = 4
    with pytest.raises(ValueError, match="cannot exceed streams"):
        TrainingRegime.model_validate(value.model_dump())
    value.stages[0].execution.workers = 2
    stage = value.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.root = "lethal-target-v1"
    with pytest.raises(ValueError, match="diagnostic roots"):
        TrainingRegime.model_validate(value.model_dump())


def test_producers_preserve_complete_learning_and_failure_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.set_num_threads(1)
    value = recipe()
    stage = value.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.streams = 4
    before = _producers()
    monkeypatch.setattr(execution, "PROGRESS_EXPORT_SECONDS", 0.0)
    monkeypatch.setattr(execution, "PROGRESS_EXPORT_SHARE", float("inf"))
    with VerifyStore(tmp_path / "verify.sqlite") as store:
        serial = execution.execute_regime(value, 197, tmp_path / "serial", store)
        stage.execution.workers = 4
        parallel = execution.execute_regime(value, 197, tmp_path / "parallel", store)
        assert serial.status == parallel.status == "completed"
        assert checkpoint_witnesses(tmp_path / "serial") == checkpoint_witnesses(
            tmp_path / "parallel"
        )
        assert_learning_diagnostics_equal(serial.stages[0], parallel.stages[0])
        assert _producers() == before
        failed = interrupt(value, tmp_path / "failed", store, monkeypatch)
        assert _producers() == before
        resumed = execution.execute_regime(
            value, 197, tmp_path / "resumed", store, resume_from=failed.id
        )
        assert resumed.status == "completed"
        expected, actual = load_update(parallel, store), load_update(resumed, store)
        for name in ("learner", "optimizer", "ema", "minibatch_rng", "torch_rng"):
            assert_state_equal(getattr(expected, name), getattr(actual, name))
        assert_state_equal(expected.collector.buffers, actual.collector.buffers)
        assert_state_equal(
            expected.collector.sampling_rng, actual.collector.sampling_rng
        )
        assert expected.collector.journal == actual.collector.journal
        assert_learning_diagnostics_equal(parallel.stages[0], resumed.stages[0])
    assert _producers() == before


def test_collection_proof_preserves_learning_and_matches_workers(
    tmp_path: Path,
) -> None:
    torch.set_num_threads(1)
    value = recipe()
    stage = value.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.streams = 4
    original = SeatRoutedCollector.collect
    results = []
    for workers in (1, 4):
        stage.execution.workers = workers
        root = tmp_path / str(workers)
        root.mkdir()
        with collection_witnesses(root), VerifyStore(root / "verify.sqlite") as store:
            run = execution.execute_regime(value, 197, root / "run", store)
        assert run.status == "completed"
        assert SeatRoutedCollector.collect is original
        proofs = [
            json.loads(path.read_text())
            for path in sorted(root.glob("collection-witness-*.json"))
        ]
        assert len(proofs) == 3
        results.append((proofs, checkpoint_witnesses(root / "run")))
    assert results[0] == results[1]
    with VerifyStore(tmp_path / "plain.sqlite") as store:
        plain = execution.execute_regime(value, 197, tmp_path / "plain", store)
    assert plain.status == "completed"
    assert results[0][1] == checkpoint_witnesses(tmp_path / "plain")
