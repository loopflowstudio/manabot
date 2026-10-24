"""Default phase timers record where time goes without changing training.

Timers end at device waits the code already performs, add no synchronization
and draw no training randomness, so the learned state and diagnostics must be
bit-identical with timing on or off.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from manabot.infra.config import MatchConfig
from manabot.infra.profiler import Profiler
from manabot.training.execution import execute_regime
from manabot.training.learning.learner import Learner
from manabot.training.learning.objectives import update_iteration
from manabot.training.regime import (
    DampedPolicyLearning,
    Learning,
    Schedule,
    TrainingRegime,
    TrainSelfPlay,
)
from manabot.verify.store import VerifyStore
import managym
from tests.training.test_two_seat_learning import _agent, _collector

DAMPED_PHASES = {
    "Batch transfer and behavior checks",
    "Targets",
    "Selection and health diagnostics",
    "Minibatches",
    "Minibatches/Forward and loss",
    "Minibatches/Backward and gradient clipping",
    "Minibatches/Optimizer step",
    "Minibatches/Minibatch diagnostics",
}
SURROGATE_PHASES = DAMPED_PHASES - {
    "Batch transfer and behavior checks",
    "Targets",
} | {"Batch transfer, targets and behavior checks"}


def _rules() -> list[tuple[DampedPolicyLearning | Learning, set[str]]]:
    damped = DampedPolicyLearning(
        gradient="damped_policy", advantage_quantile=0, min_advantage=0
    )
    surrogate = Learning(
        retained_fraction=1,
        epochs=2,
        minibatches=2,
        learning_rate=Schedule(initial=0.01),
        tau=Schedule(initial=0),
        collection_kl=0,
    )
    return [(damped, DAMPED_PHASES), (surrogate, SURROGATE_PHASES)]


def _update(
    learner: Learner, learning: DampedPolicyLearning | Learning, batch: object
) -> dict[str, object]:
    with learner.profiler.track("Learning"):
        return dict(
            update_iteration(
                learner, batch, learning, 0.5, np.random.default_rng(3), iteration=1
            )
        )


@pytest.mark.parametrize("rule", range(2))
def test_learning_timers_leave_state_and_diagnostics_identical(rule: int) -> None:
    learning, phases = _rules()[rule]
    agent = _agent()
    batch = _collector(agent).collect(agent, 1)
    timed = Learner(deepcopy(agent), "cpu", minibatch_rows=2)
    untimed = Learner(deepcopy(agent), "cpu", minibatch_rows=2)
    untimed.profiler = Profiler(enabled=False)

    assert _update(timed, learning, batch) == _update(untimed, learning, batch)
    for (name, a), (_, b) in zip(
        timed.agent.state_dict().items(), untimed.agent.state_dict().items()
    ):
        assert torch.equal(a, b), name

    recorded = {
        path.removeprefix("Learning/")
        for path in timed.profiler.get_stats()
        if path.startswith("Learning/")
    }
    assert recorded == phases
    assert not untimed.profiler.get_stats()


def test_nested_phases_never_exceed_their_parent() -> None:
    learning, _ = _rules()[0]
    agent = _agent()
    batch = _collector(agent).collect(agent, 1)
    learner = Learner(agent, "cpu", minibatch_rows=2)
    _update(learner, learning, batch)
    stats = learner.profiler.get_stats()
    total = stats["Learning"]["total_time"]
    top = [v for k, v in stats.items() if k.count("/") == 1]
    assert sum(v["total_time"] for v in top) <= total
    minibatches = stats["Learning/Minibatches"]["total_time"]
    nested = [v for k, v in stats.items() if k.startswith("Learning/Minibatches/")]
    assert sum(v["total_time"] for v in nested) <= minibatches
    calls = stats["Learning/Minibatches/Forward and loss"]["count"]
    assert stats["Learning/Minibatches/Optimizer step"]["count"] == calls


def test_collection_timers_leave_the_batch_identical() -> None:
    agent = _agent()
    timed, untimed = _collector(agent), _collector(agent)
    timed.profiler, untimed.profiler = Profiler(), Profiler(enabled=False)
    a, b = timed.collect(agent, 1), untimed.collect(agent, 1)
    for field in ("actions", "logprobs", "rewards", "dones", "values"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    assert any(path.startswith("Collection/") for path in timed.profiler.get_stats())


def test_self_play_stage_records_both_timing_trees_by_default(tmp_path: Path) -> None:
    agent = _agent()
    stage = TrainSelfPlay(
        id="learn",
        operation="train_self_play",
        updates=2,
        streams=2,
        transitions=1,
        learning_seats="both",
        games_per_stream=1,
        learning=Learning(epochs=1, minibatches=1, retained_fraction=1, ema=0.9),
    )
    regime = TrainingRegime(
        id="phase-timing-smoke",
        world=managym.WORLD_VERSION,
        match=MatchConfig(),
        agent=agent.config,
        stages=[stage],
        wall_seconds=120,
        schedule_clock="iteration_fraction",
    )
    with VerifyStore(tmp_path / "store.sqlite") as store:
        run = execute_regime(regime, 17, tmp_path / "run", store)
    record = run.stages[0]
    assert record.collection_timings and record.learning_timings
    assert record.learning_timings["EMA update"].calls == 2
    assert record.learning_timings["Minibatches/Forward and loss"].calls >= 2
    top = sum(t.seconds for p, t in record.learning_timings.items() if "/" not in p)
    assert top <= record.learning_seconds
    collected = sum(
        t.seconds for p, t in record.collection_timings.items() if "/" not in p
    )
    assert collected <= record.collection_seconds
