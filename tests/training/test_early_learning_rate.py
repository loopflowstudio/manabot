"""Early learning-rate screen declaration and report projection; no training."""

import subprocess
import sys
import time

import pytest

from experiments.runners import early_learning_rate as screen
from manabot.training.checkpoint_queue import stop_process_group
from manabot.training.models import AtaraxosMoveLearning, TrainSelfPlay


def test_arms_change_only_a_constant_learning_rate() -> None:
    base = screen.recipe()
    resolved = screen.declaration(base, screen.schedule()).resolve()
    assert [case.name for case in resolved.cases] == [
        f"{screen.NAME}-{name}" for name, _ in screen.ARMS
    ]
    controls = []
    for case, (_, rate) in zip(resolved.cases, screen.ARMS, strict=True):
        (stage,) = case.regime.stages
        assert isinstance(stage, TrainSelfPlay)
        assert isinstance(stage.learning, AtaraxosMoveLearning)
        assert stage.updates == screen.UPDATES
        # Equal clamps make the iteration schedule constant from the first
        # update through well past the end of the run.
        assert {stage.learning.rates(i)[0] for i in (1, 300, 1500, 100_000)} == {rate}
        data = case.regime.model_dump(mode="json")
        data.pop("id")
        learning = data["stages"][0]["learning"]
        learning.pop("learning_rate_min")
        learning.pop("learning_rate_max")
        controls.append(data)
    assert controls[0] == controls[1] == controls[2]
    # The control arm is the rate the unmodified schedule holds early on.
    (source,) = base.stages
    assert isinstance(source, TrainSelfPlay)
    assert isinstance(source.learning, AtaraxosMoveLearning)
    assert source.learning.rates(1)[0] == screen.ARMS[0][1]
    assert source.learning.rates(screen.UPDATES)[0] == screen.ARMS[0][1]


def test_schedule_pairs_seeds_and_reserves_held_out_deals() -> None:
    plan = screen.schedule()
    assert plan.seeds == screen.SEEDS
    assert plan.checkpoint_updates == 300 and screen.UPDATES % 300 == 0
    assert len(plan.monitoring.protocol.deal_seeds) * 4 == 100
    assert plan.monitoring.include_initial
    assert not set(plan.monitoring.protocol.deal_seeds) & set(screen.RESERVED_DEALS)
    assert plan.wall_seconds == 16 * 3600
    plan.admit_regime(screen.recipe())


def test_windows_summarize_update_diagnostics() -> None:
    def row(norm: float, kl: float, rejected: int) -> dict[str, object]:
        return {
            "collection_kl": kl,
            "entropy": 1.0,
            "value_loss": 0.5,
            "gradient_norm": norm,
            "numerical": {
                "clip_fraction": 0.25,
                "legal_logit_gap_max": norm * 10,
                "skipped_steps": 2,
                "rejected_steps": rejected,
                "nonfinite_count": 0,
            },
        }

    first, tail = screen.windows(
        [row(0.1, 0.002, 0), row(0.5, 0.004, 1), row(0.2, 0.01, 0)], 0.267, 2
    )
    assert (first.first, first.last, tail.first, tail.last) == (0, 2, 2, 3)
    assert first.collection_kl == pytest.approx(0.003)
    assert first.clipped_fraction == 0.5 and tail.clipped_fraction == 0
    assert first.gradient_norm_median == pytest.approx(0.3)
    assert first.legal_logit_gap_max == 5
    assert (first.skipped_steps, first.rejected_steps) == (4, 1)
    with pytest.raises(ValueError, match="numerical health"):
        screen.windows([{"gradient_norm": 1.0}], 0.267, 2)


def test_stopping_an_exited_unreaped_group_is_not_an_error() -> None:
    # Darwin answers EPERM for a group holding only zombies; the coordinator
    # must treat that as already stopped instead of abandoning the experiment.
    process = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    time.sleep(0.5)
    assert stop_process_group(process) == 0
