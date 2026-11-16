"""Host reports preserve selected membership and the actual MPS optimizer step."""

from collections.abc import Iterator
from copy import deepcopy
import time

import pytest
import torch
from torch import Tensor

from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import MatchConfig, ModelConfig, RewardConfig
from manabot.model.actor_critic import ActorCritic
from manabot.training.learning import damped_policy
from manabot.training.learning.learner import Learner
from manabot.training.learning.selection import (
    SelectionGroup,
    selected_moves,
    selection_diagnostics,
)
from manabot.training.regime import DampedPolicyLearning
from manabot.training.rollout.collection import RolloutBatch, SeatRoutedCollector


@pytest.fixture(autouse=True)
def _one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _inputs() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    advantages = torch.tensor(
        [
            [-1.0, 0.0],
            [1.0, 0.25],
            [-0.5, -0.25],
            [0.5, 0.0],
            [-1.0, 0.125],
            [1.0, -0.125],
            [-0.5, 0.0],
            [0.5, 0.0],
        ]
    )
    residuals = advantages * 0.3 + 0.1
    selected = selected_moves(advantages, 0.5, 0.25)
    ends = torch.zeros_like(selected)
    ends[0, 0] = ends[-1, 0] = True
    action_types = torch.tensor([[1, 2]]).expand_as(advantages).long()
    return advantages, residuals, selected, ends, action_types


def test_cpu_selection_report_covers_ties_empty_groups_and_censored_rows() -> None:
    report = selection_diagnostics(*_inputs())
    assert [
        (r["action_type"], r["terminal_distance"], r["rows"], r["retained"])
        for r in report
    ] == [
        (1, "terminal", 2, 2),
        (1, "1-4", 4, 4),
        (1, "5+", 2, 2),
        (2, "censored", 8, 0),
    ]
    assert report[0]["advantage_quantiles"] == [-1.0, -0.625, -0.25, 0.125, 0.5]
    assert report[-1]["retained_advantage_quantiles"] == []
    assert report[-1]["retained_residual_abs_mean"] is None


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_mps_report_matches_host_inputs_and_keeps_device_membership() -> None:
    host = _inputs()
    device = tuple(t.to("mps") for t in host)
    before = tuple(t.clone() for t in device)
    expected = selection_diagnostics(*host)
    actual = selection_diagnostics(*device)
    assert len(actual) == len(expected)
    for observed, reference in zip(actual, expected, strict=True):
        for name in ("action_type", "terminal_distance", "rows", "retained"):
            assert observed[name] == reference[name]
        for name in (
            "advantage_quantiles",
            "retained_advantage_quantiles",
            "residual_abs_mean",
            "retained_residual_abs_mean",
        ):
            if reference[name] is None:
                assert observed[name] is None
            else:
                # Host reductions of the exact copied FP32 inputs; allow only
                # 1e-7 absolute / 1e-6 relative report rounding, never count drift.
                assert observed[name] == pytest.approx(
                    reference[name], abs=1e-7, rel=1e-6
                )
    for original, after in zip(before, device, strict=True):
        assert after.device.type == "mps"
        assert torch.equal(original, after)
    assert torch.equal(device[2], selected_moves(device[0], 0.5, 0.25))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_mps_report_leaves_optimizer_inputs_losses_gradients_and_adam_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One two-row minibatch per arm, with/without descriptive reporting.

    This isolates reporting from learning on the same device and same batch;
    it does not assert cross-device trajectory equivalence.
    """
    torch.manual_seed(83)
    agent = ActorCritic(
        ObservationSpace(), ModelConfig(hidden_dim=8, num_attention_heads=2)
    ).to("mps")
    collector = SeatRoutedCollector(
        agent.observation_space,
        Match(MatchConfig()),
        Reward(RewardConfig()),
        num_envs=2,
        seed=83,
        opponent_mode="self",
        learning_seats="one",
        batch_closure="exact-quota",
        device="mps",
        workers=1,
    )
    try:
        batch = collector.collect(agent, 1, deadline_monotonic=time.perf_counter() + 30)
    finally:
        collector.close()
    learning = DampedPolicyLearning(
        gradient="damped_policy", advantage_quantile=0.5, min_advantage=0
    )
    original_loss = damped_policy.damped_policy_loss
    original_targets = damped_policy._targets
    reports: list[list[SelectionGroup]] = []
    inputs: list[list[Tensor]] = []
    learners: list[Learner] = []
    losses: list[float] = []
    target_inputs: list[damped_policy.DampedTargets] = []

    def targets(
        learner: Learner,
        batch: RolloutBatch,
        learning: DampedPolicyLearning,
        bootstrap_agent: ActorCritic | None = None,
    ) -> damped_policy.DampedTargets:
        result = original_targets(learner, batch, learning, bootstrap_agent)
        target_inputs.append(result)
        return result

    def no_report(*args: Tensor) -> list[SelectionGroup]:
        return []

    def report(*args: Tensor) -> list[SelectionGroup]:
        result = selection_diagnostics(*args)
        reports.append(result)
        return result

    def loss(
        logits: Tensor,
        actions: Tensor,
        advantages: Tensor,
        behavior_logs: Tensor,
        reference: Tensor,
        valid: Tensor,
        *,
        clip: float,
        collection_kl: float,
        tau: float,
    ) -> Tensor:
        inputs.append(
            [
                t.detach().cpu().clone()
                for t in (logits, actions, advantages, behavior_logs, reference, valid)
            ]
        )
        return original_loss(
            logits,
            actions,
            advantages,
            behavior_logs,
            reference,
            valid,
            clip=clip,
            collection_kl=collection_kl,
            tau=tau,
        )

    monkeypatch.setattr(damped_policy, "damped_policy_loss", loss)
    monkeypatch.setattr(damped_policy, "_targets", targets)
    for reporting in (no_report, report):
        learner = Learner(deepcopy(agent), "mps", minibatch_rows=2)
        monkeypatch.setattr(damped_policy, "selection_diagnostics", reporting)
        result = damped_policy.update_damped_iteration(learner, batch, learning, 1)
        assert result["numerical"]["optimizer_steps"] == 1
        learners.append(learner)
        losses.append(result["loss"])
    assert len(reports) == 1 and reports[0]
    assert len(inputs) == 2
    assert len(target_inputs) == 2
    for name in ("advantages", "values", "selected"):
        assert torch.equal(
            getattr(target_inputs[0], name), getattr(target_inputs[1], name)
        )
    assert losses[0] == losses[1]
    for expected, actual in zip(inputs[0], inputs[1], strict=True):
        assert torch.equal(expected, actual)
    for first, second in zip(
        learners[0].agent.parameters(), learners[1].agent.parameters(), strict=True
    ):
        assert torch.equal(first, second)
        if first.grad is None:
            assert second.grad is None
        else:
            assert second.grad is not None and torch.equal(first.grad, second.grad)
        left, right = (
            learners[0].optimizer.state[first],
            learners[1].optimizer.state[second],
        )
        assert left.keys() == right.keys()
        for key in left:
            assert torch.equal(left[key], right[key])
