"""Deterministic numerical controls, including real collection and failed runs.

Set MANABOT_NUMERICS_DEVICE=cuda for the bounded CUDA admission suite. An
explicit unavailable CUDA request fails; CPU results never count as CUDA proof.
"""

from collections.abc import Iterator
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from torch import Tensor

from manabot.env import Match, ObservationSpace, Reward
from manabot.infra.hypers import AgentSpec, MatchHypers, RewardHypers
from manabot.model.agent import Agent
from manabot.model.policy_distribution import (
    NumericalError,
    check_behavior,
    check_probabilities,
    legal_mask,
    policy_logs,
)
from manabot.sim.net_opponent import NetOpponentTrainer, SeatRoutedCollector
from manabot.training import execution
from manabot.training.ataraxos import damped_policy_loss, update_move_iteration
from manabot.training.health import NumericalHealth, OptimizerHealth
from manabot.training.models import AtaraxosMoveLearning, TrainingRegime, TrainSelfPlay
from manabot.verify.store import VerifyStore

DEVICE = os.environ.get("MANABOT_NUMERICS_DEVICE", "cpu")
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _runtime() -> Iterator[None]:
    assert DEVICE in {"cpu", "cuda"}
    if DEVICE == "cuda":
        assert torch.cuda.is_available(), "CUDA admission requested without CUDA"
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _tensor(
    values: list[list[float]], *, grad: bool = False, double: bool = False
) -> Tensor:
    return torch.tensor(
        values,
        device=DEVICE,
        dtype=torch.float64 if double else torch.float32,
        requires_grad=grad,
    )


@pytest.mark.parametrize("gap", [0.0, 80.0, 104.0, 1000.0])
def test_extreme_logs_loss_and_gradients_match_double(gap: float) -> None:
    valid = torch.tensor([[True, True, False]], device=DEVICE)
    behavior_logs = policy_logs(_tensor([[0.0, -gap, 900.0]]), valid)
    actions = torch.tensor([0], device=DEVICE)
    check_behavior(
        behavior_logs, behavior_logs.exp(), actions, behavior_logs[:, 0], valid
    )
    # Moving away from concentrated behavior exercises reverse KL on the
    # underflowed alternative, which cannot be recovered from behavior.exp().
    logits = _tensor([[-0.3, 0.4, 900.0]], grad=True)
    reference = _tensor([[0.5, 0.5, 0.0]])
    loss = damped_policy_loss(
        logits,
        actions,
        _tensor([[0.7]]).flatten(),
        behavior_logs,
        reference,
        valid,
        clip=0.2,
        collection_kl=0.1,
        tau=0.05,
    )
    loss.sum().backward()
    expected_logits = logits.detach().double().requires_grad_()
    expected = damped_policy_loss(
        expected_logits,
        actions,
        _tensor([[0.7]], double=True).flatten(),
        policy_logs(_tensor([[0.0, -gap, 900.0]], double=True), valid),
        reference.double(),
        valid,
        clip=0.2,
        collection_kl=0.1,
        tau=0.05,
    )
    expected.sum().backward()
    torch.testing.assert_close(loss, expected, rtol=2e-6, atol=2e-6)
    assert logits.grad is not None and expected_logits.grad is not None
    torch.testing.assert_close(
        logits.grad.double(), expected_logits.grad, rtol=2e-6, atol=2e-6
    )
    assert torch.isfinite(logits.grad).all() and logits.grad[0, 2] == 0
    if gap >= 104:
        assert behavior_logs.exp()[0, 1] == 0


def test_extreme_sampled_ratio_does_not_overflow_float32() -> None:
    # A float32 subnormal action is still sampleable on a non-flushing kernel.
    logits = _tensor([[0.0, 0.0]], grad=True)
    behavior = _tensor([[0.0, -100.0]])
    loss = damped_policy_loss(
        logits,
        torch.tensor([1], device=DEVICE),
        _tensor([[1.0]]).flatten(),
        behavior,
        _tensor([[0.5, 0.5]]),
        torch.ones_like(logits, dtype=torch.bool),
        clip=0.2,
        collection_kl=0.1,
        tau=0,
    )
    loss.sum().backward()
    assert torch.isfinite(loss).all() and torch.isfinite(logits.grad).all()


@pytest.mark.parametrize(
    "probabilities, invariant",
    [
        ([[float("nan"), 0.5, 0]], "nonfinite_probabilities"),
        ([[-0.1, 1.1, 0]], "negative_probabilities"),
        ([[1.0, 0.0, 0]], "nonpositive_legal_support"),
        ([[0.4, 0.5, 0.1]], "illegal_probability_mass"),
        ([[0.2, 0.2, 0]], "probability_normalization"),
    ],
)
def test_each_original_invariant_is_identified(
    probabilities: list[list[float]], invariant: str
) -> None:
    valid = torch.tensor([[True, True, False]], device=DEVICE)
    with pytest.raises(NumericalError, match=invariant) as caught:
        check_probabilities(_tensor(probabilities), valid, positive=True)
    assert "failed" in caught.value.tensors


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_and_empty_legal_support(bad: float) -> None:
    logits = _tensor([[0.0, bad]])
    with pytest.raises(NumericalError, match="nonfinite_legal_logits"):
        policy_logs(logits, torch.ones_like(logits, dtype=torch.bool))
    with pytest.raises(NumericalError, match="empty_legal_support"):
        policy_logs(logits, torch.zeros_like(logits, dtype=torch.bool))


def test_forced_action_is_finite_with_exact_zero_gradient() -> None:
    logits = _tensor([[1e9, -1e9, 0]], grad=True)
    valid = torch.tensor([[False, True, False]], device=DEVICE)
    logs = policy_logs(logits.detach(), valid)
    loss = damped_policy_loss(
        logits,
        torch.tensor([1], device=DEVICE),
        _tensor([[1.0]]).flatten(),
        logs,
        logs.exp(),
        valid,
        clip=0.2,
        collection_kl=0.1,
        tau=0.1,
    )
    loss.sum().backward()
    assert logits.grad is not None and not logits.grad.any()
    assert torch.isfinite(loss).all()


def test_behavior_mismatch_rejected_without_epsilon() -> None:
    valid = torch.tensor([[True, True]], device=DEVICE)
    logs = policy_logs(_tensor([[0.0, -104.0]]), valid)
    action = torch.tensor([0], device=DEVICE)
    with pytest.raises(NumericalError, match="behavior_log_probability_mismatch"):
        check_behavior(logs, _tensor([[1.0, 1e-30]]), action, logs[:, 0], valid)
    with pytest.raises(NumericalError, match="selected_log_mismatch"):
        check_behavior(logs, logs.exp(), action, logs[:, 0] - 1, valid)
    with pytest.raises(NumericalError, match="zero_sampled_probability"):
        check_behavior(logs, logs.exp(), action + 1, logs[:, 1], valid)


def _health() -> NumericalHealth:
    return NumericalHealth(
        optimizer_steps=0, rejected_steps=0, skipped_steps=0, sampled_steps=0
    )


@pytest.mark.parametrize(
    "mode", ["learning", "zero_rate", "detached", "zero", "invalid"]
)
def test_known_learning_and_ineffective_controls(mode: str) -> None:
    # A fully enumerated two-action reward objective is an exact, low-noise
    # positive learning control. It claims no improvement at playing MTG.
    model = torch.nn.Linear(1, 2, bias=False, device=DEVICE)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0 if mode == "zero_rate" else 0.05
    )
    health = _health()
    observer = OptimizerHealth(model, optimizer, health, iteration=1)
    before_rng = torch.get_rng_state().clone()
    valid = torch.ones((2, 2), device=DEVICE, dtype=torch.bool)
    initial = model.weight.detach().clone()
    for _ in range(12):
        logits = model(torch.ones((2, 1), device=DEVICE))
        behavior = policy_logs(logits.detach(), valid)
        loss = (
            damped_policy_loss(
                logits,
                torch.arange(2, device=DEVICE),
                torch.tensor([1.0, -1.0], device=DEVICE),
                behavior,
                torch.full_like(logits, 0.5),
                valid,
                clip=0.2,
                collection_kl=0.1,
                tau=0,
            )
            * behavior.exp()[0]
        ).sum()
        if mode == "detached":
            with pytest.raises(NumericalError, match="detached_objective"):
                observer.step(loss.detach(), 1.0)
            assert health["rejected_steps"] == 1
            break
        if mode == "invalid":
            with pytest.raises(NumericalError, match="nonfinite_objective"):
                observer.step(loss * float("nan"), 1.0)
            break
        observer.step(loss * 0 if mode == "zero" else loss, 1.0)
    torch.testing.assert_close(torch.get_rng_state(), before_rng)
    if mode == "learning":
        assert model(torch.ones((1, 1), device=DEVICE)).softmax(-1)[0, 0] > 0.7
        assert (
            health["parameter_delta_l2"] > 0 and health["gradient_nonzero_tensors"] > 0
        )
    else:
        torch.testing.assert_close(model.weight, initial)
    if mode == "zero_rate":
        assert health["gradient_norm_before"] > 0 and health["parameter_delta_l2"] == 0
    if mode == "zero":
        assert (
            health["gradient_zero_tensors"] == 1
            and health["gradient_missing_tensors"] == 0
        )


def test_missing_and_frozen_components_and_unsampled_cadence() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)).to(DEVICE)
    model[1].bias.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters())
    health = _health()
    loss = model[0](torch.ones((1, 1), device=DEVICE)).square().mean()
    OptimizerHealth(model, optimizer, health, 1).step(loss, 0.01)
    assert health["gradient_missing_tensors"] == 1
    assert health["frozen_parameter_tensors"] == 1
    assert health["gradient_norm_after"] <= 0.010001
    health = _health()
    OptimizerHealth(model, optimizer, health, 2).step(
        model(torch.ones((1, 1), device=DEVICE)).square().mean(), 1.0
    )
    assert health["sampled_steps"] == 0 and "parameter_delta_l2" not in health


class ExtremeAgent(Agent):
    def forward_distribution(self, obs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        logits, values = super().forward_distribution(obs)
        # Keep real game/observation/learner flow, make one legal alternative
        # numerically vanish in a deterministic regression, not by long training.
        order = torch.arange(logits.shape[-1], device=logits.device)
        return logits - order * 110.0, values


def test_extreme_real_collector_to_learner() -> None:
    space = ObservationSpace()
    agent = ExtremeAgent(space, AgentSpec(hidden_dim=8, num_attention_heads=2)).to(
        DEVICE
    )
    collector = SeatRoutedCollector(
        space,
        Match(MatchHypers()),
        Reward(RewardHypers()),
        num_envs=2,
        opponent_mode="self",
        device=DEVICE,
    )
    batch = collector.collect(agent, 4)
    assert np.isfinite(batch.log_probabilities[batch.obs["actions_valid"] > 0]).all()
    trainer = cast(
        NetOpponentTrainer,
        SimpleNamespace(
            agent=agent,
            collector=collector,
            optimizer=torch.optim.Adam(agent.parameters()),
            experiment=SimpleNamespace(device=DEVICE),
            _obs_to_tensors=NetOpponentTrainer._obs_to_tensors,
        ),
    )
    learning = AtaraxosMoveLearning(
        gradient="ataraxos_move", advantage_quantile=0, min_advantage=0
    )
    result = update_move_iteration(trainer, batch, learning, 1)
    assert result["numerical"]["legal_underflow_count"] > 0
    assert result["numerical"]["optimizer_steps"] == 4
    assert result["numerical"]["parameter_delta_l2"] > 0
    assert all(torch.isfinite(p).all() for p in agent.parameters())


@pytest.mark.parametrize("fault", ["none", "capture", "telemetry"])
def test_failed_update_retains_state_and_excludes_prior_overhead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    recipe = TrainingRegime.model_validate_json(
        (ROOT / "experiments/regimes/ataraxos-move-scalar.json").read_text()
    )
    recipe.stages = recipe.stages[:1]
    recipe.agent.hidden_dim = 8
    stage = recipe.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.transitions = 2
    stage.updates = 2
    stage.execution.device = DEVICE
    stage.learning.min_advantage = 0
    offset = [0.0]
    clock = execution.time.perf_counter
    monkeypatch.setattr(execution.time, "perf_counter", lambda: clock() + offset[0])
    original_update = execution.update_iteration
    calls = [0]

    def update(*args: object, **kwargs: object) -> object:
        calls[0] += 1
        if calls[0] == 1:
            return original_update(*args, **kwargs)
        offset[0] += 0.25
        batch = args[1]
        probabilities = torch.as_tensor(batch.probabilities).clone()
        probabilities[-1, -1, 0] = float("nan")
        check_probabilities(
            probabilities,
            torch.as_tensor(batch.obs["actions_valid"]) > 0,
            positive=False,
        )
        raise AssertionError("expected invariant failure")

    original_save = VerifyStore.save_training_run

    def save(self: VerifyStore, run: object) -> None:
        if fault == "telemetry" and run.status == "failed":
            raise OSError("fixture telemetry failure")
        original_save(self, run)
        # Simulates expensive initialization/persistence accumulated before the
        # failing operation. Keep watchdog limits unchanged and generous.
        if run.status == "running":
            offset[0] += 2.0

    monkeypatch.setattr(execution, "update_iteration", update)
    monkeypatch.setattr(VerifyStore, "save_training_run", save)
    if fault == "capture":

        def failed_capture(*args: object, **kwargs: object) -> list[Path]:
            raise OSError("fixture disk failure")

        monkeypatch.setattr(execution, "save_incident", failed_capture)
    with VerifyStore(tmp_path / "training.sqlite") as store:
        with pytest.raises(NumericalError, match="nonfinite_probabilities"):
            execution.execute_regime(recipe, 123, tmp_path / "run", store)
    if fault == "telemetry":
        assert (
            "nonfinite_probabilities"
            in json.loads((tmp_path / "run/terminal-error.json").read_text())["error"]
        )
        return
    run = json.loads((tmp_path / "run/run.json").read_text())
    record = run["stages"][0]
    assert run["status"] == "failed"
    assert record["numerical_failure"]["invariant"] == "nonfinite_probabilities"
    if fault == "capture":
        assert "incident capture failed: OSError" in run["error"]
        return
    assert "incident capture failed" not in run["error"]
    assert len(record["diagnostics"]) == 1
    assert record["learning_seconds"] >= 0.25
    assert record["diagnostic_seconds"] >= 2
    assert record["learning_seconds"] < record["diagnostic_seconds"]
    artifacts = record["rejected_artifacts"]
    assert set(artifacts) == {
        "numerical/tensors.pt",
        "numerical/state.pt",
        "numerical/incident.json",
    }
    metadata = json.loads(
        Path(artifacts["numerical/incident.json"]["path"]).read_text()
    )
    assert metadata["iteration"] == 2 and metadata["completed_updates"] == 1
    assert metadata["source"]["training_source_sha256"]
    tensors = torch.load(artifacts["numerical/tensors.pt"]["path"], weights_only=True)
    assert torch.isnan(tensors["probabilities"]).any()
    assert tensors["row_indexes"].tolist() == [7]
    state = torch.load(artifacts["numerical/state.pt"]["path"], weights_only=False)
    assert (
        state["optimizer"]["state"]
        and state["model"]
        and state["collector_rng"].numel()
    )
    assert not record["artifacts"]  # No invalid replacement policy was exported.


@pytest.mark.parametrize("bad", [float("nan"), -1.0, 0.5])
def test_mask_corruption_is_not_reinterpreted_as_legality(bad: float) -> None:
    with pytest.raises(NumericalError, match="invalid_action_mask"):
        legal_mask(_tensor([[1.0, bad]]))


def test_nonfinite_learner_logits_use_incident_error() -> None:
    valid = torch.tensor([[True, True]], device=DEVICE)
    with pytest.raises(NumericalError, match="nonfinite_legal_logits"):
        damped_policy_loss(
            _tensor([[float("nan"), 0.0]]),
            torch.tensor([0], device=DEVICE),
            _tensor([[1.0]]).flatten(),
            _tensor([[0.5, 0.5]]).log(),
            _tensor([[0.5, 0.5]]),
            valid,
            clip=0.2,
            collection_kl=0.1,
            tau=0.1,
        )


def test_nonfinite_gradient_rejects_before_adam() -> None:
    model = torch.nn.Linear(1, 1, bias=False, device=DEVICE)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = torch.optim.Adam(model.parameters())
    health = _health()
    loss = model.weight.sqrt().sum()  # finite loss, infinite derivative at zero
    with pytest.raises(NumericalError, match="nonfinite_gradients") as caught:
        OptimizerHealth(model, optimizer, health, 1).step(loss, 1.0)
    assert caught.value.health["rejected_steps"] == 1
    assert not optimizer.state
    assert torch.isfinite(model.weight).all()


def test_optimizer_health_preserves_exact_updates_and_device_rng() -> None:
    model = torch.nn.Linear(2, 2, device=DEVICE)
    control = torch.nn.Linear(2, 2, device=DEVICE)
    control.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    baseline = torch.optim.Adam(control.parameters(), lr=0.01)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if DEVICE == "cuda" else None
    for iteration in (1, 2, 25):
        loss = model(torch.ones((4, 2), device=DEVICE)).square().mean()
        OptimizerHealth(model, optimizer, _health(), iteration).step(loss, 0.5)
        baseline.zero_grad(set_to_none=True)
        control(torch.ones((4, 2), device=DEVICE)).square().mean().backward()
        torch.nn.utils.clip_grad_norm_(
            control.parameters(), 0.5, error_if_nonfinite=True
        )
        baseline.step()
    for actual, expected in zip(model.parameters(), control.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), cpu_rng, rtol=0, atol=0)
    if cuda_rng is not None:
        torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)
