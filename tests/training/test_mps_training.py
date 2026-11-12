"""MPS admission, unchanged CPU arithmetic, and one production self-play update."""

from copy import deepcopy
import math
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Literal

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from experiments import training_throughput_packing as packing
from experiments.tla_frontier import frozen
from manabot import cli
from manabot.env.match import Match, Reward
from manabot.env.observation import (
    FACT_DETAIL_INDEX,
    FACT_DIM,
    FACT_EFFECT,
    FACT_ID_INDEX,
    FACT_KIND_INDEX,
    FACT_OBJECT_INDEX,
    FACT_VALID_INDEX,
    ObservationSpace,
)
from manabot.env.shared_effects import SHARED_EFFECT_SHAPES
from manabot.experiment.job import TrainingJob
from manabot.infra.config import RewardConfig
from manabot.infra.identity import canonical_sha256
from manabot.model import semantic_cards, shared_effects
from manabot.model.actor_critic import ActorCritic
from manabot.model.facts import FactEncoder
from manabot.model.numerics import embedding_rows, float64_operand
from manabot.model.policy_distribution import clipped_surrogate
from manabot.model.semantic_cards import (
    RecordEmbeddings,
    RecordNamespace,
    SemanticCardEncoder,
)
from manabot.model.shared_effects import SharedEffectEncoder
from manabot.recipes.baseline import baseline
from manabot.recipes.device import accelerator
from manabot.recipes.local import local_smoke
from manabot.training import execution
from manabot.training.learning import damped_policy
from manabot.training.learning.learner import Learner
from manabot.training.learning.selection import UpdateDiagnostics
from manabot.training.mps import mps_torch_error
from manabot.training.regime import (
    DampedPolicyLearning,
    Execution,
    Learning,
    Precision,
    TrainingRegime,
    TrainSelfPlay,
)
from manabot.training.rollout.collection import RolloutBatch, SeatRoutedCollector
import manabot.training.stages as stages
from manabot.verify.store import VerifyStore
import managym


def test_mps_admitted_without_changing_saved_regimes() -> None:
    assert Execution(device="mps").precision == "float32"
    assert Execution().model_dump() == {
        "device": "cpu",
        "precision": "float32",
        "workers": 1,
        "threads": 1,
        "wall_seconds": 300,
        "memory_bytes": 32 * 1024**3,
    }
    saved = TrainingJob.model_validate_json(
        packing.runs("HEAD", "d1")[0].job.config or ""
    )
    assert canonical_sha256(saved.regime.model_dump(mode="json")) == (
        "5684d15da200eef2c9169b75cce8b54dd643e22a0201a148be80bf2958505b84"
    )
    for plan in (local_smoke(device="mps"), baseline(device="mps").regime):
        assert TrainingRegime.model_validate_json(plan.model_dump_json()) == plan
        assert plan.stages[0].execution.device == "mps"


@pytest.mark.parametrize("precision", ["tf32", "bf16"])
def test_mps_rejects_reduced_precision(precision: Precision) -> None:
    with pytest.raises(ValueError, match="requires CUDA"):
        Execution(device="mps", precision=precision)


def test_mps_rejects_process_recovery_and_belief() -> None:
    data = local_smoke(device="mps").model_dump()
    data["recovery_max_microsteps"] = 100
    with pytest.raises(ValueError, match="MPS process recovery is unsupported"):
        TrainingRegime.model_validate(data)
    data["recovery_max_microsteps"] = None
    data["agent"]["belief_count_buckets"] = 4
    with pytest.raises(
        ValueError, match="MPS requires self-contained ordinary self-play"
    ):
        TrainingRegime.model_validate(data)


def test_unavailable_mps_fails_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with VerifyStore(tmp_path / "runs.sqlite") as store:
        with pytest.raises(
            ValueError, match="MPS requested but unavailable; CPU fallback is forbidden"
        ):
            execution.execute_regime(
                local_smoke(steps=1, seconds=600, learning_seats="one", device="mps"),
                197,
                tmp_path / "unavailable",
                store,
            )


@pytest.mark.parametrize(
    "version,admitted",
    [("2.10.0", False), ("2.14.1", False), ("2.15.0", True), ("3.0.0+cpu", True)],
)
def test_mps_torch_minimum_gates_selection(
    version: str, admitted: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch, "__version__", version)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert (mps_torch_error() is None) == admitted
    assert accelerator() == ("mps" if admitted else None)
    assert local_smoke().stages[0].execution.device == ("mps" if admitted else "cpu")
    if admitted:
        assert baseline().regime.stages[0].execution.device == "mps"
    else:
        for device in (None, "mps"):
            with pytest.raises(
                ValueError, match=f"installed Torch {version} is below that minimum"
            ):
                baseline(device=device)


def test_old_torch_mps_stage_fails_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch, "__version__", "2.10.0")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    with VerifyStore(tmp_path / "runs.sqlite") as store:
        with pytest.raises(
            ValueError,
            match=(
                "MPS training requires Torch 2.15 or newer; installed Torch 2.10.0 "
                "is below that minimum"
            ),
        ):
            execution.execute_regime(
                local_smoke(steps=1, seconds=600, learning_seats="one", device="mps"),
                197,
                tmp_path / "old-torch",
                store,
            )
    assert not (tmp_path / "old-torch" / "checkpoints").exists()


def test_float64_operand_keeps_cpu_tensor_ops_and_gradients_identical() -> None:
    logs = torch.tensor([-2.0, 0.0, 2.0, 90.0], requires_grad=True)
    advantages = torch.tensor([0.5, -0.5, 1.0, 1.0])
    assert float64_operand(logs) is logs
    assert float64_operand(advantages) is advantages
    ratio = logs.double().exp()
    expected = -torch.minimum(
        ratio * advantages.detach(), ratio.clamp(0.8, 1.2) * advantages.detach()
    )
    actual = clipped_surrogate(logs, advantages, 0.2)
    assert actual.device == logs.device and actual.dtype == torch.float64
    assert torch.equal(actual, expected)
    actual_grad = torch.autograd.grad(actual.sum(), logs)[0]
    expected_grad = torch.autograd.grad(expected.sum(), logs)[0]
    assert torch.equal(actual_grad, expected_grad)


@pytest.mark.parametrize("recipe", ["local", "baseline"])
def test_cli_threads_device_and_rejects_regime_overrides(
    recipe: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    def run(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(
            status="completed",
            id="test",
            notebook="baseline.ipynb",
            stages=[SimpleNamespace(artifacts={"raw": {"path": "policy.pt"}})],
        )

    monkeypatch.setattr(cli, f"run_{recipe}", run)
    runner = CliRunner()
    result = runner.invoke(cli.app, ["train", "--recipe", recipe, "--device", "mps"])
    assert result.exit_code == 0, result.output
    assert calls[0]["device"] == "mps"
    for option in ("--regime", "--regime-name"):
        result = runner.invoke(cli.app, ["train", option, "unused", "--device", "mps"])
        assert result.exit_code != 0
        assert "--device" in result.output and "recipe" in result.output


def _assert_finite(value: object) -> None:
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for item in value.values():
            _assert_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_finite(item)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_mps_host_loss_and_rng_preservation() -> None:
    logs = torch.tensor([-2.0, 0.0, 2.0, 90.0], device="mps", requires_grad=True)
    advantages = torch.tensor([0.5, -0.5, 1.0, 1.0], device="mps")
    loss = clipped_surrogate(logs, advantages, 0.2)
    assert loss.device.type == "cpu" and loss.dtype == torch.float64
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert logs.grad is not None and logs.grad.device.type == "mps"
    assert torch.isfinite(logs.grad).all() and logs.grad.abs().sum() > 0

    generator = torch.Generator(device="mps").manual_seed(198)
    state = generator.get_state()
    weights = torch.ones((4, 3), device="mps")
    expected = torch.multinomial(weights, 1, generator=generator)
    generator.set_state(state)
    assert torch.equal(torch.multinomial(weights, 1, generator=generator), expected)
    cpu_state, mps_state = torch.get_rng_state(), torch.mps.get_rng_state()
    with torch.random.fork_rng(devices=[0], device_type="mps"):
        torch.rand(4)
        torch.rand(4, device="mps")
    assert torch.equal(torch.get_rng_state(), cpu_state)
    assert torch.equal(torch.mps.get_rng_state(), mps_state)
    assert stages.cuda_peak_metrics("mps") == {}
    assert stages.mps_memory_metrics("cpu") == {}
    assert stages.mps_memory_metrics("cuda") == {}
    assert all(value >= 0 for value in stages.mps_memory_metrics("mps").values())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_mps_one_update_and_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = local_smoke(steps=1, seconds=600, learning_seats="one", device="mps")
    stage = plan.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.learning.min_advantage = 0
    observed: list[UpdateDiagnostics] = []
    original = stages.update_iteration

    def update(
        learner: Learner,
        batch: RolloutBatch,
        learning: Learning | DampedPolicyLearning,
        progress: float,
        rng: np.random.Generator,
        *,
        iteration: int = 1,
        bootstrap_agent: ActorCritic | None = None,
    ) -> UpdateDiagnostics:
        assert all(p.device.type == "mps" for p in learner.agent.parameters())
        result = original(
            learner,
            batch,
            learning,
            progress,
            rng,
            iteration=iteration,
            bootstrap_agent=bootstrap_agent,
        )
        assert all(torch.isfinite(p).all() for p in learner.agent.parameters())
        _assert_finite(result)
        observed.append(result)
        return result

    monkeypatch.setattr(stages, "update_iteration", update)
    with VerifyStore(tmp_path / "runs.sqlite") as store:
        run = execution.execute_regime(plan, 197, tmp_path / "mps", store)
    assert run.status == "completed", run.error
    record = run.stages[0]
    assert record.actual_device is not None
    assert torch.device(record.actual_device).type == "mps"
    assert record.updates == 1 and (record.optimizer_updates or 0) > 0
    assert len(observed) == 1
    assert Path(record.artifacts["raw"]["path"]).is_file()
    _assert_finite(record.model_dump())


@pytest.mark.parametrize("family", ["none", "name", "detail"])
def test_empty_record_embedding_cpu_loss_gradients_and_adam_equivalent(
    family: Literal["none", "name", "detail"], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compare legacy embeddings with the MPS empty-slice algebra on CPU.

    A zero contribution must retain a zero (not absent) gradient: Adam can
    still update that parameter using momentum from an earlier minibatch.
    """
    torch.set_num_threads(1)
    torch.manual_seed(17)
    cards = SemanticCardEncoder("world", 4, records=True)
    facts = torch.zeros(1, 4, FACT_DIM)
    if family != "none":
        row = facts[0, 0]
        row[FACT_OBJECT_INDEX] = 1
        row[FACT_KIND_INDEX] = FACT_EFFECT
        row[FACT_VALID_INDEX] = 1
        for column, name in (
            (FACT_ID_INDEX, "Modification::AddKeywords"),
            (FACT_DETAIL_INDEX, "flying"),
        ):
            if column == FACT_DETAIL_INDEX and family != "detail":
                continue
            word = managym.learning_token_value(name) & 0xFFFFFFFF
            row[column : column + 2] = torch.tensor([word >> 16, word & 0xFFFF])
    encoder = FactEncoder(4, 4, vocabulary="catalog")
    with torch.no_grad():
        encoder.output.weight.copy_(torch.eye(4))
    original = torch.nn.ModuleList([cards, encoder])
    rewritten = deepcopy(original)
    current_cpu = deepcopy(original)

    def legacy(embedding: torch.nn.Embedding, rows: torch.Tensor) -> torch.Tensor:
        return embedding(rows)

    def empty_slice(embedding: torch.nn.Embedding, rows: torch.Tensor) -> torch.Tensor:
        if rows.numel() == 0:
            return embedding.weight[:0].reshape(*rows.shape, embedding.embedding_dim)
        return embedding(rows)

    losses: list[torch.Tensor] = []
    gradients: list[dict[str, torch.Tensor | None]] = []
    for model, method in (
        (original, legacy),
        (rewritten, empty_slice),
        (current_cpu, None),
    ):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        # Seed momentum, then compare the actual fact loss with missing families.
        sum(parameter.square().sum() for parameter in model.parameters()).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with monkeypatch.context() as patch:
            if method is not None:
                patch.setattr(semantic_cards, "embedding_rows", method)
            records = RecordEmbeddings(model[0], torch.zeros(len(cards.tokens) + 1, 4))
            loss = model[1](facts, torch.ones(1, 2, 4), records).square().mean()
            loss.backward()
        losses.append(loss.detach())
        gradients.append(
            {
                name: None if parameter.grad is None else parameter.grad.clone()
                for name, parameter in model.named_parameters()
            }
        )
        shared_grad = model[0].token_embedding.weight.grad
        assert shared_grad is not None
        if family == "none":
            assert torch.count_nonzero(shared_grad) == 0
        optimizer.step()
        for parameter, state in optimizer.state.items():
            assert int(state["step"]) == (1 if parameter.grad is None else 2)
    for index in (1, 2):
        assert torch.equal(losses[0], losses[index])
        for name, expected in gradients[0].items():
            actual = gradients[index][name]
            if expected is None:
                assert actual is None
            else:
                assert actual is not None and torch.equal(actual, expected), name
        for expected, actual in zip(
            original.parameters(),
            (rewritten, current_cpu)[index - 1].parameters(),
            strict=True,
        ):
            assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_mps_empty_record_families_backward() -> None:
    """Both catalog entry points can produce an empty [0,D] embedding."""
    cards = SemanticCardEncoder("world", 4, records=True).to("mps")
    empty = torch.empty(0, dtype=torch.long, device="mps")
    for family in ("names", "tokens"):
        cards.zero_grad(set_to_none=True)
        values = (
            cards.record_names(RecordNamespace.TOKEN, empty)
            if family == "names"
            else cards.record_tokens(empty, empty)
        )
        assert values.shape == (0, 4) and values.device.type == "mps"
        values.sum().backward()
        gradient = cards.token_embedding.weight.grad
        assert gradient is not None and torch.count_nonzero(gradient) == 0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_mps_w64_single_selected_row_without_program_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two real decisions, original w64 model/inputs, one selected update row.

    Missing program facts reproduced EmbeddingBackward0's empty-placeholder
    assertion. Exact quota avoids the production complete-game collection.
    """
    torch.set_num_threads(1)
    torch.manual_seed(197)
    plan = frozen()[0]["w64"]
    stage = plan.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    assert isinstance(stage.learning, DampedPolicyLearning)
    agent = ActorCritic(ObservationSpace(plan.observation), plan.agent).to("mps")
    collector = SeatRoutedCollector(
        agent.observation_space,
        Match(plan.root_match),
        Reward(RewardConfig()),
        num_envs=2,
        seed=197,
        opponent_mode="self",
        learning_seats="one",
        batch_closure="exact-quota",
        device="mps",
        workers=1,
        deck_source=plan.deck_source,
    )
    try:
        batch = collector.collect(agent, 1, deadline_monotonic=time.perf_counter() + 40)
    finally:
        collector.close()
    assert batch.actions.shape == (1, 2)
    original = damped_policy._targets

    def singleton(
        learner: Learner,
        batch: RolloutBatch,
        learning: DampedPolicyLearning,
        bootstrap_agent: ActorCritic | None = None,
    ) -> damped_policy.DampedTargets:
        targets = original(learner, batch, learning, bootstrap_agent)
        selected = torch.zeros_like(targets.selected)
        selected.flatten()[0] = True
        return damped_policy.DampedTargets(targets.advantages, targets.values, selected)

    empty_calls: list[int] = []

    def checked(embedding: torch.nn.Embedding, rows: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and rows.numel() == 0:
            empty_calls.append(rows.numel())
        return embedding_rows(embedding, rows)

    monkeypatch.setattr(damped_policy, "_targets", singleton)
    monkeypatch.setattr(semantic_cards, "embedding_rows", checked)
    learner = Learner(agent, "mps", minibatch_rows=256)
    result = stages.update_iteration(
        learner, batch, stage.learning, 0, np.random.default_rng(197), iteration=1
    )
    assert empty_calls
    assert result["retained"] == result["optimizer_exposures"] == 1
    assert result["numerical"]["optimizer_steps"] == 1
    _assert_finite(result)
    assert all(torch.isfinite(parameter).all() for parameter in agent.parameters())

    # PPO shares the catalog owner; an empty actor filter still trains its critic.
    empty_calls.clear()
    result = stages.update_iteration(
        learner,
        batch,
        Learning(epochs=1, minibatches=1, filter_scope="actor", min_advantage=1e6),
        0,
        np.random.default_rng(197),
        iteration=1,
    )
    assert empty_calls
    assert result["actor_exposures"] == 0
    assert result["critic_exposures"] == result["optimizer_exposures"] == 2
    assert result["numerical"]["optimizer_steps"] == 1
    assert result["numerical"]["gradient_norm_critic_readout"] > 0
    _assert_finite(result)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="requires Apple MPS"
            ),
        ),
    ],
)
def test_empty_shared_effect_details_backward(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live header/member need not have detail tokens (e.g. a numeric buff)."""
    torch.manual_seed(19)
    values = {
        key: torch.zeros(1, *shape, device=device)
        for key, shape in SHARED_EFFECT_SHAPES.items()
    }
    values["effects"][0, 0, 10] = 1  # Controller object.
    values["effects"][0, 0, 16:18] = 1  # One member, live header.
    values["effect_members"][0, 0] = 1  # Object 1, ordinal 1.
    objects = torch.ones(1, 2, 4, device=device)
    encoder = SharedEffectEncoder(4).to(device)
    before = deepcopy(encoder)
    loss = (
        encoder(
            values["effects"],
            values["effect_details"],
            values["effect_members"],
            objects,
        )
        .square()
        .mean()
    )
    loss.backward()
    assert torch.isfinite(loss)
    for embedding in (encoder.detail, encoder.detail_kind):
        assert embedding.weight.grad is not None
        assert torch.count_nonzero(embedding.weight.grad) == 0
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in encoder.parameters()
    )
    if device == "cpu":
        # Exercise the workaround algebra on CPU against unchanged CPU calls.
        def empty_slice(
            embedding: torch.nn.Embedding, rows: torch.Tensor
        ) -> torch.Tensor:
            if rows.numel() == 0:
                return embedding.weight[:0].reshape(
                    *rows.shape, embedding.embedding_dim
                )
            return embedding(rows)

        monkeypatch.setattr(shared_effects, "embedding_rows", empty_slice)
        expected = (
            before(
                values["effects"],
                values["effect_details"],
                values["effect_members"],
                objects,
            )
            .square()
            .mean()
        )
        expected.backward()
        assert torch.equal(loss, expected)
        for actual, reference in zip(
            encoder.parameters(), before.parameters(), strict=True
        ):
            assert actual.grad is not None and reference.grad is not None
            assert torch.equal(actual.grad, reference.grad)
