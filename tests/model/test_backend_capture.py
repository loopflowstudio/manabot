"""Real captures replay the production update's flattened targets and loss."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from manabot.backends import resolve
from manabot.backends.parity import (
    check,
    installed_backends,
    main,
    minibatch_loss,
    minibatches,
)
from manabot.backends.torch.capture import capture
from manabot.backends.torch.checkpoint import load_checkpoint
from manabot.backends.torch.policy import ActorCritic
from manabot.infra.config import MatchConfig, ModelConfig, ObservationConfig
from manabot.infra.identity import file_sha256
from manabot.policy.checkpoint import from_torch_state
from manabot.policy.prepare import Prepared
from manabot.training.learning import DampedPolicyLearning
from manabot.training.loops.self_play import CLOCKS, SelfPlay
import managym
from tests.fixtures.training_records import recording


@pytest.fixture(scope="module")
def captured(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("backend-capture")
    settings = SelfPlay(
        match=MatchConfig(
            hero_deck={"Mountain": 8, "Gray Ogre": 8},
            villain_deck={"Forest": 8, "Llanowar Elves": 8},
        ),
        # The framework container implements input v9 only.
        observation=ObservationConfig(policy_input_version=9),
        model=ModelConfig(
            hidden_dim=8,
            num_attention_heads=2,
            value_kind="scalar",
            value_aggregation="masked_mean",
            world=managym.WORLD_SETUP,
        ),
        streams=2,
        transitions=4,
        bucketed_shapes=True,
        learning=DampedPolicyLearning(
            gradient="damped_policy",
            min_advantage=0,
            advantage_quantile=0,
            max_grad_norm=0.01,
        ),
    )
    out = tmp_path / "capture"
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ETUDE_OUT", str(tmp_path / "run"))
        with recording(
            experiment="capture-test",
            clocks=CLOCKS,
            seed=197,
            directory=tmp_path / "track",
        ) as run:
            capture(out, settings, run)
    return out


def test_capture_replays_both_seat_update(captured: Path) -> None:
    out = captured
    meta = json.loads((out / "capture.json").read_text())
    assert meta["calls"] > 0 and meta["minibatches"] > 1
    assert meta["engine"] >= 0
    agent, space = load_checkpoint(str(out / "initial.pt"), bucketed_shapes=True)
    weights = from_torch_state(
        agent.config,
        space.config,
        {k: v.numpy() for k, v in agent.state_dict().items()},
        parameter_names=dict(agent.named_parameters()),
        readings=agent.readings,
    )
    policy, learner = resolve("torch").load(weights)
    rows = 0
    for obs, targets, settings, recorded in minibatches(out, None):
        assert targets.actions.shape == targets.advantages.shape
        assert targets.value_targets.shape == targets.actions.shape
        assert len(targets.actions) <= 2
        loss = minibatch_loss(
            resolve("torch"),
            policy,
            learner,
            policy.prepare(obs),
            targets,
            settings,
        )
        np.testing.assert_allclose(
            float(loss.detach()), recorded["loss"], atol=1e-5, rtol=1e-4
        )
        learner.set_learning_rate(settings.learning_rate)
        norm = learner.step(loss, settings.max_grad_norm)
        assert torch.isfinite(norm)
        rows += len(targets.actions)
    assert rows == meta["learner_decisions"]


@pytest.mark.parametrize(
    ("backend", "device", "precision"),
    [
        ("torch", "cpu", "float32"),
        ("torch", "mps", "bfloat16"),
        ("mlx", "mps", "bfloat16"),
    ],
)
def test_parity_cli_keeps_execution_target_after_step(
    captured: Path, tmp_path: Path, backend: str, device: str, precision: str
) -> None:
    if backend not in installed_backends():
        pytest.skip(f"requires {backend}")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires MPS")
    checkpoint = captured / "initial.pt"
    out = tmp_path / "report.json"
    threads = torch.get_num_threads()
    code = main(
        [
            str(checkpoint),
            "--capture",
            str(captured),
            "--backend",
            backend,
            "--device",
            device,
            "--precision",
            precision,
            "--bucketed",
            "--threads",
            "1",
            "--minibatches",
            "2",
            "--collection",
            "1",
            "--json",
            str(out),
        ]
    )
    receipt = json.loads(out.read_text())[backend]
    assert receipt["execution_error"] is None
    assert receipt["actual_device"] == receipt["requested_device"] == device
    assert receipt["observed_forward_dtype"] == precision
    assert receipt["bucketed_shapes"] and receipt["step_checked"]
    assert (
        receipt["minibatch_indices"] == [0, 1] and receipt["minibatches_checked"] == 2
    )
    assert receipt["collection_indices"] == [0] and receipt["collection_checked"] == 1
    assert receipt["checkpoint_sha256"] == file_sha256(checkpoint)
    assert receipt["threads"] == 1 and torch.get_num_threads() == threads
    assert code == int(bool(receipt["failures"]))
    if precision == "float32":
        assert not receipt["failures"]


def test_empty_collection_cannot_pass(captured: Path) -> None:
    report = check(
        captured / "initial.pt", captured, 1, 0, backend_name="torch", threads=1
    )
    assert report.failures() == ["collection samples"]


def test_nonfinite_collection_output_cannot_pass(
    captured: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    forward = ActorCritic.forward_distribution

    def nonfinite(
        self: ActorCritic, batch: Prepared[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, values = forward(self, batch)
        if self.bucketed_shapes and not self.training:
            values = torch.full_like(values, float("nan"))
        return logits, values

    monkeypatch.setattr(ActorCritic, "forward_distribution", nonfinite)
    report = check(
        captured / "initial.pt",
        captured,
        1,
        1,
        backend_name="torch",
        bucketed_shapes=True,
        threads=1,
    )
    assert "nonfinite collection value" in str(report.execution_error)
    assert "execution" in report.failures()
