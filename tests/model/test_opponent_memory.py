"""Whole-prefix memories reach both heads and survive ordinary policy tooling."""

from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch

import manabot.backends.torch as torch_backend
from manabot.backends.torch.arrays import on_device
from manabot.backends.torch.checkpoint import load_checkpoint, save_checkpoint
from manabot.backends.torch.collection import CollectionSession
from manabot.backends.torch.learner import Learner
from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.env.opponent_memory import blank_memory, encode_records
from manabot.infra.config import ModelConfig, ObservationConfig, RewardConfig
from manabot.training.regime import DampedPolicyLearning
from manabot.training.rollout.collection import SeatRoutedCollector
from manabot.training.updates.damped_policy import update_damped_iteration
import managym
from managym.opponent_memory import (
    DecisionContext,
    OpponentDecisionRecord,
    PublicChoice,
)


def memory_input() -> tuple[ObservationSpace, dict[str, np.ndarray]]:
    space = ObservationSpace(
        ObservationConfig(max_opponent_records=4, max_opponent_details=1024)
    )
    env = managym.Env(seed=111)
    env.reset(Match().to_rust())
    encoded = dict(space.encode(env.observation_for_player(env.current_agent_index())))
    record = OpponentDecisionRecord(
        ordinal=0,
        first_position=0,
        last_position=0,
        count=1,
        kind="Priority",
        context=DecisionContext(
            turn=3,
            phase=1,
            step=3,
            life=(20, 18),
            hand=(6, 3),
            board_counts=((0,) * 8, (0,) * 8),
            mana_sources=[],
        ),
        response=[],
        choice=PublicChoice(
            kind="PassPriority",
            mode=None,
            ability=None,
            named=None,
            name_row=None,
            declared=None,
            references=[],
        ),
    )
    encoded.update(encode_records([record], 4, 1024))
    return space, {key: value[None] for key, value in encoded.items()}


def policy(
    space: ObservationSpace,
    encoder: Literal["transformer", "gru"],
    summaries: int,
    *,
    live: bool = False,
    bucketed: bool = False,
    aggregation: str = "masked_mean",
) -> ActorCritic:
    torch.set_num_threads(1)
    torch.manual_seed(221)
    model = ActorCritic(
        space,
        ModelConfig.model_validate(
            {
                "world": managym.WORLD_SETUP,
                "hidden_dim": 16,
                "opponent_memory": encoder,
                "opponent_memory_summaries": summaries,
                "attention_on": True,
                "attention_layers": 2,
                "policy_film": True,
                "value_aggregation": aggregation,
            }
        ),
        live_tokens=live,
        bucketed_shapes=bucketed,
    )
    # FiLM starts at zero to preserve the incumbent's initial function. Exercise
    # its learned path so objectless action sensitivity cannot pass via focus.
    assert model.policy_film is not None
    with torch.no_grad():
        model.policy_film.weight.normal_(0, 0.1)
    return model


def forward(
    model: ActorCritic, obs: dict[str, np.ndarray]
) -> tuple[torch.Tensor, torch.Tensor]:
    return model.forward_distribution(on_device(model.prepare(obs), "cpu"))


@pytest.mark.parametrize("encoder", ["transformer", "gru"])
@pytest.mark.parametrize("summaries", [1, 4])
def test_opponent_memory_changes_probabilities_value_and_both_encoder_gradients(
    encoder: Literal["transformer", "gru"], summaries: int
) -> None:
    space, obs = memory_input()
    model = policy(space, encoder, summaries)
    # Make every legal action objectless. Their distinct native action features
    # remain, proving the actor can read memory through the selected board join.
    obs["action_focus"] = np.full_like(obs["action_focus"], -1)
    logits, value = forward(model, obs)
    blank_logits, blank_value = forward(model, blank_memory(obs))
    assert not torch.allclose(
        logits.softmax(-1), blank_logits.softmax(-1), atol=1e-8, rtol=0
    )
    assert not torch.allclose(value, blank_value, atol=1e-8, rtol=0)
    assert model.opponent_memory is not None
    for loss in (logits.softmax(-1)[:, 0].sum(), value.sum()):
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        grads = [
            p.grad for p in model.opponent_memory.parameters() if p.grad is not None
        ]
        assert grads and all(torch.isfinite(grad).all() for grad in grads)
        assert sum(float(grad.abs().sum()) for grad in grads) > 0
        temporal = (
            model.opponent_memory.gru
            if encoder == "gru"
            else model.opponent_memory.transformer
        )
        assert temporal is not None
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in temporal.parameters()
        )


@pytest.mark.parametrize("encoder", ["transformer", "gru"])
@pytest.mark.parametrize("summaries", [1, 4])
@pytest.mark.parametrize(
    "aggregation",
    ["masked_mean", "value_token", "historical_mean", "attention_readout"],
)
def test_opponent_memory_live_compact_buckets_and_empty_rows_agree(
    encoder: Literal["transformer", "gru"], summaries: int, aggregation: str
) -> None:
    space, one = memory_input()
    blank = blank_memory(one)
    mixed = {key: np.concatenate((value, blank[key]), 0) for key, value in one.items()}
    reference = policy(space, encoder, summaries, aggregation=aggregation).eval()
    expected = forward(reference, mixed)
    for live, bucketed in ((False, True), (True, False), (True, True)):
        candidate = policy(
            space,
            encoder,
            summaries,
            live=live,
            bucketed=bucketed,
            aggregation=aggregation,
        ).eval()
        candidate.load_state_dict(reference.state_dict())
        actual = forward(candidate, mixed)
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    prepared = on_device(reference.prepare(mixed), "cpu")
    assert prepared.memory is not None and reference.opponent_memory is not None
    summaries_tensor = reference.opponent_memory(
        prepared.memory,
        reference.semantic_cards.token_embedding,
        reference._gather_object_embeddings(prepared).programs,
        reference.name_candidates.embedding,
    )
    assert torch.count_nonzero(summaries_tensor[1]) == 0
    # Compaction is the ordinary CPU inference path, with summaries restored
    # separately from all focus coordinates and the trailing critic token.
    compact = reference.forward_distribution(
        on_device(reference.prepare(mixed, inference=True), "cpu")
    )
    for a, b in zip(compact, expected, strict=True):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("encoder", ["transformer", "gru"])
def test_opponent_memory_checkpoint_round_trip(
    tmp_path: Path, encoder: Literal["transformer", "gru"]
) -> None:
    space, obs = memory_input()
    model = policy(space, encoder, 4).eval()
    expected = forward(model, obs)
    path = tmp_path / "memory.safetensors"
    save_checkpoint(model, space, path, player_configs=Match().to_rust())
    restored, restored_space = load_checkpoint(
        str(path), live_tokens=True, bucketed_shapes=True
    )
    assert restored_space == space
    assert restored.config.opponent_memory == encoder
    for a, b in zip(forward(restored.eval(), obs), expected, strict=True):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_opponent_memory_rejects_missing_companions_and_unowned_details() -> None:
    space, obs = memory_input()
    model = policy(space, "gru", 1)
    with pytest.raises(ValueError, match="companion"):
        model.prepare(
            {key: value for key, value in obs.items() if key != "opponent_details"}
        )
    obs["opponent_records"][0, 0, 14] = 1
    with pytest.raises(ValueError, match="partition"):
        model.prepare(obs)


def test_opponent_memory_rejects_noncanonical_words_with_the_same_catalog_join() -> (
    None
):
    space, obs = memory_input()
    model = policy(space, "gru", 1)
    header = obs["opponent_records"][0, 0]
    # Catalog keys keep the low u32, so adding 2**32 aliases the admitted
    # name. Each float is still exact, but the high word is no longer u16.
    header[12] += 65536
    with pytest.raises(ValueError, match="integer words"):
        model.prepare(obs)


@pytest.mark.parametrize("encoder", ["transformer", "gru"])
def test_opponent_memory_collects_bootstraps_and_learns_from_real_prefixes(
    encoder: Literal["transformer", "gru"],
) -> None:
    # Explicit fixture capacities, not the unmeasured pilot's capacity choice.
    space = ObservationSpace(
        ObservationConfig(max_opponent_records=64, max_opponent_details=65536)
    )
    model = policy(space, encoder, 4)
    collector = SeatRoutedCollector(
        space,
        Match(),
        Reward(RewardConfig()),
        num_envs=1,
        learning_seats="one",
        batch_closure="exact-quota",
        numerics=CollectionSession(device="cpu", seed=111, precision="float32"),
    )
    try:
        batch = collector.collect(model, 16)
        assert batch.actions.size == 16
        assert (
            batch.observations(np.arange(batch.actions.size))
            .observation["opponent_records"][..., 17]
            .any()
        )
        assert model.opponent_memory is not None
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.opponent_memory.named_parameters()
        }
        learner = Learner(model, "cpu", minibatch_rows=4)
        diagnostics = update_damped_iteration(
            torch_backend,
            learner,
            batch,
            DampedPolicyLearning(
                gradient="damped_policy",
                filter_scope="actor",
                advantage_quantile=0,
                min_advantage=0,
            ),
            1,
        )
        assert diagnostics["optimizer_exposures"] == 16
        assert any(
            not torch.equal(before[name], parameter)
            for name, parameter in model.opponent_memory.named_parameters()
        )
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    finally:
        collector.close()


def test_no_history_control_retains_logs_without_using_them(tmp_path: Path) -> None:
    space, obs = memory_input()
    config = ModelConfig(world=managym.WORLD_SETUP, hidden_dim=16)
    torch.set_num_threads(1)
    torch.manual_seed(23)
    model = ActorCritic(space, config).eval()
    assert model.opponent_memory is None
    assert model.prepare(obs).memory is None
    blank = blank_memory(obs)
    for actual, expected in zip(
        forward(model, obs), forward(model, blank), strict=True
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    # Retaining the shared native input must not grow the incumbent's vocabulary
    # or parameters, or change its current-state predictions.
    ordinary = {
        k: v
        for k, v in obs.items()
        if k not in ("opponent_records", "opponent_details")
    }
    torch.manual_seed(23)
    incumbent = ActorCritic(ObservationSpace(), config).eval()
    assert list(model.state_dict()) == list(incumbent.state_dict())
    for actual, expected in zip(
        forward(model, obs), forward(incumbent, ordinary), strict=True
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    path = tmp_path / "control.safetensors"
    save_checkpoint(model, space, path, player_configs=Match().to_rust())
    restored, restored_space = load_checkpoint(str(path))
    assert restored_space == space
    for actual, expected in zip(
        forward(restored.eval(), obs), forward(model, obs), strict=True
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    with pytest.raises(ValueError, match="contract"):
        model.prepare(ordinary)
    malformed = {**obs, "opponent_details": obs["opponent_details"][:, :1]}
    with pytest.raises(ValueError):
        model.prepare(malformed)


@pytest.mark.parametrize("field", ["opponent_records", "opponent_details"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.5])
@pytest.mark.parametrize("padding", [False, True])
def test_memory_admission_rejects_invalid_live_and_padding_values(
    field: str, value: float, padding: bool
) -> None:
    from manabot.policy.opponent_memory import admit_memory

    _, obs = memory_input()
    obs[field][0, -1 if padding else 0, 0] = value
    with pytest.raises(ValueError):
        admit_memory(obs)
