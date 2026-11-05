"""Typed state treatments preserve native decisions and train/export normally."""

from copy import deepcopy
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch

from experiments.state_representation_v7 import recipe
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.model.actor_critic import ActorCritic
from manabot.model.architecture import architecture_identity
from manabot.model.checkpoint import load_checkpoint, save_checkpoint
from manabot.model.records import RecordReducer
from manabot.model.state_records import SharedFactEncoder, TypedContinuationEncoder
from manabot.training.execution import execute_regime
from manabot.verify.representation import shared_effect_root
from manabot.verify.store import VerifyStore
from manabot.verify.text_observation import collision_branches, native_fixtures
import managym

Arm = Literal["typed_pool", "typed_gru"]


def _batch(encoded: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {key: torch.from_numpy(value)[None] for key, value in encoded.items()}


@pytest.mark.parametrize("arm", ("typed_pool", "typed_gru"))
def test_native_choices_remain_distinct_and_reload(tmp_path: Path, arm: Arm) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(73200)
    space = ObservationSpace(ObservationConfig(policy_input_version=5))
    agent = ActorCritic(
        space, ModelConfig(hidden_dim=16, state_encoder=arm, policy_film=True)
    )
    branches = collision_branches()
    encoded = [
        space.encode(env.observation_for_player(env.current_agent_index()))
        for env in branches
    ]
    outputs = [agent(_batch(obs)) for obs in encoded]
    assert not torch.equal(outputs[0][0], outputs[1][0])
    assert not torch.equal(outputs[0][1], outputs[1][1])
    (
        outputs[0][1].sum()
        + outputs[0][0][_batch(encoded[0])["actions_valid"].bool()].sum()
    ).backward()
    assert agent.continuation is not None
    assert (
        sum(
            p.grad.abs().sum()
            for p in agent.continuation.parameters()
            if p.grad is not None
        )
        > 0
    )
    path = tmp_path / "policy.pt"
    players = [managym.PlayerConfig(name, {"Mountain": 40}) for name in ("a", "b")]
    save_checkpoint(agent, space, path, player_configs=players)
    restored, _ = load_checkpoint(str(path))
    assert restored.config.state_encoder == arm
    for expected, actual in zip(outputs[0], restored(_batch(encoded[0])), strict=True):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    payload = torch.load(path, weights_only=False)
    payload["hypers"]["agent_hypers"]["state_encoder"] = "rows"
    torch.save(payload, path)
    with pytest.raises((ValueError, RuntimeError)):
        load_checkpoint(str(path))


@pytest.mark.parametrize("arm", ("typed_pool", "typed_gru"))
def test_shared_facts_preserve_membership_multiplicity_and_padding(arm: Arm) -> None:
    torch.manual_seed(73200)
    encoder = SharedFactEncoder(8, arm)
    objects = torch.randn(1, 3, 8, requires_grad=True)
    rows = ObservationConfig(policy_input_version=5).fact_rows
    facts = torch.zeros(1, rows, 12)
    # Same payload, distinct objects; a duplicate on object one is significant.
    facts[0, :3] = torch.tensor(
        [[obj, 6, 1, 2, 3, 4, 0.1, 0.2, 3, 7, 11, 1] for obj in (1, 2, 1)]
    )
    expected = encoder(facts, objects)
    delta = expected - objects
    torch.testing.assert_close(delta[:, 0], 2 * delta[:, 1])
    assert delta[:, 2].abs().sum() == 0
    permuted = facts[:, torch.randperm(rows)]
    torch.testing.assert_close(expected, encoder(permuted, objects))
    poisoned = facts.clone()
    poisoned[0, 3:, :11] = float("nan")
    torch.testing.assert_close(expected, encoder(poisoned, objects))
    changed = facts.clone()
    changed[0, 0, 8] = 1
    assert not torch.equal(expected, encoder(changed, objects))
    expected.sum().backward()
    assert encoder.source.weight.grad is not None
    assert encoder.source.weight.grad.abs().sum() > 0
    changed[0, 0, 0] = 4
    with pytest.raises(ValueError, match="object"):
        encoder(changed, objects)


@pytest.mark.parametrize("arm", ("typed_pool", "typed_gru"))
def test_native_viewer_projection_and_object_joins(arm: Arm) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(73200)
    space = ObservationSpace(ObservationConfig(policy_input_version=5))
    agent = ActorCritic(space, ModelConfig(hidden_dim=16, state_encoder=arm))
    for env in native_fixtures().values():
        actor = env.current_agent_index()
        for viewer in (actor, 1 - actor):
            obs = env.observation_for_player(viewer)
            native = env.encode_observation(obs, 5)
            reference = space.encode(obs)
            for key in native:
                np.testing.assert_array_equal(native[key], reference[key])
            if viewer == actor:
                logits, value = agent(_batch(native))
                assert torch.isfinite(
                    logits[_batch(native)["actions_valid"].bool()]
                ).all()
                assert torch.isfinite(value).all()


def test_historical_defaults_and_admission() -> None:
    assert "state_encoder" not in ModelConfig().model_dump()
    space = ObservationSpace(ObservationConfig(policy_input_version=5))
    identities = {
        architecture_identity(ModelConfig(state_encoder=arm), space)
        for arm in ("rows", "typed_pool", "typed_gru")
    }
    assert len(identities) == 3
    with pytest.raises(ValueError, match="v5"):
        ActorCritic(
            ObservationSpace(ObservationConfig(policy_input_version=4)),
            ModelConfig(state_encoder="typed_gru"),
        )


@pytest.mark.parametrize("arm", ("typed_pool", "typed_gru"))
def test_treatment_trains_with_existing_executor(tmp_path: Path, arm: Arm) -> None:
    plan = recipe(arm, smoke=True)
    with VerifyStore(tmp_path / "training.sqlite") as store:
        run = execute_regime(plan, 73200, tmp_path / "run", store)
    assert run.status == "completed"
    stage = run.stages[0]
    assert stage.updates == 1 and stage.learner_transitions > 0
    assert stage.optimizer_exposures > 0
    restored, _ = load_checkpoint(stage.artifacts["raw"]["path"])
    assert restored.config.state_encoder == arm


@pytest.mark.parametrize("arm", ("typed_pool", "typed_gru"))
@pytest.mark.parametrize(
    "frame_kind,edge_kind,column",
    [(2, 7, 24), (10, 8, 24), (10, 9, 28), (11, 1, 7), (12, 1, 7)],
    ids=("inherited", "remembered", "frequency", "turn-order", "combat-order"),
)
def test_t6_captures_lifetimes_and_schedules_reach_typed_context(
    arm: Arm, frame_kind: int, edge_kind: int, column: int
) -> None:
    torch.manual_seed(73200)
    encoder = TypedContinuationEncoder(8, arm)
    objects = torch.randn(1, 2, 8, requires_grad=True)
    definitions = torch.randn(1, 64, 2, 8, requires_grad=True)
    rows = torch.zeros(1, 64, 49)
    rows[0, 0, :7] = torch.tensor([1, 1, 0, frame_kind, 1, 1, 1])
    rows[0, 0, 31] = 1
    subject_row = 0 if edge_kind == 1 else 1
    if subject_row:
        rows[0, 1] = rows[0, 0]
        rows[0, 1, 0] = edge_kind
    rows[0, subject_row, column] = 1
    summary, frames = encoder(rows, objects, definitions)
    changed = rows.clone()
    changed[0, subject_row, column] = 2
    other_summary, other_frames = encoder(changed, objects, definitions)
    assert not torch.allclose(summary, other_summary)
    assert not torch.allclose(frames[:, 0], other_frames[:, 0])
    # Moving storage rows preserves the frame and its joined subjects.
    permuted = encoder(rows.flip(1), objects, definitions.flip(1))
    torch.testing.assert_close(summary, permuted[0])
    torch.testing.assert_close(frames, permuted[1])
    summary.sum().backward()
    assert objects.grad is not None and objects.grad.abs().sum() > 0
    assert definitions.grad is not None and definitions.grad.abs().sum() > 0


@pytest.mark.parametrize("arm", ("typed_pool", "typed_gru"))
def test_frame_and_object_relabeling_preserves_joins(arm: Arm) -> None:
    torch.manual_seed(73200)
    encoder = TypedContinuationEncoder(8, arm)
    objects = torch.randn(1, 3, 8, requires_grad=True)
    definitions = torch.randn(1, 64, 2, 8, requires_grad=True)
    rows = torch.zeros(1, 64, 49)
    for i, (handle, parent, kind, source) in enumerate(((1, 0, 7, 1), (2, 1, 8, 2))):
        rows[0, i, :7] = torch.tensor([1, handle, parent, kind, 1, 1, source])
        rows[0, i, 31] = 1
    rows[0, 2] = rows[0, 1]
    rows[0, 2, 0] = 2
    rows[0, 2, 22] = rows[0, 2, 29] = 1
    rows[0, 2, 24] = 3
    expected_context, expected_frames = encoder(rows, objects, definitions)
    renamed = rows.clone()
    renamed[0, :3, 1] = torch.tensor([9, 4, 4])
    renamed[0, :3, 2] = torch.tensor([0, 9, 9])
    # Reverse objects, repairing source and target references.
    for col in (6, 24):
        named = renamed[..., col] > 0
        renamed[..., col][named] = 4 - renamed[..., col][named]
    actual_context, actual_frames = encoder(
        renamed.flip(1), objects.flip(1), definitions.flip(1)
    )
    torch.testing.assert_close(actual_context, expected_context)
    torch.testing.assert_close(actual_frames[:, [8, 3]], expected_frames[:, :2])
    expected_context.sum().backward()
    assert objects.grad is not None and (objects.grad.abs().sum(-1) > 0).all()
    assert definitions.grad is not None and definitions.grad.abs().sum() > 0
    renamed[0, 0, 2] = 4
    with pytest.raises(ValueError, match="cyclic"):
        encoder(renamed, objects, definitions)


@pytest.mark.parametrize("arm", ("typed_pool", "typed_gru"))
def test_stack_order_is_semantic_but_row_storage_order_is_not(arm: Arm) -> None:
    torch.manual_seed(73200)
    encoder = TypedContinuationEncoder(8, arm)
    objects = torch.randn(1, 2, 8)
    definitions = torch.randn(1, 64, 2, 8)
    rows = torch.zeros(1, 64, 49)
    for i in range(2):
        rows[0, i, :7] = torch.tensor([1, i + 1, 0, 1, 1, 1, i + 1])
        rows[0, i, 31] = 1
    summary, frames = encoder(rows, objects, definitions)
    permuted = encoder(rows.flip(1), objects, definitions.flip(1))
    torch.testing.assert_close(permuted[0], summary)
    torch.testing.assert_close(permuted[1], frames)
    rows[0, :2, 1] = torch.tensor([2, 1])
    changed, _ = encoder(rows, objects, definitions)
    assert not torch.allclose(changed, summary, atol=1e-7, rtol=1e-7)


def test_unused_continuation_parameters_keep_absent_gradients() -> None:
    encoder = TypedContinuationEncoder(8, "typed_pool")
    rows = torch.zeros(2, 64, 49)
    rows[:, 0, :7] = torch.tensor([1, 1, 0, 7, 1, 1, 1])
    rows[:, 0, 31] = 1
    summary, _ = encoder(rows, torch.randn(2, 2, 8), torch.randn(2, 64, 2, 8))
    summary.square().sum().backward()
    assert all(p.grad is None for p in encoder.edge.parameters())
    assert all(p.grad is None for p in encoder.structures[0].parameters())
    assert all(p.grad is None for p in encoder.structures[1].parameters())
    assert encoder.structures[2].weight.grad is not None


def test_native_shared_effects_keep_timestamps_and_reject_v5_overflow() -> None:
    env = shared_effect_root(6)
    space = ObservationSpace(ObservationConfig(policy_input_version=5))
    obs = env.observation_for_player(env.current_agent_index())
    native = env.encode_observation(obs, 5)
    expected = space.encode(obs)
    for key in native:
        np.testing.assert_array_equal(native[key], expected[key])
    facts = native["facts"][native["facts"][:, 11] == 1]
    assert len(facts) == 120
    assert len(np.unique(facts[:, 1:11], axis=0)) == 12
    assert len(np.unique(facts[:, 9])) == 6
    overflowing = shared_effect_root(7)
    obs = overflowing.observation_for_player(overflowing.current_agent_index())
    for encode in (space.encode, lambda obs: overflowing.encode_observation(obs, 5)):
        with pytest.raises((ValueError, RuntimeError), match="facts"):
            encode(obs)


@pytest.mark.parametrize("arm", ("typed_pool", "typed_gru"))
def test_flat_records_preserve_sequence_order_and_gradients(arm: Arm) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(91)
    reducer = RecordReducer(8, arm)
    reference = deepcopy(reducer)
    lengths = [3, 1, 5, 3, 2]
    tokens = torch.randn(sum(lengths), 8, requires_grad=True)
    separate = tokens.detach().clone().requires_grad_()
    actual = reducer.reduce_tokens(tokens, lengths)
    expected_rows: list[torch.Tensor] = []
    for sequence in separate.split(lengths):
        if reference.gru is None:
            pooled = sequence.sum(0)
        else:
            _, hidden = reference.gru(sequence.unsqueeze(0))
            pooled = hidden[-1, 0]
        expected_rows.append(reference.output(torch.tanh(pooled)))
    expected = torch.stack(expected_rows)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(tokens.grad, separate.grad)
    for parameter, other in zip(
        reducer.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(parameter.grad, other.grad)


@pytest.mark.parametrize("arm", ("typed_pool", "typed_gru"))
def test_continuation_batches_preserve_independent_frames_and_gradients(
    arm: Arm,
) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(93)
    encoder = TypedContinuationEncoder(8, arm)
    reference = deepcopy(encoder)
    objects = torch.randn(3, 3, 8, requires_grad=True)
    definitions = torch.randn(3, 64, 2, 8, requires_grad=True)
    relations = torch.zeros(3, 64, 49)
    for batch in (0, 2):
        for row, (handle, parent, kind) in enumerate(((4, 0, 1), (9, 4, 7))):
            relations[batch, row, :7] = torch.tensor([1, handle, parent, kind, 1, 1, 1])
            relations[batch, row, 31] = 1
        relations[batch, 2] = relations[batch, 1]
        relations[batch, 2, 0] = 2
        relations[batch, 2, 24] = 3
    relations[1, :, :31] = float("nan")
    relations.requires_grad_()
    copies = [
        x.detach().clone().requires_grad_() for x in (relations, objects, definitions)
    ]
    actual = encoder(relations, objects, definitions)
    each = [reference(*(x[i : i + 1] for x in copies)) for i in range(3)]
    expected = tuple(torch.cat([item[column] for item in each]) for column in (0, 1))
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b)
    sum(x.square().sum() for x in actual).backward()
    sum(x.square().sum() for x in expected).backward()
    for a, b in zip((relations, objects, definitions), copies, strict=True):
        torch.testing.assert_close(a.grad, b.grad)
    for parameter, other in zip(
        encoder.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(parameter.grad, other.grad)
