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
from manabot.env.decision_memory import blank_memory, encode_records
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig, RewardConfig
from manabot.training.regime import DampedPolicyLearning
from manabot.training.rollout.collection import SeatRoutedCollector
from manabot.training.updates.damped_policy import update_damped_iteration
import managym
from managym.decision_memory import (
    DecisionContext,
    DecisionRecord,
    DecisionReference,
    PublicChoice,
)


def memory_input() -> tuple[ObservationSpace, dict[str, np.ndarray]]:
    space = ObservationSpace(
        ObservationConfig(
            max_decision_records=4,
            max_decision_reference_details=1024,
            max_decision_references=1024,
            max_decision_members=16384,
        )
    )
    env = managym.Env(seed=111)
    env.reset(Match().to_rust())
    encoded = dict(space.encode(env.observation_for_player(env.current_agent_index())))
    record = DecisionRecord(
        actor=2,
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
    encoded.update(encode_records([record], 4, 1024, 1024, 16384))
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
                "decision_memory": encoder,
                "decision_memory_summaries": summaries,
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
        for layer in (model.attention, *model.extra_attention):
            assert layer.memory_gate is not None
            layer.memory_gate.fill_(0.5)
    return model


def forward(
    model: ActorCritic, obs: dict[str, np.ndarray]
) -> tuple[torch.Tensor, torch.Tensor]:
    return model.forward_distribution(on_device(model.prepare(obs), "cpu"))


@pytest.mark.parametrize("encoder", ["transformer", "gru"])
@pytest.mark.parametrize("summaries", [1, 4])
def test_decision_memory_changes_probabilities_value_and_both_encoder_gradients(
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
    assert model.decision_memory is not None
    for loss in (logits.softmax(-1)[:, 0].sum(), value.sum()):
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        grads = [
            p.grad for p in model.decision_memory.parameters() if p.grad is not None
        ]
        assert grads and all(torch.isfinite(grad).all() for grad in grads)
        assert sum(float(grad.abs().sum()) for grad in grads) > 0
        temporal = (
            model.decision_memory.gru
            if encoder == "gru"
            else model.decision_memory.transformer
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
def test_decision_memory_live_compact_buckets_and_empty_rows_agree(
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
    assert prepared.memory is not None and reference.decision_memory is not None
    summaries_tensor = reference.decision_memory(
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
def test_decision_memory_checkpoint_round_trip(
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
    assert restored.config.decision_memory == encoder
    for a, b in zip(forward(restored.eval(), obs), expected, strict=True):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_decision_memory_rejects_missing_companions_and_unowned_details() -> None:
    space, obs = memory_input()
    model = policy(space, "gru", 1)
    with pytest.raises(ValueError, match="companion"):
        model.prepare(
            {
                key: value
                for key, value in obs.items()
                if key != "decision_reference_details"
            }
        )
    obs["decision_records"][0, 0, 14] = 1
    with pytest.raises(ValueError, match="partition"):
        model.prepare(obs)


def test_decision_memory_rejects_noncanonical_words_with_the_same_catalog_join() -> (
    None
):
    space, obs = memory_input()
    model = policy(space, "gru", 1)
    header = obs["decision_records"][0, 0]
    # Catalog keys keep the low u32, so adding 2**32 aliases the admitted
    # name. Each float is still exact, but the high word is no longer u16.
    header[12] += 65536
    with pytest.raises(ValueError, match="integer words"):
        model.prepare(obs)


@pytest.mark.parametrize("encoder", ["transformer", "gru"])
def test_decision_memory_collects_bootstraps_and_learns_from_real_prefixes(
    encoder: Literal["transformer", "gru"],
) -> None:
    # Explicit fixture capacities, not the unmeasured pilot's capacity choice.
    space = ObservationSpace(
        ObservationConfig(
            max_decision_records=64,
            max_decision_reference_details=65536,
            max_decision_references=1024,
            max_decision_members=16384,
        )
    )
    model = policy(space, encoder, 1)
    with torch.no_grad():
        for layer in (model.attention, *model.extra_attention):
            assert layer.memory_gate is not None
            layer.memory_gate.zero_()
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
            .observation["decision_records"][..., 19]
            .any()
        )
        assert model.decision_memory is not None
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.decision_memory.named_parameters()
        }
        learner = Learner(model, "cpu", minibatch_rows=16)
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
        assert all(
            torch.equal(before[name], parameter)
            for name, parameter in model.decision_memory.named_parameters()
        )
        assert any(value != 0 for value in model.memory_gates().values())
        update_damped_iteration(
            torch_backend,
            learner,
            batch,
            DampedPolicyLearning(
                gradient="damped_policy",
                filter_scope="actor",
                advantage_quantile=0,
                min_advantage=0,
            ),
            2,
        )
        assert any(
            not torch.equal(before[name], parameter)
            for name, parameter in model.decision_memory.named_parameters()
        )
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    finally:
        collector.close()


def test_control_retains_logs_without_using_them(tmp_path: Path) -> None:
    space, obs = memory_input()
    config = ModelConfig(world=managym.WORLD_SETUP, hidden_dim=16)
    torch.set_num_threads(1)
    torch.manual_seed(23)
    model = ActorCritic(space, config).eval()
    assert model.decision_memory is None
    assert model.prepare(obs).memory is None
    blank = blank_memory(obs)
    for actual, expected in zip(
        forward(model, obs), forward(model, blank), strict=True
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    # A log-enabled trunk reserves the exact vocabulary read by its branches.
    # The true history-disabled incumbent remains a separate throughput control.
    ordinary = {k: v for k, v in obs.items() if not k.startswith("decision_")}
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
    malformed = {
        **obs,
        "decision_reference_details": obs["decision_reference_details"][:, :1],
    }
    with pytest.raises(ValueError):
        model.prepare(malformed)


@pytest.mark.parametrize(
    "field",
    [
        "decision_records",
        "decision_reference_details",
        "decision_references",
        "decision_members",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.5])
@pytest.mark.parametrize("padding", [False, True])
def test_memory_admission_rejects_invalid_live_and_padding_values(
    field: str, value: float, padding: bool
) -> None:
    from manabot.policy.decision_memory import admit_memory

    _, obs = _shared_reference_input()
    obs[field][0, -1 if padding else 0, 0] = value
    with pytest.raises(ValueError):
        admit_memory(obs)


def _shared_reference_input() -> tuple[ObservationSpace, dict[str, np.ndarray]]:
    space, obs = memory_input()
    reference = DecisionReference(
        role=2,
        definition=None,
        appearance=None,
        name=None,
        description="",
        power=1,
        toughness=2,
        tapped=False,
        ready=True,
        attacking=False,
        blocking=False,
        program=[(4, 1), (4, 7)],
        mana=[(0, 0, 0, 1, 0, 0)],
    )
    record = DecisionRecord(
        actor=2,
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
            mana_sources=[reference],
        ),
        response=[reference, reference],
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
    obs.update(
        {
            key: value[None]
            for key, value in encode_records([record], 4, 1024, 1024, 16384).items()
        }
    )
    return space, obs


def test_decision_memory_snapshot_routes_preserve_groups_repetitions_and_empty_viewer() -> (
    None
):
    space, one = _shared_reference_input()
    blank = blank_memory(one)
    raw = {
        key: np.concatenate((value, blank[key], value), 0) for key, value in one.items()
    }
    model = policy(space, "gru", 1)
    rows = model.prepare(raw).memory
    assert rows is not None
    assert len(rows.reference_numbers) == 2
    np.testing.assert_array_equal(rows.member_references, [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(rows.member_groups, [0, 1, 1, 3, 4, 4])
    np.testing.assert_array_equal(rows.record_batches, [0, 2])
    assert len(rows.detail_references) == 16
    np.testing.assert_array_equal(rows.detail_references, [0] * 8 + [1] * 8)


@pytest.mark.parametrize(
    "change", ["numeric_binding", "reference_binding", "group", "repetition"]
)
def test_decision_memory_distinguishes_previously_colliding_bindings(
    change: str,
) -> None:
    space, original = _shared_reference_input()
    changed = {key: value.copy() for key, value in original.items()}
    if change == "numeric_binding":
        details = changed["decision_reference_details"]
        details[0, :2, 2:4] = details[0, 1::-1, 2:4]
    elif change == "reference_binding":
        references = changed["decision_references"]
        # Swap bound power and toughness words, preserving their numeric sum.
        references[0, 0, 2:6] = references[0, 0, [4, 5, 2, 3]]
    elif change == "group":
        changed["decision_records"][0, 0, 15:18] = [0, 3, 0]
        changed["decision_members"][0, :3, 1] = [0, 1, 2]
    else:
        changed["decision_records"][0, 0, 16] = 1
        changed["decision_members"][0, 2] = 0
    model = policy(space, "gru", 1)

    def summary(raw: dict[str, np.ndarray]) -> torch.Tensor:
        prepared = on_device(model.prepare(raw), "cpu")
        assert prepared.memory is not None and model.decision_memory is not None
        state = model._gather_object_embeddings(prepared)
        return model.decision_memory(
            prepared.memory,
            model.semantic_cards.token_embedding,
            model.name_candidates.public_definitions(state.programs),
            model.name_candidates.embedding,
        )

    assert not torch.allclose(summary(original), summary(changed), atol=1e-7, rtol=0)


def test_definition_only_and_full_snapshot_use_identical_semantic_features() -> None:
    from manabot.backends.torch.arrays import array_fields
    from manabot.env.decision_memory import integer_words
    from managym.decision_memory import definition_programs

    space, full = _shared_reference_input()
    definition = managym.catalog_tokens()["registry_names"].index("Lightning Bolt")
    program = definition_programs()[definition].tokens
    references = full["decision_references"]
    references[0, 0, 1] = definition + 1
    references[0, 0, 2:6] = 0
    mana = full["decision_reference_details"][0, 2:8].copy()
    details = full["decision_reference_details"]
    details.fill(0)
    encoded = np.array(
        [
            [0, kind, *integer_words(value), index, 1]
            for index, (kind, value) in enumerate(program)
        ],
        np.float32,
    )
    details[0, : len(encoded)] = encoded
    details[0, len(encoded) : len(encoded) + 6] = mana
    references[0, 0, 11] = len(encoded) + 6
    short = {key: value.copy() for key, value in full.items()}
    short["decision_references"][0, 0, 11] = 6
    short["decision_references"][0, 0, 13] = 1
    short["decision_reference_details"].fill(0)
    short["decision_reference_details"][0, :6] = mana
    model = policy(space, "gru", 1)
    full_rows = array_fields(model.prepare(full), np.ndarray)
    short_rows = array_fields(model.prepare(short), np.ndarray)
    assert full_rows.keys() == short_rows.keys()
    for key in full_rows:
        np.testing.assert_array_equal(full_rows[key], short_rows[key], err_msg=key)
    for a, b in zip(forward(model, full), forward(model, short), strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize(
    "field",
    [
        "decision_records",
        "decision_reference_details",
        "decision_references",
        "decision_members",
    ],
)
def test_shared_snapshot_admission_rejects_orphan_slices_and_invalid_joins(
    field: str,
) -> None:
    from manabot.policy.decision_memory import admit_memory

    _, obs = _shared_reference_input()
    if field == "decision_records":
        obs[field][0, 0, 14] = 1
    elif field == "decision_reference_details":
        obs[field][0, 9, 5] = 1
    elif field == "decision_references":
        obs[field][0, 0, 10] = 1
    else:
        obs[field][0, 0, 0] = 2
    with pytest.raises(ValueError):
        admit_memory(obs)


def test_snapshot_unknown_after_public_reference_keeps_its_owned_empty_program() -> (
    None
):
    from manabot.policy.decision_memory import admit_memory

    space, obs = _shared_reference_input()
    obs["decision_references"][0, 1, 10] = 8
    obs["decision_references"][0, 1, 12] = 1
    obs["decision_records"][0, 0, 17] = 1
    obs["decision_members"][0, 3] = [2, 0]
    admit_memory(obs)
    rows = policy(space, "gru", 1).prepare(obs).memory
    assert rows is not None and len(rows.reference_numbers) == 2
    assert not rows.reference_numbers[1].any()
    np.testing.assert_array_equal(rows.member_references, [0, 0, 0, 1])


def test_snapshot_boolean_tokens_reject_non_boolean_values() -> None:
    from manabot.policy.decision_memory import admit_memory

    _, obs = _shared_reference_input()
    obs["decision_reference_details"][0, 0, 1] = 5
    obs["decision_reference_details"][0, 0, 3] = 2
    with pytest.raises(ValueError, match="boolean"):
        admit_memory(obs)


def _two_reference_input() -> tuple[ObservationSpace, dict[str, np.ndarray]]:
    space, obs = _shared_reference_input()
    refs, details = obs["decision_references"], obs["decision_reference_details"]
    refs[0, 1] = refs[0, 0]
    refs[0, 1, [3, 5, 10]] = [3, 4, 8]
    details[0, 8:16] = details[0, :8]
    details[0, 8:10, 3] = [5, 11]
    obs["decision_records"][0, 0, 15:18] = [2, 1, 1]
    obs["decision_members"][0, :4] = [[1, 0], [2, 1], [1, 0], [2, 0]]
    return space, obs


@pytest.mark.parametrize("change", ["between_references", "response_choice_swap"])
def test_bound_references_and_groups_distinguish_equal_additive_totals(
    change: str,
) -> None:
    space, before = _two_reference_input()
    after = {key: value.copy() for key, value in before.items()}
    if change == "between_references":
        refs = after["decision_references"]
        refs[0, :2, 2:4] = refs[0, 1::-1, 2:4]
    else:
        after["decision_members"][0, 2:4, 0] = [2, 1]
    # Both references still occur twice, all group sizes and group-local
    # positions stay fixed, and each field's numeric total is unchanged.
    # The former independent linear sums could not distinguish these inputs.
    model = policy(space, "gru", 1)

    def summary(raw: dict[str, np.ndarray]) -> torch.Tensor:
        prepared = on_device(model.prepare(raw), "cpu")
        assert prepared.memory is not None and model.decision_memory is not None
        state = model._gather_object_embeddings(prepared)
        return model.decision_memory(
            prepared.memory,
            model.semantic_cards.token_embedding,
            model.name_candidates.public_definitions(state.programs),
            model.name_candidates.embedding,
        )

    assert not torch.allclose(summary(before), summary(after), atol=1e-7, rtol=0)


def test_definition_shortcut_rejects_ineligible_registered_rows_before_lookup() -> None:
    from managym.decision_memory import definition_programs

    space, obs = _shared_reference_input()
    definition = next(
        index for index, row in enumerate(definition_programs()) if not row.eligible
    )
    mana = obs["decision_reference_details"][0, 2:8].copy()
    obs["decision_reference_details"].fill(0)
    obs["decision_reference_details"][0, :6] = mana
    obs["decision_references"][0, 0, [1, 11, 13]] = [definition + 1, 6, 1]
    # Even a no-head control must reject this malformed representation.
    control = ActorCritic(space, ModelConfig(world=managym.WORLD_SETUP, hidden_dim=16))
    with pytest.raises(ValueError, match="ineligible.*shortcut"):
        control.prepare(obs)


def test_no_head_selected_programs_ignore_admitted_historical_definitions() -> None:
    from managym.decision_memory import definition_programs

    space, obs = _shared_reference_input()
    definition = next(
        index for index, row in enumerate(definition_programs()) if not row.eligible
    )
    obs["decision_references"][0, 0, 1] = definition + 1
    config = ModelConfig(world=managym.WORLD_SETUP, hidden_dim=16)
    torch.manual_seed(223)
    control = ActorCritic(space, config).eval()
    actual = control.forward_distribution(
        on_device(control.prepare(obs, inference=True), "cpu")
    )
    expected = control.forward_distribution(
        on_device(control.prepare(blank_memory(obs), inference=True), "cpu")
    )
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("field", "row", "column", "value"),
    [
        ("decision_records", 0, 1, 1),  # ordinal must start at zero
        ("decision_records", 0, 3, 0),  # positive multiplicity
        ("decision_records", 0, 7, 1),  # chronological span
        ("decision_records", 0, 9, 1),
        ("decision_records", 0, 10, 5),
        ("decision_records", 0, 11, 12),
        ("decision_records", 0, 18, 0),
        ("decision_records", 0, 15, -1),
        ("decision_records", 0, 15, 1e30),  # reject before allocation/conversion
        ("decision_records", 0, 63, 2),
        ("decision_records", 0, 64, 1),  # unavailable selector
        ("decision_records", 0, 73, 1),  # unavailable declaration
        ("decision_records", 0, 21, 65536),
        ("decision_members", 1, 0, 0),
        ("decision_members", 1, 0, 3),
        ("decision_members", 2, 1, 1),  # new group starts at zero
        ("decision_members", 3, 1, 1),
        ("decision_references", 0, 0, 3),
        ("decision_references", 0, 1, -1),
        ("decision_references", 0, 3, 65536),
        ("decision_references", 0, 6, 2),
        ("decision_references", 0, 11, -1),
        ("decision_references", 0, 11, 1e30),
        ("decision_references", 0, 13, 1),  # shortcut cannot have a program
        ("decision_reference_details", 0, 1, 0),
        ("decision_reference_details", 0, 1, 10),
        ("decision_reference_details", 0, 2, 65536),
        ("decision_reference_details", 1, 0, 1),  # mana before remaining program
        ("decision_reference_details", 1, 4, 0),
        ("decision_reference_details", 2, 1, 4),
        ("decision_reference_details", 2, 4, 2),
    ],
)
def test_memory_admission_rejects_malformed_owned_tables(
    field: str, row: int, column: int, value: float
) -> None:
    from manabot.policy.decision_memory import admit_memory

    _, obs = _two_reference_input()
    obs[field][0, row, column] = value
    with pytest.raises(ValueError):
        admit_memory(obs)


@pytest.mark.parametrize("change", ["first_occurrence", "unreachable", "interleaved"])
def test_memory_admission_requires_reachable_snapshots_in_first_occurrence_order(
    change: str,
) -> None:
    from manabot.policy.decision_memory import admit_memory

    _, obs = _two_reference_input()
    if change == "first_occurrence":
        obs["decision_members"][0, :4, 0] = [2, 1, 2, 1]
    elif change == "unreachable":
        obs["decision_members"][0, :4, 0] = 1
    else:
        obs["decision_reference_details"][0, [1, 2]] = obs[
            "decision_reference_details"
        ][0, [2, 1]]
    with pytest.raises(ValueError):
        admit_memory(obs)


def test_memory_admission_preserves_own_records_coalescing_and_empty_groups() -> None:
    from manabot.policy.decision_memory import admit_memory

    _, obs = _two_reference_input()
    headers = obs["decision_records"]
    headers[0, 0, [3, 9]] = [3, 2]  # three coalesced opponent decisions
    headers[0, 1] = headers[0, 0]
    headers[0, 1, [1, 3, 7, 9, 14, 18]] = [1, 1, 3, 3, 4, 1]
    headers[0, 1, 15:18] = 0  # own record with no members
    # Available zero/false selectors differ from unavailable selectors.
    headers[0, 1, [63, 66, 69, 72]] = 1
    blank = blank_memory(obs)
    mixed = {
        key: np.concatenate((blank[key], value, value), 0) for key, value in obs.items()
    }
    admit_memory(mixed)
    # Compact capacities and noncontiguous views have the same contract.
    compact = dict(mixed)
    for key, count in zip(
        (
            "decision_records",
            "decision_reference_details",
            "decision_references",
            "decision_members",
        ),
        (2, 16, 2, 4),
        strict=True,
    ):
        compact[key] = mixed[key][:, :count]
    admit_memory(compact)
    admit_memory({key: value[::-1] for key, value in compact.items()})
    # Spans must remain contiguous across the actor boundary too.
    mixed["decision_records"][1, 1, 7] = 4
    with pytest.raises(ValueError, match="span"):
        admit_memory(mixed)


@pytest.mark.parametrize("include_own", [False, True])
def test_no_head_admission_respects_own_seat_configuration(include_own: bool) -> None:
    from manabot.env.observation import ObservationSpace

    space, obs = _shared_reference_input()
    space = ObservationSpace(
        space.encoder.config.model_copy(
            update={"decision_memory_include_own": include_own}
        )
    )
    obs["decision_records"][0, 0, 18] = 1
    control = ActorCritic(space, ModelConfig(world=managym.WORLD_SETUP, hidden_dim=16))
    if include_own:
        assert control.prepare(obs).memory is None
    else:
        with pytest.raises(ValueError, match="own-seat"):
            control.prepare(obs)


@pytest.mark.parametrize(
    "field",
    [
        "decision_records",
        "decision_reference_details",
        "decision_references",
        "decision_members",
    ],
)
def test_no_head_admission_rejects_changed_companion_capacity(field: str) -> None:
    space, obs = _shared_reference_input()
    obs[field] = np.pad(obs[field], ((0, 0), (0, 1), (0, 0)))
    control = ActorCritic(space, ModelConfig(world=managym.WORLD_SETUP, hidden_dim=16))
    with pytest.raises(ValueError, match="capacity mismatch"):
        control.prepare(obs)
