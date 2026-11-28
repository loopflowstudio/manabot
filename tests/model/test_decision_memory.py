"""Whole-prefix memories reach both heads and survive ordinary policy tooling."""

from dataclasses import replace
import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import TypeAdapter
import pytest
import torch

import manabot.backends.torch as torch_backend
from manabot.backends.torch.arrays import on_device
from manabot.backends.torch.checkpoint import load_checkpoint, save_checkpoint
from manabot.backends.torch.collection import CollectionSession
from manabot.backends.torch.learner import Learner
from manabot.backends.torch.policy import ActorCritic
from manabot.env.decision_memory import MEMORY_FIELDS
from manabot.env.match import Match, Reward
from manabot.env.memory_store import gather_memory
from manabot.env.observation import ObservationSpace, encode_policy_batch
from manabot.infra.config import ModelConfig, ObservationConfig, RewardConfig
from manabot.policy.admission import admit
from manabot.policy.prepare import RecordNamespace, record_key
from manabot.policy.spans import ObservationSelection
from manabot.training.learning import DampedPolicyLearning
from manabot.training.rollout.collection import (
    close_collector,
    collect,
    open_collector,
)
from manabot.training.updates.damped_policy import update_damped_iteration
import managym
from managym.decision_memory import (
    DecisionContext,
    DecisionRecord,
    DecisionReference,
    PublicChoice,
)
from tests.fixtures.memory_rows import MemoryRows, memory_rows, retained


def encode_with_records(
    space: ObservationSpace,
    obs: managym.Observation,
    records: list[DecisionRecord],
) -> managym.ObservationBatch:
    """`obs` restored with `records` as its decision memory, encoded natively.

    A decoded log starts a fresh lineage, so the whole log is written.
    """
    payload = json.loads(obs.toJSON())
    payload["decision_memory"] = TypeAdapter(list[DecisionRecord]).dump_python(
        records, mode="json"
    )
    batch = encode_policy_batch(
        space, [managym.Observation.from_json(json.dumps(payload))]
    )
    counts = rows_of(space, batch).spans.tables["decision_records"].counts
    assert int(counts.sum()) == len(records)
    return batch


type MemoryInput = managym.ObservationBatch | ObservationSelection


def blank(batch: managym.ObservationBatch) -> managym.ObservationBatch:
    return batch.blank_decision_memory()


def rows_of(
    space: ObservationSpace, *batches: managym.ObservationBatch
) -> ObservationSelection:
    """Concatenate single-row batches in order."""
    return ObservationSelection(space, tuple((batch, 0) for batch in batches))


def memory_root() -> tuple[ObservationSpace, managym.Observation]:
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
    return space, env.observation_for_player(env.current_agent_index())


def memory_input() -> tuple[ObservationSpace, managym.ObservationBatch]:
    space, obs = memory_root()
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
    return space, encode_with_records(space, obs, [record])


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
    model: ActorCritic, obs: MemoryInput, *, objectless: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    prepared = model.prepare(obs)
    if objectless:
        prepared = replace(prepared, focus_mask=np.zeros_like(prepared.focus_mask))
    return model.forward_distribution(on_device(prepared, "cpu"))


@pytest.mark.parametrize("encoder", ["transformer", "gru"])
@pytest.mark.parametrize("summaries", [1, 4])
def test_decision_memory_changes_probabilities_value_and_both_encoder_gradients(
    encoder: Literal["transformer", "gru"], summaries: int
) -> None:
    space, obs = memory_input()
    model = policy(space, encoder, summaries)
    # Make every legal action objectless. Their distinct native action features
    # remain, proving the actor can read memory through the selected board join.
    logits, value = forward(model, obs, objectless=True)
    blank_logits, blank_value = forward(model, blank(obs), objectless=True)
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
    mixed = rows_of(space, one, blank(one))
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


def test_decision_memory_rejects_noncanonical_words_with_the_same_catalog_join() -> (
    None
):
    space, obs = memory_input()
    rows = memory_rows(space, obs)
    # Catalog keys keep the low u32, so adding 2**32 aliases the admitted
    # name. Each float is still exact, but the high word is no longer u16.
    rows["decision_records"][0, 12] += 65536
    with pytest.raises(ValueError, match="integer words"):
        retained(space, [(obs, 0)], [rows])


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
    collector = open_collector(
        space,
        Match(),
        Reward(RewardConfig()),
        num_envs=1,
        learning_seats="one",
        batch_closure="exact-quota",
        numerics=CollectionSession(device="cpu", seed=111, precision="float32"),
    )
    try:
        batch = collect(collector, model, 16)
        assert batch.actions.size == 16
        assert gather_memory(
            batch.observations(
                np.arange(batch.actions.size)
            ).observation.memory.prefixes
        )["decision_records"][..., 19].any()
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
        close_collector(collector)


def test_control_retains_logs_without_using_them(tmp_path: Path) -> None:
    space, obs = memory_input()
    config = ModelConfig(world=managym.WORLD_SETUP, hidden_dim=16)
    torch.set_num_threads(1)
    torch.manual_seed(23)
    model = ActorCritic(space, config).eval()
    assert model.decision_memory is None
    assert model.prepare(obs).memory is None
    for actual, expected in zip(
        forward(model, obs), forward(model, blank(obs)), strict=True
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    # A log-enabled trunk reserves the exact vocabulary read by its branches.
    # The true history-disabled incumbent remains a separate throughput control.
    path = tmp_path / "control.safetensors"
    save_checkpoint(model, space, path, player_configs=Match().to_rust())
    restored, restored_space = load_checkpoint(str(path))
    assert restored_space == space
    for actual, expected in zip(
        forward(restored.eval(), obs), forward(model, obs), strict=True
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


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
    space, batch, rows = _shared_reference_input()
    rows[field][-1 if padding else 0, 0] = value
    with pytest.raises(ValueError):
        retained(space, [(batch, 0)], [rows])


def _shared_reference_input() -> tuple[
    ObservationSpace, managym.ObservationBatch, MemoryRows
]:
    """One record with two references, and its editable memory tables."""
    space, root = memory_root()
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
    batch = encode_with_records(space, root, [record])
    return space, batch, memory_rows(space, batch)


def test_decision_memory_snapshot_routes_preserve_groups_repetitions_and_empty_viewer() -> (
    None
):
    space, one, _ = _shared_reference_input()
    raw = rows_of(space, one, blank(one), one)
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
    space, batch, original = _shared_reference_input()
    changed = {key: value.copy() for key, value in original.items()}
    if change == "numeric_binding":
        details = changed["decision_reference_details"]
        details[:2, 2:4] = details[1::-1, 2:4]
    elif change == "reference_binding":
        references = changed["decision_references"]
        # Swap bound power and toughness words, preserving their numeric sum.
        references[0, 2:6] = references[0, [4, 5, 2, 3]]
    elif change == "group":
        changed["decision_records"][0, 15:18] = [0, 3, 0]
        changed["decision_members"][:3, 1] = [0, 1, 2]
    else:
        changed["decision_records"][0, 16] = 1
        changed["decision_members"][2] = 0
    model = policy(space, "gru", 1)

    def summary(rows: MemoryRows) -> torch.Tensor:
        raw = retained(space, [(batch, 0)], [rows])
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

    space, batch, full = _shared_reference_input()
    definition = managym.catalog_tokens()["registry_names"].index("Lightning Bolt")
    program = definition_programs()[definition].tokens
    references = full["decision_references"]
    references[0, 1] = definition + 1
    references[0, 2:6] = 0
    mana = full["decision_reference_details"][2:8].copy()
    details = full["decision_reference_details"]
    details.fill(0)
    encoded = np.array(
        [
            [0, kind, *integer_words(value), index, 1]
            for index, (kind, value) in enumerate(program)
        ],
        np.float32,
    )
    details[: len(encoded)] = encoded
    details[len(encoded) : len(encoded) + 6] = mana
    references[0, 11] = len(encoded) + 6
    short = {key: value.copy() for key, value in full.items()}
    short["decision_references"][0, 11] = 6
    short["decision_references"][0, 13] = 1
    short["decision_reference_details"].fill(0)
    short["decision_reference_details"][:6] = mana
    model = policy(space, "gru", 1)
    full_input = retained(space, [(batch, 0)], [full])
    short_input = retained(space, [(batch, 0)], [short])
    full_rows = array_fields(model.prepare(full_input), np.ndarray)
    short_rows = array_fields(model.prepare(short_input), np.ndarray)
    assert full_rows.keys() == short_rows.keys()
    for key in full_rows:
        np.testing.assert_array_equal(full_rows[key], short_rows[key], err_msg=key)
    for a, b in zip(
        forward(model, full_input), forward(model, short_input), strict=True
    ):
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
    space, batch, rows = _shared_reference_input()
    if field == "decision_records":
        rows[field][0, 14] = 1
    elif field == "decision_reference_details":
        rows[field][9, 5] = 1
    elif field == "decision_references":
        rows[field][0, 10] = 1
    else:
        rows[field][0, 0] = 2
    with pytest.raises(ValueError):
        retained(space, [(batch, 0)], [rows])


def test_snapshot_unknown_after_public_reference_keeps_its_owned_empty_program() -> (
    None
):
    space, batch, edited = _shared_reference_input()
    edited["decision_references"][1, 10] = 8
    edited["decision_references"][1, 12] = 1
    edited["decision_records"][0, 17] = 1
    edited["decision_members"][3] = [2, 0]
    obs = retained(space, [(batch, 0)], [edited])
    rows = policy(space, "gru", 1).prepare(obs).memory
    assert rows is not None and len(rows.reference_numbers) == 2
    assert not rows.reference_numbers[1].any()
    np.testing.assert_array_equal(rows.member_references, [0, 0, 0, 1])


def test_snapshot_boolean_tokens_reject_non_boolean_values() -> None:
    space, batch, rows = _shared_reference_input()
    rows["decision_reference_details"][0, 1] = 5
    rows["decision_reference_details"][0, 3] = 2
    with pytest.raises(ValueError, match="boolean"):
        retained(space, [(batch, 0)], [rows])


def _two_reference_input() -> tuple[
    ObservationSpace, managym.ObservationBatch, MemoryRows
]:
    space, batch, rows = _shared_reference_input()
    refs, details = rows["decision_references"], rows["decision_reference_details"]
    refs[1] = refs[0]
    refs[1, [3, 5, 10]] = [3, 4, 8]
    details[8:16] = details[:8]
    details[8:10, 3] = [5, 11]
    rows["decision_records"][0, 15:18] = [2, 1, 1]
    rows["decision_members"][:4] = [[1, 0], [2, 1], [1, 0], [2, 0]]
    return space, batch, rows


@pytest.mark.parametrize("change", ["between_references", "response_choice_swap"])
def test_bound_references_and_groups_distinguish_equal_additive_totals(
    change: str,
) -> None:
    space, batch, before = _two_reference_input()
    after = {key: value.copy() for key, value in before.items()}
    if change == "between_references":
        refs = after["decision_references"]
        refs[:2, 2:4] = refs[1::-1, 2:4]
    else:
        after["decision_members"][2:4, 0] = [2, 1]
    # Both references still occur twice, all group sizes and group-local
    # positions stay fixed, and each field's numeric total is unchanged.
    # The former independent linear sums could not distinguish these inputs.
    model = policy(space, "gru", 1)

    def summary(rows: MemoryRows) -> torch.Tensor:
        raw = retained(space, [(batch, 0)], [rows])
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

    space, batch, rows = _shared_reference_input()
    definition = next(
        index for index, row in enumerate(definition_programs()) if not row.eligible
    )
    mana = rows["decision_reference_details"][2:8].copy()
    rows["decision_reference_details"].fill(0)
    rows["decision_reference_details"][:6] = mana
    rows["decision_references"][0, [1, 11, 13]] = [definition + 1, 6, 1]
    # Even a no-head control must reject this malformed representation.
    control = ActorCritic(space, ModelConfig(world=managym.WORLD_SETUP, hidden_dim=16))
    with pytest.raises(ValueError, match="ineligible.*shortcut"):
        control.prepare(retained(space, [(batch, 0)], [rows]))


def test_no_head_selected_programs_ignore_admitted_historical_definitions() -> None:
    from managym.decision_memory import definition_programs

    space, batch, rows = _shared_reference_input()
    definition = next(
        index for index, row in enumerate(definition_programs()) if not row.eligible
    )
    rows["decision_references"][0, 1] = definition + 1
    obs = retained(space, [(batch, 0)], [rows])
    config = ModelConfig(world=managym.WORLD_SETUP, hidden_dim=16)
    torch.manual_seed(223)
    control = ActorCritic(space, config).eval()
    actual = control.forward_distribution(
        on_device(control.prepare(obs, inference=True), "cpu")
    )
    expected = control.forward_distribution(
        on_device(control.prepare(blank(batch), inference=True), "cpu")
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
    space, batch, rows = _two_reference_input()
    retained(space, [(batch, 0)], [rows])
    rows[field][row, column] = value
    with pytest.raises(ValueError):
        retained(space, [(batch, 0)], [rows])


@pytest.mark.parametrize("change", ["first_occurrence", "unreachable", "interleaved"])
def test_memory_admission_requires_reachable_snapshots_in_first_occurrence_order(
    change: str,
) -> None:
    space, batch, rows = _two_reference_input()
    if change == "first_occurrence":
        rows["decision_members"][:4, 0] = [2, 1, 2, 1]
    elif change == "unreachable":
        rows["decision_members"][:4, 0] = 1
    else:
        details = rows["decision_reference_details"]
        details[[1, 2]] = details[[2, 1]]
    with pytest.raises(ValueError):
        retained(space, [(batch, 0)], [rows])


def test_memory_admission_preserves_own_records_coalescing_and_empty_groups() -> None:
    space, batch, rows = _two_reference_input()
    headers = rows["decision_records"]
    headers[0, [3, 9]] = [3, 2]  # three coalesced opponent decisions
    headers[1] = headers[0]
    headers[1, [1, 3, 7, 9, 14, 18]] = [1, 1, 3, 3, 4, 1]
    headers[1, 15:18] = 0  # own record with no members
    # Available zero/false selectors differ from unavailable selectors.
    headers[1, [63, 66, 69, 72]] = 1
    empty = memory_rows(space, blank(batch))
    retained(space, [(batch, 0)] * 3, [empty, rows, rows])
    # Spans must remain contiguous across the actor boundary too.
    headers[1, 7] = 4
    with pytest.raises(ValueError, match="span"):
        retained(space, [(batch, 0)], [rows])


@pytest.mark.parametrize("include_own", [False, True])
def test_no_head_admission_respects_own_seat_configuration(include_own: bool) -> None:
    from manabot.env.observation import ObservationSpace

    space, batch, rows = _shared_reference_input()
    space = ObservationSpace(
        space.config.model_copy(update={"decision_memory_include_own": include_own})
    )
    rows["decision_records"][0, 18] = 1
    obs = retained(space, [(batch, 0)], [rows])
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
    space, batch, rows = _shared_reference_input()
    capacities = [space.shapes[key][0] for key in MEMORY_FIELDS]
    capacities[MEMORY_FIELDS.index(field)] += 1
    obs = retained(space, [(batch, 0)], [rows], capacities=tuple(capacities))
    control = ActorCritic(space, ModelConfig(world=managym.WORLD_SETUP, hidden_dim=16))
    with pytest.raises(ValueError, match="capacity mismatch"):
        control.prepare(obs)


def test_native_memory_spans_prepare_the_same_chronological_records() -> None:
    space, _ = memory_root()
    env = managym.Env(seed=111, decision_memory=True)
    env.reset(Match().to_rust())
    # Adjacent float32 numbers above 2**24 are two integers apart.
    env.scenario_set_life(0, 16_777_217)
    env.scenario_set_life(1, 16_777_219)
    early = env.clone_env()
    for _ in range(4):
        env.step(0)
    games = [early, env]
    batch = managym.encode_observation_batch(games, space.layout)
    dtype = np.dtype(
        [
            (name, dtype)
            for name, dtype, _ in space.layout.csr_fields["decision_records"]
        ]
    )
    records = np.frombuffer(batch.tables()["decision_records"][2], dtype)
    assert records["life_viewer"].dtype == np.int32
    assert set(records["life_viewer"]) | set(records["life_opponent"]) == {
        16_777_217,
        16_777_219,
    }
    assert np.all(records["valid"] == 1)
    model = policy(space, "gru", 1)
    packed = model.prepare(batch)
    assert packed.memory is not None
    # The same rows restored from an archive (folded as retained prefixes)
    # give the policy the same outputs.
    selection = ObservationSelection(space, ((batch, 0), (batch, 1)))
    archived = retained(
        space,
        selection.coordinates,
        [
            {key: selection.project(key)[row] for key in MEMORY_FIELDS}
            for row in range(2)
        ],
    )
    model.eval()
    for a, b in zip(forward(model, batch), forward(model, archived), strict=True):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    spec = model._admission_spec()
    key = record_key(RecordNamespace.TOKEN, records["kind"][:1])[0]
    keys = spec.tables.record_name_keys
    incomplete = replace(spec.tables, record_name_keys=keys[keys != key])
    with pytest.raises(ValueError, match="outside the record vocabulary"):
        admit(batch, replace(spec, tables=incomplete), bucketed=False)
