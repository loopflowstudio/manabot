"""Bucketed shapes agree with the default path on admitted inputs."""

from collections.abc import Callable
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from manabot.backends.torch.policy import ActorCritic, PolicyInference
from manabot.backends.torch.rl import NumericalError
from manabot.backends.torch.rollout import AdmittedRollout
from manabot.env.observation import (
    FACT_OBJECT_INDEX,
    ObservationSpace,
    column,
    encode_observation,
)
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.policy.buckets import (
    AXES,
    bucket,
    bucket_observation,
    bucket_sizes,
    row_needs,
    sizes_of,
)
from manabot.policy.prepare import prepare
from manabot.training.rollout.collection import collect
from manabot.verify.representation import shared_effect_root
import managym
from tests.model.prepared_routes import device_input, host_inputs
from tests.model.test_inference import PolicyFixture, _policy


def _bucketed(reference: ActorCritic) -> ActorCritic:
    """A bucketed copy of a policy with identical weights."""
    agent = ActorCritic(
        reference.observation_space, reference.config, bucketed_shapes=True
    ).eval()
    agent.load_state_dict(reference.state_dict())
    # Fresh fact and link outputs are zero; give those joins real signal.
    with torch.no_grad():
        for original, copy in zip(
            reference.parameters(), agent.parameters(), strict=True
        ):
            noise = torch.randn_like(original) * 0.05
            original.add_(noise)
            copy.add_(noise)
    return agent


def _host(obs: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {key: value.numpy() for key, value in obs.items()}


def _later_decisions(case: PolicyFixture) -> dict[str, torch.Tensor]:
    """Decisions from played turns: permanents, facts and public history."""
    batch = collect(case.collector, case.agent, 24)
    return {
        key: torch.from_numpy(value[-8:, 0]).float()
        for key, value in batch.observations().observation.items()
    }


def test_bucket_rule() -> None:
    assert [bucket(n, 512) for n in (0, 1, 8, 9, 59, 133, 512)] == [
        8,
        8,
        8,
        16,
        64,
        256,
        512,
    ]
    assert bucket(31, 96) == 32
    assert bucket(70, 96) == 96
    assert bucket(0, 4) == 4


def test_row_needs_keep_strided_padding_and_referenced_objects() -> None:
    space = ObservationSpace(
        ObservationConfig(
            policy_history_version=1,
            max_cards_per_player=9,
            max_permanents_per_player=5,
            max_stack_objects=3,
        )
    )
    obs = {key: value[::-2] for key, value in space.allocate(9).items()}
    # Full coordinates are players [0,2), cards [2,11), [11,20),
    # permanents [20,25), [25,30), and stack [30,33).
    obs["actions_valid"][:2, 0] = 1
    obs["action_focus"][:] = -1
    obs["action_focus"][0, 0, 0] = 10
    obs["action_focus"][1, 0, 0] = 19
    obs["facts"] = obs["facts"][:, ::-1, ::-1]
    obs["facts"][2, 3, FACT_OBJECT_INDEX] = 25.75
    obs["facts"][3, 2, FACT_OBJECT_INDEX] = 30
    obs["events"][4, 1, column("events", "target_context")] = 32.5
    obs["events"][0, 2, column("events", "source_context")] = np.inf
    obs["events"][1, 1, column("events", "target_context")] = 33
    obs["decklists"] = obs["decklists"][:, ::-1, ::-1, ::-1]
    obs["decklists"][0, 0, 5, -1] = np.nan
    obs["decklists"][1, 1, 2, 0] = 1
    obs["decklists"][1, 0, 8, 1] = -0.0
    obs["known_hand"] = obs["known_hand"].astype(">f8")
    obs["known_hand"][4, 1, 6, 0] = 1e-200
    # Conversion to float32 before subtracting one would name the next card
    # in the second seat. The float32 spacing changes at 16.
    obs["effects"] = obs["effects"].astype(np.float64)
    obs["effects"][2, 0, column("effects", "source")] = 17 - 7e-7
    obs["effects"][2, 0, column("effects", "detail_count")] = 3.75
    obs["effects"][2, 1, column("effects", "detail_count")] = np.inf

    expected = np.zeros((5, len(AXES)), dtype=np.int64)
    for axis, rows in {
        "actions": [1, 1, 0, 0, 0],
        "facts": [0, 0, 4, 3, 0],
        "effects": [0, 0, 2, 0, 0],
        "details": [0, 0, 3, 0, 0],
        "cards": [9, 9, 5, 0, 0],
        "permanents": [0, 0, 5, 5, 0],
        "stack": [0, 0, 0, 0, 3],
        "events": [3, 2, 0, 0, 2],
        "known": [0, 0, 0, 0, 7],
        "decklists": [6, 3, 0, 0, 0],
    }.items():
        expected[:, AXES.index(axis)] = rows
    np.testing.assert_array_equal(row_needs(obs, space), expected, strict=True)


def test_bucket_extents_cover_full_csr_spans_in_narrow_exports() -> None:
    space = ObservationSpace()
    obs = space.allocate(1)
    headers = obs["effects"].astype(np.uint8)
    headers[0, 0, column("effects", "member_offset")] = 250
    headers[0, 0, column("effects", "member_count")] = 10
    obs["effects"] = headers
    needs = row_needs(obs, space)
    assert needs is not None and needs[0, AXES.index("members")] == 260


def test_remapped_references_are_every_object_table_column() -> None:
    """A new object-table reference must be remapped, not trimmed silently."""
    layout = managym.observation_layout(policy_history=True)
    # Give each reference its own batch item, so no other join can keep its target.
    joins: dict[tuple[str, str], tuple[int, int]] = {}
    for key, array in layout.items():
        for name, field in array["columns"].items():
            encoding = field["encoding"].lower()
            if field["kind"] == "reference" and "object" in encoding:
                joins[key, name] = (
                    field["index"],
                    0 if encoding.startswith("zero-based") else 1,
                )
        element = array.get("element")
        if element is not None and element["kind"] == "reference":
            assert key == "action_focus"
            joins[key, ""] = 0, 0
    space = ObservationSpace(ObservationConfig(policy_history_version=1))
    assert {
        (key, index, base)
        for key, fields in space.layout.object_references.items()
        for index, base in fields
    } == {
        (key, index if name else None, base)
        for (key, name), (index, base) in joins.items()
    }
    count = len(joins)
    obs = space.allocate(count)
    obs["actions_valid"][:, 0] = obs["events_valid"][:, 0] = 1
    for key in ("relations", "effects", "facts"):
        obs[key][:, 0, column(key, "valid")] = 1
    obs["effects"][:, 0, column("effects", "member_count")] = 1
    # Last stack slot, after the removable card and permanent padding.
    target = (
        1
        + 2 * (obs["agent_cards"].shape[1] + obs["agent_permanents"].shape[1])
        + obs["stack"].shape[1]
    )
    for row, ((key, _), (index, base)) in enumerate(joins.items()):
        obs[key][row, 0, index] = target + base
    extents = space.layout.axis_extents(obs)
    assert extents is not None
    np.testing.assert_array_equal(
        extents["stack"], np.full(count, obs["stack"].shape[1])
    )
    measured = bucket_observation(obs, space, objects=True)
    assert measured is not None and measured.buckets.trims_objects
    policy = ActorCritic(
        space,
        ModelConfig(
            hidden_dim=8,
            num_attention_heads=2,
            recent_events=True,
            world=managym.WORLD_SETUP,
        ),
    )
    batch = prepare(measured, policy._admission_spec().tables)
    assert (
        batch.effects is not None
        and batch.events is not None
        and batch.stack_targets is not None
    )
    destinations = {
        ("facts", "object"): batch.facts.targets.reshape(count, -1)[:, 0],
        ("effects", "controller"): batch.effects.controller,
        ("effects", "source"): batch.effects.source_object,
        ("effect_members", "object"): batch.effects.routes,
        ("stack", "source"): batch.stack_targets.reshape(count, -1)[:, 0],
        ("action_focus", ""): batch.focus[:, 0],
    }
    for i, name in enumerate(("source", "reference")):
        destinations["relations", name] = batch.relation_objects[i].reshape(count, -1)[
            :, 0
        ]
    for i, name in enumerate(("source_context", "target_context")):
        destinations["events", name] = batch.events.focus[:, 0, i]
    for i, name in enumerate(("attached_to", "blocking")):
        targets = batch.link_targets[i].reshape(count, 2, -1)
        for seat, key in enumerate(("agent_permanents", "opponent_permanents")):
            destinations[key, name] = targets[:, seat, 0]
    assert destinations.keys() == joins.keys()
    objects = batch.validity.shape[1]
    for row, key in enumerate(joins):
        assert destinations[key][row] == (row + 1) * objects - 1, key


@pytest.mark.parametrize(
    ("history", "streams"), [(False, False), (True, False), (True, True)]
)
def test_outputs_and_gradients_match_default(history: bool, streams: bool) -> None:
    """`streams` compares the `stream_norm` and `focus_film` layout."""
    torch.manual_seed(302)
    case = _policy(history=history)
    obs = _later_decisions(case)
    reference = case.agent
    if streams:
        reference = ActorCritic(
            reference.observation_space,
            reference.config.model_copy(
                update={"stream_norm": True, "focus_film": True}
            ),
        ).eval()
    agent = _bucketed(reference)
    buckets = bucket_sizes(_host(obs), agent.observation_space, objects=True)
    assert buckets is not None and buckets.trims_objects
    outputs = []
    for model in (reference, agent):
        logits, values = model.forward_distribution(device_input(model, obs))
        outputs.append((logits, values))
        legal = obs["actions_valid"].bool()
        (logits[legal].square().mean() + values.square().mean()).backward()
    (expected_logits, expected_values), (logits, values) = outputs
    assert logits.shape == expected_logits.shape
    torch.testing.assert_close(logits, expected_logits, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(values, expected_values, atol=1e-5, rtol=1e-5)
    for (name, a), b in zip(
        reference.named_parameters(), agent.parameters(), strict=True
    ):
        if a.grad is None:
            # The complete catalog may add exactly-zero program gradients.
            assert b.grad is None or not b.grad.any(), name
        else:
            assert b.grad is not None, name
            torch.testing.assert_close(b.grad, a.grad, atol=1e-5, rtol=1e-4)
    with torch.no_grad(), PolicyInference(agent) as scope:
        prepared = scope.prepare(host_inputs(obs))
        assert isinstance(prepared.bucket_needs, np.ndarray)
        assert (
            sizes_of(
                prepared.bucket_needs,
                agent.observation_space,
                objects=True,
            )
            == buckets
        )
        assert prepared.facts.kinds.shape == (len(obs["facts"]), buckets.sizes["facts"])
        served, _ = agent.forward_distribution(prepared)
    torch.testing.assert_close(served, expected_logits, atol=1e-5, rtol=1e-5)


def _set_fact_object(value: float) -> Callable[[dict[str, torch.Tensor]], None]:
    def plant(obs: dict[str, torch.Tensor]) -> None:
        obs["facts"][0, 0] = 0
        obs["facts"][0, 0, 11] = 1
        obs["facts"][0, 0, 1] = 1
        obs["facts"][0, 0, FACT_OBJECT_INDEX] = value

    return plant


def _set_focus(value: float) -> Callable[[dict[str, torch.Tensor]], None]:
    def plant(obs: dict[str, torch.Tensor]) -> None:
        slot = int(obs["actions_valid"][0].nonzero()[0])
        obs["action_focus"][0, slot, 0] = value

    return plant


def _shrink_facts(obs: dict[str, torch.Tensor]) -> None:
    obs["facts"] = obs["facts"][:, :100]


@pytest.mark.parametrize(
    ("plant", "message"),
    [
        (_set_fact_object(10_000), "invalid fact object reference"),
        (_set_fact_object(2.5), "invalid fact object reference"),
        (_set_fact_object(1e-12), "invalid fact object reference"),
        (_set_focus(10_000), "invalid action focus reference"),
        (_shrink_facts, "facts"),
    ],
)
def test_rejections_match_default(
    plant: Callable[[dict[str, torch.Tensor]], None],
    message: str,
) -> None:
    case = _policy()
    obs = _later_decisions(case)
    agent = _bucketed(case.agent)
    plant(obs)
    errors = []
    for model in (case.agent, agent):
        with pytest.raises(ValueError, match=message) as error:
            model.forward_distribution(device_input(model, obs))
        errors.append((type(error.value), str(error.value)))
    assert errors[0] == errors[1]


def test_admission_preserves_padding_errors_and_original_mask_evidence() -> None:
    case = _policy()
    agent = _bucketed(case.agent)
    obs = case.obs
    obs["effects"][-1, -1, column("effects", "valid")] = np.nan
    for model in (agent, case.agent):
        with pytest.raises(ValueError, match="^state record validity must be binary$"):
            model.prepare(obs)
    obs["actions_valid"][-1, -1] = 0.25
    for model in (agent, case.agent):
        with pytest.raises(NumericalError, match="^invalid_action_mask$") as rejected:
            model.prepare(obs)
        np.testing.assert_array_equal(
            rejected.value.tensors["mask"], obs["actions_valid"]
        )
        assert rejected.value.tensors["failed"][-1, -1]


def test_reconfigured_fact_capacity_checks_original_buffer_shape() -> None:
    case = _policy()
    agent = _bucketed(case.agent)
    agent.prepare(case.obs)
    # The old allocation and its measured prefix cannot stand in for a new contract.
    agent.observation_space.config.max_facts = 8
    for model in (case.agent, agent):
        with pytest.raises(ValueError, match=r"^expected \[batch, 8, \d+\] facts$"):
            model.prepare(case.obs)


def test_bucketed_admission_keeps_effect_order_bounds_and_batch_strides() -> None:
    case = _policy()
    env = shared_effect_root(1)
    encoded = encode_observation(
        case.agent.observation_space,
        env.observation_for_player(env.current_agent_index()),
    )
    obs = {key: np.stack([value, value]) for key, value in encoded.items()}
    bucketed = _bucketed(case.agent)
    order = column("effect_members", "order")
    # Membership order belongs to the declared table, not its live prefix.
    obs["effect_members"][1, 0, order] = obs["effect_members"].shape[1]
    for model in (case.agent, bucketed):
        model.check_observation(obs)
    for key, field, value, message in (
        (
            "effect_members",
            "order",
            obs["effect_members"][1, 1, order],
            "duplicate effect membership order",
        ),
        ("effects", "member_offset", 1, "invalid effect CSR partition"),
        ("effects", "member_count", -1, "invalid effect CSR offset/count reference"),
        ("effect_details", "kind", 0, "live effect detail has no kind"),
    ):
        idx = (1, 0, column(key, field))
        saved = obs[key][idx]
        obs[key][idx] = value
        for model in (case.agent, bucketed):
            with pytest.raises(ValueError, match=f"^{message}$"):
                model.check_observation(obs)
        obs[key][idx] = saved


def test_learner_rechecks_edited_padding_beyond_recorded_extents() -> None:
    case = _policy(bucketed_shapes=True)
    batch = collect(case.collector, case.agent, 1)
    row = batch.recorded.index[0]
    batch.recorded.rows["facts"][row, -1, column("facts", "valid")] = 2
    with pytest.raises(ValueError, match="^fact validity must be binary$"):
        AdmittedRollout(case.agent, batch).admit_rows(np.array([0]), 1)


@pytest.mark.parametrize("recorded", [True, False], ids=["mixed", "absent-table"])
def test_learner_preserves_each_rows_relation_floor(recorded: bool) -> None:
    case = _policy(bucketed_shapes=True)
    masks = case.obs["actions_valid"]
    legal_slots = [np.flatnonzero(mask) for mask in masks]
    # Collection measures this continuation before packing drops the masked action.
    masked_slot = np.flatnonzero(masks[0] == 0)[0]
    case.obs["actions"][0, masked_slot, column("actions", "continuation")] = 64
    batch = collect(case.collector, case.agent, 1)
    needs = batch.recorded.needs
    assert needs is not None
    positions = [
        int(np.flatnonzero(batch.recorded.index == row).item()) for row in (0, 1)
    ]
    assert needs[0, AXES.index("relations")] == 64
    if recorded:
        needs[1] = -1
    else:
        batch.recorded = replace(batch.recorded, needs=None)

    indexes = np.array(positions)
    separate = AdmittedRollout(case.agent, batch)
    grouped = AdmittedRollout(case.agent, batch)
    separate.admit_rows(indexes, 1)
    grouped.admit_rows(indexes, 2)

    for row, (position, slots) in enumerate(zip(positions, legal_slots, strict=True)):
        selection = np.array([position])
        single, _ = separate.prepare_rows(selection)
        together, _ = grouped.prepare_rows(selection)
        expected_relations = 64 if recorded and row == 0 else 8
        assert single.relation_features.shape[1] == expected_relations
        np.testing.assert_array_equal(single.legal_slots, slots)
        np.testing.assert_equal(asdict(single), asdict(together))


def test_reference_to_padding_object_is_kept() -> None:
    """A focus on the last permanent slot keeps that slot, as the default reads it."""
    case = _policy()
    obs = _later_decisions(case)
    agent = _bucketed(case.agent)
    objects = case.agent._is_agent_template.shape[1]
    _set_focus(objects - 1)(obs)
    buckets = bucket_sizes(_host(obs), agent.observation_space, objects=True)
    assert buckets is not None
    assert buckets.sizes["permanents"] == buckets.capacities["permanents"]
    with torch.no_grad():
        expected, _ = case.agent.forward_distribution(device_input(case.agent, obs))
        actual, _ = agent.forward_distribution(device_input(agent, obs))
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("objects", [True, False])
def test_row_needs_regroup_to_the_measured_buckets(objects: bool) -> None:
    """Needs measured once per row give any grouping of those rows the buckets
    measuring the group directly gives, so later epochs need not measure."""
    case = _policy(history=True)
    space = case.agent.observation_space
    batch = collect(case.collector, case.agent, 24)
    obs = {
        key: torch.from_numpy(value[:, 0]).float()
        for key, value in batch.observations().observation.items()
    }
    # A fractional object reference in one row, which only that row needs.
    _set_fact_object(3.7)(obs)
    host = _host(obs)
    needs = row_needs(host, space)
    assert needs is not None and len(needs) == len(host["actions_valid"])
    rng = np.random.default_rng(0)
    for _ in range(20):
        rows = rng.choice(len(needs), size=rng.integers(1, len(needs)), replace=False)
        group = {key: value[rows] for key, value in host.items()}
        assert sizes_of(needs[rows], space, objects=objects) == bucket_sizes(
            group, space, objects=objects
        )


def test_collection_records_each_rows_measured_needs() -> None:
    """These decisions retain all measured extents after legal-action packing."""
    case = _policy(history=True, bucketed_shapes=True)
    batch = collect(case.collector, case.agent, 12)
    rows = np.arange(batch.actions.size)
    recorded = batch.recorded.gather_needs(rows)
    assert recorded is not None
    host = batch.observations(rows).observation
    assert np.array_equal(recorded, row_needs(host, case.agent.observation_space))
