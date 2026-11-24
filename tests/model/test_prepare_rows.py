"""Host preparation preserves the admitted observation's joins and coordinates."""

from dataclasses import asdict, replace
from functools import lru_cache

import numpy as np
import pytest
import torch

from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match
from manabot.env.observation import (
    FACT_CONTINUATION_PROGRAM,
    FACT_DETAIL_INDEX,
    FACT_KIND_INDEX,
    FACT_OBJECT_INDEX,
    FACT_SOURCE_INDEX,
    FACT_VALID_INDEX,
    FACT_VALUE_INDEX,
    PERMANENT_DEFINITION_INDEX,
    ObservationBuffers,
    ObservationSpace,
    column,
    encode_observation,
)
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.policy.admission import Spec, admit_native, check
from manabot.policy.buckets import MIN_BUCKET, bucket_observation
from manabot.policy.prepare import (
    OBJECT_VALIDITY_KEYS,
    AttentionLayout,
    Layout,
    Prepared,
    Tables,
    prepare,
    select_attention_columns,
)
from manabot.policy.readings import fresh_readings, legacy_readings, read_numeric
from manabot.policy.spans import observation_spans
from manabot.verify.representation import shared_effect_root
import managym
from tests.model.prepared_routes import device_input


@lru_cache(maxsize=3)
def _observations(
    version: int, history: bool
) -> tuple[ObservationSpace, dict[str, np.ndarray]]:
    space = ObservationSpace(
        ObservationConfig(
            policy_input_version=version, policy_history_version=int(history)
        )
    )
    env = managym.Env(seed=11)
    env.reset(Match().to_rust())
    rows: list[dict[str, np.ndarray]] = []
    for _ in range(3):
        obs = env.observation_for_player(env.current_agent_index())
        rows.append(encode_observation(space, obs))
        env.step(0)
    return space, {key: np.stack([row[key] for row in rows]) for key in rows[0]}


def _tables(agent: ActorCritic) -> Tables:
    return Tables.from_buffers(
        {name: value.detach().numpy() for name, value in agent.named_buffers()},
        Layout(agent.readings),
    )


def original_object_slots(
    obs: ObservationBuffers, views: ObservationBuffers
) -> np.ndarray:
    """Original coordinates in compact order, directly from each segment's width."""
    parts = []
    start = 0
    for key in (*OBJECT_VALIDITY_KEYS, "stack_valid"):
        if key in obs:
            parts.append(start + np.arange(views[key].shape[1]))
            start += obs[key].shape[1]
    return np.concatenate(parts)


@pytest.mark.parametrize(
    "kind",
    [
        "default",
        "masked_mean",
        "history",
        "wdl",
        "film",
        "no_attention",
        "two_layers",
        "hashed",
        "no_world",
        "v10",
    ],
)
@pytest.mark.parametrize("bucketed", [False, True])
def test_prepared_joins_match_admitted_rows(kind: str, bucketed: bool) -> None:
    space, obs = _observations(10 if kind == "v10" else 9, kind == "history")
    config = ModelConfig(
        hidden_dim=8,
        num_attention_heads=2,
        world=None if kind == "no_world" else managym.WORLD_SETUP,
        value_aggregation="masked_mean" if kind == "masked_mean" else "historical_mean",
        recent_events=kind == "history",
        value_kind="categorical_wdl" if kind == "wdl" else "scalar",
        policy_film=kind == "film",
        attention_on=kind != "no_attention",
        attention_layers=2 if kind == "two_layers" else 1,
        record_vocabulary="hashed" if kind == "hashed" else "catalog",
    )
    agent = ActorCritic(space, config, bucketed_shapes=bucketed)
    tensors = {key: torch.from_numpy(value) for key, value in obs.items()}
    tables = _tables(agent)
    spec = Spec(
        config,
        space,
        tables,
        0 if agent.card_definitions is None else agent.card_definitions.num_embeddings,
    )
    check(obs, spec, obs["actions_valid"].astype(bool))
    device_input(agent, tensors)
    measured = (
        bucket_observation(obs, space, objects=agent.object_trimming())
        if bucketed
        else None
    )
    assert (measured is not None) == bucketed
    trimmed = measured.rows if measured is not None else obs
    slots = original_object_slots(obs, trimmed)
    prepared = prepare(
        measured or obs,
        tables,
        binding=spec.binding,
        attention="live_tokens",
    )
    expected_cards = tables.rows(trimmed["semantic_cards"])
    expected_permanents = tables.rows(
        np.stack(
            [
                trimmed[key][..., PERMANENT_DEFINITION_INDEX]
                for key in ("agent_permanents", "opponent_permanents")
            ],
            axis=1,
        )
    )
    np.testing.assert_array_equal(prepared.card_rows, expected_cards)
    np.testing.assert_array_equal(prepared.permanent_rows, expected_permanents)
    np.testing.assert_array_equal(
        prepared.known_rows, tables.rows(trimmed["known_hand"][..., 0])
    )
    np.testing.assert_array_equal(prepared.known_counts, trimmed["known_hand"][..., 1])
    np.testing.assert_array_equal(prepared.legal, obs["actions_valid"].astype(bool))
    b, a = np.nonzero(prepared.legal)
    # Every legal row in order, read by its readings.
    np.testing.assert_array_equal(
        prepared.action_features,
        read_numeric(obs["actions"][b, a], agent.readings["actions"]),
    )
    objects = prepared.validity.shape[1]
    focus = trimmed["action_focus"][b, a]
    np.testing.assert_array_equal(
        prepared.focus, b[:, None] * objects + np.searchsorted(slots, focus.clip(min=0))
    )
    np.testing.assert_array_equal(
        prepared.live_index, np.flatnonzero(prepared.validity)
    )
    np.testing.assert_array_equal(
        prepared.live_offsets, np.r_[0, (prepared.validity != 0).sum(1).cumsum()]
    )
    ownership = agent._is_agent_template[:, slots]
    np.testing.assert_array_equal(prepared.is_agent, ownership.expand(3, -1).numpy())
    compact = prepare(
        measured or obs,
        tables,
        attention="used_columns",
        facts="padded" if bucketed else "used_columns",
    )
    assert compact.facts.kinds.shape[1] == (
        trimmed["facts"].shape[1]
        if bucketed
        else (trimmed["facts"][..., FACT_VALID_INDEX] == 1).any(0).sum()
    )
    if kind == "v10":
        turn_state = read_numeric(
            trimmed["player_turn_state"][:, :1], agent.readings["player_turn_state"]
        )
        np.testing.assert_array_equal(
            prepared.agent_player[..., -turn_state.shape[-1] :], turn_state
        )
    if not tables.catalog_records:
        facts = np.where(
            (trimmed["facts"][..., FACT_VALID_INDEX] == 1)[..., None],
            trimmed["facts"],
            0,
        )
        detail = (facts[..., FACT_DETAIL_INDEX].astype(np.int64) << 16) | facts[
            ..., FACT_DETAIL_INDEX + 1
        ].astype(np.int64)
        np.testing.assert_array_equal(prepared.facts.detail_rows, detail % 4096)
        np.testing.assert_array_equal(
            prepared.facts.source_rows,
            facts[..., FACT_SOURCE_INDEX].astype(np.int64) % 4096,
        )
        assert prepared.facts.columns.shape[-1] == agent.facts.numeric.in_features == 7
    if kind == "history":
        assert prepared.events is not None
        np.testing.assert_array_equal(
            prepared.events.validity, trimmed["events_valid"].astype(bool)
        )
    assert prepared.binding == spec.binding


def test_padded_numeric_features_match_dense_readings_at_each_slot() -> None:
    space, original = _observations(11, False)
    obs = {key: rows.copy() for key, rows in original.items()}
    agent = ActorCritic(space, ModelConfig(hidden_dim=8, num_attention_heads=2))

    # External object rows can carry numeric data even in a masked slot.
    for key in ("opponent_cards", "agent_permanents", "stack"):
        assert obs[key + "_valid"][1, -1] == 0
        obs[key][1, -1, column(key, "power")] = -0.3

    # Relation padding is ignored. Include an isolated live row after a gap.
    relations = obs["relations"]
    relations.fill(0)
    relations[:, -2:, column("relations", "mana_generic")] = 3
    relations[:2, -1, column("relations", "valid")] = 1
    relations[:2, -1, column("relations", "frame")] = 2
    relations[:2, -1, column("relations", "parent")] = 1
    relations[1, -1, column("relations", "frame_kind")] = 7

    prepared = agent.prepare(obs)
    for seat, key in enumerate(("agent_cards", "opponent_cards")):
        numeric = read_numeric(obs[key], agent.readings[key])
        expected = np.concatenate(
            (numeric, obs["card_turn_state"][:, seat, :, None]), -1
        )
        np.testing.assert_array_equal(getattr(prepared, key), expected)
    for seat, key in enumerate(("agent_permanents", "opponent_permanents")):
        np.testing.assert_array_equal(
            prepared.permanents[:, seat], read_numeric(obs[key], agent.readings[key])
        )
    np.testing.assert_array_equal(
        prepared.stack, read_numeric(obs["stack"], agent.readings["stack"])
    )
    live = relations[..., column("relations", "valid")] == 1
    expected_relations = np.where(live[..., None], relations, 0)
    for name in ("frame", "parent"):
        expected_relations[1, -1, column("relations", name)] = 0
    np.testing.assert_array_equal(
        prepared.relation_features,
        read_numeric(expected_relations, agent.readings["relations"]),
    )


def test_belief_rows_extend_all_flat_object_coordinates() -> None:
    space, original = _observations(9, False)
    obs = dict(original)
    obs.update(
        {
            "belief_card_def_ids": np.array([[1, 2]] * 3),
            "belief_owner_role_ids": np.array([[0, 1]] * 3),
            "belief_hidden_zone_ids": np.array([[1, 2]] * 3),
            "belief_count_probabilities": np.array(
                [[[0.2, 0.8], [1, 0]]] * 3, np.float32
            ),
            "belief_validity": np.array([[1, 0]] * 3, np.float32),
            "belief_globals": np.zeros((3, 2), np.float32),
        }
    )
    config = ModelConfig(
        hidden_dim=8,
        num_attention_heads=2,
        belief_count_buckets=2,
        belief_card_vocab_size=3,
    )
    agent = ActorCritic(space, config)
    tables = _tables(agent)
    spec = Spec(config, space, tables, agent.card_definitions.num_embeddings)
    check(obs, spec, obs["actions_valid"].astype(bool))
    batch = prepare(obs, tables, attention="live_tokens", value_token=True)
    assert batch.belief is not None
    visible_objects = sum(original[key].shape[1] for key in OBJECT_VALIDITY_KEYS)
    assert batch.validity.shape[1] == visible_objects + 3
    np.testing.assert_array_equal(batch.validity[:, -3:], [[1, 0, 1]] * 3)
    np.testing.assert_array_equal(batch.is_agent[:, -3:], [[True, False, False]] * 3)
    b, a = np.nonzero(batch.legal)
    np.testing.assert_array_equal(
        batch.focus,
        b[:, None] * batch.validity.shape[1] + obs["action_focus"][b, a].clip(min=0),
    )
    assert batch.live_offsets[-1] == np.count_nonzero(batch.validity) + batch.batch
    obs["belief_count_probabilities"] = np.zeros((3, 2, 2), np.float32)
    with pytest.raises(ValueError, match="valid belief rows must be normalized"):
        check(obs, spec, obs["actions_valid"].astype(bool))


@pytest.mark.parametrize("legacy", [False, True])
def test_bucketed_prepare_translates_only_extracted_joins(legacy: bool) -> None:
    space = ObservationSpace(ObservationConfig(policy_history_version=1))
    env = shared_effect_root(1)
    encoded = encode_observation(
        space, env.observation_for_player(env.current_agent_index())
    )
    obs = {key: np.stack([value, value]) for key, value in encoded.items()}
    # Join across removed padding to the first opposing permanent.
    target = (
        2
        + 2 * space.config.max_cards_per_player
        + space.config.max_permanents_per_player
    )
    obs["opponent_permanents"][:, 0, column("opponent_permanents", "blocking")] = (
        target + 1
    )
    obs["stack"][:, 0, column("stack", "source")] = target + 1
    obs["action_focus"][obs["actions_valid"] == 1] = [-1, target]
    events = obs["events"]
    obs["events_valid"][:, 0] = 1
    events[:, 0, column("events", "source_context")] = -1
    events[:, 0, column("events", "target_context")] = target
    relations = obs["relations"]
    for slot, kind in enumerate((7, 8, 10, 2)):
        for name, value in (
            ("valid", 1),
            ("frame", slot + 1),
            ("frame_kind", kind),
            ("reference", target + 1),
        ):
            relations[:, slot, column("relations", name)] = value
    relations[:, 4, column("relations", "reference")] = np.nan
    config = ModelConfig(
        hidden_dim=8,
        num_attention_heads=2,
        world=managym.WORLD_SETUP,
        recent_events=True,
        value_aggregation="masked_mean",
    )
    readings = legacy_readings(space) if legacy else fresh_readings(space)
    agent = ActorCritic(space, config, bucketed_shapes=True, readings=readings)
    actual = agent.prepare(obs)
    measured = bucket_observation(obs, space, objects=True)
    assert measured is not None and measured.buckets.trims_objects
    # Independently renumber raw coordinates by their positions among kept slots.
    slots = original_object_slots(obs, measured.rows)
    compact = {key: value.copy() for key, value in measured.rows.items()}
    for key, fields in space.layout.object_references.items():
        for index, base in fields:
            values = compact[key] if index is None else compact[key][..., index]
            present = values >= base
            values[present] = np.searchsorted(slots, values[present] - base) + base
    compact["actions_valid"] = obs["actions_valid"]
    expected = prepare(
        compact,
        agent._admission_spec().tables,
        binding=actual.binding,
        value_token=agent.value_token is not None,
    )
    expected = replace(expected, bucket_needs=measured.needs)
    np.testing.assert_equal(asdict(actual), asdict(expected))
    # A pending frame masks its positional columns even for saved scalar readings.
    trimmed = compact["relations"]
    dense = np.where(
        (trimmed[..., column("relations", "valid")] == 1)[..., None], trimmed, 0
    )
    dense[:, :3, column("relations", "frame") : column("relations", "frame_kind")] = 0
    np.testing.assert_array_equal(
        actual.relation_features,
        read_numeric(dense, readings["relations"]),
        strict=True,
    )
    assert obs["stack"][0, 0, column("stack", "source")] == target + 1


@pytest.mark.parametrize("catalog", [False, True])
def test_effect_routes_preserve_memberships_and_padding(catalog: bool) -> None:
    space = ObservationSpace()
    env = shared_effect_root(1)
    encoded = encode_observation(
        space, env.observation_for_player(env.current_agent_index())
    )
    obs = {key: np.stack((value, value)) for key, value in encoded.items()}
    config = ModelConfig(
        hidden_dim=8,
        num_attention_heads=2,
        world=managym.WORLD_SETUP,
        record_vocabulary="catalog" if catalog else "hashed",
    )
    agent = ActorCritic(space, config)
    tables = _tables(agent)
    check(obs, Spec(config, space, tables, 0), obs["actions_valid"].astype(bool))
    batch = prepare(obs, tables)
    effects = batch.effects
    assert effects is not None
    records = np.argwhere(obs["effects"][..., 17] == 1)
    expected_routes: list[int] = []
    expected_records: list[int] = []
    for i, (b, slot) in enumerate(records):
        header = obs["effects"][b, slot]
        start, count = header[15:17].astype(int)
        expected_routes.extend(
            (
                b * batch.validity.shape[1]
                + obs["effect_members"][b, start : start + count, 0].astype(int)
                - 1
            ).tolist()
        )
        expected_records.extend([i] * count)
    np.testing.assert_array_equal(effects.routes, expected_routes)
    np.testing.assert_array_equal(effects.member_records, expected_records)
    assert sum(effects.schedule.lengths) == len(effects.token_indexes)
    # Invalid header payloads are ignored before any lookup or numeric cast.
    obs["effects"][..., :17][obs["effects"][..., 17] == 0] = np.nan
    check(obs, Spec(config, space, tables, 0), obs["actions_valid"].astype(bool))
    poisoned = prepare(obs, tables).effects
    assert poisoned is not None
    np.testing.assert_array_equal(poisoned.numeric, effects.numeric)
    np.testing.assert_array_equal(poisoned.routes, effects.routes)


def test_program_integers_stay_raw_and_sequences_keep_their_boundaries() -> None:
    space, original = _observations(9, False)
    obs = {key: value.copy() for key, value in original.items()}
    obs["facts"][:] = 0
    signed = [-2147483648, 70000, -71]
    for b in range(3):
        for slot, number in enumerate(signed):
            row = obs["facts"][b, slot]
            row[FACT_OBJECT_INDEX] = 1
            row[FACT_KIND_INDEX] = FACT_CONTINUATION_PROGRAM
            row[FACT_VALID_INDEX] = 1
            row[FACT_VALUE_INDEX : FACT_VALUE_INDEX + 2] = [4, slot if slot < 2 else 0]
            word = number & 0xFFFFFFFF
            row[FACT_DETAIL_INDEX : FACT_DETAIL_INDEX + 2] = [word >> 16, word & 65535]
    config = ModelConfig(hidden_dim=8, num_attention_heads=2)
    agent = ActorCritic(space, config)
    tables = _tables(agent)
    check(
        obs,
        Spec(config, space, tables, agent.card_definitions.num_embeddings),
        obs["actions_valid"].astype(bool),
    )
    batch = prepare(obs, tables)
    programs = batch.facts.programs
    assert programs is not None
    np.testing.assert_array_equal(
        programs.numbers, np.tile(signed, 3).astype(np.float32)
    )
    assert programs.schedule.lengths == (2, 1, 2, 1, 2, 1)
    np.testing.assert_array_equal(
        programs.schedule.gather[programs.schedule.unpack],
        np.arange(9),
    )


def test_v10_object_buckets_keep_turn_rows_aligned() -> None:
    space, original = _observations(10, False)
    obs = {key: value.copy() for key, value in original.items()}
    agent = ActorCritic(
        space,
        ModelConfig(
            hidden_dim=8, num_attention_heads=2, value_aggregation="masked_mean"
        ),
        bucketed_shapes=True,
    )
    obs["card_turn_state"][:, :, 31] = 1
    measured = bucket_observation(obs, space, objects=True)
    assert measured is not None
    trimmed = measured.rows
    assert trimmed["card_turn_state"].shape[2] == trimmed["agent_cards"].shape[1]
    assert trimmed["card_turn_state"].shape[2] >= 32
    batch = prepare(measured, _tables(agent))
    np.testing.assert_array_equal(
        batch.agent_cards[..., -1], trimmed["card_turn_state"][:, 0]
    )
    np.testing.assert_array_equal(
        batch.opponent_cards[..., -1], trimmed["card_turn_state"][:, 1]
    )


@pytest.mark.parametrize("catalog", [False, True])
def test_v10_masked_mean_forward_and_gradients_survive_object_buckets(
    catalog: bool,
) -> None:
    """Turn-state rows stay attached to cards when object padding is removed."""
    torch.set_num_threads(1)
    space, original = _observations(10, False)
    obs = {key: torch.from_numpy(value.copy()) for key, value in original.items()}
    obs["player_turn_state"][:, 0, 0] = 1
    obs["card_turn_state"][:, 0, 0] = 2
    config = ModelConfig(
        hidden_dim=8,
        num_attention_heads=2,
        value_aggregation="masked_mean",
        world=managym.WORLD_SETUP if catalog else None,
    )
    torch.manual_seed(76)
    padded = ActorCritic(space, config).eval()
    bucketed = ActorCritic(space, config, bucketed_shapes=True).eval()
    bucketed.load_state_dict(padded.state_dict())
    outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for model in (padded, bucketed):
        logits, values = model(device_input(model, obs))
        outputs.append((logits, values))
        (
            logits[obs["actions_valid"].bool()].square().mean() + values.square().mean()
        ).backward()
    for left, right in zip(outputs[0], outputs[1], strict=True):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
    for left, right in zip(padded.parameters(), bucketed.parameters(), strict=True):
        if left.grad is None or right.grad is None:
            assert left.grad is right.grad
        else:
            torch.testing.assert_close(left.grad, right.grad, atol=1e-6, rtol=1e-4)
    with torch.no_grad():
        for left, right in zip(
            padded(device_input(padded, obs)),
            bucketed(device_input(bucketed, obs)),
            strict=True,
        ):
            torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("bucketed", [False, True])
def test_native_batch_preserves_prepared_rows_and_full_external_admission(
    bucketed: bool,
) -> None:
    space = ObservationSpace()
    env = shared_effect_root(2)
    branch = env.clone_env()
    branch.step(0)
    engines = [branch, env, branch]
    batch = managym.encode_observation_batch(engines, space.layout)
    dense = space.allocate(len(engines))
    counts = managym.encode_observations_into(
        engines, space.layout, dense, len(engines)
    )
    assert batch.action_counts == counts
    policy = ActorCritic(
        space,
        ModelConfig(hidden_dim=8, num_attention_heads=2, world=managym.WORLD_SETUP),
        bucketed_shapes=bucketed,
        live_tokens=True,
    )
    for inference in (False, True):
        expected = policy.prepare(dense, inference=inference)
        actual = policy.prepare(batch, inference=inference)
        np.testing.assert_equal(asdict(actual), asdict(expected))
    with pytest.raises(ValueError, match="layout differs"):
        other = ActorCritic(
            ObservationSpace(space.config.model_copy(update={"max_actions": 3})),
            policy.config,
        )
        other.check_observation(batch)
    spec = policy._admission_spec()
    assert spec.tables.definition_rows is not None
    lookup = spec.tables.definition_rows.copy()
    # The native definition fits the runtime catalog, but this checkpoint cannot read it.
    used = dense["agent_permanents"][..., PERMANENT_DEFINITION_INDEX]
    lookup[int(used[used > 0][0])] = -1
    with pytest.raises(ValueError, match="unadmitted semantic definition"):
        admit_native(
            batch,
            replace(spec, tables=replace(spec.tables, definition_rows=lookup)),
            bucketed=bucketed,
        )
    dense["actions_valid"][0, 0] = 0.5
    dense["actions_valid"].flags.writeable = False
    with pytest.raises(ValueError, match="invalid_action_mask"):
        policy.prepare(dense)


def test_bucketed_attention_columns_pad_used_columns_to_a_bucket() -> None:
    used = np.array([0, 3, 5, 40, 41])
    columns = select_attention_columns(used, 234, bucketed=True)
    assert columns is not None and len(columns) == MIN_BUCKET
    assert set(used) <= set(columns)
    assert (np.diff(columns) > 0).all()
    many = np.arange(0, 2 * 40, 2)
    assert len(select_attention_columns(many, 234, bucketed=True)) == 64
    np.testing.assert_array_equal(
        select_attention_columns(used, 234, bucketed=False), used
    )
    # Dense attention when the bucket fills the table, it is tiny, or it is empty.
    assert select_attention_columns(np.arange(70), 128, bucketed=True) is None
    assert select_attention_columns(np.array([1, 2]), 8, bucketed=True) is None
    assert select_attention_columns(np.array([], np.int64), 234, bucketed=True) is None
    assert select_attention_columns(np.arange(234), 234, bucketed=False) is None


def test_bucketed_columns_add_only_empty_columns_on_native_and_dense_rows() -> None:
    space = ObservationSpace()
    env = shared_effect_root(2)
    branch = env.clone_env()
    branch.step(0)
    engines = [branch, env, branch]
    native = managym.encode_observation_batch(engines, space.layout)
    dense = space.allocate(len(engines))
    managym.encode_observations_into(engines, space.layout, dense, len(engines))
    agent = ActorCritic(
        space,
        ModelConfig(hidden_dim=8, num_attention_heads=2, world=managym.WORLD_SETUP),
    )
    tables = _tables(agent)
    spans = observation_spans(native, space, bucketed=False, objects=False)
    batch = prepare(spans, tables, attention="bucketed_columns")
    np.testing.assert_equal(
        asdict(batch), asdict(prepare(dense, tables, attention="bucketed_columns"))
    )
    columns = batch.attention_columns
    assert columns is not None
    assert len(columns) >= MIN_BUCKET and len(columns) & (len(columns) - 1) == 0
    used = np.flatnonzero((batch.validity != 0).any(0))
    assert set(used) <= set(columns)
    assert not batch.validity[:, np.setdiff1d(columns, used)].any()
    padded = prepare(dense, tables)
    assert padded.attention_columns is None
    np.testing.assert_equal(
        asdict(replace(batch, attention_columns=None)), asdict(padded)
    )
    with pytest.raises(ValueError, match="attention layout"):
        prepare(dense, tables, attention="bucketed")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fact layout"):
        prepare(dense, tables, facts="bucketed")  # type: ignore[arg-type]


def test_torch_learners_and_dense_attention_prepare_no_columns() -> None:
    space, obs = _observations(11, False)
    config = ModelConfig(hidden_dim=8, num_attention_heads=2)
    agent = ActorCritic(space, config)
    assert agent.prepare(obs).attention_columns is None
    assert agent.prepare(obs, inference=True).attention_columns is not None
    off = ActorCritic(space, config.model_copy(update={"attention_on": False}))
    assert off.prepare(obs, inference=True).attention_columns is None
    live = ActorCritic(space, config, live_tokens=True).prepare(obs, inference=True)
    assert live.attention_columns is None and live.live_index is not None


def _belief_observations() -> tuple[ObservationSpace, dict[str, np.ndarray]]:
    space, original = _observations(9, False)
    obs = dict(original)
    obs.update(
        {
            "belief_card_def_ids": np.array([[1, 2]] * 3),
            "belief_owner_role_ids": np.array([[0, 1]] * 3),
            "belief_hidden_zone_ids": np.array([[1, 2]] * 3),
            "belief_count_probabilities": np.array(
                [[[0.2, 0.8], [1, 0]]] * 3, np.float32
            ),
            "belief_validity": np.array([[1, 0]] * 3, np.float32),
            "belief_globals": np.zeros((3, 2), np.float32),
        }
    )
    return space, obs


DEVICES = ["cpu", *(["mps"] if torch.backends.mps.is_available() else [])]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "kind",
    [
        "historical_mean",
        "masked_mean",
        "value_token",
        "attention_readout",
        "bucketed_shapes",
        "belief",
    ],
)
def test_bucketed_columns_keep_logits_and_values(kind: str, device: str) -> None:
    torch.manual_seed(0)
    if kind == "belief":
        space, obs = _belief_observations()
        extra = {"belief_count_buckets": 2, "belief_card_vocab_size": 3}
    else:
        space, obs = _observations(11, False)
        extra = {}
    aggregations = ("masked_mean", "value_token", "attention_readout")
    config = ModelConfig(
        hidden_dim=16,
        num_attention_heads=2,
        attention_layers=2,
        value_aggregation=kind if kind in aggregations else "historical_mean",
        **extra,
    )
    agent = ActorCritic(space, config, bucketed_shapes=kind == "bucketed_shapes")
    tables = _tables(agent)
    agent.to(device)
    rows = (
        bucket_observation(obs, space, objects=agent.object_trimming())
        if kind == "bucketed_shapes"
        else obs
    )
    assert rows is not None

    def run(attention: AttentionLayout) -> tuple[np.ndarray, np.ndarray, Prepared]:
        batch = prepare(
            rows,
            tables,
            attention=attention,
            value_token=agent.value_token is not None,
            binding=agent._binding(),
        )
        logits, values = agent.evaluate(batch)
        return logits, values, batch

    logits, values, padded = run("padded")
    compact_logits, compact_values, compact = run("bucketed_columns")
    assert compact.attention_columns is not None
    assert len(compact.attention_columns) < padded.validity.shape[1]
    legal = padded.legal
    np.testing.assert_array_equal(compact_logits[~legal], logits[~legal])
    np.testing.assert_allclose(
        compact_logits[legal], logits[legal], atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(compact_values, values, atol=1e-5, rtol=1e-5)
