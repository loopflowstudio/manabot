"""Host preparation preserves the admitted observation's joins and coordinates."""

from dataclasses import asdict, replace
from functools import lru_cache

import numpy as np
import pytest
import torch

from manabot.backends.torch.arrays import on_device
from manabot.backends.torch.policy import ActorCritic
from manabot.belief.encoding import BeliefRows
from manabot.env.match import Match
from manabot.env.observation import (
    FACT_CONTINUATION_PROGRAM,
    PERMANENT_DEFINITION_INDEX,
    ObservationSpace,
    column,
    encode_policy_batch,
)
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.policy.admission import Spec, admit
from manabot.policy.buckets import MIN_BUCKET, bucket
from manabot.policy.prepare import (
    OBJECT_KEYS,
    OBJECT_VALIDITY_KEYS,
    AttentionLayout,
    Layout,
    Prepared,
    Tables,
    prepare,
    select_attention_columns,
)
from manabot.policy.readings import fresh_readings, legacy_readings, read_numeric
from manabot.policy.spans import ObservationSpans, RowSpans, observation_spans
from manabot.verify.representation import shared_effect_root
from manabot.verify.text_observation import priority_root
import managym
from tests.model.prepared_routes import device_input


@lru_cache(maxsize=3)
def _observations(history: bool) -> tuple[ObservationSpace, managym.ObservationBatch]:
    space = ObservationSpace(ObservationConfig(policy_history_version=int(history)))
    env = managym.Env(seed=11)
    env.reset(Match().to_rust())
    observations: list[managym.Observation] = []
    for _ in range(3):
        observations.append(env.observation_for_player(env.current_agent_index()))
        env.step(0)
    return space, encode_policy_batch(space, observations)


def _effect_batch(space: ObservationSpace, rows: int = 2) -> managym.ObservationBatch:
    env = shared_effect_root(1)
    obs = env.observation_for_player(env.current_agent_index())
    return encode_policy_batch(space, [obs] * rows)


def _tables(agent: ActorCritic) -> Tables:
    return Tables.from_buffers(
        {name: value.detach().numpy() for name, value in agent.named_buffers()},
        Layout(agent.readings),
    )


def _spans(
    batch: managym.ObservationBatch,
    space: ObservationSpace,
    *,
    bucketed: bool = False,
    objects: bool = True,
) -> ObservationSpans:
    return observation_spans(batch, space, bucketed=bucketed, objects=objects)


def kept_object_slots(space: ObservationSpace, spans: ObservationSpans) -> np.ndarray:
    """Original coordinates in compact order, directly from each segment's width."""
    parts = []
    start = 0
    for key in OBJECT_KEYS:
        parts.append(start + np.arange(spans.capacities[key]))
        start += space.shapes[key][0]
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
    ],
)
@pytest.mark.parametrize("bucketed", [False, True])
def test_prepared_joins_match_admitted_rows(kind: str, bucketed: bool) -> None:
    space, obs = _observations(kind == "history")
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
    tables = _tables(agent)
    spec = Spec(
        config,
        space,
        tables,
        0 if agent.card_definitions is None else agent.card_definitions.num_embeddings,
    )
    device_input(agent, obs)
    measured = admit(obs, spec, bucketed=bucketed)
    assert (measured.buckets is not None) == bucketed
    full = _spans(obs, space)
    slots = kept_object_slots(space, measured)
    prepared = prepare(measured, tables, binding=spec.binding, attention="live_tokens")

    def trimmed(key: str) -> np.ndarray:
        return measured.dense(key)

    expected_cards = tables.rows(trimmed("semantic_cards"))
    expected_permanents = tables.rows(
        np.stack(
            [
                trimmed(key)[..., PERMANENT_DEFINITION_INDEX]
                for key in ("agent_permanents", "opponent_permanents")
            ],
            axis=1,
        )
    )
    np.testing.assert_array_equal(prepared.card_rows, expected_cards)
    np.testing.assert_array_equal(prepared.permanent_rows, expected_permanents)
    np.testing.assert_array_equal(
        prepared.known_rows, tables.rows(trimmed("known_hand")[..., 0])
    )
    np.testing.assert_array_equal(prepared.known_counts, trimmed("known_hand")[..., 1])
    legal = full.dense("actions_valid").astype(bool)
    np.testing.assert_array_equal(prepared.legal, legal)
    b, a = np.nonzero(legal)
    # Every legal row in order, read by its readings.
    readings = agent.readings["actions"]
    np.testing.assert_array_equal(
        prepared.action_features,
        read_numeric(full.dense("actions")[b, a][:, : len(readings)], readings),
    )
    objects = prepared.validity.shape[1]
    focus = trimmed("action_focus")[b, a]
    np.testing.assert_array_equal(
        prepared.focus, b[:, None] * objects + np.searchsorted(slots, focus.clip(min=0))
    )
    np.testing.assert_array_equal(
        prepared.live_index, np.flatnonzero(prepared.validity)
    )
    np.testing.assert_array_equal(
        prepared.live_offsets, np.r_[0, (prepared.validity != 0).sum(1).cumsum()]
    )
    stack = full.dense("stack")
    fixed_owners = agent._is_agent_template.expand(len(stack), -1).numpy()
    owners = np.concatenate(
        (fixed_owners, stack[..., column("stack", "is_mine")] > 0), axis=1
    )
    np.testing.assert_array_equal(prepared.is_agent, owners[:, slots])
    compact = prepare(
        measured,
        tables,
        attention="used_columns",
        facts="padded" if bucketed else "used_columns",
    )
    assert compact.facts.kinds.shape[1] == (
        measured.capacities["facts"]
        if bucketed
        else int(measured.tables["facts"].counts.max(initial=0))
    )
    turn_state = read_numeric(
        trimmed("player_turn_state")[:, :1], agent.readings["player_turn_state"]
    )
    np.testing.assert_array_equal(
        prepared.agent_player[..., -turn_state.shape[-1] :], turn_state
    )
    if kind == "history":
        assert prepared.events is not None
        np.testing.assert_array_equal(
            prepared.events.validity, trimmed("events_valid").astype(bool)
        )
    assert prepared.binding == spec.binding


def test_compact_attention_places_live_objects_and_masks_shorter_boards() -> None:
    space = ObservationSpace()
    obs = encode_policy_batch(
        space,
        [
            priority_root(("Lightning Bolt",), lands).scenario_refresh()
            for lands in (("Mountain",), ("Mountain", "Gray Ogre"))
        ],
    )
    spans = _spans(obs, space)
    tables = _tables(
        ActorCritic(space, ModelConfig(hidden_dim=8, num_attention_heads=2))
    )
    prepared = prepare(spans, tables, attention="live_tokens", value_token=True)
    compact = prepared.compact_attention
    assert compact is not None and prepared.live_index is not None

    # A value token follows the last object slot, even on the shorter board.
    present = np.pad(prepared.validity, ((0, 0), (0, 1)), constant_values=1)
    expected = [np.flatnonzero(row) + b * len(row) for b, row in enumerate(present)]
    longest = max(map(len, expected))
    assert len(expected[0]) < longest
    slots = np.full((obs.batch_size, longest), -1)
    slots[compact.batches, compact.slots] = prepared.live_index
    np.testing.assert_array_equal(
        slots,
        [np.pad(row, (0, longest - len(row)), constant_values=-1) for row in expected],
    )
    np.testing.assert_array_equal(
        compact.key_mask, np.where(slots >= 0, 0.0, -np.inf)[:, None, None, :]
    )
    assert on_device(prepared, "cpu").compact_attention is compact
    assert prepare(spans, tables).compact_attention is None


def test_padded_numeric_features_match_dense_readings_at_each_slot() -> None:
    space, obs = _observations(False)
    spans = _spans(obs, space)
    agent = ActorCritic(space, ModelConfig(hidden_dim=8, num_attention_heads=2))
    prepared = agent.prepare(obs)

    def readings(key: str) -> np.ndarray:
        values = spans.dense(key)
        return read_numeric(
            values[..., : len(agent.readings[key])], agent.readings[key]
        )

    turn_state = spans.dense("card_turn_state")
    for seat, key in enumerate(("agent_cards", "opponent_cards")):
        expected = np.concatenate((readings(key), turn_state[:, seat, :, None]), -1)
        np.testing.assert_array_equal(getattr(prepared, key), expected)
    for seat, key in enumerate(("agent_permanents", "opponent_permanents")):
        np.testing.assert_array_equal(prepared.permanents[:, seat], readings(key))
    np.testing.assert_array_equal(prepared.stack, readings("stack"))
    np.testing.assert_array_equal(prepared.relation_features, readings("relations"))


def belief_batch() -> tuple[
    ObservationSpace, managym.ObservationBatch, BeliefRows[np.ndarray]
]:
    space = ObservationSpace()
    game = managym.Env(seed=11)
    raw, _ = game.reset(Match().to_rust())
    obs = managym.encode_observation_batch([raw, raw, raw], space.layout)
    belief = BeliefRows(
        cards=np.array([[1, 2]] * 3),
        owners=np.array([[0, 1]] * 3),
        zones=np.array([[1, 2]] * 3),
        probabilities=np.array([[[0.2, 0.8], [1, 0]]] * 3, np.float32),
        validity=np.array([[0, 1], [0, 0], [0, 1]], np.float32),
        globals=np.zeros((3, 2), np.float32),
    )
    return space, obs, belief


def test_belief_rows_extend_all_flat_object_coordinates() -> None:
    space, obs, belief = belief_batch()
    config = ModelConfig(
        hidden_dim=8,
        num_attention_heads=2,
        belief_count_buckets=2,
        belief_card_vocab_size=3,
    )
    agent = ActorCritic(space, config)
    tables = _tables(agent)
    spec = Spec(config, space, tables, agent.card_definitions.num_embeddings)
    spans = admit(obs, spec, bucketed=False, belief=belief)
    batch = prepare(
        spans, tables, belief=belief, attention="live_tokens", value_token=True
    )
    assert batch.belief is belief
    visible_objects = sum(space.shapes[key][0] for key in OBJECT_VALIDITY_KEYS)
    assert batch.validity.shape[1] == visible_objects + 3
    np.testing.assert_array_equal(
        batch.validity[:, -3:], [[0, 1, 1], [0, 0, 1], [0, 1, 1]]
    )
    np.testing.assert_array_equal(batch.is_agent[:, -3:], [[True, False, False]] * 3)
    decisions, offers = np.nonzero(batch.legal)
    focus = spans.dense("action_focus")[decisions, offers]
    np.testing.assert_array_equal(
        batch.focus, decisions[:, None] * batch.validity.shape[1] + focus.clip(min=0)
    )
    expected = np.column_stack((batch.validity, np.ones(batch.batch)))
    np.testing.assert_array_equal(batch.live_index, np.flatnonzero(expected))
    np.testing.assert_array_equal(np.diff(batch.live_offsets), expected.sum(1))
    for attention in ("used_columns", "bucketed_columns"):
        columns = prepare(
            spans, tables, belief=belief, attention=attention
        ).attention_columns
        assert columns is not None
        assert set(np.flatnonzero(batch.validity.any(0))).issubset(columns)
    with pytest.raises(ValueError, match="valid belief rows must be normalized"):
        agent.prepare(
            obs,
            belief=replace(belief, probabilities=np.zeros_like(belief.probabilities)),
        )
    with pytest.raises(ValueError, match="batch size"):
        agent.prepare(obs, belief=replace(belief, cards=belief.cards[:1]))
    with pytest.raises(ValueError, match="binary"):
        agent.prepare(obs, belief=replace(belief, validity=belief.validity + 0.5))
    with pytest.raises(ValueError, match="belief-free"):
        admit(obs, replace(spec, model=ModelConfig()), bucketed=False, belief=belief)


@pytest.mark.parametrize("legacy", [False, True])
def test_bucketed_prepare_translates_only_extracted_joins(legacy: bool) -> None:
    """Every object join of a trimmed batch names the same object as untrimmed."""
    space = ObservationSpace(ObservationConfig(policy_history_version=1))
    obs = _effect_batch(space)
    config = ModelConfig(
        hidden_dim=8,
        num_attention_heads=2,
        world=managym.WORLD_SETUP,
        recent_events=True,
        value_aggregation="masked_mean",
    )
    readings = legacy_readings(space) if legacy else fresh_readings(space)
    padded = ActorCritic(space, config, readings=readings)
    agent = ActorCritic(space, config, bucketed_shapes=True, readings=readings)
    expected = padded.prepare(obs)
    actual = agent.prepare(obs)
    measured = _spans(obs, space, bucketed=True)
    assert measured.buckets is not None and measured.buckets.trims_objects
    slots = kept_object_slots(space, measured)
    full_objects = expected.validity.shape[1]
    objects = actual.validity.shape[1]
    assert objects == len(slots) < full_objects

    def translated(flat: np.ndarray) -> np.ndarray:
        batch, slot = np.divmod(np.asarray(flat), full_objects)
        assert np.isin(slot, slots).all()
        return batch * objects + np.searchsorted(slots, slot)

    present = expected.focus_mask > 0
    np.testing.assert_array_equal(
        actual.focus[present], translated(expected.focus[present])
    )
    assert expected.effects is not None and actual.effects is not None
    for name in ("controller", "source_object", "routes"):
        mask = getattr(expected.effects, f"{name}_mask", None)
        values = getattr(expected.effects, name)
        live = np.ones(len(values), bool) if mask is None else mask > 0
        assert live.any() or name == "source_object", name
        np.testing.assert_array_equal(
            getattr(actual.effects, name)[live], translated(values[live]), name
        )
    width = measured.capacities["agent_permanents"]
    for targets, masks, kept in zip(
        expected.link_targets, expected.link_masks, actual.link_targets, strict=True
    ):
        full = targets.reshape(obs.batch_size, 2, -1)[..., :width]
        linked = masks.reshape(obs.batch_size, 2, -1)[..., :width] > 0
        np.testing.assert_array_equal(
            kept.reshape(obs.batch_size, 2, width)[linked], translated(full[linked])
        )
    np.testing.assert_array_equal(actual.legal, expected.legal)
    np.testing.assert_array_equal(actual.action_features, expected.action_features)
    kept_relations = actual.relation_features.shape[1]
    np.testing.assert_array_equal(
        actual.relation_features, expected.relation_features[:, :kept_relations]
    )
    assert not expected.relation_features[:, kept_relations:].any()


@pytest.mark.parametrize("catalog", [False, True])
def test_effect_routes_preserve_memberships(catalog: bool) -> None:
    space = ObservationSpace()
    obs = _effect_batch(space)
    config = ModelConfig(
        hidden_dim=8,
        num_attention_heads=2,
        world=managym.WORLD_SETUP,
        record_vocabulary="catalog" if catalog else "hashed",
    )
    agent = ActorCritic(space, config)
    tables = _tables(agent)
    spans = admit(obs, Spec(config, space, tables, 0), bucketed=False)
    batch = prepare(spans, tables)
    effects = batch.effects
    assert effects is not None
    headers = spans.tables["effects"]
    members = spans.tables["effect_members"]
    record_batches, _ = headers.coordinates
    expected_routes: list[int] = []
    expected_records: list[int] = []
    for i, (b, header) in enumerate(zip(record_batches, headers.integers, strict=True)):
        start = header[column("effects", "member_offset")]
        count = header[column("effects", "member_count")]
        owned = members.group(int(b))[start : start + count]
        expected_routes.extend(
            (b * batch.validity.shape[1] + owned[:, 0].astype(int) - 1).tolist()
        )
        expected_records.extend([i] * count)
    assert expected_records
    np.testing.assert_array_equal(effects.routes, expected_routes)
    np.testing.assert_array_equal(effects.member_records, expected_records)
    assert sum(effects.schedule.lengths) == len(effects.token_indexes)


def _with_table(spans: ObservationSpans, key: str, table: RowSpans) -> ObservationSpans:
    return replace(spans, tables={**spans.tables, key: table})


def test_program_integers_stay_raw_and_sequences_keep_their_boundaries() -> None:
    space, obs = _observations(False)
    spans = _spans(obs, space)
    facts = spans.tables["facts"]
    signed = [-2147483648, 70000, -71]
    records = np.zeros(3 * len(signed), facts.layout.dtype)
    records["object"] = 1
    records["kind"] = FACT_CONTINUATION_PROGRAM
    records["valid"] = 1
    records["value"] = 4
    records["value_2"] = np.tile([0, 1, 0], 3)
    records["detail"] = np.tile(signed, 3)
    programs_input = _with_table(
        spans,
        "facts",
        RowSpans(
            records, facts.layout, np.arange(0, 10, 3, dtype=np.int32), facts.groups
        ),
    )
    config = ModelConfig(hidden_dim=8, num_attention_heads=2)
    agent = ActorCritic(space, config)
    batch = prepare(programs_input, _tables(agent))
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


def _marked_turn_state(spans: ObservationSpans) -> ObservationSpans:
    """Distinct card turn-state values, so a misaligned card row is visible."""
    table = spans.tables["card_turn_state"]
    records = np.arange(1, len(table.records) + 1, dtype=table.records.dtype)
    return _with_table(spans, "card_turn_state", replace(table, records=records))


def test_object_buckets_keep_turn_rows_aligned() -> None:
    space, obs = _observations(False)
    agent = ActorCritic(
        space,
        ModelConfig(
            hidden_dim=8, num_attention_heads=2, value_aggregation="masked_mean"
        ),
        bucketed_shapes=True,
    )
    measured = _marked_turn_state(_spans(obs, space, bucketed=True))
    assert measured.buckets is not None and measured.buckets.trims_objects
    turn_state = measured.dense("card_turn_state")
    assert turn_state.shape[2] == measured.capacities["agent_cards"]
    assert turn_state.any()
    batch = prepare(measured, _tables(agent))
    np.testing.assert_array_equal(batch.agent_cards[..., -1], turn_state[:, 0])
    np.testing.assert_array_equal(batch.opponent_cards[..., -1], turn_state[:, 1])


@pytest.mark.parametrize("catalog", [False, True])
def test_masked_mean_forward_and_gradients_survive_object_buckets(
    catalog: bool,
) -> None:
    """Turn-state rows stay attached to cards when object padding is removed."""
    torch.set_num_threads(1)
    space, obs = _observations(False)
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
    legal = torch.from_numpy(_spans(obs, space).dense("actions_valid")).bool()

    def prepared(model: ActorCritic) -> Prepared[torch.Tensor]:
        spans = _marked_turn_state(
            _spans(
                obs,
                space,
                bucketed=model.bucketed_shapes,
                objects=model.object_trimming(),
            )
        )
        host = prepare(
            spans,
            model._admission_spec().tables,
            binding=model._binding(),
            value_token=model.value_token is not None,
        )
        return device_input(model, host)

    outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for model in (padded, bucketed):
        logits, values = model(prepared(model))
        outputs.append((logits, values))
        (logits[legal].square().mean() + values.square().mean()).backward()
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
def test_native_batch_binds_to_the_policy_layout_and_catalog(bucketed: bool) -> None:
    space = ObservationSpace()
    env = shared_effect_root(2)
    branch = env.clone_env()
    branch.step(0)
    batch = managym.encode_observation_batch([branch, env, branch], space.layout)
    policy = ActorCritic(
        space,
        ModelConfig(hidden_dim=8, num_attention_heads=2, world=managym.WORLD_SETUP),
        bucketed_shapes=bucketed,
        live_tokens=True,
    )
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
    used = _spans(batch, space).dense("agent_permanents")[
        ..., PERMANENT_DEFINITION_INDEX
    ]
    lookup[int(used[used > 0][0])] = -1
    with pytest.raises(ValueError, match="unadmitted semantic definition"):
        admit(
            batch,
            replace(spec, tables=replace(spec.tables, definition_rows=lookup)),
            bucketed=bucketed,
        )


def test_bucketed_attention_columns_pad_used_columns_to_a_bucket() -> None:
    used = np.array([0, 3, 5, 40, 41])
    columns = select_attention_columns(used, 234, bucketed=True)
    assert columns is not None and len(columns) == MIN_BUCKET
    assert set(used) <= set(columns)
    assert (np.diff(columns) > 0).all()
    many = np.arange(0, 2 * 40, 2)
    many_columns = select_attention_columns(many, 234, bucketed=True)
    assert many_columns is not None and len(many_columns) == 64
    exact = np.arange(0, 32, 2)
    np.testing.assert_array_equal(
        select_attention_columns(exact, 234, bucketed=True), exact
    )
    np.testing.assert_array_equal(
        select_attention_columns(used, 234, bucketed=False), used
    )
    # Dense attention when the bucket fills the table, it is tiny, or it is empty.
    assert select_attention_columns(np.arange(70), 128, bucketed=True) is None
    assert select_attention_columns(np.array([1, 2]), 8, bucketed=True) is None
    assert select_attention_columns(np.array([], np.int64), 234, bucketed=True) is None
    assert select_attention_columns(np.arange(234), 234, bucketed=False) is None


def test_bucketed_columns_add_only_empty_columns() -> None:
    space = ObservationSpace()
    env = shared_effect_root(2)
    branch = env.clone_env()
    branch.step(0)
    native = managym.encode_observation_batch([branch, env, branch], space.layout)
    agent = ActorCritic(
        space,
        ModelConfig(hidden_dim=8, num_attention_heads=2, world=managym.WORLD_SETUP),
    )
    tables = _tables(agent)
    spans = _spans(native, space, objects=False)
    batch = prepare(spans, tables, attention="bucketed_columns")
    columns = batch.attention_columns
    assert columns is not None
    assert len(columns) >= MIN_BUCKET and len(columns) & (len(columns) - 1) == 0
    used = np.flatnonzero((batch.validity != 0).any(0))
    assert set(used) <= set(columns)
    assert not batch.validity[:, np.setdiff1d(columns, used)].any()
    padded = prepare(spans, tables)
    assert padded.attention_columns is None
    np.testing.assert_equal(
        asdict(replace(batch, attention_columns=None)), asdict(padded)
    )
    with pytest.raises(ValueError, match="attention layout"):
        prepare(spans, tables, attention="bucketed")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fact layout"):
        prepare(spans, tables, facts="bucketed")  # type: ignore[arg-type]


def test_torch_learners_and_dense_attention_prepare_no_columns() -> None:
    space, obs = _observations(False)
    config = ModelConfig(hidden_dim=8, num_attention_heads=2)
    agent = ActorCritic(space, config)
    assert agent.prepare(obs).attention_columns is None
    cpu_batch = agent.prepare(obs, inference=True)
    assert cpu_batch.attention_columns is not None
    off = ActorCritic(space, config.model_copy(update={"attention_on": False}))
    assert off.prepare(obs, inference=True).attention_columns is None
    live = ActorCritic(space, config, live_tokens=True).prepare(obs, inference=True)
    assert live.attention_columns is None and live.live_index is not None
    if torch.backends.mps.is_available():
        columns = agent.to("mps").prepare(obs, inference=True).attention_columns
        assert columns is not None
        assert len(columns) == bucket(
            len(cpu_batch.attention_columns), cpu_batch.validity.shape[1]
        )
        assert set(cpu_batch.attention_columns) <= set(columns)
        assert agent.prepare(obs).attention_columns is None


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
    belief = None
    if kind == "belief":
        space, native, belief = belief_batch()
    else:
        space, native = _observations(False)
    aggregations = ("masked_mean", "value_token", "attention_readout")
    config = ModelConfig(
        hidden_dim=16,
        num_attention_heads=2,
        attention_layers=2,
        value_aggregation=kind if kind in aggregations else "historical_mean",
        belief_count_buckets=2 if kind == "belief" else 0,
        belief_card_vocab_size=3 if kind == "belief" else 0,
    )
    agent = ActorCritic(space, config, bucketed_shapes=kind == "bucketed_shapes")
    tables = _tables(agent)
    agent.to(device)
    rows = _spans(
        native,
        space,
        bucketed=kind == "bucketed_shapes",
        objects=agent.object_trimming(),
    )

    def run(
        attention: AttentionLayout,
    ) -> tuple[np.ndarray, np.ndarray, Prepared[np.ndarray]]:
        batch = prepare(
            rows,
            tables,
            attention=attention,
            belief=belief,
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
