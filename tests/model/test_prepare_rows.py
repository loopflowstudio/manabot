"""Host preparation preserves the admitted observation's joins and coordinates."""

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
    TURN_STATE_DIM,
    ObservationSpace,
)
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.policy.admission import Spec, check
from manabot.policy.buckets import bucket_sizes, object_columns, trim_observation
from manabot.policy.checkpoint import layout
from manabot.policy.prepare import OBJECT_VALIDITY_KEYS, Tables, prepare
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
        rows.append(space.encode(obs))
        env.step(0)
    return space, {key: np.stack([row[key] for row in rows]) for key in rows[0]}


def _tables(agent: ActorCritic) -> Tables:
    return Tables.from_buffers(
        {name: value.detach().numpy() for name, value in agent.named_buffers()},
        layout(agent.observation_space),
    )


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
    buckets = (
        bucket_sizes(
            obs, agent.observation_space.encoder, objects=agent.object_trimming()
        )
        if bucketed
        else None
    )
    trimmed = obs
    if buckets is not None:
        trimmed = trim_observation(obs, buckets)
    prepared = prepare(
        trimmed,
        tables,
        bucketed=bucketed,
        binding=spec.binding,
        action_capacity=space.encoder.max_actions,
        live_tokens=True,
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
    np.testing.assert_array_equal(prepared.action_features, obs["actions"][b, a, :-2])
    objects = prepared.validity.shape[1]
    focus = trimmed["action_focus"][b, a]
    np.testing.assert_array_equal(
        prepared.focus, b[:, None] * objects + focus.clip(min=0)
    )
    np.testing.assert_array_equal(
        prepared.live_index, np.flatnonzero(prepared.validity)
    )
    np.testing.assert_array_equal(
        prepared.live_offsets, np.r_[0, (prepared.validity != 0).sum(1).cumsum()]
    )
    ownership = agent._is_agent_template
    if buckets is not None and buckets.trims_objects:
        ownership = torch.cat(
            [ownership[:, part] for part in object_columns(buckets)], 1
        )
    np.testing.assert_array_equal(prepared.is_agent, ownership.expand(3, -1).numpy())
    compact = prepare(trimmed, tables, inference=True, bucketed=bucketed)
    assert compact.facts.kinds.shape[1] == (
        trimmed["facts"].shape[1]
        if bucketed
        else (trimmed["facts"][..., FACT_VALID_INDEX] == 1).any(0).sum()
    )
    if kind == "v10":
        np.testing.assert_array_equal(
            prepared.agent_player[..., -TURN_STATE_DIM:],
            trimmed["player_turn_state"][:, :1],
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
        assert prepared.facts.columns.shape[-1] == 5
    if kind == "history":
        assert prepared.events is not None
        np.testing.assert_array_equal(
            prepared.events.validity, trimmed["events_valid"].astype(bool)
        )
    assert prepared.binding == spec.binding


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
    batch = prepare(obs, tables, live_tokens=True, value_token=True)
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


@pytest.mark.parametrize("catalog", [False, True])
def test_effect_routes_preserve_memberships_and_padding(catalog: bool) -> None:
    space = ObservationSpace()
    env = shared_effect_root(1)
    encoded = space.encode(env.observation_for_player(env.current_agent_index()))
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
    buckets = bucket_sizes(obs, agent.observation_space.encoder, objects=True)
    assert buckets is not None
    trimmed = trim_observation(obs, buckets)
    assert trimmed["card_turn_state"].shape[2] == trimmed["agent_cards"].shape[1]
    assert trimmed["card_turn_state"].shape[2] >= 32
    batch = prepare(trimmed, _tables(agent), bucketed=True)
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
