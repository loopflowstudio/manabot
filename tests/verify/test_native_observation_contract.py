"""The native observation encoder's contract on constructed positions.

There is one encoder (`managym/src/agent/observation_encoder.rs`). These tests
assert what named columns and rows hold, that vector rows equal scalar encoding
and that capacities are respected or refused with `managym.ObservationCapacity`.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

# Local imports
from manabot.env.observation import (
    ACTION_MODE_INDEX,
    ACTION_PAYS_DIRECTLY_INDEX,
    CARD_DIM,
    FACT_COPY,
    FACT_COPY_EXCEPTION,
    FACT_COUNTER,
    FACT_DURATION_INDEX,
    FACT_ID_INDEX,
    FACT_KIND_INDEX,
    FACT_OBJECT_INDEX,
    FACT_ORDER_INDEX,
    FACT_SOURCE_INDEX,
    FACT_TRANSFORMS,
    FACT_USED_ABILITY,
    FACT_VALID_INDEX,
    FACT_VALUE_INDEX,
    PERMANENT_DIM,
    PLAYER_NO_MAXIMUM_HAND_SIZE_INDEX,
    RELATION_THIS_TURN_INDEX,
    RELATION_VALID_INDEX,
    RELATION_X_INDEX,
    ObservationConfig,
    ObservationSpace,
    column,
    step_buffers,
)
from manabot.infra.config import UnsupportedPolicyInput
from manabot.managym.enums import ActionEnum
from manabot.semantic.text_observation import capture_text_decision, submit_response
from manabot.verify.text_observation import priority_root
import managym
from tests.fixtures.observation_rows import observation_arrays, vector_arrays


def make_player_configs() -> list[managym.PlayerConfig]:
    return [
        managym.PlayerConfig("Red Mage", {"Mountain": 10, "Gray Ogre": 20}),
        managym.PlayerConfig("Green Mage", {"Forest": 10, "Llanowar Elves": 20}),
    ]


def random_valid_action(obs: managym.Observation, rng: np.random.Generator) -> int:
    return int(rng.integers(0, len(obs.action_space.actions)))


def assert_row_observation_equal(
    rows: dict[str, np.ndarray],
    env_index: int,
    encoded_obs: dict[str, np.ndarray],
) -> None:
    for key, expected in encoded_obs.items():
        np.testing.assert_array_equal(
            rows[key][env_index], expected, err_msg=f"env={env_index} key={key}"
        )


def test_focus_indices_name_hand_rows_and_pad_unused_slots() -> None:
    env = managym.Env(seed=7)
    env.reset(make_player_configs())
    env.step(0)
    obs, *_ = env.step(0)
    space = ObservationSpace()
    focus = observation_arrays(space, obs)["action_focus"]

    # This opening offers plays from hand and a pass. The hand follows both players.
    hand = [card.id for card in obs.agent_cards]
    actions = obs.action_space.actions
    assert any(action.focus for action in actions) and any(
        not action.focus for action in actions
    )
    for row, action in zip(focus[: len(actions)], actions, strict=True):
        expected = [2 + hand.index(card) for card in action.focus]
        np.testing.assert_array_equal(row[: len(expected)], expected)
        assert (row[len(expected) :] == -1).all()
    assert (focus[len(actions) :] == -1).all()


def test_restored_observation_encodes_without_a_live_game() -> None:
    env = managym.Env(seed=5, skip_trivial=True)
    space = ObservationSpace(ObservationConfig())
    obs, _ = env.reset(make_player_configs())
    restored = managym.Observation.from_json(obs.toJSON())
    del env  # a restored observation encodes with no live game

    expected = observation_arrays(space, restored)
    assert int(expected["agent_cards_valid"].sum()) == 7


def test_policy_action_count_overflow_is_a_capacity_error() -> None:
    assert managym.policy_action_count(70, 64, "first-legal") == 64
    with pytest.raises(managym.ObservationCapacity, match="actions 70 > 64"):
        managym.policy_action_count(70, 64)


def test_vector_env_buffer_path_matches_compatibility_step_path() -> None:
    rng = np.random.default_rng(1234)
    config = ObservationConfig()
    space = ObservationSpace(config)
    num_envs = 4
    player_configs = make_player_configs()

    compat = managym.VectorEnv(
        num_envs=num_envs,
        seed=123,
        skip_trivial=True,
        player_policies=("agent", "random"),
    )
    hot = managym.VectorEnv(
        num_envs=num_envs,
        seed=123,
        skip_trivial=True,
        player_policies=("agent", "random"),
    )
    buffers = step_buffers(num_envs)
    hot.set_observation_layout(buffers, layout=space.layout)

    compat_reset = compat.reset_all(player_configs)
    hot.reset_all_into_buffers(player_configs)

    for env_index, (obs, _) in enumerate(compat_reset):
        assert_row_observation_equal(
            vector_arrays(space, hot.observation_rows()),
            env_index,
            observation_arrays(space, obs),
        )
    np.testing.assert_allclose(buffers["rewards"], np.zeros(num_envs, dtype=np.float64))
    np.testing.assert_array_equal(
        buffers["terminated"], np.zeros(num_envs, dtype=np.uint8)
    )
    np.testing.assert_array_equal(
        buffers["truncated"], np.zeros(num_envs, dtype=np.uint8)
    )

    last_compat_step = compat_reset
    for _ in range(60):
        actions = [random_valid_action(obs, rng) for obs, _ in last_compat_step]
        compat_step = compat.step(actions)
        hot.step_into_buffers(actions)

        expected_rewards = np.array([step[1] for step in compat_step], dtype=np.float64)
        expected_terminated = np.array(
            [step[2] for step in compat_step], dtype=np.uint8
        )
        expected_truncated = np.array([step[3] for step in compat_step], dtype=np.uint8)
        np.testing.assert_allclose(buffers["rewards"], expected_rewards, atol=1e-12)
        np.testing.assert_array_equal(buffers["terminated"], expected_terminated)
        np.testing.assert_array_equal(buffers["truncated"], expected_truncated)

        for env_index, (obs, _, _, _, _) in enumerate(compat_step):
            assert_row_observation_equal(
                vector_arrays(space, hot.observation_rows()),
                env_index,
                observation_arrays(space, obs),
            )

        hot_infos = [dict(info) for info in hot.get_last_info()]
        compat_infos = [dict(info) for _, _, _, _, info in compat_step]
        assert hot_infos == compat_infos

        last_compat_step = [(obs, info) for obs, _, _, _, info in compat_step]
        if expected_terminated.any() or expected_truncated.any():
            # Finished slots stay parked; deal fresh games on both paths.
            last_compat_step = compat.reset_all(player_configs)
            hot.reset_all_into_buffers(player_configs)
            for env_index, (obs, _) in enumerate(last_compat_step):
                assert_row_observation_equal(
                    vector_arrays(space, hot.observation_rows()),
                    env_index,
                    observation_arrays(space, obs),
                )


def test_vector_env_requires_a_registered_layout_before_buffered_reset() -> None:
    unconfigured = managym.VectorEnv(
        num_envs=2,
        seed=1000,
        skip_trivial=True,
        player_policies=("agent", "agent"),
    )
    with pytest.raises(RuntimeError, match="set_observation_layout"):
        unconfigured.reset_all_into_buffers(make_player_configs())


def test_elemental_indestructible_encoding_preserves_outside_and_definition_columns() -> (
    None
):
    env = managym.Env(seed=161, skip_trivial=False)
    env.reset(make_player_configs())
    env.step(0)
    env.step(0)
    env.scenario_force_battlefield(0, "Gray Ogre")
    obs = env.scenario_refresh()
    cards = obs.agent_cards
    card = cards[0]
    keywords = card.keywords
    keywords.indestructible = True
    card.keywords = keywords
    cards[0] = card
    obs.agent_cards = cards
    permanents = obs.agent_permanents
    permanent = permanents[0]
    keywords = permanent.keywords
    keywords.indestructible = True
    permanent.keywords = keywords
    permanents[0] = permanent
    obs.agent_permanents = permanents

    actual = observation_arrays(ObservationSpace(), obs)
    assert actual["agent_cards"].shape[-1] == CARD_DIM
    assert actual["agent_permanents"].shape[-1] == PERMANENT_DIM
    assert actual["agent_cards"][0, 38] == 1
    assert actual["agent_cards"][0, 37] == 0
    assert actual["agent_permanents"][0, 25] == 1
    assert actual["agent_permanents"][0, 24] == permanent.registry_key + 1


def _token_low(name: str) -> int:
    """The low 16 bits of a learning-token value, as fact rows store it."""
    return managym.catalog_token_value(name) & 0xFFFF


def test_encodes_every_observed_permanent_and_card_field() -> None:
    env = managym.Env(seed=11, skip_trivial=False)
    env.reset(make_player_configs())
    env.step(0)
    env.step(0)
    env.scenario_force_battlefield(0, "Mountain")
    env.scenario_force_battlefield(1, "Forest")
    obs = env.scenario_refresh()
    mine, theirs = obs.agent_permanents[0], obs.opponent_permanents[0]
    mine.counters = {"stun": 1, "fire": 2, "double_strike": 3}
    mine.attached_to = theirs.id
    mine.name, mine.creature_types = "Humble Merchant", ["Citizen"]
    mine.is_granted_land, mine.granted_firebending = True, 4
    mine.temp_subtypes = ["Island", "Construct", "Citizen"]
    mine.attacking, mine.blocking, mine.abilities_lost = True, theirs.id, True
    mine.used_abilities = [1, 5]
    # Entry order ranks this permanent first.
    assert mine.timestamp is None and theirs.timestamp is None
    assert mine.id < theirs.id
    # Continuous-effect records travel as shared effects
    # (test_shared_effect_transport.py). Copy exceptions share the modification
    # vocabulary, but have their own fact kind and use the copied permanent's
    # timestamp and definition join.
    copy = managym.CopyEffect()
    copy.name, copy.registry_key = "Forest", theirs.registry_key
    nonlegendary = managym.ContinuousEffect()
    nonlegendary.modification = "Nonlegendary"
    nonlegendary.duration, nonlegendary.timestamp = "Indefinite", mine.id
    nonlegendary.source_registry_key = theirs.registry_key
    keywords = managym.ContinuousEffect()
    keywords.modification, keywords.keywords = "AddKeywords", ["flying", "trample"]
    keywords.duration, keywords.timestamp = "Indefinite", mine.id
    keywords.source_registry_key = theirs.registry_key
    copy.exceptions = [nonlegendary, keywords]
    mine.copy = copy
    mine.dealt_damage_this_turn = True
    obs.agent_permanents = [mine]
    agent = obs.agent
    agent.keeps_red_mana = agent.no_maximum_hand_size = True
    agent.extra_turns = 2
    agent.additional_combats = 1
    history = agent.turn_history
    history.attacked = history.land_entered = True
    history.permanents_sacrificed, history.cards_drawn = 3, 2
    agent.turn_history = history
    obs.agent = agent

    space = ObservationSpace()
    actual = observation_arrays(space, obs)
    row = actual["agent_permanents"][0]
    own = 2 + 2 * space.config.max_cards_per_player
    host = own + space.config.max_permanents_per_player
    # Attacking (column 27), then exhausted, once each turn, granted
    # firebending, attachment join, granted land, blocking join, lost
    # abilities, timestamp rank, dealt damage this turn, validity.
    np.testing.assert_allclose(
        row[27:37], [1, 0, 0, 0.4, host + 1, 1, host + 1, 1, 1, 1]
    )
    assert row[column("agent_permanents", "valid")] == 1
    # Fact rows, in order: counters, added subtypes, set name and creature
    # type, used abilities, copy and its exceptions (one row per keyword).
    facts = actual["facts"]
    kinds = facts[facts[:, FACT_VALID_INDEX] == 1, FACT_KIND_INDEX].tolist()
    assert kinds == [1, 1, 1, 2, 2, 2, 3, 4, 5, 5, 7, 8, 8, 8]
    copies = facts[np.isin(facts[:, FACT_KIND_INDEX], [FACT_COPY, FACT_COPY_EXCEPTION])]
    np.testing.assert_array_equal(copies[:, FACT_OBJECT_INDEX], [own + 1] * 4)
    np.testing.assert_array_equal(
        copies[:, FACT_SOURCE_INDEX], [theirs.registry_key + 1] * 4
    )
    np.testing.assert_array_equal(copies[:, FACT_ORDER_INDEX], [1] * 4)
    np.testing.assert_array_equal(copies[:, FACT_DURATION_INDEX], [2] * 4)
    assert not actual["effects"].any()
    np.testing.assert_allclose(
        actual["agent_player"][0, PLAYER_NO_MAXIMUM_HAND_SIZE_INDEX:],
        [1, obs.turn.turn_number / 20, 0, 1, 1, 0.3, 1, 0.2, 0, 0, 0, 0, 2, 1, 1, 0],
    )
    assert actual["agent_cards"][0, CARD_DIM - 1] == 1


@pytest.mark.parametrize("capacity", [512, 256, 140])
def test_fact_capacity_fills_exactly_and_rejects_overflow(capacity: int) -> None:
    """A full fact block keeps every row; one more is an error, never a prefix."""
    env = managym.Env(seed=11, skip_trivial=False)
    env.reset(make_player_configs())
    env.step(0)
    env.step(0)
    env.scenario_force_battlefield(0, "Mountain")
    obs = env.scenario_refresh()
    mine = obs.agent_permanents[0]
    # Controlled transport fixture: each used ability index is one fact row.
    mine.used_abilities = list(range(capacity))
    obs.agent_permanents = [mine]
    space = ObservationSpace(ObservationConfig(max_facts=capacity))
    native = observation_arrays(space, obs)
    facts = native["facts"]
    assert facts.shape == (capacity, 12)
    assert len(np.unique(facts, axis=0)) == capacity
    assert facts[:, FACT_VALID_INDEX].tolist() == [1] * capacity
    assert facts[:, FACT_KIND_INDEX].tolist() == [FACT_USED_ABILITY] * capacity

    mine.used_abilities = list(range(capacity + 1))
    obs.agent_permanents = [mine]
    message = f"observation capacity exceeded: facts {capacity + 1} > {capacity}"
    with pytest.raises(managym.ObservationCapacity, match=message):
        observation_arrays(space, obs)


@pytest.mark.parametrize("frequency", [1, 2])
def test_waiting_delayed_trigger_rows_name_frequency_and_lifetime(
    frequency: int,
) -> None:
    card = "Fatal Fissure" if frequency == 1 else "Ruinous Waterbending"
    env = priority_root((card,), ("Swamp",) * 7, opponent=("Gray Ogre",))
    obs = env.scenario_refresh()
    ogre = obs.opponent_permanents[0].id
    for _ in range(12):
        if any(frame.kind == 10 for frame in obs.continuation):
            break
        actions = obs.action_space.actions
        ranked = sorted(
            range(len(actions)),
            key=lambda index: (
                actions[index].action_type
                not in (
                    ActionEnum.PRIORITY_CAST_SPELL,
                    ActionEnum.CHOOSE_TARGET,
                    ActionEnum.PAY_COST,
                )
                or (
                    actions[index].action_type == ActionEnum.CHOOSE_TARGET
                    and ogre not in actions[index].focus
                ),
                actions[index].action_type != ActionEnum.PRIORITY_PASS_PRIORITY,
            ),
        )
        obs, *_ = env.step(ranked[0])
    else:
        raise AssertionError(f"{card} trigger is not waiting")
    relations = observation_arrays(ObservationSpace(), obs)["relations"]
    waiting = relations[:, 3] == 10
    header = relations[waiting & (relations[:, 0] == 1)][0]
    assert header[28] == (1 if frequency == 1 else 3)
    assert header[RELATION_THIS_TURN_INDEX] == 1
    lifetime = relations[waiting & (relations[:, 0] == 9)][0]
    assert lifetime[28] == frequency
    if frequency == 1:
        watched = relations[waiting & (relations[:, 0] == 3)][0]
        remembered = relations[waiting & (relations[:, 0] == 8)][0]
        assert remembered[24] == watched[24] > 0


def test_transform_fact_and_retained_schedule_relations() -> None:
    env = priority_root((), ("Mountain", "Gray Ogre"))
    obs = env.scenario_refresh()
    mine = obs.agent_permanents[-1]
    mine.transforms = 2
    obs.agent_permanents = [*obs.agent_permanents[:-1], mine]
    reference = {"object": mine.id, "definition": mine.registry_key, "player": None}
    empty = {"object": None, "definition": None, "player": None}
    frame: dict[str, object] = {
        "kind": 2,
        "decision": 0,
        "parent": None,
        "controller": obs.agent.player_index,
        "source": reference,
        "mode": None,
        "ability": 1,
        "kicked": False,
        "active_clause": None,
        "targets": [],
        "context": None,
        "sacrifice": None,
        "cost": None,
        "hybrid": [],
        "lesson": False,
        "waterbend": False,
        "can_decline": False,
        "mana_outputs": [],
        "resolutions_this_turn": 0,
    }
    frames = [
        frame | {"source_transforms": 1, "inherited_from": reference},
        frame | {"kind": 3, "x": 6},
        frame | {"kind": 6, "decision": 24, "can_decline": True},
        frame | {"kind": 6, "decision": 25, "mode": 3, "can_decline": True},
        frame | {"kind": 10, "decision": 7, "delayed_frequency": 2},
        frame | {"kind": 11, "source": empty, "mode": 0},
        frame | {"kind": 12, "source": empty, "mode": 1, "attackers": "land creatures"},
    ]
    space = ObservationSpace()
    actual = observation_arrays(space, obs)
    # Retained frames are read-only: the native environment remains the
    # execution authority, and an offline reader encodes the capture.
    retained = managym.Observation.from_json(
        obs.toJSON(), continuation_json=json.dumps(frames)
    )
    relations = observation_arrays(space, retained)["relations"]
    assert (relations[:, 0] == 7).any(), "inherited definition joins learning tokens"
    frequency = relations[relations[:, 0] == 9]
    assert len(frequency) == 1 and frequency[0, 28] == 2
    assert (relations[:, 3] == 12).any(), "pending combat is visible"
    assert (relations[:, RELATION_X_INDEX] == 6).any(), "paid X is retained"
    rows = actual["facts"][actual["facts"][:, FACT_KIND_INDEX] == FACT_TRANSFORMS]
    assert len(rows) and (rows[:, FACT_OBJECT_INDEX] > 0).all()


def test_saved_action_capacity_is_retained_only_for_the_current_input() -> None:
    saved = {"policy_input_version": 11, "max_actions": 128}
    assert ObservationConfig.from_checkpoint(saved).max_actions == 128
    # A configuration that names no input is the first one, never today's.
    with pytest.raises(UnsupportedPolicyInput):
        ObservationConfig.from_checkpoint({"max_actions": 128})


def test_bumi_scheduled_combat_restriction_is_a_fact_row() -> None:
    env = priority_root((), ("Bumi, Unleashed", "Swamp"), opponent=("Swamp",))
    obs = env.scenario_refresh()
    for _ in range(100):
        if any(frame.kind == 12 for frame in obs.continuation):
            break
        actions = obs.action_space.actions
        attack = next(
            (
                i
                for i, action in enumerate(actions)
                if action.action_type == ActionEnum.DECLARE_ATTACKER and action.declared
            ),
            None,
        )
        quiet = next(
            (
                i
                for i, action in enumerate(actions)
                if action.action_type == ActionEnum.PRIORITY_PASS_PRIORITY
            ),
            0,
        )
        obs, *_ = env.step(attack if attack is not None else quiet)
    else:
        raise AssertionError("Bumi's additional combat was not scheduled")
    facts = observation_arrays(ObservationSpace(), obs)["facts"]
    restriction = facts[facts[:, FACT_KIND_INDEX] == 10]
    assert len(restriction) == 1 and restriction[0, FACT_OBJECT_INDEX] > 0
    assert "land" in next(
        frame.attackers for frame in obs.continuation if frame.kind == 12
    )


@pytest.mark.parametrize("capacity", [128, 140, 256, 512])
def test_vector_rows_derive_fact_capacity(capacity: int) -> None:
    config = ObservationConfig(max_facts=capacity)
    space = ObservationSpace(config)
    env = managym.VectorEnv(num_envs=1, seed=213, skip_trivial=True)
    env.set_observation_layout(step_buffers(1), layout=space.layout)
    obs, _ = env.reset_all(make_player_configs())[0]
    env.reset_seeds_into_buffers([213])
    rows = vector_arrays(space, env.observation_rows())
    assert rows["facts"].shape == (1, capacity, 12)
    assert_row_observation_equal(rows, 0, observation_arrays(space, obs))
    scalar = managym.Env(seed=213, skip_trivial=True)
    scalar.reset(make_player_configs())
    obs, *_ = scalar.step(0)
    env.step_into_buffers([0])
    rows = vector_arrays(space, env.observation_rows())
    assert_row_observation_equal(rows, 0, observation_arrays(space, obs))


def test_merchant_food_tap_count_is_the_payment_frame_mode() -> None:
    """Paying with "Tap two untapped Foods" leaves a payment frame whose mode
    column (a one-based selector) counts the taps still required."""
    env = priority_root(
        ("Unlucky Cabbage Merchant", "Unlucky Cabbage Merchant", "Llanowar Elves"),
        ("The Cabbage Merchant", "Forest", "Forest", "Forest", "Forest"),
    )
    owner = env.current_agent_index()
    assert owner is not None
    for food_count in (1, 2):
        obs = env.observation_for_player(owner)
        cast_index = next(
            index
            for index, action in enumerate(obs.action_space.actions)
            if int(action.action_type) == ActionEnum.PRIORITY_CAST_SPELL
        )
        env.step(cast_index)
        for _ in range(16):
            obs = env.observation_for_player(owner)
            if (
                sum(
                    perm.characteristics.name == "Food" for perm in obs.agent_permanents
                )
                == food_count
                and not obs.stack_objects
            ):
                break
            actor = env.current_agent_index()
            assert actor is not None
            pending = env.observation_for_player(actor)
            index = next(
                index
                for index, action in enumerate(pending.action_space.actions)
                if int(action.action_type)
                in (ActionEnum.PAY_COST, ActionEnum.PRIORITY_PASS_PRIORITY)
            )
            env.step(index)
        else:
            raise AssertionError("Food creation did not resolve")
    # Every Forest is tapped; the Foods pay for the Elves.
    decision = capture_text_decision(env)
    cast = next(a for a in decision.actions if "Llanowar Elves" in a.description)
    submit_response(env, decision, cast.reference)
    decision = capture_text_decision(env)
    activation = next(
        action
        for action in decision.actions
        if "Activate" in action.description
        and "The Cabbage Merchant" in action.description
    )
    submit_response(env, decision, activation.reference)
    space = ObservationSpace()
    for remaining in (2, 1):
        actor = env.current_agent_index()
        assert actor is not None
        obs = env.observation_for_player(actor)
        assert obs.continuation[-1].mode == remaining
        relations = observation_arrays(space, obs)["relations"]
        valid = relations[relations[:, RELATION_VALID_INDEX] == 1]
        frame = valid[valid[:, 3] == obs.continuation[-1].kind]
        assert frame[:, 7].tolist() == [remaining + 1]
        decision = capture_text_decision(env)
        assert f"{remaining} tap selections still required" in decision.text
        assert all(
            "Tap Food to pay" in action.description for action in decision.actions
        )
        submit_response(env, decision, decision.actions[0].reference)


def test_one_creature_tapped_for_mana_or_to_convoke_renders_both_taps() -> None:
    """Both taps share TAP_FOR_COST; the native flag names which pays directly."""
    env = priority_root(
        ("Appeal to Eirdu",),
        ("Savannah Lions", "Gray Ogre", "Hill Giant", "Llanowar Elves"),
    )
    for chosen in ("Cast C1 (Appeal to Eirdu)", "Target P2 (Gray Ogre)", "Decline"):
        decision = capture_text_decision(env)
        action = next(a for a in decision.actions if a.description.startswith(chosen))
        submit_response(env, decision, action.reference)
    decision = capture_text_decision(env)
    assert [action.description for action in decision.actions] == [
        "Tap for mana P4 (Llanowar Elves). Native description: "
        "Tap Llanowar Elves for {G}",
        "Tap for cost P1 (Savannah Lions). Native description: "
        "Tap Savannah Lions to pay {W} (convoke)",
        "Tap for cost P2 (Gray Ogre). Native description: "
        "Tap Gray Ogre to pay {1} (convoke)",
        "Tap for cost P3 (Hill Giant). Native description: "
        "Tap Hill Giant to pay {1} (convoke)",
        "Tap for cost P4 (Llanowar Elves). Native description: "
        "Tap Llanowar Elves to pay {1} (convoke)",
    ]

    actor = env.current_agent_index()
    assert actor is not None
    obs = env.observation_for_player(actor)
    space = ObservationSpace()
    reference = observation_arrays(space, obs)
    batch = managym.encode_observation_batch([env], space.layout)
    assert list(batch.action_counts) == [5]
    rows = vector_arrays(space, [(batch, 0)])
    for key, value in reference.items():
        np.testing.assert_array_equal(rows[key][0], value, err_msg=key)
    actions = reference["actions"]
    assert actions.shape[1] == 30
    np.testing.assert_array_equal(
        reference["action_focus"][0], reference["action_focus"][4]
    )
    np.testing.assert_array_equal(
        actions[0, :ACTION_PAYS_DIRECTLY_INDEX], actions[4, :ACTION_PAYS_DIRECTLY_INDEX]
    )
    np.testing.assert_array_equal(
        actions[:5, ACTION_PAYS_DIRECTLY_INDEX], [0, 1, 1, 1, 1]
    )

    root = env.state_digest()
    for index, committed in (
        (0, "Committed to this cost: {G} from Llanowar Elves."),
        (4, "Committed to this cost: tapped Llanowar Elves to convoke."),
    ):
        fork = env.clone_env()
        forked = capture_text_decision(fork)
        submit_response(fork, forked, forked.actions[index].reference)
        after = capture_text_decision(fork)
        assert committed in after.text.splitlines()
        assert not any("Llanowar Elves" in a.description for a in after.actions)
    assert env.state_digest() == root


def test_tap_pays_choice_names_its_payment_and_focuses_the_tapped_creature() -> None:
    """A tap-pays mode names the payment; its offers focus the creature tapped."""
    env = priority_root(
        ("Appeal to Eirdu",),
        ("Savannah Lions", "Savannah Lions", "Gray Ogre", "Hill Giant"),
    )
    for chosen in (
        "Cast C1 (Appeal to Eirdu)",
        "Target P3 (Gray Ogre)",
        "Decline",
        "Tap for cost P1 (Savannah Lions)",
    ):
        decision = capture_text_decision(env)
        action = next(a for a in decision.actions if a.description.startswith(chosen))
        submit_response(env, decision, action.reference)
    decision = capture_text_decision(env)
    # Mode 0 is {W} by convoke and mode 5 {1} by convoke; text is one-based.
    assert [action.description for action in decision.actions] == [
        "Choose mode P1 (Savannah Lions); mode 1. Native description: "
        "Savannah Lions pays {W} (convoke)",
        "Choose mode P1 (Savannah Lions); mode 6. Native description: "
        "Savannah Lions pays {1} (convoke)",
    ]

    actor = env.current_agent_index()
    assert actor is not None
    obs = env.observation_for_player(actor)
    assert [action.mode_index for action in obs.action_space.actions] == [0, 5]
    space = ObservationSpace()
    reference = observation_arrays(space, obs)
    batch = managym.encode_observation_batch([env], space.layout)
    assert list(batch.action_counts) == [2]
    rows = vector_arrays(space, [(batch, 0)])
    for key, value in reference.items():
        np.testing.assert_array_equal(rows[key][0], value, err_msg=key)
    np.testing.assert_array_equal(reference["actions"][:2, ACTION_MODE_INDEX], [1, 6])
    lions = next(
        index
        for index, permanent in enumerate(obs.agent_permanents)
        if permanent.tapped
    )
    lions_row = 2 + 2 * space.config.max_cards_per_player + lions
    np.testing.assert_array_equal(
        reference["action_focus"][:2], [[lions_row, -1], [lions_row, -1]]
    )


def test_scout_s_warning_record_and_minus_counters_reach_text_and_facts() -> None:
    env = priority_root(("Scout's Warning", "Gray Ogre"), ("Plains", "Gray Ogre"))
    owner = env.current_agent_index()
    assert owner is not None
    decision = capture_text_decision(env)
    cast = next(
        action
        for action in decision.actions
        if "Cast" in action.description and "Scout's Warning" in action.description
    )
    submit_response(env, decision, cast.reference)
    for _ in range(16):
        obs = env.observation_for_player(owner)
        if not obs.stack_objects and env.current_agent_index() == owner:
            break
        actor = env.current_agent_index()
        assert actor is not None
        pending = env.observation_for_player(actor)
        index = next(
            index
            for index, action in enumerate(pending.action_space.actions)
            if int(action.action_type)
            in (ActionEnum.PAY_COST, ActionEnum.PRIORITY_PASS_PRIORITY)
        )
        env.step(index)
    else:
        raise AssertionError("Scout's Warning did not resolve")
    obs = env.observation_for_player(owner)
    assert [effect.modification for effect in obs.agent.effects] == [
        "NextCreatureCastFlash"
    ]
    decision = capture_text_decision(env)
    assert (
        "the next creature card cast can be cast as though it had flash"
        in decision.text
    )
    space = ObservationSpace()
    permanents = obs.agent_permanents
    index, ogre = next(
        (i, p) for i, p in enumerate(permanents) if p.characteristics.power is not None
    )
    ogre.counters = {"minus_one_minus_one": 1}
    obs.agent_permanents = [ogre if p.id == ogre.id else p for p in permanents]
    facts = observation_arrays(space, obs)["facts"]
    counters = facts[
        (facts[:, FACT_VALID_INDEX] == 1) & (facts[:, FACT_KIND_INDEX] == FACT_COUNTER)
    ]
    token = managym.catalog_token_value("CounterKind::minus_one_minus_one")
    own = 2 + 2 * space.config.max_cards_per_player
    assert counters[
        :, [FACT_OBJECT_INDEX, FACT_ID_INDEX + 1, FACT_VALUE_INDEX]
    ].tolist() == [[own + index + 1, token & 0xFFFF, pytest.approx(0.1)]]


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_stack_spells_are_separate_from_card_rows() -> None:
    env = priority_root(("Lightning Bolt",), ("Mountain", "Gray Ogre"))
    obs = env.scenario_refresh()
    bolt = next(
        index
        for index, action in enumerate(obs.action_space.actions)
        if int(action.action_type) == ActionEnum.PRIORITY_CAST_SPELL
    )
    obs, *_ = env.step(bolt)
    for _ in range(8):
        if obs.stack_objects:
            break
        actions = obs.action_space.actions
        obs, *_ = env.step(
            next(
                (
                    i
                    for i, action in enumerate(actions)
                    if int(action.action_type)
                    in (ActionEnum.CHOOSE_TARGET, ActionEnum.PAY_COST)
                ),
                0,
            )
        )
    assert len(obs.stack_objects) == 1
    v11 = observation_arrays(ObservationSpace(), obs)
    cards = len(obs.agent_cards)
    assert int(v11["agent_cards_valid"].sum()) == cards
    assert int(v11["stack_valid"].sum()) == 1
