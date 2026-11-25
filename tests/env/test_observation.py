"""Observation space layout and native encoding with minimal capacities.

Covers the Python enum mirrors, the space's keys and shapes, validity masks,
feature normalization, focus indices, events and action-capacity refusal.
Capacity refusal for cards, permanents and focus is in
`tests/verify/test_native_buffer_capacities.py`.
"""

import json
from typing import Tuple

import numpy as np
import pytest

# Local imports
from manabot.env.observation import (
    EventEntityKindEnum,
    EventTypeEnum,
    ObservationConfig,
    ObservationSpace,
    PhaseEnum,
    StepEnum,
    ZoneEnum,
    encode_observation,
)
import managym

# ─────────────────────────── Fixtures ───────────────────────────


@pytest.fixture(scope="session")
def config():
    """A configuration with minimal capacities for testing."""
    return ObservationConfig(
        max_cards_per_player=8,
        max_permanents_per_player=2,
        max_actions=8,
        max_focus_objects=2,
    )


@pytest.fixture(scope="session")
def observation_space(config):
    """The observation space of the minimal configuration."""
    return ObservationSpace(config)


@pytest.fixture(scope="session")
def player_configs():
    """Create consistent player configurations with minimal decks."""
    player_a = managym.PlayerConfig("Alice", {"Mountain": 10, "Gray Ogre": 20})
    player_b = managym.PlayerConfig("Bob", {"Forest": 10, "Llanowar Elves": 20})
    return [player_a, player_b]


@pytest.fixture
def env():
    """Create a fresh environment for each test."""
    return managym.Env()


@pytest.fixture
def observation(env, player_configs) -> managym.Observation:
    """Get a fresh observation from a newly reset environment."""
    obs, _ = env.reset(player_configs)
    return obs


@pytest.fixture
def two_observations(
    env, player_configs
) -> Tuple[managym.Observation, managym.Observation]:
    """Get two consecutive observations for testing observation independence."""
    obs1, _ = env.reset(player_configs)
    # Take an action to get a different game state
    obs2, _, _, _, _ = env.step(0)
    return obs1, obs2


# ─────────────────────────── Tests ───────────────────────────


class TestEnumParity:
    """Verify that our Python enums match the managym ones exactly."""

    @pytest.mark.parametrize(
        "py_enum, cpp_enum",
        [
            (ZoneEnum.LIBRARY, managym.ZoneEnum.LIBRARY),
            (ZoneEnum.HAND, managym.ZoneEnum.HAND),
            (ZoneEnum.BATTLEFIELD, managym.ZoneEnum.BATTLEFIELD),
            (ZoneEnum.GRAVEYARD, managym.ZoneEnum.GRAVEYARD),
            (ZoneEnum.STACK, managym.ZoneEnum.STACK),
            (ZoneEnum.EXILE, managym.ZoneEnum.EXILE),
            (ZoneEnum.COMMAND, managym.ZoneEnum.COMMAND),
        ],
    )
    def test_zone_enum(self, py_enum, cpp_enum):
        assert int(cpp_enum) == py_enum

    @pytest.mark.parametrize(
        "py_enum, cpp_enum",
        [
            (PhaseEnum.BEGINNING, managym.PhaseEnum.BEGINNING),
            (PhaseEnum.PRECOMBAT_MAIN, managym.PhaseEnum.PRECOMBAT_MAIN),
            (PhaseEnum.COMBAT, managym.PhaseEnum.COMBAT),
            (PhaseEnum.POSTCOMBAT_MAIN, managym.PhaseEnum.POSTCOMBAT_MAIN),
            (PhaseEnum.ENDING, managym.PhaseEnum.ENDING),
        ],
    )
    def test_phase_enum(self, py_enum, cpp_enum):
        assert int(cpp_enum) == py_enum

    @pytest.mark.parametrize(
        "py_enum, cpp_enum",
        [
            (StepEnum.BEGINNING_UNTAP, managym.StepEnum.BEGINNING_UNTAP),
            (StepEnum.BEGINNING_UPKEEP, managym.StepEnum.BEGINNING_UPKEEP),
            (StepEnum.BEGINNING_DRAW, managym.StepEnum.BEGINNING_DRAW),
            (StepEnum.PRECOMBAT_MAIN_STEP, managym.StepEnum.PRECOMBAT_MAIN_STEP),
            (StepEnum.COMBAT_BEGIN, managym.StepEnum.COMBAT_BEGIN),
            (
                StepEnum.COMBAT_DECLARE_ATTACKERS,
                managym.StepEnum.COMBAT_DECLARE_ATTACKERS,
            ),
            (
                StepEnum.COMBAT_DECLARE_BLOCKERS,
                managym.StepEnum.COMBAT_DECLARE_BLOCKERS,
            ),
            (StepEnum.COMBAT_DAMAGE, managym.StepEnum.COMBAT_DAMAGE),
            (StepEnum.COMBAT_END, managym.StepEnum.COMBAT_END),
            (StepEnum.POSTCOMBAT_MAIN_STEP, managym.StepEnum.POSTCOMBAT_MAIN_STEP),
            (StepEnum.ENDING_END, managym.StepEnum.ENDING_END),
            (StepEnum.ENDING_CLEANUP, managym.StepEnum.ENDING_CLEANUP),
            (
                StepEnum.COMBAT_FIRST_STRIKE_DAMAGE,
                managym.StepEnum.COMBAT_FIRST_STRIKE_DAMAGE,
            ),
        ],
    )
    def test_step_enum(self, py_enum, cpp_enum):
        assert int(cpp_enum) == py_enum

    @pytest.mark.parametrize(
        "py_enum, cpp_enum",
        [
            (EventTypeEnum.CARD_MOVED, managym.EventTypeEnum.CARD_MOVED),
            (EventTypeEnum.DAMAGE_DEALT, managym.EventTypeEnum.DAMAGE_DEALT),
            (EventTypeEnum.LIFE_CHANGED, managym.EventTypeEnum.LIFE_CHANGED),
            (EventTypeEnum.SPELL_CAST, managym.EventTypeEnum.SPELL_CAST),
            (EventTypeEnum.SPELL_RESOLVED, managym.EventTypeEnum.SPELL_RESOLVED),
            (EventTypeEnum.SPELL_COUNTERED, managym.EventTypeEnum.SPELL_COUNTERED),
            (
                EventTypeEnum.ABILITY_TRIGGERED,
                managym.EventTypeEnum.ABILITY_TRIGGERED,
            ),
            (
                EventTypeEnum.COMBAT_ATTACKERS_DECLARED,
                managym.EventTypeEnum.COMBAT_ATTACKERS_DECLARED,
            ),
            (
                EventTypeEnum.BLOCKERS_DECLARED,
                managym.EventTypeEnum.BLOCKERS_DECLARED,
            ),
            (
                EventTypeEnum.COMBAT_DAMAGE_DEALT,
                managym.EventTypeEnum.COMBAT_DAMAGE_DEALT,
            ),
            (
                EventTypeEnum.PERMANENTS_DIED,
                managym.EventTypeEnum.PERMANENTS_DIED,
            ),
            (EventTypeEnum.TURN_STARTED, managym.EventTypeEnum.TURN_STARTED),
        ],
    )
    def test_event_type_enum(self, py_enum, cpp_enum):
        assert int(cpp_enum) == py_enum

    @pytest.mark.parametrize(
        "py_enum, cpp_enum",
        [
            (EventEntityKindEnum.NONE, managym.EventEntityKindEnum.NONE),
            (EventEntityKindEnum.CARD, managym.EventEntityKindEnum.CARD),
            (EventEntityKindEnum.PERMANENT, managym.EventEntityKindEnum.PERMANENT),
            (EventEntityKindEnum.PLAYER, managym.EventEntityKindEnum.PLAYER),
            (EventEntityKindEnum.OBJECT, managym.EventEntityKindEnum.OBJECT),
        ],
    )
    def test_event_entity_kind_enum(self, py_enum, cpp_enum):
        assert int(cpp_enum) == py_enum


def test_retained_turn_codes_decode_to_rules_order(
    env: managym.Env,
    observation: managym.Observation,
) -> None:
    wire = json.loads(observation.toJSON())
    semantic = json.loads(env.semantic_observation_json(env.current_agent_index()))
    # Each tuple is in rules order. Its entries are the old wire codes.
    for field, enum, codes in (
        ("phase", PhaseEnum, (0, 1, 2, 3, 4)),
        ("step", StepEnum, (0, 1, 2, 3, 4, 5, 6, 12, 7, 8, 9, 10, 11)),
    ):
        for current, legacy in enumerate(codes):
            wire["turn"][field] = legacy
            semantic["viewer_state"]["turn"][field] = legacy
            managym.validate_observation_json(json.dumps(semantic))
            restored = managym.Observation.from_json(json.dumps(wire))
            assert int(getattr(restored.turn, field)) == current
            named = restored.toJSON()
            assert json.loads(named)["turn"][field] == enum(current).name
            assert managym.Observation.from_json(named).toJSON() == named


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("phase", 5),
        ("phase", False),
        ("step", 13),
        ("step", 1.0),
        ("step", None),
        ("step", "FirstStrikeCombatDamage"),
    ],
)
def test_invalid_turn_fields_are_rejected(
    observation: managym.Observation, field: str, value: object
) -> None:
    wire = json.loads(observation.toJSON())
    wire["turn"][field] = value
    with pytest.raises(ValueError):
        managym.Observation.from_json(json.dumps(wire))


class TestObservationSpace:
    """The space's capacities, keys and shapes."""

    def test_default_padding_targets_202_objects(self):
        # 2 players + 60 cards x 2 + 40 permanents x 2 (permanent capacity
        # raised 30 -> 40: token-heavy GW games exceeded 30 battlefield
        # entries and truncated).
        default_space = ObservationSpace(ObservationConfig())
        total_objects = (
            2
            + default_space.config.max_cards_per_player * 2
            + default_space.config.max_permanents_per_player * 2
        )
        assert total_objects == 202

    def get_expected_keys(self) -> set[str]:
        """Get the complete set of expected keys in an encoded observation."""
        return {
            # Players
            "agent_player",
            "agent_player_valid",
            "opponent_player",
            "opponent_player_valid",
            # Cards
            "agent_cards",
            "agent_cards_valid",
            "opponent_cards",
            "opponent_cards_valid",
            # Permanents
            "agent_permanents",
            "agent_permanents_valid",
            "opponent_permanents",
            "opponent_permanents_valid",
            "stack",
            "stack_valid",
            "player_turn_state",
            "card_turn_state",
            # Actions
            "actions",
            "actions_valid",
            "action_focus",
            # Events
            "events",
            "events_valid",
            "semantic_cards",
            "known_hand",
            "known_library",
            "decklists",
            "relations",
            "facts",
            "effects",
            "effect_details",
            "effect_members",
            "name_candidates",
        }

    def test_encode_observation(
        self, observation_space: ObservationSpace, observation: managym.Observation
    ) -> None:
        """Test that encoding produces arrays with correct shapes and no invalid values."""
        encoded = encode_observation(observation_space, observation)

        # Verify all expected keys are present
        expected_keys = self.get_expected_keys()
        assert set(encoded.keys()) == expected_keys, (
            f"Keys mismatch. Missing: {expected_keys - set(encoded.keys())}"
        )

        # Check shapes match specification
        for key in encoded:
            expected_shape = observation_space.shapes[key]
            assert encoded[key].shape == expected_shape, f"Shape mismatch for {key}"
            assert not np.isnan(encoded[key]).any(), f"NaN found in {key}"
            assert not np.isinf(encoded[key]).any(), f"Inf found in {key}"

    def test_validity_masks(self, observation_space, observation, config):
        """Test that validity masks correctly reflect the game state."""
        encoded = encode_observation(observation_space, observation)

        # Check cards validity
        agent_cards_valid = encoded["agent_cards_valid"]
        actual_cards = len(observation.agent_cards)
        expected_valid = actual_cards
        valid_sum = int(agent_cards_valid.sum())  # Convert to Python int for comparison
        assert valid_sum == expected_valid, (
            f"Expected {expected_valid} valid cards, got {valid_sum}"
        )

        # Verify invalid slots are zeroed
        for key in [
            "agent_cards",
            "opponent_cards",
            "agent_permanents",
            "opponent_permanents",
            "events",
        ]:
            valid_key = f"{key}_valid"
            invalid = ~encoded[valid_key].astype(bool)
            if np.any(invalid):
                array_slice = encoded[key][invalid]
                assert (array_slice == 0).all(), (
                    f"Non-zero values found in invalid {key} slots: {array_slice[array_slice != 0]}"
                )

    def test_player_features_include_turn_and_are_normalized(
        self, observation_space, observation
    ):
        encoded = encode_observation(observation_space, observation)

        for key, player in (
            ("agent_player", observation.agent),
            ("opponent_player", observation.opponent),
        ):
            vec = encoded[key][0]

            assert vec[0] == pytest.approx(player.life / 20.0)
            assert vec[1] == pytest.approx(float(player.is_active))

            zone_slice = vec[2:9]
            expected_zones = np.array(player.zone_counts, dtype=np.float32) / 60.0
            np.testing.assert_allclose(zone_slice, expected_zones, atol=1e-6)

            phase_slice = vec[9:14]
            step_slice = vec[14:26]
            assert phase_slice.sum() == pytest.approx(1.0)
            assert step_slice.sum() == pytest.approx(1.0)
            assert phase_slice[int(observation.turn.phase)] == pytest.approx(1.0)
            assert step_slice[int(observation.turn.step)] == pytest.approx(1.0)

    def test_card_and_permanent_features_are_partitioned_and_normalized(
        self, observation_space, observation, config
    ):
        encoded = encode_observation(observation_space, observation)

        for i, card in enumerate(observation.agent_cards):
            vec = encoded["agent_cards"][i]
            assert vec[7] == pytest.approx(1.0)
            assert vec[8] == pytest.approx(card.power / 10.0)
            assert vec[9] == pytest.approx(card.toughness / 10.0)
            assert vec[10] == pytest.approx(card.mana_cost.mana_value / 10.0)

        for i, card in enumerate(observation.opponent_cards):
            vec = encoded["opponent_cards"][i]
            assert vec[7] == pytest.approx(0.0)
            assert vec[8] == pytest.approx(card.power / 10.0)
            assert vec[9] == pytest.approx(card.toughness / 10.0)
            assert vec[10] == pytest.approx(card.mana_cost.mana_value / 10.0)

        for i, perm in enumerate(observation.agent_permanents):
            vec = encoded["agent_permanents"][i]
            assert vec[0] == pytest.approx(1.0)
            assert vec[2] == pytest.approx(perm.damage / 10.0)

        for i, perm in enumerate(observation.opponent_permanents):
            vec = encoded["opponent_permanents"][i]
            assert vec[0] == pytest.approx(0.0)
            assert vec[2] == pytest.approx(perm.damage / 10.0)

    def test_focus_indices(self, observation_space, observation, config):
        """Test that action focus indices are valid."""
        encoded = encode_observation(observation_space, observation)
        focus = encoded["action_focus"]

        # Calculate valid range for focus indices
        total_objects = (
            2 + config.max_cards_per_player * 2 + config.max_permanents_per_player * 2
        )

        # All indices should be either -1 (no focus) or within valid range
        valid = (focus == -1) | ((focus >= 0) & (focus < total_objects))
        assert valid.all(), f"Invalid focus indices found: {focus[~valid]}"

    def test_event_features_round_trip(self, observation_space, observation, config):
        encoded = encode_observation(observation_space, observation)
        learning_events = [
            event
            for event in observation.recent_events
            if int(event.event_type) <= int(EventTypeEnum.ABILITY_TRIGGERED)
        ]
        events = learning_events[-config.max_events :]
        valid_count = min(len(learning_events), config.max_events)

        assert int(encoded["events_valid"].sum()) == valid_count
        for i, event in enumerate(events):
            np.testing.assert_allclose(
                encoded["events"][i],
                np.array(
                    [
                        float(int(event.event_type)),
                        float(int(event.source_kind)),
                        float(event.source_id),
                        float(int(event.target_kind)),
                        float(event.target_id),
                        float(event.amount),
                        float(event.controller_id),
                    ],
                    dtype=np.float32,
                ),
                atol=1e-6,
            )

    def test_action_space_overflow_rejects_incomplete_policy_input(
        self, observation: managym.Observation
    ) -> None:
        count = len(observation.action_space.actions)
        assert count > 1
        space = ObservationSpace(ObservationConfig(max_actions=count - 1))
        with pytest.raises(
            managym.ObservationCapacity,
            match=rf"capacity exceeded: actions {count} > {count - 1}",
        ):
            encode_observation(space, observation)
        assert len(observation.action_space.actions) == count


if __name__ == "__main__":
    pytest.main([__file__])


def test_native_batch_freezes_live_rows_and_retains_first_legal(
    player_configs: list[managym.PlayerConfig],
) -> None:
    env = managym.Env(seed=17)
    env.reset(player_configs)
    space = ObservationSpace(ObservationConfig(max_actions=1))
    with pytest.raises(managym.ObservationCapacity):
        managym.encode_observation_batch([env], space.layout)
    batch = managym.encode_observation_batch([env, env], space.layout, "first-legal")
    assert batch.action_counts == [1, 1]
    tables = batch.tables()
    seat_axis, offsets, data = tables["actions"]
    assert not seat_axis
    np.testing.assert_array_equal(np.frombuffer(offsets, np.int32), [0, 1, 2])
    fields = batch.layout.csr_fields["actions"]
    dtype = np.dtype([(name, dtype) for name, dtype, _ in fields])
    values = np.frombuffer(data, dtype)
    assert values.shape == (2,)
    assert dtype["valid"] == np.uint8
    assert dtype["mode"] == np.int32
    assert dtype["x"] == np.float32
    assert values["valid"].tolist() == [1, 1]
    for source in (values, np.frombuffer(offsets, np.int32)):
        with pytest.raises(ValueError):
            source.setflags(write=True)
    tables.clear()
    env.step(0)
    assert batch.tables()["actions"] == (False, offsets, data)
    with pytest.raises(TypeError):
        managym.ObservationBatch()


def test_vector_step_batch_keeps_outcomes_and_emitted_events(
    player_configs: list[managym.PlayerConfig],
) -> None:
    space = ObservationSpace()
    dense, packed = (managym.VectorEnv(2, seed=19) for _ in range(2))
    for env in (dense, packed):
        env.reset_all(player_configs)
    for actions in ([0, 1], [0, 0]):
        expected = dense.step(actions)
        batch, rewards, terminated, truncated = packed.step_observation_batch(
            actions, space.layout
        )
        for i, (obs, reward, terminal, limit, _) in enumerate(expected):
            assert (rewards[i], terminated[i], truncated[i]) == (
                reward,
                terminal,
                limit,
            )
            encoded = encode_observation(space, obs)
            table = batch.tables()["events"]
            offsets = np.frombuffer(table[1], np.int32)
            values = np.frombuffer(table[2], np.int32).reshape(
                -1, space.shapes["events"][-1]
            )
            np.testing.assert_array_equal(
                values[offsets[i] : offsets[i + 1]],
                encoded["events"][encoded["events_valid"] == 1],
            )
            assert batch.action_counts[i] == int(encoded["actions_valid"].sum())


def test_csr_fields_cover_every_logical_column_and_optional_table() -> None:
    for history in (False, True):
        layout = managym.ObservationLayout(
            policy_history=history,
            max_decision_records=8,
            max_decision_references=8,
            max_decision_reference_details=16,
            max_decision_members=16,
        )
        arrays = layout.arrays
        assert set(layout.csr_fields) == {
            key for key in arrays if not key.endswith("_valid")
        }
        for key, fields in layout.csr_fields.items():
            columns = arrays[key]["columns"]
            if not columns:
                assert fields == [
                    ("value", "uint8" if key == "card_turn_state" else "int32", 0)
                ]
                continue
            names = list(columns)
            logical = []
            for name, dtype, index in fields:
                assert index == len(logical)
                if names[index] == name + "_high":
                    assert dtype == "int32"
                    logical.extend((name + "_high", name + "_low"))
                else:
                    logical.append(name)
                    kind = columns[name]
                    if kind["kind"] == "flag" or (
                        key.startswith("decision_")
                        and (
                            name.endswith("_present")
                            or name
                            in (
                                "valid",
                                "declared",
                                "tapped",
                                "ready",
                                "attacking",
                                "blocking",
                            )
                        )
                    ):
                        assert dtype == "uint8"
                    elif kind.get("scale", 1) > 1:
                        assert dtype == "float32"
                    else:
                        assert dtype == "int32"
            assert logical == names
