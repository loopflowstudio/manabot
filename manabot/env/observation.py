"""The policy input's layout: named arrays, their shapes and the columns the model reads.

The native encoder (`managym/src/agent/observation_encoder.rs`) is the only
writer of these arrays; `managym.observation_layout()` publishes its layout.
This module mirrors the part manabot reads:

- the column constants below (`PERMANENT_DEFINITION_INDEX`, `FACT_VALUE_INDEX`,
  ...), held to the native layout by `tests/env/test_named_observation_layout.py`;
- `ObservationEncoder`, the layout of one `ObservationConfig`: shapes, dtypes
  and `allocate`. Despite its name it encodes nothing;
- `ObservationSpace`, the Gymnasium space, whose `encode` returns fresh buffers
  written by `managym.encode_observation_into`.

Native reads the configuration back from the buffers' shapes, so every buffer
is derived from the space the caller supplies, never a native default.
"""

from collections.abc import Mapping
from typing import ItemsView, KeysView, ValuesView

import gymnasium as gym
import numpy as np

from manabot.env.decision_memory import MEMORY_WIDTHS
from manabot.env.legacy_input import LEGACY_PERMANENT_DIM
from manabot.env.shared_effects import SHARED_EFFECT_SHAPES
from manabot.infra.config import ObservationConfig
import managym

# Many modules import the enum mirrors from here; managym.enums owns them.
from managym.enums import (  # noqa: F401
    ActionEnum,
    ActionSpaceEnum,
    EventEntityKindEnum,
    EventTypeEnum,
    PhaseEnum,
    StepEnum,
    ZoneEnum,
)

# Host observations at policy, simulation and rollout boundaries.
type ObservationBuffers = Mapping[str, np.ndarray]

# Column constants manabot reads. Native `observation_layout.rs` owns the
# layout; `tests/env/test_named_observation_layout.py` holds each constant to
# the column of that name, so a moved column fails there, not silently here.
# Add a constant only for a column the model, preparation or a test reads.

# Action rows: one-hot family, declaration, mode/ability selectors,
# continuation join, validity.
ACTION_TYPE_DIM = len(ActionEnum)
ACTION_DECLARED_INDEX = ACTION_TYPE_DIM
ACTION_MODE_INDEX = ACTION_DECLARED_INDEX + 1
ACTION_ABILITY_INDEX = ACTION_MODE_INDEX + 1
ACTION_CONTINUATION_INDEX = ACTION_ABILITY_INDEX + 1
MAX_ACTION_SELECTOR = (1 << 24) - 1
# Name candidate columns: name row + 1, definition + 1, then the first and
# last name rows (+ 1) of a naming group choice.
NAME_DIM = 4

# Mana restricted to spending on Lesson, Shrine and Ally spells: three blocks
# of W/U/B/R/G/C.
PLAYER_RESTRICTED_INDEX = 40
PLAYER_NO_MAXIMUM_HAND_SIZE_INDEX = 59
PLAYER_DIM = 74
# Cast-permission terms follow the printed characteristics; validity is last.
CARD_PERMISSION_INDEX = 40
CARD_DIM = 56

PERMANENT_DEFINITION_INDEX = 24
PERMANENT_ONCE_EACH_TURN_INDEX = 29
# Object index + 1 of the attachment's host and of the attacker this creature
# blocks: joins, never numeric features.
PERMANENT_ATTACHED_INDEX = 31
PERMANENT_BLOCKING_INDEX = 33
# Rank of its timestamp among the observation's timestamps.
PERMANENT_TIMESTAMP_INDEX = 35
PERMANENT_MANA_VALUE_INDEX = 37
PERMANENT_TYPES_INDEX = 38  # land..battle
PERMANENT_DIM = 51
# Permanent columns that hold object links (joins), in model parameter order.
PERMANENT_LINK_INDEXES = (PERMANENT_ATTACHED_INDEX, PERMANENT_BLOCKING_INDEX)

# Input v11 stack rows, top first: one per spell or ability.
STACK_IS_MINE_INDEX = 0
STACK_DEFINITION_INDEX = 4
STACK_SOURCE_INDEX = 5  # object index + 1 of an ability's source: a join
STACK_DIM = 37

RELATION_ROWS = 64
RELATION_DIM = 49
RELATION_VALID_INDEX = 31
RELATION_ELIGIBILITY_INDEX = 32
RELATION_X_INDEX = 33
RELATION_THIS_TURN_INDEX = 48  # a waiting delayed trigger ends this turn

# Fact rows: one open-vocabulary fact about a player or permanent, joined to
# it like relation references. Ids are learning-token values split into 16-bit
# halves (exact in float32).
# Model numeric scale for timestamp ranks, independent of row allocation.
FACT_TIMESTAMP_SCALE = 128
FACT_OBJECT_INDEX = 0
FACT_KIND_INDEX = 1
FACT_ID_INDEX = 2  # high, low
FACT_DETAIL_INDEX = 4  # high, low: a keyword, subtype or color an effect names
FACT_VALUE_INDEX = 6  # amounts / 10: count, power or N; then toughness
FACT_DURATION_INDEX = 8  # 1 until end of turn, 2 indefinite, 3 while controlling source
FACT_ORDER_INDEX = 9  # timestamp rank (1-based)
FACT_SOURCE_INDEX = 10  # source definition + 1
FACT_VALID_INDEX = 11
FACT_DIM = 12
# FactKind values (native `observation_encoder::FactKind`).
FACT_COUNTER = 1
FACT_SET_NAME = 3
FACT_USED_ABILITY = 5
# Reserved: continuous-effect records are shared-effect rows, never fact rows.
FACT_EFFECT = 6
FACT_COPY = 7
FACT_COPY_EXCEPTION = 8
FACT_TRANSFORMS = 9
FACT_COMBAT_RESTRICTION = 10
FACT_EFFECT_PROGRAM = 11
FACT_CONTINUATION_PROGRAM = 14
FACT_KINDS = 14
# The id of an effect program that states a player's pooled mana with a
# spend effect or combat expiry (native `observation_encoder::POOLED_MANA`).
POOLED_MANA = "PooledMana"

# Decklist rows: definition id + 1, then main, unseen, sideboard and
# remaining-sideboard counts. A closed opponent's rows are all zero.
DECKLIST_DIM = 5
# Input v10/v11 turn state. `player_turn_state` has one row per seat, agent
# first: life lost this turn / 20, spells cast this turn / 10, then mana that
# can't be spent to cast nonartifact spells by W/U/B/R/G/C / 10.
# `card_turn_state` aligns with the seats' card rows: 1 for a graveyard card
# put there from the battlefield this turn.
TURN_STATE_LIFE_LOST_INDEX = 0
TURN_STATE_SPELLS_CAST_INDEX = 1
TURN_STATE_NONARTIFACT_MANA_INDEX = 2
TURN_STATE_DIM = 8


class ObservationEncoder:
    """The layout of one `ObservationConfig`: array shapes, dtypes and allocation.

    Native code writes the arrays (`ObservationSpace.encode`). Construction
    refuses a configuration whose decision-memory capacities are incomplete or
    that asks for memory under a deprecated input.
    """

    def __init__(self, config: ObservationConfig) -> None:
        capacities = (
            config.max_decision_records,
            config.max_decision_reference_details,
            config.max_decision_references,
            config.max_decision_members,
        )
        if any(capacities) and not all(capacities):
            raise ValueError("decision memory requires all companion capacities")
        # Input v11 lists each game thing once; deprecated v9/v10 rows also
        # list battlefield objects and spells as cards and have no stack array.
        self.one_token = config.policy_input_version == 11
        if config.max_decision_records and not self.one_token:
            raise ValueError("decision memory requires policy input v11")
        self.config = config
        self.turn_state = config.policy_input_version in (10, 11)
        self.cards_per_player = config.max_cards_per_player
        self.perms_per_player = config.max_permanents_per_player
        self.max_stack_objects = config.max_stack_objects if self.one_token else 0
        # Naming is bounded group prompts, so action capacity is independent
        # of vocabulary size.
        self.max_actions = config.max_actions
        self.max_focus_objects = config.max_focus_objects
        self.max_events = config.max_events
        self.num_zones = len(ZoneEnum.__members__)

        # Row widths; native `observation_layout()` names every column.
        self.player_dim = PLAYER_DIM
        self.card_dim = CARD_DIM
        self.permanent_dim = PERMANENT_DIM if self.one_token else LEGACY_PERMANENT_DIM
        self.relation_dim = RELATION_DIM
        # Family one-hot, then declaration, mode, ability, continuation join
        # and validity.
        self.action_dim = len(ActionEnum.__members__) + 5
        self.event_dim = 12 if config.policy_history_version == 1 else 7

    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            **(
                {
                    key: (capacity, MEMORY_WIDTHS[key])
                    for key, capacity in (
                        ("decision_records", self.config.max_decision_records),
                        ("decision_references", self.config.max_decision_references),
                        (
                            "decision_reference_details",
                            self.config.max_decision_reference_details,
                        ),
                        ("decision_members", self.config.max_decision_members),
                    )
                }
                if self.config.max_decision_records
                else {}
            ),
            "relations": (RELATION_ROWS, self.relation_dim),
            "facts": (self.config.max_facts, FACT_DIM),
            **SHARED_EFFECT_SHAPES,
            "semantic_cards": (2, self.cards_per_player),
            "known_hand": (2, self.cards_per_player, 2),
            **(
                {
                    "known_library": tuple(
                        managym.observation_layout()["known_library"]["shape"]
                    )
                }
                if self.one_token
                else {}
            ),
            "decklists": (2, self.cards_per_player, DECKLIST_DIM),
            **(
                {
                    "player_turn_state": (2, TURN_STATE_DIM),
                    "card_turn_state": (2, self.cards_per_player),
                }
                if self.turn_state
                else {}
            ),
            # Game objects
            "agent_player": (1, self.player_dim),
            "opponent_player": (1, self.player_dim),
            "agent_cards": (self.cards_per_player, self.card_dim),
            "opponent_cards": (self.cards_per_player, self.card_dim),
            "agent_permanents": (self.perms_per_player, self.permanent_dim),
            "opponent_permanents": (self.perms_per_player, self.permanent_dim),
            **(
                {
                    "stack": (self.max_stack_objects, STACK_DIM),
                    "stack_valid": (self.max_stack_objects,),
                }
                if self.one_token
                else {}
            ),
            "name_candidates": (self.max_actions, NAME_DIM),
            "actions": (self.max_actions, self.action_dim),
            "events": (self.max_events, self.event_dim),
            "action_focus": (self.max_actions, self.max_focus_objects),
            "agent_player_valid": (1,),
            "opponent_player_valid": (1,),
            "agent_cards_valid": (self.cards_per_player,),
            "opponent_cards_valid": (self.cards_per_player,),
            "agent_permanents_valid": (self.perms_per_player,),
            "opponent_permanents_valid": (self.perms_per_player,),
            "actions_valid": (self.max_actions,),
            "events_valid": (self.max_events,),
        }

    @property
    def dtypes(self) -> dict[str, np.dtype]:
        return {
            key: np.int32 if key == "action_focus" else np.float32
            for key in self.shapes
        }

    def allocate(self, *leading_dims: int) -> dict[str, np.ndarray]:
        dtypes = self.dtypes
        return {
            key: np.zeros((*leading_dims, *shape), dtype=dtypes[key])
            for key, shape in self.shapes.items()
        }


class ObservationSpace(gym.spaces.Space):
    """
    Gymnasium-compatible observation space optimized for attention processing.
    """

    def __init__(self, config: ObservationConfig | None = None):
        super().__init__(shape=None, dtype=None)
        self.encoder = ObservationEncoder(config or ObservationConfig())
        self.spaces = gym.spaces.Dict(
            {
                name: gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=shape, dtype=np.float32
                )
                for name, shape in self.encoder.shapes.items()
            }
        )
        self.shapes = self.encoder.shapes

    def sample(self, mask=None) -> dict[str, np.ndarray]:
        return {name: space.sample() for name, space in self.spaces.items()}

    def contains(self, x: ObservationBuffers) -> bool:
        return all(
            name in x and x[name].shape == self.shapes[name]
            for name in self.spaces.keys()
        )

    def encode(self, obs: managym.Observation) -> ObservationBuffers:
        """Fresh buffers for ``obs``, written by the native encoder.

        Any observation encodes: live, restored with ``Observation.from_json``
        or a viewer projection. Raises ``managym.ObservationCapacity`` when it
        does not fit this space and ``ValueError`` for state the configured
        input cannot represent.
        """
        out = self.encoder.allocate()
        managym.encode_observation_into(obs, out, self.encoder.config.action_overflow)
        return out

    @property
    def shape(self) -> tuple | None:
        return None

    def __getitem__(self, key: str) -> gym.spaces.Space:
        return self.spaces[key]

    def keys(self) -> KeysView:
        return self.spaces.keys()

    def values(self) -> ValuesView:
        return self.spaces.values()

    def items(self) -> ItemsView:
        return self.spaces.items()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ObservationSpace):
            return False
        if set(self.spaces.keys()) != set(other.spaces.keys()):
            return False
        for key in self.spaces:
            if self.spaces[key] != other.spaces[key]:
                return False
        return True
