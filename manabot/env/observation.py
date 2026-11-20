"""The policy input: named arrays, their shapes and the columns the model reads.

Native code owns the policy input. `managym.encode_observation_into` (one
observation) and the vector env (many) write it, and
`managym.observation_layout()` publishes its layout: array names, axes, fixed
extents, dtypes and named columns. Nothing here is a second copy of that
layout:

- `ObservationSpace` is one `ObservationConfig` plus the layout derived from
  it: `shapes` are the native arrays with each caller-selected (`None`)
  extent filled from the config's capacities, and `allocate` builds them;
- `encode_observation(space, obs)` returns fresh buffers for one observation;
- the column constants below (`PERMANENT_DEFINITION_INDEX`, `FACT_VALUE_INDEX`,
  ...) and fact kinds are looked up by name in the native layout at import, so
  a renamed native column fails the import instead of being read from the
  wrong place.

Native reads the configuration back from the buffers' shapes, so every buffer
is derived from the space the caller supplies, never a native default.
"""

from collections.abc import Mapping
from typing import Any, ItemsView, KeysView, ValuesView
import warnings

import gymnasium as gym
import numpy as np

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

# Release that removes `ObservationSpace.encode`, `space.encoder` and
# `ObservationEncoder`.
REMOVAL_RELEASE = "0.4.0"

# Input v11 with public history and decision memory: every array a v11
# reader can meet. Legacy v9/v10 rows are `observation_layout(legacy=True)`.
_LAYOUT = managym.observation_layout(policy_history=True, decision_memory=True)


def column(array: str, name: str) -> int:
    """Index of the named native column of `array` in input v11."""
    return _LAYOUT[array]["columns"][name]["index"]


def columns(array: str, *names: str) -> list[int]:
    """Indexes of the named native columns of `array`, in the order given."""
    return [column(array, name) for name in names]


def column_names(array: str) -> list[str]:
    """The native column names of `array` in input v11, in column order."""
    return list(_LAYOUT[array]["columns"])


def width(array: str) -> int:
    """Feature width of `array` in input v11."""
    return len(_LAYOUT[array]["columns"])


# Column constants manabot reads, by native column name. Add one only for a
# column the model, preparation or a test reads.

# Action rows: one-hot family, declaration, mode/ability selectors,
# continuation join, validity, then appended columns.
ACTION_TYPE_DIM = column("actions", "declared")
ACTION_DECLARED_INDEX = column("actions", "declared")
ACTION_MODE_INDEX = column("actions", "mode")
ACTION_ABILITY_INDEX = column("actions", "ability")
ACTION_CONTINUATION_INDEX = column("actions", "continuation")
ACTION_VALID_INDEX = column("actions", "valid")
# In v11 this flag marks taps that directly pay a cost (convoke, improvise,
# waterbend). Its position preserves the saved 22-column prefix.
ACTION_PAYS_DIRECTLY_INDEX = column("actions", "pays_directly")
MAX_ACTION_SELECTOR = (1 << 24) - 1
# Name candidate columns: name row + 1, definition + 1, then the first and
# last name rows (+ 1) of a naming group choice.
NAME_DIM = width("name_candidates")

# Mana restricted to spending on Lesson, Shrine and Ally spells: three blocks
# of W/U/B/R/G/C.
PLAYER_RESTRICTED_INDEX = column("agent_player", "restricted_lesson_white")
PLAYER_NO_MAXIMUM_HAND_SIZE_INDEX = column("agent_player", "no_maximum_hand_size")
PLAYER_DIM = width("agent_player")
# Cast-permission terms follow the printed characteristics; validity is last.
CARD_PERMISSION_INDEX = column("agent_cards", "permission")
CARD_DIM = width("agent_cards")

PERMANENT_DEFINITION_INDEX = column("agent_permanents", "definition")
PERMANENT_ONCE_EACH_TURN_INDEX = column("agent_permanents", "once_each_turn_used")
# Object index + 1 of the attachment's host and of the attacker this creature
# blocks: joins, never numeric features.
PERMANENT_ATTACHED_INDEX = column("agent_permanents", "attached_to")
PERMANENT_BLOCKING_INDEX = column("agent_permanents", "blocking")
# Rank of its timestamp among the observation's timestamps.
PERMANENT_TIMESTAMP_INDEX = column("agent_permanents", "timestamp")
PERMANENT_MANA_VALUE_INDEX = column("agent_permanents", "mana_value")
PERMANENT_DIM = width("agent_permanents")
# Permanent columns that hold object links (joins), in model parameter order.
PERMANENT_LINK_INDEXES = (PERMANENT_ATTACHED_INDEX, PERMANENT_BLOCKING_INDEX)

# Input v11 stack rows, top first: one per spell or ability.
STACK_IS_MINE_INDEX = column("stack", "is_mine")
STACK_DEFINITION_INDEX = column("stack", "definition")
STACK_SOURCE_INDEX = column("stack", "source")  # object index + 1: a join
STACK_DIM = width("stack")

RELATION_ROWS = _LAYOUT["relations"]["shape"][0]
RELATION_DIM = width("relations")
RELATION_VALID_INDEX = column("relations", "valid")
RELATION_ELIGIBILITY_INDEX = column("relations", "legal_at_resolution_start")
RELATION_X_INDEX = column("relations", "x")
# A waiting delayed trigger ends this turn.
RELATION_THIS_TURN_INDEX = column("relations", "this_turn")

# Fact rows: one open-vocabulary fact about a player or permanent, joined to
# it like relation references. Ids are learning-token values split into 16-bit
# halves (exact in float32).
FACT_OBJECT_INDEX = column("facts", "object")
FACT_KIND_INDEX = column("facts", "kind")
FACT_ID_INDEX = column("facts", "id_high")  # high, low
# High, low: a keyword, subtype or color an effect names.
FACT_DETAIL_INDEX = column("facts", "detail_high")
# Amounts / 10: count, power or N; then toughness.
FACT_VALUE_INDEX = column("facts", "value")
# 1 until end of turn, 2 indefinite, 3 while controlling source, 4 until your
# next turn.
FACT_DURATION_INDEX = column("facts", "duration")
FACT_ORDER_INDEX = column("facts", "timestamp")  # timestamp rank (1-based)
FACT_SOURCE_INDEX = column("facts", "source_definition")  # definition + 1
FACT_VALID_INDEX = column("facts", "valid")
FACT_DIM = width("facts")
# Fact kinds by native name (`observation_encoder::FactKind`).
_FACT_KINDS: dict[str, int] = _LAYOUT["facts"]["row_kind_names"]
FACT_COUNTER = _FACT_KINDS["counter"]
FACT_SET_NAME = _FACT_KINDS["set_name"]
FACT_USED_ABILITY = _FACT_KINDS["used_ability"]
FACT_COPY = _FACT_KINDS["copy"]
FACT_COPY_EXCEPTION = _FACT_KINDS["copy_exception"]
FACT_TRANSFORMS = _FACT_KINDS["transforms"]
FACT_COMBAT_RESTRICTION = _FACT_KINDS["combat_restriction"]
FACT_EFFECT_PROGRAM = _FACT_KINDS["effect_program"]
FACT_CONTINUATION_PROGRAM = _FACT_KINDS["continuation_program"]
# Largest kind; sizes the model's fact-kind embedding.
FACT_KINDS = max(_FACT_KINDS.values())
# Reserved and never written: continuous-effect records are shared-effect
# rows. Preparation still names it among the kinds whose detail is a name.
FACT_EFFECT = 6
# The id of an effect program that states a player's pooled mana with a
# spend effect or combat expiry (native `observation_encoder::POOLED_MANA`).
POOLED_MANA = "PooledMana"
# The id of an effect program that states a permanent's ward price when the
# ward-cost column cannot, as "Ward—Pay 7 life." (native `observation_encoder::WARD`).
WARD = "Ward"

# Decklist rows: definition id + 1, then main, unseen, sideboard and
# remaining-sideboard counts. A closed opponent's rows are all zero.
DECKLIST_DIM = width("decklists")
# Input v10/v11 turn state. `player_turn_state` has one row per seat, agent
# first: life lost this turn / 20, spells cast this turn / 10, then mana that
# can't be spent to cast nonartifact spells by W/U/B/R/G/C / 10.
# `card_turn_state` aligns with the seats' card rows: 1 for a graveyard card
# put there from the battlefield this turn.
TURN_STATE_LIFE_LOST_INDEX = column("player_turn_state", "life_lost")
TURN_STATE_SPELLS_CAST_INDEX = column("player_turn_state", "spells_cast")
TURN_STATE_NONARTIFACT_MANA_INDEX = column(
    "player_turn_state", "nonartifact_mana_white"
)
TURN_STATE_DIM = width("player_turn_state")


def capacities(config: ObservationConfig) -> dict[str, int]:
    """The extent of each native caller-selected axis under `config`."""
    return {
        "card": config.max_cards_per_player,
        "definition": config.max_cards_per_player,
        "permanent": config.max_permanents_per_player,
        "stack": config.max_stack_objects,
        "action": config.max_actions,
        "focus": config.max_focus_objects,
        "event": config.max_events,
        "fact": config.max_facts,
        "opponent_record": config.max_decision_records,
        "opponent_reference": config.max_decision_references,
        "opponent_reference_detail": config.max_decision_reference_details,
        "opponent_member": config.max_decision_members,
    }


class ObservationSpace(gym.spaces.Space):
    """One `ObservationConfig` and the native layout of its policy input.

    `shapes` and `dtypes` are `managym.observation_layout()` for the config's
    input version, history and memory, with every `None` extent filled from
    its capacities. Construction refuses a configuration whose decision-memory
    capacities are incomplete or that asks for memory under a deprecated
    input. The space writes nothing; `encode_observation` does.
    """

    def __init__(self, config: ObservationConfig | None = None) -> None:
        super().__init__(shape=None, dtype=None)
        config = config or ObservationConfig()
        memory = (
            config.max_decision_records,
            config.max_decision_reference_details,
            config.max_decision_references,
            config.max_decision_members,
        )
        if any(memory) and not all(memory):
            raise ValueError("decision memory requires all companion capacities")
        # Input v11 lists each game thing once; deprecated v9/v10 rows also
        # list battlefield objects and spells as cards and have no stack array.
        self.one_token = config.policy_input_version == 11
        if config.max_decision_records and not self.one_token:
            raise ValueError("decision memory requires policy input v11")
        self.config = config
        self.turn_state = config.policy_input_version in (10, 11)
        layout = managym.observation_layout(
            policy_history=config.policy_history_version == 1,
            turn_state=self.turn_state,
            legacy=not self.one_token,
            decision_memory=bool(config.max_decision_records),
        )
        extents = capacities(config)
        self.shapes: dict[str, tuple[int, ...]] = {
            key: tuple(
                extents[axis] if extent is None else extent
                for axis, extent in zip(array["axes"], array["shape"], strict=True)
            )
            for key, array in layout.items()
        }
        # Numpy scalar types: checkpoints record `str()` of each.
        self.dtypes: dict[str, type[np.generic]] = {
            key: np.dtype(array["dtype"]).type for key, array in layout.items()
        }
        self.spaces = gym.spaces.Dict(
            {
                name: gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=shape, dtype=np.float32
                )
                for name, shape in self.shapes.items()
            }
        )

    def allocate(self, *leading_dims: int) -> dict[str, np.ndarray]:
        """Zeroed buffers for this space, with `leading_dims` before each shape."""
        return {
            key: np.zeros((*leading_dims, *shape), dtype=self.dtypes[key])
            for key, shape in self.shapes.items()
        }

    @property
    def encoder(self) -> "ObservationSpace":
        """Deprecated: the space is its own layout."""
        warnings.warn(
            "ObservationSpace.encoder is deprecated and removed in "
            f"{REMOVAL_RELEASE}; read config, shapes, dtypes and allocate from "
            "the space",
            DeprecationWarning,
            stacklevel=2,
        )
        return self

    def encode(self, obs: managym.Observation) -> ObservationBuffers:
        """Deprecated: call `encode_observation(space, obs)`."""
        warnings.warn(
            "ObservationSpace.encode is deprecated and removed in "
            f"{REMOVAL_RELEASE}; call "
            "manabot.env.observation.encode_observation(space, obs)",
            DeprecationWarning,
            stacklevel=2,
        )
        return encode_observation(self, obs)

    def sample(self, mask: Any = None) -> dict[str, np.ndarray]:
        return self.spaces.sample()

    def contains(self, x: ObservationBuffers) -> bool:
        return all(
            name in x and x[name].shape == self.shapes[name]
            for name in self.spaces.keys()
        )

    def __getitem__(self, key: str) -> gym.spaces.Space:
        return self.spaces[key]

    def keys(self) -> KeysView:
        return self.spaces.keys()

    def values(self) -> ValuesView:
        return self.spaces.values()

    def items(self) -> ItemsView:
        return self.spaces.items()

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ObservationSpace)
            and self.keys() == other.keys()
            and all(space == other[name] for name, space in self.items())
        )


def encode_observation(
    space: ObservationSpace, obs: managym.Observation
) -> ObservationBuffers:
    """Fresh buffers for `obs` in `space`, written by the native encoder.

    Any observation encodes: live, restored with ``Observation.from_json`` or
    a viewer projection. Raises ``managym.ObservationCapacity`` when it does
    not fit the space and ``ValueError`` for state the configured input cannot
    represent.
    """
    out = space.allocate()
    managym.encode_observation_into(obs, out, space.config.action_overflow)
    return out


def __getattr__(name: str) -> Any:
    if name == "ObservationEncoder":
        warnings.warn(
            f"ObservationEncoder is deprecated and removed in {REMOVAL_RELEASE}; "
            "ObservationSpace(config) holds the layout",
            DeprecationWarning,
            stacklevel=2,
        )
        return ObservationSpace
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
