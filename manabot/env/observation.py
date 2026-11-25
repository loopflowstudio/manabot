"""The policy input: named arrays, their shapes and the columns the model reads.

Native code owns the policy input. `managym.encode_observation_batch` and the
vector env encode packed `managym.ObservationBatch` rows against a resolved
`managym.ObservationLayout`, and
`managym.observation_layout()` publishes its schema: array names, axes, fixed
extents, dtypes, named columns and row-kind maps. Nothing here is a second copy
of that layout:

- `ObservationSpace` is one `ObservationConfig` plus `space.layout`
  (`managym.ObservationLayout` at the config's input version, history, memory
  and capacities), from which `shapes` and `dtypes` come;
- `encode_policy_batch(space, observations)` encodes viewer projections;
- the column constants below (`PERMANENT_DEFINITION_INDEX`, `FACT_VALUE_INDEX`,
  ...) and fact kinds are looked up by name in the native layout at import, so
  a renamed native column fails the import instead of being read from the
  wrong place.

Every batch is encoded under the layout the caller's space supplies, never a
native default.
"""

from collections.abc import Sequence
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

__all__ = [
    "ObservationSpace",
    "column",
    "columns",
    "column_names",
    "width",
    "encode_policy_batch",
]

# Release that removes the `PolicyObservation` alias here.
REMOVAL_RELEASE = "0.3.0"

# Input v11 with public history and decision memory: every array a v11
# reader can meet.
V11_LAYOUT = managym.observation_layout(policy_history=True, decision_memory=True)


def column(array: str, name: str) -> int:
    """Index of the named native column of `array` in input v11."""
    return V11_LAYOUT[array]["columns"][name]["index"]


def columns(array: str, *names: str) -> list[int]:
    """Indexes of the named native columns of `array`, in the order given."""
    return [column(array, name) for name in names]


def column_names(array: str) -> list[str]:
    """The native column names of `array` in input v11, in column order."""
    return list(V11_LAYOUT[array]["columns"])


def width(array: str) -> int:
    """Feature width of `array` in input v11."""
    return len(V11_LAYOUT[array]["columns"])


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

RELATION_ROWS = V11_LAYOUT["relations"]["shape"][0]
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
_FACT_KINDS: dict[str, int] = V11_LAYOUT["facts"]["row_kind_names"]
FACT_COUNTER = _FACT_KINDS["counter"]
FACT_SET_NAME = _FACT_KINDS["set_name"]
FACT_ABILITY_RESOLUTION = _FACT_KINDS["ability_resolution"]
FACT_TRIGGERING_AMOUNT = _FACT_KINDS["triggering_amount"]
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


class ObservationSpace(gym.spaces.Space):
    """One `ObservationConfig` and the native layout of its policy input.

    `shapes` and `dtypes` come from the configured native `layout`.
    Construction refuses a configuration whose decision-memory
    capacities are incomplete. The space writes nothing; `encode_policy_batch` encodes.
    """

    def __init__(self, config: ObservationConfig | None = None) -> None:
        super().__init__(shape=None, dtype=None)
        config = config or ObservationConfig()
        self.config = config
        self.layout = managym.ObservationLayout(
            policy_input_version=config.policy_input_version,
            policy_history=config.policy_history_version == 1,
            max_cards_per_player=config.max_cards_per_player,
            max_permanents_per_player=config.max_permanents_per_player,
            max_stack_objects=config.max_stack_objects,
            max_actions=config.max_actions,
            max_focus_objects=config.max_focus_objects,
            max_events=config.max_events,
            max_facts=config.max_facts,
            max_decision_records=config.max_decision_records,
            max_decision_references=config.max_decision_references,
            max_decision_reference_details=config.max_decision_reference_details,
            max_decision_members=config.max_decision_members,
        )
        self.shapes = self.layout.shapes
        self.axis_buffers = self.layout.axis_buffers
        # Checkpoints record str() of each NumPy scalar type.
        self.dtypes = {
            key: np.dtype(dtype).type for key, dtype in self.layout.dtypes.items()
        }
        self.spaces = gym.spaces.Dict(
            {
                name: gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=shape, dtype=np.float32
                )
                for name, shape in self.shapes.items()
            }
        )

    def __reduce__(self) -> tuple[type, tuple[ObservationConfig]]:
        return ObservationSpace, (self.config,)

    def sample(self, mask: Any = None) -> dict[str, np.ndarray]:
        return self.spaces.sample()

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


def step_buffers(n: int) -> dict[str, np.ndarray]:
    """Per-stream outcomes the native vector writes beside its observation rows."""
    return {
        "rewards": np.zeros(n, dtype=np.float64),
        "terminated": np.zeros(n, dtype=np.uint8),
        "truncated": np.zeros(n, dtype=np.uint8),
    }


def encode_policy_batch(
    space: ObservationSpace,
    observations: Sequence[managym.Observation],
    *,
    blank_decision_memory: bool = False,
) -> managym.ObservationBatch:
    """Encode viewer projections in order using the checkpoint's capacity and overflow policy."""
    batch = managym.encode_observation_batch(
        observations, space.layout, space.config.action_overflow
    )
    return batch.blank_decision_memory() if blank_decision_memory else batch


def __getattr__(name: str) -> Any:
    if name == "PolicyObservation":
        from manabot.policy.spans import PolicyObservation

        warnings.warn(
            f"PolicyObservation moved to manabot.policy.spans; removal in {REMOVAL_RELEASE}",
            DeprecationWarning,
            stacklevel=2,
        )
        return PolicyObservation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
