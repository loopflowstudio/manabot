"""
observation.py
Defines a Gymnasium-compatible observation space from managym observations.

This module is responsible for:
1. Converting managym observations into fixed-size tensors,
2. Structuring tensors in a natural way for the agent to build a policy, and
3. Defining the observation space as a Gymnasium space.

Additional updates in this version:
- Focus object indices are naturally gathered when encoding actions.
- Validity masks (if provided) are passed to the GameObject Encoder, Action Encoder,
  Policy head, and multi-head attention.
- The last element of each card and permanent feature vector is reserved as an
  is_valid flag.
"""

from typing import Dict, ItemsView, KeysView, List, Tuple, ValuesView

import gymnasium as gym

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
import numpy as np

# Local imports
from manabot.infra.config import ObservationConfig
from manabot.infra.log import getLogger
import managym

# Native action tensor: family, declaration, mode/ability selectors, validity.
ACTION_TYPE_DIM = len(ActionEnum)
ACTION_DECLARED_INDEX = ACTION_TYPE_DIM
ACTION_MODE_INDEX = ACTION_DECLARED_INDEX + 1
ACTION_ABILITY_INDEX = ACTION_MODE_INDEX + 1
ACTION_CONTINUATION_INDEX = ACTION_ABILITY_INDEX + 1
MAX_ACTION_SELECTOR = (1 << 24) - 1
PERMANENT_DEFINITION_INDEX = 24
RELATION_ROWS = 64
LEGACY_RELATION_DIM = 32
RELATION_DIM = 33
# V4 appends eligibility; row validity keeps its v2/v3 position.
RELATION_VALID_INDEX = 31
# Policy input v5 encodes every Observation fact; managym's
# observation_encoder.rs owns the layout and these mirror it exactly. V5 widens
# the player, card and permanent rows; v1-v4 keep the legacy widths.
RELATION_DIM_V5 = 48
RELATION_X_INDEX = 33
RELATION_KEYWORDS_INDEX = 34  # 14 granted spell keywords, KeywordData order
SPEND_SUBTYPES = ("Lesson", "Shrine", "Ally")
LEGACY_PLAYER_DIM = 40
PLAYER_RESTRICTED_INDEX = LEGACY_PLAYER_DIM
PLAYER_HAND_ONLY_INDEX = PLAYER_RESTRICTED_INDEX + len(SPEND_SUBTYPES) * 6
PLAYER_NO_MAXIMUM_HAND_SIZE_INDEX = PLAYER_HAND_ONLY_INDEX + 1
PLAYER_TURN_INDEX = PLAYER_NO_MAXIMUM_HAND_SIZE_INDEX + 1
PLAYER_SEAT_INDEX = PLAYER_TURN_INDEX + 1
PLAYER_KEEPS_RED_MANA_INDEX = PLAYER_SEAT_INDEX + 1
# Turn history: permanents sacrificed this turn.
PLAYER_SACRIFICED_INDEX = PLAYER_KEEPS_RED_MANA_INDEX + 1
PLAYER_DIM = PLAYER_SACRIFICED_INDEX + 1
LEGACY_CARD_DIM = 41
# V5 card terms start where v4's validity flag was; validity stays last.
CARD_PERMISSION_INDEX = LEGACY_CARD_DIM - 1
CARD_X_INDEX = CARD_PERMISSION_INDEX + 12
CARD_DIM = CARD_X_INDEX + 2 + 1
LEGACY_PERMANENT_DIM = 29
# V5 permanent terms start where v4's validity flag was; validity stays last.
# Open-ended vocabularies are fact rows (below), not columns.
PERMANENT_EXHAUSTED_INDEX = LEGACY_PERMANENT_DIM - 1
PERMANENT_ONCE_EACH_TURN_INDEX = PERMANENT_EXHAUSTED_INDEX + 1
PERMANENT_FIREBENDING_INDEX = PERMANENT_ONCE_EACH_TURN_INDEX + 1
# Object index + 1 of the attachment's host: a join, never a numeric feature.
PERMANENT_ATTACHED_INDEX = PERMANENT_FIREBENDING_INDEX + 1
PERMANENT_GRANTED_LAND_INDEX = PERMANENT_ATTACHED_INDEX + 1
# Object index + 1 of the attacker this creature blocks: a join.
PERMANENT_BLOCKING_INDEX = PERMANENT_GRANTED_LAND_INDEX + 1
PERMANENT_ABILITIES_LOST_INDEX = PERMANENT_BLOCKING_INDEX + 1
# Rank of its timestamp among the observation's timestamps.
PERMANENT_TIMESTAMP_INDEX = PERMANENT_ABILITIES_LOST_INDEX + 1
PERMANENT_DIM = PERMANENT_TIMESTAMP_INDEX + 2
# V5 fact rows: one open-vocabulary fact about a player or permanent, joined to
# it like relation references. Ids are learning-token values split into 16-bit
# halves (exact in float32).
FACT_ROWS = 128
FACT_OBJECT_INDEX = 0
FACT_KIND_INDEX = 1
FACT_ID_INDEX = 2  # high, low
FACT_DETAIL_INDEX = 4  # high, low: a keyword, subtype or color an effect names
FACT_VALUE_INDEX = 6  # amounts / 10: count, power or N; then toughness
FACT_DURATION_INDEX = 8  # 1 until end of turn, 2 indefinite
FACT_ORDER_INDEX = 9  # timestamp rank (1-based)
FACT_SOURCE_INDEX = 10  # source definition + 1
FACT_VALID_INDEX = 11
FACT_DIM = 12
# FactKind values (observation_encoder.rs).
FACT_COUNTER = 1
FACT_ADDED_SUBTYPE = 2
FACT_SET_NAME = 3
FACT_SET_CREATURE_TYPE = 4
FACT_USED_ABILITY = 5
FACT_EFFECT = 6
FACT_COPY = 7
FACT_COPY_EXCEPTION = 8
FACT_KINDS = 8
# Permanent columns that hold object links (joins), in model parameter order.
PERMANENT_LINK_INDEXES = (PERMANENT_ATTACHED_INDEX, PERMANENT_BLOCKING_INDEX)


# -----------------------------------------------------------------------------
# Observation Encoding
# -----------------------------------------------------------------------------
def _split_id(ident: int) -> Tuple[float, float]:
    """A vocabulary id as 16-bit halves, exact in float32."""
    ident &= 0xFFFFFFFF
    return float(ident >> 16), float(ident & 0xFFFF)


def _timestamp(perm: managym.Permanent) -> int:
    """A permanent's timestamp: renewed by an attachment, otherwise its id."""
    return perm.id if perm.timestamp is None else perm.timestamp


def _timestamp_ranks(obs: managym.Observation) -> Dict[int, float]:
    """Rank (1-based) of every permanent and effect timestamp in `obs`."""
    permanents = [*obs.agent_permanents, *obs.opponent_permanents]
    stamps = {_timestamp(perm) for perm in permanents}
    for effects in (
        obs.agent.effects,
        obs.opponent.effects,
        *(perm.effects for perm in permanents),
    ):
        stamps.update(effect.timestamp for effect in effects)
    return {stamp: float(rank + 1) for rank, stamp in enumerate(sorted(stamps))}


class ObservationEncoder:
    """
    Encodes each typed chunk (player, cards, permanents, etc.) into fixed-size arrays,
    along with corresponding ID and validity masks. These validity masks can be used
    later by the agent to mask out invalid slots in attention, action encoding, etc.
    """

    def __init__(self, config: ObservationConfig) -> None:
        self.config = config
        self.shows_everything = config.policy_input_version >= 5
        if self.shows_everything:
            self.relation_dim = RELATION_DIM_V5
        elif config.policy_input_version >= 4:
            self.relation_dim = RELATION_DIM
        else:
            self.relation_dim = LEGACY_RELATION_DIM
        self.cards_per_player = config.max_cards_per_player
        self.perms_per_player = config.max_permanents_per_player
        self.max_actions = config.max_actions
        self.max_focus_objects = config.max_focus_objects
        self.max_events = config.max_events

        self.num_phases = len(PhaseEnum.__members__)
        self.num_steps = len(StepEnum.__members__)
        self.num_zones = len(ZoneEnum.__members__)
        self.num_actions = len(ActionEnum.__members__)
        self.num_event_types = len(EventTypeEnum.__members__)
        self.num_event_entity_kinds = len(EventEntityKindEnum.__members__)

        # Define dimensions for players, cards, permanents.
        # Player: life + is_active + zones + phase/step one-hots + gy
        # lessons + until-end-of-combat mana + pooled and spend-restricted colors;
        # v5 adds restricted mana by spending subtype, the hand-only lock, no
        # maximum hand size, turn number, seat and kept red mana.
        # Card: zone one-hot + is_mine + P/T + mana value + 6 type flags +
        # 12 keywords + is_token/is_ally/is_lesson tags + ward flag/cost +
        # kicker flag/cost + hexproof + outside + indestructible + legendary;
        # v5 adds cast-permission terms and announced X; validity is last.
        current = self.shows_everything
        self.player_dim = PLAYER_DIM if current else LEGACY_PLAYER_DIM
        self.card_dim = CARD_DIM if current else LEGACY_CARD_DIM
        # Permanent: is_mine, tapped, damage, summoning sick, +1/+1 counters,
        # can't-be-blocked-this-turn, effective power/toughness, animated
        # (earthbent land), exile-linkage (Jailer), 13 effective-keyword flags
        # (printed + until-EOT grants), death-to-exile replacement, definition,
        # temporary Avatar subtype, attacking; v5 adds used exhaust and
        # once-each-turn ability counts, granted firebending, attachment join,
        # granted land, blocking join, lost abilities and timestamp rank;
        # validity is last. Counter kinds, effect-set subtypes and
        # characteristics, used abilities and continuous-effect records are
        # v5 fact rows.
        self.permanent_dim = PERMANENT_DIM if current else LEGACY_PERMANENT_DIM
        self.event_dim = 12 if config.policy_history_version == 1 else 7

        # Preserve attack/decline independently of action family and focus.
        self.action_dim = (
            self.num_actions + 5
        )  # declaration, mode, ability, continuation join, validity

        # This maps object IDS to their ultimate Game Object index.
        # This is used to map the focus object indices in the action space
        # to the actual object indices in the game object tensor.
        self.object_to_index: Dict[int, int] = {}
        self.current_object_index = 0

    @property
    def shapes(self) -> Dict[str, Tuple[int, ...]]:
        return {
            **(
                {"relations": (RELATION_ROWS, self.relation_dim)}
                if self.config.policy_input_version >= 2
                else {}
            ),
            **({"facts": (FACT_ROWS, FACT_DIM)} if self.shows_everything else {}),
            "semantic_cards": (2, self.cards_per_player),
            "known_hand": (2, self.cards_per_player, 2),
            # Game objects
            "agent_player": (1, self.player_dim),
            "opponent_player": (1, self.player_dim),
            "agent_cards": (self.cards_per_player, self.card_dim),
            "opponent_cards": (self.cards_per_player, self.card_dim),
            "agent_permanents": (self.perms_per_player, self.permanent_dim),
            "opponent_permanents": (self.perms_per_player, self.permanent_dim),
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
    def dtypes(self) -> Dict[str, np.dtype]:
        return {
            key: np.int32 if key == "action_focus" else np.float32
            for key in self.shapes
        }

    def allocate(self, *leading_dims: int) -> Dict[str, np.ndarray]:
        return {
            key: np.zeros((*leading_dims, *shape), dtype=self.dtypes[key])
            for key, shape in self.shapes.items()
        }

    def encode(self, obs: managym.Observation) -> Dict[str, np.ndarray]:
        # Fixed training tensors must never turn a complete rules decision into
        # a smaller, apparently legal policy decision.
        for label, size, capacity in (
            ("actions", len(obs.action_space.actions), self.max_actions),
            ("agent cards", len(obs.agent_cards), self.cards_per_player),
            ("opponent cards", len(obs.opponent_cards), self.cards_per_player),
            ("agent permanents", len(obs.agent_permanents), self.perms_per_player),
            (
                "opponent permanents",
                len(obs.opponent_permanents),
                self.perms_per_player,
            ),
        ):
            if size > capacity:
                raise ValueError(
                    f"Observation capacity exceeded: {label} {size} > {capacity}"
                )
        if any(
            len(action.focus) > self.max_focus_objects
            for action in obs.action_space.actions
        ):
            raise ValueError("Observation capacity exceeded: action focus")
        for action in obs.action_space.actions:
            if action.continuation_index is not None and (
                self.config.policy_input_version < 2
                or not 0 <= action.continuation_index < len(obs.continuation)
            ):
                raise ValueError("invalid action continuation reference")
        out = {
            "semantic_cards": np.zeros((2, self.cards_per_player), dtype=np.float32),
            "known_hand": np.zeros((2, self.cards_per_player, 2), dtype=np.float32),
        }

        ## NOTE: It is very important that we encode in this exact
        ## order.
        # 1. ActorCritic player
        # 2. Opponent player
        # 3. ActorCritic cards
        # 4. Opponent cards
        # 5. ActorCritic permanents
        # 6. Opponent permanents
        # This ensures that the object_to_index mapping is always
        # consistent and matches the actual array order, because
        # this is also the concatenation order.

        log = getLogger(__name__).getChild("encode")

        self.object_to_index = {}
        self.current_object_index = 0

        for key, player in (
            ("agent_player", obs.agent),
            ("opponent_player", obs.opponent),
        ):
            out[key] = self._encode_player_features(player, obs.turn)[np.newaxis, ...]
        for key, cards, is_mine in (
            ("agent_cards", obs.agent_cards, 1.0),
            ("opponent_cards", obs.opponent_cards, 0.0),
        ):
            out[key] = self._encode_cards(
                cards, is_mine=is_mine, viewer=obs.agent.id, turn=obs.turn.turn_number
            )
        # Outside copies occupy explicit rows after visible owner cards. They
        # have no zone bits; the outside marker identifies their location.
        for seat, (cards, player) in enumerate(
            ((obs.agent_cards, obs.agent), (obs.opponent_cards, obs.opponent))
        ):
            if len(player.known_hand) > self.cards_per_player:
                raise ValueError("known hand exceeds observation capacity")
            for slot, card in enumerate(cards):
                out["semantic_cards"][seat, slot] = card.registry_key + 1
            for slot, (definition, count) in enumerate(
                sorted(player.known_hand.items())
            ):
                out["known_hand"][seat, slot] = (int(definition) + 1, count)
        outside = obs.agent_sideboard
        start = len(obs.agent_cards)
        if start + len(outside) > self.cards_per_player:
            raise ValueError(
                "observation capacity exceeded: agent_cards plus sideboard"
            )
        for offset, card in enumerate(outside):
            out["semantic_cards"][0, start + offset] = card.registry_key + 1
            row = start + offset
            out["agent_cards"][row, 7] = 1.0
            out["agent_cards"][row, 37] = 1.0
            out["agent_cards"][row, -1] = 1.0
            self.object_to_index[card.candidate_id] = 2 + row
        for key, perms, is_mine in (
            ("agent_permanents", obs.agent_permanents, 1.0),
            ("opponent_permanents", obs.opponent_permanents, 0.0),
        ):
            out[key] = self._encode_perms(perms, is_mine=is_mine)
        if self.shows_everything:
            # Hosts may be listed after their attachments, so links follow indexing.
            ranks = _timestamp_ranks(obs)
            for key, perms in (
                ("agent_permanents", obs.agent_permanents),
                ("opponent_permanents", obs.opponent_permanents),
            ):
                for row, perm in enumerate(perms):
                    for column, linked in (
                        (PERMANENT_ATTACHED_INDEX, perm.attached_to),
                        (PERMANENT_BLOCKING_INDEX, perm.blocking),
                    ):
                        if linked is not None:
                            out[key][row, column] = (
                                self.object_to_index.get(linked, -1) + 1
                            )
                    out[key][row, PERMANENT_TIMESTAMP_INDEX] = ranks[_timestamp(perm)]
            out["facts"] = self._encode_facts(obs, ranks)

        # Validity masks
        out["agent_player_valid"] = np.ones((1,), dtype=np.float32)
        out["opponent_player_valid"] = np.ones((1,), dtype=np.float32)
        for key in (
            "agent_cards",
            "opponent_cards",
            "agent_permanents",
            "opponent_permanents",
        ):
            out[f"{key}_valid"] = out[key][..., -1].astype(np.float32)

        # Actionspace
        out["actions"], out["action_focus"] = self._encode_actions(obs)
        out["actions_valid"] = out["actions"][..., -1].astype(np.float32)
        if self.config.policy_history_version == 1:
            rows = obs.encode_policy_history(self.max_events, self.cards_per_player)
            out["events"] = np.zeros(
                (self.max_events, self.event_dim), dtype=np.float32
            )
            out["events_valid"] = np.zeros(self.max_events, dtype=np.float32)
            if rows:
                out["events"][: len(rows)] = rows
                out["events_valid"][: len(rows)] = 1
        else:
            out["events"], out["events_valid"] = self._encode_events(obs.recent_events)

        if self.config.policy_input_version >= 2:
            out["relations"] = self._encode_relations(
                obs.continuation, obs.agent.player_index
            )

        for key, value in out.items():
            log.debug(f"[SHAPES] {key}: {value.shape}")
        return out

    def _encode_facts(
        self, obs: managym.Observation, ranks: Dict[int, float]
    ) -> np.ndarray:
        """Reference implementation of native v5 fact rows."""
        rows: list[np.ndarray] = []

        def push(
            obj: int,
            kind: int,
            ident: int,
            values: Tuple[float, float],
            detail: int = 0,
            provenance: tuple[float, int, int | None] | None = None,
        ) -> None:
            """One row; `provenance` is an effect's (duration, timestamp,
            source registry key), as native `Fact.effect`."""
            row = np.zeros(FACT_DIM, dtype=np.float32)
            row[FACT_OBJECT_INDEX] = self.object_to_index.get(obj, -1) + 1
            row[FACT_KIND_INDEX] = kind
            row[FACT_ID_INDEX : FACT_ID_INDEX + 2] = _split_id(ident)
            row[FACT_DETAIL_INDEX : FACT_DETAIL_INDEX + 2] = _split_id(detail)
            row[FACT_VALUE_INDEX : FACT_VALUE_INDEX + 2] = values
            if provenance is not None:
                duration, timestamp, source = provenance
                row[FACT_DURATION_INDEX] = duration
                row[FACT_ORDER_INDEX] = ranks[timestamp]
                if source is not None:
                    row[FACT_SOURCE_INDEX] = source + 1
            row[FACT_VALID_INDEX] = 1.0
            rows.append(row)

        token = managym.learning_token_value
        permanents = [*obs.agent_permanents, *obs.opponent_permanents]
        for perm in permanents:
            if perm.plus1_counters > 0:
                push(
                    perm.id,
                    FACT_COUNTER,
                    token("CounterKind::plus_one_plus_one"),
                    (perm.plus1_counters / 10.0, 0.0),
                )
            for kind, count in sorted(perm.counters.items()):
                push(
                    perm.id,
                    FACT_COUNTER,
                    token(f"CounterKind::{kind}"),
                    (count / 10.0, 0.0),
                )
            for subtype in perm.temp_subtypes:
                push(perm.id, FACT_ADDED_SUBTYPE, token(subtype), (1.0, 0.0))
            if perm.name is not None:
                push(perm.id, FACT_SET_NAME, token(perm.name), (1.0, 0.0))
            for creature_type in perm.creature_types or ():
                push(perm.id, FACT_SET_CREATURE_TYPE, token(creature_type), (1.0, 0.0))
            for index in perm.used_abilities:
                push(perm.id, FACT_USED_ABILITY, index, (1.0, 0.0))

        def push_modification(
            obj: int, kind: int, effect: managym.ContinuousEffect
        ) -> None:
            ident = token(f"Modification::{effect.modification}")
            amount = effect.power if effect.power is not None else effect.amount
            values = (
                (amount or 0) / 10.0,
                (effect.toughness or 0) / 10.0,
            )
            details = [
                *(token(name) for name in (*effect.keywords, *effect.subtypes)),
                *(
                    [token(f"Color::{effect.color}")]
                    if effect.color is not None
                    else []
                ),
            ] or [0]
            duration = 1.0 if effect.duration == "EndOfTurn" else 2.0
            provenance = (duration, effect.timestamp, effect.source_registry_key)
            for detail in details:
                push(obj, kind, ident, values, detail, provenance)

        for perm in permanents:
            if perm.copy is None:
                continue
            push(
                perm.id,
                FACT_COPY,
                token("CopyEffect"),
                (0.0, 0.0),
                provenance=(2.0, _timestamp(perm), perm.copy.registry_key),
            )
            for exception in perm.copy.exceptions:
                push_modification(perm.id, FACT_COPY_EXCEPTION, exception)
        holders = [
            (obs.agent.id, obs.agent.effects),
            (obs.opponent.id, obs.opponent.effects),
            *((perm.id, perm.effects) for perm in permanents),
        ]
        for obj, effects in holders:
            for effect in effects:
                push_modification(obj, FACT_EFFECT, effect)
        if len(rows) > FACT_ROWS:
            raise ValueError(
                f"observation capacity exceeded: facts {len(rows)} > {FACT_ROWS}"
            )
        out = np.zeros((FACT_ROWS, FACT_DIM), dtype=np.float32)
        if rows:
            out[: len(rows)] = rows
        return out

    def _encode_relations(
        self, frames: list[managym.ContinuationFrame], viewer: int
    ) -> np.ndarray:
        """Reference implementation of native current-continuation relation rows."""
        rows: list[np.ndarray] = []

        def _reference(row: np.ndarray, value: managym.VisibleReference) -> None:
            row[23] = 0 if value.definition is None else value.definition + 1
            row[24] = self.object_to_index.get(value.object, -1) + 1
            if value.player is not None:
                row[25] = row[24] = 1 if value.player == viewer else 2

        for index, frame in enumerate(frames):
            row = np.zeros(self.relation_dim, dtype=np.float32)
            row[:5] = (
                1,
                index + 1,
                0 if frame.parent is None else frame.parent + 1,
                frame.kind,
                1 if frame.controller == viewer else 2,
            )
            row[5] = (
                0 if frame.source.definition is None else frame.source.definition + 1
            )
            row[6] = self.object_to_index.get(frame.source.object, -1) + 1
            row[7:12] = (
                0 if frame.mode is None else frame.mode + 1,
                0 if frame.ability is None else frame.ability + 1,
                frame.kicked,
                0 if frame.active_clause is None else frame.active_clause + 1,
                frame.cost is not None,
            )
            if frame.cost is not None:
                row[12:19] = frame.cost
            row[19:22] = (frame.lesson, frame.waterbend, frame.can_decline)
            row[30] = frame.resolutions_this_turn
            row[28] = frame.decision
            row[RELATION_VALID_INDEX] = 1
            if self.shows_everything:
                row[RELATION_X_INDEX] = frame.x
                # Granted keywords fill the row's remaining columns.
                row[RELATION_KEYWORDS_INDEX:] = frame.granted_keywords
            rows.append(row)
            for ordinal, target in enumerate(frame.targets, 1):
                target_row = row.copy()
                target_row[0], target_row[22], target_row[29] = (
                    2,
                    target.clause + 1,
                    ordinal,
                )
                if self.relation_dim >= RELATION_DIM:
                    legal = target.legal_at_resolution_start
                    target_row[32] = 0 if legal is None else (1 if legal else 2)
                _reference(target_row, target.reference)
                rows.append(target_row)
            for kind, reference in ((3, frame.context), (4, frame.sacrifice)):
                if reference is not None:
                    context_row = row.copy()
                    context_row[0], context_row[22], context_row[29] = kind, 1, 1
                    _reference(context_row, reference)
                    rows.append(context_row)
            for pair in frame.hybrid:
                hybrid = row.copy()
                hybrid[0], hybrid[26], hybrid[27] = 5, pair[0] + 1, pair[1] + 1
                rows.append(hybrid)
            for output in frame.mana_outputs:
                for restricted, amounts in (
                    (False, output.mana),
                    (True, output.lesson),
                ):
                    produced = row.copy()
                    produced[0], produced[7] = 6, output.mode + 1
                    produced[12:18] = amounts
                    produced[19] = restricted
                    rows.append(produced)
        if len(rows) > RELATION_ROWS:
            raise ValueError("observation capacity exceeded: continuation relations")
        result = np.zeros((RELATION_ROWS, self.relation_dim), dtype=np.float32)
        if rows:
            result[: len(rows)] = rows
        return result

    # -------------------------------------------------------------------------
    # Players (with validity mask support)
    # -------------------------------------------------------------------------
    def _encode_player_features(
        self, player: managym.Player, turn: managym.Turn
    ) -> np.ndarray:
        arr = np.zeros(self.player_dim, dtype=np.float32)
        arr[0] = float(player.life) / 20.0
        arr[1] = float(player.is_active)
        zone_start = 2
        zone_counts = np.asarray(player.zone_counts[: self.num_zones], dtype=np.float32)
        arr[zone_start : zone_start + len(zone_counts)] = zone_counts / 60.0
        zone_end = zone_start + self.num_zones

        phase_start = zone_end
        phase = int(turn.phase)
        if 0 <= phase < self.num_phases:
            arr[phase_start + phase] = 1.0

        step_start = phase_start + self.num_phases
        step = int(turn.step)
        if 0 <= step < self.num_steps:
            arr[step_start + step] = 1.0

        arr[step_start + self.num_steps] = float(player.graveyard_lessons) / 10.0
        arr[step_start + self.num_steps + 1] = float(player.combat_mana) / 10.0

        arr[28:34] = np.asarray(player.mana, dtype=np.float32) / 10.0
        arr[34:40] = np.asarray(player.lesson_mana, dtype=np.float32) / 10.0
        if self.player_dim == PLAYER_DIM:
            # A class admitting several subtypes counts under each.
            for index, subtype in enumerate(SPEND_SUBTYPES):
                start = PLAYER_RESTRICTED_INDEX + 6 * index
                for restricted in player.restricted_mana:
                    if subtype in (*restricted.spells, *restricted.abilities):
                        arr[start : start + 6] += (
                            np.asarray(restricted.mana, dtype=np.float32) / 10.0
                        )
            arr[PLAYER_HAND_ONLY_INDEX] = float(player.hand_only_until is not None)
            arr[PLAYER_NO_MAXIMUM_HAND_SIZE_INDEX] = float(player.no_maximum_hand_size)
            arr[PLAYER_TURN_INDEX] = float(turn.turn_number) / 20.0
            # Seat 0 is on the play.
            arr[PLAYER_SEAT_INDEX] = float(player.player_index)
            arr[PLAYER_KEEPS_RED_MANA_INDEX] = float(player.keeps_red_mana)
            arr[PLAYER_SACRIFICED_INDEX] = (
                float(player.turn_history.permanents_sacrificed) / 10.0
            )

        self.object_to_index[player.id] = self.current_object_index
        self.current_object_index += 1
        return arr

    # -------------------------------------------------------------------------
    # Cards (with validity mask support)
    # -------------------------------------------------------------------------
    def _encode_cards(
        self, cards: List[managym.Card], is_mine: float, viewer: int, turn: int
    ) -> np.ndarray:
        return self._encode_objects(
            items=cards,
            max_items=self.cards_per_player,
            dim=self.card_dim,
            label="Card list",
            encode_item=lambda card: self._encode_card_features(
                card, is_mine, viewer, turn
            ),
        )

    def _encode_card_features(
        self, card: managym.Card, is_mine: float, viewer: int, turn: int
    ) -> np.ndarray:
        arr = np.zeros(self.card_dim, dtype=np.float32)
        i = 0
        zone_val = int(card.zone) & 0xFF
        if 0 <= zone_val < self.num_zones:
            arr[i + zone_val] = 1.0
        i += self.num_zones
        arr[i] = is_mine
        i += 1
        arr[i] = float(card.power) / 10.0
        i += 1
        arr[i] = float(card.toughness) / 10.0
        i += 1
        arr[i] = float(card.mana_cost.mana_value) / 10.0
        i += 1
        arr[i] = float(card.card_types.is_land)
        i += 1
        arr[i] = float(card.card_types.is_creature)
        i += 1
        arr[i] = float(card.card_types.is_artifact)
        i += 1
        arr[i] = float(card.card_types.is_enchantment)
        i += 1
        arr[i] = float(card.card_types.is_planeswalker)
        i += 1
        arr[i] = float(card.card_types.is_battle)
        i += 1
        arr[i] = float(card.keywords.flying)
        i += 1
        arr[i] = float(card.keywords.reach)
        i += 1
        arr[i] = float(card.keywords.haste)
        i += 1
        arr[i] = float(card.keywords.vigilance)
        i += 1
        arr[i] = float(card.keywords.trample)
        i += 1
        arr[i] = float(card.keywords.first_strike)
        i += 1
        arr[i] = float(card.keywords.double_strike)
        i += 1
        arr[i] = float(card.keywords.deathtouch)
        i += 1
        arr[i] = float(card.keywords.lifelink)
        i += 1
        arr[i] = float(card.keywords.defender)
        i += 1
        arr[i] = float(card.keywords.menace)
        i += 1
        arr[i] = float(card.keywords.flash)
        i += 1
        arr[i] = float(card.is_token)
        i += 1
        arr[i] = float(card.is_ally)
        i += 1
        arr[i] = float(card.is_lesson)
        i += 1
        arr[i] = float(card.ward_cost > 0)
        i += 1
        arr[i] = float(card.ward_cost) / 10.0
        i += 1
        arr[i] = float(card.kicker_cost > 0)
        i += 1
        arr[i] = float(card.kicker_cost) / 10.0
        i += 1
        arr[i] = float(card.keywords.hexproof)
        arr[38] = float(card.keywords.indestructible)
        arr[39] = float(card.is_legendary)
        if self.card_dim == CARD_DIM:
            permission = card.cast_permission
            if permission is not None:
                base = CARD_PERMISSION_INDEX
                arr[base] = 1.0
                arr[base + 1] = float(permission.player_id == viewer)
                if permission.cost is not None:
                    arr[base + 2] = 1.0
                    arr[base + 3] = float(permission.cost.mana_value) / 10.0
                arr[base + 4] = float(permission.this_turn)
                arr[base + 5] = float(permission.own_turn_only)
                # Not usable until a later turn than this one.
                arr[base + 6] = float(
                    permission.after_turn is not None and turn <= permission.after_turn
                )
                arr[base + 7] = float(permission.source_permanent_id is not None)
                arr[base + 8] = float(permission.flash)
                arr[base + 9] = float(permission.any_mana)
                arr[base + 10] = float(permission.waterbend)
                arr[base + 11] = float(permission.foretold)
            if card.x is not None:
                arr[CARD_X_INDEX] = 1.0
                arr[CARD_X_INDEX + 1] = float(card.x) / 10.0
        # Set validity flag (card exists)
        arr[-1] = 1.0
        return arr

    # -------------------------------------------------------------------------
    # Permanents (with validity mask support)
    # -------------------------------------------------------------------------
    def _encode_perms(
        self, perms: List[managym.Permanent], is_mine: float
    ) -> np.ndarray:
        return self._encode_objects(
            items=perms,
            max_items=self.perms_per_player,
            dim=self.permanent_dim,
            label="Permanent list",
            encode_item=lambda perm: self._encode_permanent_features(perm, is_mine),
        )

    def _encode_permanent_features(
        self, perm: managym.Permanent, is_mine: float
    ) -> np.ndarray:
        arr = np.zeros(self.permanent_dim, dtype=np.float32)
        arr[0] = is_mine
        arr[1] = float(perm.tapped)
        arr[2] = float(perm.damage) / 10.0
        arr[3] = float(perm.is_summoning_sick)
        arr[4] = float(perm.plus1_counters) / 10.0
        arr[5] = float(perm.cant_be_blocked_this_turn)
        arr[6] = float(perm.power) / 10.0
        arr[7] = float(perm.toughness) / 10.0
        arr[8] = float(perm.is_animated)
        arr[9] = float(perm.has_exile_link)
        # Effective keywords (printed + until-EOT grants) — mirrors the Rust
        # encoder's permanent keyword block exactly.
        arr[10] = float(perm.keywords.flying)
        arr[11] = float(perm.keywords.reach)
        arr[12] = float(perm.keywords.haste)
        arr[13] = float(perm.keywords.flash)
        arr[14] = float(perm.keywords.vigilance)
        arr[15] = float(perm.keywords.trample)
        arr[16] = float(perm.keywords.first_strike)
        arr[17] = float(perm.keywords.double_strike)
        arr[18] = float(perm.keywords.deathtouch)
        arr[19] = float(perm.keywords.lifelink)
        arr[20] = float(perm.keywords.defender)
        arr[21] = float(perm.keywords.menace)
        arr[22] = float(perm.keywords.hexproof)
        arr[23] = float(perm.exile_if_dies_this_turn)
        arr[25] = float(perm.keywords.indestructible)
        arr[26] = float("Avatar" in perm.temp_subtypes)
        arr[27] = float(perm.attacking)
        arr[PERMANENT_DEFINITION_INDEX] = perm.registry_key + 1
        if self.permanent_dim == PERMANENT_DIM:
            arr[PERMANENT_EXHAUSTED_INDEX] = float(perm.exhausted_abilities)
            arr[PERMANENT_ONCE_EACH_TURN_INDEX] = float(perm.once_each_turn_used)
            arr[PERMANENT_FIREBENDING_INDEX] = float(perm.granted_firebending) / 10.0
            # PERMANENT_ATTACHED_INDEX, PERMANENT_BLOCKING_INDEX and
            # PERMANENT_TIMESTAMP_INDEX are written once every object is indexed.
            arr[PERMANENT_GRANTED_LAND_INDEX] = float(perm.is_granted_land)
            arr[PERMANENT_ABILITIES_LOST_INDEX] = float(perm.abilities_lost)
        # Set validity flag (permanent exists)
        arr[-1] = 1.0
        return arr

    # -------------------------------------------------------------------------
    # Actions (with focus object indices and validity masking)
    # -------------------------------------------------------------------------
    def _encode_actions(
        self, obs: managym.Observation
    ) -> Tuple[np.ndarray, np.ndarray]:
        log = getLogger(__name__).getChild("encode_actions")
        arr = np.zeros((self.max_actions, self.action_dim), dtype=np.float32)
        action_focus = np.full(
            (self.max_actions, self.max_focus_objects), -1, dtype=np.int32
        )
        actions = obs.action_space.actions
        if len(actions) > self.max_actions:
            raise ValueError(
                f"observation capacity exceeded: actions {len(actions)} > {self.max_actions}"
            )
        for idx, action in enumerate(actions):
            arr[idx, -1] = 1.0
            action_type = int(action.action_type)
            if 0 <= action_type < self.num_actions:
                arr[idx, action_type] = 1.0

            arr[idx, ACTION_DECLARED_INDEX] = float(action.declared is True)
            for column, selector in (
                (ACTION_MODE_INDEX, action.mode_index),
                (ACTION_ABILITY_INDEX, action.ability_index),
                (ACTION_CONTINUATION_INDEX, action.continuation_index),
            ):
                if selector is not None:
                    if not 0 <= selector <= MAX_ACTION_SELECTOR:
                        raise ValueError(
                            "observation capacity exceeded: action selector"
                        )
                    arr[idx, column] = selector + 1

            if len(action.focus) > self.max_focus_objects:
                raise ValueError("observation capacity exceeded: action focus")
            for focus_index, fid in enumerate(action.focus):
                index = self.object_to_index.get(fid, -1)
                if index == -1:
                    log.warning(
                        f"Invalid focus object ID {fid} for action {action_type}."
                    )
                action_focus[idx, focus_index] = index

        return arr, action_focus

    def _encode_events(
        self, events: List[managym.EventData]
    ) -> Tuple[np.ndarray, np.ndarray]:
        log = getLogger(__name__).getChild("encode_events")
        arr = np.zeros((self.max_events, self.event_dim), dtype=np.float32)
        valid = np.zeros((self.max_events,), dtype=np.float32)
        # Values after ABILITY_TRIGGERED are viewer presentation facts. They
        # share the committed event window but never alter the policy tensor.
        learning_events = [
            event
            for event in events
            if int(event.event_type) <= int(EventTypeEnum.ABILITY_TRIGGERED)
        ]
        if len(learning_events) > self.max_events:
            log.warning(
                f"Event list truncated: {len(learning_events)} -> {self.max_events}"
            )
        ordered_events = learning_events[-self.max_events :]
        for i, event in enumerate(ordered_events):
            arr[i] = np.array(
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
            )
            valid[i] = 1.0
        return arr, valid

    def _encode_objects(
        self,
        items: List,
        max_items: int,
        dim: int,
        label: str,
        encode_item,
    ) -> np.ndarray:
        feat = np.zeros((max_items, dim), dtype=np.float32)
        if len(items) > max_items:
            raise ValueError(
                f"observation capacity exceeded: {label} {len(items)} > {max_items}"
            )

        for i, item in enumerate(items):
            feat[i] = encode_item(item)
            self.object_to_index[item.id] = self.current_object_index
            self.current_object_index += 1

        self.current_object_index += max_items - len(items)
        return feat


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

    def sample(self, mask=None) -> Dict[str, np.ndarray]:
        return {name: space.sample() for name, space in self.spaces.items()}

    def contains(self, x: Dict[str, np.ndarray]) -> bool:
        return all(
            name in x and x[name].shape == self.shapes[name]
            for name in self.spaces.keys()
        )

    def encode(self, obs: managym.Observation) -> Dict[str, np.ndarray]:
        return self.encoder.encode(obs)

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
