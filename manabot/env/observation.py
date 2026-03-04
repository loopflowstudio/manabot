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

from enum import IntEnum
from typing import Dict, ItemsView, KeysView, List, Tuple, ValuesView

import gymnasium as gym
import numpy as np

# Local imports
from manabot.infra.hypers import ObservationSpaceHypers
from manabot.infra.log import getLogger
import managym


# -----------------------------------------------------------------------------
# Game Enums - mirror managym for validation
# -----------------------------------------------------------------------------
class PhaseEnum(IntEnum):
    """Mirrors managym.PhaseEnum for validation."""

    BEGINNING = 0
    PRECOMBAT_MAIN = 1
    COMBAT = 2
    POSTCOMBAT_MAIN = 3
    ENDING = 4


class StepEnum(IntEnum):
    """Mirrors managym.StepEnum for validation."""

    BEGINNING_UNTAP = 0
    BEGINNING_UPKEEP = 1
    BEGINNING_DRAW = 2
    PRECOMBAT_MAIN_STEP = 3
    COMBAT_BEGIN = 4
    COMBAT_DECLARE_ATTACKERS = 5
    COMBAT_DECLARE_BLOCKERS = 6
    COMBAT_DAMAGE = 7
    COMBAT_END = 8
    POSTCOMBAT_MAIN_STEP = 9
    ENDING_END = 10
    ENDING_CLEANUP = 11


class ActionEnum(IntEnum):
    """Mirrors managym.ActionEnum for validation."""

    PRIORITY_PLAY_LAND = 0
    PRIORITY_CAST_SPELL = 1
    PRIORITY_PASS_PRIORITY = 2
    DECLARE_ATTACKER = 3
    DECLARE_BLOCKER = 4


class ZoneEnum(IntEnum):
    """Mirrors managym.ZoneEnum for validation."""

    LIBRARY = 0
    HAND = 1
    BATTLEFIELD = 2
    GRAVEYARD = 3
    STACK = 4
    EXILE = 5
    COMMAND = 6


# -----------------------------------------------------------------------------
# Observation Encoding
# -----------------------------------------------------------------------------
class ObservationEncoder:
    """
    Encodes each typed chunk (player, cards, permanents, etc.) into fixed-size arrays,
    along with corresponding ID and validity masks. These validity masks can be used
    later by the agent to mask out invalid slots in attention, action encoding, etc.
    """

    def __init__(self, hypers: ObservationSpaceHypers):
        self.hypers = hypers
        self.cards_per_player = hypers.max_cards_per_player
        self.perms_per_player = hypers.max_permanents_per_player
        self.max_actions = hypers.max_actions
        self.max_focus_objects = hypers.max_focus_objects

        self.num_phases = len(PhaseEnum.__members__)
        self.num_steps = len(StepEnum.__members__)
        self.num_zones = len(ZoneEnum.__members__)
        self.num_actions = len(ActionEnum.__members__)

        # Define dimensions for players, cards, permanents.
        self.player_dim = 2 + self.num_zones + self.num_phases + self.num_steps
        self.card_dim = (self.num_zones + 1 + 2 + 1 + 6) + 1
        self.permanent_dim = 4 + 1

        # Action space dimension: action type + validity bit.
        self.action_dim = self.num_actions + 1  # validity bit

        # This maps object IDS to their ultimate Game Object index.
        # This is used to map the focus object indices in the action space
        # to the actual object indices in the game object tensor.
        self.object_to_index: Dict[int, int] = {}
        self.current_object_index = 0

    @property
    def shapes(self) -> Dict[str, Tuple[int, ...]]:
        return {
            # Game objects
            "agent_player": (1, self.player_dim),
            "opponent_player": (1, self.player_dim),
            "agent_cards": (self.cards_per_player, self.card_dim),
            "opponent_cards": (self.cards_per_player, self.card_dim),
            "agent_permanents": (self.perms_per_player, self.permanent_dim),
            "opponent_permanents": (self.perms_per_player, self.permanent_dim),
            "actions": (self.max_actions, self.action_dim),
            "action_focus": (self.max_actions, self.max_focus_objects),
            "agent_player_valid": (1,),
            "opponent_player_valid": (1,),
            "agent_cards_valid": (self.cards_per_player,),
            "opponent_cards_valid": (self.cards_per_player,),
            "agent_permanents_valid": (self.perms_per_player,),
            "opponent_permanents_valid": (self.perms_per_player,),
            "actions_valid": (self.max_actions,),
        }

    def encode(self, obs: managym.Observation) -> Dict[str, np.ndarray]:
        out = {}

        ## NOTE: It is very important that we encode in this exact
        ## order.
        # 1. Agent player
        # 2. Opponent player
        # 3. Agent cards
        # 4. Opponent cards
        # 5. Agent permanents
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
            out[key] = self._encode_cards(cards, is_mine=is_mine)
        for key, perms, is_mine in (
            ("agent_permanents", obs.agent_permanents, 1.0),
            ("opponent_permanents", obs.opponent_permanents, 0.0),
        ):
            out[key] = self._encode_perms(perms, is_mine=is_mine)

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

        for key, value in out.items():
            log.debug(f"[SHAPES] {key}: {value.shape}")
        return out

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
        zone_counts = np.asarray(
            player.zone_counts[: self.num_zones], dtype=np.float32
        )
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

        self.object_to_index[player.id] = self.current_object_index
        self.current_object_index += 1
        return arr

    # -------------------------------------------------------------------------
    # Cards (with validity mask support)
    # -------------------------------------------------------------------------
    def _encode_cards(self, cards: List[managym.Card], is_mine: float) -> np.ndarray:
        log = getLogger(__name__).getChild("encode_cards")
        feat = np.zeros((self.cards_per_player, self.card_dim), dtype=np.float32)
        if len(cards) > self.cards_per_player:
            log.warning(
                f"Card list truncated: {len(cards)} -> {self.cards_per_player}"
            )
        ordered_cards = cards[: self.cards_per_player]
        for i, card in enumerate(ordered_cards):
            feat[i] = self._encode_card_features(card, is_mine)
            self.object_to_index[card.id] = self.current_object_index
            self.current_object_index += 1
        unused_slots = self.cards_per_player - len(ordered_cards)
        self.current_object_index += unused_slots
        return feat

    def _encode_card_features(self, card: managym.Card, is_mine: float) -> np.ndarray:
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
        # Set validity flag (card exists)
        arr[-1] = 1.0
        return arr

    # -------------------------------------------------------------------------
    # Permanents (with validity mask support)
    # -------------------------------------------------------------------------
    def _encode_perms(
        self, perms: List[managym.Permanent], is_mine: float
    ) -> np.ndarray:
        log = getLogger(__name__).getChild("encode_permanents")
        feat = np.zeros((self.perms_per_player, self.permanent_dim), dtype=np.float32)
        if len(perms) > self.perms_per_player:
            log.warning(
                f"Permanent list truncated: {len(perms)} -> {self.perms_per_player}"
            )
        ordered_perms = perms[: self.perms_per_player]
        for i, perm in enumerate(ordered_perms):
            feat[i] = self._encode_permanent_features(perm, is_mine)
            self.object_to_index[perm.id] = self.current_object_index
            self.current_object_index += 1
        unused_slots = self.perms_per_player - len(ordered_perms)
        self.current_object_index += unused_slots
        return feat

    def _encode_permanent_features(
        self, perm: managym.Permanent, is_mine: float
    ) -> np.ndarray:
        arr = np.zeros(self.permanent_dim, dtype=np.float32)
        arr[0] = is_mine
        arr[1] = float(perm.tapped)
        arr[2] = float(perm.damage) / 10.0
        arr[3] = float(perm.is_summoning_sick)
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
        valid_actions = np.zeros(self.max_actions, dtype=bool)
        actions = obs.action_space.actions
        # We'll accumulate the focus indices for each action here.
        action_focus_indices = []  # Expected shape: (max_actions, max_focus_objects)
        if len(actions) > self.max_actions:
            log.warning(f"Action space truncated: {len(actions)} -> {self.max_actions}")
        for idx, action in enumerate(actions[: self.max_actions]):
            valid_actions[idx] = True
            action_type = int(action.action_type)
            if 0 <= action_type < self.num_actions:
                arr[idx, action_type] = 1.0

            focus_ids = action.focus[: self.max_focus_objects]
            indices = []
            for fid in focus_ids:
                index = self.object_to_index.get(fid, -1)
                if index == -1:
                    log.warning(
                        f"Invalid focus object ID {fid} for action {action_type}."
                    )
                indices.append(index)
            # Pad the indices to ensure we have exactly self.max_focus_objects entries.
            if len(indices) < self.max_focus_objects:
                indices.extend([-1] * (self.max_focus_objects - len(indices)))
            action_focus_indices.append(indices)

        arr[..., -1] = valid_actions.astype(np.float32)
        # Pad action_focus_indices so shape is always
        # (max_actions, max_focus_objects).
        unused_slots = self.max_actions - len(action_focus_indices)
        if unused_slots > 0:
            action_focus_indices.extend([[-1] * self.max_focus_objects] * unused_slots)

        return arr, np.array(action_focus_indices, dtype=np.int32)


class ObservationSpace(gym.spaces.Space):
    """
    Gymnasium-compatible observation space optimized for attention processing.
    """

    def __init__(self, hypers: ObservationSpaceHypers = ObservationSpaceHypers()):
        super().__init__(shape=None, dtype=None)
        self.encoder = ObservationEncoder(hypers)
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
