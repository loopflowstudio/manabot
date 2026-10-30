use std::{collections::HashMap, fmt};

use crate::{
    agent::observation::{CardData, EventData, Observation, PermanentData, PlayerData, TurnData},
    flow::turn::{PhaseKind, StepKind},
    state::mana::SPEND_SUBTYPES,
};

/// Player and card widths before policy input v5, which appends cast
/// permissions, the hand-only lock and restricted mana by spending subtype.
pub const LEGACY_PLAYER_DIM: usize = 40;
/// v5 player columns: restricted mana each spending subtype may use
/// (`SPEND_SUBTYPES` × W/U/B/R/G/C), then the hand-only casting lock, the
/// hand-size freedom, the turn number, the seat (0 is on the play), kept
/// red mana, then turn history: attacked, permanents sacrificed, a land
/// entered and cards drawn this turn.
pub const PLAYER_RESTRICTED_INDEX: usize = LEGACY_PLAYER_DIM;
pub const PLAYER_HAND_ONLY_INDEX: usize = PLAYER_RESTRICTED_INDEX + 3 * 6;
pub const PLAYER_NO_MAXIMUM_HAND_SIZE_INDEX: usize = PLAYER_HAND_ONLY_INDEX + 1;
pub const PLAYER_TURN_INDEX: usize = PLAYER_NO_MAXIMUM_HAND_SIZE_INDEX + 1;
pub const PLAYER_SEAT_INDEX: usize = PLAYER_TURN_INDEX + 1;
pub const PLAYER_KEEPS_RED_MANA_INDEX: usize = PLAYER_SEAT_INDEX + 1;
pub const PLAYER_ATTACKED_INDEX: usize = PLAYER_KEEPS_RED_MANA_INDEX + 1;
pub const PLAYER_SACRIFICED_INDEX: usize = PLAYER_ATTACKED_INDEX + 1;
pub const PLAYER_LAND_ENTERED_INDEX: usize = PLAYER_SACRIFICED_INDEX + 1;
pub const PLAYER_CARDS_DRAWN_INDEX: usize = PLAYER_LAND_ENTERED_INDEX + 1;
pub const PLAYER_DIM: usize = PLAYER_CARDS_DRAWN_INDEX + 1;
pub const LEGACY_CARD_DIM: usize = 41;
/// v5 card columns replace v4's trailing validity flag, which moves to the
/// new last column: 12 cast-permission terms, then announced X. A face-down
/// card is foretold, so its permission's foretold term marks it.
pub const CARD_PERMISSION_INDEX: usize = LEGACY_CARD_DIM - 1;
pub const CARD_X_INDEX: usize = CARD_PERMISSION_INDEX + 12;
pub const CARD_DIM: usize = CARD_X_INDEX + 2 + 1;
pub const PERMANENT_DEFINITION_INDEX: usize = 24;
pub const LEGACY_PERMANENT_DIM: usize = 29;
// V5 permanent terms start where v4's validity flag was; validity stays last.
pub const PERMANENT_EXHAUSTED_INDEX: usize = LEGACY_PERMANENT_DIM - 1;
pub const PERMANENT_ONCE_EACH_TURN_INDEX: usize = PERMANENT_EXHAUSTED_INDEX + 1;
pub const PERMANENT_FIREBENDING_INDEX: usize = PERMANENT_ONCE_EACH_TURN_INDEX + 1;
/// Counters other than +1/+1 (CR 122.1): stun, fire, double strike.
pub const PERMANENT_STUN_INDEX: usize = PERMANENT_FIREBENDING_INDEX + 1;
pub const PERMANENT_FIRE_INDEX: usize = PERMANENT_STUN_INDEX + 1;
pub const PERMANENT_DOUBLE_STRIKE_COUNTERS_INDEX: usize = PERMANENT_FIRE_INDEX + 1;
/// Object index + 1 of the permanent this Aura or Equipment is attached to;
/// zero when unattached. A join like action focus, never a numeric feature.
pub const PERMANENT_ATTACHED_INDEX: usize = PERMANENT_DOUBLE_STRIKE_COUNTERS_INDEX + 1;
/// An effect sets the name and creature types (Honest Work). The values
/// follow from the setter, which the attachment join identifies.
pub const PERMANENT_CHARACTERISTICS_SET_INDEX: usize = PERMANENT_ATTACHED_INDEX + 1;
pub const PERMANENT_GRANTED_LAND_INDEX: usize = PERMANENT_CHARACTERISTICS_SET_INDEX + 1;
/// Temporary subtypes beyond v4's Avatar flag (column 26): one flag each,
/// then a count of any other, so none is silently dropped.
pub const PERMANENT_TEMP_SUBTYPES_INDEX: usize = PERMANENT_GRANTED_LAND_INDEX + 1;
pub const TEMP_SUBTYPES: [&str; 6] = [
    "Plains",
    "Island",
    "Swamp",
    "Mountain",
    "Forest",
    "Construct",
];
pub const PERMANENT_OTHER_TEMP_SUBTYPES_INDEX: usize =
    PERMANENT_TEMP_SUBTYPES_INDEX + TEMP_SUBTYPES.len();
/// Object index + 1 of the attacker this creature blocks; a join.
pub const PERMANENT_BLOCKING_INDEX: usize = PERMANENT_OTHER_TEMP_SUBTYPES_INDEX + 1;
pub const PERMANENT_ABILITIES_LOST_INDEX: usize = PERMANENT_BLOCKING_INDEX + 1;
/// Spent activation limits by ability: one flag per ability index below
/// `USED_ABILITY_FLAGS`, then a count of any later index.
pub const PERMANENT_USED_ABILITIES_INDEX: usize = PERMANENT_ABILITIES_LOST_INDEX + 1;
pub const USED_ABILITY_FLAGS: usize = 4;
pub const PERMANENT_OTHER_USED_ABILITIES_INDEX: usize =
    PERMANENT_USED_ABILITIES_INDEX + USED_ABILITY_FLAGS;
pub const PERMANENT_DEALT_DAMAGE_INDEX: usize = PERMANENT_OTHER_USED_ABILITIES_INDEX + 1;
pub const PERMANENT_DIM: usize = PERMANENT_DEALT_DAMAGE_INDEX + 2;
pub const ACTION_TYPE_DIM: usize = 17;
/// Declaration choice is separate from its family: attack and decline must
/// have distinct policy inputs even when they focus the same permanent.
pub const ACTION_DECLARED_INDEX: usize = ACTION_TYPE_DIM;
pub const ACTION_MODE_INDEX: usize = ACTION_DECLARED_INDEX + 1;
pub const ACTION_ABILITY_INDEX: usize = ACTION_MODE_INDEX + 1;
pub const ACTION_CONTINUATION_INDEX: usize = ACTION_ABILITY_INDEX + 1;
pub const ACTION_DIM: usize = ACTION_TYPE_DIM + 5;
/// Selectors encode index+1, reserving zero for absence. Keep f32 integers exact.
pub const MAX_ACTION_SELECTOR: usize = (1 << 24) - 1;
pub const EVENT_DIM: usize = 7;
pub const ZONE_DIM: usize = 7;
pub const PHASE_DIM: usize = 5;
pub const STEP_DIM: usize = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObservationEncoderConfig {
    pub policy_input_version: u8,
    pub max_cards_per_player: usize,
    pub max_permanents_per_player: usize,
    pub max_actions: usize,
    pub max_focus_objects: usize,
    pub max_events: usize,
    pub policy_history: bool,
}

impl Default for ObservationEncoderConfig {
    fn default() -> Self {
        Self {
            max_cards_per_player: 60,
            max_permanents_per_player: 40,
            max_actions: 64,
            max_focus_objects: 2,
            max_events: 32,
            policy_history: false,
            policy_input_version: 5,
        }
    }
}

impl ObservationEncoderConfig {
    pub fn relation_dim(&self) -> usize {
        match self.policy_input_version {
            5.. => super::continuation::RELATION_DIM_V5,
            4 => super::continuation::RELATION_DIM,
            _ => super::continuation::LEGACY_RELATION_DIM,
        }
    }

    pub fn player_dim(&self) -> usize {
        if self.policy_input_version >= 5 {
            PLAYER_DIM
        } else {
            LEGACY_PLAYER_DIM
        }
    }

    pub fn card_dim(&self) -> usize {
        if self.policy_input_version >= 5 {
            CARD_DIM
        } else {
            LEGACY_CARD_DIM
        }
    }

    pub fn permanent_dim(&self) -> usize {
        if self.policy_input_version >= 5 {
            PERMANENT_DIM
        } else {
            LEGACY_PERMANENT_DIM
        }
    }

    pub fn relations_len(&self) -> usize {
        if self.policy_input_version >= 2 {
            super::continuation::RELATION_ROWS * self.relation_dim()
        } else {
            0
        }
    }

    pub fn cards_len(&self) -> usize {
        self.max_cards_per_player * self.card_dim()
    }

    pub fn permanents_len(&self) -> usize {
        self.max_permanents_per_player * self.permanent_dim()
    }

    pub fn actions_len(&self) -> usize {
        self.max_actions * ACTION_DIM
    }

    pub fn action_focus_len(&self) -> usize {
        self.max_actions * self.max_focus_objects
    }

    pub fn event_dim(&self) -> usize {
        if self.policy_history {
            super::policy_history::HISTORY_DIM
        } else {
            EVENT_DIM
        }
    }

    pub fn events_len(&self) -> usize {
        self.max_events * self.event_dim()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EncodedObservation {
    pub relations: Vec<f32>,
    pub semantic_cards: Vec<f32>,
    pub known_hand: Vec<f32>,
    pub agent_player: Vec<f32>,
    pub opponent_player: Vec<f32>,
    pub agent_cards: Vec<f32>,
    pub opponent_cards: Vec<f32>,
    pub agent_permanents: Vec<f32>,
    pub opponent_permanents: Vec<f32>,
    pub actions: Vec<f32>,
    pub events: Vec<f32>,
    pub action_focus: Vec<i32>,
    pub agent_player_valid: Vec<f32>,
    pub opponent_player_valid: Vec<f32>,
    pub agent_cards_valid: Vec<f32>,
    pub opponent_cards_valid: Vec<f32>,
    pub agent_permanents_valid: Vec<f32>,
    pub opponent_permanents_valid: Vec<f32>,
    pub actions_valid: Vec<f32>,
    pub events_valid: Vec<f32>,
}

pub struct EncodedObservationMut<'a> {
    pub relations: &'a mut [f32],
    pub semantic_cards: &'a mut [f32],
    pub known_hand: &'a mut [f32],
    pub agent_player: &'a mut [f32],
    pub opponent_player: &'a mut [f32],
    pub agent_cards: &'a mut [f32],
    pub opponent_cards: &'a mut [f32],
    pub agent_permanents: &'a mut [f32],
    pub opponent_permanents: &'a mut [f32],
    pub actions: &'a mut [f32],
    pub events: &'a mut [f32],
    pub action_focus: &'a mut [i32],
    pub agent_player_valid: &'a mut [f32],
    pub opponent_player_valid: &'a mut [f32],
    pub agent_cards_valid: &'a mut [f32],
    pub opponent_cards_valid: &'a mut [f32],
    pub agent_permanents_valid: &'a mut [f32],
    pub opponent_permanents_valid: &'a mut [f32],
    pub actions_valid: &'a mut [f32],
    pub events_valid: &'a mut [f32],
}

impl EncodedObservation {
    /// Borrow the owned buffers for repeated encoding without allocating.
    pub fn buffers_mut(&mut self) -> EncodedObservationMut<'_> {
        EncodedObservationMut {
            relations: &mut self.relations,
            semantic_cards: &mut self.semantic_cards,
            known_hand: &mut self.known_hand,
            agent_player: &mut self.agent_player,
            opponent_player: &mut self.opponent_player,
            agent_cards: &mut self.agent_cards,
            opponent_cards: &mut self.opponent_cards,
            agent_permanents: &mut self.agent_permanents,
            opponent_permanents: &mut self.opponent_permanents,
            actions: &mut self.actions,
            events: &mut self.events,
            action_focus: &mut self.action_focus,
            agent_player_valid: &mut self.agent_player_valid,
            opponent_player_valid: &mut self.opponent_player_valid,
            agent_cards_valid: &mut self.agent_cards_valid,
            opponent_cards_valid: &mut self.opponent_cards_valid,
            agent_permanents_valid: &mut self.agent_permanents_valid,
            opponent_permanents_valid: &mut self.opponent_permanents_valid,
            actions_valid: &mut self.actions_valid,
            events_valid: &mut self.events_valid,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObservationEncodeError {
    Capacity {
        field: &'static str,
        capacity: usize,
        actual: usize,
    },
    InvalidLength {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for ObservationEncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Capacity {
                field,
                capacity,
                actual,
            } => write!(
                f,
                "observation capacity exceeded: {field} {actual} > {capacity}"
            ),
            Self::InvalidLength {
                field,
                expected,
                actual,
            } => write!(
                f,
                "invalid length for {field}: expected {expected}, got {actual}"
            ),
        }
    }
}

impl std::error::Error for ObservationEncodeError {}

pub fn encode(
    obs: &Observation,
    config: &ObservationEncoderConfig,
) -> Result<EncodedObservation, ObservationEncodeError> {
    let mut encoded = EncodedObservation {
        relations: vec![0.0; config.relations_len()],
        semantic_cards: vec![0.0; 2 * config.max_cards_per_player],
        known_hand: vec![0.0; 4 * config.max_cards_per_player],
        agent_player: vec![0.0; config.player_dim()],
        opponent_player: vec![0.0; config.player_dim()],
        agent_cards: vec![0.0; config.cards_len()],
        opponent_cards: vec![0.0; config.cards_len()],
        agent_permanents: vec![0.0; config.permanents_len()],
        opponent_permanents: vec![0.0; config.permanents_len()],
        actions: vec![0.0; config.actions_len()],
        events: vec![0.0; config.events_len()],
        action_focus: vec![-1; config.action_focus_len()],
        agent_player_valid: vec![0.0; 1],
        opponent_player_valid: vec![0.0; 1],
        agent_cards_valid: vec![0.0; config.max_cards_per_player],
        opponent_cards_valid: vec![0.0; config.max_cards_per_player],
        agent_permanents_valid: vec![0.0; config.max_permanents_per_player],
        opponent_permanents_valid: vec![0.0; config.max_permanents_per_player],
        actions_valid: vec![0.0; config.max_actions],
        events_valid: vec![0.0; config.max_events],
    };

    encode_into(obs, config, encoded.buffers_mut())?;
    Ok(encoded)
}

pub fn encode_into(
    obs: &Observation,
    config: &ObservationEncoderConfig,
    out: EncodedObservationMut<'_>,
) -> Result<(), ObservationEncodeError> {
    validate_buffer_len(
        "semantic_cards",
        out.semantic_cards.len(),
        2 * config.max_cards_per_player,
    )?;
    validate_buffer_len(
        "known_hand",
        out.known_hand.len(),
        4 * config.max_cards_per_player,
    )?;
    validate_buffer_len("agent_player", out.agent_player.len(), config.player_dim())?;
    validate_buffer_len(
        "opponent_player",
        out.opponent_player.len(),
        config.player_dim(),
    )?;
    validate_buffer_len("agent_cards", out.agent_cards.len(), config.cards_len())?;
    validate_buffer_len(
        "opponent_cards",
        out.opponent_cards.len(),
        config.cards_len(),
    )?;
    validate_buffer_len(
        "agent_permanents",
        out.agent_permanents.len(),
        config.permanents_len(),
    )?;
    validate_buffer_len(
        "opponent_permanents",
        out.opponent_permanents.len(),
        config.permanents_len(),
    )?;
    validate_buffer_len("actions", out.actions.len(), config.actions_len())?;
    validate_buffer_len(
        "action_focus",
        out.action_focus.len(),
        config.action_focus_len(),
    )?;
    validate_buffer_len("events", out.events.len(), config.events_len())?;
    validate_buffer_len("agent_player_valid", out.agent_player_valid.len(), 1)?;
    validate_buffer_len("opponent_player_valid", out.opponent_player_valid.len(), 1)?;
    validate_buffer_len(
        "agent_cards_valid",
        out.agent_cards_valid.len(),
        config.max_cards_per_player,
    )?;
    validate_buffer_len(
        "opponent_cards_valid",
        out.opponent_cards_valid.len(),
        config.max_cards_per_player,
    )?;
    validate_buffer_len(
        "agent_permanents_valid",
        out.agent_permanents_valid.len(),
        config.max_permanents_per_player,
    )?;
    validate_buffer_len(
        "opponent_permanents_valid",
        out.opponent_permanents_valid.len(),
        config.max_permanents_per_player,
    )?;
    validate_buffer_len("actions_valid", out.actions_valid.len(), config.max_actions)?;
    validate_buffer_len("events_valid", out.events_valid.len(), config.max_events)?;

    for (field, actual, capacity) in [
        (
            "agent_cards plus sideboard",
            obs.agent_cards.len() + obs.agent_sideboard.len(),
            config.max_cards_per_player,
        ),
        (
            "opponent_cards",
            obs.opponent_cards.len(),
            config.max_cards_per_player,
        ),
        (
            "agent_permanents",
            obs.agent_permanents.len(),
            config.max_permanents_per_player,
        ),
        (
            "opponent_permanents",
            obs.opponent_permanents.len(),
            config.max_permanents_per_player,
        ),
        (
            "actions",
            obs.action_space.actions.len(),
            config.max_actions,
        ),
    ] {
        if actual > capacity {
            return Err(ObservationEncodeError::Capacity {
                field,
                capacity,
                actual,
            });
        }
    }
    for action in &obs.action_space.actions {
        if let Some(index) = action.continuation_index {
            if config.policy_input_version < 2 || index >= obs.continuation.len() {
                return Err(ObservationEncodeError::Capacity {
                    field: "action continuation reference",
                    capacity: if config.policy_input_version >= 2 {
                        obs.continuation.len()
                    } else {
                        0
                    },
                    actual: index + 1,
                });
            }
        }
        for index in [action.mode_index, action.ability_index]
            .into_iter()
            .flatten()
        {
            if index > MAX_ACTION_SELECTOR {
                return Err(ObservationEncodeError::Capacity {
                    field: "action selector",
                    capacity: MAX_ACTION_SELECTOR,
                    actual: index,
                });
            }
        }
        if action.focus.len() > config.max_focus_objects {
            return Err(ObservationEncodeError::Capacity {
                field: "action focus",
                capacity: config.max_focus_objects,
                actual: action.focus.len(),
            });
        }
    }
    out.semantic_cards.fill(0.0);
    out.known_hand.fill(0.0);
    out.agent_player.fill(0.0);
    out.opponent_player.fill(0.0);
    out.agent_cards.fill(0.0);
    out.opponent_cards.fill(0.0);
    out.agent_permanents.fill(0.0);
    out.opponent_permanents.fill(0.0);
    out.actions.fill(0.0);
    out.events.fill(0.0);
    out.action_focus.fill(-1);
    out.agent_player_valid.fill(0.0);
    out.opponent_player_valid.fill(0.0);
    out.agent_cards_valid.fill(0.0);
    out.opponent_cards_valid.fill(0.0);
    out.agent_permanents_valid.fill(0.0);
    out.opponent_permanents_valid.fill(0.0);
    out.actions_valid.fill(0.0);
    out.events_valid.fill(0.0);

    for (seat, (cards, player)) in [
        (&obs.agent_cards, &obs.agent),
        (&obs.opponent_cards, &obs.opponent),
    ]
    .into_iter()
    .enumerate()
    {
        if player.known_hand.len() > config.max_cards_per_player {
            return Err(ObservationEncodeError::Capacity {
                field: "known hand",
                capacity: config.max_cards_per_player,
                actual: player.known_hand.len(),
            });
        }
        for (slot, card) in cards.iter().enumerate() {
            out.semantic_cards[seat * config.max_cards_per_player + slot] =
                (card.registry_key + 1) as f32;
        }
        for (slot, (definition, count)) in player.known_hand.iter().enumerate() {
            let offset = (seat * config.max_cards_per_player + slot) * 2;
            out.known_hand[offset] = (definition + 1) as f32;
            out.known_hand[offset + 1] = *count as f32;
        }
    }
    for (slot, card) in obs.agent_sideboard.iter().enumerate() {
        out.semantic_cards[obs.agent_cards.len() + slot] = (card.registry_key + 1) as f32;
    }

    let mut object_to_index: HashMap<i32, i32> = HashMap::with_capacity(
        2 + obs.agent_cards.len()
            + obs.opponent_cards.len()
            + obs.agent_sideboard.len()
            + obs.agent_permanents.len()
            + obs.opponent_permanents.len(),
    );
    let mut current_object_index: i32 = 0;

    encode_player_features(
        &obs.agent,
        &obs.turn,
        out.agent_player,
        &mut object_to_index,
        &mut current_object_index,
    );
    out.agent_player_valid[0] = 1.0;

    encode_player_features(
        &obs.opponent,
        &obs.turn,
        out.opponent_player,
        &mut object_to_index,
        &mut current_object_index,
    );
    out.opponent_player_valid[0] = 1.0;

    let card_context = CardContext {
        width: config.card_dim(),
        max_cards: config.max_cards_per_player,
        viewer_id: obs.agent.id,
        turn_number: obs.turn.turn_number,
    };
    encode_cards(
        &obs.agent_cards,
        1.0,
        &card_context,
        out.agent_cards,
        out.agent_cards_valid,
        &mut object_to_index,
        &mut current_object_index,
    );
    encode_cards(
        &obs.opponent_cards,
        0.0,
        &card_context,
        out.opponent_cards,
        out.opponent_cards_valid,
        &mut object_to_index,
        &mut current_object_index,
    );

    for (offset, card) in obs.agent_sideboard.iter().enumerate() {
        let index = obs.agent_cards.len() + offset;
        let width = config.card_dim();
        let row = &mut out.agent_cards[index * width..(index + 1) * width];
        row[7] = 1.0;
        row[37] = 1.0; // outside inventory; all seven zone bits remain zero
        row[width - 1] = 1.0;
        out.agent_cards_valid[index] = 1.0;
        object_to_index.insert(card.candidate_id, 2 + index as i32);
    }

    encode_permanents(
        &obs.agent_permanents,
        1.0,
        config,
        out.agent_permanents,
        out.agent_permanents_valid,
        &mut object_to_index,
        &mut current_object_index,
    );
    encode_permanents(
        &obs.opponent_permanents,
        0.0,
        config,
        out.opponent_permanents,
        out.opponent_permanents_valid,
        &mut object_to_index,
        &mut current_object_index,
    );
    if config.policy_input_version >= 5 {
        // Hosts may be listed after their attachments, so links follow indexing.
        let link = |object: Option<i32>| {
            object
                .and_then(|object| object_to_index.get(&object))
                .map_or(0.0, |index| (index + 1) as f32)
        };
        for (permanents, out) in [
            (&obs.agent_permanents, &mut *out.agent_permanents),
            (&obs.opponent_permanents, &mut *out.opponent_permanents),
        ] {
            for (row, permanent) in out
                .as_chunks_mut::<PERMANENT_DIM>()
                .0
                .iter_mut()
                .zip(permanents)
            {
                row[PERMANENT_ATTACHED_INDEX] = link(permanent.attached_to);
                row[PERMANENT_BLOCKING_INDEX] = link(permanent.blocking);
            }
        }
    }

    if config.policy_input_version >= 2 {
        super::continuation::encode_into(
            &obs.continuation,
            obs.agent.player_index as usize,
            &object_to_index,
            out.relations,
        )?;
    }

    encode_actions(
        obs,
        config.max_actions,
        config.max_focus_objects,
        out.actions,
        out.actions_valid,
        out.action_focus,
        &object_to_index,
    );
    if config.policy_history {
        for (index, row) in
            super::policy_history::encode(obs, config.max_events, config.max_cards_per_player)
                .iter()
                .enumerate()
        {
            out.events[index * config.event_dim()..(index + 1) * config.event_dim()]
                .copy_from_slice(row);
            out.events_valid[index] = 1.0;
        }
    } else {
        encode_events(
            &obs.recent_events,
            config.max_events,
            out.events,
            out.events_valid,
        );
    }

    Ok(())
}

fn validate_buffer_len(
    field: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), ObservationEncodeError> {
    if actual != expected {
        return Err(ObservationEncodeError::InvalidLength {
            field,
            expected,
            actual,
        });
    }
    Ok(())
}

fn encode_player_features(
    player: &PlayerData,
    turn: &TurnData,
    out: &mut [f32],
    object_to_index: &mut HashMap<i32, i32>,
    current_object_index: &mut i32,
) {
    out[0] = player.life as f32 / 20.0;
    out[1] = bool_to_f32(player.is_active);

    for i in 0..ZONE_DIM {
        out[2 + i] = player.zone_counts[i] as f32 / 60.0;
    }

    let phase_index = phase_index(turn.phase);
    if phase_index < PHASE_DIM {
        out[9 + phase_index] = 1.0;
    }

    let step_index = step_index(turn.step);
    if step_index < STEP_DIM {
        out[14 + step_index] = 1.0;
    }

    out[26] = player.graveyard_lessons as f32 / 10.0;
    out[27] = player.combat_mana as f32 / 10.0;
    for slot in 0..6 {
        out[28 + slot] = player.mana[slot] as f32 / 10.0;
        out[34 + slot] = player.lesson_mana[slot] as f32 / 10.0;
    }
    if out.len() == PLAYER_DIM {
        // Restricted mana each spending subtype may use, by color. A class
        // admitting several subtypes counts under each, so these columns
        // overlap and need not sum to the total above.
        for (index, subtype) in SPEND_SUBTYPES.iter().enumerate() {
            for class in player
                .restricted_mana
                .iter()
                .filter(|class| class.admits(subtype))
            {
                for slot in 0..6 {
                    out[PLAYER_RESTRICTED_INDEX + 6 * index + slot] +=
                        class.mana[slot] as f32 / 10.0;
                }
            }
        }
        out[PLAYER_HAND_ONLY_INDEX] = bool_to_f32(player.hand_only_until.is_some());
        out[PLAYER_NO_MAXIMUM_HAND_SIZE_INDEX] = bool_to_f32(player.no_maximum_hand_size);
        out[PLAYER_TURN_INDEX] = turn.turn_number as f32 / 20.0;
        // Seat 0 is on the play.
        out[PLAYER_SEAT_INDEX] = player.player_index as f32;
        out[PLAYER_KEEPS_RED_MANA_INDEX] = bool_to_f32(player.keeps_red_mana);
        let history = &player.turn_history;
        out[PLAYER_ATTACKED_INDEX] = bool_to_f32(history.attacked);
        out[PLAYER_SACRIFICED_INDEX] = history.permanents_sacrificed as f32 / 10.0;
        out[PLAYER_LAND_ENTERED_INDEX] = bool_to_f32(history.land_entered);
        out[PLAYER_CARDS_DRAWN_INDEX] = history.cards_drawn as f32 / 10.0;
    }

    object_to_index.insert(player.id, *current_object_index);
    *current_object_index += 1;
}

/// Facts every card row of one observation reads.
struct CardContext {
    width: usize,
    max_cards: usize,
    viewer_id: i32,
    turn_number: u32,
}

fn encode_cards(
    cards: &[CardData],
    is_mine: f32,
    context: &CardContext,
    out: &mut [f32],
    out_valid: &mut [f32],
    object_to_index: &mut HashMap<i32, i32>,
    current_object_index: &mut i32,
) {
    let ordered_cards = cards.iter().take(context.max_cards);
    let mut encoded_count = 0;

    for (i, card) in ordered_cards.enumerate() {
        let start = i * context.width;
        let end = start + context.width;
        encode_card_features(card, is_mine, context, &mut out[start..end]);
        out_valid[i] = 1.0;
        object_to_index.insert(card.id, *current_object_index);
        *current_object_index += 1;
        encoded_count += 1;
    }

    *current_object_index += (context.max_cards.saturating_sub(encoded_count)) as i32;
}

fn encode_card_features(card: &CardData, is_mine: f32, context: &CardContext, out: &mut [f32]) {
    let zone_index = (card.zone as i32 & 0xFF) as usize;
    if zone_index < ZONE_DIM {
        out[zone_index] = 1.0;
    }
    out[7] = is_mine;
    out[8] = card.power as f32 / 10.0;
    out[9] = card.toughness as f32 / 10.0;
    out[10] = card.mana_cost.mana_value as f32 / 10.0;
    out[11] = bool_to_f32(card.card_types.is_land);
    out[12] = bool_to_f32(card.card_types.is_creature);
    out[13] = bool_to_f32(card.card_types.is_artifact);
    out[14] = bool_to_f32(card.card_types.is_enchantment);
    out[15] = bool_to_f32(card.card_types.is_planeswalker);
    out[16] = bool_to_f32(card.card_types.is_battle);
    out[17] = bool_to_f32(card.keywords.flying);
    out[18] = bool_to_f32(card.keywords.reach);
    out[19] = bool_to_f32(card.keywords.haste);
    out[20] = bool_to_f32(card.keywords.vigilance);
    out[21] = bool_to_f32(card.keywords.trample);
    out[22] = bool_to_f32(card.keywords.first_strike);
    out[23] = bool_to_f32(card.keywords.double_strike);
    out[24] = bool_to_f32(card.keywords.deathtouch);
    out[25] = bool_to_f32(card.keywords.lifelink);
    out[26] = bool_to_f32(card.keywords.defender);
    out[27] = bool_to_f32(card.keywords.menace);
    out[28] = bool_to_f32(card.keywords.flash);
    out[29] = bool_to_f32(card.is_token);
    out[30] = bool_to_f32(card.is_ally);
    out[31] = bool_to_f32(card.is_lesson);
    out[32] = bool_to_f32(card.ward_cost > 0);
    out[33] = card.ward_cost as f32 / 10.0;
    out[34] = bool_to_f32(card.kicker_cost > 0);
    out[35] = card.kicker_cost as f32 / 10.0;
    out[36] = bool_to_f32(card.keywords.hexproof);
    out[38] = bool_to_f32(card.keywords.indestructible);
    out[39] = bool_to_f32(card.is_legendary);
    if context.width == CARD_DIM {
        if let Some(permission) = &card.cast_permission {
            let base = CARD_PERMISSION_INDEX;
            out[base] = 1.0;
            out[base + 1] = bool_to_f32(permission.player_id == context.viewer_id);
            if let Some(cost) = &permission.cost {
                out[base + 2] = 1.0;
                out[base + 3] = cost.mana_value as f32 / 10.0;
            }
            out[base + 4] = bool_to_f32(permission.this_turn);
            out[base + 5] = bool_to_f32(permission.own_turn_only);
            // Not yet usable: "cast it on a later turn" and this is that turn
            // or earlier.
            out[base + 6] = bool_to_f32(
                permission
                    .after_turn
                    .is_some_and(|turn| context.turn_number <= turn),
            );
            out[base + 7] = bool_to_f32(permission.source_permanent_id.is_some());
            out[base + 8] = bool_to_f32(permission.flash);
            out[base + 9] = bool_to_f32(permission.any_mana);
            out[base + 10] = bool_to_f32(permission.waterbend);
            out[base + 11] = bool_to_f32(permission.foretold);
        }
        if let Some(x) = card.x {
            out[CARD_X_INDEX] = 1.0;
            out[CARD_X_INDEX + 1] = x as f32 / 10.0;
        }
    }
    out[context.width - 1] = 1.0; // validity flag
}

fn encode_permanents(
    permanents: &[PermanentData],
    is_mine: f32,
    config: &ObservationEncoderConfig,
    out: &mut [f32],
    out_valid: &mut [f32],
    object_to_index: &mut HashMap<i32, i32>,
    current_object_index: &mut i32,
) {
    let max_permanents = config.max_permanents_per_player;
    let width = config.permanent_dim();
    let ordered_permanents = permanents.iter().take(max_permanents);
    let mut encoded_count = 0;

    for (i, permanent) in ordered_permanents.enumerate() {
        let start = i * width;
        let end = start + width;
        encode_permanent_features(permanent, is_mine, &mut out[start..end]);
        out_valid[i] = 1.0;
        object_to_index.insert(permanent.id, *current_object_index);
        *current_object_index += 1;
        encoded_count += 1;
    }

    *current_object_index += (max_permanents.saturating_sub(encoded_count)) as i32;
}

fn encode_permanent_features(permanent: &PermanentData, is_mine: f32, out: &mut [f32]) {
    out[0] = is_mine;
    out[1] = bool_to_f32(permanent.tapped);
    out[2] = permanent.damage as f32 / 10.0;
    out[3] = bool_to_f32(permanent.is_summoning_sick);
    out[4] = permanent.plus1_counters as f32 / 10.0;
    out[5] = bool_to_f32(permanent.cant_be_blocked_this_turn);
    out[6] = permanent.power as f32 / 10.0;
    out[7] = permanent.toughness as f32 / 10.0;
    out[8] = bool_to_f32(permanent.is_animated);
    out[9] = bool_to_f32(permanent.has_exile_link);
    // Effective keywords (printed + until-EOT grants) — agents see what the
    // permanent can do right now, not just what's printed on the card.
    out[10] = bool_to_f32(permanent.keywords.flying);
    out[11] = bool_to_f32(permanent.keywords.reach);
    out[12] = bool_to_f32(permanent.keywords.haste);
    out[13] = bool_to_f32(permanent.keywords.flash);
    out[14] = bool_to_f32(permanent.keywords.vigilance);
    out[15] = bool_to_f32(permanent.keywords.trample);
    out[16] = bool_to_f32(permanent.keywords.first_strike);
    out[17] = bool_to_f32(permanent.keywords.double_strike);
    out[18] = bool_to_f32(permanent.keywords.deathtouch);
    out[19] = bool_to_f32(permanent.keywords.lifelink);
    out[20] = bool_to_f32(permanent.keywords.defender);
    out[21] = bool_to_f32(permanent.keywords.menace);
    out[22] = bool_to_f32(permanent.keywords.hexproof);
    out[23] = bool_to_f32(permanent.exile_if_dies_this_turn);
    out[25] = bool_to_f32(permanent.keywords.indestructible);
    out[26] = bool_to_f32(permanent.temp_subtypes.iter().any(|s| s == "Avatar"));
    out[27] = bool_to_f32(permanent.attacking);
    out[PERMANENT_DEFINITION_INDEX] = (permanent.registry_key + 1) as f32;
    if out.len() == PERMANENT_DIM {
        // Ability-use limits and granted firebending.
        out[PERMANENT_EXHAUSTED_INDEX] = permanent.exhausted_abilities as f32;
        out[PERMANENT_ONCE_EACH_TURN_INDEX] = permanent.once_each_turn_used as f32;
        out[PERMANENT_FIREBENDING_INDEX] = permanent.granted_firebending as f32 / 10.0;
        out[PERMANENT_STUN_INDEX] = permanent.stun_counters as f32 / 10.0;
        out[PERMANENT_FIRE_INDEX] = permanent.fire_counters as f32 / 10.0;
        out[PERMANENT_DOUBLE_STRIKE_COUNTERS_INDEX] =
            permanent.double_strike_counters as f32 / 10.0;
        // PERMANENT_ATTACHED_INDEX is written once every object is indexed.
        out[PERMANENT_CHARACTERISTICS_SET_INDEX] =
            bool_to_f32(permanent.name.is_some() || permanent.creature_types.is_some());
        out[PERMANENT_GRANTED_LAND_INDEX] = bool_to_f32(permanent.is_granted_land);
        for (offset, subtype) in TEMP_SUBTYPES.iter().enumerate() {
            out[PERMANENT_TEMP_SUBTYPES_INDEX + offset] =
                bool_to_f32(permanent.temp_subtypes.iter().any(|s| s == subtype));
        }
        out[PERMANENT_OTHER_TEMP_SUBTYPES_INDEX] = permanent
            .temp_subtypes
            .iter()
            .filter(|s| *s != "Avatar" && !TEMP_SUBTYPES.contains(&s.as_str()))
            .count() as f32;
        // PERMANENT_BLOCKING_INDEX is written once every object is indexed.
        out[PERMANENT_ABILITIES_LOST_INDEX] = bool_to_f32(permanent.abilities_lost);
        for index in &permanent.used_abilities {
            if *index < USED_ABILITY_FLAGS {
                out[PERMANENT_USED_ABILITIES_INDEX + index] = 1.0;
            } else {
                out[PERMANENT_OTHER_USED_ABILITIES_INDEX] += 1.0;
            }
        }
        out[PERMANENT_DEALT_DAMAGE_INDEX] = bool_to_f32(permanent.dealt_damage_this_turn);
    }
    out[out.len() - 1] = 1.0;
}

fn encode_actions(
    obs: &Observation,
    max_actions: usize,
    max_focus_objects: usize,
    out_actions: &mut [f32],
    out_valid: &mut [f32],
    out_focus: &mut [i32],
    object_to_index: &HashMap<i32, i32>,
) {
    for (action_index, action) in obs
        .action_space
        .actions
        .iter()
        .take(max_actions)
        .enumerate()
    {
        let action_start = action_index * ACTION_DIM;
        let action_end = action_start + ACTION_DIM;
        let row = &mut out_actions[action_start..action_end];

        let action_type_index = action.action_type as usize;
        if action_type_index < ACTION_TYPE_DIM {
            row[action_type_index] = 1.0;
        }
        row[ACTION_DECLARED_INDEX] = bool_to_f32(action.declared.unwrap_or(false));
        row[ACTION_MODE_INDEX] = action.mode_index.map_or(0.0, |index| (index + 1) as f32);
        row[ACTION_ABILITY_INDEX] = action.ability_index.map_or(0.0, |index| (index + 1) as f32);
        row[ACTION_CONTINUATION_INDEX] = action
            .continuation_index
            .map_or(0.0, |index| (index + 1) as f32);
        row[ACTION_DIM - 1] = 1.0;
        out_valid[action_index] = 1.0;

        let focus_start = action_index * max_focus_objects;
        for (focus_index, focus_id) in action.focus.iter().take(max_focus_objects).enumerate() {
            out_focus[focus_start + focus_index] =
                object_to_index.get(focus_id).copied().unwrap_or(-1);
        }
    }
}

fn encode_events(events: &[EventData], max_events: usize, out: &mut [f32], out_valid: &mut [f32]) {
    // Presentation-only domain facts are additive to `recent_events`, but the
    // fixed policy ABI must remain byte-for-byte compatible.
    let learning_events = events
        .iter()
        .filter(|event| event.event_type <= 6)
        .collect::<Vec<_>>();
    let start_index = learning_events.len().saturating_sub(max_events);
    for (event_index, event) in learning_events.into_iter().skip(start_index).enumerate() {
        let row_start = event_index * EVENT_DIM;
        let row = &mut out[row_start..row_start + EVENT_DIM];
        row[0] = event.event_type as f32;
        row[1] = event.source_kind as f32;
        row[2] = event.source_id as f32;
        row[3] = event.target_kind as f32;
        row[4] = event.target_id as f32;
        row[5] = event.amount as f32;
        row[6] = event.controller_id as f32;
        out_valid[event_index] = 1.0;
    }
}

fn phase_index(phase: PhaseKind) -> usize {
    match phase {
        PhaseKind::Beginning => 0,
        PhaseKind::PrecombatMain => 1,
        PhaseKind::Combat => 2,
        PhaseKind::PostcombatMain => 3,
        PhaseKind::Ending => 4,
    }
}

fn step_index(step: StepKind) -> usize {
    match step {
        StepKind::Untap => 0,
        StepKind::Upkeep => 1,
        StepKind::Draw => 2,
        StepKind::Main => 3,
        StepKind::BeginningOfCombat => 4,
        StepKind::DeclareAttackers => 5,
        StepKind::DeclareBlockers => 6,
        StepKind::CombatDamage => 7,
        StepKind::EndOfCombat => 8,
        StepKind::PostcombatMain => 9,
        StepKind::End => 10,
        StepKind::Cleanup => 11,
    }
}

fn bool_to_f32(value: bool) -> f32 {
    if value {
        1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        agent::{
            action::{ActionSpaceKind, ActionType},
            observation::{
                ActionOption, ActionSpaceData, CardData, CardTypeData, EventData, KeywordData,
                Observation, PermanentData, PlayerData, TurnData,
            },
        },
        flow::turn::{PhaseKind, StepKind},
        state::{mana::ManaCost, zone::ZoneType},
    };

    use super::{
        encode, encode_into, EncodedObservationMut, ObservationEncodeError,
        ObservationEncoderConfig, ACTION_DIM, EVENT_DIM, LEGACY_CARD_DIM, LEGACY_PERMANENT_DIM,
        LEGACY_PLAYER_DIM, ZONE_DIM,
    };

    #[test]
    fn selector_encoding_is_exact_and_rejects_overflow() {
        use super::{ACTION_ABILITY_INDEX, ACTION_MODE_INDEX, MAX_ACTION_SELECTOR};
        let mut obs = sample_observation();
        obs.action_space.actions[0].mode_index = Some(0);
        obs.action_space.actions[1].ability_index = Some(MAX_ACTION_SELECTOR);
        let config = ObservationEncoderConfig::default();
        let encoded = encode(&obs, &config).unwrap();
        assert_eq!(encoded.actions[ACTION_MODE_INDEX], 1.0);
        assert_eq!(encoded.actions[ACTION_ABILITY_INDEX], 0.0);
        assert_eq!(
            encoded.actions[ACTION_DIM + ACTION_ABILITY_INDEX],
            16777216.0
        );
        for mode in [false, true] {
            let mut obs = sample_observation();
            let action = &mut obs.action_space.actions[0];
            if mode {
                action.mode_index = Some(MAX_ACTION_SELECTOR + 1);
            } else {
                action.ability_index = Some(MAX_ACTION_SELECTOR + 1);
            }
            assert!(matches!(
                encode(&obs, &config),
                Err(ObservationEncodeError::Capacity { .. })
            ));
        }
    }

    fn sample_observation() -> Observation {
        Observation {
            continuation: Vec::new(),
            game_over: false,
            won: false,
            turn: TurnData {
                turn_number: 1,
                phase: PhaseKind::Combat,
                step: StepKind::DeclareAttackers,
                active_player_id: 10,
                agent_player_id: 10,
            },
            action_space: ActionSpaceData {
                action_space_type: ActionSpaceKind::Priority,
                actions: vec![
                    ActionOption {
                        action_type: ActionType::PriorityCastSpell,
                        focus: vec![111, 333],
                        declared: None,
                        description: None,
                        mode_index: None,
                        ability_index: None,
                        continuation_index: None,
                    },
                    ActionOption {
                        action_type: ActionType::ChooseTarget,
                        focus: vec![444],
                        declared: None,
                        description: None,
                        mode_index: None,
                        ability_index: None,
                        continuation_index: None,
                    },
                    ActionOption {
                        action_type: ActionType::DeclareBlocker,
                        focus: vec![999],
                        declared: Some(true),
                        description: None,
                        mode_index: None,
                        ability_index: None,
                        continuation_index: None,
                    },
                ],
                focus: Vec::new(),
            },
            agent: PlayerData {
                player_index: 0,
                id: 10,
                is_agent: true,
                is_active: true,
                life: 20,
                zone_counts: [40, 2, 1, 0, 0, 0, 0],
                graveyard_lessons: 0,
                known_hand: Default::default(),
                sideboard_counts: Default::default(),
                remaining_sideboard_counts: Default::default(),
                keeps_red_mana: false,
                turn_history: Default::default(),
                combat_mana: 0,
                mana: [0; 6],
                lesson_mana: [0; 6],
                restricted_mana: Vec::new(),
                hand_only_until: None,
                no_maximum_hand_size: false,
            },
            agent_sideboard: Vec::new(),
            agent_cards: vec![
                make_card(111, ZoneType::Hand, true, 2, 2, 1),
                make_card(112, ZoneType::Battlefield, true, 3, 3, 2),
            ],
            stack_objects: vec![],
            agent_permanents: vec![make_permanent(333, true)],
            opponent: PlayerData {
                player_index: 1,
                id: 20,
                is_agent: false,
                is_active: false,
                life: 18,
                zone_counts: [39, 3, 1, 0, 0, 0, 0],
                graveyard_lessons: 0,
                known_hand: Default::default(),
                sideboard_counts: Default::default(),
                remaining_sideboard_counts: Default::default(),
                keeps_red_mana: false,
                turn_history: Default::default(),
                combat_mana: 0,
                mana: [0; 6],
                lesson_mana: [0; 6],
                restricted_mana: Vec::new(),
                hand_only_until: None,
                no_maximum_hand_size: false,
            },
            opponent_cards: vec![make_card(221, ZoneType::Hand, false, 1, 1, 1)],
            opponent_permanents: vec![make_permanent(444, false)],
            policy_history: Default::default(),
            recent_events: vec![
                EventData {
                    event_type: 3,
                    source_kind: 1,
                    source_id: 112,
                    target_kind: 0,
                    target_id: -1,
                    amount: 0,
                    controller_id: 10,
                    from_zone: -1,
                    to_zone: -1,
                    source_incarnation: -1,
                    target_incarnation: -1,
                },
                EventData {
                    event_type: 2,
                    source_kind: 3,
                    source_id: 20,
                    target_kind: 3,
                    target_id: 20,
                    amount: -3,
                    controller_id: -1,
                    from_zone: -1,
                    to_zone: -1,
                    source_incarnation: -1,
                    target_incarnation: -1,
                },
                EventData {
                    event_type: 11,
                    source_kind: 3,
                    source_id: 10,
                    target_kind: 0,
                    target_id: -1,
                    amount: 2,
                    controller_id: 10,
                    from_zone: -1,
                    to_zone: -1,
                    source_incarnation: -1,
                    target_incarnation: -1,
                },
            ],
        }
    }

    #[test]
    fn attack_and_decline_have_distinct_policy_features() {
        let mut observation = sample_observation();
        observation.action_space.action_space_type = ActionSpaceKind::DeclareAttacker;
        observation.action_space.actions = [true, false]
            .into_iter()
            .map(|declared| ActionOption {
                action_type: ActionType::DeclareAttacker,
                focus: vec![333],
                declared: Some(declared),
                description: None,
                mode_index: None,
                ability_index: None,
                continuation_index: None,
            })
            .collect();
        let config = ObservationEncoderConfig::default();
        let encoded = encode(&observation, &config).expect("attack decision encodes");
        assert_ne!(
            &encoded.actions[..ACTION_DIM - 1],
            &encoded.actions[ACTION_DIM..2 * ACTION_DIM - 1],
            "the policy must be able to prefer attacking or declining"
        );
        assert_eq!(
            &encoded.action_focus[..config.max_focus_objects],
            &encoded.action_focus[config.max_focus_objects..2 * config.max_focus_objects]
        );
        assert_eq!(&encoded.actions_valid[..2], &[1.0, 1.0]);
    }

    fn make_card(
        id: i32,
        zone: ZoneType,
        mine: bool,
        power: i32,
        toughness: i32,
        mana_value: u8,
    ) -> CardData {
        CardData {
            zone,
            owner_id: if mine { 10 } else { 20 },
            id,
            registry_key: id,
            name: format!("Card {id}"),
            text_box: String::new(),
            power,
            toughness,
            is_token: false,
            is_legendary: false,
            is_ally: false,
            is_lesson: false,
            ward_cost: 0,
            kicker_cost: 0,
            face_down: false,
            cast_permission: None,
            x: None,
            card_types: CardTypeData {
                is_castable: true,
                is_permanent: true,
                is_non_land_permanent: true,
                is_non_creature_permanent: false,
                is_spell: true,
                is_creature: true,
                is_land: false,
                is_planeswalker: false,
                is_enchantment: false,
                is_artifact: false,
                is_kindred: false,
                is_battle: false,
            },
            keywords: KeywordData::default(),
            mana_cost: ManaCost {
                purpose: Default::default(),
                cost: [0; 7],
                mana_value,
                hybrid: Vec::new(),
                x: 0,
            },
        }
    }

    fn make_permanent(id: i32, mine: bool) -> PermanentData {
        PermanentData {
            id,
            registry_key: 0,
            controller_id: if mine { 10 } else { 20 },
            tapped: false,
            damage: 0,
            is_summoning_sick: false,
            plus1_counters: 0,
            cant_be_blocked_this_turn: false,
            power: 2,
            toughness: 2,
            is_animated: false,
            has_exile_link: false,
            exile_if_dies_this_turn: false,
            ..Default::default()
        }
    }

    #[test]
    fn encode_assigns_deterministic_object_indices_and_focus() {
        let obs = sample_observation();
        let config = ObservationEncoderConfig {
            max_cards_per_player: 3,
            max_permanents_per_player: 2,
            max_actions: 3,
            max_focus_objects: 2,
            max_events: 2,
            policy_history: false,
            policy_input_version: 4,
        };

        let encoded = encode(&obs, &config).unwrap();

        // Agent and opponent player vectors are populated with turn one-hot slices.
        assert_eq!(encoded.agent_player[9 + 2], 1.0);
        assert_eq!(encoded.agent_player[14 + 5], 1.0);
        assert_eq!(encoded.opponent_player[9 + 2], 1.0);

        // Validity flags for populated card/permanent slots.
        assert_eq!(encoded.agent_cards[LEGACY_CARD_DIM - 1], 1.0);
        assert_eq!(encoded.agent_cards[LEGACY_CARD_DIM * 2 - 1], 1.0);
        assert_eq!(encoded.agent_cards[LEGACY_CARD_DIM * 3 - 1], 0.0);
        assert_eq!(encoded.agent_permanents[LEGACY_PERMANENT_DIM - 1], 1.0);
        assert_eq!(
            encoded.agent_permanents[LEGACY_PERMANENT_DIM - 1 + LEGACY_PERMANENT_DIM],
            0.0
        );

        // Expected object indices with padding-aware ordering:
        // players: 0,1; agent cards: [2,3,4]; opponent cards: [5,6,7];
        // agent permanents: [8,9]; opponent permanents: [10,11]
        assert_eq!(encoded.action_focus[0], 2);
        assert_eq!(encoded.action_focus[1], 8);
        assert_eq!(encoded.action_focus[2], 10);
        assert_eq!(encoded.action_focus[3], -1);
        assert_eq!(encoded.action_focus[4], -1);
        assert_eq!(encoded.action_focus[5], -1);

        // Unknown focus IDs map to -1.
        assert_eq!(encoded.action_focus[4], -1);

        // Validity vectors match populated slots.
        assert_eq!(encoded.agent_player_valid, vec![1.0]);
        assert_eq!(encoded.opponent_player_valid, vec![1.0]);
        assert_eq!(encoded.agent_cards_valid, vec![1.0, 1.0, 0.0]);
        assert_eq!(encoded.opponent_cards_valid, vec![1.0, 0.0, 0.0]);
        assert_eq!(encoded.agent_permanents_valid, vec![1.0, 0.0]);
        assert_eq!(encoded.opponent_permanents_valid, vec![1.0, 0.0]);
        assert_eq!(encoded.actions_valid, vec![1.0, 1.0, 1.0]);
        assert_eq!(encoded.events_valid, vec![1.0, 1.0]);
        assert_eq!(encoded.events[0], 3.0);
        assert_eq!(encoded.events[EVENT_DIM], 2.0);
    }

    #[test]
    fn learn_outside_focus_is_complete_and_capacity_errors_are_explicit() {
        use crate::agent::observation::SideboardCardData;
        let mut obs = sample_observation();
        obs.agent_sideboard = (0..40)
            .map(|i| SideboardCardData {
                candidate_id: 1000 + i,
                owner_id: obs.agent.id,
                registry_key: 7,
                name: "Accumulate Wisdom".into(),
                text_box: String::new(),
            })
            .collect();
        obs.action_space.actions = obs
            .agent_sideboard
            .iter()
            .map(|card| ActionOption {
                action_type: ActionType::LearnTakeLesson,
                focus: vec![card.candidate_id],
                declared: None,
                description: None,
                mode_index: None,
                ability_index: None,
                continuation_index: None,
            })
            .collect();
        let config = ObservationEncoderConfig::default();
        let encoded = encode(&obs, &config).unwrap();
        let last = 39;
        assert_eq!(encoded.actions[last * ACTION_DIM + 14], 1.0);
        assert_eq!(
            encoded.action_focus[last * config.max_focus_objects],
            2 + obs.agent_cards.len() as i32 + last as i32
        );
        let row = (obs.agent_cards.len() + last) * config.card_dim();
        assert_eq!(encoded.agent_cards[row..row + ZONE_DIM], [0.0; ZONE_DIM]);
        assert_eq!(encoded.agent_cards[row + 37], 1.0);
        assert_eq!(encoded.agent_cards[row + config.card_dim() - 1], 1.0);
        for undersized in [
            ObservationEncoderConfig {
                max_actions: 32,
                ..config
            },
            ObservationEncoderConfig {
                max_cards_per_player: 40,
                ..config
            },
            ObservationEncoderConfig {
                max_focus_objects: 0,
                ..config
            },
        ] {
            assert!(matches!(
                encode(&obs, &undersized),
                Err(ObservationEncodeError::Capacity { .. })
            ));
        }
    }

    #[test]
    fn encode_into_validates_output_lengths() {
        let obs = sample_observation();
        let config = ObservationEncoderConfig {
            max_cards_per_player: 2,
            max_permanents_per_player: 1,
            max_actions: 2,
            max_focus_objects: 2,
            max_events: 1,
            policy_history: false,
            policy_input_version: 4,
        };

        let mut agent_player = vec![0.0; 1];
        let mut semantic_cards = vec![0.0; 4];
        let mut known_hand = vec![0.0; 8];
        let mut opponent_player = vec![0.0; LEGACY_PLAYER_DIM];
        let mut agent_cards = vec![0.0; 2 * LEGACY_CARD_DIM];
        let mut opponent_cards = vec![0.0; 2 * LEGACY_CARD_DIM];
        let mut agent_permanents = vec![0.0; LEGACY_PERMANENT_DIM];
        let mut opponent_permanents = vec![0.0; LEGACY_PERMANENT_DIM];
        let mut actions = vec![0.0; 2 * ACTION_DIM];
        let mut events = vec![0.0; EVENT_DIM];
        let mut action_focus = vec![-1; 2 * 2];
        let mut agent_player_valid = vec![0.0; 1];
        let mut opponent_player_valid = vec![0.0; 1];
        let mut agent_cards_valid = vec![0.0; 2];
        let mut opponent_cards_valid = vec![0.0; 2];
        let mut agent_permanents_valid = vec![0.0; 1];
        let mut opponent_permanents_valid = vec![0.0; 1];
        let mut actions_valid = vec![0.0; 2];
        let mut events_valid = vec![0.0; 1];

        let out = EncodedObservationMut {
            relations: &mut [],
            semantic_cards: &mut semantic_cards,
            known_hand: &mut known_hand,
            agent_player: &mut agent_player,
            opponent_player: &mut opponent_player,
            agent_cards: &mut agent_cards,
            opponent_cards: &mut opponent_cards,
            agent_permanents: &mut agent_permanents,
            opponent_permanents: &mut opponent_permanents,
            actions: &mut actions,
            events: &mut events,
            action_focus: &mut action_focus,
            agent_player_valid: &mut agent_player_valid,
            opponent_player_valid: &mut opponent_player_valid,
            agent_cards_valid: &mut agent_cards_valid,
            opponent_cards_valid: &mut opponent_cards_valid,
            agent_permanents_valid: &mut agent_permanents_valid,
            opponent_permanents_valid: &mut opponent_permanents_valid,
            actions_valid: &mut actions_valid,
            events_valid: &mut events_valid,
        };

        let err = encode_into(&obs, &config, out).expect_err("invalid length must fail");
        assert_eq!(
            err.to_string(),
            format!("invalid length for agent_player: expected {LEGACY_PLAYER_DIM}, got 1")
        );
    }
}
