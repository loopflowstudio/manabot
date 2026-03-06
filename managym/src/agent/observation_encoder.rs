use std::{collections::HashMap, fmt};

use crate::{
    agent::observation::{CardData, Observation, PermanentData, PlayerData, TurnData},
    flow::turn::{PhaseKind, StepKind},
};

pub const PLAYER_DIM: usize = 26;
pub const CARD_DIM: usize = 18;
pub const PERMANENT_DIM: usize = 5;
pub const ACTION_TYPE_DIM: usize = 6;
pub const ACTION_DIM: usize = ACTION_TYPE_DIM + 1;
pub const ZONE_DIM: usize = 7;
pub const PHASE_DIM: usize = 5;
pub const STEP_DIM: usize = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObservationEncoderConfig {
    pub max_cards_per_player: usize,
    pub max_permanents_per_player: usize,
    pub max_actions: usize,
    pub max_focus_objects: usize,
}

impl Default for ObservationEncoderConfig {
    fn default() -> Self {
        Self {
            max_cards_per_player: 60,
            max_permanents_per_player: 30,
            max_actions: 20,
            max_focus_objects: 2,
        }
    }
}

impl ObservationEncoderConfig {
    pub fn cards_len(&self) -> usize {
        self.max_cards_per_player * CARD_DIM
    }

    pub fn permanents_len(&self) -> usize {
        self.max_permanents_per_player * PERMANENT_DIM
    }

    pub fn actions_len(&self) -> usize {
        self.max_actions * ACTION_DIM
    }

    pub fn action_focus_len(&self) -> usize {
        self.max_actions * self.max_focus_objects
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EncodedObservation {
    pub agent_player: Vec<f32>,
    pub opponent_player: Vec<f32>,
    pub agent_cards: Vec<f32>,
    pub opponent_cards: Vec<f32>,
    pub agent_permanents: Vec<f32>,
    pub opponent_permanents: Vec<f32>,
    pub actions: Vec<f32>,
    pub action_focus: Vec<i32>,
    pub agent_player_valid: Vec<f32>,
    pub opponent_player_valid: Vec<f32>,
    pub agent_cards_valid: Vec<f32>,
    pub opponent_cards_valid: Vec<f32>,
    pub agent_permanents_valid: Vec<f32>,
    pub opponent_permanents_valid: Vec<f32>,
    pub actions_valid: Vec<f32>,
}

pub struct EncodedObservationMut<'a> {
    pub agent_player: &'a mut [f32],
    pub opponent_player: &'a mut [f32],
    pub agent_cards: &'a mut [f32],
    pub opponent_cards: &'a mut [f32],
    pub agent_permanents: &'a mut [f32],
    pub opponent_permanents: &'a mut [f32],
    pub actions: &'a mut [f32],
    pub action_focus: &'a mut [i32],
    pub agent_player_valid: &'a mut [f32],
    pub opponent_player_valid: &'a mut [f32],
    pub agent_cards_valid: &'a mut [f32],
    pub opponent_cards_valid: &'a mut [f32],
    pub agent_permanents_valid: &'a mut [f32],
    pub opponent_permanents_valid: &'a mut [f32],
    pub actions_valid: &'a mut [f32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObservationEncodeError {
    InvalidLength {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for ObservationEncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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

pub fn encode(obs: &Observation, config: &ObservationEncoderConfig) -> EncodedObservation {
    let mut encoded = EncodedObservation {
        agent_player: vec![0.0; PLAYER_DIM],
        opponent_player: vec![0.0; PLAYER_DIM],
        agent_cards: vec![0.0; config.cards_len()],
        opponent_cards: vec![0.0; config.cards_len()],
        agent_permanents: vec![0.0; config.permanents_len()],
        opponent_permanents: vec![0.0; config.permanents_len()],
        actions: vec![0.0; config.actions_len()],
        action_focus: vec![-1; config.action_focus_len()],
        agent_player_valid: vec![0.0; 1],
        opponent_player_valid: vec![0.0; 1],
        agent_cards_valid: vec![0.0; config.max_cards_per_player],
        opponent_cards_valid: vec![0.0; config.max_cards_per_player],
        agent_permanents_valid: vec![0.0; config.max_permanents_per_player],
        opponent_permanents_valid: vec![0.0; config.max_permanents_per_player],
        actions_valid: vec![0.0; config.max_actions],
    };

    let out = EncodedObservationMut {
        agent_player: &mut encoded.agent_player,
        opponent_player: &mut encoded.opponent_player,
        agent_cards: &mut encoded.agent_cards,
        opponent_cards: &mut encoded.opponent_cards,
        agent_permanents: &mut encoded.agent_permanents,
        opponent_permanents: &mut encoded.opponent_permanents,
        actions: &mut encoded.actions,
        action_focus: &mut encoded.action_focus,
        agent_player_valid: &mut encoded.agent_player_valid,
        opponent_player_valid: &mut encoded.opponent_player_valid,
        agent_cards_valid: &mut encoded.agent_cards_valid,
        opponent_cards_valid: &mut encoded.opponent_cards_valid,
        agent_permanents_valid: &mut encoded.agent_permanents_valid,
        opponent_permanents_valid: &mut encoded.opponent_permanents_valid,
        actions_valid: &mut encoded.actions_valid,
    };

    encode_into(obs, config, out).expect("internal encode buffer lengths are always valid");
    encoded
}

pub fn encode_into(
    obs: &Observation,
    config: &ObservationEncoderConfig,
    out: EncodedObservationMut<'_>,
) -> Result<(), ObservationEncodeError> {
    validate_buffer_len("agent_player", out.agent_player.len(), PLAYER_DIM)?;
    validate_buffer_len("opponent_player", out.opponent_player.len(), PLAYER_DIM)?;
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

    out.agent_player.fill(0.0);
    out.opponent_player.fill(0.0);
    out.agent_cards.fill(0.0);
    out.opponent_cards.fill(0.0);
    out.agent_permanents.fill(0.0);
    out.opponent_permanents.fill(0.0);
    out.actions.fill(0.0);
    out.action_focus.fill(-1);
    out.agent_player_valid.fill(0.0);
    out.opponent_player_valid.fill(0.0);
    out.agent_cards_valid.fill(0.0);
    out.opponent_cards_valid.fill(0.0);
    out.agent_permanents_valid.fill(0.0);
    out.opponent_permanents_valid.fill(0.0);
    out.actions_valid.fill(0.0);

    let mut object_to_index: HashMap<i32, i32> = HashMap::new();
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

    encode_cards(
        &obs.agent_cards,
        1.0,
        config.max_cards_per_player,
        out.agent_cards,
        out.agent_cards_valid,
        &mut object_to_index,
        &mut current_object_index,
    );
    encode_cards(
        &obs.opponent_cards,
        0.0,
        config.max_cards_per_player,
        out.opponent_cards,
        out.opponent_cards_valid,
        &mut object_to_index,
        &mut current_object_index,
    );

    encode_permanents(
        &obs.agent_permanents,
        1.0,
        config.max_permanents_per_player,
        out.agent_permanents,
        out.agent_permanents_valid,
        &mut object_to_index,
        &mut current_object_index,
    );
    encode_permanents(
        &obs.opponent_permanents,
        0.0,
        config.max_permanents_per_player,
        out.opponent_permanents,
        out.opponent_permanents_valid,
        &mut object_to_index,
        &mut current_object_index,
    );

    encode_actions(
        obs,
        config.max_actions,
        config.max_focus_objects,
        out.actions,
        out.actions_valid,
        out.action_focus,
        &object_to_index,
    );

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

    object_to_index.insert(player.id, *current_object_index);
    *current_object_index += 1;
}

fn encode_cards(
    cards: &[CardData],
    is_mine: f32,
    max_cards: usize,
    out: &mut [f32],
    out_valid: &mut [f32],
    object_to_index: &mut HashMap<i32, i32>,
    current_object_index: &mut i32,
) {
    let ordered_cards = cards.iter().take(max_cards);
    let mut encoded_count = 0;

    for (i, card) in ordered_cards.enumerate() {
        let start = i * CARD_DIM;
        let end = start + CARD_DIM;
        encode_card_features(card, is_mine, &mut out[start..end]);
        out_valid[i] = 1.0;
        object_to_index.insert(card.id, *current_object_index);
        *current_object_index += 1;
        encoded_count += 1;
    }

    *current_object_index += (max_cards.saturating_sub(encoded_count)) as i32;
}

fn encode_card_features(card: &CardData, is_mine: f32, out: &mut [f32]) {
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
    out[17] = 1.0;
}

fn encode_permanents(
    permanents: &[PermanentData],
    is_mine: f32,
    max_permanents: usize,
    out: &mut [f32],
    out_valid: &mut [f32],
    object_to_index: &mut HashMap<i32, i32>,
    current_object_index: &mut i32,
) {
    let ordered_permanents = permanents.iter().take(max_permanents);
    let mut encoded_count = 0;

    for (i, permanent) in ordered_permanents.enumerate() {
        let start = i * PERMANENT_DIM;
        let end = start + PERMANENT_DIM;
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
    out[4] = 1.0;
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
        row[ACTION_DIM - 1] = 1.0;
        out_valid[action_index] = 1.0;

        let focus_start = action_index * max_focus_objects;
        for (focus_index, focus_id) in action.focus.iter().take(max_focus_objects).enumerate() {
            out_focus[focus_start + focus_index] =
                object_to_index.get(focus_id).copied().unwrap_or(-1);
        }
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
                ActionOption, ActionSpaceData, CardData, CardTypeData, KeywordData, Observation,
                PermanentData, PlayerData, TurnData,
            },
        },
        flow::turn::{PhaseKind, StepKind},
        state::{mana::ManaCost, zone::ZoneType},
    };

    use super::{
        encode, encode_into, EncodedObservationMut, ObservationEncoderConfig, ACTION_DIM, CARD_DIM,
        PERMANENT_DIM, PLAYER_DIM,
    };

    fn sample_observation() -> Observation {
        Observation {
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
                    },
                    ActionOption {
                        action_type: ActionType::ChooseTarget,
                        focus: vec![444],
                    },
                    ActionOption {
                        action_type: ActionType::DeclareBlocker,
                        focus: vec![999],
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
            },
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
            },
            opponent_cards: vec![make_card(221, ZoneType::Hand, false, 1, 1, 1)],
            opponent_permanents: vec![make_permanent(444, false)],
        }
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
            power,
            toughness,
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
                cost: [0; 7],
                mana_value,
            },
        }
    }

    fn make_permanent(id: i32, mine: bool) -> PermanentData {
        PermanentData {
            id,
            controller_id: if mine { 10 } else { 20 },
            tapped: false,
            damage: 0,
            is_summoning_sick: false,
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
        };

        let encoded = encode(&obs, &config);

        // Agent and opponent player vectors are populated with turn one-hot slices.
        assert_eq!(encoded.agent_player[9 + 2], 1.0);
        assert_eq!(encoded.agent_player[14 + 5], 1.0);
        assert_eq!(encoded.opponent_player[9 + 2], 1.0);

        // Validity flags for populated card/permanent slots.
        assert_eq!(encoded.agent_cards[17], 1.0);
        assert_eq!(encoded.agent_cards[17 + 18], 1.0);
        assert_eq!(encoded.agent_cards[17 + 18 * 2], 0.0);
        assert_eq!(encoded.agent_permanents[4], 1.0);
        assert_eq!(encoded.agent_permanents[4 + 5], 0.0);

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
    }

    #[test]
    fn encode_into_validates_output_lengths() {
        let obs = sample_observation();
        let config = ObservationEncoderConfig {
            max_cards_per_player: 2,
            max_permanents_per_player: 1,
            max_actions: 2,
            max_focus_objects: 2,
        };

        let mut agent_player = vec![0.0; 1];
        let mut opponent_player = vec![0.0; PLAYER_DIM];
        let mut agent_cards = vec![0.0; 2 * CARD_DIM];
        let mut opponent_cards = vec![0.0; 2 * CARD_DIM];
        let mut agent_permanents = vec![0.0; PERMANENT_DIM];
        let mut opponent_permanents = vec![0.0; PERMANENT_DIM];
        let mut actions = vec![0.0; 2 * ACTION_DIM];
        let mut action_focus = vec![-1; 2 * 2];
        let mut agent_player_valid = vec![0.0; 1];
        let mut opponent_player_valid = vec![0.0; 1];
        let mut agent_cards_valid = vec![0.0; 2];
        let mut opponent_cards_valid = vec![0.0; 2];
        let mut agent_permanents_valid = vec![0.0; 1];
        let mut opponent_permanents_valid = vec![0.0; 1];
        let mut actions_valid = vec![0.0; 2];

        let out = EncodedObservationMut {
            agent_player: &mut agent_player,
            opponent_player: &mut opponent_player,
            agent_cards: &mut agent_cards,
            opponent_cards: &mut opponent_cards,
            agent_permanents: &mut agent_permanents,
            opponent_permanents: &mut opponent_permanents,
            actions: &mut actions,
            action_focus: &mut action_focus,
            agent_player_valid: &mut agent_player_valid,
            opponent_player_valid: &mut opponent_player_valid,
            agent_cards_valid: &mut agent_cards_valid,
            opponent_cards_valid: &mut opponent_cards_valid,
            agent_permanents_valid: &mut agent_permanents_valid,
            opponent_permanents_valid: &mut opponent_permanents_valid,
            actions_valid: &mut actions_valid,
        };

        let err = encode_into(&obs, &config, out).expect_err("invalid length must fail");
        assert_eq!(
            err.to_string(),
            "invalid length for agent_player: expected 26, got 1"
        );
    }
}
