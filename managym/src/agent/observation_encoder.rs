use super::observation_layout::{
    action, card, decklist, event, fact, known_hand, name, permanent, player, turn_state,
};

use std::{
    collections::{BTreeMap, HashMap},
    fmt,
    sync::LazyLock,
};

use crate::{
    agent::observation::{
        CardData, ContinuousEffectData, EventData, Observation, PermanentData, PlayerData,
        RestrictedManaData, TurnData,
    },
    cards::tokens::name_value,
    flow::turn::{PhaseKind, StepKind},
    state::mana::ManaRestriction,
};

// Feature widths. Column indices, names and value encodings live in
// `observation_layout`; open-ended vocabularies (counter kinds, subtypes, set
// characteristics, used abilities) are fact rows, not columns. Fact vocabulary
// ids are the learning tokens' values (`cards::tokens::name_value` of the same
// Rust or printed name), split into 16-bit halves so f32 keeps them exact.
pub const PLAYER_DIM: usize = player::WIDTH;
pub const CARD_DIM: usize = card::WIDTH;
pub const PERMANENT_DIM: usize = permanent::WIDTH;
pub const FACT_DIM: usize = fact::WIDTH;

/// What a fact row states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum FactKind {
    /// Counters of one kind (+1/+1 included); value: the count.
    Counter = 1,
    /// A subtype an effect added.
    AddedSubtype = 2,
    /// The name an effect set.
    SetName = 3,
    /// A creature type an effect set.
    SetCreatureType = 4,
    /// An activated ability (id: its index) whose use limit is spent.
    UsedAbility = 5,
    // 6 is reserved: effect records are shared-effect rows, and later kinds
    // keep their values.
    /// The copy effect that made a permanent a copy (CR 707.2); source: the
    /// copied printed definition, order: the copy's timestamp.
    Copy = 7,
    /// An exception that is part of a copy's copiable values (CR 707.9b):
    /// id its `Modification`, one row per keyword, subtype or color it names.
    CopyException = 8,
    /// Transform count: id 0 for a permanent, frame index + 1 for a continuation.
    Transforms = 9,
    /// Attacker predicate token for a combat continuation (id: frame index).
    CombatRestriction = 10,
    /// A rules program in learning tokens (detail: token value, values:
    /// token kind and ordinal): a copy exception's payload under its
    /// `Modification` id, or a player's pooled mana with a spend effect
    /// under `POOLED_MANA`.
    EffectProgram = 11,
    EffectiveColor = 12,
    KeywordCounterTimestamp = 13,
    ContinuationProgram = 14,
}

impl FactKind {
    pub const ALL: [FactKind; 13] = [
        FactKind::Counter,
        FactKind::AddedSubtype,
        FactKind::SetName,
        FactKind::SetCreatureType,
        FactKind::UsedAbility,
        FactKind::Copy,
        FactKind::CopyException,
        FactKind::Transforms,
        FactKind::CombatRestriction,
        FactKind::EffectProgram,
        FactKind::EffectiveColor,
        FactKind::KeywordCounterTimestamp,
        FactKind::ContinuationProgram,
    ];
}

pub const ACTION_TYPE_DIM: usize = action::DECLARED;
/// Declaration choice is separate from its family: attack and decline must
/// have distinct policy inputs even when they focus the same permanent.
pub const ACTION_DIM: usize = action::WIDTH;
/// Selectors encode index+1, reserving zero for absence. Keep f32 integers exact.
pub const MAX_ACTION_SELECTOR: usize = (1 << 24) - 1;
pub const EVENT_DIM: usize = event::WIDTH;
pub const ZONE_DIM: usize = 7;
pub const PHASE_DIM: usize = 5;
pub const STEP_DIM: usize = 12;

/// Policy support selection only; the game's legal actions remain complete.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ActionOverflow {
    #[default]
    Error,
    FirstLegal,
}

impl std::str::FromStr for ActionOverflow {
    type Err = &'static str;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "error" => Ok(Self::Error),
            "first-legal" => Ok(Self::FirstLegal),
            _ => Err("unknown action overflow policy"),
        }
    }
}

/// Retain a prefix without renumbering native actions or their Commands.
pub fn policy_action_count(
    count: usize,
    capacity: usize,
    overflow: ActionOverflow,
) -> Result<usize, ObservationEncodeError> {
    if count > capacity && matches!(overflow, ActionOverflow::Error) {
        return Err(ObservationEncodeError::Capacity {
            field: "actions",
            capacity,
            actual: count,
        });
    }
    Ok(count.min(capacity))
}
/// Name candidate columns per action row: name row + 1, definition + 1, then
/// the first and last name rows (+ 1) of a naming group choice.
pub const NAME_DIM: usize = name::WIDTH;

/// Decklist rows: definition id + 1, then the main, unseen, sideboard and
/// remaining-sideboard counts (`PlayerData` decklist maps).
pub const DECKLIST_DIM: usize = decklist::WIDTH;

// Input v10 adds turn state v9 has no place for, in two buffers whose
// presence selects it. `player_turn_state` has one row per seat (agent
// first), columns named by `observation_layout::turn_state`.
// `card_turn_state` aligns with the seats' card rows like `semantic_cards`:
// 1 for a graveyard card put there from the battlefield this turn.
pub const TURN_STATE_DIM: usize = turn_state::WIDTH;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObservationEncoderConfig {
    pub max_facts: usize,
    pub max_cards_per_player: usize,
    pub max_permanents_per_player: usize,
    pub max_actions: usize,
    pub action_overflow: ActionOverflow,
    pub max_focus_objects: usize,
    pub max_events: usize,
    pub policy_history: bool,
    /// Policy input v10: write `player_turn_state` and `card_turn_state`.
    /// Without them, an observation that needs that state is refused.
    pub turn_state: bool,
}

impl Default for ObservationEncoderConfig {
    fn default() -> Self {
        Self {
            max_cards_per_player: 60,
            max_permanents_per_player: 40,
            max_actions: 512,
            action_overflow: ActionOverflow::Error,
            max_focus_objects: 2,
            max_events: 32,
            policy_history: false,
            turn_state: false,
            max_facts: 512,
        }
    }
}

impl ObservationEncoderConfig {
    /// Candidate rows share the action axis, but never consume object capacity.
    pub fn name_candidates_len(&self) -> usize {
        NAME_DIM * self.max_actions
    }

    pub fn relations_len(&self) -> usize {
        super::continuation::RELATION_ROWS * super::continuation::RELATION_DIM
    }

    /// Two seats of decklist rows.
    pub fn decklists_len(&self) -> usize {
        2 * self.max_cards_per_player * DECKLIST_DIM
    }

    pub fn effects_len(&self) -> usize {
        super::shared_effects::EFFECT_ROWS * super::shared_effects::EFFECT_DIM
    }

    pub fn effect_details_len(&self) -> usize {
        super::shared_effects::DETAIL_ROWS * super::shared_effects::DETAIL_DIM
    }

    pub fn effect_members_len(&self) -> usize {
        super::shared_effects::MEMBER_ROWS * super::shared_effects::MEMBER_DIM
    }

    /// Two seat rows of v10 player turn state; empty for v9.
    pub fn player_turn_state_len(&self) -> usize {
        if self.turn_state {
            2 * TURN_STATE_DIM
        } else {
            0
        }
    }

    /// Two seats of v10 card turn state; empty for v9.
    pub fn card_turn_state_len(&self) -> usize {
        if self.turn_state {
            2 * self.max_cards_per_player
        } else {
            0
        }
    }

    pub fn facts_len(&self) -> usize {
        self.max_facts * FACT_DIM
    }

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
    pub effects: Vec<f32>,
    pub effect_details: Vec<f32>,
    pub effect_members: Vec<f32>,
    pub name_candidates: Vec<f32>,
    pub facts: Vec<f32>,
    pub semantic_cards: Vec<f32>,
    pub known_hand: Vec<f32>,
    pub decklists: Vec<f32>,
    pub player_turn_state: Vec<f32>,
    pub card_turn_state: Vec<f32>,
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
    pub effects: &'a mut [f32],
    pub effect_details: &'a mut [f32],
    pub effect_members: &'a mut [f32],
    pub name_candidates: &'a mut [f32],
    pub facts: &'a mut [f32],
    pub semantic_cards: &'a mut [f32],
    pub known_hand: &'a mut [f32],
    pub decklists: &'a mut [f32],
    pub player_turn_state: &'a mut [f32],
    pub card_turn_state: &'a mut [f32],
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
            effects: &mut self.effects,
            effect_details: &mut self.effect_details,
            effect_members: &mut self.effect_members,
            name_candidates: &mut self.name_candidates,
            facts: &mut self.facts,
            semantic_cards: &mut self.semantic_cards,
            known_hand: &mut self.known_hand,
            decklists: &mut self.decklists,
            player_turn_state: &mut self.player_turn_state,
            card_turn_state: &mut self.card_turn_state,
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
    InvalidShared(&'static str),
    InvalidName(&'static str),
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
    /// A v9 observation that needs state only input v10 holds.
    RequiresInputV10(&'static str),
    /// State the policy input has no columns or rows for; encoding it would
    /// hide a distinction the game makes.
    Unrepresentable(String),
}

impl fmt::Display for ObservationEncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidName(reason) => write!(f, "invalid name candidate: {reason}"),
            Self::RequiresInputV10(state) => write!(
                f,
                "policy input v10 required: this observation needs {state}, which v9 cannot hold"
            ),
            Self::InvalidShared(reason) => write!(f, "invalid shared effects: {reason}"),
            Self::Unrepresentable(state) => {
                write!(f, "the policy input cannot represent {state}")
            }
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
    #[cfg(feature = "benchmark")]
    let allocation_timer = crate::benchmark_search::Timer::new("encoding/native_allocation");
    let mut encoded = EncodedObservation {
        relations: vec![0.0; config.relations_len()],
        effects: vec![0.0; config.effects_len()],
        effect_details: vec![0.0; config.effect_details_len()],
        effect_members: vec![0.0; config.effect_members_len()],
        name_candidates: vec![0.0; config.name_candidates_len()],
        facts: vec![0.0; config.facts_len()],
        semantic_cards: vec![0.0; 2 * config.max_cards_per_player],
        known_hand: vec![0.0; 4 * config.max_cards_per_player],
        decklists: vec![0.0; config.decklists_len()],
        player_turn_state: vec![0.0; config.player_turn_state_len()],
        card_turn_state: vec![0.0; config.card_turn_state_len()],
        agent_player: vec![0.0; PLAYER_DIM],
        opponent_player: vec![0.0; PLAYER_DIM],
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

    #[cfg(feature = "benchmark")]
    drop(allocation_timer);
    #[cfg(feature = "benchmark")]
    let _encode_timer = crate::benchmark_search::Timer::new("encoding/native_encode");
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
    validate_buffer_len("decklists", out.decklists.len(), config.decklists_len())?;
    validate_buffer_len(
        "player_turn_state",
        out.player_turn_state.len(),
        config.player_turn_state_len(),
    )?;
    validate_buffer_len(
        "card_turn_state",
        out.card_turn_state.len(),
        config.card_turn_state_len(),
    )?;
    if !config.turn_state {
        if let Some(state) = input_v10_state(obs) {
            return Err(ObservationEncodeError::RequiresInputV10(state));
        }
    }
    if let Some(state) = unrepresentable_state(obs) {
        return Err(ObservationEncodeError::Unrepresentable(state));
    }
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
        "name_candidates",
        out.name_candidates.len(),
        config.name_candidates_len(),
    )?;
    validate_buffer_len(
        "action_focus",
        out.action_focus.len(),
        config.action_focus_len(),
    )?;
    validate_buffer_len("events", out.events.len(), config.events_len())?;
    validate_buffer_len("effects", out.effects.len(), config.effects_len())?;
    validate_buffer_len(
        "effect_details",
        out.effect_details.len(),
        config.effect_details_len(),
    )?;
    validate_buffer_len(
        "effect_members",
        out.effect_members.len(),
        config.effect_members_len(),
    )?;
    validate_buffer_len("facts", out.facts.len(), config.facts_len())?;
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
            policy_action_count(
                obs.action_space.actions.len(),
                config.max_actions,
                config.action_overflow,
            )?,
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
    for action in obs.action_space.actions.iter().take(config.max_actions) {
        if let Some(definition) = action.name_definition {
            if action.name_index.is_none() || definition as usize > MAX_ACTION_SELECTOR {
                return Err(ObservationEncodeError::InvalidName(
                    "missing identity or definition exceeds exact integer capacity",
                ));
            }
        }

        if let Some(index) = action.continuation_index {
            if index >= obs.continuation.len() {
                return Err(ObservationEncodeError::Capacity {
                    field: "action continuation reference",
                    capacity: obs.continuation.len(),
                    actual: index + 1,
                });
            }
        }
        for index in [action.mode_index, action.ability_index, action.name_index]
            .into_iter()
            .flatten()
            .chain(action.name_group.into_iter().flatten())
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
    out.player_turn_state.fill(0.0);
    out.card_turn_state.fill(0.0);
    out.known_hand.fill(0.0);
    out.decklists.fill(0.0);
    out.agent_player.fill(0.0);
    out.opponent_player.fill(0.0);
    out.agent_cards.fill(0.0);
    out.opponent_cards.fill(0.0);
    out.agent_permanents.fill(0.0);
    out.opponent_permanents.fill(0.0);
    out.actions.fill(0.0);
    out.name_candidates.fill(0.0);
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
            let offset = (seat * config.max_cards_per_player + slot) * known_hand::WIDTH;
            out.known_hand[offset + known_hand::DEFINITION] = (definition + 1) as f32;
            out.known_hand[offset + known_hand::COUNT] = *count as f32;
        }
    }
    for (slot, card) in obs.agent_sideboard.iter().enumerate() {
        out.semantic_cards[obs.agent_cards.len() + slot] = (card.registry_key + 1) as f32;
    }
    if config.turn_state {
        encode_turn_state(obs, config, out.player_turn_state, out.card_turn_state);
    }
    for (seat, player) in [&obs.agent, &obs.opponent].into_iter().enumerate() {
        let rows = decklist_rows(player);
        if rows.len() > config.max_cards_per_player {
            return Err(ObservationEncodeError::Capacity {
                field: "decklist definitions",
                capacity: config.max_cards_per_player,
                actual: rows.len(),
            });
        }
        for (slot, row) in rows.iter().enumerate() {
            let offset = (seat * config.max_cards_per_player + slot) * DECKLIST_DIM;
            out.decklists[offset..offset + DECKLIST_DIM].copy_from_slice(row);
        }
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
        let width = CARD_DIM;
        let row = &mut out.agent_cards[index * width..(index + 1) * width];
        row[card::IS_MINE] = 1.0;
        row[card::OUTSIDE] = 1.0; // outside inventory; all seven zone bits remain zero
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
    // Hosts may be listed after their attachments, so links follow indexing.
    let ranks = timestamp_ranks(obs);
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
            row[permanent::ATTACHED_TO] = link(permanent.attached_to);
            row[permanent::BLOCKING] = link(permanent.blocking);
            row[permanent::TIMESTAMP] = ranks[&permanent.timestamp()];
        }
    }
    encode_facts(obs, &object_to_index, &ranks, out.facts)?;
    super::shared_effects::encode(
        obs,
        &object_to_index,
        &ranks,
        out.effects,
        out.effect_details,
        out.effect_members,
    )?;

    super::continuation::encode_into(
        &obs.continuation,
        obs.agent.player_index as usize,
        &object_to_index,
        out.relations,
    )?;

    encode_actions(
        obs,
        config,
        out.actions,
        out.actions_valid,
        out.action_focus,
        out.name_candidates,
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

// ---------------------------------------------------------------------------
// Input v10: turn state
// ---------------------------------------------------------------------------

fn encode_turn_state(
    obs: &Observation,
    config: &ObservationEncoderConfig,
    players: &mut [f32],
    cards: &mut [f32],
) {
    for (seat, (player, seat_cards)) in [
        (&obs.agent, &obs.agent_cards),
        (&obs.opponent, &obs.opponent_cards),
    ]
    .into_iter()
    .enumerate()
    {
        let row = &mut players[seat * TURN_STATE_DIM..(seat + 1) * TURN_STATE_DIM];
        row[turn_state::LIFE_LOST] = player.turn_history.life_lost as f32 / 20.0;
        row[turn_state::SPELLS_CAST] = player.spells_cast as f32 / 10.0;
        for class in player
            .restricted_mana
            .iter()
            .filter(|class| class.except_nonartifact_spells)
        {
            for slot in 0..6 {
                row[turn_state::NONARTIFACT_MANA_WHITE + slot] += class.mana[slot] as f32 / 10.0;
            }
        }
        for (slot, card) in seat_cards.iter().enumerate() {
            cards[seat * config.max_cards_per_player + slot] =
                bool_to_f32(card.from_battlefield_this_turn);
        }
    }
}

/// Which v10-only state a definition's rules read, found by walking its
/// serialized value for the rules terms that ask for it.
#[derive(Clone, Copy, Debug, Default)]
struct TurnStateReads {
    life_lost: bool,
    from_battlefield: bool,
}

impl TurnStateReads {
    fn of(definition: &crate::state::card::CardDefinition) -> Self {
        fn walk(value: &serde_json::Value, reads: &mut TurnStateReads) {
            match value {
                serde_json::Value::Object(map) => {
                    for (key, value) in map {
                        match key.as_str() {
                            "OpponentLostLifeThisTurnAtLeast" => reads.life_lost = true,
                            "from_battlefield_this_turn" if *value == true => {
                                reads.from_battlefield = true
                            }
                            _ => {}
                        }
                        walk(value, reads);
                    }
                }
                serde_json::Value::Array(values) => {
                    values.iter().for_each(|value| walk(value, reads))
                }
                _ => {}
            }
        }
        let mut reads = Self::default();
        if let Ok(value) = serde_json::to_value(definition) {
            walk(&value, &mut reads);
        }
        reads
    }
}

/// Per definition of the world's pack, by `CardDefId`; computed once.
fn turn_state_reads(registry_key: i32) -> TurnStateReads {
    static READS: std::sync::OnceLock<Vec<TurnStateReads>> = std::sync::OnceLock::new();
    let reads = READS.get_or_init(|| {
        let pack = crate::cards::content_pack::default_content_pack();
        let mut reads = Vec::new();
        for (id, definition) in pack.definition_entries() {
            let index = id.0 as usize;
            if reads.len() <= index {
                reads.resize(index + 1, TurnStateReads::default());
            }
            reads[index] = TurnStateReads::of(definition);
        }
        reads
    });
    usize::try_from(registry_key)
        .ok()
        .and_then(|key| reads.get(key))
        .copied()
        .unwrap_or_default()
}

/// The subtypes with restricted-mana columns, in column order. A class
/// admitting several subtypes counts under each column.
const RESTRICTED_SUBTYPE_COLUMNS: [&str; 3] = ["Lesson", "Shrine", "Ally"];

/// The restrictions the player columns state exactly. With the total, the
/// overlapping Lesson and Shrine columns solve uniquely for these classes and
/// no others: a Lesson-or-Ally class beside a Shrine class sums like a Lesson
/// class beside a Shrine-or-Ally one. Powerstone mana has its own v10 columns.
fn representable(restriction: &ManaRestriction) -> bool {
    static REPRESENTABLE: LazyLock<[ManaRestriction; 4]> = LazyLock::new(|| {
        [
            ManaRestriction::spell(["Lesson"]),
            ManaRestriction::spell(["Lesson", "Shrine"]),
            ManaRestriction::only(["Ally"], ["Ally"]),
            ManaRestriction::ExceptNonartifactSpells,
        ]
    });
    REPRESENTABLE.contains(restriction)
}

/// Restricted mana in `obs` that the player columns cannot state exactly, if
/// any. Encoders refuse it before writing a buffer; its subtypes stay
/// complete in `RestrictedManaData` and the text observation.
pub fn unrepresentable_state(obs: &Observation) -> Option<String> {
    obs.agent
        .restricted_mana
        .iter()
        .chain(&obs.opponent.restricted_mana)
        .map(RestrictedManaData::restriction)
        .find(|restriction| !representable(restriction))
        .map(|restriction| format!("mana restricted to {}", restriction.describe()))
}

/// The v10-only state this observation needs, if any: a definition visible
/// to the viewer reads it and it is present. V9 refuses such an observation
/// before writing a buffer rather than hide state a visible card reads.
pub fn input_v10_state(obs: &Observation) -> Option<&'static str> {
    let players = [&obs.agent, &obs.opponent];
    let cards = || obs.agent_cards.iter().chain(&obs.opponent_cards);
    let visible_reads = || {
        cards()
            .map(|card| card.registry_key)
            .chain(
                obs.agent_permanents
                    .iter()
                    .chain(&obs.opponent_permanents)
                    .map(|permanent| permanent.registry_key),
            )
            .chain(
                obs.stack_objects
                    .iter()
                    .map(|object| object.source_card_registry_key),
            )
            .map(turn_state_reads)
    };
    if players
        .iter()
        .any(|player| player.turn_history.life_lost > 0)
        && visible_reads().any(|reads| reads.life_lost)
    {
        return Some("life lost this turn");
    }
    if cards().any(|card| card.from_battlefield_this_turn)
        && visible_reads().any(|reads| reads.from_battlefield)
    {
        return Some("a card put into a graveyard from the battlefield this turn");
    }
    // Only a Powerstone-like source produces it, so its presence alone is
    // the need.
    if players.iter().any(|player| {
        player
            .restricted_mana
            .iter()
            .any(|class| class.except_nonartifact_spells && class.mana.iter().any(|n| *n > 0))
    }) {
        return Some("mana that can't be spent to cast nonartifact spells");
    }
    None
}

/// One row per definition in the union of a known player's main deck and
/// sideboard, sorted by definition id; none for a closed decklist. Unseen
/// counts are a subset of the main deck's definitions plus retrieved
/// sideboard copies, so the union covers every count map.
fn decklist_rows(player: &PlayerData) -> Vec<[f32; DECKLIST_DIM]> {
    let maps = [
        &player.main_counts,
        &player.unseen_counts,
        &player.sideboard_counts,
        &player.remaining_sideboard_counts,
    ];
    let mut rows: std::collections::BTreeMap<u32, [f32; DECKLIST_DIM]> = Default::default();
    for (column, counts) in maps.into_iter().enumerate() {
        for (definition, count) in counts.iter().flatten() {
            let row = rows.entry(*definition).or_insert_with(|| {
                let mut row = [0.0; DECKLIST_DIM];
                row[decklist::DEFINITION] = (definition + 1) as f32;
                row
            });
            row[column + decklist::MAIN] = *count as f32;
        }
    }
    rows.into_values().collect()
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
    out[player::LIFE] = player.life as f32 / 20.0;
    out[player::IS_ACTIVE] = bool_to_f32(player.is_active);

    for i in 0..ZONE_DIM {
        out[player::LIBRARY_COUNT + i] = player.zone_counts[i] as f32 / 60.0;
    }

    let phase_index = phase_index(turn.phase);
    if phase_index < PHASE_DIM {
        out[player::PHASE_BEGINNING + phase_index] = 1.0;
    }

    let step_index = step_index(turn.step);
    if step_index < STEP_DIM {
        out[player::STEP_UNTAP + step_index] = 1.0;
    }

    out[player::GRAVEYARD_LESSONS] = player.graveyard_lessons as f32 / 10.0;
    out[player::COMBAT_MANA] = player.combat_mana as f32 / 10.0;
    for slot in 0..6 {
        out[player::MANA_WHITE + slot] = player.mana[slot] as f32 / 10.0;
        out[player::LESSON_MANA_WHITE + slot] = player.lesson_mana[slot] as f32 / 10.0;
    }
    // Restricted mana each column's subtype may use, by color. A class
    // admitting several subtypes counts under each, so these columns
    // overlap and need not sum to the total above.
    for (index, subtype) in RESTRICTED_SUBTYPE_COLUMNS.iter().enumerate() {
        for class in player
            .restricted_mana
            .iter()
            .filter(|class| class.admits(subtype))
        {
            for slot in 0..6 {
                out[player::RESTRICTED_LESSON_WHITE + 6 * index + slot] +=
                    class.mana[slot] as f32 / 10.0;
            }
        }
    }
    out[player::HAND_ONLY] = bool_to_f32(player.hand_only_until.is_some());
    out[player::NO_MAXIMUM_HAND_SIZE] = bool_to_f32(player.no_maximum_hand_size);
    out[player::TURN] = turn.turn_number as f32 / 20.0;
    // Seat 0 is on the play.
    out[player::SEAT] = player.player_index as f32;
    out[player::KEEPS_RED_MANA] = bool_to_f32(player.keeps_red_mana);
    let history = &player.turn_history;
    out[player::ATTACKED] = bool_to_f32(history.attacked);
    out[player::PERMANENTS_SACRIFICED] = history.permanents_sacrificed as f32 / 10.0;
    out[player::LAND_ENTERED] = bool_to_f32(history.land_entered);
    out[player::CARDS_DRAWN] = history.cards_drawn as f32 / 10.0;
    for (index, kind) in crate::flow::turn::Bending::ALL.into_iter().enumerate() {
        out[player::BENT_WATER + index] = bool_to_f32(history.bent & kind as u8 != 0);
    }
    out[player::EXTRA_TURNS] = player.extra_turns as f32;
    out[player::ADDITIONAL_COMBATS] = player.additional_combats as f32;

    object_to_index.insert(player.id, *current_object_index);
    *current_object_index += 1;
}

/// Facts every card row of one observation reads.
struct CardContext {
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
        let start = i * CARD_DIM;
        let end = start + CARD_DIM;
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
        out[card::ZONE_LIBRARY + zone_index] = 1.0;
    }
    out[card::IS_MINE] = is_mine;
    out[card::POWER] = card.power as f32 / 10.0;
    out[card::TOUGHNESS] = card.toughness as f32 / 10.0;
    out[card::MANA_VALUE] = card.mana_cost.mana_value as f32 / 10.0;
    out[card::IS_LAND] = bool_to_f32(card.card_types.is_land);
    out[card::IS_CREATURE] = bool_to_f32(card.card_types.is_creature);
    out[card::IS_ARTIFACT] = bool_to_f32(card.card_types.is_artifact);
    out[card::IS_ENCHANTMENT] = bool_to_f32(card.card_types.is_enchantment);
    out[card::IS_PLANESWALKER] = bool_to_f32(card.card_types.is_planeswalker);
    out[card::IS_BATTLE] = bool_to_f32(card.card_types.is_battle);
    out[card::FLYING] = bool_to_f32(card.keywords.flying);
    out[card::REACH] = bool_to_f32(card.keywords.reach);
    out[card::HASTE] = bool_to_f32(card.keywords.haste);
    out[card::VIGILANCE] = bool_to_f32(card.keywords.vigilance);
    out[card::TRAMPLE] = bool_to_f32(card.keywords.trample);
    out[card::FIRST_STRIKE] = bool_to_f32(card.keywords.first_strike);
    out[card::DOUBLE_STRIKE] = bool_to_f32(card.keywords.double_strike);
    out[card::DEATHTOUCH] = bool_to_f32(card.keywords.deathtouch);
    out[card::LIFELINK] = bool_to_f32(card.keywords.lifelink);
    out[card::DEFENDER] = bool_to_f32(card.keywords.defender);
    out[card::MENACE] = bool_to_f32(card.keywords.menace);
    out[card::FLASH] = bool_to_f32(card.keywords.flash);
    out[card::IS_TOKEN] = bool_to_f32(card.is_token);
    out[card::IS_ALLY] = bool_to_f32(card.is_ally);
    out[card::IS_LESSON] = bool_to_f32(card.is_lesson);
    out[card::HAS_WARD] = bool_to_f32(card.ward_cost > 0);
    out[card::WARD_COST] = card.ward_cost as f32 / 10.0;
    out[card::HAS_KICKER] = bool_to_f32(card.kicker_cost > 0);
    out[card::KICKER_COST] = card.kicker_cost as f32 / 10.0;
    out[card::HEXPROOF] = bool_to_f32(card.keywords.hexproof);
    out[card::INDESTRUCTIBLE] = bool_to_f32(card.keywords.indestructible);
    out[card::IS_LEGENDARY] = bool_to_f32(card.is_legendary);
    if let Some(permission) = &card.cast_permission {
        out[card::PERMISSION] = 1.0;
        out[card::PERMISSION_IS_MINE] = bool_to_f32(permission.player_id == context.viewer_id);
        if let Some(cost) = &permission.cost {
            out[card::PERMISSION_HAS_COST] = 1.0;
            out[card::PERMISSION_MANA_VALUE] = cost.mana_value as f32 / 10.0;
        }
        out[card::PERMISSION_THIS_TURN] = bool_to_f32(permission.this_turn);
        out[card::PERMISSION_OWN_TURN_ONLY] = bool_to_f32(permission.own_turn_only);
        // Not yet usable: "cast it on a later turn" and this is that turn
        // or earlier.
        out[card::PERMISSION_NOT_YET] = bool_to_f32(
            permission
                .after_turn
                .is_some_and(|turn| context.turn_number <= turn),
        );
        out[card::PERMISSION_HAS_SOURCE] = bool_to_f32(permission.source_permanent_id.is_some());
        out[card::PERMISSION_FLASH] = bool_to_f32(permission.flash);
        out[card::PERMISSION_ANY_MANA] = bool_to_f32(permission.any_mana);
        out[card::PERMISSION_WATERBEND] = bool_to_f32(permission.waterbend);
        out[card::PERMISSION_FORETOLD] = bool_to_f32(permission.foretold);
        out[card::PERMISSION_THROUGH_NEXT_TURN] = bool_to_f32(permission.through_next_turn);
    }
    if let Some(x) = card.x {
        out[card::HAS_X] = 1.0;
        out[card::X] = x as f32 / 10.0;
    }
    out[card::VALID] = 1.0;
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
    let width = PERMANENT_DIM;
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
    out[permanent::IS_MINE] = is_mine;
    out[permanent::TAPPED] = bool_to_f32(permanent.tapped);
    out[permanent::DAMAGE] = permanent.damage as f32 / 10.0;
    out[permanent::SUMMONING_SICK] = bool_to_f32(permanent.is_summoning_sick);
    out[permanent::PLUS_ONE_COUNTERS] = permanent.plus1_counters as f32 / 10.0;
    out[permanent::CANT_BE_BLOCKED_THIS_TURN] = bool_to_f32(permanent.cant_be_blocked_this_turn);
    out[permanent::POWER] = permanent.power as f32 / 10.0;
    out[permanent::TOUGHNESS] = permanent.toughness as f32 / 10.0;
    out[permanent::ANIMATED] = bool_to_f32(permanent.is_animated);
    out[permanent::HAS_EXILE_LINK] = bool_to_f32(permanent.has_exile_link);
    // Effective keywords (printed + until-EOT grants) — agents see what the
    // permanent can do right now, not just what's printed on the card.
    out[permanent::FLYING] = bool_to_f32(permanent.keywords.flying);
    out[permanent::REACH] = bool_to_f32(permanent.keywords.reach);
    out[permanent::HASTE] = bool_to_f32(permanent.keywords.haste);
    out[permanent::FLASH] = bool_to_f32(permanent.keywords.flash);
    out[permanent::VIGILANCE] = bool_to_f32(permanent.keywords.vigilance);
    out[permanent::TRAMPLE] = bool_to_f32(permanent.keywords.trample);
    out[permanent::FIRST_STRIKE] = bool_to_f32(permanent.keywords.first_strike);
    out[permanent::DOUBLE_STRIKE] = bool_to_f32(permanent.keywords.double_strike);
    out[permanent::DEATHTOUCH] = bool_to_f32(permanent.keywords.deathtouch);
    out[permanent::LIFELINK] = bool_to_f32(permanent.keywords.lifelink);
    out[permanent::DEFENDER] = bool_to_f32(permanent.keywords.defender);
    out[permanent::MENACE] = bool_to_f32(permanent.keywords.menace);
    out[permanent::HEXPROOF] = bool_to_f32(permanent.keywords.hexproof);
    out[permanent::EXILE_IF_DIES_THIS_TURN] = bool_to_f32(permanent.exile_if_dies_this_turn);
    out[permanent::INDESTRUCTIBLE] = bool_to_f32(permanent.keywords.indestructible);
    out[permanent::ADDED_AVATAR] =
        bool_to_f32(permanent.temp_subtypes.iter().any(|s| s == "Avatar"));
    out[permanent::ATTACKING] = bool_to_f32(permanent.attacking);
    out[permanent::DEFINITION] = (permanent.registry_key + 1) as f32;
    // Ability-use limits and granted firebending.
    out[permanent::EXHAUSTED_ABILITIES] = permanent.exhausted_abilities as f32;
    out[permanent::ONCE_EACH_TURN_USED] = permanent.once_each_turn_used as f32;
    out[permanent::GRANTED_FIREBENDING] = permanent.granted_firebending as f32 / 10.0;
    // permanent::ATTACHED_TO and permanent::BLOCKING are written
    // once every object is indexed; permanent::TIMESTAMP with facts.
    out[permanent::GRANTED_LAND] = bool_to_f32(permanent.is_granted_land);
    out[permanent::ABILITIES_LOST] = bool_to_f32(permanent.abilities_lost);
    out[permanent::DEALT_DAMAGE_THIS_TURN] = bool_to_f32(permanent.dealt_damage_this_turn);
    out[out.len() - 1] = 1.0;
}

/// The id of an `EffectProgram` on a player that states pooled mana with a
/// spend effect (`SpendEffectManaData::program`).
pub const POOLED_MANA: &str = "PooledMana";

/// One fact row before its object join and timestamp rank are known.
struct Fact {
    object: i32,
    kind: FactKind,
    id: i32,
    detail: i32,
    values: [f32; 2],
    effect: Option<(f32, i32, Option<i32>)>,
}

/// The facts `PermanentData` and `PlayerData` state as open vocabularies, in
/// object order (agent, opponent, agent permanents, opponent permanents).
fn facts(obs: &Observation) -> Vec<Fact> {
    let mut out = Vec::new();
    let mut push = |object: i32, kind: FactKind, id: i32, value: f32| {
        out.push(Fact {
            object,
            kind,
            id,
            detail: 0,
            values: [value, 0.0],
            effect: None,
        })
    };
    let permanents = || obs.agent_permanents.iter().chain(&obs.opponent_permanents);
    for permanent in permanents() {
        let object = permanent.id;
        if permanent.plus1_counters > 0 {
            push(
                object,
                FactKind::Counter,
                name_value("CounterKind::plus_one_plus_one"),
                permanent.plus1_counters as f32 / 10.0,
            );
        }
        for (kind, count) in &permanent.counters {
            push(
                object,
                FactKind::Counter,
                name_value(&format!("CounterKind::{kind}")),
                *count as f32 / 10.0,
            );
        }
        if let Some(colors) = &permanent.colors {
            if colors.is_empty() {
                push(
                    object,
                    FactKind::EffectiveColor,
                    name_value("Color::Colorless"),
                    1.0,
                );
            }
            for color in colors {
                push(
                    object,
                    FactKind::EffectiveColor,
                    name_value(&format!("Color::{color}")),
                    1.0,
                );
            }
        }
        for subtype in &permanent.temp_subtypes {
            push(object, FactKind::AddedSubtype, name_value(subtype), 1.0);
        }
        if let Some(name) = &permanent.name {
            push(object, FactKind::SetName, name_value(name), 1.0);
        }
        for creature_type in permanent.creature_types.iter().flatten() {
            push(
                object,
                FactKind::SetCreatureType,
                name_value(creature_type),
                1.0,
            );
        }
        if permanent.transforms > 0 {
            push(object, FactKind::Transforms, 0, permanent.transforms as f32);
        }
        for index in &permanent.used_abilities {
            push(object, FactKind::UsedAbility, *index as i32, 1.0);
        }
    }
    for (index, frame) in obs.continuation.iter().enumerate() {
        if frame.source_transforms > 0 {
            push(
                frame.source.object.unwrap_or(
                    if frame.controller == obs.agent.player_index as usize {
                        obs.agent.id
                    } else {
                        obs.opponent.id
                    },
                ),
                FactKind::Transforms,
                (index + 1) as i32,
                frame.source_transforms as f32,
            );
        }
    }
    for (index, frame) in obs.continuation.iter().enumerate() {
        for program in [&frame.program, &frame.effect_execution] {
            for (ordinal, &(kind, value)) in program.iter().enumerate() {
                out.push(Fact {
                    object: if frame.controller == obs.agent.player_index as usize {
                        obs.agent.id
                    } else {
                        obs.opponent.id
                    },
                    kind: FactKind::ContinuationProgram,
                    id: (index + 1) as i32,
                    detail: value,
                    values: [kind as f32, ordinal as f32],
                    effect: None,
                });
            }
        }
        if let Some(attackers) = &frame.attackers {
            out.push(Fact {
                object: if frame.controller == obs.agent.player_index as usize {
                    obs.agent.id
                } else {
                    obs.opponent.id
                },
                kind: FactKind::CombatRestriction,
                id: (index + 1) as i32,
                detail: name_value(attackers),
                values: [0.0, 0.0],
                effect: None,
            });
        }
    }
    for permanent in permanents() {
        let Some(copy) = &permanent.copy else {
            continue;
        };
        out.push(Fact {
            object: permanent.id,
            kind: FactKind::Copy,
            id: name_value("CopyEffect"),
            detail: 0,
            values: [0.0, 0.0],
            effect: Some((2.0, permanent.timestamp(), Some(copy.registry_key))),
        });
        for exception in &copy.exceptions {
            copy_exception_facts(permanent.id, exception, &mut out);
        }
    }
    for permanent in permanents() {
        for (kind, &timestamp) in &permanent.keyword_counter_timestamps {
            out.push(Fact {
                object: permanent.id,
                kind: FactKind::KeywordCounterTimestamp,
                id: name_value(&format!("CounterKind::{kind}")),
                detail: 0,
                values: [0.0, 0.0],
                effect: Some((0.0, timestamp, None)),
            });
        }
    }
    // Pooled mana with a spend effect: one program per class on its player.
    for player in [&obs.agent, &obs.opponent] {
        for class in &player.spend_effect_mana {
            for (ordinal, &(kind, value)) in class.program.iter().enumerate() {
                out.push(Fact {
                    object: player.id,
                    kind: FactKind::EffectProgram,
                    id: name_value(POOLED_MANA),
                    detail: value,
                    values: [f32::from(kind), ordinal as f32],
                    effect: None,
                });
            }
        }
    }
    out
}

/// The rows of one copy exception; effect records themselves are
/// shared-effect rows.
fn copy_exception_facts(object: i32, effect: &ContinuousEffectData, out: &mut Vec<Fact>) {
    let kind = FactKind::CopyException;
    let id = name_value(&format!("Modification::{}", effect.modification));
    let duration = super::shared_effects::duration_code(&effect.duration).unwrap_or(2.0);
    let values = [
        effect.power.or(effect.amount).unwrap_or(0) as f32 / 10.0,
        effect.toughness.unwrap_or(0) as f32 / 10.0,
    ];
    let details: Vec<i32> = effect
        .keywords
        .iter()
        .chain(&effect.subtypes)
        .map(|name| name_value(name))
        .chain(
            effect
                .color
                .iter()
                .map(|color| name_value(&format!("Color::{color}"))),
        )
        .collect();
    for (ordinal, &(token_kind, token_value)) in effect.program.iter().enumerate() {
        out.push(Fact {
            object,
            kind: FactKind::EffectProgram,
            id,
            detail: token_value,
            values: [f32::from(token_kind), ordinal as f32],
            effect: Some((duration, effect.timestamp, effect.source_registry_key)),
        });
    }
    for detail in if details.is_empty() { vec![0] } else { details } {
        out.push(Fact {
            object,
            kind,
            id,
            detail,
            values,
            effect: Some((duration, effect.timestamp, effect.source_registry_key)),
        });
    }
}

/// Rank of each timestamp among the observation's permanents and effect
/// records (1-based), so order is visible without exposing raw ids.
fn timestamp_ranks(obs: &Observation) -> BTreeMap<i32, f32> {
    let permanents = obs.agent_permanents.iter().chain(&obs.opponent_permanents);
    let mut stamps: Vec<i32> = permanents.map(PermanentData::timestamp).collect();
    stamps.extend(
        obs.shared_effects
            .iter()
            .flatten()
            .map(|record| record.effect.timestamp),
    );
    stamps.extend(
        obs.agent_permanents
            .iter()
            .chain(&obs.opponent_permanents)
            .flat_map(|p| p.keyword_counter_timestamps.values().copied()),
    );
    stamps.sort_unstable();
    stamps.dedup();
    stamps
        .into_iter()
        .enumerate()
        .map(|(rank, stamp)| (stamp, (rank + 1) as f32))
        .collect()
}

pub(super) fn split_id(id: i32) -> [f32; 2] {
    let id = id as u32;
    [(id >> 16) as f32, (id & 0xffff) as f32]
}

fn encode_facts(
    obs: &Observation,
    object_to_index: &HashMap<i32, i32>,
    ranks: &BTreeMap<i32, f32>,
    out: &mut [f32],
) -> Result<(), ObservationEncodeError> {
    let facts = facts(obs);
    let capacity = out.len() / FACT_DIM;
    if facts.len() > capacity {
        return Err(ObservationEncodeError::Capacity {
            field: "facts",
            capacity,
            actual: facts.len(),
        });
    }
    out.fill(0.0);
    for (row, fact) in out.as_chunks_mut::<FACT_DIM>().0.iter_mut().zip(&facts) {
        row[fact::OBJECT] = object_to_index
            .get(&fact.object)
            .map_or(0.0, |index| (index + 1) as f32);
        row[fact::KIND] = fact.kind as u8 as f32;
        row[fact::ID_HIGH..fact::ID_HIGH + 2].copy_from_slice(&split_id(fact.id));
        row[fact::DETAIL_HIGH..fact::DETAIL_HIGH + 2].copy_from_slice(&split_id(fact.detail));
        row[fact::VALUE..fact::VALUE + 2].copy_from_slice(&fact.values);
        if let Some((duration, timestamp, source)) = fact.effect {
            row[fact::DURATION] = duration;
            row[fact::TIMESTAMP] = ranks[&timestamp];
            row[fact::SOURCE_DEFINITION] = source.map_or(0.0, |key| (key + 1) as f32);
        }
        row[fact::VALID] = 1.0;
    }
    Ok(())
}

fn encode_actions(
    obs: &Observation,
    config: &ObservationEncoderConfig,
    out_actions: &mut [f32],
    out_valid: &mut [f32],
    out_focus: &mut [i32],
    out_names: &mut [f32],
    object_to_index: &HashMap<i32, i32>,
) {
    // encode_into has validated the retained policy support and its focus capacity.
    for (action_index, action) in obs
        .action_space
        .actions
        .iter()
        .take(config.max_actions)
        .enumerate()
    {
        let action_start = action_index * ACTION_DIM;
        let action_end = action_start + ACTION_DIM;
        let row = &mut out_actions[action_start..action_end];

        let action_type_index = action.action_type as usize;
        if action_type_index < ACTION_TYPE_DIM {
            row[action_type_index] = 1.0;
        }
        row[action::DECLARED] = bool_to_f32(action.declared.unwrap_or(false));
        let name_row = &mut out_names[NAME_DIM * action_index..NAME_DIM * (action_index + 1)];
        match (action.name_index, action.name_group) {
            // A name is categorical; its position is never a numeric mode feature.
            (Some(index), _) => {
                name_row[name::NAME] = (index + 1) as f32;
                name_row[name::DEFINITION] =
                    action.name_definition.map_or(0.0, |id| (id + 1) as f32);
            }
            // A group is its public member span.
            (_, Some([first, last])) => {
                name_row[name::GROUP_FIRST] = (first + 1) as f32;
                name_row[name::GROUP_LAST] = (last + 1) as f32;
            }
            _ => {
                row[action::MODE] = action.mode_index.map_or(0.0, |index| (index + 1) as f32);
            }
        }
        row[action::ABILITY] = action.ability_index.map_or(0.0, |index| (index + 1) as f32);
        row[action::CONTINUATION] = action
            .continuation_index
            .map_or(0.0, |index| (index + 1) as f32);
        row[action::VALID] = 1.0;
        out_valid[action_index] = 1.0;

        let focus_start = action_index * config.max_focus_objects;
        for (focus_index, focus_id) in action.focus.iter().enumerate() {
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
        row[event::KIND] = event.event_type as f32;
        row[event::SOURCE_KIND] = event.source_kind as f32;
        row[event::SOURCE_ID] = event.source_id as f32;
        row[event::TARGET_KIND] = event.target_kind as f32;
        row[event::TARGET_ID] = event.target_id as f32;
        row[event::AMOUNT] = event.amount as f32;
        row[event::CONTROLLER] = event.controller_id as f32;
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
        encode, encode_into, fact, EncodedObservationMut, ObservationEncodeError,
        ObservationEncoderConfig, ACTION_DIM, CARD_DIM, EVENT_DIM, FACT_DIM, PERMANENT_DIM,
        PLAYER_DIM, ZONE_DIM,
    };

    #[test]
    fn selector_encoding_is_exact_and_rejects_overflow() {
        use super::{action, MAX_ACTION_SELECTOR};
        let mut obs = sample_observation();
        obs.action_space.actions[0].mode_index = Some(0);
        obs.action_space.actions[1].ability_index = Some(MAX_ACTION_SELECTOR);
        let config = ObservationEncoderConfig::default();
        let encoded = encode(&obs, &config).unwrap();
        assert_eq!(encoded.actions[action::MODE], 1.0);
        assert_eq!(encoded.actions[action::ABILITY], 0.0);
        assert_eq!(encoded.actions[ACTION_DIM + action::ABILITY], 16777216.0);
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

    #[test]
    fn fact_capacity_rejects_overflow_and_preserves_every_row() {
        let mut obs = sample_observation();
        obs.agent_permanents[0].used_abilities = (0..140).collect();
        let config = ObservationEncoderConfig::default();
        assert!(matches!(
            encode(
                &obs,
                &ObservationEncoderConfig {
                    max_facts: 128,
                    ..config
                }
            ),
            Err(ObservationEncodeError::Capacity {
                field: "facts",
                capacity: 128,
                actual: 140,
            })
        ));
        let encoded = encode(&obs, &config).unwrap();
        assert_eq!(encoded.facts.len(), 512 * FACT_DIM);
        for (index, row) in encoded.facts.as_chunks::<FACT_DIM>().0.iter().enumerate() {
            if index < 140 {
                assert_eq!(row[fact::VALID], 1.0);
                assert_eq!(row[fact::KIND], 5.0);
                assert_eq!(row[fact::ID_HIGH + 1], index as f32);
                assert_eq!(row[fact::TIMESTAMP], 0.0);
                assert_eq!(row[fact::SOURCE_DEFINITION], 0.0);
            } else {
                assert!(row.iter().all(|value| *value == 0.0));
            }
        }
        let exact = encode(
            &obs,
            &ObservationEncoderConfig {
                max_facts: 140,
                ..config
            },
        )
        .unwrap();
        assert_eq!(exact.facts, encoded.facts[..140 * FACT_DIM]);
    }

    fn sample_observation() -> Observation {
        Observation {
            shared_effects: Some(Vec::new()),
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
                        name_definition: None,
                        name_index: None,
                        name_group: None,
                        ability_index: None,
                        continuation_index: None,
                    },
                    ActionOption {
                        action_type: ActionType::ChooseTarget,
                        focus: vec![444],
                        declared: None,
                        description: None,
                        mode_index: None,
                        name_definition: None,
                        name_index: None,
                        name_group: None,
                        ability_index: None,
                        continuation_index: None,
                    },
                    ActionOption {
                        action_type: ActionType::DeclareBlocker,
                        focus: vec![999],
                        declared: Some(true),
                        description: None,
                        mode_index: None,
                        name_definition: None,
                        name_index: None,
                        name_group: None,
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
                main_counts: Default::default(),
                unseen_counts: Default::default(),
                sideboard_counts: Default::default(),
                remaining_sideboard_counts: Default::default(),
                keeps_red_mana: false,
                turn_history: Default::default(),
                spells_cast: 0,
                extra_turns: 0,
                additional_combats: 0,
                combat_mana: 0,
                mana: [0; 6],
                lesson_mana: [0; 6],
                restricted_mana: Vec::new(),
                spend_effect_mana: Vec::new(),
                hand_only_until: None,
                no_maximum_hand_size: false,
                effects: Vec::new(),
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
                main_counts: Default::default(),
                unseen_counts: Default::default(),
                sideboard_counts: Default::default(),
                remaining_sideboard_counts: Default::default(),
                keeps_red_mana: false,
                turn_history: Default::default(),
                spells_cast: 0,
                extra_turns: 0,
                additional_combats: 0,
                combat_mana: 0,
                mana: [0; 6],
                lesson_mana: [0; 6],
                restricted_mana: Vec::new(),
                spend_effect_mana: Vec::new(),
                hand_only_until: None,
                no_maximum_hand_size: false,
                effects: Vec::new(),
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
                name_definition: None,
                name_index: None,
                name_group: None,
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
            from_battlefield_this_turn: false,
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
            colors: None,
            keyword_counter_timestamps: Default::default(),
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
            ..ObservationEncoderConfig::default()
        };

        let encoded = encode(&obs, &config).unwrap();

        // Agent and opponent player vectors are populated with turn one-hot slices.
        assert_eq!(encoded.agent_player[9 + 2], 1.0);
        assert_eq!(encoded.agent_player[14 + 5], 1.0);
        assert_eq!(encoded.opponent_player[9 + 2], 1.0);

        // Validity flags for populated card/permanent slots.
        assert_eq!(encoded.agent_cards[CARD_DIM - 1], 1.0);
        assert_eq!(encoded.agent_cards[CARD_DIM * 2 - 1], 1.0);
        assert_eq!(encoded.agent_cards[CARD_DIM * 3 - 1], 0.0);
        assert_eq!(encoded.agent_permanents[PERMANENT_DIM - 1], 1.0);
        assert_eq!(
            encoded.agent_permanents[PERMANENT_DIM - 1 + PERMANENT_DIM],
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
                name_definition: None,
                name_index: None,
                name_group: None,
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
        let row = (obs.agent_cards.len() + last) * CARD_DIM;
        assert_eq!(encoded.agent_cards[row..row + ZONE_DIM], [0.0; ZONE_DIM]);
        assert_eq!(encoded.agent_cards[row + 37], 1.0);
        assert_eq!(encoded.agent_cards[row + CARD_DIM - 1], 1.0);
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
            ..ObservationEncoderConfig::default()
        };

        let mut agent_player = vec![0.0; 1];
        let mut semantic_cards = vec![0.0; 4];
        let mut known_hand = vec![0.0; 8];
        let mut opponent_player = vec![0.0; PLAYER_DIM];
        let mut agent_cards = vec![0.0; 2 * CARD_DIM];
        let mut opponent_cards = vec![0.0; 2 * CARD_DIM];
        let mut agent_permanents = vec![0.0; PERMANENT_DIM];
        let mut opponent_permanents = vec![0.0; PERMANENT_DIM];
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
        let mut name_candidates = vec![0.0; config.name_candidates_len()];
        let mut relations = vec![0.0; config.relations_len()];
        let mut decklists = vec![0.0; config.decklists_len()];
        let mut effects = vec![0.0; config.effects_len()];
        let mut effect_details = vec![0.0; config.effect_details_len()];
        let mut effect_members = vec![0.0; config.effect_members_len()];
        let mut facts = vec![0.0; config.facts_len()];

        let out = EncodedObservationMut {
            name_candidates: &mut name_candidates,
            relations: &mut relations,
            semantic_cards: &mut semantic_cards,
            known_hand: &mut known_hand,
            decklists: &mut decklists,
            player_turn_state: &mut [],
            card_turn_state: &mut [],
            agent_player: &mut agent_player,
            opponent_player: &mut opponent_player,
            agent_cards: &mut agent_cards,
            opponent_cards: &mut opponent_cards,
            agent_permanents: &mut agent_permanents,
            opponent_permanents: &mut opponent_permanents,
            actions: &mut actions,
            events: &mut events,
            effects: &mut effects,
            effect_details: &mut effect_details,
            effect_members: &mut effect_members,
            facts: &mut facts,
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
            format!("invalid length for agent_player: expected {PLAYER_DIM}, got 1")
        );
    }
}
