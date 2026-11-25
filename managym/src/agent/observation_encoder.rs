//! Encode one acting view in checkpoint object coordinates. The row writer
//! selects dense compatibility buffers or immutable typed CSR records.

use super::observation_layout::{
    action, action_columns, card, decklist, event, fact, name, permanent, player, relation, stack,
    turn_state,
};
use super::observation_rows::{Cell, RowWriter, Table};
use Cell::{Integer, Scalar, Word};

use std::{
    collections::{BTreeMap, HashMap},
    fmt,
    sync::LazyLock,
};

use crate::{
    agent::observation::{
        CardData, CastPermissionData, ContinuousEffectData, EventData, Observation, PermanentData,
        PlayerData, RestrictedManaData, StackObjectData, StackObjectKindData, TurnData,
    },
    cards::tokens::name_value,
    flow::turn::{PhaseKind, StepKind},
    state::mana::{ManaCost, ManaRestriction},
};

// Feature widths. Column indices, names and value encodings live in
// `observation_layout`; open-ended vocabularies (counter kinds, subtypes, set
// characteristics, used abilities) are fact rows, not columns. Fact vocabulary
// ids are the learning tokens' values (`cards::tokens::name_value` of the same
// Rust or printed name). Dense destinations split a word into 16-bit halves.
pub const PLAYER_DIM: usize = player::WIDTH;
pub const CARD_DIM: usize = card::WIDTH;
pub const PERMANENT_DIM: usize = permanent::WIDTH;
pub const STACK_DIM: usize = stack::WIDTH;
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
    /// `Modification` id, a player's pooled mana with a spend effect or combat expiry
    /// under `POOLED_MANA`, or a permanent's ward price that is not a mana
    /// value under `WARD`.
    EffectProgram = 11,
    EffectiveColor = 12,
    KeywordCounterTimestamp = 13,
    ContinuationProgram = 14,
    AbilityResolution = 15,
    TriggeringAmount = 16,
}

impl FactKind {
    /// Divisor of this kind's two raw integer payloads in the logical v11 input.
    pub fn payload_divisor(self) -> u8 {
        match self {
            Self::Counter | Self::CopyException => 10,
            _ => 1,
        }
    }

    pub const ALL: [FactKind; 15] = [
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
        FactKind::AbilityResolution,
        FactKind::TriggeringAmount,
    ];
}

pub const ACTION_TYPE_DIM: usize = action::DECLARED;
/// Width of the current v11 action layout. Use the config for a selected layout.
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

/// Which object rows the encoder writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputLayout {
    /// Policy input v11: one row per game thing. Card rows hold cards outside
    /// the battlefield and stack; a permanent row holds its characteristics;
    /// the `stack` segment holds every spell and ability. Battlefield object
    /// and spell ids index their permanent and stack rows.
    OneToken,
    /// Deprecated policy inputs v9 and v10 (`legacy_input`), removed in 0.3.0.
    Legacy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObservationEncoderConfig {
    pub max_facts: usize,
    pub max_cards_per_player: usize,
    pub max_permanents_per_player: usize,
    /// Stack rows (input v11); overflow is a Capacity error.
    pub max_stack_objects: usize,
    pub layout: InputLayout,
    pub max_actions: usize,
    pub action_overflow: ActionOverflow,
    pub max_focus_objects: usize,
    pub max_events: usize,
    pub max_decision_records: usize,
    pub max_decision_reference_details: usize,
    pub max_decision_references: usize,
    pub max_decision_members: usize,
    pub policy_history: bool,
    /// Write `player_turn_state` and `card_turn_state` (inputs v10 and v11).
    /// Without them, an observation that needs that state is refused.
    pub turn_state: bool,
}

impl Default for ObservationEncoderConfig {
    fn default() -> Self {
        Self {
            max_cards_per_player: 60,
            max_permanents_per_player: 40,
            max_stack_objects: 32,
            layout: InputLayout::OneToken,
            max_actions: 512,
            action_overflow: ActionOverflow::Error,
            max_focus_objects: 2,
            max_events: 32,
            max_decision_records: 0,
            max_decision_reference_details: 0,
            max_decision_references: 0,
            max_decision_members: 0,
            policy_history: false,
            turn_state: true,
            max_facts: 512,
        }
    }
}

impl ObservationEncoderConfig {
    /// Two seats, each with a definition and position per retained fact.
    pub fn known_library_len(&self) -> usize {
        if self.layout == InputLayout::OneToken {
            4 * crate::state::knowledge::KNOWN_LIBRARY_ROWS
        } else {
            0
        }
    }

    /// Candidate rows share the action axis, but never consume object capacity.
    pub fn name_candidates_len(&self) -> usize {
        NAME_DIM * self.max_actions
    }

    pub fn relations_len(&self) -> usize {
        super::continuation::RELATION_ROWS * self.relation_dim()
    }

    pub fn relation_dim(&self) -> usize {
        match self.layout {
            InputLayout::Legacy => super::legacy_input::RELATION_DIM,
            InputLayout::OneToken => super::continuation::RELATION_DIM,
        }
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

    /// Columns in one seat's v10/v11 player turn state row.
    pub fn turn_state_dim(&self) -> usize {
        super::observation_layout::turn_state_columns(self.layout).len()
    }

    /// Two seat rows of v10/v11 player turn state; empty for v9.
    pub fn player_turn_state_len(&self) -> usize {
        if self.turn_state {
            2 * self.turn_state_dim()
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

    pub fn player_dim(&self) -> usize {
        self.layout.player_columns().len()
    }

    /// Permanent row width: v11's, or the deprecated v9/v10 rows'.
    pub fn permanent_dim(&self) -> usize {
        match self.layout {
            InputLayout::OneToken => PERMANENT_DIM,
            InputLayout::Legacy => super::legacy_input::PERMANENT_DIM,
        }
    }

    pub fn permanents_len(&self) -> usize {
        self.max_permanents_per_player * self.permanent_dim()
    }

    /// Stack rows the layout writes; none for v9/v10.
    pub fn stack_rows(&self) -> usize {
        match self.layout {
            InputLayout::OneToken => self.max_stack_objects,
            InputLayout::Legacy => 0,
        }
    }

    pub fn stack_len(&self) -> usize {
        self.stack_rows() * STACK_DIM
    }

    pub fn action_dim(&self) -> usize {
        action_columns(self.layout).len()
    }

    pub fn actions_len(&self) -> usize {
        self.max_actions * self.action_dim()
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

/// Deprecated dense encoded observation container; removed in 0.3.0.
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
    pub known_library: Vec<f32>,
    pub decklists: Vec<f32>,
    pub player_turn_state: Vec<f32>,
    pub card_turn_state: Vec<f32>,
    pub agent_player: Vec<f32>,
    pub opponent_player: Vec<f32>,
    pub agent_cards: Vec<f32>,
    pub opponent_cards: Vec<f32>,
    pub agent_permanents: Vec<f32>,
    pub opponent_permanents: Vec<f32>,
    pub stack: Vec<f32>,
    pub actions: Vec<f32>,
    pub events: Vec<f32>,
    pub decision_records: Vec<f32>,
    pub decision_reference_details: Vec<f32>,
    pub decision_references: Vec<f32>,
    pub decision_members: Vec<f32>,
    pub action_focus: Vec<i32>,
    pub agent_player_valid: Vec<f32>,
    pub opponent_player_valid: Vec<f32>,
    pub agent_cards_valid: Vec<f32>,
    pub opponent_cards_valid: Vec<f32>,
    pub agent_permanents_valid: Vec<f32>,
    pub opponent_permanents_valid: Vec<f32>,
    pub stack_valid: Vec<f32>,
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
    pub known_library: &'a mut [f32],
    pub decklists: &'a mut [f32],
    pub player_turn_state: &'a mut [f32],
    pub card_turn_state: &'a mut [f32],
    pub agent_player: &'a mut [f32],
    pub opponent_player: &'a mut [f32],
    pub agent_cards: &'a mut [f32],
    pub opponent_cards: &'a mut [f32],
    pub agent_permanents: &'a mut [f32],
    pub opponent_permanents: &'a mut [f32],
    pub stack: &'a mut [f32],
    pub actions: &'a mut [f32],
    pub events: &'a mut [f32],
    pub decision_records: &'a mut [f32],
    pub decision_reference_details: &'a mut [f32],
    pub decision_references: &'a mut [f32],
    pub decision_members: &'a mut [f32],
    pub action_focus: &'a mut [i32],
    pub agent_player_valid: &'a mut [f32],
    pub opponent_player_valid: &'a mut [f32],
    pub agent_cards_valid: &'a mut [f32],
    pub opponent_cards_valid: &'a mut [f32],
    pub agent_permanents_valid: &'a mut [f32],
    pub opponent_permanents_valid: &'a mut [f32],
    pub stack_valid: &'a mut [f32],
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
            known_library: &mut self.known_library,
            decklists: &mut self.decklists,
            player_turn_state: &mut self.player_turn_state,
            card_turn_state: &mut self.card_turn_state,
            agent_player: &mut self.agent_player,
            opponent_player: &mut self.opponent_player,
            agent_cards: &mut self.agent_cards,
            opponent_cards: &mut self.opponent_cards,
            agent_permanents: &mut self.agent_permanents,
            opponent_permanents: &mut self.opponent_permanents,
            stack: &mut self.stack,
            actions: &mut self.actions,
            events: &mut self.events,
            decision_records: &mut self.decision_records,
            decision_reference_details: &mut self.decision_reference_details,
            decision_references: &mut self.decision_references,
            decision_members: &mut self.decision_members,
            action_focus: &mut self.action_focus,
            agent_player_valid: &mut self.agent_player_valid,
            opponent_player_valid: &mut self.opponent_player_valid,
            agent_cards_valid: &mut self.agent_cards_valid,
            opponent_cards_valid: &mut self.opponent_cards_valid,
            agent_permanents_valid: &mut self.agent_permanents_valid,
            opponent_permanents_valid: &mut self.opponent_permanents_valid,
            stack_valid: &mut self.stack_valid,
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
    /// Missing permanent or spell identity, which input v11 requires.
    HistoricalObservation,
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
            Self::HistoricalObservation => write!(
                f,
                "policy input v11 requires permanent and spell identities; \
                 this historical observation encodes only as input v9/v10"
            ),
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
    let mut encoded = allocate(config);
    #[cfg(feature = "benchmark")]
    drop(allocation_timer);
    #[cfg(feature = "benchmark")]
    let _encode_timer = crate::benchmark_search::Timer::new("encoding/native_encode");
    encode_into(obs, config, encoded.buffers_mut())?;
    Ok(encoded)
}

/// A reusable one-observation workspace shared by dense and CSR writers.
pub fn allocate(config: &ObservationEncoderConfig) -> EncodedObservation {
    EncodedObservation {
        relations: vec![0.0; config.relations_len()],
        effects: vec![0.0; config.effects_len()],
        effect_details: vec![0.0; config.effect_details_len()],
        effect_members: vec![0.0; config.effect_members_len()],
        name_candidates: vec![0.0; config.name_candidates_len()],
        facts: vec![0.0; config.facts_len()],
        semantic_cards: vec![0.0; 2 * config.max_cards_per_player],
        known_hand: vec![0.0; 4 * config.max_cards_per_player],
        known_library: vec![0.0; config.known_library_len()],
        decklists: vec![0.0; config.decklists_len()],
        player_turn_state: vec![0.0; config.player_turn_state_len()],
        card_turn_state: vec![0.0; config.card_turn_state_len()],
        agent_player: vec![0.0; config.player_dim()],
        opponent_player: vec![0.0; config.player_dim()],
        agent_cards: vec![0.0; config.cards_len()],
        opponent_cards: vec![0.0; config.cards_len()],
        agent_permanents: vec![0.0; config.permanents_len()],
        opponent_permanents: vec![0.0; config.permanents_len()],
        stack: vec![0.0; config.stack_len()],
        actions: vec![0.0; config.actions_len()],
        events: vec![0.0; config.events_len()],
        decision_records: vec![
            0.0;
            config.max_decision_records * super::decision_memory::RECORD_DIM
        ],
        decision_reference_details: vec![
            0.0;
            config.max_decision_reference_details
                * super::decision_memory::DETAIL_DIM
        ],
        decision_references: vec![
            0.0;
            config.max_decision_references
                * super::decision_memory::REFERENCE_DIM
        ],
        decision_members: vec![
            0.0;
            config.max_decision_members * super::decision_memory::MEMBER_DIM
        ],
        action_focus: vec![-1; config.action_focus_len()],
        agent_player_valid: vec![0.0; 1],
        opponent_player_valid: vec![0.0; 1],
        agent_cards_valid: vec![0.0; config.max_cards_per_player],
        opponent_cards_valid: vec![0.0; config.max_cards_per_player],
        agent_permanents_valid: vec![0.0; config.max_permanents_per_player],
        opponent_permanents_valid: vec![0.0; config.max_permanents_per_player],
        stack_valid: vec![0.0; config.stack_rows()],
        actions_valid: vec![0.0; config.max_actions],
        events_valid: vec![0.0; config.max_events],
    }
}

/// Deprecated dense observation encoder; removed in 0.3.0.
pub fn encode_into(
    obs: &Observation,
    config: &ObservationEncoderConfig,
    out: EncodedObservationMut<'_>,
) -> Result<(), ObservationEncodeError> {
    encode_into_memory(obs, config, out, None)
}

/// Deprecated incremental dense observation encoder; removed in 0.3.0.
pub fn encode_into_incremental(
    obs: &Observation,
    config: &ObservationEncoderConfig,
    out: EncodedObservationMut<'_>,
    cursor: &mut super::decision_memory::MemoryEncoding,
) -> Result<(), ObservationEncodeError> {
    let result = encode_into_memory(obs, config, out, Some(cursor));
    if result.is_err() {
        *cursor = Default::default();
    }
    result
}

fn encode_into_memory(
    obs: &Observation,
    config: &ObservationEncoderConfig,
    mut out: EncodedObservationMut<'_>,
    cursor: Option<&mut super::decision_memory::MemoryEncoding>,
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
    validate_buffer_len(
        "known_library",
        out.known_library.len(),
        config.known_library_len(),
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
    validate_buffer_len("stack", out.stack.len(), config.stack_len())?;
    validate_buffer_len("stack_valid", out.stack_valid.len(), config.stack_rows())?;
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
    validate_buffer_len(
        "decision_records",
        out.decision_records.len(),
        config.max_decision_records * super::decision_memory::RECORD_DIM,
    )?;
    validate_buffer_len(
        "decision_reference_details",
        out.decision_reference_details.len(),
        config.max_decision_reference_details * super::decision_memory::DETAIL_DIM,
    )?;
    validate_buffer_len(
        "decision_references",
        out.decision_references.len(),
        config.max_decision_references * super::decision_memory::REFERENCE_DIM,
    )?;
    validate_buffer_len(
        "decision_members",
        out.decision_members.len(),
        config.max_decision_members * super::decision_memory::MEMBER_DIM,
    )?;
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
    validate_buffer_len("relations", out.relations.len(), config.relations_len())?;
    out.relations.fill(0.0);
    out.effects.fill(0.0);
    out.effect_details.fill(0.0);
    out.effect_members.fill(0.0);
    out.name_candidates.fill(0.0);
    out.facts.fill(0.0);
    out.semantic_cards.fill(0.0);
    out.known_hand.fill(0.0);
    out.known_library.fill(0.0);
    out.decklists.fill(0.0);
    out.player_turn_state.fill(0.0);
    out.card_turn_state.fill(0.0);
    out.agent_player.fill(0.0);
    out.opponent_player.fill(0.0);
    out.agent_cards.fill(0.0);
    out.opponent_cards.fill(0.0);
    out.agent_permanents.fill(0.0);
    out.opponent_permanents.fill(0.0);
    out.stack.fill(0.0);
    out.actions.fill(0.0);
    out.events.fill(0.0);
    out.action_focus.fill(-1);
    out.agent_player_valid.fill(0.0);
    out.opponent_player_valid.fill(0.0);
    out.agent_cards_valid.fill(0.0);
    out.opponent_cards_valid.fill(0.0);
    out.agent_permanents_valid.fill(0.0);
    out.opponent_permanents_valid.fill(0.0);
    out.stack_valid.fill(0.0);
    out.actions_valid.fill(0.0);
    out.events_valid.fill(0.0);
    encode_rows(obs, config, &mut out, cursor)
}

pub fn encode_rows(
    obs: &Observation,
    config: &ObservationEncoderConfig,
    out: &mut impl RowWriter,
    cursor: Option<&mut super::decision_memory::MemoryEncoding>,
) -> Result<(), ObservationEncodeError> {
    // Deprecated v9/v10 rows read the card lists they were trained on.
    let legacy;
    let obs = match config.layout {
        InputLayout::Legacy => {
            legacy = super::legacy_input::view(obs);
            &legacy
        }
        InputLayout::OneToken => {
            if obs.historical() {
                return Err(ObservationEncodeError::HistoricalObservation);
            }
            obs
        }
    };
    if !config.turn_state {
        if let Some(state) = input_v10_state(obs) {
            return Err(ObservationEncodeError::RequiresInputV10(state));
        }
    }
    if let Some(state) = unrepresentable_state(obs) {
        return Err(ObservationEncodeError::Unrepresentable(state));
    }
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
            "stack",
            if config.layout == InputLayout::OneToken {
                obs.stack_objects.len()
            } else {
                0
            },
            config.stack_rows(),
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
    if config.max_decision_records != 0
        || config.max_decision_reference_details != 0
        || config.max_decision_references != 0
        || config.max_decision_members != 0
    {
        if config.layout != InputLayout::OneToken
            || [
                config.max_decision_records,
                config.max_decision_reference_details,
                config.max_decision_references,
                config.max_decision_members,
            ]
            .contains(&0)
        {
            return Err(ObservationEncodeError::InvalidShared(
                "decision memory requires v11 and all companion capacities",
            ));
        }
        let log = &obs.decision_memory;
        for (field, actual, capacity) in [
            ("decision_records", log.len(), config.max_decision_records),
            (
                "decision_references",
                log.reference_count(),
                config.max_decision_references,
            ),
            (
                "decision_reference_details",
                log.detail_count(),
                config.max_decision_reference_details,
            ),
            (
                "decision_members",
                log.member_count(),
                config.max_decision_members,
            ),
        ] {
            if actual > capacity {
                return Err(ObservationEncodeError::Capacity {
                    field,
                    actual,
                    capacity,
                });
            }
        }
        super::decision_memory::write_suffix(log, cursor.unwrap_or(&mut Default::default()), out)?;
    }

    // Every link addresses the checkpoint's configured object coordinates, even
    // when its destination stores only live rows.
    let object_capacity = config
        .max_cards_per_player
        .saturating_add(config.max_permanents_per_player)
        .saturating_mul(2)
        .saturating_add(2)
        .saturating_add(config.stack_rows());
    check_capacity("object coordinates", object_capacity, i32::MAX as usize)?;
    let mut objects = HashMap::with_capacity(
        2 + obs.agent_cards.len()
            + obs.agent_sideboard.len()
            + obs.opponent_cards.len()
            + 2 * (obs.agent_permanents.len()
                + obs.opponent_permanents.len()
                + obs.stack_objects.len()),
    );
    let permanent_start = 2 + 2 * config.max_cards_per_player;
    let stack_start = permanent_start + 2 * config.max_permanents_per_player;
    let seats = [
        (
            &obs.agent,
            &obs.agent_cards,
            &obs.agent_permanents,
            Table::AgentPlayer,
            Table::AgentCards,
            Table::AgentPermanents,
        ),
        (
            &obs.opponent,
            &obs.opponent_cards,
            &obs.opponent_permanents,
            Table::OpponentPlayer,
            Table::OpponentCards,
            Table::OpponentPermanents,
        ),
    ];
    for (seat, &(player, cards, permanents, ..)) in seats.iter().enumerate() {
        objects.insert(player.id, seat as i32);
        for (i, card) in cards.iter().enumerate() {
            objects.insert(card.id, (2 + seat * config.max_cards_per_player + i) as i32);
        }
        for (i, permanent) in permanents.iter().enumerate() {
            let index = (permanent_start + seat * config.max_permanents_per_player + i) as i32;
            objects.insert(permanent.id, index);
            if config.layout == InputLayout::OneToken {
                if let Some(id) = permanent.object_id {
                    objects.insert(id, index);
                }
            }
        }
    }
    for (i, card) in obs.agent_sideboard.iter().enumerate() {
        objects.insert(card.candidate_id, (2 + obs.agent_cards.len() + i) as i32);
    }
    if config.layout == InputLayout::OneToken {
        for (i, object) in obs.stack_objects.iter().enumerate() {
            let index = (stack_start + i) as i32;
            objects.insert(object.stack_object_id, index);
            if let Some(spell) = &object.spell {
                objects.insert(spell.object_id, index);
            }
        }
    }
    let link = |id: Option<i32>| {
        Integer(
            id.and_then(|id| objects.get(&id))
                .map_or(0, |n| i64::from(*n) + 1),
        )
    };
    let ranks = timestamp_ranks(obs);
    let combat_damage_flags = |seats: &[usize]| {
        [obs.agent.player_index, obs.opponent.player_index]
            .map(|seat| flag(seats.contains(&(seat as usize))))
    };
    let context = CardContext {
        viewer_id: obs.agent.id,
        turn_number: obs.turn.turn_number,
    };
    for (seat, (player, cards, permanents, player_table, card_table, permanent_table)) in
        seats.into_iter().enumerate()
    {
        let mine = flag(seat == 0);
        let mut row = [Cell::ZERO; PLAYER_DIM];
        encode_player_features(player, &obs.turn, &mut row);
        row[player::DECKLISTS_OPEN] = flag(!obs.decklists_closed);
        // The deprecated v9/v10 rows end before the first-strike step flag.
        out.write(player_table, 0, 0, &row[..config.player_dim()])?;
        if config.turn_state {
            let row = turn_state_row(player, config.layout);
            out.write(
                Table::PlayerTurnState,
                0,
                seat,
                &row[..config.turn_state_dim()],
            )?;
        }
        for (slot, card) in cards.iter().enumerate() {
            let mut row = [Cell::ZERO; CARD_DIM];
            encode_card_features(card, mine, &context, &mut row);
            out.write(card_table, 0, slot, &row)?;
            out.write(
                Table::SemanticCards,
                seat,
                slot,
                &[Integer(i64::from(card.registry_key) + 1)],
            )?;
            if config.turn_state {
                out.write(
                    Table::CardTurnState,
                    seat,
                    slot,
                    &[flag(card.from_battlefield_this_turn)],
                )?;
            }
        }
        if seat == 0 {
            for (i, card) in obs.agent_sideboard.iter().enumerate() {
                let slot = cards.len() + i;
                let mut row = [Cell::ZERO; CARD_DIM];
                for column in [card::IS_MINE, card::OUTSIDE, card::VALID] {
                    row[column] = Integer(1);
                }
                out.write(card_table, 0, slot, &row)?;
                out.write(
                    Table::SemanticCards,
                    seat,
                    slot,
                    &[Integer(i64::from(card.registry_key) + 1)],
                )?;
                if config.turn_state {
                    out.write(Table::CardTurnState, seat, slot, &[Cell::ZERO])?;
                }
            }
        }
        for (slot, permanent) in permanents.iter().enumerate() {
            let mut row = [Cell::ZERO; PERMANENT_DIM];
            let row = &mut row[..config.permanent_dim()];
            encode_permanent_features(permanent, mine, config.layout, row);
            if config.layout == InputLayout::OneToken {
                [
                    row[permanent::COMBAT_DAMAGE_TO_ME],
                    row[permanent::COMBAT_DAMAGE_TO_OPPONENT],
                ] = combat_damage_flags(&permanent.combat_damage_to_players);
            }
            row[permanent::ATTACHED_TO] = link(permanent.attached_to);
            row[permanent::BLOCKING] = link(permanent.blocking);
            row[permanent::TIMESTAMP] = Integer(ranks[&permanent.timestamp()]);
            out.write(permanent_table, 0, slot, row)?;
        }
        check_capacity(
            "known hand",
            player.known_hand.len(),
            config.max_cards_per_player,
        )?;
        for (i, (definition, count)) in player.known_hand.iter().enumerate() {
            out.write(
                Table::KnownHand,
                seat,
                i,
                &[Integer(i64::from(*definition) + 1), Integer(*count as i64)],
            )?;
        }
        if config.layout == InputLayout::OneToken {
            check_capacity(
                "known library",
                player.known_library.len(),
                crate::state::knowledge::KNOWN_LIBRARY_ROWS,
            )?;
            for (i, (position, definition)) in player.known_library.iter().enumerate() {
                out.write(
                    Table::KnownLibrary,
                    seat,
                    i,
                    &[
                        Integer(i64::from(*definition) + 1),
                        Integer(*position as i64 + 1),
                    ],
                )?;
            }
        }
        let deck = decklist_rows(player);
        check_capacity(
            "decklist definitions",
            deck.len(),
            config.max_cards_per_player,
        )?;
        for (i, row) in deck.iter().enumerate() {
            out.write(Table::Decklists, seat, i, row)?;
        }
    }
    if config.layout == InputLayout::OneToken {
        for (slot, object) in obs.stack_objects.iter().enumerate() {
            let mut row = [Cell::ZERO; STACK_DIM];
            encode_stack_features(object, obs.agent.id, &mut row);
            [
                row[stack::COMBAT_DAMAGE_TO_ME],
                row[stack::COMBAT_DAMAGE_TO_OPPONENT],
            ] = combat_damage_flags(&object.combat_damage_to_players);
            row[stack::SOURCE] = link(object.source_permanent_id);
            out.write(Table::Stack, 0, slot, &row)?;
        }
    }
    encode_facts(obs, config.layout, &objects, &ranks, config.max_facts, out)?;
    super::shared_effects::encode(
        obs,
        &objects,
        &ranks,
        config.effect_details_len() / super::shared_effects::DETAIL_DIM,
        out,
    )?;
    let continuation_rows = super::continuation::write_rows(
        &obs.continuation,
        obs.agent.player_index as usize,
        &objects,
        config.relation_dim(),
        out,
    )?;

    if config.layout == InputLayout::OneToken {
        let mut next = continuation_rows;
        for source in obs.agent_permanents.iter().chain(&obs.opponent_permanents) {
            for &target in &source.damaged_permanents {
                check_capacity(
                    "damage relations",
                    next + 1,
                    super::continuation::RELATION_ROWS,
                )?;
                let mut row = [Cell::ZERO; relation::WIDTH];
                // A standalone pair owns its pooling group, separate from pending work.
                row[relation::FRAME] =
                    Integer((obs.continuation.len() + next - continuation_rows + 1) as i64);
                row[relation::KIND] = Integer(11);
                row[relation::SOURCE] = link(Some(source.id));
                row[relation::REFERENCE] = link(Some(target));
                row[relation::VALID] = Integer(1);
                out.write(Table::Relations, 0, next, &row)?;
                next += 1;
            }
        }
    }
    encode_actions(obs, config, out, &objects)?;
    if config.policy_history {
        let rows = match config.layout {
            InputLayout::OneToken => {
                super::policy_history::encode(obs, config.max_events, &objects)
            }
            InputLayout::Legacy => {
                super::legacy_input::history(obs, config.max_events, config.max_cards_per_player)
            }
        };
        for (i, row) in rows.into_iter().enumerate() {
            out.write(Table::Events, 0, i, &row.map(Integer))?;
        }
    } else {
        encode_events(&obs.recent_events, config.max_events, out)?;
    }
    Ok(())
}

pub(super) fn check_capacity(
    field: &'static str,
    actual: usize,
    capacity: usize,
) -> Result<(), ObservationEncodeError> {
    if actual > capacity {
        Err(ObservationEncodeError::Capacity {
            field,
            actual,
            capacity,
        })
    } else {
        Ok(())
    }
}

fn turn_state_row(player: &PlayerData, layout: InputLayout) -> [Cell; TURN_STATE_DIM] {
    let mut row = [Cell::ZERO; TURN_STATE_DIM];
    if layout == InputLayout::OneToken {
        let history = &player.turn_history;
        row[turn_state::LIFE_GAINS] = Integer(i64::from(history.life_gains));
        row[turn_state::LIFE_GAINED] = Integer(i64::from(history.life_gained));
        row[turn_state::MANA_SPENT_CASTING] = Integer(i64::from(history.mana_spent_casting));
        row[turn_state::CREATURES_DIED] = Integer(i64::from(history.creatures_died));
        row[turn_state::PERMANENTS_LEFT] = Integer(i64::from(history.permanents_left));
        row[turn_state::NONLAND_PERMANENTS_LEFT] =
            Integer(i64::from(history.nonland_permanents_left));
        row[turn_state::WARPED_THIS_TURN] = flag(history.warped_this_turn);
    }
    row[turn_state::LIFE_LOST] = Scalar(player.turn_history.life_lost as f32 / 20.0);
    row[turn_state::SPELLS_CAST] = Scalar(player.spells_cast as f32 / 10.0);
    for class in player
        .restricted_mana
        .iter()
        .filter(|class| class.except_nonartifact_spells)
    {
        for slot in 0..6 {
            let cell = &mut row[turn_state::NONARTIFACT_MANA_WHITE + slot];
            *cell = Scalar(cell.scalar() + class.mana[slot] as f32 / 10.0);
        }
    }
    row
}

/// Which v10-only state a definition's rules read, found by walking its
/// serialized value for the rules terms that ask for it.
#[derive(Clone, Copy, Debug, Default)]
struct TurnStateReads {
    life_lost: bool,
    from_battlefield: bool,
}

impl TurnStateReads {
    fn of(definition: &crate::state::card::ObjectDefinition) -> Self {
        fn walk(value: &serde_json::Value, reads: &mut TurnStateReads) {
            match value {
                serde_json::Value::Object(map) => {
                    for (key, value) in map {
                        match key.as_str() {
                            "OpponentLostLifeThisTurnAtLeast" | "LifeLostThisTurn" => {
                                reads.life_lost = true
                            }
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
pub(crate) fn representable(restriction: &ManaRestriction) -> bool {
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
fn decklist_rows(player: &PlayerData) -> Vec<[Cell; DECKLIST_DIM]> {
    let maps = [
        &player.main_counts,
        &player.unseen_counts,
        &player.sideboard_counts,
        &player.remaining_sideboard_counts,
    ];
    let mut rows: std::collections::BTreeMap<u32, [Cell; DECKLIST_DIM]> = Default::default();
    for (column, counts) in maps.into_iter().enumerate() {
        for (definition, count) in counts.iter().flatten() {
            let row = rows.entry(*definition).or_insert_with(|| {
                let mut row = [Cell::ZERO; DECKLIST_DIM];
                row[decklist::DEFINITION] = Integer(i64::from(*definition) + 1);
                row
            });
            row[column + decklist::MAIN] = Integer(*count as i64);
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

fn encode_player_features(player: &PlayerData, turn: &TurnData, out: &mut [Cell]) {
    out[player::LIFE] = Scalar(player.life as f32 / 20.0);
    out[player::IS_ACTIVE] = flag(player.is_active);

    for i in 0..ZONE_DIM {
        out[player::LIBRARY_COUNT + i] = Scalar(player.zone_counts[i] as f32 / 60.0);
    }

    let phase_index = usize::from(phase_index(turn.phase));
    if phase_index < PHASE_DIM {
        out[player::PHASE_BEGINNING + phase_index] = Integer(1);
    }

    let step_index = usize::from(step_index(turn.step));
    if step_index < STEP_DIM {
        out[player::STEP_UNTAP + step_index] = Integer(1);
    }

    out[player::GRAVEYARD_LESSONS] = Scalar(player.graveyard_lessons as f32 / 10.0);
    out[player::COMBAT_MANA] = Scalar(player.combat_mana as f32 / 10.0);
    for slot in 0..6 {
        out[player::MANA_WHITE + slot] = Scalar(player.mana[slot] as f32 / 10.0);
        out[player::LESSON_MANA_WHITE + slot] = Scalar(player.lesson_mana[slot] as f32 / 10.0);
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
                let cell = &mut out[player::RESTRICTED_LESSON_WHITE + 6 * index + slot];
                *cell = Scalar(cell.scalar() + class.mana[slot] as f32 / 10.0);
            }
        }
    }
    out[player::HAND_ONLY] = flag(player.hand_only_until.is_some());
    out[player::NO_MAXIMUM_HAND_SIZE] = flag(player.no_maximum_hand_size);
    out[player::STEP_FIRST_STRIKE_COMBAT_DAMAGE] =
        flag(turn.step == StepKind::FirstStrikeCombatDamage);
    out[player::TURN] = Scalar(turn.turn_number as f32 / 20.0);
    // Seat 0 is on the play.
    out[player::SEAT] = Integer(player.player_index as i64);
    out[player::KEEPS_RED_MANA] = flag(player.keeps_red_mana);
    let history = &player.turn_history;
    out[player::ATTACKED] = flag(history.attacked);
    out[player::PERMANENTS_SACRIFICED] = Scalar(history.permanents_sacrificed as f32 / 10.0);
    out[player::LAND_ENTERED] = flag(history.land_entered);
    out[player::CARDS_DRAWN] = Scalar(history.cards_drawn as f32 / 10.0);
    for (index, kind) in crate::flow::turn::Bending::ALL.into_iter().enumerate() {
        out[player::BENT_WATER + index] = flag(history.bent & kind as u8 != 0);
    }
    out[player::EXTRA_TURNS] = Integer(player.extra_turns as i64);
    out[player::ADDITIONAL_COMBATS] = Integer(player.additional_combats as i64);
}

/// Facts every card row of one observation reads.
struct CardContext {
    viewer_id: i32,
    turn_number: u32,
}

fn encode_card_features(card: &CardData, is_mine: Cell, context: &CardContext, out: &mut [Cell]) {
    let zone_index = (card.zone as i32 & 0xFF) as usize;
    if zone_index < ZONE_DIM {
        out[card::ZONE_LIBRARY + zone_index] = Integer(1);
    }
    out[card::IS_MINE] = is_mine;
    out[card::POWER] = Scalar(card.power as f32 / 10.0);
    out[card::TOUGHNESS] = Scalar(card.toughness as f32 / 10.0);
    out[card::MANA_VALUE] = Scalar(card.mana_cost.mana_value as f32 / 10.0);
    out[card::IS_LAND] = flag(card.card_types.is_land);
    out[card::IS_CREATURE] = flag(card.card_types.is_creature);
    out[card::IS_ARTIFACT] = flag(card.card_types.is_artifact);
    out[card::IS_ENCHANTMENT] = flag(card.card_types.is_enchantment);
    out[card::IS_PLANESWALKER] = flag(card.card_types.is_planeswalker);
    out[card::IS_BATTLE] = flag(card.card_types.is_battle);
    out[card::FLYING] = flag(card.keywords.flying);
    out[card::REACH] = flag(card.keywords.reach);
    out[card::HASTE] = flag(card.keywords.haste);
    out[card::VIGILANCE] = flag(card.keywords.vigilance);
    out[card::TRAMPLE] = flag(card.keywords.trample);
    out[card::FIRST_STRIKE] = flag(card.keywords.first_strike);
    out[card::DOUBLE_STRIKE] = flag(card.keywords.double_strike);
    out[card::DEATHTOUCH] = flag(card.keywords.deathtouch);
    out[card::LIFELINK] = flag(card.keywords.lifelink);
    out[card::DEFENDER] = flag(card.keywords.defender);
    out[card::MENACE] = flag(card.keywords.menace);
    out[card::FLASH] = flag(card.keywords.flash);
    out[card::IS_TOKEN] = flag(card.is_token);
    out[card::IS_ALLY] = flag(card.is_ally);
    out[card::IS_LESSON] = flag(card.is_lesson);
    out[card::HAS_WARD] =
        flag(card.ward_cost > 0 || !card.ward_program.is_empty() || card.ward_life > 0);
    out[card::WARD_COST] = Scalar(card.ward_cost as f32 / 10.0);
    out[card::HAS_KICKER] = flag(card.kicker_cost > 0);
    out[card::KICKER_COST] = Scalar(card.kicker_cost as f32 / 10.0);
    out[card::HEXPROOF] = flag(card.keywords.hexproof);
    out[card::INDESTRUCTIBLE] = flag(card.keywords.indestructible);
    out[card::IS_LEGENDARY] = flag(card.is_legendary);
    if let Some(permission) = card.cast_permissions.first() {
        out[card::PERMISSION] = Integer(1);
        out[card::PERMISSION_IS_MINE] = flag(permission.player_id == context.viewer_id);
        if let Some(cost) = &permission.cost {
            out[card::PERMISSION_HAS_COST] = Integer(1);
            out[card::PERMISSION_MANA_VALUE] = Scalar(cost.mana_value as f32 / 10.0);
        }
        out[card::PERMISSION_THIS_TURN] = flag(permission.this_turn);
        out[card::PERMISSION_OWN_TURN_ONLY] = flag(permission.own_turn_only);
        // Not yet usable: "cast it on a later turn" and this is that turn
        // or earlier.
        out[card::PERMISSION_NOT_YET] = flag(
            permission
                .after_turn
                .is_some_and(|turn| context.turn_number <= turn),
        );
        out[card::PERMISSION_HAS_SOURCE] = flag(permission.source_permanent_id.is_some());
        out[card::PERMISSION_FLASH] = flag(permission.flash);
        out[card::PERMISSION_ANY_MANA] = flag(permission.any_mana);
        out[card::PERMISSION_WATERBEND] = flag(permission.waterbend);
        out[card::PERMISSION_FORETOLD] = flag(permission.foretold);
        out[card::PERMISSION_THROUGH_NEXT_TURN] = flag(permission.through_next_turn);
    }
    if let Some(x) = card.x {
        out[card::HAS_X] = Integer(1);
        out[card::X] = Scalar(x as f32 / 10.0);
    }
    out[card::VALID] = Integer(1);
}

fn encode_permanent_features(
    permanent: &PermanentData,
    is_mine: Cell,
    layout: InputLayout,
    out: &mut [Cell],
) {
    out[permanent::IS_MINE] = is_mine;
    out[permanent::TAPPED] = flag(permanent.tapped);
    out[permanent::DAMAGE] = Scalar(permanent.damage as f32 / 10.0);
    out[permanent::SUMMONING_SICK] = flag(permanent.is_summoning_sick);
    out[permanent::PLUS_ONE_COUNTERS] = Scalar(permanent.plus1_counters as f32 / 10.0);
    out[permanent::CANT_BE_BLOCKED_THIS_TURN] = flag(permanent.cant_be_blocked_this_turn);
    out[permanent::POWER] = Scalar(permanent.power as f32 / 10.0);
    out[permanent::TOUGHNESS] = Scalar(permanent.toughness as f32 / 10.0);
    out[permanent::ANIMATED] = flag(permanent.is_animated);
    out[permanent::HAS_EXILE_LINK] = flag(permanent.has_exile_link);
    // Effective keywords (printed + until-EOT grants) — agents see what the
    // permanent can do right now, not just what's printed on the card.
    out[permanent::FLYING] = flag(permanent.keywords.flying);
    out[permanent::REACH] = flag(permanent.keywords.reach);
    out[permanent::HASTE] = flag(permanent.keywords.haste);
    out[permanent::FLASH] = flag(permanent.keywords.flash);
    out[permanent::VIGILANCE] = flag(permanent.keywords.vigilance);
    out[permanent::TRAMPLE] = flag(permanent.keywords.trample);
    out[permanent::FIRST_STRIKE] = flag(permanent.keywords.first_strike);
    out[permanent::DOUBLE_STRIKE] = flag(permanent.keywords.double_strike);
    out[permanent::DEATHTOUCH] = flag(permanent.keywords.deathtouch);
    out[permanent::LIFELINK] = flag(permanent.keywords.lifelink);
    out[permanent::DEFENDER] = flag(permanent.keywords.defender);
    out[permanent::MENACE] = flag(permanent.keywords.menace);
    out[permanent::HEXPROOF] = flag(permanent.keywords.hexproof);
    out[permanent::EXILE_IF_DIES_THIS_TURN] = flag(permanent.exile_if_dies_this_turn);
    out[permanent::INDESTRUCTIBLE] = flag(permanent.keywords.indestructible);
    out[permanent::ADDED_AVATAR] = flag(permanent.temp_subtypes.iter().any(|s| s == "Avatar"));
    out[permanent::ATTACKING] = flag(permanent.attacking);
    out[permanent::DEFINITION] = Integer(i64::from(permanent.registry_key) + 1);
    // Ability-use limits and granted firebending.
    out[permanent::EXHAUSTED_ABILITIES] = Integer(permanent.exhausted_abilities as i64);
    out[permanent::ONCE_EACH_TURN_USED] = Integer(permanent.once_each_turn_used as i64);
    out[permanent::GRANTED_FIREBENDING] = Scalar(permanent.granted_firebending as f32 / 10.0);
    // permanent::ATTACHED_TO and permanent::BLOCKING are written
    // once every object is indexed; permanent::TIMESTAMP with facts.
    out[permanent::GRANTED_LAND] = flag(permanent.is_granted_land);
    out[permanent::ABILITIES_LOST] = flag(permanent.abilities_lost);
    out[permanent::DEALT_DAMAGE_THIS_TURN] = flag(permanent.dealt_damage_this_turn);
    // Input v11: the characteristics only a card row held in v9/v10.
    if layout == InputLayout::OneToken {
        let card = &permanent.characteristics;
        out[permanent::ATTACKED_THIS_TURN] = flag(permanent.attacked_this_turn);
        out[permanent::ENTERED_THIS_TURN] = flag(permanent.entered_this_turn);
        out[permanent::MANA_VALUE] = Scalar(f32::from(permanent.mana_value()) / 10.0);
        out[permanent::IS_LAND] = flag(card.card_types.is_land);
        out[permanent::IS_CREATURE] = flag(card.card_types.is_creature);
        out[permanent::IS_ARTIFACT] = flag(card.card_types.is_artifact);
        out[permanent::IS_ENCHANTMENT] = flag(card.card_types.is_enchantment);
        out[permanent::IS_PLANESWALKER] = flag(card.card_types.is_planeswalker);
        out[permanent::IS_BATTLE] = flag(card.card_types.is_battle);
        out[permanent::IS_LEGENDARY] = flag(card.is_legendary);
        out[permanent::IS_TOKEN] = flag(permanent.is_token);
        out[permanent::IS_ALLY] = flag(card.is_ally);
        out[permanent::IS_LESSON] = flag(card.is_lesson);
        out[permanent::HAS_WARD] =
            flag(card.ward_cost > 0 || !card.ward_program.is_empty() || card.ward_life > 0);
        out[permanent::WARD_COST] = Scalar(card.ward_cost as f32 / 10.0);
        out[permanent::VALID] = Integer(1);
        out[permanent::FIRST_STRIKE_PARTICIPANT] = flag(permanent.first_strike_participant);
        out[permanent::FACE_DOWN] = flag(permanent.face_down);
    } else {
        out[super::legacy_input::PERMANENT_VALID] = Integer(1);
    }
}

/// Stack rows, top first, each indexed by its stack object id; a spell's card
/// id names the same row.
fn encode_stack_features(object: &StackObjectData, viewer_id: i32, out: &mut [Cell]) {
    out[stack::IS_MINE] = flag(object.controller_id == viewer_id);
    out[match object.kind {
        StackObjectKindData::Spell => stack::SPELL,
        StackObjectKindData::ActivatedAbility => stack::ACTIVATED_ABILITY,
        StackObjectKindData::TriggeredAbility => stack::TRIGGERED_ABILITY,
    }] = Integer(1);
    // A spell's definition is its card's; an ability's is its source's.
    let definition = object
        .spell
        .as_ref()
        .map_or(object.source_card_registry_key, |card| card.registry_key);
    out[stack::DEFINITION] = Integer(i64::from(definition) + 1);
    // stack::SOURCE is written once every object is indexed.
    out[stack::ABILITY] = Integer(object.ability_index.map_or(0, |index| index as i64 + 1));
    out[stack::MODE] = Integer(object.chosen_mode.map_or(0, |mode| mode as i64 + 1));
    let x = match &object.spell {
        Some(card) => card.has_announced_x.then_some(object.x),
        None => (object.x != 0).then_some(object.x),
    };
    if let Some(x) = x {
        out[stack::HAS_X] = Integer(1);
        out[stack::X] = Scalar(x as f32 / 10.0);
    }
    if let Some(value) = object.effective_mana_value() {
        out[stack::EFFECTIVE_MANA_VALUE] = Scalar(f32::from(value) / 10.0);
    }
    out[stack::KICKED] = flag(object.kicked);
    out[stack::FROM_PREPARED] = flag(object.from_prepared);
    out[stack::FLASHBACK] = flag(object.flashback);
    out[stack::MANA_SPENT] = Integer(i64::from(object.mana_spent));
    out[stack::SOURCE_POWER] = Scalar(object.source_power.unwrap_or_default() as f32 / 10.0);
    out[stack::PAID_AMOUNT] = Scalar(object.paid_amount.unwrap_or_default() as f32 / 10.0);
    out[stack::ABILITY_RESOLUTIONS] = Integer(i64::from(object.ability_resolutions));
    if let Some(card) = &object.spell {
        out[stack::CAST_FROM_GRAVEYARD] = flag(card.cast_from_graveyard);
        out[stack::IS_COPY] = flag(card.is_copy);
        out[stack::MANA_VALUE] = Scalar(card.mana_cost.mana_value as f32 / 10.0);
        out[stack::POWER] = Scalar(card.power as f32 / 10.0);
        out[stack::TOUGHNESS] = Scalar(card.toughness as f32 / 10.0);
        out[stack::IS_LAND] = flag(card.card_types.is_land);
        out[stack::IS_CREATURE] = flag(card.card_types.is_creature);
        out[stack::IS_ARTIFACT] = flag(card.card_types.is_artifact);
        out[stack::IS_ENCHANTMENT] = flag(card.card_types.is_enchantment);
        out[stack::IS_PLANESWALKER] = flag(card.card_types.is_planeswalker);
        out[stack::IS_BATTLE] = flag(card.card_types.is_battle);
        out[stack::IS_LEGENDARY] = flag(card.is_legendary);
        // Printed keywords and the ones a static grants it on the stack.
        let granted = object.granted_keywords.clone().unwrap_or_default().flags();
        for (column, (printed, granted)) in
            card.keywords.flags().into_iter().zip(granted).enumerate()
        {
            out[stack::FLYING + column] = flag(printed || granted);
        }
    }
    out[stack::VALID] = Integer(1);
}

/// The id of an `EffectProgram` on a player that states pooled mana with a
/// spend effect or combat expiry (`PooledManaClassData::program`).
pub const POOLED_MANA: &str = "PooledMana";

/// The id of an effect program that states a permanent's ward price when
/// the `ward_cost` column cannot ("Ward—Pay 7 life.").
pub const WARD: &str = "Ward";

/// Effect program labels for overlapping grants and consumed type allowances.
pub const PLAY_GRANT: &str = "CastPermission::Grant";
pub const PLAY_GRANT_SOURCE: &str = "CastPermission::GrantSource";
pub const SPENT_TYPE: &str = "CastPermission::SpentType";

/// One fact row before its object join and timestamp rank are known.
struct Fact {
    object: i32,
    kind: FactKind,
    id: i32,
    detail: i32,
    values: [i64; 2],
    effect: Option<(i64, i32, Option<i32>)>,
}

/// The facts `PlayerData`, `CardData` and `PermanentData` state as open
/// vocabularies, in object order (agent, opponent, agent cards, opponent
/// cards, agent permanents, opponent permanents).
fn facts(obs: &Observation, layout: InputLayout) -> Vec<Fact> {
    let mut out = Vec::new();
    let mut push = |object: i32, kind: FactKind, id: i32, value: i64| {
        out.push(Fact {
            object,
            kind,
            id,
            detail: 0,
            values: [value, 0],
            effect: None,
        })
    };
    for player in [&obs.agent, &obs.opponent] {
        for (kind, count) in &player.counters {
            push(
                player.id,
                FactKind::Counter,
                name_value(&format!("CounterKind::{kind}")),
                i64::from(*count),
            );
        }
    }
    // Counters on a visible card off the battlefield ("with a stash counter").
    if layout == InputLayout::OneToken {
        for card in obs.agent_cards.iter().chain(&obs.opponent_cards) {
            for (kind, count) in &card.counters {
                push(
                    card.id,
                    FactKind::Counter,
                    name_value(&format!("CounterKind::{kind}")),
                    i64::from(*count),
                );
            }
        }
    }
    let permanents = || obs.agent_permanents.iter().chain(&obs.opponent_permanents);
    for permanent in permanents() {
        let object = permanent.id;
        if permanent.plus1_counters > 0 {
            push(
                object,
                FactKind::Counter,
                name_value("CounterKind::plus_one_plus_one"),
                i64::from(permanent.plus1_counters),
            );
        }
        for (kind, count) in &permanent.counters {
            push(
                object,
                FactKind::Counter,
                name_value(&format!("CounterKind::{kind}")),
                i64::from(*count),
            );
        }
        if let Some(colors) = &permanent.colors {
            if colors.is_empty() {
                push(
                    object,
                    FactKind::EffectiveColor,
                    name_value("Color::Colorless"),
                    1,
                );
            }
            for color in colors {
                push(
                    object,
                    FactKind::EffectiveColor,
                    name_value(&format!("Color::{color}")),
                    1,
                );
            }
        }
        for subtype in &permanent.temp_subtypes {
            push(object, FactKind::AddedSubtype, name_value(subtype), 1);
        }
        if let Some(name) = &permanent.name {
            push(object, FactKind::SetName, name_value(name), 1);
        }
        for creature_type in permanent.creature_types.iter().flatten() {
            push(
                object,
                FactKind::SetCreatureType,
                name_value(creature_type),
                1,
            );
        }
        if permanent.transforms > 0 {
            push(object, FactKind::Transforms, 0, permanent.transforms as i64);
        }
        if layout == InputLayout::OneToken {
            for &(index, count) in &permanent.ability_resolutions {
                push(
                    object,
                    FactKind::AbilityResolution,
                    index as i32,
                    i64::from(count),
                );
            }
        }
        for index in &permanent.used_abilities {
            push(object, FactKind::UsedAbility, *index as i32, 1);
        }
    }
    let frame_controller = |frame: &super::continuation::ContinuationFrame| {
        if frame.controller == obs.agent.player_index as usize {
            obs.agent.id
        } else {
            obs.opponent.id
        }
    };
    for (index, frame) in obs.continuation.iter().enumerate() {
        let controller = frame_controller(frame);
        if layout == InputLayout::OneToken && frame.triggering_amount > 0 {
            push(
                controller,
                FactKind::TriggeringAmount,
                (index + 1) as i32,
                i64::from(frame.triggering_amount),
            );
        }
        if frame.source_transforms > 0 {
            push(
                frame.source.object.unwrap_or(controller),
                FactKind::Transforms,
                (index + 1) as i32,
                frame.source_transforms as i64,
            );
        }
    }
    for (index, frame) in obs.continuation.iter().enumerate() {
        let controller = frame_controller(frame);
        for program in [&frame.program, &frame.effect_execution] {
            for (ordinal, &(kind, value)) in program.iter().enumerate() {
                out.push(Fact {
                    object: controller,
                    kind: FactKind::ContinuationProgram,
                    id: (index + 1) as i32,
                    detail: value,
                    values: [kind as i64, ordinal as i64],
                    effect: None,
                });
            }
        }
        if let Some(attackers) = &frame.attackers {
            out.push(Fact {
                object: controller,
                kind: FactKind::CombatRestriction,
                id: (index + 1) as i32,
                detail: name_value(attackers),
                values: [0; 2],
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
            values: [0; 2],
            effect: Some((2, permanent.timestamp(), Some(copy.registry_key))),
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
                values: [0; 2],
                effect: Some((0, timestamp, None)),
            });
        }
    }
    // A ward price that is not a mana value: one program on its permanent.
    for permanent in permanents() {
        for (ordinal, &(kind, value)) in permanent.characteristics.ward_program.iter().enumerate() {
            out.push(Fact {
                object: permanent.id,
                kind: FactKind::EffectProgram,
                id: name_value(WARD),
                detail: value,
                values: [kind as i64, ordinal as i64],
                effect: None,
            });
        }
    }
    if layout == InputLayout::OneToken {
        permission_facts(obs, &mut out);
    }
    // Spend effects and combat expiry: one program per pooled class on its player.
    for player in [&obs.agent, &obs.opponent] {
        for class in &player.pooled_mana_classes {
            for (ordinal, &(kind, value)) in class.program.iter().enumerate() {
                out.push(Fact {
                    object: player.id,
                    kind: FactKind::EffectProgram,
                    id: name_value(POOLED_MANA),
                    detail: value,
                    values: [kind as i64, ordinal as i64],
                    effect: None,
                });
            }
        }
    }
    out
}

/// One play grant's terms as learning tokens. The TLA terms are a fixed
/// tuple; later terms follow it only when set, in a stable order (adventure,
/// source-controller, flashback), and the named [`PlayGrantExtension`]
/// closes the sequence, so a grant without later terms keeps the tokens it
/// had when those terms did not exist.
struct PlayGrantTerms<'a> {
    terms: PlayGrantBase<'a>,
    on_adventure: bool,
    follows_source_controller: bool,
    flashback: bool,
    extension: Option<PlayGrantExtension<'a>>,
}

type PlayGrantBase<'a> = (
    bool,
    &'a Option<ManaCost>,
    bool,
    bool,
    bool,
    Option<u32>,
    bool,
    bool,
    bool,
    bool,
    bool,
);

/// Grant terms added after the positional tuple, each omitted when unset.
#[derive(serde::Serialize)]
struct PlayGrantExtension<'a> {
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    prepared: bool,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    face_down: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    condition: Option<&'a str>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    play_lands: bool,
}

impl<'a> PlayGrantTerms<'a> {
    fn of(grant: &'a CastPermissionData, viewer: i32) -> Self {
        let extension =
            (grant.prepared || grant.face_down || grant.condition.is_some() || grant.play_lands)
                .then_some(PlayGrantExtension {
                    prepared: grant.prepared,
                    face_down: grant.face_down,
                    condition: grant.condition.as_deref(),
                    play_lands: grant.play_lands,
                });
        Self {
            terms: (
                grant.player_id == viewer,
                &grant.cost,
                grant.this_turn,
                grant.through_next_turn,
                grant.own_turn_only,
                grant.after_turn,
                grant.source_permanent_id.is_some(),
                grant.flash,
                grant.any_mana,
                grant.waterbend,
                grant.foretold,
            ),
            on_adventure: grant.on_adventure,
            follows_source_controller: grant.follows_source_controller,
            flashback: grant.flashback,
            extension,
        }
    }

    /// How many later terms follow the TLA tuple.
    fn tail(&self) -> usize {
        if self.extension.is_some() {
            4
        } else if self.flashback {
            3
        } else if self.follows_source_controller {
            2
        } else {
            usize::from(self.on_adventure)
        }
    }

    /// Whether the grant carries terms the card columns do not.
    fn has_tail(&self) -> bool {
        self.tail() > 0
    }
}

impl serde::Serialize for PlayGrantTerms<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeTuple;
        let tail = self.tail();
        if tail == 0 {
            return self.terms.serialize(serializer);
        }
        let mut tuple = serializer.serialize_tuple(tail + 1)?;
        tuple.serialize_element(&self.terms)?;
        for flag in [
            self.on_adventure,
            self.follows_source_controller,
            self.flashback,
        ]
        .into_iter()
        .take(tail.min(3))
        {
            tuple.serialize_element(&flag)?;
        }
        if let Some(extension) = &self.extension {
            tuple.serialize_element(extension)?;
        }
        tuple.end()
    }
}

fn permission_facts(obs: &Observation, out: &mut Vec<Fact>) {
    // Each occurrence starts at ordinal zero. The owner join identifies the
    // card or granting permanent; player terms are relative to this viewer.
    let mut program = |object, label, tokens: &[(u16, i32)]| {
        for (ordinal, &(kind, value)) in tokens.iter().enumerate() {
            out.push(Fact {
                object,
                kind: FactKind::EffectProgram,
                id: name_value(label),
                detail: value,
                values: [i64::from(kind), ordinal as i64],
                effect: None,
            });
        }
    };
    let pack = crate::cards::content_pack::default_content_pack;
    for card in obs.agent_cards.iter().chain(&obs.opponent_cards) {
        for grant in &card.cast_permissions {
            let overlapping = card.cast_permissions.len() > 1;
            let terms = PlayGrantTerms::of(grant, obs.agent.id);
            if !overlapping && grant.source_permanent_id.is_none() && !terms.has_tail() {
                continue;
            }
            let tokens =
                crate::cards::tokens::rules_tokens(&terms, pack()).expect("public grant terms");
            program(card.id, PLAY_GRANT, &tokens);
            // Adjacent source and card programs describe the same occurrence.
            // A singleton still carries its terms in the existing card columns.
            if let Some(source) = grant.source_permanent_id {
                program(source, PLAY_GRANT_SOURCE, &tokens);
            }
        }
    }
    for player in [&obs.agent, &obs.opponent] {
        for &(source, kind) in &player.spent_graveyard_types {
            let terms = (player.id == obs.agent.id, kind);
            program(
                source as i32,
                SPENT_TYPE,
                &crate::cards::tokens::rules_tokens(&terms, pack())
                    .expect("public graveyard allowance"),
            );
        }
    }
}

/// The rows of one copy exception; effect records themselves are
/// shared-effect rows.
fn copy_exception_facts(object: i32, effect: &ContinuousEffectData, out: &mut Vec<Fact>) {
    let kind = FactKind::CopyException;
    let id = name_value(&format!("Modification::{}", effect.modification));
    let duration = super::shared_effects::duration_code(&effect.duration).unwrap_or(2);
    let values = [
        i64::from(effect.power.or(effect.amount).unwrap_or(0)),
        i64::from(effect.toughness.unwrap_or(0)),
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
            values: [token_kind as i64, ordinal as i64],
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
fn timestamp_ranks(obs: &Observation) -> BTreeMap<i32, i64> {
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
        .map(|(rank, stamp)| (stamp, (rank + 1) as i64))
        .collect()
}

fn encode_facts(
    obs: &Observation,
    layout: InputLayout,
    object_to_index: &HashMap<i32, i32>,
    ranks: &BTreeMap<i32, i64>,
    capacity: usize,
    out: &mut impl RowWriter,
) -> Result<(), ObservationEncodeError> {
    let facts = facts(obs, layout);
    if facts.len() > capacity {
        return Err(ObservationEncodeError::Capacity {
            field: "facts",
            capacity,
            actual: facts.len(),
        });
    }
    for (slot, fact) in facts.iter().enumerate() {
        let mut row = [Cell::ZERO; FACT_DIM];
        row[fact::OBJECT] = Integer(
            object_to_index
                .get(&fact.object)
                .map_or(0, |index| i64::from(*index) + 1),
        );
        row[fact::KIND] = Integer(fact.kind as u8 as i64);
        row[fact::ID_HIGH] = Word(fact.id);
        row[fact::DETAIL_HIGH] = Word(fact.detail);
        row[fact::VALUE..fact::VALUE + 2].copy_from_slice(
            &fact
                .values
                .map(|n| Cell::Payload(n, fact.kind.payload_divisor())),
        );
        if let Some((duration, timestamp, source)) = fact.effect {
            row[fact::DURATION] = Integer(duration);
            row[fact::TIMESTAMP] = Integer(ranks[&timestamp]);
            row[fact::SOURCE_DEFINITION] = Integer(source.map_or(0, |key| i64::from(key) + 1));
        }
        row[fact::VALID] = Integer(1);
        out.write(Table::Facts, 0, slot, &row)?;
    }
    Ok(())
}

fn encode_actions(
    obs: &Observation,
    config: &ObservationEncoderConfig,
    out: &mut impl RowWriter,
    object_to_index: &HashMap<i32, i32>,
) -> Result<(), ObservationEncodeError> {
    let mut focus = vec![Integer(-1); config.max_focus_objects];
    for (action_index, action) in obs
        .action_space
        .actions
        .iter()
        .take(config.max_actions)
        .enumerate()
    {
        let mut row = [Cell::ZERO; ACTION_DIM];
        let row = &mut row[..config.action_dim()];
        let action_type_index = action.action_type as usize;
        if action_type_index < ACTION_TYPE_DIM {
            row[action_type_index] = Integer(1);
        }
        row[action::DECLARED] = flag(action.declared.unwrap_or(false));
        let mut name_row = [Cell::ZERO; NAME_DIM];
        match (action.name_index, action.name_group) {
            // A name is categorical; its position is never a numeric mode feature.
            (Some(index), _) => {
                name_row[name::NAME] = Integer(index as i64 + 1);
                name_row[name::DEFINITION] =
                    Integer(action.name_definition.map_or(0, |id| id as i64 + 1));
            }
            // A group is its public member span.
            (_, Some([first, last])) => {
                name_row[name::GROUP_FIRST] = Integer(first as i64 + 1);
                name_row[name::GROUP_LAST] = Integer(last as i64 + 1);
            }
            _ => {
                row[action::MODE] = Integer(action.mode_index.map_or(0, |index| index as i64 + 1));
            }
        }
        row[action::ABILITY] = Integer(action.ability_index.map_or(0, |index| index as i64 + 1));
        row[action::CONTINUATION] = Integer(
            action
                .continuation_index
                .map_or(0, |index| index as i64 + 1),
        );
        row[action::VALID] = Integer(1);
        if let Some(column) = row.get_mut(action::PAYS_DIRECTLY) {
            *column = flag(action.pays_directly);
        }
        if let (true, Some(x), Some([has_x, value])) = (
            action.announces_x,
            action.mode_index,
            row.get_mut(action::HAS_X..=action::X),
        ) {
            *has_x = Integer(1);
            *value = Scalar(x as f32 / 10.0);
        }
        if let Some([has_cost, value, quota]) =
            row.get_mut(action::HAS_ROUTE_COST..=action::QUOTA_TYPE)
        {
            *has_cost = flag(action.route_mana_value.is_some());
            *value = Scalar(f32::from(action.route_mana_value.unwrap_or_default()) / 10.0);
            *quota = Integer(i64::from(action.quota_type.unwrap_or_default()));
        }
        if let Some(profile) = row.get_mut(action::SPELL_PROFILE) {
            *profile = Integer(i64::from(action.spell_profile));
        }
        if let Some(route) = row.get_mut(action::PLAY_ROUTE) {
            *route = Integer(i64::from(action.play_route.unwrap_or_default()));
        }
        out.write(Table::Actions, 0, action_index, row)?;
        out.write(Table::NameCandidates, 0, action_index, &name_row)?;
        focus.fill(Integer(-1));
        for (target, id) in focus.iter_mut().zip(&action.focus) {
            *target = Integer(i64::from(object_to_index.get(id).copied().unwrap_or(-1)));
        }
        out.write(Table::ActionFocus, 0, action_index, &focus)?;
    }
    Ok(())
}

fn encode_events(
    events: &[EventData],
    max_events: usize,
    out: &mut impl RowWriter,
) -> Result<(), ObservationEncodeError> {
    // Presentation-only domain facts are additive to `recent_events`, but the
    // fixed policy ABI must remain byte-for-byte compatible.
    let learning_events = events
        .iter()
        .filter(|event| event.event_type <= 6)
        .collect::<Vec<_>>();
    let start_index = learning_events.len().saturating_sub(max_events);
    for (i, event) in learning_events.into_iter().skip(start_index).enumerate() {
        let mut row = [Cell::ZERO; EVENT_DIM];
        row[event::KIND] = Integer(event.event_type as i64);
        row[event::SOURCE_KIND] = Integer(event.source_kind as i64);
        row[event::SOURCE_ID] = Integer(event.source_id as i64);
        row[event::TARGET_KIND] = Integer(event.target_kind as i64);
        row[event::TARGET_ID] = Integer(event.target_id as i64);
        row[event::AMOUNT] = Integer(event.amount as i64);
        row[event::CONTROLLER] = Integer(event.controller_id as i64);
        out.write(Table::Events, 0, i, &row)?;
    }
    Ok(())
}

pub(super) fn phase_index(phase: PhaseKind) -> u8 {
    match phase {
        PhaseKind::Beginning => 0,
        PhaseKind::PrecombatMain => 1,
        PhaseKind::Combat => 2,
        PhaseKind::PostcombatMain => 3,
        PhaseKind::Ending => 4,
    }
}

pub(super) fn step_index(step: StepKind) -> u8 {
    match step {
        StepKind::Untap => 0,
        StepKind::Upkeep => 1,
        StepKind::Draw => 2,
        StepKind::Main => 3,
        StepKind::BeginningOfCombat => 4,
        StepKind::DeclareAttackers => 5,
        StepKind::DeclareBlockers => 6,
        StepKind::CombatDamage | StepKind::FirstStrikeCombatDamage => 7,
        StepKind::EndOfCombat => 8,
        StepKind::PostcombatMain => 9,
        StepKind::End => 10,
        StepKind::Cleanup => 11,
    }
}

fn flag(value: bool) -> Cell {
    Integer(i64::from(value))
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
        action, encode, encode_into, fact, permanent, EncodedObservationMut,
        ObservationEncodeError, ObservationEncoderConfig, ACTION_DIM, CARD_DIM, EVENT_DIM,
        FACT_DIM, PERMANENT_DIM, PLAYER_DIM, ZONE_DIM,
    };

    #[test]
    fn selector_encoding_is_exact_and_rejects_overflow() {
        use super::MAX_ACTION_SELECTOR;
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
    fn packed_rows_preserve_full_signed_words_and_event_ids() {
        use crate::agent::{observation_layout::configured_arrays, observation_rows::PackedRows};
        let mut obs = sample_observation();
        obs.agent_permanents[0].characteristics.ward_program =
            vec![(4, -16_777_217), (4, i32::MIN)];
        obs.recent_events[0].source_id = 16_777_217;
        let config = ObservationEncoderConfig::default();
        let arrays = configured_arrays(&config).unwrap();
        let mut batch = PackedRows::new(&arrays);
        batch.append(&obs, &config).unwrap();
        // Decode by the exported field order, independently of the dense projection.
        let words = |table: &str, name: &str| -> Vec<i32> {
            let mut offset = None;
            let mut width = 0;
            for field in arrays[table].packed_fields(table) {
                if field.name == name {
                    offset = Some(width);
                }
                width += if field.storage.dtype() == "uint8" {
                    1
                } else {
                    4
                };
            }
            let offset = offset.unwrap();
            batch.tables[table]
                .values
                .chunks_exact(width)
                .map(|row| i32::from_ne_bytes(row[offset..offset + 4].try_into().unwrap()))
                .collect()
        };
        assert_eq!(words("facts", "detail"), [-16_777_217, i32::MIN]);
        assert_eq!(words("events", "source_id")[0], 16_777_217);
        assert_eq!(batch.tables["facts"].offsets, [0, 2]);
        assert_eq!(batch.tables["card_turn_state"].offsets, [0, 1, 2]);
        assert_eq!(batch.tables["card_turn_state"].values, [0, 0]);
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
            decklists_closed: false,
            shared_effects: Some(Vec::new()),
            continuation: Vec::new(),
            game_over: false,
            won: false,
            turn: TurnData {
                creature_died_this_turn: false,
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
                        pays_directly: false,
                        announces_x: false,
                        spell_profile: 0,
                        route_mana_value: None,
                        quota_type: None,
                        play_route: None,
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
                        pays_directly: false,
                        announces_x: false,
                        spell_profile: 0,
                        route_mana_value: None,
                        quota_type: None,
                        play_route: None,
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
                        pays_directly: false,
                        announces_x: false,
                        spell_profile: 0,
                        route_mana_value: None,
                        quota_type: None,
                        play_route: None,
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
                known_library: Default::default(),
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
                pooled_mana_classes: Vec::new(),
                hand_only_until: None,
                spent_graveyard_types: vec![],
                counters: Default::default(),
                no_maximum_hand_size: false,
                effects: Vec::new(),
            },
            agent_sideboard: Vec::new(),
            agent_cards: vec![make_card(111, ZoneType::Hand, true, 2, 2, 1)],
            stack_objects: vec![],
            agent_permanents: vec![PermanentData {
                object_id: Some(112),
                owner_id: 10,
                characteristics: make_card(112, ZoneType::Battlefield, true, 3, 3, 2)
                    .characteristics,
                ..make_permanent(333, true)
            }],
            opponent: PlayerData {
                player_index: 1,
                id: 20,
                is_agent: false,
                is_active: false,
                life: 18,
                zone_counts: [39, 3, 1, 0, 0, 0, 0],
                graveyard_lessons: 0,
                known_hand: Default::default(),
                known_library: Default::default(),
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
                pooled_mana_classes: Vec::new(),
                hand_only_until: None,
                spent_graveyard_types: vec![],
                counters: Default::default(),
                no_maximum_hand_size: false,
                effects: Vec::new(),
            },
            opponent_cards: vec![make_card(221, ZoneType::Hand, false, 1, 1, 1)],
            opponent_permanents: vec![PermanentData {
                object_id: Some(445),
                owner_id: 20,
                characteristics: make_card(445, ZoneType::Battlefield, false, 2, 2, 2)
                    .characteristics,
                ..make_permanent(444, false)
            }],
            policy_history: Default::default(),
            decision_memory: Default::default(),
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
                pays_directly: false,
                announces_x: false,
                spell_profile: 0,
                route_mana_value: None,
                quota_type: None,
                play_route: None,
            })
            .collect();
        let config = ObservationEncoderConfig::default();
        let encoded = encode(&observation, &config).expect("attack decision encodes");
        assert_eq!(encoded.actions[action::DECLARED], 1.0);
        assert_eq!(encoded.actions[config.action_dim() + action::DECLARED], 0.0);
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
            characteristics: crate::agent::observation::CharacteristicsData {
                name: format!("Card {id}"),
                text_box: String::new(),
                power,
                toughness,
                is_legendary: false,
                is_ally: false,
                is_lesson: false,
                ward_cost: 0,
                ward_life: 0,
                ward_program: Vec::new(),
                kicker_cost: 0,
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
            },
            zone,
            owner_id: if mine { 10 } else { 20 },
            id,
            registry_key: id,
            is_token: false,
            face_down: false,
            cast_permissions: vec![],
            x: None,
            from_battlefield_this_turn: false,
            counters: Default::default(),
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
            characteristics: crate::agent::observation::CharacteristicsData {
                power: 2,
                toughness: 2,
                ..Default::default()
            },
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
            max_decision_records: 0,
            max_decision_reference_details: 0,
            max_decision_references: 0,
            max_decision_members: 0,
            policy_history: false,
            ..ObservationEncoderConfig::default()
        };

        let encoded = encode(&obs, &config).unwrap();

        // Agent and opponent player vectors are populated with turn one-hot slices.
        assert_eq!(encoded.agent_player[9 + 2], 1.0);
        assert_eq!(encoded.agent_player[14 + 5], 1.0);
        assert_eq!(encoded.opponent_player[9 + 2], 1.0);

        // Validity flags for populated card/permanent slots: the battlefield
        // object is its permanent row only.
        assert_eq!(encoded.agent_cards[CARD_DIM - 1], 1.0);
        assert_eq!(encoded.agent_cards[CARD_DIM * 2 - 1], 0.0);
        assert_eq!(encoded.agent_permanents[permanent::VALID], 1.0);
        assert_eq!(
            encoded.agent_permanents[permanent::VALID + PERMANENT_DIM],
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
        assert_eq!(encoded.agent_cards_valid, vec![1.0, 0.0, 0.0]);
        assert_eq!(encoded.opponent_cards_valid, vec![1.0, 0.0, 0.0]);
        assert_eq!(encoded.agent_permanents_valid, vec![1.0, 0.0]);
        assert_eq!(encoded.opponent_permanents_valid, vec![1.0, 0.0]);
        assert_eq!(encoded.actions_valid, vec![1.0, 1.0, 1.0]);
        assert_eq!(encoded.events_valid, vec![1.0, 1.0]);
        assert_eq!(encoded.events[0], 3.0);
        assert_eq!(encoded.events[EVENT_DIM], 2.0);
        // The battlefield card's characteristics are its permanent row's.
        assert_eq!(
            encoded.agent_permanents[permanent::MANA_VALUE],
            0.2,
            "mana value / 10"
        );
        assert_eq!(encoded.agent_permanents[permanent::IS_CREATURE], 1.0);
    }

    #[test]
    fn a_battlefield_object_and_a_spell_are_one_row_each() {
        use super::{stack, StackObjectData, StackObjectKindData};
        let mut obs = sample_observation();
        let spell = make_card(555, ZoneType::Stack, false, 0, 0, 3);
        obs.stack_objects = vec![
            StackObjectData {
                combat_damage_to_players: Vec::new(),
                source_power: None,
                paid_amount: None,
                mana_spent: 0,
                ability_resolutions: 0,
                description: None,
                stack_object_id: 600,
                kind: StackObjectKindData::Spell,
                controller_id: 20,
                source_card_registry_key: 555,
                source_permanent_id: None,
                chosen_mode: Some(1),
                ability_index: None,
                targets: Vec::new(),
                x: 0,
                granted_keywords: None,
                kicked: true,
                from_prepared: true,
                flashback: false,
                spell: Some(crate::agent::observation::SpellData::from_legacy_card(
                    spell,
                )),
            },
            StackObjectData {
                combat_damage_to_players: Vec::new(),
                source_power: None,
                paid_amount: None,
                mana_spent: 0,
                ability_resolutions: 0,
                description: None,
                stack_object_id: 601,
                kind: StackObjectKindData::ActivatedAbility,
                controller_id: 10,
                source_card_registry_key: 112,
                source_permanent_id: Some(333),
                chosen_mode: None,
                ability_index: Some(0),
                targets: Vec::new(),
                x: 0,
                granted_keywords: None,
                kicked: false,
                from_prepared: false,
                flashback: false,
                spell: None,
            },
        ];
        // Focus on the battlefield object's card id, the spell's card id,
        // its stack id and the ability.
        for (action, focus) in
            obs.action_space
                .actions
                .iter_mut()
                .zip([vec![112, 555], vec![600, 601], vec![445]])
        {
            action.focus = focus;
        }
        let config = ObservationEncoderConfig::default();
        let encoded = encode(&obs, &config).unwrap();
        let permanents = 2 + 2 * config.max_cards_per_player as i32;
        let stack_start = permanents + 2 * config.max_permanents_per_player as i32;
        assert_eq!(&encoded.action_focus[0..2], &[permanents, stack_start]);
        assert_eq!(&encoded.action_focus[2..4], &[stack_start, stack_start + 1]);
        assert_eq!(
            encoded.action_focus[4],
            permanents + config.max_permanents_per_player as i32
        );
        assert_eq!(encoded.stack_valid[..3], [1.0, 1.0, 0.0]);
        let spell_row = &encoded.stack[..super::STACK_DIM];
        assert_eq!(spell_row[stack::IS_MINE], 0.0);
        assert_eq!(spell_row[stack::SPELL], 1.0);
        assert_eq!(spell_row[stack::DEFINITION], 556.0);
        assert_eq!(spell_row[stack::MODE], 2.0);
        assert_eq!(spell_row[stack::KICKED], 1.0);
        assert_eq!(spell_row[stack::FROM_PREPARED], 1.0);
        assert_eq!(spell_row[stack::FLASHBACK], 0.0);
        assert_eq!(spell_row[stack::IS_CREATURE], 1.0);
        let ability_row = &encoded.stack[super::STACK_DIM..2 * super::STACK_DIM];
        assert_eq!(ability_row[stack::IS_MINE], 1.0);
        assert_eq!(ability_row[stack::ACTIVATED_ABILITY], 1.0);
        assert_eq!(ability_row[stack::DEFINITION], 113.0);
        assert_eq!(ability_row[stack::ABILITY], 1.0);
        assert_eq!(ability_row[stack::SOURCE], (permanents + 1) as f32);
        // No card row holds either object.
        assert_eq!(encoded.agent_cards_valid.iter().sum::<f32>(), 1.0);
        assert_eq!(encoded.opponent_cards_valid.iter().sum::<f32>(), 1.0);
    }

    #[test]
    fn the_legacy_layout_lists_battlefield_objects_and_spells_as_cards() {
        use super::InputLayout;
        let obs = sample_observation();
        let config = ObservationEncoderConfig {
            layout: InputLayout::Legacy,
            turn_state: false,
            ..ObservationEncoderConfig::default()
        };
        let encoded = encode(&obs, &config).unwrap();
        assert_eq!(&encoded.agent_cards_valid[..3], &[1.0, 1.0, 0.0]);
        assert_eq!(
            encoded.agent_permanents.len(),
            config.max_permanents_per_player * super::super::legacy_input::PERMANENT_DIM
        );
        assert!(encoded.stack.is_empty());
        // The battlefield card row keeps its own object index.
        let focus = encode(
            &Observation {
                action_space: ActionSpaceData {
                    actions: vec![ActionOption {
                        focus: vec![112],
                        ..obs.action_space.actions[0].clone()
                    }],
                    ..obs.action_space.clone()
                },
                ..obs.clone()
            },
            &config,
        )
        .unwrap()
        .action_focus[0];
        assert_eq!(focus, 3);
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
                pays_directly: false,
                announces_x: false,
                spell_profile: 0,
                route_mana_value: None,
                quota_type: None,
                play_route: None,
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
            max_decision_records: 0,
            max_decision_reference_details: 0,
            max_decision_references: 0,
            max_decision_members: 0,
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
        let mut player_turn_state = vec![0.0; config.player_turn_state_len()];
        let mut card_turn_state = vec![0.0; config.card_turn_state_len()];
        let mut stack = vec![0.0; config.stack_len()];
        let mut stack_valid = vec![0.0; config.stack_rows()];
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
            known_library: &mut [0.0; 80],
            decklists: &mut decklists,
            player_turn_state: &mut player_turn_state,
            card_turn_state: &mut card_turn_state,
            agent_player: &mut agent_player,
            opponent_player: &mut opponent_player,
            agent_cards: &mut agent_cards,
            opponent_cards: &mut opponent_cards,
            agent_permanents: &mut agent_permanents,
            opponent_permanents: &mut opponent_permanents,
            stack: &mut stack,
            stack_valid: &mut stack_valid,
            actions: &mut actions,
            events: &mut events,
            decision_records: &mut [],
            decision_reference_details: &mut [],
            decision_references: &mut [],
            decision_members: &mut [],
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
