// PyO3's #[pymethods] macro triggers false-positive `useless_conversion`
// warnings in generated wrappers under strict clippy settings.
#![allow(clippy::useless_conversion)]
#![allow(unexpected_cfgs)]

#[cfg(feature = "python")]
use crate::state::player::{ManaFloats, ManaPayment};
#[cfg(feature = "python")]
use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

#[cfg(feature = "python")]
use pyo3::{
    buffer::PyBuffer,
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyDict, PyList, PyModule},
};
#[cfg(feature = "python")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "python")]
use serde_json::{json, Value};

#[cfg(feature = "python")]
use crate::{
    agent::{
        action::{ActionSpaceKind, ActionType, AgentError},
        env::{Env, HandSnapshot, PreparedPossibleWorldMaterializer},
        observation::{
            ActionOption, ActionSpaceData, CardData, CardTypeData, CastPermissionData,
            CharacteristicsData, ContinuousEffectData, CopyData, EventData, EventEntityKind,
            EventType, KeywordData, Observation, PermanentData, PlayerData, PooledManaClassData,
            RestrictedManaData, SideboardCardData, StackObjectData, StackObjectKindData,
            StackTargetData, StackTargetKindData, TurnData,
        },
        observation_encoder::ObservationEncoderConfig,
        shared_effects,
        structured_offer::{OfferId as StructuredOfferId, OfferSubmission, StructuredOfferSet},
        vector_env::ObservationRead,
    },
    decision::Command as SemanticCommand,
    experience::{
        Command as ExperienceCommand, CommandId, MatchId, OfferId as ExperienceOfferId, PromptId,
        Revision,
    },
    flow::turn::{PhaseKind, PlayerTurnHistory, StepKind},
    python::convert::info_dict_to_pydict,
    search_state::SearchStateWitness,
    state::{
        deck::Deck,
        mana::ManaCost,
        player::{DecklistDisclosure, PlayerConfig},
        zone::ZoneType,
    },
};

#[cfg(feature = "python")]
#[pymethods]
impl HandSnapshot {
    fn __repr__(&self) -> String {
        format!(
            "HandSnapshot(player={}, game_id={}, seed={}, revision={}, cards={:?})",
            self.player, self.game_id, self.seed, self.revision, self.cards
        )
    }
}

#[cfg(feature = "python")]
pyo3::create_exception!(_managym, PyAgentError, PyRuntimeError);

#[cfg(feature = "python")]
pub(crate) fn map_agent_err(err: AgentError) -> PyErr {
    PyAgentError::new_err(err.to_string())
}

#[cfg(feature = "python")]
pyo3::create_exception!(_managym, ObservationCapacity, PyValueError);

/// The one Python form of an encoding failure: `ObservationCapacity` when the
/// observation does not fit its buffers, `ValueError` otherwise. The message
/// is the native one.
#[cfg(feature = "python")]
pub(crate) fn map_encode_err(
    err: crate::agent::observation_encoder::ObservationEncodeError,
) -> PyErr {
    use crate::agent::observation_encoder::ObservationEncodeError;
    match err {
        ObservationEncodeError::Capacity { .. } => ObservationCapacity::new_err(err.to_string()),
        _ => PyValueError::new_err(err.to_string()),
    }
}

#[cfg(feature = "python")]
pub(super) fn warn_deprecated(py: Python<'_>, message: &str) -> PyResult<()> {
    py.import_bound("warnings")?.call_method1(
        "warn",
        (
            message,
            py.get_type_bound::<pyo3::exceptions::PyDeprecationWarning>(),
            2,
        ),
    )?;
    Ok(())
}

#[cfg(feature = "python")]
fn numpy_array<'py, T: pyo3::buffer::Element>(
    py: Python<'py>,
    empty: &Bound<'py, PyAny>,
    data: &[T],
    shape: &[usize],
    dtype: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let array = empty.call1((shape.to_vec(), dtype))?;
    PyBuffer::<T>::get_bound(&array)?.copy_from_slice(py, data)?;
    Ok(array)
}

#[cfg(feature = "python")]
fn encoded_to_dict<'py>(
    py: Python<'py>,
    encoded: crate::agent::observation_encoder::EncodedObservation,
    config: &ObservationEncoderConfig,
) -> PyResult<Bound<'py, PyDict>> {
    use crate::agent::{continuation, decision_memory, observation_encoder as enc};
    let empty = py.import_bound("numpy")?.getattr("empty")?;
    let dict = PyDict::new_bound(py);
    let put = |name: &str, data: &[f32], shape: &[usize]| {
        dict.set_item(name, numpy_array(py, &empty, data, shape, "float32")?)
    };

    put(
        "name_candidates",
        &encoded.name_candidates,
        &[config.max_actions, enc::NAME_DIM],
    )?;
    put(
        "effects",
        &encoded.effects,
        &[shared_effects::EFFECT_ROWS, shared_effects::EFFECT_DIM],
    )?;
    put(
        "effect_details",
        &encoded.effect_details,
        &[shared_effects::DETAIL_ROWS, shared_effects::DETAIL_DIM],
    )?;
    put(
        "effect_members",
        &encoded.effect_members,
        &[shared_effects::MEMBER_ROWS, shared_effects::MEMBER_DIM],
    )?;
    if config.turn_state {
        put(
            "player_turn_state",
            &encoded.player_turn_state,
            &[2, enc::TURN_STATE_DIM],
        )?;
        put(
            "card_turn_state",
            &encoded.card_turn_state,
            &[2, config.max_cards_per_player],
        )?;
    }
    put("facts", &encoded.facts, &[config.max_facts, enc::FACT_DIM])?;
    put(
        "relations",
        &encoded.relations,
        &[continuation::RELATION_ROWS, continuation::RELATION_DIM],
    )?;
    put("agent_player", &encoded.agent_player, &[1, enc::PLAYER_DIM])?;
    put(
        "opponent_player",
        &encoded.opponent_player,
        &[1, enc::PLAYER_DIM],
    )?;
    let cards = [config.max_cards_per_player, enc::CARD_DIM];
    put("agent_cards", &encoded.agent_cards, &cards)?;
    put("opponent_cards", &encoded.opponent_cards, &cards)?;
    let permanents = [config.max_permanents_per_player, config.permanent_dim()];
    put("agent_permanents", &encoded.agent_permanents, &permanents)?;
    put(
        "opponent_permanents",
        &encoded.opponent_permanents,
        &permanents,
    )?;
    if config.layout == enc::InputLayout::OneToken {
        put(
            "stack",
            &encoded.stack,
            &[config.max_stack_objects, enc::STACK_DIM],
        )?;
        put(
            "stack_valid",
            &encoded.stack_valid,
            &[config.max_stack_objects],
        )?;
    }
    put(
        "actions",
        &encoded.actions,
        &[config.max_actions, config.action_dim()],
    )?;
    if config.max_decision_records > 0 {
        put(
            "decision_records",
            &encoded.decision_records,
            &[config.max_decision_records, decision_memory::RECORD_DIM],
        )?;
        put(
            "decision_reference_details",
            &encoded.decision_reference_details,
            &[
                config.max_decision_reference_details,
                decision_memory::DETAIL_DIM,
            ],
        )?;
        put(
            "decision_references",
            &encoded.decision_references,
            &[
                config.max_decision_references,
                decision_memory::REFERENCE_DIM,
            ],
        )?;
        put(
            "decision_members",
            &encoded.decision_members,
            &[config.max_decision_members, decision_memory::MEMBER_DIM],
        )?;
    }
    put(
        "events",
        &encoded.events,
        &[config.max_events, config.event_dim()],
    )?;
    dict.set_item(
        "action_focus",
        numpy_array(
            py,
            &empty,
            &encoded.action_focus,
            &[config.max_actions, config.max_focus_objects],
            "int32",
        )?,
    )?;
    put("agent_player_valid", &encoded.agent_player_valid, &[1])?;
    put(
        "opponent_player_valid",
        &encoded.opponent_player_valid,
        &[1],
    )?;
    put(
        "agent_cards_valid",
        &encoded.agent_cards_valid,
        &[config.max_cards_per_player],
    )?;
    put(
        "opponent_cards_valid",
        &encoded.opponent_cards_valid,
        &[config.max_cards_per_player],
    )?;
    put(
        "agent_permanents_valid",
        &encoded.agent_permanents_valid,
        &[config.max_permanents_per_player],
    )?;
    put(
        "opponent_permanents_valid",
        &encoded.opponent_permanents_valid,
        &[config.max_permanents_per_player],
    )?;
    put(
        "actions_valid",
        &encoded.actions_valid,
        &[config.max_actions],
    )?;
    put("events_valid", &encoded.events_valid, &[config.max_events])?;
    put(
        "semantic_cards",
        &encoded.semantic_cards,
        &[2, config.max_cards_per_player],
    )?;
    if config.known_library_len() > 0 {
        put(
            "known_library",
            &encoded.known_library,
            &[2, crate::state::knowledge::KNOWN_LIBRARY_ROWS, 2],
        )?;
    }
    put(
        "known_hand",
        &encoded.known_hand,
        &[2, config.max_cards_per_player, 2],
    )?;
    put(
        "decklists",
        &encoded.decklists,
        &[2, config.max_cards_per_player, enc::DECKLIST_DIM],
    )?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "ZoneEnum", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum ZoneEnum {
    Library = 0,
    Hand = 1,
    Battlefield = 2,
    Graveyard = 3,
    Stack = 4,
    Exile = 5,
    Command = 6,
}

#[cfg(feature = "python")]
#[pymethods]
impl ZoneEnum {
    #[classattr]
    const LIBRARY: Self = Self::Library;
    #[classattr]
    const HAND: Self = Self::Hand;
    #[classattr]
    const BATTLEFIELD: Self = Self::Battlefield;
    #[classattr]
    const GRAVEYARD: Self = Self::Graveyard;
    #[classattr]
    const STACK: Self = Self::Stack;
    #[classattr]
    const EXILE: Self = Self::Exile;
    #[classattr]
    const COMMAND: Self = Self::Command;

    fn __int__(&self) -> i32 {
        *self as i32
    }

    fn __index__(&self) -> i32 {
        *self as i32
    }
}

#[cfg(feature = "python")]
impl From<ZoneType> for ZoneEnum {
    fn from(value: ZoneType) -> Self {
        match value {
            ZoneType::Library => Self::Library,
            ZoneType::Hand => Self::Hand,
            ZoneType::Battlefield => Self::Battlefield,
            ZoneType::Graveyard => Self::Graveyard,
            ZoneType::Stack => Self::Stack,
            ZoneType::Exile => Self::Exile,
            ZoneType::Command => Self::Command,
        }
    }
}

#[cfg(feature = "python")]
impl From<ZoneEnum> for ZoneType {
    fn from(value: ZoneEnum) -> Self {
        match value {
            ZoneEnum::Library => Self::Library,
            ZoneEnum::Hand => Self::Hand,
            ZoneEnum::Battlefield => Self::Battlefield,
            ZoneEnum::Graveyard => Self::Graveyard,
            ZoneEnum::Stack => Self::Stack,
            ZoneEnum::Exile => Self::Exile,
            ZoneEnum::Command => Self::Command,
        }
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "PhaseEnum", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum PhaseEnum {
    Beginning = 0,
    PrecombatMain = 1,
    Combat = 2,
    PostcombatMain = 3,
    Ending = 4,
}

#[cfg(feature = "python")]
#[pymethods]
impl PhaseEnum {
    #[classattr]
    const BEGINNING: Self = Self::Beginning;
    #[classattr]
    const PRECOMBAT_MAIN: Self = Self::PrecombatMain;
    #[classattr]
    const COMBAT: Self = Self::Combat;
    #[classattr]
    const POSTCOMBAT_MAIN: Self = Self::PostcombatMain;
    #[classattr]
    const ENDING: Self = Self::Ending;

    fn __int__(&self) -> i32 {
        *self as i32
    }

    fn __index__(&self) -> i32 {
        *self as i32
    }
}

#[cfg(feature = "python")]
impl From<PhaseKind> for PhaseEnum {
    fn from(value: PhaseKind) -> Self {
        match value {
            PhaseKind::Beginning => Self::Beginning,
            PhaseKind::PrecombatMain => Self::PrecombatMain,
            PhaseKind::Combat => Self::Combat,
            PhaseKind::PostcombatMain => Self::PostcombatMain,
            PhaseKind::Ending => Self::Ending,
        }
    }
}

#[cfg(feature = "python")]
impl From<PhaseEnum> for PhaseKind {
    fn from(value: PhaseEnum) -> Self {
        match value {
            PhaseEnum::Beginning => Self::Beginning,
            PhaseEnum::PrecombatMain => Self::PrecombatMain,
            PhaseEnum::Combat => Self::Combat,
            PhaseEnum::PostcombatMain => Self::PostcombatMain,
            PhaseEnum::Ending => Self::Ending,
        }
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "StepEnum", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum StepEnum {
    BeginningUntap = 0,
    BeginningUpkeep = 1,
    BeginningDraw = 2,
    PrecombatMainStep = 3,
    CombatBegin = 4,
    CombatDeclareAttackers = 5,
    CombatDeclareBlockers = 6,
    CombatDamage = 7,
    CombatEnd = 8,
    PostcombatMainStep = 9,
    EndingEnd = 10,
    EndingCleanup = 11,
}

#[cfg(feature = "python")]
#[pymethods]
impl StepEnum {
    #[classattr]
    const BEGINNING_UNTAP: Self = Self::BeginningUntap;
    #[classattr]
    const BEGINNING_UPKEEP: Self = Self::BeginningUpkeep;
    #[classattr]
    const BEGINNING_DRAW: Self = Self::BeginningDraw;
    #[classattr]
    const PRECOMBAT_MAIN_STEP: Self = Self::PrecombatMainStep;
    #[classattr]
    const COMBAT_BEGIN: Self = Self::CombatBegin;
    #[classattr]
    const COMBAT_DECLARE_ATTACKERS: Self = Self::CombatDeclareAttackers;
    #[classattr]
    const COMBAT_DECLARE_BLOCKERS: Self = Self::CombatDeclareBlockers;
    #[classattr]
    const COMBAT_DAMAGE: Self = Self::CombatDamage;
    #[classattr]
    const COMBAT_END: Self = Self::CombatEnd;
    #[classattr]
    const POSTCOMBAT_MAIN_STEP: Self = Self::PostcombatMainStep;
    #[classattr]
    const ENDING_END: Self = Self::EndingEnd;
    #[classattr]
    const ENDING_CLEANUP: Self = Self::EndingCleanup;

    fn __int__(&self) -> i32 {
        *self as i32
    }

    fn __index__(&self) -> i32 {
        *self as i32
    }
}

#[cfg(feature = "python")]
impl From<StepKind> for StepEnum {
    fn from(value: StepKind) -> Self {
        match value {
            StepKind::Untap => Self::BeginningUntap,
            StepKind::Upkeep => Self::BeginningUpkeep,
            StepKind::Draw => Self::BeginningDraw,
            StepKind::Main => Self::PrecombatMainStep,
            StepKind::BeginningOfCombat => Self::CombatBegin,
            StepKind::DeclareAttackers => Self::CombatDeclareAttackers,
            StepKind::DeclareBlockers => Self::CombatDeclareBlockers,
            StepKind::CombatDamage => Self::CombatDamage,
            StepKind::EndOfCombat => Self::CombatEnd,
            StepKind::PostcombatMain => Self::PostcombatMainStep,
            StepKind::End => Self::EndingEnd,
            StepKind::Cleanup => Self::EndingCleanup,
        }
    }
}

#[cfg(feature = "python")]
impl From<StepEnum> for StepKind {
    fn from(value: StepEnum) -> Self {
        match value {
            StepEnum::BeginningUntap => Self::Untap,
            StepEnum::BeginningUpkeep => Self::Upkeep,
            StepEnum::BeginningDraw => Self::Draw,
            StepEnum::PrecombatMainStep => Self::Main,
            StepEnum::CombatBegin => Self::BeginningOfCombat,
            StepEnum::CombatDeclareAttackers => Self::DeclareAttackers,
            StepEnum::CombatDeclareBlockers => Self::DeclareBlockers,
            StepEnum::CombatDamage => Self::CombatDamage,
            StepEnum::CombatEnd => Self::EndOfCombat,
            StepEnum::PostcombatMainStep => Self::PostcombatMain,
            StepEnum::EndingEnd => Self::End,
            StepEnum::EndingCleanup => Self::Cleanup,
        }
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "ActionEnum", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum ActionEnum {
    PriorityPlayLand = 0,
    PriorityCastSpell = 1,
    PriorityPassPriority = 2,
    DeclareAttacker = 3,
    DeclareBlocker = 4,
    ChooseTarget = 5,
    PriorityActivateAbility = 6,
    ScryKeep = 7,
    ScryBottom = 8,
    SelectCard = 9,
    DeclineChoice = 10,
    PayCost = 11,
    ChooseMode = 12,
    TapForCost = 13,
    LearnTakeLesson = 14,
    LearnDiscard = 15,
    ChooseTrigger = 16,
}

#[cfg(feature = "python")]
#[pymethods]
impl ActionEnum {
    #[classattr]
    const PRIORITY_PLAY_LAND: Self = Self::PriorityPlayLand;
    #[classattr]
    const PRIORITY_CAST_SPELL: Self = Self::PriorityCastSpell;
    #[classattr]
    const PRIORITY_PASS_PRIORITY: Self = Self::PriorityPassPriority;
    #[classattr]
    const DECLARE_ATTACKER: Self = Self::DeclareAttacker;
    #[classattr]
    const DECLARE_BLOCKER: Self = Self::DeclareBlocker;
    #[classattr]
    const CHOOSE_TARGET: Self = Self::ChooseTarget;
    #[classattr]
    const PRIORITY_ACTIVATE_ABILITY: Self = Self::PriorityActivateAbility;
    #[classattr]
    const SCRY_KEEP: Self = Self::ScryKeep;
    #[classattr]
    const SCRY_BOTTOM: Self = Self::ScryBottom;
    #[classattr]
    const SELECT_CARD: Self = Self::SelectCard;
    #[classattr]
    const LEARN_TAKE_LESSON: Self = Self::LearnTakeLesson;
    #[classattr]
    const LEARN_DISCARD: Self = Self::LearnDiscard;
    #[classattr]
    const CHOOSE_TRIGGER: Self = Self::ChooseTrigger;
    #[classattr]
    const DECLINE_CHOICE: Self = Self::DeclineChoice;
    #[classattr]
    const PAY_COST: Self = Self::PayCost;
    #[classattr]
    const CHOOSE_MODE: Self = Self::ChooseMode;
    #[classattr]
    const TAP_FOR_COST: Self = Self::TapForCost;

    fn __int__(&self) -> i32 {
        *self as i32
    }

    fn __index__(&self) -> i32 {
        *self as i32
    }
}

#[cfg(feature = "python")]
impl From<ActionType> for ActionEnum {
    fn from(value: ActionType) -> Self {
        match value {
            ActionType::PriorityPlayLand => Self::PriorityPlayLand,
            ActionType::PriorityCastSpell => Self::PriorityCastSpell,
            ActionType::PriorityPassPriority => Self::PriorityPassPriority,
            ActionType::DeclareAttacker => Self::DeclareAttacker,
            ActionType::DeclareBlocker => Self::DeclareBlocker,
            ActionType::ChooseTarget => Self::ChooseTarget,
            ActionType::PriorityActivateAbility => Self::PriorityActivateAbility,
            ActionType::ScryKeep => Self::ScryKeep,
            ActionType::ScryBottom => Self::ScryBottom,
            ActionType::LearnTakeLesson => Self::LearnTakeLesson,
            ActionType::LearnDiscard => Self::LearnDiscard,
            ActionType::ChooseTrigger => Self::ChooseTrigger,
            ActionType::SelectCard => Self::SelectCard,
            ActionType::DeclineChoice => Self::DeclineChoice,
            ActionType::PayCost => Self::PayCost,
            ActionType::ChooseMode => Self::ChooseMode,
            ActionType::TapForCost => Self::TapForCost,
        }
    }
}

#[cfg(feature = "python")]
impl From<ActionEnum> for ActionType {
    fn from(value: ActionEnum) -> Self {
        match value {
            ActionEnum::PriorityPlayLand => Self::PriorityPlayLand,
            ActionEnum::PriorityCastSpell => Self::PriorityCastSpell,
            ActionEnum::PriorityPassPriority => Self::PriorityPassPriority,
            ActionEnum::DeclareAttacker => Self::DeclareAttacker,
            ActionEnum::DeclareBlocker => Self::DeclareBlocker,
            ActionEnum::ChooseTarget => Self::ChooseTarget,
            ActionEnum::PriorityActivateAbility => Self::PriorityActivateAbility,
            ActionEnum::ScryKeep => Self::ScryKeep,
            ActionEnum::ScryBottom => Self::ScryBottom,
            ActionEnum::LearnTakeLesson => Self::LearnTakeLesson,
            ActionEnum::LearnDiscard => Self::LearnDiscard,
            ActionEnum::ChooseTrigger => Self::ChooseTrigger,
            ActionEnum::SelectCard => Self::SelectCard,
            ActionEnum::DeclineChoice => Self::DeclineChoice,
            ActionEnum::PayCost => Self::PayCost,
            ActionEnum::ChooseMode => Self::ChooseMode,
            ActionEnum::TapForCost => Self::TapForCost,
        }
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "ActionSpaceEnum", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum ActionSpaceEnum {
    GameOver = 0,
    Priority = 1,
    DeclareAttacker = 2,
    DeclareBlocker = 3,
    ChooseTarget = 4,
    Scry = 5,
    LookAndSelect = 6,
    PayOrNot = 7,
    Modal = 8,
    Learn = 9,
    Waterbend = 10,
    Discard = 11,
    LegendRule = 12,
    OrderTriggers = 13,
}

#[cfg(feature = "python")]
#[pyclass(name = "EventTypeEnum", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum EventTypeEnum {
    CardMoved = EventType::CardMoved as i32,
    DamageDealt = EventType::DamageDealt as i32,
    LifeChanged = EventType::LifeChanged as i32,
    SpellCast = EventType::SpellCast as i32,
    SpellResolved = EventType::SpellResolved as i32,
    SpellCountered = EventType::SpellCountered as i32,
    AbilityTriggered = EventType::AbilityTriggered as i32,
    CombatAttackersDeclared = EventType::CombatAttackersDeclared as i32,
    BlockersDeclared = EventType::BlockersDeclared as i32,
    CombatDamageDealt = EventType::CombatDamageDealt as i32,
    PermanentsDied = EventType::PermanentsDied as i32,
    TurnStarted = EventType::TurnStarted as i32,
    CardRevealed = EventType::CardRevealed as i32,
    AbilityCountered = EventType::AbilityCountered as i32,
}

#[cfg(feature = "python")]
#[pymethods]
impl EventTypeEnum {
    #[classattr]
    const CARD_MOVED: Self = Self::CardMoved;
    #[classattr]
    const DAMAGE_DEALT: Self = Self::DamageDealt;
    #[classattr]
    const LIFE_CHANGED: Self = Self::LifeChanged;
    #[classattr]
    const SPELL_CAST: Self = Self::SpellCast;
    #[classattr]
    const SPELL_RESOLVED: Self = Self::SpellResolved;
    #[classattr]
    const SPELL_COUNTERED: Self = Self::SpellCountered;
    #[classattr]
    const ABILITY_TRIGGERED: Self = Self::AbilityTriggered;
    #[classattr]
    const COMBAT_ATTACKERS_DECLARED: Self = Self::CombatAttackersDeclared;
    #[classattr]
    const BLOCKERS_DECLARED: Self = Self::BlockersDeclared;
    #[classattr]
    const COMBAT_DAMAGE_DEALT: Self = Self::CombatDamageDealt;
    #[classattr]
    const PERMANENTS_DIED: Self = Self::PermanentsDied;
    #[classattr]
    const TURN_STARTED: Self = Self::TurnStarted;
    #[classattr]
    const CARD_REVEALED: Self = Self::CardRevealed;
    #[classattr]
    const ABILITY_COUNTERED: Self = Self::AbilityCountered;

    fn __int__(&self) -> i32 {
        *self as i32
    }

    fn __index__(&self) -> i32 {
        *self as i32
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "EventEntityKindEnum", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum EventEntityKindEnum {
    None = EventEntityKind::None as i32,
    Card = EventEntityKind::Card as i32,
    Permanent = EventEntityKind::Permanent as i32,
    Player = EventEntityKind::Player as i32,
    Object = EventEntityKind::Object as i32,
    Definition = EventEntityKind::Definition as i32,
}

#[cfg(feature = "python")]
#[pymethods]
impl EventEntityKindEnum {
    #[classattr]
    const NONE: Self = Self::None;
    #[classattr]
    const CARD: Self = Self::Card;
    #[classattr]
    const PERMANENT: Self = Self::Permanent;
    #[classattr]
    const PLAYER: Self = Self::Player;
    #[classattr]
    const OBJECT: Self = Self::Object;
    #[classattr]
    const DEFINITION: Self = Self::Definition;

    fn __int__(&self) -> i32 {
        *self as i32
    }

    fn __index__(&self) -> i32 {
        *self as i32
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl ActionSpaceEnum {
    #[classattr]
    const GAME_OVER: Self = Self::GameOver;
    #[classattr]
    const PRIORITY: Self = Self::Priority;
    #[classattr]
    const DECLARE_ATTACKER: Self = Self::DeclareAttacker;
    #[classattr]
    const DECLARE_BLOCKER: Self = Self::DeclareBlocker;
    #[classattr]
    const CHOOSE_TARGET: Self = Self::ChooseTarget;
    #[classattr]
    const SCRY: Self = Self::Scry;
    #[classattr]
    const LOOK_AND_SELECT: Self = Self::LookAndSelect;
    #[classattr]
    const PAY_OR_NOT: Self = Self::PayOrNot;
    #[classattr]
    const MODAL: Self = Self::Modal;
    #[classattr]
    const LEARN: Self = Self::Learn;
    #[classattr]
    const WATERBEND: Self = Self::Waterbend;
    #[classattr]
    const DISCARD: Self = Self::Discard;
    #[classattr]
    const LEGEND_RULE: Self = Self::LegendRule;
    #[classattr]
    const ORDER_TRIGGERS: Self = Self::OrderTriggers;

    fn __int__(&self) -> i32 {
        *self as i32
    }

    fn __index__(&self) -> i32 {
        *self as i32
    }
}

#[cfg(feature = "python")]
impl From<ActionSpaceKind> for ActionSpaceEnum {
    fn from(value: ActionSpaceKind) -> Self {
        match value {
            ActionSpaceKind::GameOver => Self::GameOver,
            ActionSpaceKind::Priority => Self::Priority,
            ActionSpaceKind::DeclareAttacker => Self::DeclareAttacker,
            ActionSpaceKind::DeclareBlocker => Self::DeclareBlocker,
            ActionSpaceKind::ChooseTarget => Self::ChooseTarget,
            ActionSpaceKind::Scry => Self::Scry,
            ActionSpaceKind::LookAndSelect => Self::LookAndSelect,
            ActionSpaceKind::PayOrNot => Self::PayOrNot,
            ActionSpaceKind::Modal => Self::Modal,
            ActionSpaceKind::Learn => Self::Learn,
            ActionSpaceKind::Waterbend => Self::Waterbend,
            ActionSpaceKind::Discard => Self::Discard,
            ActionSpaceKind::LegendRule => Self::LegendRule,
            ActionSpaceKind::OrderTriggers => Self::OrderTriggers,
        }
    }
}

#[cfg(feature = "python")]
impl From<ActionSpaceEnum> for ActionSpaceKind {
    fn from(value: ActionSpaceEnum) -> Self {
        match value {
            ActionSpaceEnum::GameOver => Self::GameOver,
            ActionSpaceEnum::Priority => Self::Priority,
            ActionSpaceEnum::DeclareAttacker => Self::DeclareAttacker,
            ActionSpaceEnum::DeclareBlocker => Self::DeclareBlocker,
            ActionSpaceEnum::ChooseTarget => Self::ChooseTarget,
            ActionSpaceEnum::Scry => Self::Scry,
            ActionSpaceEnum::LookAndSelect => Self::LookAndSelect,
            ActionSpaceEnum::PayOrNot => Self::PayOrNot,
            ActionSpaceEnum::Modal => Self::Modal,
            ActionSpaceEnum::Learn => Self::Learn,
            ActionSpaceEnum::Waterbend => Self::Waterbend,
            ActionSpaceEnum::Discard => Self::Discard,
            ActionSpaceEnum::LegendRule => Self::LegendRule,
            ActionSpaceEnum::OrderTriggers => Self::OrderTriggers,
        }
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "StackObjectKindEnum", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum StackObjectKindEnum {
    Spell = 0,
    ActivatedAbility = 1,
    TriggeredAbility = 2,
}

#[cfg(feature = "python")]
#[pymethods]
impl StackObjectKindEnum {
    #[classattr]
    const SPELL: Self = Self::Spell;
    #[classattr]
    const TRIGGERED_ABILITY: Self = Self::TriggeredAbility;
    #[classattr]
    const ACTIVATED_ABILITY: Self = Self::ActivatedAbility;

    fn __int__(&self) -> i32 {
        *self as i32
    }

    fn __index__(&self) -> i32 {
        *self as i32
    }
}

#[cfg(feature = "python")]
impl From<StackObjectKindData> for StackObjectKindEnum {
    fn from(value: StackObjectKindData) -> Self {
        match value {
            StackObjectKindData::Spell => Self::Spell,
            StackObjectKindData::ActivatedAbility => Self::ActivatedAbility,
            StackObjectKindData::TriggeredAbility => Self::TriggeredAbility,
        }
    }
}

#[cfg(feature = "python")]
impl From<StackObjectKindEnum> for StackObjectKindData {
    fn from(value: StackObjectKindEnum) -> Self {
        match value {
            StackObjectKindEnum::Spell => Self::Spell,
            StackObjectKindEnum::ActivatedAbility => Self::ActivatedAbility,
            StackObjectKindEnum::TriggeredAbility => Self::TriggeredAbility,
        }
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "StackTargetKindEnum", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum StackTargetKindEnum {
    Player = 0,
    Permanent = 1,
    StackObject = 2,
    Card = 3,
}

#[cfg(feature = "python")]
#[pymethods]
impl StackTargetKindEnum {
    #[classattr]
    const PLAYER: Self = Self::Player;
    #[classattr]
    const PERMANENT: Self = Self::Permanent;
    #[classattr]
    const STACK_OBJECT: Self = Self::StackObject;
    #[classattr]
    const CARD: Self = Self::Card;

    fn __int__(&self) -> i32 {
        *self as i32
    }

    fn __index__(&self) -> i32 {
        *self as i32
    }
}

#[cfg(feature = "python")]
impl From<StackTargetKindData> for StackTargetKindEnum {
    fn from(value: StackTargetKindData) -> Self {
        match value {
            StackTargetKindData::Player => Self::Player,
            StackTargetKindData::Permanent => Self::Permanent,
            StackTargetKindData::StackObject => Self::StackObject,
            StackTargetKindData::Card => Self::Card,
        }
    }
}

#[cfg(feature = "python")]
impl From<StackTargetKindEnum> for StackTargetKindData {
    fn from(value: StackTargetKindEnum) -> Self {
        match value {
            StackTargetKindEnum::Player => Self::Player,
            StackTargetKindEnum::Permanent => Self::Permanent,
            StackTargetKindEnum::StackObject => Self::StackObject,
            StackTargetKindEnum::Card => Self::Card,
        }
    }
}

/// A seat's setup: player name and [`Deck`], and whether the deck is submitted
/// to the world (`world == WORLD_SETUP`), so game creation admits it.
/// `decklist`/`sideboard` read the deck's sections under their receipt names.
#[cfg(feature = "python")]
#[pyclass(name = "PlayerConfig")]
#[derive(Clone)]
pub struct PyPlayerConfig {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get)]
    pub deck: Deck,
    pub submitted: bool,
    /// `"open"` (default) or `"closed"`: whether the opponent knows this
    /// player's main deck and sideboard.
    pub legacy_disclosure: Option<DecklistDisclosure>,
}

/// `PlayerConfig.world` names only the world; authored content packs are retired.
#[cfg(feature = "python")]
fn submission(world: Option<&str>) -> PyResult<bool> {
    match world {
        None => Ok(false),
        Some(crate::world::WORLD_POOL) => Ok(true),
        Some(other) => Err(PyValueError::new_err(format!(
            "PlayerConfig.world must be None or {:?}; content pack {other:?} is retired \
             (every match runs on the world's one catalog)",
            crate::world::WORLD_POOL
        ))),
    }
}

#[cfg(feature = "python")]
fn parse_disclosure(value: &str) -> PyResult<DecklistDisclosure> {
    match value {
        "open" => Ok(DecklistDisclosure::Open),
        "closed" => Ok(DecklistDisclosure::Closed),
        other => Err(PyValueError::new_err(format!(
            "decklist disclosure must be 'open' or 'closed', got {other:?}"
        ))),
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl PyPlayerConfig {
    #[new]
    #[pyo3(signature = (name, decklist, sideboard=None, world=None, disclosure=None))]
    fn new(
        py: Python<'_>,
        name: String,
        decklist: BTreeMap<String, usize>,
        sideboard: Option<BTreeMap<String, usize>>,
        world: Option<&str>,
        disclosure: Option<&str>,
    ) -> PyResult<Self> {
        if disclosure.is_some() {
            warn_disclosure(py)?;
        }
        Ok(Self {
            name,
            deck: Deck::new(decklist, sideboard.unwrap_or_default()),
            submitted: submission(world)?,
            legacy_disclosure: disclosure.map(parse_disclosure).transpose()?,
        })
    }

    #[getter]
    fn world(&self) -> Option<&'static str> {
        self.submitted.then_some(crate::world::WORLD_POOL)
    }

    #[setter]
    fn set_world(&mut self, world: Option<&str>) -> PyResult<()> {
        self.submitted = submission(world)?;
        Ok(())
    }

    #[getter]
    fn decklist(&self) -> BTreeMap<String, usize> {
        self.deck.main.clone()
    }

    #[getter]
    fn sideboard(&self) -> BTreeMap<String, usize> {
        self.deck.sideboard.clone()
    }

    #[getter]
    fn disclosure(&self, py: Python<'_>) -> PyResult<&'static str> {
        warn_disclosure(py)?;
        Ok(disclosure_name(self.legacy_disclosure.unwrap_or_default()))
    }

    #[setter]
    fn set_disclosure(&mut self, py: Python<'_>, value: &str) -> PyResult<()> {
        warn_disclosure(py)?;
        self.legacy_disclosure = Some(parse_disclosure(value)?);
        Ok(())
    }
}

#[cfg(feature = "python")]
impl From<PyPlayerConfig> for PlayerConfig {
    fn from(value: PyPlayerConfig) -> Self {
        PlayerConfig {
            name: value.name,
            deck: value.deck,
            submitted: value.submitted,
        }
    }
}

#[cfg(feature = "python")]
impl From<PlayerConfig> for PyPlayerConfig {
    fn from(value: PlayerConfig) -> Self {
        PyPlayerConfig {
            name: value.name,
            deck: value.deck,
            submitted: value.submitted,
            legacy_disclosure: None,
        }
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "Player")]
#[derive(Clone)]
pub struct PyPlayer {
    #[pyo3(get, set)]
    pub player_index: i32,
    #[pyo3(get, set)]
    pub id: i32,
    #[pyo3(get, set)]
    pub is_agent: bool,
    #[pyo3(get, set)]
    pub is_active: bool,
    #[pyo3(get, set)]
    pub life: i32,
    #[pyo3(get, set)]
    pub zone_counts: Vec<i32>,
    #[pyo3(get, set)]
    pub graveyard_lessons: i32,
    #[pyo3(get)]
    pub known_hand: BTreeMap<u32, u32>,
    #[pyo3(get)]
    pub known_library: BTreeMap<u32, u32>,
    /// Decklist counts by definition; `None` when the opponent's decklist is
    /// closed to this observation's viewer.
    #[pyo3(get)]
    pub main_counts: Option<BTreeMap<u32, u32>>,
    #[pyo3(get)]
    pub unseen_counts: Option<BTreeMap<u32, u32>>,
    #[pyo3(get)]
    pub sideboard_counts: Option<BTreeMap<u32, u32>>,
    #[pyo3(get)]
    pub remaining_sideboard_counts: Option<BTreeMap<u32, u32>>,
    #[pyo3(get, set)]
    pub keeps_red_mana: bool,
    /// What this player has done this turn.
    #[pyo3(get, set)]
    pub turn_history: PyPlayerTurnHistory,
    /// Spells this player cast this turn.
    #[pyo3(get, set)]
    pub spells_cast: u32,
    #[pyo3(get, set)]
    pub extra_turns: u32,
    #[pyo3(get, set)]
    pub additional_combats: u32,
    #[pyo3(get, set)]
    pub no_maximum_hand_size: bool,
    #[pyo3(get, set)]
    pub combat_mana: i32,
    #[pyo3(get)]
    pub mana: [i32; 6],
    #[pyo3(get)]
    pub lesson_mana: [i32; 6],
    #[pyo3(get)]
    pub restricted_mana: Vec<PyRestrictedMana>,
    /// Pooled mana with a spend effect or combat expiry, by class.
    #[pyo3(get)]
    pub pooled_mana_classes: Vec<PyPooledManaClass>,
    /// Player id whose next turn ends this player's hand-only casting.
    #[pyo3(get)]
    pub hand_only_until: Option<i32>,
    /// Continuous effects on this player, in timestamp order.
    #[pyo3(get, set)]
    pub effects: Vec<PyContinuousEffect>,
}

/// One continuous-effect record on a player or permanent.
#[cfg(feature = "python")]
#[pyclass(name = "ContinuousEffect")]
#[derive(Clone, Default)]
pub struct PyContinuousEffect {
    #[pyo3(get, set)]
    pub program: Vec<(u16, i32)>,
    #[pyo3(get, set)]
    pub description: Option<String>,
    #[pyo3(get, set)]
    pub modification: String,
    #[pyo3(get, set)]
    pub power: Option<i32>,
    #[pyo3(get, set)]
    pub toughness: Option<i32>,
    #[pyo3(get, set)]
    pub amount: Option<i32>,
    #[pyo3(get, set)]
    pub keywords: Vec<String>,
    #[pyo3(get, set)]
    pub subtypes: Vec<String>,
    #[pyo3(get, set)]
    pub color: Option<String>,
    #[pyo3(get, set)]
    pub duration: String,
    #[pyo3(get, set)]
    pub source_registry_key: Option<i32>,
    #[pyo3(get, set)]
    pub source_name: Option<String>,
    #[pyo3(get, set)]
    pub timestamp: i32,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyContinuousEffect {
    #[new]
    fn py_new() -> Self {
        Self::default()
    }
}

#[cfg(feature = "python")]
impl From<ContinuousEffectData> for PyContinuousEffect {
    fn from(value: ContinuousEffectData) -> Self {
        Self {
            program: value.program,
            description: value.description,
            modification: value.modification,
            power: value.power,
            toughness: value.toughness,
            amount: value.amount,
            keywords: value.keywords,
            subtypes: value.subtypes,
            color: value.color,
            duration: value.duration,
            source_registry_key: value.source_registry_key,
            source_name: value.source_name,
            timestamp: value.timestamp,
        }
    }
}

#[cfg(feature = "python")]
impl From<PyContinuousEffect> for ContinuousEffectData {
    fn from(value: PyContinuousEffect) -> Self {
        Self {
            program: value.program,
            description: value.description,
            modification: value.modification,
            power: value.power,
            toughness: value.toughness,
            amount: value.amount,
            keywords: value.keywords,
            subtypes: value.subtypes,
            color: value.color,
            duration: value.duration,
            source_registry_key: value.source_registry_key,
            source_name: value.source_name,
            timestamp: value.timestamp,
        }
    }
}

/// What a player has done this turn.
#[cfg(feature = "python")]
#[pyclass(name = "PlayerTurnHistory")]
#[derive(Clone, Default)]
pub struct PyPlayerTurnHistory {
    #[pyo3(get, set)]
    pub attacked: bool,
    #[pyo3(get, set)]
    pub permanents_sacrificed: u32,
    #[pyo3(get, set)]
    pub land_entered: bool,
    #[pyo3(get, set)]
    pub cards_drawn: u32,
    /// Bending kinds done this turn: bits water 1, earth 2, fire 4, air 8.
    #[pyo3(get, set)]
    pub bent: u8,
    /// Life lost this turn (CR 119.3).
    #[pyo3(get, set)]
    pub life_lost: u32,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyPlayerTurnHistory {
    #[new]
    fn py_new() -> Self {
        Self::default()
    }
}

#[cfg(feature = "python")]
impl From<PlayerTurnHistory> for PyPlayerTurnHistory {
    fn from(value: PlayerTurnHistory) -> Self {
        Self {
            attacked: value.attacked,
            permanents_sacrificed: value.permanents_sacrificed,
            land_entered: value.land_entered,
            cards_drawn: value.cards_drawn,
            bent: value.bent,
            life_lost: value.life_lost,
        }
    }
}

#[cfg(feature = "python")]
impl From<PyPlayerTurnHistory> for PlayerTurnHistory {
    fn from(value: PyPlayerTurnHistory) -> Self {
        Self {
            attacked: value.attacked,
            permanents_sacrificed: value.permanents_sacrificed,
            land_entered: value.land_entered,
            cards_drawn: value.cards_drawn,
            bent: value.bent,
            life_lost: value.life_lost,
        }
    }
}

/// The copy effect that made a permanent a copy: the printed definition at
/// the root of the copy chain and the exceptions in its copiable values.
#[cfg(feature = "python")]
#[pyclass(name = "CopyEffect")]
#[derive(Clone, Default)]
pub struct PyCopyEffect {
    #[pyo3(get, set)]
    pub registry_key: i32,
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub exceptions: Vec<PyContinuousEffect>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyCopyEffect {
    #[new]
    fn py_new() -> Self {
        Self::default()
    }
}

#[cfg(feature = "python")]
impl From<CopyData> for PyCopyEffect {
    fn from(value: CopyData) -> Self {
        Self {
            registry_key: value.registry_key,
            name: value.name,
            exceptions: value.exceptions.into_iter().map(Into::into).collect(),
        }
    }
}

#[cfg(feature = "python")]
impl From<PyCopyEffect> for CopyData {
    fn from(value: PyCopyEffect) -> Self {
        Self {
            registry_key: value.registry_key,
            name: value.name,
            exceptions: value.exceptions.into_iter().map(Into::into).collect(),
        }
    }
}

/// Mana spendable only on the named spell or ability-source subtypes, or
/// (`except_nonartifact_spells`) on anything but nonartifact spells.
#[cfg(feature = "python")]
#[pyclass(name = "RestrictedMana")]
#[derive(Clone)]
pub struct PyRestrictedMana {
    #[pyo3(get)]
    pub spells: Vec<String>,
    #[pyo3(get)]
    pub abilities: Vec<String>,
    #[pyo3(get)]
    pub except_nonartifact_spells: bool,
    #[pyo3(get)]
    pub mana: [i32; 6],
}

/// Pooled mana of one class with a spend effect or combat expiry: its qualifiers,
/// amounts by W/U/B/R/G/C and learning-token program.
#[cfg(feature = "python")]
#[pyclass(name = "PooledManaClass")]
#[derive(Clone)]
pub struct PyPooledManaClass {
    #[pyo3(get)]
    pub combat: bool,
    #[pyo3(get)]
    pub restriction: Option<String>,
    #[pyo3(get)]
    pub effect: Option<String>,
    #[pyo3(get)]
    pub mana: [i32; 6],
    #[pyo3(get)]
    pub program: Vec<(u16, i32)>,
}

#[cfg(feature = "python")]
impl From<PooledManaClassData> for PyPooledManaClass {
    fn from(value: PooledManaClassData) -> Self {
        Self {
            combat: value.combat,
            restriction: value.restriction,
            effect: value.effect,
            mana: value.mana,
            program: value.program,
        }
    }
}

#[cfg(feature = "python")]
impl From<PyPooledManaClass> for PooledManaClassData {
    fn from(value: PyPooledManaClass) -> Self {
        Self {
            combat: value.combat,
            restriction: value.restriction,
            effect: value.effect,
            mana: value.mana,
            program: value.program,
        }
    }
}

#[cfg(feature = "python")]
impl From<RestrictedManaData> for PyRestrictedMana {
    fn from(value: RestrictedManaData) -> Self {
        Self {
            spells: value.spells,
            abilities: value.abilities,
            except_nonartifact_spells: value.except_nonartifact_spells,
            mana: value.mana,
        }
    }
}

#[cfg(feature = "python")]
impl From<PyRestrictedMana> for RestrictedManaData {
    fn from(value: PyRestrictedMana) -> Self {
        Self {
            spells: value.spells,
            abilities: value.abilities,
            except_nonartifact_spells: value.except_nonartifact_spells,
            mana: value.mana,
        }
    }
}

#[cfg(feature = "python")]
impl From<PlayerData> for PyPlayer {
    fn from(value: PlayerData) -> Self {
        Self {
            player_index: value.player_index,
            id: value.id,
            is_agent: value.is_agent,
            is_active: value.is_active,
            life: value.life,
            zone_counts: value.zone_counts.to_vec(),
            graveyard_lessons: value.graveyard_lessons,
            known_hand: value.known_hand,
            known_library: value.known_library,
            main_counts: value.main_counts,
            unseen_counts: value.unseen_counts,
            sideboard_counts: value.sideboard_counts,
            remaining_sideboard_counts: value.remaining_sideboard_counts,
            keeps_red_mana: value.keeps_red_mana,
            turn_history: value.turn_history.into(),
            spells_cast: value.spells_cast,
            extra_turns: value.extra_turns,
            additional_combats: value.additional_combats,
            no_maximum_hand_size: value.no_maximum_hand_size,
            combat_mana: value.combat_mana,
            mana: value.mana,
            lesson_mana: value.lesson_mana,
            restricted_mana: value.restricted_mana.into_iter().map(Into::into).collect(),
            pooled_mana_classes: value
                .pooled_mana_classes
                .into_iter()
                .map(Into::into)
                .collect(),
            hand_only_until: value.hand_only_until,
            effects: value.effects.into_iter().map(Into::into).collect(),
        }
    }
}

#[cfg(feature = "python")]
impl From<PyPlayer> for PlayerData {
    fn from(value: PyPlayer) -> Self {
        let mut zone_counts = [0_i32; 7];
        for (index, out) in zone_counts.iter_mut().enumerate() {
            *out = value.zone_counts.get(index).copied().unwrap_or(0);
        }

        Self {
            player_index: value.player_index,
            id: value.id,
            is_agent: value.is_agent,
            is_active: value.is_active,
            life: value.life,
            zone_counts,
            graveyard_lessons: value.graveyard_lessons,
            known_hand: value.known_hand,
            known_library: value.known_library,
            main_counts: value.main_counts,
            unseen_counts: value.unseen_counts,
            sideboard_counts: value.sideboard_counts,
            remaining_sideboard_counts: value.remaining_sideboard_counts,
            keeps_red_mana: value.keeps_red_mana,
            turn_history: value.turn_history.into(),
            spells_cast: value.spells_cast,
            extra_turns: value.extra_turns,
            additional_combats: value.additional_combats,
            no_maximum_hand_size: value.no_maximum_hand_size,
            combat_mana: value.combat_mana,
            mana: value.mana,
            lesson_mana: value.lesson_mana,
            restricted_mana: value.restricted_mana.into_iter().map(Into::into).collect(),
            pooled_mana_classes: value
                .pooled_mana_classes
                .into_iter()
                .map(Into::into)
                .collect(),
            hand_only_until: value.hand_only_until,
            effects: value.effects.into_iter().map(Into::into).collect(),
        }
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "Turn")]
#[derive(Clone)]
pub struct PyTurn {
    #[pyo3(get, set)]
    pub turn_number: i32,
    #[pyo3(get, set)]
    pub phase: PhaseEnum,
    #[pyo3(get, set)]
    pub step: StepEnum,
    #[pyo3(get, set)]
    pub active_player_id: i32,
    #[pyo3(get, set)]
    pub agent_player_id: i32,
}

#[cfg(feature = "python")]
impl From<TurnData> for PyTurn {
    fn from(value: TurnData) -> Self {
        Self {
            turn_number: value.turn_number as i32,
            phase: value.phase.into(),
            step: value.step.into(),
            active_player_id: value.active_player_id,
            agent_player_id: value.agent_player_id,
        }
    }
}

#[cfg(feature = "python")]
impl From<PyTurn> for TurnData {
    fn from(value: PyTurn) -> Self {
        Self {
            turn_number: value.turn_number.max(0) as u32,
            phase: value.phase.into(),
            step: value.step.into(),
            active_player_id: value.active_player_id,
            agent_player_id: value.agent_player_id,
        }
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "CardTypes")]
#[derive(Clone)]
pub struct PyCardTypes {
    #[pyo3(get, set)]
    pub is_castable: bool,
    #[pyo3(get, set)]
    pub is_permanent: bool,
    #[pyo3(get, set)]
    pub is_non_land_permanent: bool,
    #[pyo3(get, set)]
    pub is_non_creature_permanent: bool,
    #[pyo3(get, set)]
    pub is_spell: bool,
    #[pyo3(get, set)]
    pub is_creature: bool,
    #[pyo3(get, set)]
    pub is_land: bool,
    #[pyo3(get, set)]
    pub is_planeswalker: bool,
    #[pyo3(get, set)]
    pub is_enchantment: bool,
    #[pyo3(get, set)]
    pub is_artifact: bool,
    #[pyo3(get, set)]
    pub is_kindred: bool,
    #[pyo3(get, set)]
    pub is_battle: bool,
}

#[cfg(feature = "python")]
impl From<CardTypeData> for PyCardTypes {
    fn from(value: CardTypeData) -> Self {
        Self {
            is_castable: value.is_castable,
            is_permanent: value.is_permanent,
            is_non_land_permanent: value.is_non_land_permanent,
            is_non_creature_permanent: value.is_non_creature_permanent,
            is_spell: value.is_spell,
            is_creature: value.is_creature,
            is_land: value.is_land,
            is_planeswalker: value.is_planeswalker,
            is_enchantment: value.is_enchantment,
            is_artifact: value.is_artifact,
            is_kindred: value.is_kindred,
            is_battle: value.is_battle,
        }
    }
}

#[cfg(feature = "python")]
impl From<PyCardTypes> for CardTypeData {
    fn from(value: PyCardTypes) -> Self {
        Self {
            is_castable: value.is_castable,
            is_permanent: value.is_permanent,
            is_non_land_permanent: value.is_non_land_permanent,
            is_non_creature_permanent: value.is_non_creature_permanent,
            is_spell: value.is_spell,
            is_creature: value.is_creature,
            is_land: value.is_land,
            is_planeswalker: value.is_planeswalker,
            is_enchantment: value.is_enchantment,
            is_artifact: value.is_artifact,
            is_kindred: value.is_kindred,
            is_battle: value.is_battle,
        }
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "Keywords")]
#[derive(Clone)]
pub struct PyKeywords {
    #[pyo3(get, set)]
    pub flying: bool,
    #[pyo3(get, set)]
    pub reach: bool,
    #[pyo3(get, set)]
    pub haste: bool,
    #[pyo3(get, set)]
    pub flash: bool,
    #[pyo3(get, set)]
    pub vigilance: bool,
    #[pyo3(get, set)]
    pub trample: bool,
    #[pyo3(get, set)]
    pub first_strike: bool,
    #[pyo3(get, set)]
    pub double_strike: bool,
    #[pyo3(get, set)]
    pub deathtouch: bool,
    #[pyo3(get, set)]
    pub lifelink: bool,
    #[pyo3(get, set)]
    pub defender: bool,
    #[pyo3(get, set)]
    pub menace: bool,
    #[pyo3(get, set)]
    pub hexproof: bool,
    #[pyo3(get, set)]
    pub indestructible: bool,
}

#[cfg(feature = "python")]
impl From<KeywordData> for PyKeywords {
    fn from(value: KeywordData) -> Self {
        Self {
            flying: value.flying,
            reach: value.reach,
            haste: value.haste,
            flash: value.flash,
            vigilance: value.vigilance,
            trample: value.trample,
            first_strike: value.first_strike,
            double_strike: value.double_strike,
            deathtouch: value.deathtouch,
            lifelink: value.lifelink,
            defender: value.defender,
            menace: value.menace,
            hexproof: value.hexproof,
            indestructible: value.indestructible,
        }
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "ManaCost")]
#[derive(Clone)]
pub struct PyManaCost {
    #[pyo3(get)]
    pub hybrid: Vec<(u8, u8)>,
    #[pyo3(get, set)]
    pub cost: Vec<i32>,
    #[pyo3(get, set)]
    pub mana_value: i32,
    /// Number of {X} symbols; `mana_value` counts none of them (CR 202.3e).
    #[pyo3(get)]
    pub x: u8,
}

#[cfg(feature = "python")]
impl From<ManaCost> for PyManaCost {
    fn from(value: ManaCost) -> Self {
        Self {
            hybrid: value
                .hybrid
                .iter()
                .map(|pip| (pip[0] as u8, pip[1] as u8))
                .collect(),
            cost: value.cost[..6].iter().map(|v| i32::from(*v)).collect(),
            mana_value: i32::from(value.mana_value),
            x: value.x,
        }
    }
}

#[cfg(feature = "python")]
impl From<PyManaCost> for ManaCost {
    fn from(value: PyManaCost) -> Self {
        let mut cost = [0_u8; 7];
        for (index, amount) in value.cost.iter().take(6).enumerate() {
            let clamped = (*amount).clamp(0, u8::MAX as i32) as u8;
            cost[index] = clamped;
        }
        cost[6] = 0;
        let mana_value = value.mana_value.clamp(0, u8::MAX as i32) as u8;

        Self {
            purpose: Default::default(),
            cost,
            mana_value,
            x: value.x,
            hybrid: value
                .hybrid
                .into_iter()
                .map(|(left, right)| {
                    let colors = [
                        crate::state::mana::Color::White,
                        crate::state::mana::Color::Blue,
                        crate::state::mana::Color::Black,
                        crate::state::mana::Color::Red,
                        crate::state::mana::Color::Green,
                    ];
                    [colors[left as usize], colors[right as usize]]
                })
                .collect(),
        }
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "ObservationCharacteristics", frozen)]
#[derive(Clone)]
pub struct PyObservationCharacteristics {
    data: CharacteristicsData,
}
#[cfg(feature = "python")]
#[pymethods]
impl PyObservationCharacteristics {
    #[getter]
    fn name(&self) -> String {
        self.data.name.clone()
    }
    #[getter]
    fn text_box(&self) -> String {
        self.data.text_box.clone()
    }
    #[getter]
    fn power(&self) -> i32 {
        self.data.power
    }
    #[getter]
    fn toughness(&self) -> i32 {
        self.data.toughness
    }
    #[getter]
    fn is_legendary(&self) -> bool {
        self.data.is_legendary
    }
    #[getter]
    fn is_ally(&self) -> bool {
        self.data.is_ally
    }
    #[getter]
    fn is_lesson(&self) -> bool {
        self.data.is_lesson
    }
    #[getter]
    fn ward_cost(&self) -> i32 {
        self.data.ward_cost
    }
    #[getter]
    fn ward_life(&self) -> i32 {
        self.data.ward_life
    }
    #[getter]
    fn ward_program(&self) -> Vec<(u16, i32)> {
        self.data.ward_program.clone()
    }
    #[getter]
    fn kicker_cost(&self) -> i32 {
        self.data.kicker_cost
    }
    #[getter]
    fn card_types(&self) -> PyCardTypes {
        self.data.card_types.clone().into()
    }
    #[getter]
    fn keywords(&self) -> PyKeywords {
        self.data.keywords.clone().into()
    }
    #[getter]
    fn mana_cost(&self) -> PyManaCost {
        self.data.mana_cost.clone().into()
    }
}

#[cfg(feature = "python")]
fn warn_observation_alias(py: Python<'_>, old: &str, new: &str) -> PyResult<()> {
    py.import_bound("warnings")?.call_method1(
        "warn",
        (
            format!("{old} is deprecated; use {new}; removal in 0.4.0"),
            py.get_type_bound::<pyo3::exceptions::PyDeprecationWarning>(),
            // A native getter has no Python frame, so level 1 is the caller.
            1,
        ),
    )?;
    Ok(())
}

#[cfg(feature = "python")]
#[pyclass(name = "Card")]
#[derive(Clone)]
pub struct PyCard {
    pub data: CardData,
}
#[cfg(feature = "python")]
#[pymethods]
impl PyCard {
    #[getter]
    fn zone(&self) -> ZoneEnum {
        self.data.zone.into()
    }
    #[setter]
    fn set_zone(&mut self, value: ZoneEnum) {
        self.data.zone = value.into();
    }
    #[getter]
    fn owner_id(&self) -> i32 {
        self.data.owner_id
    }
    #[setter]
    fn set_owner_id(&mut self, value: i32) {
        self.data.owner_id = value;
    }
    #[getter]
    fn id(&self) -> i32 {
        self.data.id
    }
    #[setter]
    fn set_id(&mut self, value: i32) {
        self.data.id = value;
    }
    #[getter]
    fn registry_key(&self) -> i32 {
        self.data.registry_key
    }
    #[setter]
    fn set_registry_key(&mut self, value: i32) {
        self.data.registry_key = value;
    }
    #[getter]
    fn name(&self, py: Python<'_>) -> PyResult<String> {
        warn_observation_alias(py, "name", "characteristics.name")?;
        Ok(self.data.name.clone())
    }
    #[setter]
    fn set_name(&mut self, py: Python<'_>, value: String) -> PyResult<()> {
        warn_observation_alias(py, "name", "characteristics.name")?;
        self.data.name = value;
        Ok(())
    }
    #[getter]
    fn text_box(&self, py: Python<'_>) -> PyResult<String> {
        warn_observation_alias(py, "text_box", "characteristics.text_box")?;
        Ok(self.data.text_box.clone())
    }
    #[setter]
    fn set_text_box(&mut self, py: Python<'_>, value: String) -> PyResult<()> {
        warn_observation_alias(py, "text_box", "characteristics.text_box")?;
        self.data.text_box = value;
        Ok(())
    }
    #[getter]
    fn power(&self, py: Python<'_>) -> PyResult<i32> {
        warn_observation_alias(py, "power", "characteristics.power")?;
        Ok(self.data.power)
    }
    #[setter]
    fn set_power(&mut self, py: Python<'_>, value: i32) -> PyResult<()> {
        warn_observation_alias(py, "power", "characteristics.power")?;
        self.data.power = value;
        Ok(())
    }
    #[getter]
    fn toughness(&self, py: Python<'_>) -> PyResult<i32> {
        warn_observation_alias(py, "toughness", "characteristics.toughness")?;
        Ok(self.data.toughness)
    }
    #[setter]
    fn set_toughness(&mut self, py: Python<'_>, value: i32) -> PyResult<()> {
        warn_observation_alias(py, "toughness", "characteristics.toughness")?;
        self.data.toughness = value;
        Ok(())
    }
    #[getter]
    fn is_token(&self) -> bool {
        self.data.is_token
    }
    #[setter]
    fn set_is_token(&mut self, value: bool) {
        self.data.is_token = value;
    }
    #[getter]
    fn is_legendary(&self, py: Python<'_>) -> PyResult<bool> {
        warn_observation_alias(py, "is_legendary", "characteristics.is_legendary")?;
        Ok(self.data.is_legendary)
    }
    #[setter]
    fn set_is_legendary(&mut self, py: Python<'_>, value: bool) -> PyResult<()> {
        warn_observation_alias(py, "is_legendary", "characteristics.is_legendary")?;
        self.data.is_legendary = value;
        Ok(())
    }
    #[getter]
    fn is_ally(&self, py: Python<'_>) -> PyResult<bool> {
        warn_observation_alias(py, "is_ally", "characteristics.is_ally")?;
        Ok(self.data.is_ally)
    }
    #[setter]
    fn set_is_ally(&mut self, py: Python<'_>, value: bool) -> PyResult<()> {
        warn_observation_alias(py, "is_ally", "characteristics.is_ally")?;
        self.data.is_ally = value;
        Ok(())
    }
    #[getter]
    fn is_lesson(&self, py: Python<'_>) -> PyResult<bool> {
        warn_observation_alias(py, "is_lesson", "characteristics.is_lesson")?;
        Ok(self.data.is_lesson)
    }
    #[setter]
    fn set_is_lesson(&mut self, py: Python<'_>, value: bool) -> PyResult<()> {
        warn_observation_alias(py, "is_lesson", "characteristics.is_lesson")?;
        self.data.is_lesson = value;
        Ok(())
    }
    #[getter]
    fn ward_cost(&self, py: Python<'_>) -> PyResult<i32> {
        warn_observation_alias(py, "ward_cost", "characteristics.ward_cost")?;
        Ok(self.data.ward_cost)
    }
    #[setter]
    fn set_ward_cost(&mut self, py: Python<'_>, value: i32) -> PyResult<()> {
        warn_observation_alias(py, "ward_cost", "characteristics.ward_cost")?;
        self.data.ward_cost = value;
        Ok(())
    }
    #[getter]
    fn kicker_cost(&self, py: Python<'_>) -> PyResult<i32> {
        warn_observation_alias(py, "kicker_cost", "characteristics.kicker_cost")?;
        Ok(self.data.kicker_cost)
    }
    #[setter]
    fn set_kicker_cost(&mut self, py: Python<'_>, value: i32) -> PyResult<()> {
        warn_observation_alias(py, "kicker_cost", "characteristics.kicker_cost")?;
        self.data.kicker_cost = value;
        Ok(())
    }
    #[getter]
    fn card_types(&self, py: Python<'_>) -> PyResult<PyCardTypes> {
        warn_observation_alias(py, "card_types", "characteristics.card_types")?;
        Ok(self.data.card_types.clone().into())
    }
    #[setter]
    fn set_card_types(&mut self, py: Python<'_>, value: PyCardTypes) -> PyResult<()> {
        warn_observation_alias(py, "card_types", "characteristics.card_types")?;
        self.data.card_types = value.into();
        Ok(())
    }
    #[getter]
    fn keywords(&self, py: Python<'_>) -> PyResult<PyKeywords> {
        warn_observation_alias(py, "keywords", "characteristics.keywords")?;
        Ok(self.data.keywords.clone().into())
    }
    #[setter]
    fn set_keywords(&mut self, py: Python<'_>, value: PyKeywords) -> PyResult<()> {
        warn_observation_alias(py, "keywords", "characteristics.keywords")?;
        self.data.keywords = value.into();
        Ok(())
    }
    #[getter]
    fn mana_cost(&self, py: Python<'_>) -> PyResult<PyManaCost> {
        warn_observation_alias(py, "mana_cost", "characteristics.mana_cost")?;
        Ok(self.data.mana_cost.clone().into())
    }
    #[setter]
    fn set_mana_cost(&mut self, py: Python<'_>, value: PyManaCost) -> PyResult<()> {
        warn_observation_alias(py, "mana_cost", "characteristics.mana_cost")?;
        self.data.mana_cost = value.into();
        Ok(())
    }
    #[getter]
    fn face_down(&self) -> bool {
        self.data.face_down
    }
    #[setter]
    fn set_face_down(&mut self, value: bool) {
        self.data.face_down = value;
    }
    #[getter]
    fn cast_permission(&self) -> Option<PyCastPermission> {
        self.data.cast_permission.clone().map(Into::into)
    }
    #[setter]
    fn set_cast_permission(&mut self, value: Option<PyCastPermission>) {
        self.data.cast_permission = value.map(Into::into);
    }
    #[getter]
    fn x(&self) -> Option<i32> {
        self.data.x
    }
    #[setter]
    fn set_x(&mut self, value: Option<i32>) {
        self.data.x = value;
    }
    #[getter]
    #[allow(clippy::wrong_self_convention)]
    fn from_battlefield_this_turn(&self) -> bool {
        self.data.from_battlefield_this_turn
    }
    #[setter]
    fn set_from_battlefield_this_turn(&mut self, value: bool) {
        self.data.from_battlefield_this_turn = value;
    }
    #[getter]
    fn characteristics(&self) -> PyObservationCharacteristics {
        PyObservationCharacteristics {
            data: self.data.characteristics.clone(),
        }
    }
}

/// A public permission to play an exiled card; see `CastPermissionData`.
#[cfg(feature = "python")]
#[pyclass(name = "CastPermission")]
#[derive(Clone)]
pub struct PyCastPermission {
    #[pyo3(get)]
    pub player_id: i32,
    #[pyo3(get)]
    pub cost: Option<PyManaCost>,
    #[pyo3(get)]
    pub this_turn: bool,
    #[pyo3(get)]
    pub through_next_turn: bool,
    #[pyo3(get)]
    pub own_turn_only: bool,
    #[pyo3(get)]
    pub after_turn: Option<u32>,
    #[pyo3(get)]
    pub source_permanent_id: Option<i32>,
    #[pyo3(get)]
    pub flash: bool,
    #[pyo3(get)]
    pub any_mana: bool,
    #[pyo3(get)]
    pub waterbend: bool,
    #[pyo3(get)]
    pub foretold: bool,
}

#[cfg(feature = "python")]
impl From<CastPermissionData> for PyCastPermission {
    fn from(value: CastPermissionData) -> Self {
        Self {
            player_id: value.player_id,
            cost: value.cost.map(Into::into),
            this_turn: value.this_turn,
            through_next_turn: value.through_next_turn,
            own_turn_only: value.own_turn_only,
            after_turn: value.after_turn,
            source_permanent_id: value.source_permanent_id,
            flash: value.flash,
            any_mana: value.any_mana,
            waterbend: value.waterbend,
            foretold: value.foretold,
        }
    }
}

#[cfg(feature = "python")]
impl From<PyCastPermission> for CastPermissionData {
    fn from(value: PyCastPermission) -> Self {
        Self {
            player_id: value.player_id,
            cost: value.cost.map(Into::into),
            this_turn: value.this_turn,
            through_next_turn: value.through_next_turn,
            own_turn_only: value.own_turn_only,
            after_turn: value.after_turn,
            source_permanent_id: value.source_permanent_id,
            flash: value.flash,
            any_mana: value.any_mana,
            waterbend: value.waterbend,
            foretold: value.foretold,
        }
    }
}

#[cfg(feature = "python")]
impl From<CardData> for PyCard {
    fn from(value: CardData) -> Self {
        Self { data: value }
    }
}

#[cfg(feature = "python")]
impl From<PyCard> for CardData {
    fn from(value: PyCard) -> Self {
        value.data
    }
}

#[cfg(feature = "python")]
impl From<PyKeywords> for KeywordData {
    fn from(value: PyKeywords) -> Self {
        Self {
            flying: value.flying,
            reach: value.reach,
            haste: value.haste,
            flash: value.flash,
            vigilance: value.vigilance,
            trample: value.trample,
            first_strike: value.first_strike,
            double_strike: value.double_strike,
            deathtouch: value.deathtouch,
            lifelink: value.lifelink,
            defender: value.defender,
            menace: value.menace,
            hexproof: value.hexproof,
            indestructible: value.indestructible,
        }
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "Permanent")]
#[derive(Clone)]
pub struct PyPermanent {
    pub data: PermanentData,
}
#[cfg(feature = "python")]
#[pymethods]
impl PyPermanent {
    #[getter]
    fn mana_value(&self) -> u8 {
        self.data.mana_value()
    }
    #[getter]
    fn colors(&self) -> Option<Vec<String>> {
        self.data.colors.clone()
    }
    #[setter]
    fn set_colors(&mut self, value: Option<Vec<String>>) {
        self.data.colors = value;
    }
    #[getter]
    fn keyword_counter_timestamps(&self) -> BTreeMap<String, i32> {
        self.data.keyword_counter_timestamps.clone()
    }
    #[setter]
    fn set_keyword_counter_timestamps(&mut self, value: BTreeMap<String, i32>) {
        self.data.keyword_counter_timestamps = value;
    }
    #[getter]
    fn transforms(&self) -> u32 {
        self.data.transforms
    }
    #[setter]
    fn set_transforms(&mut self, value: u32) {
        self.data.transforms = value;
    }
    #[getter]
    fn id(&self) -> i32 {
        self.data.id
    }
    #[setter]
    fn set_id(&mut self, value: i32) {
        self.data.id = value;
    }
    #[getter]
    fn registry_key(&self) -> i32 {
        self.data.registry_key
    }
    #[setter]
    fn set_registry_key(&mut self, value: i32) {
        self.data.registry_key = value;
    }
    #[getter]
    fn controller_id(&self) -> i32 {
        self.data.controller_id
    }
    #[setter]
    fn set_controller_id(&mut self, value: i32) {
        self.data.controller_id = value;
    }
    #[getter]
    fn tapped(&self) -> bool {
        self.data.tapped
    }
    #[setter]
    fn set_tapped(&mut self, value: bool) {
        self.data.tapped = value;
    }
    #[getter]
    fn damage(&self) -> i32 {
        self.data.damage
    }
    #[setter]
    fn set_damage(&mut self, value: i32) {
        self.data.damage = value;
    }
    #[getter]
    fn is_summoning_sick(&self) -> bool {
        self.data.is_summoning_sick
    }
    #[setter]
    fn set_is_summoning_sick(&mut self, value: bool) {
        self.data.is_summoning_sick = value;
    }
    #[getter]
    fn plus1_counters(&self) -> i32 {
        self.data.plus1_counters
    }
    #[setter]
    fn set_plus1_counters(&mut self, value: i32) {
        self.data.plus1_counters = value;
    }
    #[getter]
    fn cant_be_blocked_this_turn(&self) -> bool {
        self.data.cant_be_blocked_this_turn
    }
    #[setter]
    fn set_cant_be_blocked_this_turn(&mut self, value: bool) {
        self.data.cant_be_blocked_this_turn = value;
    }
    #[getter]
    fn power(&self, py: Python<'_>) -> PyResult<i32> {
        warn_observation_alias(py, "power", "characteristics.power")?;
        Ok(self.data.power)
    }
    #[setter]
    fn set_power(&mut self, py: Python<'_>, value: i32) -> PyResult<()> {
        warn_observation_alias(py, "power", "characteristics.power")?;
        self.data.power = value;
        Ok(())
    }
    #[getter]
    fn toughness(&self, py: Python<'_>) -> PyResult<i32> {
        warn_observation_alias(py, "toughness", "characteristics.toughness")?;
        Ok(self.data.toughness)
    }
    #[setter]
    fn set_toughness(&mut self, py: Python<'_>, value: i32) -> PyResult<()> {
        warn_observation_alias(py, "toughness", "characteristics.toughness")?;
        self.data.toughness = value;
        Ok(())
    }
    #[getter]
    fn is_animated(&self) -> bool {
        self.data.is_animated
    }
    #[setter]
    fn set_is_animated(&mut self, value: bool) {
        self.data.is_animated = value;
    }
    #[getter]
    fn has_exile_link(&self) -> bool {
        self.data.has_exile_link
    }
    #[setter]
    fn set_has_exile_link(&mut self, value: bool) {
        self.data.has_exile_link = value;
    }
    #[getter]
    fn exile_if_dies_this_turn(&self) -> bool {
        self.data.exile_if_dies_this_turn
    }
    #[setter]
    fn set_exile_if_dies_this_turn(&mut self, value: bool) {
        self.data.exile_if_dies_this_turn = value;
    }
    #[getter]
    fn attacking(&self) -> bool {
        self.data.attacking
    }
    #[setter]
    fn set_attacking(&mut self, value: bool) {
        self.data.attacking = value;
    }
    #[getter]
    fn exhausted_abilities(&self) -> i32 {
        self.data.exhausted_abilities
    }
    #[setter]
    fn set_exhausted_abilities(&mut self, value: i32) {
        self.data.exhausted_abilities = value;
    }
    #[getter]
    fn once_each_turn_used(&self) -> i32 {
        self.data.once_each_turn_used
    }
    #[setter]
    fn set_once_each_turn_used(&mut self, value: i32) {
        self.data.once_each_turn_used = value;
    }
    #[getter]
    fn granted_firebending(&self) -> i32 {
        self.data.granted_firebending
    }
    #[setter]
    fn set_granted_firebending(&mut self, value: i32) {
        self.data.granted_firebending = value;
    }
    #[getter]
    fn temp_subtypes(&self) -> Vec<String> {
        self.data.temp_subtypes.clone()
    }
    #[setter]
    fn set_temp_subtypes(&mut self, value: Vec<String>) {
        self.data.temp_subtypes = value;
    }
    #[getter]
    fn keywords(&self, py: Python<'_>) -> PyResult<PyKeywords> {
        warn_observation_alias(py, "keywords", "characteristics.keywords")?;
        Ok(self.data.keywords.clone().into())
    }
    #[setter]
    fn set_keywords(&mut self, py: Python<'_>, value: PyKeywords) -> PyResult<()> {
        warn_observation_alias(py, "keywords", "characteristics.keywords")?;
        self.data.keywords = value.into();
        Ok(())
    }
    #[getter]
    fn counters(&self) -> BTreeMap<String, i32> {
        self.data.counters.clone()
    }
    #[setter]
    fn set_counters(&mut self, value: BTreeMap<String, i32>) {
        self.data.counters = value;
    }
    #[getter]
    fn attached_to(&self) -> Option<i32> {
        self.data.attached_to
    }
    #[setter]
    fn set_attached_to(&mut self, value: Option<i32>) {
        self.data.attached_to = value;
    }
    #[getter]
    fn name(&self) -> Option<String> {
        self.data.name.clone()
    }
    #[setter]
    fn set_name(&mut self, value: Option<String>) {
        self.data.name = value;
    }
    #[getter]
    fn creature_types(&self) -> Option<Vec<String>> {
        self.data.creature_types.clone()
    }
    #[setter]
    fn set_creature_types(&mut self, value: Option<Vec<String>>) {
        self.data.creature_types = value;
    }
    #[getter]
    fn is_granted_land(&self) -> bool {
        self.data.is_granted_land
    }
    #[setter]
    fn set_is_granted_land(&mut self, value: bool) {
        self.data.is_granted_land = value;
    }
    #[getter]
    fn blocking(&self) -> Option<i32> {
        self.data.blocking
    }
    #[setter]
    fn set_blocking(&mut self, value: Option<i32>) {
        self.data.blocking = value;
    }
    #[getter]
    fn abilities_lost(&self) -> bool {
        self.data.abilities_lost
    }
    #[setter]
    fn set_abilities_lost(&mut self, value: bool) {
        self.data.abilities_lost = value;
    }
    #[getter]
    fn used_abilities(&self) -> Vec<usize> {
        self.data.used_abilities.clone()
    }
    #[setter]
    fn set_used_abilities(&mut self, value: Vec<usize>) {
        self.data.used_abilities = value;
    }
    #[getter]
    fn dealt_damage_this_turn(&self) -> bool {
        self.data.dealt_damage_this_turn
    }
    #[setter]
    fn set_dealt_damage_this_turn(&mut self, value: bool) {
        self.data.dealt_damage_this_turn = value;
    }
    #[getter]
    fn effects(&self) -> Vec<PyContinuousEffect> {
        self.data
            .effects
            .clone()
            .into_iter()
            .map(Into::into)
            .collect()
    }
    #[setter]
    fn set_effects(&mut self, value: Vec<PyContinuousEffect>) {
        self.data.effects = value.into_iter().map(Into::into).collect();
    }
    #[getter]
    fn timestamp(&self) -> Option<i32> {
        self.data.timestamp
    }
    #[setter]
    fn set_timestamp(&mut self, value: Option<i32>) {
        self.data.timestamp = value;
    }
    #[getter]
    fn copy(&self) -> Option<PyCopyEffect> {
        self.data.copy.clone().map(Into::into)
    }
    #[setter]
    fn set_copy(&mut self, value: Option<PyCopyEffect>) {
        self.data.copy = value.map(Into::into);
    }
    #[getter]
    fn characteristics(&self) -> PyObservationCharacteristics {
        PyObservationCharacteristics {
            data: self.data.characteristics.clone(),
        }
    }
    #[getter]
    fn object_id(&self) -> Option<i32> {
        self.data.object_id
    }
    #[getter]
    fn owner_id(&self) -> i32 {
        self.data.owner_id
    }
    #[getter]
    fn is_token(&self) -> bool {
        self.data.is_token
    }
    #[getter]
    fn card(&self, py: Python<'_>) -> PyResult<Option<PyCard>> {
        warn_observation_alias(
            py,
            "Permanent.card",
            "Permanent.characteristics and object_id/owner_id/is_token",
        )?;
        Ok(self.data.legacy_card().map(Into::into))
    }
    #[setter]
    fn set_card(&mut self, py: Python<'_>, value: Option<PyCard>) -> PyResult<()> {
        warn_observation_alias(
            py,
            "Permanent.card",
            "Permanent.characteristics and object identity",
        )?;
        if let Some(value) = value {
            let card = value.data;
            self.data.object_id = Some(card.id);
            self.data.owner_id = card.owner_id;
            self.data.registry_key = card.registry_key;
            self.data.is_token = card.is_token;
            self.data.characteristics = card.characteristics;
        } else {
            self.data.object_id = None;
        }
        Ok(())
    }
}

#[cfg(feature = "python")]
impl From<PermanentData> for PyPermanent {
    fn from(value: PermanentData) -> Self {
        Self { data: value }
    }
}

#[cfg(feature = "python")]
impl From<PyPermanent> for PermanentData {
    fn from(value: PyPermanent) -> Self {
        value.data
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "Action")]
#[derive(Clone)]
pub struct PyAction {
    #[pyo3(get, set)]
    pub description: Option<String>,
    #[pyo3(get, set)]
    pub action_type: ActionEnum,
    #[pyo3(get, set)]
    pub focus: Vec<i32>,
    #[pyo3(get, set)]
    pub declared: Option<bool>,
    #[pyo3(get, set)]
    pub mode_index: Option<usize>,
    #[pyo3(get, set)]
    pub name_definition: Option<u32>,
    #[pyo3(get, set)]
    pub name_index: Option<usize>,
    /// First and last vocabulary rows of a naming group choice.
    #[pyo3(get, set)]
    pub name_group: Option<[usize; 2]>,
    #[pyo3(get, set)]
    pub ability_index: Option<usize>,
    #[pyo3(get)]
    pub continuation_index: Option<usize>,
    /// The tap pays part of the cost itself rather than making mana.
    #[pyo3(get)]
    pub pays_directly: bool,
    /// `mode_index` is an announced number (X, a replicate count).
    #[pyo3(get)]
    pub announces_x: bool,
}

#[cfg(feature = "python")]
#[pyclass(name = "EventData")]
#[derive(Clone)]
pub struct PyEventData {
    #[pyo3(get, set)]
    pub event_type: EventTypeEnum,
    #[pyo3(get, set)]
    pub source_kind: EventEntityKindEnum,
    #[pyo3(get, set)]
    pub source_id: i32,
    #[pyo3(get, set)]
    pub target_kind: EventEntityKindEnum,
    #[pyo3(get, set)]
    pub target_id: i32,
    #[pyo3(get, set)]
    pub amount: i32,
    #[pyo3(get, set)]
    pub controller_id: i32,
    #[pyo3(get, set)]
    pub from_zone: i32,
    #[pyo3(get, set)]
    pub to_zone: i32,
    #[pyo3(get, set)]
    pub source_incarnation: i32,
    #[pyo3(get, set)]
    pub target_incarnation: i32,
}

#[cfg(feature = "python")]
impl From<EventData> for PyEventData {
    fn from(value: EventData) -> Self {
        Self {
            event_type: match value.event_type {
                x if x == EventType::CardMoved as i32 => EventTypeEnum::CardMoved,
                x if x == EventType::DamageDealt as i32 => EventTypeEnum::DamageDealt,
                x if x == EventType::LifeChanged as i32 => EventTypeEnum::LifeChanged,
                x if x == EventType::SpellCast as i32 => EventTypeEnum::SpellCast,
                x if x == EventType::SpellResolved as i32 => EventTypeEnum::SpellResolved,
                x if x == EventType::SpellCountered as i32 => EventTypeEnum::SpellCountered,
                x if x == EventType::AbilityTriggered as i32 => EventTypeEnum::AbilityTriggered,
                x if x == EventType::CombatAttackersDeclared as i32 => {
                    EventTypeEnum::CombatAttackersDeclared
                }
                x if x == EventType::BlockersDeclared as i32 => EventTypeEnum::BlockersDeclared,
                x if x == EventType::CombatDamageDealt as i32 => EventTypeEnum::CombatDamageDealt,
                x if x == EventType::PermanentsDied as i32 => EventTypeEnum::PermanentsDied,
                x if x == EventType::CardRevealed as i32 => EventTypeEnum::CardRevealed,
                x if x == EventType::AbilityCountered as i32 => EventTypeEnum::AbilityCountered,
                _ => EventTypeEnum::TurnStarted,
            },
            source_kind: match value.source_kind {
                x if x == EventEntityKind::Card as i32 => EventEntityKindEnum::Card,
                x if x == EventEntityKind::Permanent as i32 => EventEntityKindEnum::Permanent,
                x if x == EventEntityKind::Player as i32 => EventEntityKindEnum::Player,
                x if x == EventEntityKind::Object as i32 => EventEntityKindEnum::Object,
                x if x == EventEntityKind::Definition as i32 => EventEntityKindEnum::Definition,
                _ => EventEntityKindEnum::None,
            },
            source_id: value.source_id,
            target_kind: match value.target_kind {
                x if x == EventEntityKind::Card as i32 => EventEntityKindEnum::Card,
                x if x == EventEntityKind::Permanent as i32 => EventEntityKindEnum::Permanent,
                x if x == EventEntityKind::Player as i32 => EventEntityKindEnum::Player,
                x if x == EventEntityKind::Object as i32 => EventEntityKindEnum::Object,
                x if x == EventEntityKind::Definition as i32 => EventEntityKindEnum::Definition,
                _ => EventEntityKindEnum::None,
            },
            target_id: value.target_id,
            amount: value.amount,
            controller_id: value.controller_id,
            from_zone: value.from_zone,
            to_zone: value.to_zone,
            source_incarnation: value.source_incarnation,
            target_incarnation: value.target_incarnation,
        }
    }
}

#[cfg(feature = "python")]
impl From<PyEventData> for EventData {
    fn from(value: PyEventData) -> Self {
        Self {
            event_type: value.event_type as i32,
            source_kind: value.source_kind as i32,
            source_id: value.source_id,
            target_kind: value.target_kind as i32,
            target_id: value.target_id,
            amount: value.amount,
            controller_id: value.controller_id,
            from_zone: value.from_zone,
            to_zone: value.to_zone,
            source_incarnation: value.source_incarnation,
            target_incarnation: value.target_incarnation,
        }
    }
}

#[cfg(feature = "python")]
impl From<ActionOption> for PyAction {
    fn from(value: ActionOption) -> Self {
        Self {
            action_type: value.action_type.into(),
            focus: value.focus,
            declared: value.declared,
            description: value.description,
            mode_index: value.mode_index,
            name_definition: value.name_definition,
            name_index: value.name_index,
            name_group: value.name_group,
            ability_index: value.ability_index,
            continuation_index: value.continuation_index,
            pays_directly: value.pays_directly,
            announces_x: value.announces_x,
        }
    }
}

#[cfg(feature = "python")]
impl From<PyAction> for ActionOption {
    fn from(value: PyAction) -> Self {
        Self {
            action_type: value.action_type.into(),
            focus: value.focus,
            declared: value.declared,
            description: value.description,
            mode_index: value.mode_index,
            name_definition: value.name_definition,
            name_index: value.name_index,
            name_group: value.name_group,
            ability_index: value.ability_index,
            continuation_index: value.continuation_index,
            pays_directly: value.pays_directly,
            announces_x: value.announces_x,
        }
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "ActionSpace")]
#[derive(Clone)]
pub struct PyActionSpace {
    #[pyo3(get, set)]
    pub action_space_type: ActionSpaceEnum,
    #[pyo3(get, set)]
    pub actions: Vec<PyAction>,
    #[pyo3(get, set)]
    pub focus: Vec<i32>,
}

#[cfg(feature = "python")]
impl From<ActionSpaceData> for PyActionSpace {
    fn from(value: ActionSpaceData) -> Self {
        Self {
            action_space_type: value.action_space_type.into(),
            actions: value.actions.into_iter().map(PyAction::from).collect(),
            focus: value.focus,
        }
    }
}

#[cfg(feature = "python")]
impl From<PyActionSpace> for ActionSpaceData {
    fn from(value: PyActionSpace) -> Self {
        Self {
            action_space_type: value.action_space_type.into(),
            actions: value.actions.into_iter().map(ActionOption::from).collect(),
            focus: value.focus,
        }
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "StackTarget")]
#[derive(Clone)]
pub struct PyStackTarget {
    #[pyo3(get, set)]
    pub kind: StackTargetKindEnum,
    #[pyo3(get, set)]
    pub player_id: Option<i32>,
    #[pyo3(get, set)]
    pub permanent_id: Option<i32>,
    #[pyo3(get, set)]
    pub stack_object_id: Option<i32>,
    #[pyo3(get, set)]
    pub card_id: Option<i32>,
}

#[cfg(feature = "python")]
impl From<StackTargetData> for PyStackTarget {
    fn from(value: StackTargetData) -> Self {
        Self {
            kind: value.kind.into(),
            player_id: value.player_id,
            permanent_id: value.permanent_id,
            stack_object_id: value.stack_object_id,
            card_id: value.card_id,
        }
    }
}

#[cfg(feature = "python")]
impl From<PyStackTarget> for StackTargetData {
    fn from(value: PyStackTarget) -> Self {
        Self {
            kind: value.kind.into(),
            player_id: value.player_id,
            permanent_id: value.permanent_id,
            stack_object_id: value.stack_object_id,
            card_id: value.card_id,
        }
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "StackObject")]
#[derive(Clone)]
pub struct PyStackObject {
    pub data: StackObjectData,
}
#[cfg(feature = "python")]
#[pymethods]
impl PyStackObject {
    #[getter]
    fn description(&self) -> Option<String> {
        self.data.description.clone()
    }
    #[setter]
    fn set_description(&mut self, value: Option<String>) {
        self.data.description = value;
    }
    #[getter]
    fn stack_object_id(&self) -> i32 {
        self.data.stack_object_id
    }
    #[setter]
    fn set_stack_object_id(&mut self, value: i32) {
        self.data.stack_object_id = value;
    }
    #[getter]
    fn kind(&self) -> StackObjectKindEnum {
        self.data.kind.into()
    }
    #[setter]
    fn set_kind(&mut self, value: StackObjectKindEnum) {
        self.data.kind = value.into();
    }
    #[getter]
    fn controller_id(&self) -> i32 {
        self.data.controller_id
    }
    #[setter]
    fn set_controller_id(&mut self, value: i32) {
        self.data.controller_id = value;
    }
    #[getter]
    fn source_card_registry_key(&self) -> i32 {
        self.data.source_card_registry_key
    }
    #[setter]
    fn set_source_card_registry_key(&mut self, value: i32) {
        self.data.source_card_registry_key = value;
    }
    #[getter]
    fn source_permanent_id(&self) -> Option<i32> {
        self.data.source_permanent_id
    }
    #[setter]
    fn set_source_permanent_id(&mut self, value: Option<i32>) {
        self.data.source_permanent_id = value;
    }
    #[getter]
    fn chosen_mode(&self) -> Option<usize> {
        self.data.chosen_mode
    }
    #[setter]
    fn set_chosen_mode(&mut self, value: Option<usize>) {
        self.data.chosen_mode = value;
    }
    #[getter]
    fn ability_index(&self) -> Option<i32> {
        self.data.ability_index
    }
    #[setter]
    fn set_ability_index(&mut self, value: Option<i32>) {
        self.data.ability_index = value;
    }
    #[getter]
    fn targets(&self) -> Vec<PyStackTarget> {
        self.data
            .targets
            .clone()
            .into_iter()
            .map(Into::into)
            .collect()
    }
    #[setter]
    fn set_targets(&mut self, value: Vec<PyStackTarget>) {
        self.data.targets = value.into_iter().map(Into::into).collect();
    }
    #[getter]
    fn x(&self) -> i32 {
        self.data.x
    }
    #[setter]
    fn set_x(&mut self, value: i32) {
        self.data.x = value;
    }
    #[getter]
    fn granted_keywords(&self) -> Option<PyKeywords> {
        self.data.granted_keywords.clone().map(Into::into)
    }
    #[setter]
    fn set_granted_keywords(&mut self, value: Option<PyKeywords>) {
        self.data.granted_keywords = value.map(Into::into);
    }
    #[getter]
    fn cast_from_graveyard(&self) -> bool {
        self.data
            .spell
            .as_ref()
            .is_some_and(|spell| spell.cast_from_graveyard)
    }
    #[getter]
    fn kicked(&self) -> bool {
        self.data.kicked
    }
    #[setter]
    fn set_kicked(&mut self, value: bool) {
        self.data.kicked = value;
    }
    #[getter]
    fn characteristics(&self) -> Option<PyObservationCharacteristics> {
        self.data
            .spell
            .as_ref()
            .map(|s| PyObservationCharacteristics {
                data: s.characteristics.clone(),
            })
    }
    #[getter]
    fn object_id(&self) -> Option<i32> {
        self.data.spell.as_ref().map(|s| s.object_id)
    }
    #[getter]
    fn owner_id(&self) -> Option<i32> {
        self.data.spell.as_ref().map(|s| s.owner_id)
    }
    #[getter]
    fn is_copy(&self) -> Option<bool> {
        self.data.spell.as_ref().map(|s| s.is_copy)
    }
    #[getter]
    fn announced_x(&self) -> Option<i32> {
        self.data
            .spell
            .as_ref()
            .and_then(|s| s.has_announced_x.then_some(self.data.x))
    }
    #[getter]
    fn card(&self, py: Python<'_>) -> PyResult<Option<PyCard>> {
        warn_observation_alias(
            py,
            "StackObject.card",
            "StackObject.characteristics and object_id/owner_id/is_copy",
        )?;
        Ok(self
            .data
            .spell
            .as_ref()
            .map(|s| s.legacy_card(self.data.x).into()))
    }
    #[setter]
    fn set_card(&mut self, py: Python<'_>, value: Option<PyCard>) -> PyResult<()> {
        warn_observation_alias(
            py,
            "StackObject.card",
            "StackObject.characteristics and object identity",
        )?;
        if let Some(card) = &value {
            self.data.source_card_registry_key = card.data.registry_key;
            self.data.x = card.data.x.unwrap_or(0);
        }
        self.data.spell =
            value.map(|value| crate::agent::observation::SpellData::from_legacy_card(value.data));
        Ok(())
    }
}

#[cfg(feature = "python")]
impl From<StackObjectData> for PyStackObject {
    fn from(value: StackObjectData) -> Self {
        Self { data: value }
    }
}

#[cfg(feature = "python")]
impl From<PyStackObject> for StackObjectData {
    fn from(value: PyStackObject) -> Self {
        value.data
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "SideboardCard")]
#[derive(Clone)]
pub struct PySideboardCard {
    #[pyo3(get)]
    pub candidate_id: i32,
    #[pyo3(get)]
    pub owner_id: i32,
    #[pyo3(get)]
    pub registry_key: i32,
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub text_box: String,
}

#[cfg(feature = "python")]
impl From<SideboardCardData> for PySideboardCard {
    fn from(value: SideboardCardData) -> Self {
        Self {
            candidate_id: value.candidate_id,
            owner_id: value.owner_id,
            registry_key: value.registry_key,
            name: value.name,
            text_box: value.text_box,
        }
    }
}

#[cfg(feature = "python")]
impl From<PySideboardCard> for SideboardCardData {
    fn from(value: PySideboardCard) -> Self {
        Self {
            candidate_id: value.candidate_id,
            owner_id: value.owner_id,
            registry_key: value.registry_key,
            name: value.name,
            text_box: value.text_box,
        }
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "SharedEffect", frozen)]
#[derive(Clone)]
pub struct PySharedEffect {
    inner: crate::agent::shared_effects::SharedEffectData,
}

#[cfg(feature = "python")]
#[pymethods]
impl PySharedEffect {
    #[getter]
    fn effect(&self) -> PyContinuousEffect {
        self.inner.effect.clone().into()
    }
    #[getter]
    fn controller_id(&self) -> i32 {
        self.inner.controller_id
    }
    #[getter]
    fn source(&self) -> Option<crate::agent::shared_effects::EffectObject> {
        self.inner.source.clone()
    }
    #[getter]
    fn members(&self) -> Vec<crate::agent::shared_effects::EffectMembership> {
        self.inner.members.clone()
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "Observation")]
#[derive(Clone)]
pub struct PyObservation {
    #[pyo3(get, set)]
    pub decklists_closed: bool,
    #[pyo3(get)]
    pub shared_effects: Option<Vec<PySharedEffect>>,
    #[pyo3(get)]
    pub continuation: Vec<crate::agent::continuation::ContinuationFrame>,
    #[pyo3(get, set)]
    pub game_over: bool,
    #[pyo3(get, set)]
    pub won: bool,
    #[pyo3(get, set)]
    pub turn: PyTurn,
    #[pyo3(get, set)]
    pub action_space: PyActionSpace,
    #[pyo3(get, set)]
    pub agent: PyPlayer,
    #[pyo3(get, set)]
    pub agent_cards: Vec<PyCard>,
    #[pyo3(get)]
    pub agent_sideboard: Vec<PySideboardCard>,
    #[pyo3(get, set)]
    pub agent_permanents: Vec<PyPermanent>,
    #[pyo3(get, set)]
    pub opponent: PyPlayer,
    #[pyo3(get, set)]
    pub opponent_cards: Vec<PyCard>,
    #[pyo3(get, set)]
    pub opponent_permanents: Vec<PyPermanent>,
    #[pyo3(get, set)]
    pub stack_objects: Vec<PyStackObject>,
    #[pyo3(get, set)]
    pub recent_events: Vec<PyEventData>,
    pub decision_memory: crate::agent::decision_memory::DecisionLog,
    pub policy_history: crate::agent::policy_history::HistoryWindow,
}

#[cfg(feature = "python")]
impl From<Observation> for PyObservation {
    fn from(value: Observation) -> Self {
        Self {
            shared_effects: value.shared_effects.map(|records| {
                records
                    .into_iter()
                    .map(|inner| PySharedEffect { inner })
                    .collect()
            }),
            decklists_closed: value.decklists_closed,
            game_over: value.game_over,
            won: value.won,
            turn: value.turn.into(),
            action_space: value.action_space.into(),
            agent: value.agent.into(),
            agent_cards: value.agent_cards.into_iter().map(PyCard::from).collect(),
            agent_sideboard: value
                .agent_sideboard
                .into_iter()
                .map(PySideboardCard::from)
                .collect(),
            agent_permanents: value
                .agent_permanents
                .into_iter()
                .map(PyPermanent::from)
                .collect(),
            opponent: value.opponent.into(),
            opponent_cards: value.opponent_cards.into_iter().map(PyCard::from).collect(),
            opponent_permanents: value
                .opponent_permanents
                .into_iter()
                .map(PyPermanent::from)
                .collect(),
            stack_objects: value
                .stack_objects
                .into_iter()
                .map(PyStackObject::from)
                .collect(),
            policy_history: value.policy_history,
            decision_memory: value.decision_memory,
            continuation: value.continuation,
            recent_events: value
                .recent_events
                .into_iter()
                .map(PyEventData::from)
                .collect(),
        }
    }
}

#[cfg(feature = "python")]
impl From<PyObservation> for Observation {
    fn from(value: PyObservation) -> Self {
        Self {
            shared_effects: value
                .shared_effects
                .map(|records| records.into_iter().map(|record| record.inner).collect()),
            decklists_closed: value.decklists_closed,
            game_over: value.game_over,
            won: value.won,
            turn: value.turn.into(),
            action_space: value.action_space.into(),
            agent: value.agent.into(),
            agent_cards: value.agent_cards.into_iter().map(CardData::from).collect(),
            agent_sideboard: value
                .agent_sideboard
                .into_iter()
                .map(SideboardCardData::from)
                .collect(),
            agent_permanents: value
                .agent_permanents
                .into_iter()
                .map(PermanentData::from)
                .collect(),
            opponent: value.opponent.into(),
            opponent_cards: value
                .opponent_cards
                .into_iter()
                .map(CardData::from)
                .collect(),
            opponent_permanents: value
                .opponent_permanents
                .into_iter()
                .map(PermanentData::from)
                .collect(),
            stack_objects: value
                .stack_objects
                .into_iter()
                .map(StackObjectData::from)
                .collect(),
            policy_history: value.policy_history,
            decision_memory: value.decision_memory,
            continuation: value.continuation,
            recent_events: value
                .recent_events
                .into_iter()
                .map(EventData::from)
                .collect(),
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl PyObservation {
    /// Frozen viewer JSON omits continuation. A retained reader may supply
    /// its separately serialized frames; neither input grants execution authority.
    #[staticmethod]
    #[pyo3(signature = (text, *, continuation_json=None))]
    fn from_json(text: &str, continuation_json: Option<&str>) -> PyResult<Self> {
        let mut observation = crate::agent::observation_json::parse_with_history(text)
            .map_err(PyValueError::new_err)?;
        if let Some(frames) = continuation_json {
            observation.continuation = parse_continuation_frames_json(frames)?;
        }
        Ok(Self::from(observation))
    }

    fn decision_memory_json(&self) -> String {
        serde_json::to_string(&self.decision_memory).expect("decision memory serializes")
    }

    fn continuation_json(&self) -> String {
        serde_json::to_string(&self.continuation).expect("continuation serializes")
    }

    /// Deprecated input v9/v10 card rows (removed in 0.4.0): the observation's
    /// cards with battlefield objects and stack spells listed as cards again.
    fn legacy_card_lists(&self) -> (Vec<PyCard>, Vec<PyCard>) {
        let legacy = crate::agent::legacy_input::view(&Observation::from(self.clone()));
        (
            legacy.agent_cards.into_iter().map(PyCard::from).collect(),
            legacy
                .opponent_cards
                .into_iter()
                .map(PyCard::from)
                .collect(),
        )
    }

    fn validate(&self) -> bool {
        Observation::from(self.clone()).validate()
    }

    #[allow(non_snake_case)]
    fn toJSON(&self) -> String {
        Observation::from(self.clone()).to_json_with_history()
    }
}

/// Opaque origin binding. JSON is portable evidence; only this live handle can
/// authorize submission back to its originating Env and reset generation.
#[cfg(feature = "python")]
#[pyclass(name = "DecisionCapture", frozen)]
pub struct PyDecisionCapture {
    source: Arc<Mutex<Env>>,
    generation: u64,
    capture: crate::decision_capture::DecisionCapture,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyDecisionCapture {
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.capture)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "Env")]
pub struct PyEnv {
    pub(crate) inner: Arc<Mutex<Env>>,
    selected_guard: bool,
    prepared_lease: Option<PreparedBranchLease>,
}

#[cfg(feature = "python")]
#[derive(Debug, Default)]
struct PreparedBranchLeaseCounters {
    current: AtomicUsize,
    maximum: AtomicUsize,
}

#[cfg(feature = "python")]
#[derive(Debug)]
struct PreparedBranchLease {
    counters: Arc<PreparedBranchLeaseCounters>,
}

#[cfg(feature = "python")]
impl PreparedBranchLease {
    fn acquire(counters: Arc<PreparedBranchLeaseCounters>) -> Self {
        let current = counters.current.fetch_add(1, Ordering::SeqCst) + 1;
        counters.maximum.fetch_max(current, Ordering::SeqCst);
        Self { counters }
    }

    fn clone_lease(&self) -> Self {
        Self::acquire(Arc::clone(&self.counters))
    }
}

#[cfg(feature = "python")]
impl Drop for PreparedBranchLease {
    fn drop(&mut self) {
        self.counters.current.fetch_sub(1, Ordering::SeqCst);
    }
}

#[cfg(feature = "python")]
#[derive(Debug, Default, Serialize)]
struct PreparedMaterializerDiagnostics {
    space_constructions: u64,
    batch_calls: u64,
    requested_rows: u64,
    materialized_rows: u64,
    rejected_calls: u64,
    maximum_returned_batch: usize,
}

#[cfg(feature = "python")]
#[pyclass(name = "PreparedPossibleWorldMaterializer")]
pub struct PyPreparedPossibleWorldMaterializer {
    inner: PreparedPossibleWorldMaterializer,
    source: Arc<Mutex<Env>>,
    diagnostics: Mutex<PreparedMaterializerDiagnostics>,
    branch_leases: Arc<PreparedBranchLeaseCounters>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyPreparedPossibleWorldMaterializer {
    #[getter]
    fn viewer(&self) -> usize {
        self.inner.viewer()
    }

    #[getter]
    fn space_identity(&self) -> String {
        self.inner.space_identity().to_string()
    }

    #[getter]
    fn support_size(&self) -> usize {
        self.inner.support_size()
    }

    #[getter]
    fn max_batch_size(&self) -> usize {
        self.inner.max_batch_size()
    }

    #[getter]
    fn construction_count(&self) -> u64 {
        self.inner.construction_count()
    }

    fn diagnostics_json(&self) -> PyResult<String> {
        let diagnostics = self
            .diagnostics
            .lock()
            .map_err(|_| PyRuntimeError::new_err("prepared diagnostics lock poisoned"))?;
        let mut value = serde_json::to_value(&*diagnostics)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        value["current_live_prepared_branches"] =
            json!(self.branch_leases.current.load(Ordering::SeqCst));
        value["maximum_live_prepared_branches"] =
            json!(self.branch_leases.maximum.load(Ordering::SeqCst));
        serde_json::to_string(&value).map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    #[pyo3(signature = (world_indexes, seeds, refresh_opponent_commitment=false))]
    fn materialize_indexes(
        &self,
        world_indexes: Vec<usize>,
        seeds: Vec<u64>,
        refresh_opponent_commitment: bool,
    ) -> PyResult<Vec<PyEnv>> {
        if self.branch_leases.current.load(Ordering::SeqCst) != 0 {
            let mut diagnostics = self
                .diagnostics
                .lock()
                .map_err(|_| PyRuntimeError::new_err("prepared diagnostics lock poisoned"))?;
            diagnostics.rejected_calls += 1;
            return Err(PyAgentError::new_err(
                "prepared branches from the preceding batch are still live",
            ));
        }
        {
            let mut diagnostics = self
                .diagnostics
                .lock()
                .map_err(|_| PyRuntimeError::new_err("prepared diagnostics lock poisoned"))?;
            diagnostics.batch_calls += 1;
        }

        let branches = {
            let source = self
                .source
                .lock()
                .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
            let before = source.possible_world_space_construction_count();
            let result = self.inner.materialize_indexes(
                &source,
                &world_indexes,
                &seeds,
                refresh_opponent_commitment,
            );
            let constructions = source.possible_world_space_construction_count() - before;
            self.diagnostics
                .lock()
                .map_err(|_| PyRuntimeError::new_err("prepared diagnostics lock poisoned"))?
                .space_constructions += constructions;
            result
        };
        let branches = match branches {
            Ok(branches) => branches,
            Err(error) => {
                let mut diagnostics = self
                    .diagnostics
                    .lock()
                    .map_err(|_| PyRuntimeError::new_err("prepared diagnostics lock poisoned"))?;
                diagnostics.rejected_calls += 1;
                return Err(map_agent_err(error));
            }
        };
        {
            let mut diagnostics = self
                .diagnostics
                .lock()
                .map_err(|_| PyRuntimeError::new_err("prepared diagnostics lock poisoned"))?;
            diagnostics.requested_rows += world_indexes.len() as u64;
            diagnostics.materialized_rows += branches.len() as u64;
            diagnostics.maximum_returned_batch =
                diagnostics.maximum_returned_batch.max(branches.len());
        }
        Ok(branches
            .into_iter()
            .map(|branch| PyEnv {
                inner: Arc::new(Mutex::new(branch)),
                selected_guard: false,
                prepared_lease: Some(PreparedBranchLease::acquire(Arc::clone(
                    &self.branch_leases,
                ))),
            })
            .collect())
    }
}

#[cfg(feature = "python")]
const SELECTED_BRANCH_DRIVER_ID: &str = "full_clone/current_game_v1";

#[cfg(feature = "python")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SelectedMutationSite {
    World,
    Child,
    Leaf,
}

#[cfg(feature = "python")]
impl SelectedMutationSite {
    fn parse(value: &str) -> PyResult<Self> {
        match value {
            "world" => Ok(Self::World),
            "child" => Ok(Self::Child),
            "leaf" => Ok(Self::Leaf),
            _ => Err(PyAgentError::new_err(format!(
                "unknown selected mutation site {value:?}"
            ))),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::World => "world",
            Self::Child => "child",
            Self::Leaf => "leaf",
        }
    }
}

#[cfg(feature = "python")]
#[derive(Clone, Debug, Default, Serialize)]
struct SelectedSiteCounts {
    world: u64,
    child: u64,
    leaf: u64,
}

#[cfg(feature = "python")]
impl SelectedSiteCounts {
    fn increment(&mut self, site: SelectedMutationSite) -> u64 {
        let value = match site {
            SelectedMutationSite::World => &mut self.world,
            SelectedMutationSite::Child => &mut self.child,
            SelectedMutationSite::Leaf => &mut self.leaf,
        };
        *value += 1;
        *value
    }

    fn total(&self) -> u64 {
        self.world + self.child + self.leaf
    }
}

#[cfg(feature = "python")]
#[derive(Clone, Debug, Default, Deserialize)]
struct SelectedPreconditions {
    offer_id: Option<u32>,
    prompt_id: Option<u64>,
    expected_revision: Option<u64>,
    authority_hash: Option<String>,
    legal_surface_hash: Option<String>,
}

#[cfg(feature = "python")]
#[derive(Debug)]
struct SelectedBranchRuntimeState {
    match_id: String,
    audit: bool,
    next_command_id: u64,
    next_apply_sequence: u64,
    forks: SelectedSiteCounts,
    applies: SelectedSiteCounts,
    leaf_playouts: u64,
    leaf_cap_hits: u64,
    tapes: [Vec<Value>; 3],
}

#[cfg(feature = "python")]
impl SelectedBranchRuntimeState {
    fn tape_mut(&mut self, site: SelectedMutationSite) -> &mut Vec<Value> {
        match site {
            SelectedMutationSite::World => &mut self.tapes[0],
            SelectedMutationSite::Child => &mut self.tapes[1],
            SelectedMutationSite::Leaf => &mut self.tapes[2],
        }
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "SelectedBranchRuntime")]
pub struct PySelectedBranchRuntime {
    inner: Mutex<SelectedBranchRuntimeState>,
}

#[cfg(feature = "python")]
fn compact_search_witness(witness: &SearchStateWitness) -> Value {
    json!({
        "authority_hash": witness.authority.hash,
        "legal_surface_hash": witness.legal_surface.hash,
        "action_count": witness.legal_surface.action_count,
        "acting_viewer_hash": witness.acting_projection.hash,
        "viewer_hashes": [witness.viewers[0].hash, witness.viewers[1].hash],
        "event_boundary": witness.diagnostics.event_boundary,
        "rng_continuation": witness.diagnostics.rng_probe,
        "terminal": witness.diagnostics.terminal,
    })
}

#[cfg(feature = "python")]
impl PyEnv {
    fn reject_guarded_mutation(&self, operation: &str) -> PyResult<()> {
        if self.selected_guard {
            return Err(PyAgentError::new_err(format!(
                "selected production branch forbids {operation}; use SelectedBranchRuntime"
            )));
        }
        Ok(())
    }

    fn require_selected_guard(&self) -> PyResult<()> {
        if !self.selected_guard {
            return Err(PyAgentError::new_err(
                "SelectedBranchRuntime requires a guarded selected branch",
            ));
        }
        Ok(())
    }
}

/// Prompt-bound structured offers. The private Rust mapping remains attached
/// to the public projection, so Python can submit IDs but cannot forge engine
/// identities from presentation values.
#[cfg(feature = "python")]
#[pyclass(name = "StructuredOfferSet", frozen)]
#[derive(Clone)]
pub struct PyStructuredOfferSet {
    inner: StructuredOfferSet,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyStructuredOfferSet {
    fn projection_json(&self) -> PyResult<String> {
        serde_json::to_string(self.inner.projection())
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl PySelectedBranchRuntime {
    #[cfg(feature = "benchmark")]
    #[staticmethod]
    fn start_benchmark() {
        crate::benchmark_search::start();
    }

    #[cfg(feature = "benchmark")]
    #[staticmethod]
    fn finish_benchmark_json() -> PyResult<String> {
        serde_json::to_string(&crate::benchmark_search::finish())
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    #[new]
    #[pyo3(signature = (match_id, audit=false))]
    fn new(match_id: String, audit: bool) -> PyResult<Self> {
        if match_id.trim().is_empty() {
            return Err(PyAgentError::new_err(
                "selected branch match_id must not be empty",
            ));
        }
        Ok(Self {
            inner: Mutex::new(SelectedBranchRuntimeState {
                match_id,
                audit,
                next_command_id: 1,
                next_apply_sequence: 1,
                forks: SelectedSiteCounts::default(),
                applies: SelectedSiteCounts::default(),
                leaf_playouts: 0,
                leaf_cap_hits: 0,
                tapes: std::array::from_fn(|_| Vec::new()),
            }),
        })
    }

    #[getter]
    fn driver_id(&self) -> &'static str {
        SELECTED_BRANCH_DRIVER_ID
    }

    fn fork_exact(&self, source: &PyEnv, site: &str) -> PyResult<PyEnv> {
        let site = SelectedMutationSite::parse(site)?;
        let mut runtime = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("selected runtime lock poisoned"))?;
        let source = source
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        let fork = source.selected_fork().map_err(map_agent_err)?;
        runtime.forks.increment(site);
        Ok(PyEnv {
            inner: Arc::new(Mutex::new(fork)),
            selected_guard: true,
            prepared_lease: None,
        })
    }

    fn determinize(&self, branch: &PyEnv, seed: u64, perspective: usize) -> PyResult<()> {
        branch.require_selected_guard()?;
        let mut env = branch
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        env.selected_determinize(perspective, seed)
            .map_err(map_agent_err)
    }

    fn reseed_rollout(&self, branch: &PyEnv, seed: u64) -> PyResult<()> {
        branch.require_selected_guard()?;
        let mut env = branch
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        env.selected_reseed_rollout(seed).map_err(map_agent_err)
    }

    fn sample_policy_index(&self, branch: &PyEnv) -> PyResult<usize> {
        branch.require_selected_guard()?;
        let mut env = branch
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        env.random_action_index().map_err(map_agent_err)
    }

    #[pyo3(signature = (branch, site, policy_index, preconditions_json=None))]
    fn apply_policy_choice(
        &self,
        py: Python<'_>,
        branch: &PyEnv,
        site: &str,
        policy_index: usize,
        preconditions_json: Option<&str>,
    ) -> PyResult<(PyObservation, f64, bool, bool, PyObject, String)> {
        branch.require_selected_guard()?;
        let site = SelectedMutationSite::parse(site)?;
        let overrides = preconditions_json
            .map(serde_json::from_str::<SelectedPreconditions>)
            .transpose()
            .map_err(|error| {
                PyAgentError::new_err(format!("invalid selected preconditions: {error}"))
            })?
            .unwrap_or_default();

        let mut runtime = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("selected runtime lock poisoned"))?;
        let mut env = branch
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;

        let require_identity = runtime.audit
            || overrides.authority_hash.is_some()
            || overrides.legal_surface_hash.is_some();
        #[cfg(feature = "benchmark")]
        let identity_timer = crate::benchmark_search::Timer::new("witness_construction");
        let current_identity = require_identity
            .then(|| env.selected_identity())
            .transpose()
            .map_err(map_agent_err)?;
        #[cfg(feature = "benchmark")]
        drop(identity_timer);
        #[cfg(feature = "benchmark")]
        let validation_timer = crate::benchmark_search::Timer::new("command_validation");
        let current_revision = env.selected_revision().map_err(map_agent_err)?;
        let current_offers = env.structured_search_offers().map_err(map_agent_err)?;
        let current_offer = current_offers
            .projection()
            .offers
            .get(policy_index)
            .ok_or_else(|| {
                PyAgentError::new_err(format!(
                    "policy lookup key {policy_index} is out of range for {} offers",
                    current_offers.projection().offers.len()
                ))
            })?;

        let expected_offer_id = overrides.offer_id.unwrap_or(current_offer.id.0);
        let expected_prompt_id = overrides.prompt_id.unwrap_or(current_revision);
        let expected_revision = overrides.expected_revision.unwrap_or(current_revision);
        let expected_authority_hash = overrides.authority_hash.as_deref().or_else(|| {
            current_identity
                .as_ref()
                .map(|identity| identity.authority.hash.as_str())
        });
        let expected_legal_hash = overrides.legal_surface_hash.as_deref().or_else(|| {
            current_identity
                .as_ref()
                .map(|identity| identity.legal_surface.hash.as_str())
        });

        if current_offer.id.0 != expected_offer_id {
            return Err(PyAgentError::new_err("offer ID precondition mismatch"));
        }
        if current_revision != expected_prompt_id {
            return Err(PyAgentError::new_err("prompt ID precondition mismatch"));
        }
        if current_revision != expected_revision {
            return Err(PyAgentError::new_err("revision precondition mismatch"));
        }
        if expected_authority_hash.is_some_and(|expected| {
            current_identity
                .as_ref()
                .is_none_or(|identity| identity.authority.hash != expected)
        }) {
            return Err(PyAgentError::new_err("authority precondition mismatch"));
        }
        if expected_legal_hash.is_some_and(|expected| {
            current_identity
                .as_ref()
                .is_none_or(|identity| identity.legal_surface.hash != expected)
        }) {
            return Err(PyAgentError::new_err("legal-surface precondition mismatch"));
        }

        // All caller preconditions are checked above while both locks are held.
        // Construct the retained Command from those validated values.
        let command_id = format!("search.{}.{}", runtime.match_id, runtime.next_command_id);
        let command = ExperienceCommand {
            command_id: CommandId(command_id),
            match_id: MatchId(runtime.match_id.clone()),
            expected_revision: Revision(expected_revision),
            prompt_id: PromptId(expected_prompt_id),
            offer_id: ExperienceOfferId(expected_offer_id),
            answers: Vec::new(),
        };
        let submission = OfferSubmission {
            offer_id: StructuredOfferId(command.offer_id.0),
            answers: Vec::new(),
        };
        #[cfg(feature = "benchmark")]
        drop(validation_timer);
        let (observation, reward, terminated, truncated, info, native_actions) = env
            .step_structured(&current_offers, &submission)
            .map_err(map_agent_err)?;
        if native_actions != 1 {
            return Err(PyAgentError::new_err(format!(
                "selected structured apply reported {native_actions} native actions"
            )));
        }
        let post_witness = if runtime.audit {
            compact_search_witness(&env.selected_witness().map_err(map_agent_err)?)
        } else {
            Value::Null
        };

        #[cfg(feature = "benchmark")]
        let audit_timer = crate::benchmark_search::Timer::new("audit_record");
        let apply_sequence = runtime.next_apply_sequence;
        let site_sequence = runtime.applies.increment(site);
        runtime.next_apply_sequence += 1;
        runtime.next_command_id += 1;
        let record = json!({
            "site": site.as_str(),
            "policy_index": policy_index,
            "offer_id": expected_offer_id,
            "command": command,
            "source": {
                "prompt_id": expected_prompt_id,
                "expected_revision": expected_revision,
                "authority_hash": expected_authority_hash,
                "legal_surface_hash": expected_legal_hash,
            },
            "native_receipt": {
                "driver_id": SELECTED_BRANCH_DRIVER_ID,
                "apply_sequence": apply_sequence,
                "site_sequence": site_sequence,
                "accepted_apply_counter": runtime.applies.total(),
                "native_apply_count": native_actions,
                "terminal": terminated || truncated,
            },
            "post_apply_witness": post_witness,
        });
        if runtime.audit {
            runtime.tape_mut(site).push(record.clone());
        }

        let record_json = serde_json::to_string(&record)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        #[cfg(feature = "benchmark")]
        drop(audit_timer);
        #[cfg(feature = "benchmark")]
        let _conversion = crate::benchmark_search::Timer::new("returned_python_conversion");
        let py_info = info_dict_to_pydict(py, &info);
        Ok((
            PyObservation::from(observation),
            reward,
            terminated,
            truncated,
            py_info.into_any().unbind(),
            record_json,
        ))
    }

    #[pyo3(signature = (hit_cap=false))]
    fn record_leaf_playout(&self, hit_cap: bool) -> PyResult<()> {
        let mut runtime = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("selected runtime lock poisoned"))?;
        runtime.leaf_playouts += 1;
        runtime.leaf_cap_hits += u64::from(hit_cap);
        Ok(())
    }

    fn snapshot_json(&self) -> PyResult<String> {
        let runtime = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("selected runtime lock poisoned"))?;
        let tape_lengths = SelectedSiteCounts {
            world: runtime.tapes[0].len() as u64,
            child: runtime.tapes[1].len() as u64,
            leaf: runtime.tapes[2].len() as u64,
        };
        let reconciliation = if runtime.audit {
            runtime.applies.world == tape_lengths.world
                && runtime.applies.child == tape_lengths.child
                && runtime.applies.leaf == tape_lengths.leaf
                && runtime.applies.total() == tape_lengths.total()
        } else {
            true
        };
        serde_json::to_string(&json!({
            "driver_id": SELECTED_BRANCH_DRIVER_ID,
            "match_id": runtime.match_id,
            "audit": runtime.audit,
            "counters": {
                "forks": &runtime.forks,
                "applies": &runtime.applies,
                "marks": 0,
                "rollbacks": 0,
                "random_playouts": runtime.leaf_playouts,
                "random_playout_cap_hits": runtime.leaf_cap_hits,
                "indexed_fallbacks": 0,
            },
            "tape_lengths": tape_lengths,
            "tapes": {
                "world": &runtime.tapes[0],
                "child": &runtime.tapes[1],
                "leaf": &runtime.tapes[2],
            },
            "reconciliation": {
                "per_site_and_total": reconciliation,
                "zero_unmeasured_fallback": true,
            },
        }))
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl PyEnv {
    #[new]
    #[pyo3(signature = (seed=0, skip_trivial=true, enable_profiler=false, enable_behavior_tracking=false, *, player_policies=None, decision_memory_include_own=false, decision_memory=false))]
    fn new(
        py: Python<'_>,
        seed: u64,
        skip_trivial: bool,
        enable_profiler: bool,
        enable_behavior_tracking: bool,
        player_policies: Option<Vec<String>>,
        decision_memory_include_own: bool,
        decision_memory: bool,
    ) -> PyResult<Self> {
        if enable_behavior_tracking {
            py.import_bound("warnings")?.call_method1("warn", (
                "enable_behavior_tracking is a deprecated no-op; use managym.statistics.Behaviour; removal in 0.4.0",
                py.get_type_bound::<pyo3::exceptions::PyDeprecationWarning>(),
            ))?;
        }
        let policies =
            crate::python::vector_env_bindings::parse_player_policies(player_policies, None)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(
                Env::new(
                    seed,
                    skip_trivial,
                    enable_profiler,
                    enable_behavior_tracking,
                )
                .with_player_policies(policies)
                .with_decision_memory(decision_memory)
                .with_decision_memory_include_own(decision_memory_include_own),
            )),
            selected_guard: false,
            prepared_lease: None,
        })
    }

    fn reset(
        &self,
        py: Python<'_>,
        player_configs: Bound<'_, PyAny>,
    ) -> PyResult<(PyObservation, PyObject)> {
        let mut env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;

        let configs = game_setup(&player_configs)?;
        let (obs, info) = env.reset(configs).map_err(map_agent_err)?;

        let py_dict = info_dict_to_pydict(py, &info);
        Ok((PyObservation::from(obs), py_dict.into_any().unbind()))
    }

    fn enable_statistics(&self) -> PyResult<()> {
        self.reject_guarded_mutation("Env.enable_statistics")?;
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?
            .enable_statistics()
            .map_err(map_agent_err)
    }

    fn statistics(&self, py: Python<'_>) -> PyResult<PyObject> {
        let summary = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?
            .statistics();
        statistics_object(py, &summary, "summary")
    }

    fn take_statistics(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.reject_guarded_mutation("Env.take_statistics")?;
        let summaries = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?
            .take_statistics();
        statistics_object(py, &summaries, "summaries")
    }

    fn enable_profiler(&self) -> PyResult<()> {
        self.reject_guarded_mutation("Env.enable_profiler")?;
        let mut env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        if !env.profiler.is_enabled() {
            env.profiler = crate::infra::profiler::Profiler::new(true, 64);
        }
        Ok(())
    }

    fn enable_exposure_capture(&self) -> PyResult<()> {
        self.reject_guarded_mutation("Env.enable_exposure_capture")?;
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?
            .enable_exposure_capture()
            .map_err(map_agent_err)
    }

    fn enable_card_capture(&self) -> PyResult<()> {
        self.reject_guarded_mutation("Env.enable_card_capture")?;
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?
            .enable_card_capture()
            .map_err(map_agent_err)
    }

    fn card_capture_json(&self) -> PyResult<String> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?
            .card_capture_json()
            .map_err(map_agent_err)
    }

    fn set_seed(&self, seed: u64) -> PyResult<()> {
        let mut env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        env.set_seed(seed);
        Ok(())
    }

    #[pyo3(signature = (action=None))]
    fn step(
        &self,
        py: Python<'_>,
        action: Option<i64>,
    ) -> PyResult<(PyObservation, f64, bool, bool, PyObject)> {
        self.reject_guarded_mutation("Env.step(index)")?;
        let mut env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;

        let (obs, reward, terminated, truncated, info) =
            env.step(action.unwrap_or(-1)).map_err(map_agent_err)?;
        let py_dict = info_dict_to_pydict(py, &info);
        Ok((
            PyObservation::from(obs),
            reward,
            terminated,
            truncated,
            py_dict.into_any().unbind(),
        ))
    }

    /// Return the exact structured projection for the current supported
    /// priority or attacker decision.
    fn structured_offers(&self) -> PyResult<PyStructuredOfferSet> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        Ok(PyStructuredOfferSet {
            inner: env.structured_offers().map_err(map_agent_err)?,
        })
    }

    fn compound_offers(&self) -> PyResult<PyStructuredOfferSet> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        Ok(PyStructuredOfferSet {
            inner: env.compound_offers().map_err(map_agent_err)?,
        })
    }

    fn compound_commands_json(
        &self,
        offers: &PyStructuredOfferSet,
        submission_json: &str,
    ) -> PyResult<String> {
        let submission: OfferSubmission = serde_json::from_str(submission_json)
            .map_err(|error| PyAgentError::new_err(error.to_string()))?;
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        let commands = env
            .compound_commands(&offers.inner, &submission)
            .map_err(map_agent_err)?;
        serde_json::to_string(&commands).map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    /// Complete action-aligned structured surface used by the selected
    /// production search backend.
    fn structured_search_offers(&self) -> PyResult<PyStructuredOfferSet> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        Ok(PyStructuredOfferSet {
            inner: env.structured_search_offers().map_err(map_agent_err)?,
        })
    }

    /// Read-only revision, offer, and witness context for differential audit.
    #[pyo3(signature = (include_witness=true))]
    fn search_context_json(&self, include_witness: bool) -> PyResult<String> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        let revision = env.selected_revision().map_err(map_agent_err)?;
        let witness = if include_witness {
            compact_search_witness(&env.selected_witness().map_err(map_agent_err)?)
        } else {
            Value::Null
        };
        let offers = env.structured_search_offers().map_err(map_agent_err)?;
        serde_json::to_string(&json!({
            "revision": revision,
            "prompt_id": revision,
            "witness": witness,
            "offers": offers.projection(),
        }))
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    /// Compact representation-neutral witness, including terminal states.
    fn search_witness_json(&self) -> PyResult<String> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        let witness = env.selected_witness().map_err(map_agent_err)?;
        serde_json::to_string(&compact_search_witness(&witness))
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    /// Apply an ID-only submission through the atomic structured path.
    fn step_structured(
        &self,
        py: Python<'_>,
        offers: &PyStructuredOfferSet,
        submission_json: &str,
    ) -> PyResult<(PyObservation, f64, bool, bool, PyObject, usize)> {
        self.reject_guarded_mutation("direct Env.step_structured")?;
        let submission: OfferSubmission = serde_json::from_str(submission_json)
            .map_err(|error| PyAgentError::new_err(format!("invalid offer submission: {error}")))?;
        let mut env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        let (obs, reward, terminated, truncated, info, legacy_equivalent_actions) = env
            .step_structured(&offers.inner, &submission)
            .map_err(map_agent_err)?;
        let py_dict = info_dict_to_pydict(py, &info);
        Ok((
            PyObservation::from(obs),
            reward,
            terminated,
            truncated,
            py_dict.into_any().unbind(),
            legacy_equivalent_actions,
        ))
    }

    /// Apply the same semantic submission by walking the positional prompts.
    fn step_legacy_submission(
        &self,
        py: Python<'_>,
        offers: &PyStructuredOfferSet,
        submission_json: &str,
    ) -> PyResult<(PyObservation, f64, bool, bool, PyObject, usize)> {
        self.reject_guarded_mutation("Env.step_legacy_submission")?;
        let submission: OfferSubmission = serde_json::from_str(submission_json)
            .map_err(|error| PyAgentError::new_err(format!("invalid offer submission: {error}")))?;
        let mut env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        let (obs, reward, terminated, truncated, info, legacy_actions) = env
            .step_legacy_submission(&offers.inner, &submission)
            .map_err(map_agent_err)?;
        let py_dict = info_dict_to_pydict(py, &info);
        Ok((
            PyObservation::from(obs),
            reward,
            terminated,
            truncated,
            py_dict.into_any().unbind(),
            legacy_actions,
        ))
    }

    /// Canonical full-engine digest for exact differential assertions.
    fn state_digest(&self) -> PyResult<String> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        env.state_digest().map_err(map_agent_err)
    }

    fn info(&self, py: Python<'_>) -> PyResult<PyObject> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        let py_dict = info_dict_to_pydict(py, &env.info());
        Ok(py_dict.into_any().unbind())
    }

    /// Read-only binding manifest of the catalog the current match runs on.
    /// This is deliberately not repeated on observations.
    fn content_pack_manifest(&self, py: Python<'_>) -> PyResult<PyObject> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        let manifest = env.content_pack_manifest().map_err(map_agent_err)?;

        let out = PyDict::new_bound(py);
        out.set_item("schema_version", manifest.schema_version)?;
        out.set_item("content_digest", manifest.content_digest)?;
        let definitions = PyList::empty_bound(py);
        for entry in manifest.definitions {
            let definition = PyDict::new_bound(py);
            definition.set_item("card_def_id", entry.card_def_id.0)?;
            definition.set_item("key", entry.key)?;
            definition.set_item("registry_name", entry.registry_name)?;
            definitions.append(definition)?;
        }
        out.set_item("definitions", definitions)?;
        Ok(out.into_any().unbind())
    }

    /// Number of trivial decision points auto-collapsed by `skip_trivial`
    /// since the current game began. Resets to zero on `reset`.
    fn skip_trivial_count(&self) -> PyResult<usize> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        Ok(env.skip_trivial_count())
    }

    fn export_profile_baseline(&self) -> PyResult<String> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        Ok(env.export_profile_baseline())
    }

    fn compare_profile(&self, baseline: String) -> PyResult<String> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        Ok(env.compare_profile(&baseline))
    }

    /// Independent copy of this env (current game state cloned; stepping the
    /// clone never mutates the original). Profiling/tracking disabled.
    fn clone_env(&self) -> PyResult<PyEnv> {
        self.reject_guarded_mutation("Env.clone_env")?;
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        let fork = env.fork().map_err(map_agent_err)?;
        Ok(PyEnv {
            inner: Arc::new(Mutex::new(fork)),
            selected_guard: false,
            prepared_lease: self
                .prepared_lease
                .as_ref()
                .map(PreparedBranchLease::clone_lease),
        })
    }

    /// Player index holding the current decision, or None.
    fn current_agent_index(&self) -> PyResult<Option<usize>> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        Ok(env.current_agent_index())
    }

    /// Detached authoritative hand contents, including hidden information.
    fn hand_snapshot(&self, player: usize) -> PyResult<HandSnapshot> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        env.hand_snapshot(player).map_err(map_agent_err)
    }

    /// Return a fixed-viewer projection even when another player now acts.
    fn observation_for_player(&self, player_index: usize) -> PyResult<PyObservation> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        Ok(PyObservation::from(
            env.observation_for_player(player_index)
                .map_err(map_agent_err)?,
        ))
    }

    /// Read-only automatic payment plan for one offer; see `managym.payment`.
    fn payment_preview_json(
        &self,
        viewer: usize,
        revision: u64,
        fingerprint: &str,
        action_index: usize,
    ) -> PyResult<String> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        let preview = env
            .payment_preview(viewer, revision, fingerprint, action_index)
            .map_err(map_agent_err)?;
        serde_json::to_string(&preview).map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    /// Read-only whole combat declaration preview for a JSON plan; see
    /// `managym.combat`.
    fn combat_preview_json(
        &self,
        viewer: usize,
        revision: u64,
        fingerprint: &str,
        plan_json: &str,
    ) -> PyResult<String> {
        let plan: crate::flow::combat_preview::CombatPlan = serde_json::from_str(plan_json)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        let preview = env
            .combat_preview(viewer, revision, fingerprint, &plan)
            .map_err(map_agent_err)?;
        serde_json::to_string(&preview).map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    /// Shared semantic DecisionFrame for the current revision, as canonical
    /// JSON. See `managym.decision.DecisionFrame`.
    fn semantic_decision_frame_json(&self) -> PyResult<String> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        let frame = env.semantic_decision_frame().map_err(map_agent_err)?;
        serde_json::to_string(&frame).map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    /// Public event identities used to explain a committed transition.
    fn presentation_subjects_json(&self) -> PyResult<String> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        let subjects = env.presentation_subjects().map_err(map_agent_err)?;
        serde_json::to_string(&subjects).map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    fn decision_offer_subjects_json(&self, viewer: usize) -> PyResult<String> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        let subjects = env.decision_offer_subjects(viewer).map_err(map_agent_err)?;
        serde_json::to_string(&subjects).map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    fn public_schedule_json(&self) -> PyResult<String> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        let schedule = env.public_schedule().map_err(map_agent_err)?;
        serde_json::to_string(&schedule).map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    fn player_controls_json(&self) -> PyResult<String> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        let controls = env.player_controls().map_err(map_agent_err)?;
        serde_json::to_string(&controls).map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    fn decision_context_json(&self, viewer: usize) -> PyResult<String> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        let context = env.decision_context(viewer).map_err(map_agent_err)?;
        serde_json::to_string(&context).map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    #[pyo3(signature = (viewer, history_capacity=0))]
    fn capture_decision(
        &self,
        viewer: usize,
        history_capacity: usize,
    ) -> PyResult<PyDecisionCapture> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        Ok(PyDecisionCapture {
            source: Arc::clone(&self.inner),
            generation: env.capture_generation,
            capture: env
                .capture_decision(viewer, history_capacity)
                .map_err(map_agent_err)?,
        })
    }

    /// Validate origin and all included input facts under the same lock as the
    /// ordinary Command execution. A clone or a reset is a different authority.
    fn execute_captured_command_json(
        &self,
        capture: &PyDecisionCapture,
        command_json: &str,
    ) -> PyResult<String> {
        self.reject_guarded_mutation("Env.execute_captured_command_json")?;
        if !Arc::ptr_eq(&self.inner, &capture.source) {
            return Err(PyAgentError::new_err(
                "wrong_origin: capture belongs to another environment/branch",
            ));
        }
        let command: SemanticCommand = serde_json::from_str(command_json)
            .map_err(|error| PyAgentError::new_err(format!("invalid semantic command: {error}")))?;
        let mut env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        if env.capture_generation != capture.generation {
            return Err(PyAgentError::new_err(
                "stale_capture: environment was reset",
            ));
        }
        let current = env
            .capture_decision(
                capture.capture.observation.identity.viewer as usize,
                capture.capture.history.capacity,
            )
            .map_err(|error| PyAgentError::new_err(format!("stale_capture: {error}")))?;
        if current.digest != capture.capture.digest {
            return Err(PyAgentError::new_err(
                "stale_capture: captured input has changed",
            ));
        }
        let transition = env
            .execute_semantic_command(&command)
            .map_err(map_agent_err)?;
        serde_json::to_string(&transition)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    /// Composite viewer-safe Observation for `viewer`, as canonical JSON.
    fn semantic_observation_json(&self, viewer: usize) -> PyResult<String> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        let observation = env.semantic_observation(viewer).map_err(map_agent_err)?;
        #[cfg(feature = "benchmark")]
        let _serialization =
            crate::benchmark_search::Timer::new("semantic/composite_serialization");
        serde_json::to_string(&observation)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    /// Exact semantic viewer-state SHA-256, rebuilt under the Env lock.
    fn semantic_viewer_state_hash(&self, viewer: usize) -> PyResult<String> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        env.semantic_viewer_state_hash(viewer)
            .map_err(map_agent_err)
    }

    /// Viewer-safe hand-count constraints; never enumerates possible worlds.
    fn hidden_hand_constraints_json(&self, viewer: usize) -> PyResult<String> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        let projection = env.hidden_hand_constraints(viewer).map_err(map_agent_err)?;
        serde_json::to_string(&projection)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    /// Diagnostic count of full-support constructions on this Env.
    fn possible_world_space_construction_count(&self) -> PyResult<u64> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        Ok(env.possible_world_space_construction_count())
    }

    /// Canonical viewer-relative PossibleWorldSpace as read-only JSON.
    fn possible_world_space_json(&self, viewer: usize) -> PyResult<String> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        let projection = env.possible_world_space(viewer).map_err(map_agent_err)?;
        serde_json::to_string(&projection)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    /// Prepare one canonical space and retain this exact live root for bounded
    /// indexed materialization across inference batches.
    fn prepare_possible_world_materializer(
        &self,
        viewer: usize,
        expected_space_identity: &str,
        max_batch_size: usize,
    ) -> PyResult<PyPreparedPossibleWorldMaterializer> {
        let prepared = {
            let env = self
                .inner
                .lock()
                .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
            env.prepare_possible_world_materializer(viewer, expected_space_identity, max_batch_size)
                .map_err(map_agent_err)?
        };
        let diagnostics = PreparedMaterializerDiagnostics {
            space_constructions: prepared.construction_count(),
            ..PreparedMaterializerDiagnostics::default()
        };
        Ok(PyPreparedPossibleWorldMaterializer {
            inner: prepared,
            source: Arc::clone(&self.inner),
            diagnostics: Mutex::new(diagnostics),
            branch_leases: Arc::new(PreparedBranchLeaseCounters::default()),
        })
    }

    /// Evaluate one typed WorldQuery against an identity-bound canonical
    /// space and return its canonical support receipt as JSON.
    fn possible_world_support_json(
        &self,
        viewer: usize,
        space_identity: &str,
        query_json: &str,
    ) -> PyResult<String> {
        let query: crate::possible_worlds::WorldQueryWire = serde_json::from_str(query_json)
            .map_err(|error| {
                PyAgentError::new_err(format!("invalid possible-world query: {error}"))
            })?;
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        let receipt = env
            .possible_world_support(viewer, space_identity, query)
            .map_err(map_agent_err)?;
        serde_json::to_string(&receipt).map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    /// Select canonical world indexes through Rules' typed query evaluator.
    fn possible_world_condition_json(
        &self,
        viewer: usize,
        space_identity: &str,
        query_json: &str,
    ) -> PyResult<String> {
        let query: crate::possible_worlds::WorldQueryWire = serde_json::from_str(query_json)
            .map_err(|error| {
                PyAgentError::new_err(format!("invalid possible-world query: {error}"))
            })?;
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        let receipt = env
            .possible_world_condition(viewer, space_identity, query)
            .map_err(map_agent_err)?;
        serde_json::to_string(&receipt).map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    /// Materialize sampled definition counts against a complete public snapshot.
    fn materialize_sampled_hand(
        &self,
        viewer: usize,
        constraints_json: &str,
        hand: std::collections::BTreeMap<String, u32>,
        seed: u64,
    ) -> PyResult<PyEnv> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        let branch = env
            .materialize_sampled_hand(viewer, constraints_json, &hand, seed)
            .map_err(map_agent_err)?;
        Ok(PyEnv {
            inner: Arc::new(Mutex::new(branch)),
            selected_guard: false,
            prepared_lease: None,
        })
    }

    /// Materialize one identity-bound canonical world into an isolated Env.
    #[pyo3(signature = (viewer, space_identity, world_index, seed, refresh_opponent_commitment=false))]
    fn materialize_possible_world(
        &self,
        viewer: usize,
        space_identity: &str,
        world_index: usize,
        seed: u64,
        refresh_opponent_commitment: bool,
    ) -> PyResult<PyEnv> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        let branch = env
            .materialize_possible_world(
                viewer,
                space_identity,
                world_index,
                seed,
                refresh_opponent_commitment,
            )
            .map_err(map_agent_err)?;
        Ok(PyEnv {
            inner: Arc::new(Mutex::new(branch)),
            selected_guard: false,
            prepared_lease: None,
        })
    }

    /// Validate and atomically apply one revision-bound semantic Command
    /// (`{"command_id","expected_revision","offer_id","answers"}`), returning
    /// canonical JSON of `{"receipt","observation"}`. Fails closed without
    /// mutation on stale, unknown, or illegal commands.
    fn execute_semantic_command_json(&self, command_json: &str) -> PyResult<String> {
        self.reject_guarded_mutation("Env.execute_semantic_command_json")?;
        let command: SemanticCommand = serde_json::from_str(command_json)
            .map_err(|error| PyAgentError::new_err(format!("invalid semantic command: {error}")))?;
        let mut env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        let transition = env
            .execute_semantic_command(&command)
            .map_err(map_agent_err)?;
        serde_json::to_string(&transition)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    /// Apply the shared semantic Command and return the ordinary acting
    /// observation needed by Academy presentation and trace consumers.
    fn step_semantic_command(
        &self,
        py: Python<'_>,
        command_json: &str,
    ) -> PyResult<(String, PyObservation, f64, bool, bool, PyObject)> {
        self.reject_guarded_mutation("Env.step_semantic_command")?;
        let command: SemanticCommand = serde_json::from_str(command_json)
            .map_err(|error| PyAgentError::new_err(format!("invalid semantic command: {error}")))?;
        let mut env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        let (transition, observation, reward, terminated, truncated, info) =
            env.step_semantic_command(&command).map_err(map_agent_err)?;
        let transition = serde_json::to_string(&transition)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let py_dict = info_dict_to_pydict(py, &info);
        Ok((
            transition,
            PyObservation::from(observation),
            reward,
            terminated,
            truncated,
            py_dict.into_any().unbind(),
        ))
    }

    /// Cursor over committed semantic events. This is diagnostic identity,
    /// never a mutation or replay API.
    fn semantic_event_cursor(&self) -> PyResult<u64> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        env.semantic_event_cursor().map_err(map_agent_err)
    }

    fn is_game_over(&self) -> PyResult<bool> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        Ok(env.is_game_over())
    }

    fn winner_index(&self) -> PyResult<Option<usize>> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        Ok(env.winner_index())
    }

    /// Number of legal actions in the current action space.
    fn action_count(&self) -> PyResult<usize> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        env.action_count().map_err(map_agent_err)
    }

    // ------------------------------------------------------------------
    // Scenario / state-injection helpers (flow/scenario.rs).
    //
    // FOR TEST AND MEASUREMENT HARNESSES ONLY — they mutate state directly,
    // bypassing the rules engine (no events, no triggers, no costs). Used by
    // manabot/verify/competency.py to construct scored mid-game positions.
    // Priority injections need scenario_refresh() once. The naming fixture
    // instead requires a pending name decision and republishes it directly.
    // ------------------------------------------------------------------

    /// Scenario harness only: install the checked-in large naming vocabulary.
    fn scenario_naming_vocabulary(&self) -> PyResult<Vec<crate::cards::names::NameCandidate>> {
        self.reject_guarded_mutation("scenario_naming_vocabulary")?;
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?
            .scenario_naming_vocabulary()
            .map(|names| names.iter().cloned().collect())
            .map_err(map_agent_err)
    }

    /// Scenario harness: set a player's life total directly.
    fn scenario_set_life(&self, player: usize, life: i32) -> PyResult<()> {
        let mut env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        env.scenario_set_life(player, life).map_err(map_agent_err)
    }

    /// Scenario harness: move a player's entire hand to the bottom of their
    /// library.
    fn scenario_clear_hand(&self, player: usize) -> PyResult<()> {
        let mut env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        env.scenario_clear_hand(player).map_err(map_agent_err)
    }

    /// Scenario harness: move one card named `name` from the player's
    /// library (or graveyard) into their hand.
    fn scenario_force_card_in_hand(&self, player: usize, name: &str) -> PyResult<()> {
        let mut env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        env.scenario_force_card_in_hand(player, name)
            .map_err(map_agent_err)
    }

    /// Scenario harness: put a card named `name` onto the battlefield as a
    /// new permanent (no ETB triggers). `ready` clears summoning sickness.
    /// Returns the new permanent index.
    #[pyo3(signature = (player, name, ready=true))]
    fn scenario_force_battlefield(
        &self,
        player: usize,
        name: &str,
        ready: bool,
    ) -> PyResult<usize> {
        let mut env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        env.scenario_force_battlefield(player, name, ready)
            .map_err(map_agent_err)
    }

    /// Scenario harness: recompute the current priority action space after
    /// injections and return a fresh observation.
    fn scenario_refresh(&self) -> PyResult<PyObservation> {
        let mut env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        let obs = env.scenario_refresh().map_err(map_agent_err)?;
        Ok(PyObservation::from(obs))
    }

    /// Resample hidden information (opponent hand + both library orders)
    /// consistent with `perspective`'s observation. Defaults to the player
    /// holding the current decision. Public state is preserved.
    #[pyo3(signature = (seed, perspective=None))]
    fn determinize(&self, seed: u64, perspective: Option<usize>) -> PyResult<()> {
        self.reject_guarded_mutation("direct Env.determinize")?;
        let mut env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        let perspective = match perspective {
            Some(p) => p,
            None => env.current_agent_index().ok_or_else(|| {
                PyRuntimeError::new_err("determinize: no current decision player")
            })?,
        };
        env.determinize(perspective, seed).map_err(map_agent_err)
    }

    /// Play both sides uniformly-random-legal to terminal from the current
    /// state. Returns the winner index, or None on draw / step cap.
    #[pyo3(signature = (seed, max_steps=2000))]
    fn random_playout(
        &self,
        py: Python<'_>,
        seed: u64,
        max_steps: usize,
    ) -> PyResult<Option<usize>> {
        self.reject_guarded_mutation("Game::random_playout")?;
        py.allow_threads(|| {
            let mut env = self
                .inner
                .lock()
                .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
            env.random_playout(seed, max_steps).map_err(map_agent_err)
        })
    }

    /// Reference-backend RNG setup. Guarded selected branches must use the
    /// selected runtime method instead.
    fn reseed_rollout(&self, seed: u64) -> PyResult<()> {
        self.reject_guarded_mutation("direct rollout reseed")?;
        let mut env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        env.selected_reseed_rollout(seed).map_err(map_agent_err)
    }

    /// Reference-backend uniform policy sampling. This only reads the current
    /// action count and advances the branch-local RNG.
    fn random_action_index(&self) -> PyResult<usize> {
        self.reject_guarded_mutation("direct policy sampling")?;
        let mut env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        env.random_action_index().map_err(map_agent_err)
    }

    /// Flat determinized Monte Carlo evaluation of the current action space.
    ///
    /// Returns (scores, simulations, cap_hits): mean playout score per legal
    /// action for the player holding the decision (win 1.0 / loss 0.0 /
    /// draw-or-cap 0.5), total playouts, and playouts that hit the step cap.
    #[pyo3(signature = (worlds, rollouts, seed, max_steps=2000))]
    fn flat_mc_scores(
        &self,
        py: Python<'_>,
        worlds: usize,
        rollouts: usize,
        seed: u64,
        max_steps: usize,
    ) -> PyResult<(Vec<f64>, u64, u64)> {
        py.allow_threads(|| {
            let env = self
                .inner
                .lock()
                .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
            let result = env
                .flat_mc_scores(worlds, rollouts, seed, max_steps)
                .map_err(map_agent_err)?;
            Ok((result.scores, result.simulations, result.cap_hits))
        })
    }

    /// Flat MC over caller-sampled canonical world indexes. Returns no hidden
    /// hand identities; the authoritative materializer validates each row.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (viewer, space_identity, world_indexes, world_seeds, rollouts, max_steps=2000))]
    fn flat_mc_scores_for_worlds(
        &self,
        py: Python<'_>,
        viewer: usize,
        space_identity: &str,
        world_indexes: Vec<usize>,
        world_seeds: Vec<u64>,
        rollouts: usize,
        max_steps: usize,
    ) -> PyResult<(Vec<f64>, u64, u64)> {
        py.allow_threads(|| {
            let env = self
                .inner
                .lock()
                .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
            let result = env
                .flat_mc_scores_for_worlds(
                    viewer,
                    space_identity,
                    &world_indexes,
                    &world_seeds,
                    rollouts,
                    max_steps,
                )
                .map_err(map_agent_err)?;
            Ok((result.scores, result.simulations, result.cap_hits))
        })
    }

    /// Batched policy-rollout pool for the current decision point. See
    /// `RolloutPool`: worlds x rollouts simulations per legal action, root
    /// actions pre-applied, ready for batched policy-driven stepping.
    #[pyo3(signature = (worlds, rollouts, seed, max_steps=2000))]
    fn rollout_pool(
        &self,
        py: Python<'_>,
        worlds: usize,
        rollouts: usize,
        seed: u64,
        max_steps: usize,
    ) -> PyResult<crate::python::vector_env_bindings::PyRolloutPool> {
        py.allow_threads(|| {
            let env = self
                .inner
                .lock()
                .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
            let pool = env
                .rollout_pool(worlds, rollouts, seed, max_steps)
                .map_err(map_agent_err)?;
            Ok(crate::python::vector_env_bindings::PyRolloutPool::from_inner(pool))
        })
    }

    fn decision_memory_counts(&self) -> PyResult<[[usize; 4]; 2]> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?
            .decision_memory_counts()
            .map_err(map_agent_err)
    }

    /// Deprecated: encode with native default capacities. Removal in 0.4.0.
    #[pyo3(signature = (obs, max_facts=None))]
    fn encode_observation(
        &self,
        py: Python<'_>,
        obs: PyObservation,
        max_facts: Option<usize>,
    ) -> PyResult<PyObject> {
        warn_deprecated(
            py,
            "Env.encode_observation is deprecated; allocate buffers from \
             managym.observation_layout() or ObservationSpace and call \
             managym.encode_observation_into; removal in 0.4.0",
        )?;
        let rust_obs = Observation::from(obs);
        if max_facts == Some(0) {
            return Err(PyValueError::new_err(
                "max_facts requires a positive capacity",
            ));
        }
        let config = ObservationEncoderConfig {
            max_facts: max_facts.unwrap_or(ObservationEncoderConfig::default().max_facts),
            ..ObservationEncoderConfig::default()
        };
        let encoded = crate::agent::observation_encoder::encode(&rust_obs, &config)
            .map_err(map_encode_err)?;
        let out = encoded_to_dict(py, encoded, &config)?;
        Ok(out.into_any().unbind())
    }

    /// Deprecated alias of `managym.encode_observation_into`. Removal in 0.4.0.
    #[pyo3(signature = (obs, out, action_overflow="error"))]
    fn encode_observation_into(
        &self,
        py: Python<'_>,
        obs: PyObservation,
        out: Bound<'_, PyDict>,
        action_overflow: &str,
    ) -> PyResult<()> {
        warn_deprecated(
            py,
            "Env.encode_observation_into is deprecated; use \
             managym.encode_observation_into; removal in 0.4.0",
        )?;
        let config = crate::python::vector_env_bindings::input_buffer_config(&out)?;
        let layout = super::layout_bindings::ObservationLayout::from_config(config)?;
        encode_observation_into(py, obs, &layout, out, action_overflow)
    }
}

/// Write any observation (live, restored or a viewer projection) into
/// caller-owned buffers at the supplied layout;
/// companion shapes are checked before any write.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (obs, layout, out, action_overflow="error"))]
fn encode_observation_into(
    py: Python<'_>,
    obs: PyObservation,
    layout: &super::layout_bindings::ObservationLayout,
    out: Bound<'_, PyDict>,
    action_overflow: &str,
) -> PyResult<()> {
    #[cfg(feature = "benchmark")]
    let conversion_timer = crate::benchmark_search::Timer::new("encoding/python_conversion");
    let rust_obs = Observation::from(obs);
    #[cfg(feature = "benchmark")]
    drop(conversion_timer);
    #[cfg(feature = "benchmark")]
    let config_timer = crate::benchmark_search::Timer::new("encoding/buffer_config");
    let mut config = layout.config;
    config.action_overflow = action_overflow.parse().map_err(PyValueError::new_err)?;
    crate::python::vector_env_bindings::check_observation_buffers(&out, layout)?;
    #[cfg(feature = "benchmark")]
    drop(config_timer);
    let encoded =
        crate::agent::observation_encoder::encode(&rust_obs, &config).map_err(map_encode_err)?;

    #[cfg(feature = "benchmark")]
    let _copy_timer = crate::benchmark_search::Timer::new("encoding/buffer_copy");
    // Stage the complete encoding before touching the validated caller buffers.
    let copyto = py.import_bound("numpy")?.getattr("copyto")?;
    for (key, source) in encoded_to_dict(py, encoded, &config)?.iter() {
        copyto.call1((out.as_any().get_item(key)?, source))?;
    }
    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (count, capacity, action_overflow="error"))]
fn policy_action_count(count: usize, capacity: usize, action_overflow: &str) -> PyResult<usize> {
    let overflow = action_overflow.parse().map_err(PyValueError::new_err)?;
    crate::agent::observation_encoder::policy_action_count(count, capacity, overflow)
        .map_err(map_encode_err)
}

/// The state only policy input v10 holds that `obs` needs, if any (see
/// `observation_encoder::input_v10_state`); a v9 encoder refuses it.
#[cfg(feature = "python")]
#[pyfunction]
fn input_v10_state(obs: PyObservation) -> Option<&'static str> {
    crate::agent::observation_encoder::input_v10_state(&Observation::from(obs))
}

/// State in `obs` the policy input cannot represent, if any (see
/// `observation_encoder::unrepresentable_state`); every encoder refuses it.
#[cfg(feature = "python")]
#[pyfunction]
fn unrepresentable_state(obs: PyObservation) -> Option<String> {
    crate::agent::observation_encoder::unrepresentable_state(&Observation::from(obs))
}

#[cfg(feature = "python")]
fn authored_setup(name: &str) -> PyResult<&'static crate::cards::setups::AuthoredSetup> {
    crate::cards::setups::authored_setup(name)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Distinct card names this world does not support for deck entry, sorted.
/// Eligibility only; deck construction policy belongs to deck admission.
#[cfg(feature = "python")]
#[pyfunction]
fn unsupported_cards(names: Vec<String>) -> Vec<String> {
    crate::world::unsupported_cards(names.iter().map(String::as_str))
}

#[cfg(feature = "python")]
#[pymethods]
impl crate::cards::names::NameCandidate {
    #[new]
    #[pyo3(signature = (name, definition=None))]
    fn py_new(name: String, definition: Option<u32>) -> PyResult<Self> {
        if name.is_empty() {
            return Err(PyValueError::new_err("empty card name"));
        }
        Ok(Self {
            name,
            definition: definition.map(crate::state::card::CardDefId),
        })
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    #[getter]
    fn definition(&self) -> Option<u32> {
        self.definition.map(|id| id.0)
    }
}

/// Public naming vocabulary in Command order, with learning definition joins.
#[cfg(feature = "python")]
#[pyfunction]
fn naming_catalog() -> Vec<crate::cards::names::NameCandidate> {
    crate::cards::content_pack::default_content_pack()
        .name_candidates()
        .iter()
        .cloned()
        .collect()
}

/// Learning tokens for every catalog definition, derived from the engine's card
/// definitions. Rows are `CardDefId`s; `content_digest` names the content they
/// encode.
#[cfg(feature = "python")]
#[pyfunction]
fn catalog_tokens(py: Python<'_>) -> PyResult<PyObject> {
    let content = crate::cards::content_pack::default_content_pack();
    let catalog = crate::cards::tokens::catalog_tokens(content)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let out = PyDict::new_bound(py);
    out.set_item("content_digest", content.content_digest())?;
    // Binding only: a row's key, or its name when it has none, joins a
    // manifest's CardDefIds to catalog rows.
    out.set_item(
        "keys",
        content
            .definitions()
            .map(|definition| (!definition.key.is_empty()).then_some(definition.key))
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "registry_names",
        content
            .definitions()
            .map(|definition| definition.name.clone())
            .collect::<Vec<_>>(),
    )?;
    out.set_item("token_kind", catalog.token_kind)?;
    out.set_item("token_value", catalog.token_value)?;
    out.set_item("definition_offsets", catalog.definition_offsets)?;
    out.set_item("program_offsets", catalog.program_offsets)?;
    out.set_item(
        "definition_program_offsets",
        catalog.definition_program_offsets,
    )?;
    out.set_item("definition_program_rows", catalog.definition_program_rows)?;
    out.set_item(
        "definition_ref_source_tokens",
        catalog.definition_ref_source_tokens,
    )?;
    out.set_item(
        "definition_ref_target_rows",
        catalog.definition_ref_target_rows,
    )?;
    Ok(out.into())
}

/// Every catalog definition as the JSON its learning tokens serialize, in
/// `CardDefId` order. A read-only view of the engine's card definitions for
/// Python consumers such as the heuristic player.
#[cfg(feature = "python")]
#[pyfunction]
fn card_definitions_json() -> PyResult<String> {
    let content = crate::cards::content_pack::default_content_pack();
    let definitions = content.definitions().collect::<Vec<_>>();
    serde_json::to_string(&definitions).map_err(|error| PyValueError::new_err(error.to_string()))
}

/// The stable token value the learning catalog uses for a Rust name
/// (`"Effect::DealDamage"`, a field, or a printed category such as `"Ally"`).
#[cfg(feature = "python")]
#[pyfunction]
fn catalog_token_value(name: &str) -> i32 {
    crate::cards::tokens::name_value(name)
}

#[cfg(feature = "python")]
type Renamed<K> = (K, &'static str, Option<&'static str>);

/// Historical (token kind, old name, successor) and (record namespace,
/// old name, successor) entries. Definition references instead follow the
/// referenced definition identity.
#[cfg(feature = "python")]
#[pyfunction]
fn catalog_token_renames() -> (Vec<Renamed<u16>>, Vec<Renamed<&'static str>>) {
    use crate::cards::token_renames::{RECORDS, TOKENS};
    (
        TOKENS
            .iter()
            .map(|&(kind, old, new)| (kind as u16, old, new))
            .collect(),
        RECORDS.to_vec(),
    )
}

/// `(namespace, name, value, token_kind, token_value)` of one record name.
#[cfg(feature = "python")]
type RecordNameRow = (&'static str, String, i32, Option<u16>, Option<i32>);

/// Every name the fact and shared-effect encoders hash; the token is the
/// learning token card text uses for the same concept, or `None`.
#[cfg(feature = "python")]
#[pyfunction]
fn record_vocabulary() -> Vec<RecordNameRow> {
    crate::agent::record_vocabulary::record_names()
        .into_iter()
        .map(|name| {
            (
                name.namespace.as_str(),
                name.name,
                name.value,
                name.token.map(|(kind, _)| kind),
                name.token.map(|(_, value)| value),
            )
        })
        .collect()
}

/// The world pool by its public name (`"world"`, the only pool).
#[cfg(feature = "python")]
#[pyfunction]
fn content_pool(py: Python<'_>, name: &str) -> PyResult<PyObject> {
    let pool =
        crate::cards::content_pool(name).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let out = PyDict::new_bound(py);
    out.set_item("content_digest", &pool.content_digest)?;
    out.set_item("definitions", pool_rows(py, &pool.definitions)?)?;
    Ok(out.into())
}

/// The installed world's card digest.
#[cfg(feature = "python")]
#[pyfunction]
fn world_identity(py: Python<'_>) -> PyResult<PyObject> {
    let identity = crate::world::world_identity();
    let out = PyDict::new_bound(py);
    out.set_item("card_digest", identity.card_digest)?;
    Ok(out.into())
}

/// World pool rows for typed cards whose key prefix is one of `sets`, such as
/// `["tla", "lea"]` for TLA cards with Alpha's basic lands.
#[cfg(feature = "python")]
#[pyfunction]
fn set_cards(py: Python<'_>, sets: Vec<String>) -> PyResult<PyObject> {
    let sets = sets.iter().map(String::as_str).collect::<Vec<_>>();
    Ok(pool_rows(py, &crate::world::set_cards(&sets))?.into())
}

#[cfg(feature = "python")]
fn pool_rows<'py>(
    py: Python<'py>,
    rows: &[crate::cards::PoolDefinition],
) -> PyResult<Bound<'py, PyList>> {
    let definitions = PyList::empty_bound(py);
    for row in rows {
        let item = PyDict::new_bound(py);
        item.set_item("card_def_id", row.card_def_id.0)?;
        item.set_item("key", &row.key)?;
        item.set_item("scryfall_id", &row.scryfall_id)?;
        item.set_item("registry_name", &row.registry_name)?;
        item.set_item(
            "eligibility",
            match row.eligibility {
                crate::cards::Eligibility::Eligible => "eligible",
                crate::cards::Eligibility::Unsupported => "unsupported",
            },
        )?;
        item.set_item("reason", &row.reason)?;
        definitions.append(item)?;
    }
    Ok(definitions)
}

#[cfg(feature = "python")]
#[pyfunction]
fn parse_continuation_frames_json(
    text: &str,
) -> PyResult<Vec<crate::agent::continuation::ContinuationFrame>> {
    let mut value: serde_json::Value =
        serde_json::from_str(text).map_err(|e| PyValueError::new_err(e.to_string()))?;
    // Only the added nullable fact may be absent in historical captures.
    if let Some(frames) = value.as_array_mut() {
        for frame in frames {
            if let Some(targets) = frame.get_mut("targets").and_then(|v| v.as_array_mut()) {
                for target in targets {
                    if let Some(target) = target.as_object_mut() {
                        target
                            .entry("legal_at_resolution_start")
                            .or_insert(serde_json::Value::Null);
                    }
                }
            }
        }
    }
    let frames: Vec<crate::agent::continuation::ContinuationFrame> =
        serde_json::from_value(value.clone()).map_err(|e| PyValueError::new_err(e.to_string()))?;
    if frames.iter().enumerate().any(|(index, frame)| {
        !(1..=14).contains(&frame.kind)
            || (frame.kind == 10
                && (!(1..=crate::agent::continuation::MAX_DELAYED_EVENT_CODE)
                    .contains(&frame.decision)
                    || frame.parent.is_some()))
            || (frame.kind != 10 && frame.this_turn)
            || frame
                .delayed_frequency
                .is_some_and(|frequency| frame.kind != 10 || !(1..=2).contains(&frequency))
            || (frame.kind != 5
                && frame
                    .targets
                    .iter()
                    .any(|t| t.legal_at_resolution_start.is_some()))
            || frame.decision > crate::agent::continuation::MAX_DECISION_CODE
            || frame.parent.is_some_and(|parent| parent >= index)
    }) {
        return Err(PyValueError::new_err(
            "invalid continuation frame relationship",
        ));
    }
    if serde_json::to_value(&frames).unwrap() != value {
        return Err(PyValueError::new_err("noncanonical continuation frames"));
    }
    Ok(frames)
}

#[cfg(feature = "python")]
#[pyfunction(signature = (text, *, schema_version=crate::decision::SEMANTIC_DECISION_VERSION))]
fn validate_observation_json(text: &str, schema_version: u16) -> PyResult<()> {
    // Historical capture readers explicitly select 7. Live readers keep the
    // current default; validation grants no origin or execution authority.
    if !matches!(
        schema_version,
        7 | crate::decision::SEMANTIC_DECISION_VERSION
    ) {
        return Err(PyValueError::new_err(
            "unsupported semantic observation schema",
        ));
    }
    let value: serde_json::Value =
        serde_json::from_str(text).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let observation: crate::decision::Observation =
        serde_json::from_value(value.clone()).map_err(|e| PyValueError::new_err(e.to_string()))?;
    if serde_json::to_value(&observation).unwrap() != value
        || observation.identity.schema_version != schema_version
        || observation
            .decision
            .as_ref()
            .is_some_and(|frame| frame.schema_version != schema_version)
    {
        return Err(PyValueError::new_err("noncanonical semantic observation"));
    }
    let viewer = crate::agent::observation_json::parse(&observation.viewer_state.to_string())
        .map_err(PyValueError::new_err)?;
    if schema_version == 7
        && viewer.action_space.actions.iter().any(|action| {
            action.name_index.is_some()
                || action.name_definition.is_some()
                || action.name_group.is_some()
        })
    {
        return Err(PyValueError::new_err(
            "schema 7 has no public name references",
        ));
    }
    Ok(())
}

/// One deck of an authored setup, named by preset or evidence key, as a fixed
/// setup over the catalog (not submitted to the world).
#[cfg(feature = "python")]
#[pyfunction]
fn authored_deck_setup(setup: &str, deck_key: &str) -> PyResult<PyPlayerConfig> {
    Ok(authored_setup(setup)?
        .player_config(deck_key, deck_key)
        .map_err(|error| PyValueError::new_err(error.to_string()))?
        .into())
}

#[cfg(feature = "python")]
#[pyfunction]
fn authored_deck_keys(setup: &str) -> PyResult<Vec<String>> {
    Ok(authored_setup(setup)?
        .decks
        .iter()
        .map(|deck| deck.key.clone())
        .collect())
}

#[pymodule]
pub fn _managym(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "MAX_DECISION_CODE",
        crate::agent::continuation::MAX_DECISION_CODE,
    )?;
    m.add(
        "CARD_DEFINITION_COUNT",
        crate::cards::content_pack::default_content_pack().len(),
    )?;
    m.add("NAME_BRANCHING", crate::cards::names::NAME_BRANCHING)?;
    m.add("AgentError", py.get_type_bound::<PyAgentError>())?;
    m.add(
        "ObservationCapacity",
        py.get_type_bound::<ObservationCapacity>(),
    )?;

    m.add_class::<ZoneEnum>()?;
    m.add_class::<PhaseEnum>()?;
    m.add_class::<StepEnum>()?;
    m.add_class::<ActionEnum>()?;
    m.add_class::<ActionSpaceEnum>()?;
    m.add_class::<StackObjectKindEnum>()?;
    m.add_class::<StackTargetKindEnum>()?;
    m.add_class::<EventTypeEnum>()?;
    m.add_class::<EventEntityKindEnum>()?;

    m.add_class::<PyPlayerConfig>()?;
    m.add_function(wrap_pyfunction!(resolve_decklists, m)?)?;
    m.add_function(wrap_pyfunction!(authored_deck_setup, m)?)?;
    super::deck_bindings::register(py, m)?;
    m.add_function(wrap_pyfunction!(authored_deck_keys, m)?)?;
    m.add_function(wrap_pyfunction!(content_pool, m)?)?;
    m.add_function(wrap_pyfunction!(world_identity, m)?)?;
    m.add_function(wrap_pyfunction!(set_cards, m)?)?;
    m.add_function(wrap_pyfunction!(catalog_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(naming_catalog, m)?)?;
    m.add_class::<crate::cards::names::NameCandidate>()?;
    m.add_function(wrap_pyfunction!(card_definitions_json, m)?)?;
    m.add_function(wrap_pyfunction!(reduce_card_capture_json, m)?)?;
    m.add_function(wrap_pyfunction!(catalog_token_value, m)?)?;
    m.add_function(wrap_pyfunction!(catalog_token_renames, m)?)?;
    m.add_function(wrap_pyfunction!(record_vocabulary, m)?)?;
    crate::python::layout_bindings::register(m)?;
    m.add_function(wrap_pyfunction!(unsupported_cards, m)?)?;
    m.add_function(wrap_pyfunction!(validate_observation_json, m)?)?;
    m.add_function(wrap_pyfunction!(policy_action_count, m)?)?;
    m.add_function(wrap_pyfunction!(encode_observation_into, m)?)?;
    m.add_function(wrap_pyfunction!(input_v10_state, m)?)?;
    m.add_function(wrap_pyfunction!(unrepresentable_state, m)?)?;
    m.add_function(wrap_pyfunction!(parse_continuation_frames_json, m)?)?;
    m.add_class::<PyObservation>()?;
    m.add_class::<crate::agent::continuation::VisibleReference>()?;
    m.add_class::<crate::agent::continuation::ClauseTarget>()?;
    m.add_class::<crate::agent::continuation::ManaOutput>()?;
    m.add_class::<crate::agent::continuation::ContinuationFrame>()?;
    m.add_class::<PyPlayer>()?;
    m.add_class::<PyTurn>()?;
    m.add_class::<PyCard>()?;
    m.add_class::<PyObservationCharacteristics>()?;
    m.add_class::<PyCastPermission>()?;
    m.add_class::<PyRestrictedMana>()?;
    m.add_class::<PyPooledManaClass>()?;
    m.add_class::<PyPlayerTurnHistory>()?;
    m.add_class::<PySideboardCard>()?;
    m.add_class::<PyCardTypes>()?;
    m.add_class::<PyKeywords>()?;
    m.add_class::<PyManaCost>()?;
    m.add_class::<PyPermanent>()?;
    m.add_class::<PyContinuousEffect>()?;
    m.add_class::<PySharedEffect>()?;
    m.add_class::<crate::agent::shared_effects::EffectObject>()?;
    m.add_class::<crate::agent::shared_effects::EffectMembership>()?;
    m.add_class::<PyCopyEffect>()?;
    m.add_class::<PyAction>()?;
    m.add_class::<PyActionSpace>()?;
    m.add_class::<PyStackTarget>()?;
    m.add_class::<PyStackObject>()?;
    m.add_class::<PyEventData>()?;

    m.add_class::<PyStructuredOfferSet>()?;
    m.add_class::<PySelectedBranchRuntime>()?;
    m.add_class::<PyPreparedPossibleWorldMaterializer>()?;
    m.add_class::<PyDecisionCapture>()?;
    m.add_class::<PyEnv>()?;
    m.add_class::<crate::python::heuristic_bindings::PyHeuristic>()?;
    m.add_class::<HandSnapshot>()?;
    m.add_class::<ObservationRead>()?;
    crate::python::vector_env_bindings::register_vector_env_bindings(m)?;
    Ok(())
}

#[cfg(feature = "python")]
pub(crate) fn disclosure_name(value: DecklistDisclosure) -> &'static str {
    match value {
        DecklistDisclosure::Open => "open",
        DecklistDisclosure::Closed => "closed",
    }
}

#[cfg(feature = "python")]
fn warn_disclosure(py: Python<'_>) -> PyResult<()> {
    pyo3::PyErr::warn_bound(
        py,
        &py.get_type_bound::<pyo3::exceptions::PyDeprecationWarning>(),
        "PlayerConfig.disclosure is deprecated; use GameSetup(decklists=...). Removed in 0.4.0.",
        2,
    )
}

/// Read the game value directly; the deprecated seat hints are checked only at
/// this boundary, never installed on a native player or roster.
#[cfg(feature = "python")]
pub(crate) fn game_setup(value: &Bound<'_, PyAny>) -> PyResult<crate::state::player::GameSetup> {
    use crate::state::player::{DecklistMode, GameSetup};
    let players: Vec<PyPlayerConfig> = value.extract()?;
    let mut hints = players
        .iter()
        .filter_map(|player| player.legacy_disclosure)
        .map(DecklistMode::from);
    let setup_type = value.py().import_bound("managym")?.getattr("GameSetup")?;
    let (decklists, mana_floats, mana_payment, conflict) = if value.is_instance(&setup_type)? {
        let name: String = value.getattr("decklists")?.extract()?;
        let decklists = match name.as_str() {
            "mixed" => DecklistMode::Mixed,
            fixed => parse_disclosure(fixed)?.into(),
        };
        (
            decklists,
            seat_modes(value, "mana_floats", ManaFloats::from_name)?,
            seat_modes(value, "mana_payment", ManaPayment::from_name)?,
            "deprecated per-seat disclosure conflicts with GameSetup.decklists",
        )
    } else {
        // A bare seat list takes its game value from the first hint, if any.
        (
            hints.clone().next().unwrap_or_default(),
            [ManaFloats::Off; 2],
            [ManaPayment::Manual; 2],
            "deprecated per-seat disclosure values disagree",
        )
    };
    if hints.any(|hint| hint != decklists) {
        return Err(PyValueError::new_err(conflict));
    }
    Ok(GameSetup {
        players: players.into_iter().map(PlayerConfig::from).collect(),
        decklists,
        mana_floats,
        mana_payment,
    })
}

/// Read a `GameSetup` per-seat mode pair by its native names.
#[cfg(feature = "python")]
fn seat_modes<T>(
    value: &Bound<'_, PyAny>,
    field: &str,
    parse: fn(&str) -> Option<T>,
) -> PyResult<[T; 2]> {
    let seats: [String; 2] = value.getattr(field)?.extract()?;
    let [first, second] = seats.map(|name| {
        parse(&name).ok_or_else(|| PyValueError::new_err(format!("unknown {field} mode {name:?}")))
    });
    Ok([first?, second?])
}

#[cfg(feature = "python")]
pub(crate) fn python_game_setup(
    py: Python<'_>,
    players: Vec<PlayerConfig>,
    disclosure: DecklistDisclosure,
    mana_floats: [ManaFloats; 2],
    mana_payment: [ManaPayment; 2],
) -> PyResult<PyObject> {
    let players: Vec<PyPlayerConfig> = players.into_iter().map(PyPlayerConfig::from).collect();
    let kwargs = pyo3::types::PyDict::new_bound(py);
    kwargs.set_item("decklists", disclosure_name(disclosure))?;
    kwargs.set_item("mana_floats", mana_floats.map(ManaFloats::name))?;
    kwargs.set_item("mana_payment", mana_payment.map(ManaPayment::name))?;
    Ok(py
        .import_bound("managym")?
        .getattr("GameSetup")?
        .call((players,), Some(&kwargs))?
        .unbind())
}

#[cfg(feature = "python")]
pub(crate) fn statistics_object(
    py: Python<'_>,
    value: &impl serde::Serialize,
    parser: &str,
) -> PyResult<PyObject> {
    let raw = serde_json::to_string(value).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(py
        .import_bound("managym.statistics")?
        .getattr(parser)?
        .call1((raw,))?
        .unbind())
}

#[cfg(feature = "python")]
#[pyfunction]
fn resolve_decklists(player_configs: Bound<'_, PyAny>, seed: u64) -> PyResult<&'static str> {
    Ok(disclosure_name(
        game_setup(&player_configs)?.decklists.resolve(seed),
    ))
}

#[cfg(feature = "python")]
#[pyfunction]
fn reduce_card_capture_json(raw: &str) -> PyResult<String> {
    let rows = crate::flow::statistics::reduce_capture(raw)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    serde_json::to_string(&rows).map_err(|e| PyValueError::new_err(e.to_string()))
}
