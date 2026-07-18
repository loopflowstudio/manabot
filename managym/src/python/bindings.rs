// PyO3's #[pymethods] macro triggers false-positive `useless_conversion`
// warnings in generated wrappers under strict clippy settings.
#![allow(clippy::useless_conversion)]
#![allow(unexpected_cfgs)]

#[cfg(feature = "python")]
use std::{collections::HashMap, sync::Mutex};

#[cfg(feature = "python")]
use pyo3::{
    exceptions::PyRuntimeError,
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
        env::Env,
        observation::{
            ActionOption, ActionSpaceData, CardData, CardTypeData, EventData, EventEntityKind,
            EventType, KeywordData, Observation, PermanentData, PlayerData, StackObjectData,
            StackObjectKindData, StackTargetData, StackTargetKindData, TurnData,
        },
        observation_encoder::{
            ObservationEncoderConfig, ACTION_DIM, CARD_DIM, EVENT_DIM, PERMANENT_DIM, PLAYER_DIM,
        },
        structured_offer::{OfferId as StructuredOfferId, OfferSubmission, StructuredOfferSet},
    },
    decision::Command as SemanticCommand,
    experience::{
        Command as ExperienceCommand, CommandId, MatchId, OfferId as ExperienceOfferId, PromptId,
        Revision,
    },
    flow::turn::{PhaseKind, StepKind},
    python::convert::{info_dict_to_pydict, require_numpy_array, shape_to_vec},
    search_state::SearchStateWitness,
    state::{mana::ManaCost, player::PlayerConfig, zone::ZoneType},
};

#[cfg(feature = "python")]
pyo3::create_exception!(_managym, PyAgentError, PyRuntimeError);

#[cfg(feature = "python")]
pub(crate) fn map_agent_err(err: AgentError) -> PyErr {
    PyAgentError::new_err(err.to_string())
}

#[cfg(feature = "python")]
fn to_numpy_array_f32(
    py: Python<'_>,
    np: &Bound<'_, PyModule>,
    data: &[f32],
    shape: &[usize],
) -> PyResult<PyObject> {
    let array_fn = np.getattr("array")?;
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("dtype", np.getattr("float32")?)?;
    let list = PyList::new_bound(py, data.iter().copied());
    let array = array_fn.call((list,), Some(&kwargs))?;
    let reshaped = array.call_method1("reshape", (shape.to_vec(),))?;
    Ok(reshaped.unbind())
}

#[cfg(feature = "python")]
fn to_numpy_array_i32(
    py: Python<'_>,
    np: &Bound<'_, PyModule>,
    data: &[i32],
    shape: &[usize],
) -> PyResult<PyObject> {
    let array_fn = np.getattr("array")?;
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("dtype", np.getattr("int32")?)?;
    let list = PyList::new_bound(py, data.iter().copied());
    let array = array_fn.call((list,), Some(&kwargs))?;
    let reshaped = array.call_method1("reshape", (shape.to_vec(),))?;
    Ok(reshaped.unbind())
}

#[cfg(feature = "python")]
fn encoded_to_dict<'py>(
    py: Python<'py>,
    encoded: crate::agent::observation_encoder::EncodedObservation,
    config: &ObservationEncoderConfig,
) -> PyResult<Bound<'py, PyDict>> {
    let np = PyModule::import_bound(py, "numpy")?;
    let dict = PyDict::new_bound(py);

    dict.set_item(
        "agent_player",
        to_numpy_array_f32(py, &np, &encoded.agent_player, &[1, PLAYER_DIM])?,
    )?;
    dict.set_item(
        "opponent_player",
        to_numpy_array_f32(py, &np, &encoded.opponent_player, &[1, PLAYER_DIM])?,
    )?;
    dict.set_item(
        "agent_cards",
        to_numpy_array_f32(
            py,
            &np,
            &encoded.agent_cards,
            &[config.max_cards_per_player, CARD_DIM],
        )?,
    )?;
    dict.set_item(
        "opponent_cards",
        to_numpy_array_f32(
            py,
            &np,
            &encoded.opponent_cards,
            &[config.max_cards_per_player, CARD_DIM],
        )?,
    )?;
    dict.set_item(
        "agent_permanents",
        to_numpy_array_f32(
            py,
            &np,
            &encoded.agent_permanents,
            &[config.max_permanents_per_player, PERMANENT_DIM],
        )?,
    )?;
    dict.set_item(
        "opponent_permanents",
        to_numpy_array_f32(
            py,
            &np,
            &encoded.opponent_permanents,
            &[config.max_permanents_per_player, PERMANENT_DIM],
        )?,
    )?;
    dict.set_item(
        "actions",
        to_numpy_array_f32(py, &np, &encoded.actions, &[config.max_actions, ACTION_DIM])?,
    )?;
    dict.set_item(
        "events",
        to_numpy_array_f32(py, &np, &encoded.events, &[config.max_events, EVENT_DIM])?,
    )?;
    dict.set_item(
        "action_focus",
        to_numpy_array_i32(
            py,
            &np,
            &encoded.action_focus,
            &[config.max_actions, config.max_focus_objects],
        )?,
    )?;

    dict.set_item(
        "agent_player_valid",
        to_numpy_array_f32(py, &np, &encoded.agent_player_valid, &[1])?,
    )?;
    dict.set_item(
        "opponent_player_valid",
        to_numpy_array_f32(py, &np, &encoded.opponent_player_valid, &[1])?,
    )?;
    dict.set_item(
        "agent_cards_valid",
        to_numpy_array_f32(
            py,
            &np,
            &encoded.agent_cards_valid,
            &[config.max_cards_per_player],
        )?,
    )?;
    dict.set_item(
        "opponent_cards_valid",
        to_numpy_array_f32(
            py,
            &np,
            &encoded.opponent_cards_valid,
            &[config.max_cards_per_player],
        )?,
    )?;
    dict.set_item(
        "agent_permanents_valid",
        to_numpy_array_f32(
            py,
            &np,
            &encoded.agent_permanents_valid,
            &[config.max_permanents_per_player],
        )?,
    )?;
    dict.set_item(
        "opponent_permanents_valid",
        to_numpy_array_f32(
            py,
            &np,
            &encoded.opponent_permanents_valid,
            &[config.max_permanents_per_player],
        )?,
    )?;
    dict.set_item(
        "actions_valid",
        to_numpy_array_f32(py, &np, &encoded.actions_valid, &[config.max_actions])?,
    )?;
    dict.set_item(
        "events_valid",
        to_numpy_array_f32(py, &np, &encoded.events_valid, &[config.max_events])?,
    )?;

    Ok(dict)
}

#[cfg(feature = "python")]
fn fill_encoded_into_existing_buffers(
    py: Python<'_>,
    out: &Bound<'_, PyDict>,
    encoded: crate::agent::observation_encoder::EncodedObservation,
    config: &ObservationEncoderConfig,
) -> PyResult<()> {
    let np = PyModule::import_bound(py, "numpy")?;
    let copyto = np.getattr("copyto")?;
    let expected = encoded_to_dict(py, encoded, config)?;
    for (key_obj, source) in expected.iter() {
        let key = key_obj.extract::<String>()?;
        let dtype_name = source
            .getattr("dtype")?
            .getattr("name")?
            .extract::<String>()?;
        let shape = shape_to_vec(&source.getattr("shape")?)?;
        let target = require_numpy_array(out, &key, &shape, &dtype_name)?;
        copyto.call1((target, source))?;
    }

    Ok(())
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
    DiscardThenDraw = 9,
    Waterbend = 10,
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
    const DISCARD_THEN_DRAW: Self = Self::DiscardThenDraw;
    #[classattr]
    const WATERBEND: Self = Self::Waterbend;

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
            ActionSpaceKind::DiscardThenDraw => Self::DiscardThenDraw,
            ActionSpaceKind::Waterbend => Self::Waterbend,
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
            ActionSpaceEnum::DiscardThenDraw => Self::DiscardThenDraw,
            ActionSpaceEnum::Waterbend => Self::Waterbend,
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
        }
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "PlayerConfig")]
#[derive(Clone)]
pub struct PyPlayerConfig {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub decklist: HashMap<String, usize>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyPlayerConfig {
    #[new]
    fn new(name: String, decklist: HashMap<String, usize>) -> Self {
        Self { name, decklist }
    }
}

#[cfg(feature = "python")]
impl From<PyPlayerConfig> for PlayerConfig {
    fn from(value: PyPlayerConfig) -> Self {
        PlayerConfig {
            name: value.name,
            decklist: value.decklist.into_iter().collect(),
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
    #[pyo3(get, set)]
    pub combat_mana: i32,
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
            combat_mana: value.combat_mana,
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
            combat_mana: value.combat_mana,
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
        }
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "ManaCost")]
#[derive(Clone)]
pub struct PyManaCost {
    #[pyo3(get, set)]
    pub cost: Vec<i32>,
    #[pyo3(get, set)]
    pub mana_value: i32,
}

#[cfg(feature = "python")]
impl From<ManaCost> for PyManaCost {
    fn from(value: ManaCost) -> Self {
        Self {
            cost: value.cost[..6].iter().map(|v| i32::from(*v)).collect(),
            mana_value: i32::from(value.mana_value),
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

        Self { cost, mana_value }
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "Card")]
#[derive(Clone)]
pub struct PyCard {
    #[pyo3(get, set)]
    pub zone: ZoneEnum,
    #[pyo3(get, set)]
    pub owner_id: i32,
    #[pyo3(get, set)]
    pub id: i32,
    #[pyo3(get, set)]
    pub registry_key: i32,
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub power: i32,
    #[pyo3(get, set)]
    pub toughness: i32,
    #[pyo3(get, set)]
    pub is_token: bool,
    #[pyo3(get, set)]
    pub is_ally: bool,
    #[pyo3(get, set)]
    pub is_lesson: bool,
    #[pyo3(get, set)]
    pub ward_cost: i32,
    #[pyo3(get, set)]
    pub kicker_cost: i32,
    #[pyo3(get, set)]
    pub card_types: PyCardTypes,
    #[pyo3(get, set)]
    pub keywords: PyKeywords,
    #[pyo3(get, set)]
    pub mana_cost: PyManaCost,
}

#[cfg(feature = "python")]
impl From<CardData> for PyCard {
    fn from(value: CardData) -> Self {
        Self {
            zone: value.zone.into(),
            owner_id: value.owner_id,
            id: value.id,
            registry_key: value.registry_key,
            name: value.name,
            power: value.power,
            toughness: value.toughness,
            is_token: value.is_token,
            is_ally: value.is_ally,
            is_lesson: value.is_lesson,
            ward_cost: value.ward_cost,
            kicker_cost: value.kicker_cost,
            card_types: value.card_types.into(),
            keywords: value.keywords.into(),
            mana_cost: value.mana_cost.into(),
        }
    }
}

#[cfg(feature = "python")]
impl From<PyCard> for CardData {
    fn from(value: PyCard) -> Self {
        Self {
            zone: value.zone.into(),
            owner_id: value.owner_id,
            id: value.id,
            registry_key: value.registry_key,
            name: value.name,
            power: value.power,
            toughness: value.toughness,
            is_token: value.is_token,
            is_ally: value.is_ally,
            is_lesson: value.is_lesson,
            ward_cost: value.ward_cost,
            kicker_cost: value.kicker_cost,
            card_types: value.card_types.into(),
            keywords: value.keywords.into(),
            mana_cost: value.mana_cost.into(),
        }
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
        }
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "Permanent")]
#[derive(Clone)]
pub struct PyPermanent {
    #[pyo3(get, set)]
    pub id: i32,
    #[pyo3(get, set)]
    pub controller_id: i32,
    #[pyo3(get, set)]
    pub tapped: bool,
    #[pyo3(get, set)]
    pub damage: i32,
    #[pyo3(get, set)]
    pub is_summoning_sick: bool,
    #[pyo3(get, set)]
    pub plus1_counters: i32,
    #[pyo3(get, set)]
    pub cant_be_blocked_this_turn: bool,
    #[pyo3(get, set)]
    pub power: i32,
    #[pyo3(get, set)]
    pub toughness: i32,
    #[pyo3(get, set)]
    pub is_animated: bool,
    #[pyo3(get, set)]
    pub has_exile_link: bool,
    /// Effective keywords (printed + until-EOT grants).
    #[pyo3(get, set)]
    pub keywords: PyKeywords,
}

#[cfg(feature = "python")]
impl From<PermanentData> for PyPermanent {
    fn from(value: PermanentData) -> Self {
        Self {
            id: value.id,
            controller_id: value.controller_id,
            tapped: value.tapped,
            damage: value.damage,
            is_summoning_sick: value.is_summoning_sick,
            plus1_counters: value.plus1_counters,
            cant_be_blocked_this_turn: value.cant_be_blocked_this_turn,
            power: value.power,
            toughness: value.toughness,
            is_animated: value.is_animated,
            has_exile_link: value.has_exile_link,
            keywords: value.keywords.into(),
        }
    }
}

#[cfg(feature = "python")]
impl From<PyPermanent> for PermanentData {
    fn from(value: PyPermanent) -> Self {
        Self {
            id: value.id,
            controller_id: value.controller_id,
            tapped: value.tapped,
            damage: value.damage,
            is_summoning_sick: value.is_summoning_sick,
            plus1_counters: value.plus1_counters,
            cant_be_blocked_this_turn: value.cant_be_blocked_this_turn,
            power: value.power,
            toughness: value.toughness,
            is_animated: value.is_animated,
            has_exile_link: value.has_exile_link,
            keywords: value.keywords.into(),
        }
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "Action")]
#[derive(Clone)]
pub struct PyAction {
    #[pyo3(get, set)]
    pub action_type: ActionEnum,
    #[pyo3(get, set)]
    pub focus: Vec<i32>,
    #[pyo3(get, set)]
    pub declared: Option<bool>,
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
                _ => EventTypeEnum::TurnStarted,
            },
            source_kind: match value.source_kind {
                x if x == EventEntityKind::Card as i32 => EventEntityKindEnum::Card,
                x if x == EventEntityKind::Permanent as i32 => EventEntityKindEnum::Permanent,
                x if x == EventEntityKind::Player as i32 => EventEntityKindEnum::Player,
                x if x == EventEntityKind::Object as i32 => EventEntityKindEnum::Object,
                _ => EventEntityKindEnum::None,
            },
            source_id: value.source_id,
            target_kind: match value.target_kind {
                x if x == EventEntityKind::Card as i32 => EventEntityKindEnum::Card,
                x if x == EventEntityKind::Permanent as i32 => EventEntityKindEnum::Permanent,
                x if x == EventEntityKind::Player as i32 => EventEntityKindEnum::Player,
                x if x == EventEntityKind::Object as i32 => EventEntityKindEnum::Object,
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
}

#[cfg(feature = "python")]
impl From<StackTargetData> for PyStackTarget {
    fn from(value: StackTargetData) -> Self {
        Self {
            kind: value.kind.into(),
            player_id: value.player_id,
            permanent_id: value.permanent_id,
            stack_object_id: value.stack_object_id,
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
        }
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "StackObject")]
#[derive(Clone)]
pub struct PyStackObject {
    #[pyo3(get, set)]
    pub stack_object_id: i32,
    #[pyo3(get, set)]
    pub kind: StackObjectKindEnum,
    #[pyo3(get, set)]
    pub controller_id: i32,
    #[pyo3(get, set)]
    pub source_card_registry_key: i32,
    #[pyo3(get, set)]
    pub source_permanent_id: Option<i32>,
    #[pyo3(get, set)]
    pub ability_index: Option<i32>,
    #[pyo3(get, set)]
    pub targets: Vec<PyStackTarget>,
}

#[cfg(feature = "python")]
impl From<StackObjectData> for PyStackObject {
    fn from(value: StackObjectData) -> Self {
        Self {
            stack_object_id: value.stack_object_id,
            kind: value.kind.into(),
            controller_id: value.controller_id,
            source_card_registry_key: value.source_card_registry_key,
            source_permanent_id: value.source_permanent_id,
            ability_index: value.ability_index,
            targets: value.targets.into_iter().map(PyStackTarget::from).collect(),
        }
    }
}

#[cfg(feature = "python")]
impl From<PyStackObject> for StackObjectData {
    fn from(value: PyStackObject) -> Self {
        Self {
            stack_object_id: value.stack_object_id,
            kind: value.kind.into(),
            controller_id: value.controller_id,
            source_card_registry_key: value.source_card_registry_key,
            source_permanent_id: value.source_permanent_id,
            ability_index: value.ability_index,
            targets: value
                .targets
                .into_iter()
                .map(StackTargetData::from)
                .collect(),
        }
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "Observation")]
#[derive(Clone)]
pub struct PyObservation {
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
}

#[cfg(feature = "python")]
impl From<Observation> for PyObservation {
    fn from(value: Observation) -> Self {
        Self {
            game_over: value.game_over,
            won: value.won,
            turn: value.turn.into(),
            action_space: value.action_space.into(),
            agent: value.agent.into(),
            agent_cards: value.agent_cards.into_iter().map(PyCard::from).collect(),
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
            game_over: value.game_over,
            won: value.won,
            turn: value.turn.into(),
            action_space: value.action_space.into(),
            agent: value.agent.into(),
            agent_cards: value.agent_cards.into_iter().map(CardData::from).collect(),
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
    fn validate(&self) -> bool {
        if self.agent.id == self.opponent.id {
            return false;
        }
        if self.agent.is_agent == self.opponent.is_agent {
            return false;
        }

        for card in &self.agent_cards {
            if card.owner_id != self.agent.id {
                return false;
            }
        }
        for card in &self.opponent_cards {
            if card.owner_id != self.opponent.id {
                return false;
            }
        }
        for permanent in &self.agent_permanents {
            if permanent.controller_id != self.agent.id {
                return false;
            }
        }
        for permanent in &self.opponent_permanents {
            if permanent.controller_id != self.opponent.id {
                return false;
            }
        }

        true
    }

    #[allow(non_snake_case)]
    fn toJSON(&self) -> String {
        fn player_json(player: &PyPlayer) -> Value {
            json!({
                "player_index": player.player_index,
                "id": player.id,
                "is_active": player.is_active,
                "is_agent": player.is_agent,
                "life": player.life,
                "zone_counts": player.zone_counts,
            })
        }

        fn card_json(card: &PyCard) -> Value {
            json!({
                "id": card.id,
                "registry_key": card.registry_key,
                "name": card.name,
                "zone": card.zone as i32,
                "owner_id": card.owner_id,
                "power": card.power,
                "toughness": card.toughness,
                "card_types": {
                    "is_castable": card.card_types.is_castable,
                    "is_permanent": card.card_types.is_permanent,
                    "is_non_land_permanent": card.card_types.is_non_land_permanent,
                    "is_non_creature_permanent": card.card_types.is_non_creature_permanent,
                    "is_spell": card.card_types.is_spell,
                    "is_creature": card.card_types.is_creature,
                    "is_land": card.card_types.is_land,
                    "is_planeswalker": card.card_types.is_planeswalker,
                    "is_enchantment": card.card_types.is_enchantment,
                    "is_artifact": card.card_types.is_artifact,
                    "is_kindred": card.card_types.is_kindred,
                    "is_battle": card.card_types.is_battle,
                },
                "keywords": keywords_json(&card.keywords),
                "mana_cost": {
                    "cost": card.mana_cost.cost,
                    "mana_value": card.mana_cost.mana_value,
                }
            })
        }

        fn keywords_json(keywords: &PyKeywords) -> Value {
            json!({
                "flying": keywords.flying,
                "reach": keywords.reach,
                "haste": keywords.haste,
                "flash": keywords.flash,
                "vigilance": keywords.vigilance,
                "trample": keywords.trample,
                "first_strike": keywords.first_strike,
                "double_strike": keywords.double_strike,
                "deathtouch": keywords.deathtouch,
                "lifelink": keywords.lifelink,
                "defender": keywords.defender,
                "menace": keywords.menace,
                "hexproof": keywords.hexproof,
            })
        }

        fn permanent_json(permanent: &PyPermanent) -> Value {
            json!({
                "id": permanent.id,
                "controller_id": permanent.controller_id,
                "tapped": permanent.tapped,
                "damage": permanent.damage,
                "is_summoning_sick": permanent.is_summoning_sick,
                "keywords": keywords_json(&permanent.keywords),
            })
        }

        fn stack_target_json(target: &PyStackTarget) -> Value {
            json!({
                "kind": target.kind as i32,
                "player_id": target.player_id,
                "permanent_id": target.permanent_id,
                "stack_object_id": target.stack_object_id,
            })
        }

        fn stack_object_json(stack_object: &PyStackObject) -> Value {
            json!({
                "stack_object_id": stack_object.stack_object_id,
                "kind": stack_object.kind as i32,
                "controller_id": stack_object.controller_id,
                "source_card_registry_key": stack_object.source_card_registry_key,
                "source_permanent_id": stack_object.source_permanent_id,
                "ability_index": stack_object.ability_index,
                "targets": stack_object.targets.iter().map(stack_target_json).collect::<Vec<_>>(),
            })
        }

        json!({
            "game_over": self.game_over,
            "won": self.won,
            "turn": {
                "turn_number": self.turn.turn_number,
                "phase": self.turn.phase as i32,
                "step": self.turn.step as i32,
                "active_player_id": self.turn.active_player_id,
                "agent_player_id": self.turn.agent_player_id,
            },
            "action_space": {
                "type": self.action_space.action_space_type as i32,
                "actions": self
                    .action_space
                    .actions
                    .iter()
                    .map(|action| {
                        json!({
                            "type": action.action_type as i32,
                            "focus": action.focus,
                        })
                    })
                    .collect::<Vec<_>>(),
            },
            "agent": player_json(&self.agent),
            "agent_cards": self.agent_cards.iter().map(card_json).collect::<Vec<_>>(),
            "agent_permanents": self
                .agent_permanents
                .iter()
                .map(permanent_json)
                .collect::<Vec<_>>(),
            "opponent": player_json(&self.opponent),
            "opponent_cards": self.opponent_cards.iter().map(card_json).collect::<Vec<_>>(),
            "opponent_permanents": self
                .opponent_permanents
                .iter()
                .map(permanent_json)
                .collect::<Vec<_>>(),
            "stack_objects": self
                .stack_objects
                .iter()
                .map(stack_object_json)
                .collect::<Vec<_>>(),
            "recent_events": self
                .recent_events
                .iter()
                .map(|event| {
                    json!({
                        "event_type": event.event_type as i32,
                        "source_kind": event.source_kind as i32,
                        "source_id": event.source_id,
                        "target_kind": event.target_kind as i32,
                        "target_id": event.target_id,
                        "amount": event.amount,
                        "controller_id": event.controller_id,
                        "from_zone": event.from_zone,
                        "to_zone": event.to_zone,
                        "source_incarnation": event.source_incarnation,
                        "target_incarnation": event.target_incarnation,
                    })
                })
                .collect::<Vec<_>>(),
        })
        .to_string()
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "Env")]
pub struct PyEnv {
    inner: Mutex<Env>,
    selected_guard: bool,
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
            inner: Mutex::new(fork),
            selected_guard: true,
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

        let require_witness = runtime.audit
            || overrides.authority_hash.is_some()
            || overrides.legal_surface_hash.is_some();
        let current_witness = require_witness
            .then(|| env.selected_witness())
            .transpose()
            .map_err(map_agent_err)?;
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
            current_witness
                .as_ref()
                .map(|witness| witness.authority.hash.as_str())
        });
        let expected_legal_hash = overrides.legal_surface_hash.as_deref().or_else(|| {
            current_witness
                .as_ref()
                .map(|witness| witness.legal_surface.hash.as_str())
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
            current_witness
                .as_ref()
                .is_none_or(|witness| witness.authority.hash != expected)
        }) {
            return Err(PyAgentError::new_err("authority precondition mismatch"));
        }
        if expected_legal_hash.is_some_and(|expected| {
            current_witness
                .as_ref()
                .is_none_or(|witness| witness.legal_surface.hash != expected)
        }) {
            return Err(PyAgentError::new_err("legal-surface precondition mismatch"));
        }

        let command_id = format!("search.{}.{}", runtime.match_id, runtime.next_command_id);
        let command = ExperienceCommand {
            command_id: CommandId(command_id),
            match_id: MatchId(runtime.match_id.clone()),
            expected_revision: Revision(expected_revision),
            prompt_id: PromptId(expected_prompt_id),
            offer_id: ExperienceOfferId(expected_offer_id),
            answers: Vec::new(),
        };
        if command.match_id.0 != runtime.match_id {
            return Err(PyAgentError::new_err("match ID precondition mismatch"));
        }
        if command.expected_revision.0 != current_revision {
            return Err(PyAgentError::new_err("revision precondition mismatch"));
        }
        if command.prompt_id.0 != current_revision {
            return Err(PyAgentError::new_err("prompt ID precondition mismatch"));
        }
        if command.offer_id.0 != current_offer.id.0 {
            return Err(PyAgentError::new_err("offer ID precondition mismatch"));
        }
        if !command.answers.is_empty() {
            return Err(PyAgentError::new_err(
                "search policy Command unexpectedly contains answers",
            ));
        }
        let submission = OfferSubmission {
            offer_id: StructuredOfferId(command.offer_id.0),
            answers: Vec::new(),
        };
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

        let py_info = info_dict_to_pydict(py, &info);
        Ok((
            PyObservation::from(observation),
            reward,
            terminated,
            truncated,
            py_info.into_any().unbind(),
            serde_json::to_string(&record)
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?,
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
    #[pyo3(signature = (seed=0, skip_trivial=true, enable_profiler=false, enable_behavior_tracking=false))]
    fn new(
        seed: u64,
        skip_trivial: bool,
        enable_profiler: bool,
        enable_behavior_tracking: bool,
    ) -> Self {
        Self {
            inner: Mutex::new(Env::new(
                seed,
                skip_trivial,
                enable_profiler,
                enable_behavior_tracking,
            )),
            selected_guard: false,
        }
    }

    fn reset(
        &self,
        py: Python<'_>,
        player_configs: Vec<PyPlayerConfig>,
    ) -> PyResult<(PyObservation, PyObject)> {
        let mut env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;

        let configs = player_configs.into_iter().map(PlayerConfig::from).collect();
        let (obs, info) = env.reset(configs).map_err(map_agent_err)?;

        let py_dict = info_dict_to_pydict(py, &info);
        Ok((PyObservation::from(obs), py_dict.into_any().unbind()))
    }

    fn set_seed(&self, seed: u64) -> PyResult<()> {
        let mut env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        env.set_seed(seed);
        Ok(())
    }

    fn step(
        &self,
        py: Python<'_>,
        action: i64,
    ) -> PyResult<(PyObservation, f64, bool, bool, PyObject)> {
        self.reject_guarded_mutation("Env.step(index)")?;
        let mut env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;

        let (obs, reward, terminated, truncated, info) = env.step(action).map_err(map_agent_err)?;
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

    /// Read-only binding manifest for the exact immutable ContentPack used by
    /// the current match. This is deliberately not repeated on observations.
    fn content_pack_manifest(&self, py: Python<'_>) -> PyResult<PyObject> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        let manifest = env.content_pack_manifest().map_err(map_agent_err)?;

        let out = PyDict::new_bound(py);
        out.set_item("schema_version", manifest.schema_version)?;
        out.set_item("content_digest", manifest.content_digest)?;
        if let Some(compiled) = manifest.compiled_semantics {
            let semantic = PyDict::new_bound(py);
            semantic.set_item("pack_key", compiled.pack_key)?;
            semantic.set_item("ir_hash", compiled.ir_hash)?;
            semantic.set_item("source_hash", compiled.source_hash)?;
            out.set_item("compiled_semantics", semantic)?;
        } else {
            out.set_item("compiled_semantics", py.None())?;
        }
        let definitions = PyList::empty_bound(py);
        for entry in manifest.definitions {
            let definition = PyDict::new_bound(py);
            definition.set_item("card_def_id", entry.card_def_id.0)?;
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
            inner: Mutex::new(fork),
            selected_guard: false,
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

    /// Composite viewer-safe Observation for `viewer`, as canonical JSON.
    fn semantic_observation_json(&self, viewer: usize) -> PyResult<String> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        let observation = env.semantic_observation(viewer).map_err(map_agent_err)?;
        serde_json::to_string(&observation)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
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

    /// Materialize one identity-bound canonical world into an isolated Env.
    #[pyo3(signature = (viewer, space_identity, world_index, seed, refresh_opponent_priority=false))]
    fn materialize_possible_world(
        &self,
        viewer: usize,
        space_identity: &str,
        world_index: usize,
        seed: u64,
        refresh_opponent_priority: bool,
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
                refresh_opponent_priority,
            )
            .map_err(map_agent_err)?;
        Ok(PyEnv {
            inner: Mutex::new(branch),
            selected_guard: false,
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
    /// observation needed by Etude presentation and trace consumers.
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
    // Inject at a priority decision, then call scenario_refresh() once.
    // ------------------------------------------------------------------

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

    fn encode_observation(&self, py: Python<'_>, obs: PyObservation) -> PyResult<PyObject> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        let rust_obs = Observation::from(obs);
        let config = ObservationEncoderConfig::default();
        let encoded = env.encode_observation(&rust_obs);
        drop(env);
        let out = encoded_to_dict(py, encoded, &config)?;
        Ok(out.into_any().unbind())
    }

    fn encode_observation_into(
        &self,
        py: Python<'_>,
        obs: PyObservation,
        out: Bound<'_, PyDict>,
    ) -> PyResult<()> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;

        let rust_obs = Observation::from(obs);
        let config = ObservationEncoderConfig::default();
        let encoded = env.encode_observation(&rust_obs);
        drop(env);

        fill_encoded_into_existing_buffers(py, &out, encoded, &config)
    }
}

#[cfg(feature = "python")]
#[pymodule]
pub fn _managym(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("AgentError", py.get_type_bound::<PyAgentError>())?;

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
    m.add_class::<PyObservation>()?;
    m.add_class::<PyPlayer>()?;
    m.add_class::<PyTurn>()?;
    m.add_class::<PyCard>()?;
    m.add_class::<PyCardTypes>()?;
    m.add_class::<PyKeywords>()?;
    m.add_class::<PyManaCost>()?;
    m.add_class::<PyPermanent>()?;
    m.add_class::<PyAction>()?;
    m.add_class::<PyActionSpace>()?;
    m.add_class::<PyStackTarget>()?;
    m.add_class::<PyStackObject>()?;
    m.add_class::<PyEventData>()?;

    m.add_class::<PyStructuredOfferSet>()?;
    m.add_class::<PySelectedBranchRuntime>()?;
    m.add_class::<PyEnv>()?;
    crate::python::vector_env_bindings::register_vector_env_bindings(m)?;
    Ok(())
}
