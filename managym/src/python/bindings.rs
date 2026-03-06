// PyO3's #[pymethods] macro triggers false-positive `useless_conversion`
// warnings in generated wrappers under strict clippy settings.
#![allow(clippy::useless_conversion)]
#![allow(unexpected_cfgs)]

#[cfg(feature = "python")]
use std::{collections::HashMap, sync::Mutex};

#[cfg(feature = "python")]
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyDict, PyList, PyModule},
};
#[cfg(feature = "python")]
use serde_json::{json, Value};

#[cfg(feature = "python")]
use crate::{
    agent::{
        action::{ActionSpaceKind, ActionType, AgentError},
        env::Env,
        observation::{
            ActionOption, ActionSpaceData, CardData, CardTypeData, Observation, PermanentData,
            PlayerData, TurnData,
        },
        observation_encoder::{
            ObservationEncoderConfig, ACTION_DIM, CARD_DIM, PERMANENT_DIM, PLAYER_DIM,
        },
    },
    flow::turn::{PhaseKind, StepKind},
    python::convert::{info_dict_to_pydict, require_numpy_array},
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
    pub power: i32,
    #[pyo3(get, set)]
    pub toughness: i32,
    #[pyo3(get, set)]
    pub card_types: PyCardTypes,
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
            power: value.power,
            toughness: value.toughness,
            card_types: value.card_types.into(),
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
            power: value.power,
            toughness: value.toughness,
            card_types: value.card_types.into(),
            mana_cost: value.mana_cost.into(),
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
}

#[cfg(feature = "python")]
impl From<ActionOption> for PyAction {
    fn from(value: ActionOption) -> Self {
        Self {
            action_type: value.action_type.into(),
            focus: value.focus,
        }
    }
}

#[cfg(feature = "python")]
impl From<PyAction> for ActionOption {
    fn from(value: PyAction) -> Self {
        Self {
            action_type: value.action_type.into(),
            focus: value.focus,
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
                "mana_cost": {
                    "cost": card.mana_cost.cost,
                    "mana_value": card.mana_cost.mana_value,
                }
            })
        }

        fn permanent_json(permanent: &PyPermanent) -> Value {
            json!({
                "id": permanent.id,
                "controller_id": permanent.controller_id,
                "tapped": permanent.tapped,
                "damage": permanent.damage,
                "is_summoning_sick": permanent.is_summoning_sick,
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
        })
        .to_string()
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "Env")]
pub struct PyEnv {
    inner: Mutex<Env>,
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

    fn step(
        &self,
        py: Python<'_>,
        action: i64,
    ) -> PyResult<(PyObservation, f64, bool, bool, PyObject)> {
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

    fn info(&self, py: Python<'_>) -> PyResult<PyObject> {
        let env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        let py_dict = info_dict_to_pydict(py, &env.info());
        Ok(py_dict.into_any().unbind())
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

    m.add_class::<PyPlayerConfig>()?;
    m.add_class::<PyObservation>()?;
    m.add_class::<PyPlayer>()?;
    m.add_class::<PyTurn>()?;
    m.add_class::<PyCard>()?;
    m.add_class::<PyCardTypes>()?;
    m.add_class::<PyManaCost>()?;
    m.add_class::<PyPermanent>()?;
    m.add_class::<PyAction>()?;
    m.add_class::<PyActionSpace>()?;

    m.add_class::<PyEnv>()?;
    crate::python::vector_env_bindings::register_vector_env_bindings(m)?;
    Ok(())
}
