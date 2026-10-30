use std::collections::BTreeMap;

use crate::state::game_object::PlayerId;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[repr(i32)]
pub enum PhaseKind {
    // CR 500.1 — Turn phases in order.
    Beginning = 0,
    PrecombatMain = 1,
    Combat = 2,
    PostcombatMain = 3,
    Ending = 4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[repr(i32)]
pub enum StepKind {
    // CR 501-514 — Step order for a standard two-player turn.
    Untap = 0,
    Upkeep = 1,
    Draw = 2,
    Main = 3,
    BeginningOfCombat = 4,
    DeclareAttackers = 5,
    DeclareBlockers = 6,
    CombatDamage = 7,
    EndOfCombat = 8,
    PostcombatMain = 9,
    End = 10,
    Cleanup = 11,
}

pub const PHASE_STEPS: [&[StepKind]; 5] = [
    // CR 500.1 — Structured phase/step progression.
    &[StepKind::Untap, StepKind::Upkeep, StepKind::Draw],
    &[StepKind::Main],
    &[
        StepKind::BeginningOfCombat,
        StepKind::DeclareAttackers,
        StepKind::DeclareBlockers,
        StepKind::CombatDamage,
        StepKind::EndOfCombat,
    ],
    &[StepKind::PostcombatMain],
    &[StepKind::End, StepKind::Cleanup],
];

/// What one player has done this turn, for conditions that read turn history
/// ("if you attacked this turn", "your second card each turn"). The owner of
/// every such fact: the turn reset, the state hash and the policy's player
/// row all read this struct.
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    schemars::JsonSchema,
)]
#[serde(deny_unknown_fields)]
pub struct PlayerTurnHistory {
    /// Declared one or more attackers (raid; CR 508.1). Creatures put onto
    /// the battlefield attacking were never declared and do not count.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub attacked: bool,
    /// Permanents this player sacrificed (CR 701.21).
    #[serde(default, skip_serializing_if = "crate::state::is_zero")]
    pub permanents_sacrificed: u32,
    /// A land entered the battlefield under this player's control.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub land_entered: bool,
    /// Cards this player drew ("your Nth card each turn").
    #[serde(default, skip_serializing_if = "crate::state::is_zero")]
    pub cards_drawn: u32,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct TurnState {
    pub active_player: PlayerId,
    pub turn_number: u32,
    pub lands_played: u32,
    pub spells_cast: [Vec<crate::state::game_object::CardId>; 2],
    /// Each player's history this turn, by player index. Resets every turn.
    pub history: [PlayerTurnHistory; 2],
    /// Once-per-turn limits are recorded at collection, by exact source object.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub ability_triggers_this_turn: BTreeMap<(crate::state::game_object::ObjectRef, usize), u32>,
    /// Times each triggered ability `(card index, ability index)` has
    /// resolved this turn (for "the Nth time this ability has resolved this
    /// turn" gating). Resets every turn.
    pub ability_resolutions_this_turn: BTreeMap<(usize, usize), u32>,
    pub current_phase: usize,
    pub current_step: usize,
    pub step_initialized: bool,
    pub turn_based_actions_complete: bool,
    pub cleanup_priority: bool,
}

impl TurnState {
    pub fn new(active_player: PlayerId) -> Self {
        Self {
            active_player,
            turn_number: 1,
            lands_played: 0,
            spells_cast: [Vec::new(), Vec::new()],
            history: Default::default(),
            ability_resolutions_this_turn: BTreeMap::new(),
            ability_triggers_this_turn: BTreeMap::new(),
            current_phase: 0,
            current_step: 0,
            step_initialized: false,
            turn_based_actions_complete: false,
            cleanup_priority: false,
        }
    }

    pub fn current_phase_kind(&self) -> PhaseKind {
        match self.current_phase {
            0 => PhaseKind::Beginning,
            1 => PhaseKind::PrecombatMain,
            2 => PhaseKind::Combat,
            3 => PhaseKind::PostcombatMain,
            4 => PhaseKind::Ending,
            _ => unreachable!("invalid phase index"),
        }
    }

    pub fn current_step_kind(&self) -> StepKind {
        PHASE_STEPS[self.current_phase][self.current_step]
    }

    pub fn can_cast_sorceries(&self) -> bool {
        // CR 117.1a, 307.1 — Sorcery timing is restricted to main phases.
        matches!(
            self.current_phase_kind(),
            PhaseKind::PrecombatMain | PhaseKind::PostcombatMain
        )
    }

    pub fn step_has_priority(step: StepKind) -> bool {
        // CR 502.4, 514.3a — No priority in untap or most cleanup steps.
        !matches!(step, StepKind::Untap | StepKind::Cleanup)
    }

    pub fn advance_step(&mut self) {
        self.cleanup_priority = false;
        self.step_initialized = false;
        self.turn_based_actions_complete = false;

        if self.current_step + 1 < PHASE_STEPS[self.current_phase].len() {
            self.current_step += 1;
            return;
        }

        self.current_step = 0;
        if self.current_phase + 1 < PHASE_STEPS.len() {
            self.current_phase += 1;
            return;
        }

        self.current_phase = 0;
        self.turn_number += 1;
        // CR 305.2 — The land-play allowance resets each new turn.
        self.lands_played = 0;
        // Turn-scoped counters reset with the turn.
        self.history = Default::default();
        self.spells_cast = [Vec::new(), Vec::new()];
        self.ability_resolutions_this_turn.clear();
        self.ability_triggers_this_turn.clear();
        self.active_player = PlayerId((self.active_player.0 + 1) % 2);
    }
}
