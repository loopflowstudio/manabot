use crate::state::game_object::PlayerId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum PhaseKind {
    Beginning = 0,
    PrecombatMain = 1,
    Combat = 2,
    PostcombatMain = 3,
    Ending = 4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum StepKind {
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

#[derive(Clone, Debug)]
pub struct TurnState {
    pub active_player: PlayerId,
    pub turn_number: u32,
    pub lands_played: u32,
    pub current_phase: usize,
    pub current_step: usize,
    pub step_initialized: bool,
    pub turn_based_actions_complete: bool,
}

impl TurnState {
    pub fn new(active_player: PlayerId) -> Self {
        Self {
            active_player,
            turn_number: 1,
            lands_played: 0,
            current_phase: 0,
            current_step: 0,
            step_initialized: false,
            turn_based_actions_complete: false,
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
        matches!(
            self.current_phase_kind(),
            PhaseKind::PrecombatMain | PhaseKind::PostcombatMain
        )
    }

    pub fn step_has_priority(step: StepKind) -> bool {
        !matches!(step, StepKind::Untap | StepKind::Cleanup)
    }

    pub fn advance_step(&mut self) {
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
        self.lands_played = 0;
        self.active_player = PlayerId((self.active_player.0 + 1) % 2);
    }
}
