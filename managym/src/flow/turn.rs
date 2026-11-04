use std::collections::BTreeMap;

use crate::state::{
    ability::PlayerControlDuration,
    game_object::{ObjectRef, PlayerId},
    predicate::CardPredicate,
};

/// A once-each-turn use: the exact source object and its ability index.
pub type OnceEachTurnKey = (ObjectRef, usize);

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

/// An additional combat phase this turn (CR 500.8).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct AdditionalCombat {
    /// Only matching creatures can attack during it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attackers: Option<CardPredicate>,
}

/// Scheduled or active player control, in creation order. Later active
/// effects win; neither the turn's active player nor resource ownership moves.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct PlayerControl {
    pub player: PlayerId,
    pub controller: PlayerId,
    pub duration: PlayerControlDuration,
    pub active: bool,
}

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
    /// The kinds of bending this player has done: one bit per [`Bending`]
    /// ("if you've done all four this turn").
    #[serde(default, skip_serializing_if = "crate::state::is_zero")]
    pub bent: u8,
}

/// A kind of bending (CR 701.65–701.67 and firebending, CR 702.189); its
/// value is its bit in [`PlayerTurnHistory::bent`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[repr(u8)]
pub enum Bending {
    Water = 1,
    Earth = 2,
    Fire = 4,
    Air = 8,
}

impl Bending {
    pub const ALL: [Bending; 4] = [Bending::Water, Bending::Earth, Bending::Fire, Bending::Air];
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct TurnState {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub player_controls: Vec<PlayerControl>,
    pub active_player: PlayerId,
    pub turn_number: u32,
    pub lands_played: u32,
    pub spells_cast: [Vec<crate::state::game_object::CardId>; 2],
    /// Each player's history this turn, by player index. Resets every turn.
    pub history: [PlayerTurnHistory; 2],
    /// Once-each-turn uses by exact source object and ability index: a
    /// `TriggerCondition::OnceEachTurn` is recorded as it fires, an
    /// `Effect::MayOnceEachTurn` option as it is taken.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub ability_triggers_this_turn: BTreeMap<OnceEachTurnKey, u32>,
    /// Times each triggered ability `(card index, ability index)` has
    /// resolved this turn (for "the Nth time this ability has resolved this
    /// turn" gating). Resets every turn.
    pub ability_resolutions_this_turn: BTreeMap<(usize, usize), u32>,
    /// Extra turns still to be taken (CR 500.7), the most recently created
    /// last: it is taken first.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub extra_turns: Vec<PlayerId>,
    /// While extra turns are being taken: the player whose turn follows them,
    /// the one who would have played after the turn they interrupted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resume_with: Option<PlayerId>,
    /// Additional combat phases to follow the current one this turn (CR
    /// 500.8), the most recently created last; each may limit its attackers.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub additional_combats: Vec<AdditionalCombat>,
    /// During an additional combat phase that limits its attackers: which
    /// creatures can attack ("only land creatures can attack").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub combat_attackers: Option<CardPredicate>,
    pub current_phase: usize,
    pub current_step: usize,
    pub step_initialized: bool,
    pub turn_based_actions_complete: bool,
    pub cleanup_priority: bool,
}

impl TurnState {
    pub fn new(active_player: PlayerId) -> Self {
        Self {
            player_controls: Vec::new(),
            active_player,
            turn_number: 1,
            lands_played: 0,
            spells_cast: [Vec::new(), Vec::new()],
            history: Default::default(),
            ability_resolutions_this_turn: BTreeMap::new(),
            ability_triggers_this_turn: BTreeMap::new(),
            extra_turns: Vec::new(),
            resume_with: None,
            additional_combats: Vec::new(),
            combat_attackers: None,
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
        let ended = self.current_step_kind();
        self.player_controls.retain(|control| {
            !control.active
                || ended
                    != match control.duration {
                        PlayerControlDuration::NextCombat => StepKind::EndOfCombat,
                        PlayerControlDuration::NextTurn => StepKind::Cleanup,
                    }
        });
        self.advance_step_position();
        // This runs before the new interval's turn-based actions or triggers.
        let started = self.current_step_kind();
        for control in &mut self.player_controls {
            if control.player == self.active_player && !control.active {
                control.active = started
                    == match control.duration {
                        PlayerControlDuration::NextCombat => StepKind::BeginningOfCombat,
                        PlayerControlDuration::NextTurn => StepKind::Untap,
                    };
            }
        }
    }

    fn advance_step_position(&mut self) {
        self.cleanup_priority = false;
        self.step_initialized = false;
        self.turn_based_actions_complete = false;

        if self.current_step + 1 < PHASE_STEPS[self.current_phase].len() {
            self.current_step += 1;
            return;
        }

        self.current_step = 0;
        if self.current_phase_kind() == PhaseKind::Combat {
            // CR 500.8: an additional combat phase follows this one.
            if let Some(combat) = self.additional_combats.pop() {
                self.combat_attackers = combat.attackers;
                return;
            }
            self.combat_attackers = None;
        }
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
        self.active_player = self.next_turn_player();
    }

    /// Who takes the next turn: the most recent extra turn's player, else
    /// the player after the last ordinary turn (CR 500.7).
    fn next_turn_player(&mut self) -> PlayerId {
        let after_this = PlayerId((self.active_player.0 + 1) % 2);
        match self.extra_turns.pop() {
            Some(player) => {
                self.resume_with.get_or_insert(after_this);
                player
            }
            None => self.resume_with.take().unwrap_or(after_this),
        }
    }
}
