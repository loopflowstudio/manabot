//! Tests for cards filed in `src/ptk.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    play::{cast, settle, target_opponent},
};
use managym::{
    flow::turn::{PhaseKind, StepKind},
    state::{effect::Modification, game_object::PlayerId},
};

/// The opponent's whole next turn has no combat phase; the record then ends,
/// so the turn after it has combat again.
#[test]
fn empty_city_ruse_skips_the_opponents_next_combat_phases() {
    let mut s = five_color_setup(&["Empty City Ruse"]);
    cast(&mut s, "Empty City Ruse");
    target_opponent(&mut s);
    settle(&mut s);
    let skips = |s: &crate::support::game::Scenario| {
        s.game()
            .player_effects(PlayerId(1))
            .any(|effect| effect.modification == Modification::SkipsCombatPhasesNextTurn)
    };
    assert!(skips(&s));
    s.advance_to_active_step(1, StepKind::Upkeep);
    let mut phases = Vec::new();
    while s.active_player() == 1 {
        phases.push(s.current_phase());
        s.advance_default_action();
    }
    assert!(phases.contains(&PhaseKind::PostcombatMain));
    assert!(!phases.contains(&PhaseKind::Combat));
    assert!(!skips(&s));
    s.advance_to_active_step(1, StepKind::BeginningOfCombat);
}
