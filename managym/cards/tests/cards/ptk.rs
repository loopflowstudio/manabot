//! Tests for cards filed in `src/ptk.rs`, in card order.

use crate::support::{
    game::{clear_summoning_sickness, five_color_setup},
    play::{cast, choose, settle, target_opponent},
};
use managym::{
    agent::action::{Action, ActionSpaceKind},
    flow::turn::{PhaseKind, StepKind},
    state::{
        effect::Modification,
        game_object::{PlayerId, Target},
    },
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

/// Every creature able to block the targeted attacker blocks it, even with
/// another attacker to block instead; declining is not offered.
#[test]
fn taunting_challenge_makes_every_able_creature_block_its_target() {
    let mut s = five_color_setup(&["Taunting Challenge", "Hill Giant", "Gray Ogre"]);
    let giant = s.force_permanent_on_battlefield(0, "Hill Giant");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let blockers = [
        s.force_permanent_on_battlefield(1, "Hill Giant"),
        s.force_permanent_on_battlefield(1, "Gray Ogre"),
    ];
    for id in [giant, ogre] {
        clear_summoning_sickness(&mut s, id);
    }
    cast(&mut s, "Taunting Challenge");
    assert!(s.choose_target(Target::Permanent(giant)));
    settle(&mut s);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    while s.action_space().kind == ActionSpaceKind::DeclareAttacker {
        choose(&mut s, |action| {
            matches!(action, Action::DeclareAttacker { attack: true, .. })
        });
    }
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    while s.action_space().kind == ActionSpaceKind::DeclareBlocker {
        assert!(s.action_space().actions.iter().all(|action| matches!(
            action,
            Action::DeclareBlocker { attacker: Some(attacker), .. } if *attacker == giant
        )));
        s.step_action(0);
    }
    let combat = s.game().state.combat.as_ref().unwrap();
    let mut blocked = combat.attacker_to_blockers[&giant].clone();
    blocked.sort();
    assert_eq!(blocked, blockers);
    assert!(combat.attacker_to_blockers[&ogre].is_empty());
}
