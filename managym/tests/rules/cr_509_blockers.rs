use managym::{agent::action::ActionSpaceKind, flow::turn::StepKind};

use crate::support::game::*;

fn setup_combat_with_available_blocker() -> Scenario {
    let mut s = Scenario::new(
        deck(&[("Forest", 24), ("Llanowar Elves", 36)]),
        deck(&[("Forest", 24), ("Llanowar Elves", 36)]),
        91,
    );
    s.play_land_and_cast_creature(0, "Forest", "Llanowar Elves");
    s.advance_to_active_step(1, StepKind::Main);
    s.play_land_and_cast_creature(1, "Forest", "Llanowar Elves");
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    s
}

/// CR 509.1 — Defending player may declare blockers for attacking creatures.
#[test]
fn cr_509_1_declared_blocker_is_recorded_on_attacker() {
    let mut s = setup_combat_with_available_blocker();

    assert_eq!(s.action_space().kind, ActionSpaceKind::DeclareBlocker);
    s.declare_block();

    let combat = s
        .game()
        .state
        .combat
        .as_ref()
        .expect("combat state should exist");
    let attacker = *combat.attackers.first().expect("one attacker should exist");
    let blockers = combat
        .attacker_to_blockers
        .get(&attacker)
        .expect("attacker should have blocker list");
    assert_eq!(blockers.len(), 1);
}

/// Negative: choosing "no block" keeps attacker unblocked.
#[test]
fn cr_509_negative_decline_block_leaves_attacker_unblocked() {
    let mut s = setup_combat_with_available_blocker();
    s.decline_block();

    let combat = s
        .game()
        .state
        .combat
        .as_ref()
        .expect("combat state should exist");
    let attacker = *combat.attackers.first().expect("one attacker should exist");
    let blockers = combat
        .attacker_to_blockers
        .get(&attacker)
        .expect("attacker should have blocker list");
    assert!(blockers.is_empty());
}
