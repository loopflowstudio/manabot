use managym::{agent::action::ActionSpaceKind, flow::turn::StepKind};

use super::helpers::*;

/// CR 508.1 — Eligible creatures can be declared as attackers.
#[test]
fn cr_508_1_creature_can_be_declared_as_attacker() {
    let mut s = Scenario::new(forest_elves_deck(), forest_deck(), 81);
    s.setup_attacker_ready();

    assert_eq!(s.action_space().kind, ActionSpaceKind::DeclareAttacker);
    s.declare_attack();

    let elf = s
        .battlefield_permanents_named(0, "Llanowar Elves")
        .into_iter()
        .next()
        .expect("attacking elf should remain on battlefield");
    let perm = s.game().state.permanents[elf]
        .as_ref()
        .expect("permanent should exist");
    assert!(perm.attacking);
    assert!(perm.tapped);
}

/// Negative: summoning-sick creatures are not legal attackers.
#[test]
fn cr_508_negative_summoning_sick_creature_not_offered_as_attacker() {
    let mut s = Scenario::new(forest_elves_deck(), forest_deck(), 82);
    s.play_land_and_cast_creature(0, "Forest", "Llanowar Elves");

    s.advance_to_active_step(0, StepKind::DeclareAttackers);

    assert_ne!(s.action_space().kind, ActionSpaceKind::DeclareAttacker);
    assert!(s
        .game()
        .state
        .permanents
        .iter()
        .flatten()
        .all(|permanent| !permanent.attacking));
}

/// CR 508.1 — Choosing "not attacking" leaves a creature out of combat.
#[test]
fn cr_508_1_decline_attack_keeps_creature_not_attacking() {
    let mut s = Scenario::new(forest_elves_deck(), forest_deck(), 81);
    s.setup_attacker_ready();
    s.decline_attack();

    let elf = s
        .battlefield_permanents_named(0, "Llanowar Elves")
        .into_iter()
        .next()
        .expect("elf should remain on battlefield");
    let perm = s.game().state.permanents[elf]
        .as_ref()
        .expect("permanent should exist");
    assert!(!perm.attacking);
}
