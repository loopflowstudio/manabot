use managym::{
    agent::action::{Action, ActionSpaceKind, ActionType},
    flow::turn::StepKind,
};

use super::helpers::*;

fn setup_player0_elf_ready_to_attack() -> Scenario {
    let mut s = Scenario::new()
        .deck(forest_elves_deck())
        .deck(forest_deck())
        .seed(81)
        .skip_trivial(false)
        .build();

    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Forest");
    s.force_card_in_hand(0, "Llanowar Elves");

    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));
    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
    s.pass_priority();
    s.pass_priority();

    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s
}

/// CR 508.1 — Eligible creatures can be declared as attackers.
#[test]
fn cr_508_1_creature_can_be_declared_as_attacker() {
    let mut s = setup_player0_elf_ready_to_attack();

    assert_eq!(s.action_space().kind, ActionSpaceKind::DeclareAttacker);

    let attack_index = s
        .action_space()
        .actions
        .iter()
        .position(|action| matches!(action, Action::DeclareAttacker { attack: true, .. }))
        .expect("declare-attack action should exist");
    s.step_action(attack_index);

    let elf = s
        .battlefield_permanents_named(0, "Llanowar Elves")
        .into_iter()
        .next()
        .expect("attacking elf should remain on battlefield");
    let perm = s.game().state.permanents[elf.0]
        .as_ref()
        .expect("permanent should exist");
    assert!(perm.attacking);
    assert!(perm.tapped);
}

/// Negative: summoning-sick creatures are not legal attackers.
#[test]
fn cr_508_negative_summoning_sick_creature_not_offered_as_attacker() {
    let mut s = Scenario::new()
        .deck(forest_elves_deck())
        .deck(forest_deck())
        .seed(82)
        .skip_trivial(false)
        .build();

    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Forest");
    s.force_card_in_hand(0, "Llanowar Elves");

    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));
    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
    s.pass_priority();
    s.pass_priority();

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
    let mut s = setup_player0_elf_ready_to_attack();

    let decline_index = s
        .action_space()
        .actions
        .iter()
        .position(|action| matches!(action, Action::DeclareAttacker { attack: false, .. }))
        .expect("decline-attack action should exist");
    s.step_action(decline_index);

    let elf = s
        .battlefield_permanents_named(0, "Llanowar Elves")
        .into_iter()
        .next()
        .expect("elf should remain on battlefield");
    let perm = s.game().state.permanents[elf.0]
        .as_ref()
        .expect("permanent should exist");
    assert!(!perm.attacking);
}
