use managym::{
    agent::action::{Action, ActionSpaceKind, ActionType},
    flow::turn::StepKind,
};

use super::helpers::*;

fn setup_combat_with_available_blocker() -> Scenario {
    let mut s = Scenario::new()
        .deck(forest_elves_deck())
        .deck(forest_elves_deck())
        .seed(91)
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
    s.force_card_in_hand(1, "Forest");
    s.force_card_in_hand(1, "Llanowar Elves");
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));
    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
    s.pass_priority();
    s.pass_priority();

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    let attack_index = s
        .action_space()
        .actions
        .iter()
        .position(|action| matches!(action, Action::DeclareAttacker { attack: true, .. }))
        .expect("attack action should exist");
    s.step_action(attack_index);

    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    s
}

/// CR 509.1 — Defending player may declare blockers for attacking creatures.
#[test]
fn cr_509_1_declared_blocker_is_recorded_on_attacker() {
    let mut s = setup_combat_with_available_blocker();

    assert_eq!(s.action_space().kind, ActionSpaceKind::DeclareBlocker);

    let block_index = s
        .action_space()
        .actions
        .iter()
        .position(|action| {
            matches!(
                action,
                Action::DeclareBlocker {
                    attacker: Some(_),
                    ..
                }
            )
        })
        .expect("block action should exist");
    s.step_action(block_index);

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

    let no_block_index = s
        .action_space()
        .actions
        .iter()
        .position(|action| matches!(action, Action::DeclareBlocker { attacker: None, .. }))
        .expect("no-block action should exist");
    s.step_action(no_block_index);

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
