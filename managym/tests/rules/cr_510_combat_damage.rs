use managym::{
    agent::action::{Action, ActionType},
    flow::turn::StepKind,
};

use super::helpers::*;

fn setup_attacking_elf_vs_empty_board() -> Scenario {
    let mut s = Scenario::new(forest_elves_deck(), forest_deck(), 101);

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

fn setup_attacking_elf_with_blocker_available() -> Scenario {
    let mut s = Scenario::new(forest_elves_deck(), forest_elves_deck(), 102);

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
    s
}

/// CR 510.1c — Unblocked attackers deal combat damage to the defending player.
#[test]
fn cr_510_1c_unblocked_attacker_deals_player_damage() {
    let mut s = setup_attacking_elf_vs_empty_board();

    let attack_index = s
        .action_space()
        .actions
        .iter()
        .position(|action| matches!(action, Action::DeclareAttacker { attack: true, .. }))
        .expect("attack action should exist");
    s.step_action(attack_index);

    let life_before = s.life(1);
    s.advance_to_active_step(0, StepKind::CombatDamage);

    assert_eq!(s.life(1), life_before - 1);
}

/// CR 510.1a — A blocked attacker assigns damage to creatures, not the defending player.
#[test]
fn cr_510_1a_blocked_attacker_does_not_damage_player() {
    let mut s = setup_attacking_elf_with_blocker_available();

    let attack_index = s
        .action_space()
        .actions
        .iter()
        .position(|action| matches!(action, Action::DeclareAttacker { attack: true, .. }))
        .expect("attack action should exist");
    s.step_action(attack_index);

    s.advance_to_active_step(0, StepKind::DeclareBlockers);
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

    let life_before = s.life(1);
    s.advance_to_active_step(0, StepKind::CombatDamage);

    assert_eq!(s.life(1), life_before);
}

/// Negative: if no attacker is declared, no combat damage is dealt to players.
#[test]
fn cr_510_negative_no_attacker_means_no_player_damage() {
    let mut s = setup_attacking_elf_vs_empty_board();

    let decline_index = s
        .action_space()
        .actions
        .iter()
        .position(|action| matches!(action, Action::DeclareAttacker { attack: false, .. }))
        .expect("decline action should exist");
    s.step_action(decline_index);

    let life_before = s.life(1);
    s.advance_to_active_step(0, StepKind::CombatDamage);

    assert_eq!(s.life(1), life_before);
}
