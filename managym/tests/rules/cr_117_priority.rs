use managym::{
    agent::action::{ActionSpaceKind, ActionType},
    flow::turn::StepKind,
    state::game_object::{PlayerId, Target},
};

use super::helpers::*;

/// CR 117.3b — The active player receives priority first.
#[test]
fn cr_117_3b_active_player_gets_priority_first() {
    let mut s = Scenario::new(mountain_deck(), mountain_deck(), 21);

    s.advance_to_step(StepKind::Main);

    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
    assert_eq!(
        s.action_space().player,
        Some(managym::state::game_object::PlayerId(0))
    );
}

/// CR 117.4 — If all players pass in succession on an empty stack, the game advances.
#[test]
fn cr_117_4_all_passes_with_empty_stack_advances_step() {
    let mut s = Scenario::new(mountain_deck(), mountain_deck(), 22);

    s.advance_to_active_step(0, StepKind::Main);
    s.pass_priority();
    s.pass_priority();

    assert_eq!(s.current_step(), StepKind::BeginningOfCombat);
}

/// Negative: nonactive player does not get sorcery-speed actions on opponent's turn.
#[test]
fn cr_117_negative_nonactive_player_has_only_pass_priority() {
    let mut s = Scenario::new(mountain_deck(), mountain_deck(), 23);

    s.advance_to_active_step(0, StepKind::Main);
    s.pass_priority();

    assert_eq!(
        s.action_space().player,
        Some(managym::state::game_object::PlayerId(1))
    );
    s.assert_action_not_available(ActionType::PriorityPlayLand);
    s.assert_action_not_available(ActionType::PriorityCastSpell);
    s.assert_action_available(ActionType::PriorityPassPriority);
}

fn setup_nap_instant_response_window(seed: u64) -> Scenario {
    let mut s = Scenario::new(forest_elves_deck(), bolt_deck(), seed);

    s.advance_to_active_step(1, StepKind::Main);
    s.force_card_in_hand(1, "Mountain");
    s.force_card_in_hand(1, "Lightning Bolt");
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));

    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Forest");
    s.force_card_in_hand(0, "Llanowar Elves");
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));
    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
    s.pass_priority();

    s
}

/// CR 117.1a — A nonactive player can cast an instant while holding priority.
#[test]
fn cr_117_1a_instant_cast_during_opponents_turn() {
    let mut s = setup_nap_instant_response_window(24);

    assert_eq!(s.action_space().player, Some(PlayerId(1)));
    s.assert_action_available(ActionType::PriorityCastSpell);
    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
    assert_eq!(s.action_space().kind, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Player(PlayerId(0))));
}

/// CR 117.3b — After any action, the active player gets priority.
#[test]
fn cr_117_3b_after_action_ap_gets_priority() {
    let mut s = setup_nap_instant_response_window(25);

    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
    assert!(s.choose_target(Target::Player(PlayerId(0))));

    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
    assert_eq!(s.action_space().player, Some(PlayerId(0)));
}

/// Regression: nonactive player must not keep priority after casting.
#[test]
fn cr_117_regression_nap_does_not_retain_priority_after_cast() {
    let mut s = setup_nap_instant_response_window(26);

    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
    assert!(s.choose_target(Target::Player(PlayerId(0))));

    assert_ne!(s.action_space().player, Some(PlayerId(1)));
}
