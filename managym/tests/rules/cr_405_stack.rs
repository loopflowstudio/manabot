use managym::{
    agent::action::ActionType,
    flow::turn::StepKind,
    state::game_object::{PlayerId, Target},
};

use super::helpers::*;

/// CR 405.1 — Stack objects resolve in last-in, first-out order.
#[test]
fn cr_405_1_lifo_resolution() {
    let mut s = Scenario::new(bolt_deck(), mountain_deck(), 4051);

    s.force_cards_in_hand(0, "Mountain", 2);
    s.force_cards_in_hand(0, "Lightning Bolt", 2);

    s.advance_to_active_step(0, StepKind::Main);
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));

    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));

    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    assert_eq!(s.zone_size(0, managym::state::zone::ZoneType::Stack), 2);

    s.pass_priority();
    s.pass_priority();
    s.assert_life(1, 17);
    assert_eq!(s.zone_size(0, managym::state::zone::ZoneType::Stack), 1);

    s.pass_priority();
    s.pass_priority();
    s.assert_life(1, 14);
    assert_eq!(s.zone_size(0, managym::state::zone::ZoneType::Stack), 0);
}

/// CR 405.5 — After a spell resolves, active player gets priority.
#[test]
fn cr_405_5_priority_after_resolution() {
    let mut s = Scenario::new(bolt_deck(), mountain_deck(), 4055);

    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Mountain");
    s.force_card_in_hand(0, "Lightning Bolt");
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));
    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
    assert!(s.choose_target(Target::Player(PlayerId(1))));

    s.pass_priority();
    s.pass_priority();

    assert_eq!(s.action_space().player, Some(PlayerId(0)));
    assert_eq!(s.current_step(), StepKind::Main);
}

/// CR 405 — Counterspell may target a spell object on the stack.
#[test]
fn cr_405_counterspell_targets_stack_object() {
    let mut s = Scenario::new(bolt_deck(), counterspell_deck(), 4059);

    s.advance_to_active_step(1, StepKind::Main);
    s.force_cards_in_hand(1, "Island", 2);
    s.force_card_in_hand(1, "Counterspell");
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));

    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Mountain");
    s.force_card_in_hand(0, "Lightning Bolt");
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));

    s.advance_to_active_step(1, StepKind::Main);
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));

    s.advance_to_active_step(0, StepKind::Main);
    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    s.pass_priority();

    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
    let stack_target = s
        .game()
        .state
        .zones
        .stack_order()
        .last()
        .copied()
        .expect("bolt should be on stack");
    assert!(s.choose_target(Target::StackSpell(stack_target)));
}
