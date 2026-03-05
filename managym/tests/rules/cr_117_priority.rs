use managym::{
    agent::action::{ActionSpaceKind, ActionType},
    flow::turn::StepKind,
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
