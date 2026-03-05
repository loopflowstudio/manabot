use managym::{
    agent::action::ActionType,
    flow::turn::{PhaseKind, StepKind},
    state::zone::ZoneType,
};

use super::helpers::*;

/// CR 305.1 — A player may play a land during a main phase of their turn when stack is empty.
#[test]
fn cr_305_1_play_land_in_main_phase() {
    let mut s = Scenario::new(mountain_deck(), mountain_deck(), 31);

    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Mountain");

    let hand_before = s.zone_size(0, ZoneType::Hand);
    s.assert_action_available(ActionType::PriorityPlayLand);
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));

    assert_eq!(s.zone_size(0, ZoneType::Hand), hand_before - 1);
    assert_eq!(s.zone_size(0, ZoneType::Battlefield), 1);
}

/// CR 305.2 — A player normally may play only one land per turn.
#[test]
fn cr_305_2_only_one_land_per_turn() {
    let mut s = Scenario::new(mountain_deck(), mountain_deck(), 32);

    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Mountain");
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));

    s.assert_action_not_available(ActionType::PriorityPlayLand);
}

/// CR 305.1 — Land plays follow sorcery timing.
#[test]
fn cr_305_1_land_play_not_available_in_combat() {
    let mut s = Scenario::new(mountain_deck(), mountain_deck(), 33);

    s.advance_to_phase(PhaseKind::Combat);
    assert_eq!(s.current_phase(), PhaseKind::Combat);
    s.assert_action_not_available(ActionType::PriorityPlayLand);
}

/// CR 305.2 — Land play count resets as turns advance.
#[test]
fn cr_305_2_land_play_resets_next_turn() {
    let mut s = Scenario::new(mountain_deck(), mountain_deck(), 34);

    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Mountain");
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));

    s.pass_priority();
    s.pass_priority();
    s.advance_to_active_step(0, StepKind::Main);
    s.assert_action_available(ActionType::PriorityPlayLand);
}
