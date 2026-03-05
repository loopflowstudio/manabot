use managym::{agent::action::ActionType, flow::turn::StepKind};

use super::helpers::*;

fn setup_damaged_permanent_before_ending_step(
) -> (Scenario, managym::state::game_object::PermanentId) {
    let mut s = Scenario::new()
        .deck(mountain_deck())
        .deck(mountain_deck())
        .seed(111)
        .skip_trivial(false)
        .build();

    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Mountain");
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));

    let permanent_id = s
        .battlefield_permanents_named(0, "Mountain")
        .into_iter()
        .next()
        .expect("mountain should be on battlefield");

    s.game_mut().state.permanents[permanent_id.0]
        .as_mut()
        .expect("permanent should exist")
        .damage = 3;

    s.advance_to_active_step(0, StepKind::End);
    (s, permanent_id)
}

/// CR 514.2 — Marked damage is removed from permanents during cleanup.
#[test]
fn cr_514_2_cleanup_clears_marked_damage() {
    let (mut s, permanent_id) = setup_damaged_permanent_before_ending_step();

    s.pass_priority();
    s.pass_priority();

    let damage_after = s.game().state.permanents[permanent_id.0]
        .as_ref()
        .expect("permanent should still exist")
        .damage;
    assert_eq!(damage_after, 0);
}

/// Negative: marked damage remains until cleanup actually happens.
#[test]
fn cr_514_negative_damage_not_cleared_before_cleanup() {
    let (s, permanent_id) = setup_damaged_permanent_before_ending_step();

    let damage_before = s.game().state.permanents[permanent_id.0]
        .as_ref()
        .expect("permanent should still exist")
        .damage;
    assert_eq!(damage_before, 3);
}
