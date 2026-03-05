use managym::{agent::action::ActionType, flow::turn::StepKind};

use super::helpers::*;

/// Grey Ogre (2/2) with 1 non-lethal damage, positioned before cleanup.
/// Needs 3 Mountains (2R cost), so we play one land per turn for 3 turns.
fn setup_damaged_creature_before_cleanup() -> (Scenario, managym::state::game_object::PermanentId) {
    let mut s = Scenario::new(ogre_deck(), mountain_deck(), 111);

    // Play one Mountain per turn for 3 turns to afford Grey Ogre (2R).
    for i in 0..3 {
        s.advance_to_active_step(0, StepKind::Main);
        s.force_card_in_hand(0, "Mountain");
        assert!(
            s.take_action_by_type(ActionType::PriorityPlayLand),
            "land play failed on turn {i}"
        );
        s.pass_priority();
        s.pass_priority();
    }

    // Cast Grey Ogre on turn 4.
    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Grey Ogre");
    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
    s.pass_priority();
    s.pass_priority();

    let ogre_id = s
        .battlefield_permanents_named(0, "Grey Ogre")
        .into_iter()
        .next()
        .expect("ogre should be on battlefield");

    s.game_mut().state.permanents[ogre_id.0]
        .as_mut()
        .expect("permanent should exist")
        .damage = 1;

    s.advance_to_active_step(0, StepKind::End);
    (s, ogre_id)
}

/// CR 514.2 — Marked damage is removed from permanents during cleanup.
#[test]
fn cr_514_2_cleanup_clears_marked_damage() {
    let (mut s, ogre_id) = setup_damaged_creature_before_cleanup();

    s.pass_priority();
    s.pass_priority();

    let damage_after = s.game().state.permanents[ogre_id.0]
        .as_ref()
        .expect("ogre should still exist")
        .damage;
    assert_eq!(damage_after, 0);
}

/// Negative: marked damage remains until cleanup actually happens.
#[test]
fn cr_514_negative_damage_not_cleared_before_cleanup() {
    let (s, ogre_id) = setup_damaged_creature_before_cleanup();

    let damage_before = s.game().state.permanents[ogre_id.0]
        .as_ref()
        .expect("ogre should still exist")
        .damage;
    assert_eq!(damage_before, 1);
}
