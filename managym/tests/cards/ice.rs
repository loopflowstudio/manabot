//! Ice Age (`src/cardsets/ice.rs`).
use crate::helpers::{ogre_deck, pyroclasm_deck, Scenario};
use managym::{agent::action::ActionType, flow::turn::StepKind};

/// Pyroclasm deals 2 damage to each creature on both sides; creatures with
/// toughness <= 2 die via state-based actions, tougher ones survive.
#[test]
fn pyroclasm_kills_small_creatures_on_both_sides() {
    let mut s = Scenario::new(pyroclasm_deck(), ogre_deck(), 1213);

    // Two mountains over two turns to afford 1R.
    s.advance_to_active_step(0, StepKind::Main);
    s.force_cards_in_hand(0, "Mountain", 2);
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));

    // Board: 2/2 ogres on both sides, one 0/8 wall for player 0.
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.force_permanent_on_battlefield(0, "Wall of Stone");
    s.force_permanent_on_battlefield(1, "Gray Ogre");
    assert_eq!(s.battlefield_permanents_named(0, "Gray Ogre").len(), 1);
    assert_eq!(s.battlefield_permanents_named(1, "Gray Ogre").len(), 1);

    s.force_card_in_hand(0, "Pyroclasm");
    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
    s.pass_priority();
    s.pass_priority();

    // Both ogres (toughness 2) die; the wall (toughness 8) survives.
    assert!(s.battlefield_permanents_named(0, "Gray Ogre").is_empty());
    assert!(s.battlefield_permanents_named(1, "Gray Ogre").is_empty());
    assert_eq!(s.battlefield_permanents_named(0, "Wall of Stone").len(), 1);
}
