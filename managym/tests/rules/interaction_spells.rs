// Tests for interaction spells: card-draw effects (Ancestral Recall) and
// mass damage (Pyroclasm), plus their interaction with the stack.

use managym::{
    agent::action::ActionType,
    flow::turn::StepKind,
    state::{
        game_object::{PlayerId, Target},
        zone::ZoneType,
    },
};

use super::helpers::*;

/// CR 121.1 — Resolving a draw spell moves cards from library to hand.
#[test]
fn ancestral_recall_draws_three() {
    let mut s = Scenario::new(recall_deck(), island_deck(), 1211);

    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Island");
    s.force_card_in_hand(0, "Ancestral Recall");
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));
    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));

    // Spell is on the stack; snapshot zone sizes before resolution.
    s.assert_zone_size(0, ZoneType::Stack, 1);
    let hand_before = s.zone_size(0, ZoneType::Hand);
    let library_before = s.zone_size(0, ZoneType::Library);

    s.pass_priority();
    s.pass_priority();

    s.assert_zone_size(0, ZoneType::Stack, 0);
    s.assert_zone_size(0, ZoneType::Hand, hand_before + 3);
    s.assert_zone_size(0, ZoneType::Library, library_before - 3);
    // The spell itself goes to the graveyard.
    s.assert_zone_size(0, ZoneType::Graveyard, 1);
}

/// CR 704.5c — Drawing from an empty library loses the game, including
/// draws caused by spell effects.
#[test]
fn ancestral_recall_decking_loses_game() {
    let mut s = Scenario::new(recall_deck(), island_deck(), 1212);

    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Island");
    s.force_card_in_hand(0, "Ancestral Recall");
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));

    // Empty player 0's library so the draw spell decks them.
    let library: Vec<_> = s
        .game()
        .state
        .zones
        .zone_cards(ZoneType::Library, PlayerId(0)).to_vec();
    for card in library {
        s.game_mut()
            .state
            .zones
            .move_card(card, PlayerId(0), ZoneType::Graveyard);
    }

    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
    s.pass_priority();
    s.pass_priority();

    s.assert_game_over();
    s.assert_winner(1);
}

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
    s.force_permanent_on_battlefield(0, "Grey Ogre");
    s.force_permanent_on_battlefield(0, "Wall of Stone");
    s.force_permanent_on_battlefield(1, "Grey Ogre");
    assert_eq!(s.battlefield_permanents_named(0, "Grey Ogre").len(), 1);
    assert_eq!(s.battlefield_permanents_named(1, "Grey Ogre").len(), 1);

    s.force_card_in_hand(0, "Pyroclasm");
    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
    s.pass_priority();
    s.pass_priority();

    // Both ogres (toughness 2) die; the wall (toughness 8) survives.
    assert!(s.battlefield_permanents_named(0, "Grey Ogre").is_empty());
    assert!(s.battlefield_permanents_named(1, "Grey Ogre").is_empty());
    assert_eq!(s.battlefield_permanents_named(0, "Wall of Stone").len(), 1);
}

/// CR 405 — Counterspell can counter a draw spell on the stack; no cards
/// are drawn and the countered spell goes to the graveyard.
#[test]
fn counterspell_counters_ancestral_recall() {
    let mut s = Scenario::new(recall_deck(), counterspell_deck(), 1214);

    s.advance_to_active_step(1, StepKind::Main);
    s.force_cards_in_hand(1, "Island", 2);
    s.force_card_in_hand(1, "Counterspell");
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));

    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Island");
    s.force_card_in_hand(0, "Ancestral Recall");
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));

    s.advance_to_active_step(1, StepKind::Main);
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));

    s.advance_to_active_step(0, StepKind::Main);
    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
    let hand_before = s.zone_size(0, ZoneType::Hand);
    let library_before = s.zone_size(0, ZoneType::Library);
    s.pass_priority();

    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
    let recall_card = s
        .game()
        .state
        .stack_objects
        .iter()
        .find_map(|obj| match obj {
            managym::state::stack_object::StackObject::Spell(spell) => Some(spell.card),
            _ => None,
        })
        .expect("recall should be on stack");
    assert!(s.choose_target(Target::StackSpell(recall_card)));

    s.pass_priority();
    s.pass_priority();

    // Recall was countered: no cards drawn, recall in the graveyard.
    s.assert_zone_size(0, ZoneType::Stack, 0);
    s.assert_zone_size(0, ZoneType::Hand, hand_before);
    s.assert_zone_size(0, ZoneType::Library, library_before);
    s.assert_zone_size(0, ZoneType::Graveyard, 1);
}
