use managym::{agent::action::ActionType, flow::turn::StepKind, state::zone::ZoneType};

use super::helpers::*;

/// CR 601.2 — A legal cast at sorcery speed becomes available in main phase.
#[test]
fn cr_601_2_cast_spell_available_when_cost_is_payable() {
    let mut s = Scenario::new(forest_elves_deck(), mountain_deck(), 61);

    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Forest");
    s.force_card_in_hand(0, "Llanowar Elves");

    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));
    s.assert_action_available(ActionType::PriorityCastSpell);
}

/// CR 601.2i and 608.3 — Cast spells go to stack, then resolve to battlefield.
#[test]
fn cr_601_2i_spell_moves_from_stack_to_battlefield_on_resolution() {
    let mut s = Scenario::new(forest_elves_deck(), mountain_deck(), 62);

    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Forest");
    s.force_card_in_hand(0, "Llanowar Elves");

    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));
    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
    assert_eq!(s.zone_size(0, ZoneType::Stack), 1);

    s.pass_priority();
    s.pass_priority();

    assert_eq!(s.zone_size(0, ZoneType::Stack), 0);
    assert!(
        !s.battlefield_permanents_named(0, "Llanowar Elves")
            .is_empty(),
        "spell should resolve as a permanent"
    );
}

/// CR 117.1a / 307.1 — Sorcery-speed casting is unavailable during combat.
#[test]
fn cr_601_negative_no_sorcery_casting_during_combat() {
    let mut s = Scenario::new(forest_elves_deck(), mountain_deck(), 63);

    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Forest");
    s.force_card_in_hand(0, "Llanowar Elves");
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));

    s.advance_to_active_step(0, StepKind::BeginningOfCombat);
    s.assert_action_not_available(ActionType::PriorityCastSpell);
}

/// Negative: lands are played, not cast as spells.
#[test]
fn cr_601_negative_land_cards_are_not_cast_actions() {
    let mut s = Scenario::new(mountain_deck(), mountain_deck(), 64);

    s.advance_to_active_step(0, StepKind::Main);
    s.assert_action_not_available(ActionType::PriorityCastSpell);
}
