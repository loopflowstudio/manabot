use managym::{
    agent::action::{ActionSpaceKind, ActionType},
    flow::turn::StepKind,
    state::{
        game_object::{PlayerId, Target},
        zone::ZoneType,
    },
};

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

/// CR 601 / 117.1a — Instants are castable when the stack is nonempty.
#[test]
fn cr_601_instant_timing() {
    let mut s = Scenario::new(forest_elves_deck(), bolt_deck(), 65);

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

    assert_eq!(s.action_space().player, Some(PlayerId(1)));
    s.assert_action_available(ActionType::PriorityCastSpell);
}

/// CR 601.2c — Targeted spells require explicit target selection.
#[test]
fn cr_601_target_selection_required_for_targeted_spells() {
    let mut s = Scenario::new(bolt_deck(), mountain_deck(), 66);

    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Mountain");
    s.force_card_in_hand(0, "Lightning Bolt");
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));
    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));

    assert_eq!(s.action_space().kind, ActionSpaceKind::ChooseTarget);
    assert_eq!(s.zone_size(0, ZoneType::Stack), 0);
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    assert_eq!(s.zone_size(0, ZoneType::Stack), 1);
}

/// CR 601.2c subset — Spells with no legal targets are not offered.
#[test]
fn cr_601_spell_not_offered_without_legal_target() {
    let mut s = Scenario::new(counterspell_deck(), mountain_deck(), 67);

    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Island");
    s.force_card_in_hand(0, "Counterspell");
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));

    s.assert_action_not_available(ActionType::PriorityCastSpell);
}

/// CR 117.1a / 307.1 — Sorceries cannot be cast as responses on a nonempty stack.
#[test]
fn cr_601_sorcery_cannot_respond() {
    let mut s = Scenario::new(forest_elves_deck(), forest_elves_deck(), 68);

    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Forest");
    s.force_card_in_hand(0, "Llanowar Elves");
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));
    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
    s.pass_priority();

    assert_eq!(s.action_space().player, Some(PlayerId(1)));
    s.assert_action_not_available(ActionType::PriorityCastSpell);
}
