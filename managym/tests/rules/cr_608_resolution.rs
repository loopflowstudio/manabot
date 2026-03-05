use managym::{
    agent::action::ActionType,
    flow::turn::StepKind,
    state::{
        game_object::{PlayerId, Target},
        zone::ZoneType,
    },
};

use super::helpers::*;

/// CR 608.2b subset — A spell with an illegal target has no effect.
#[test]
fn cr_608_2b_illegal_target_spell_has_no_effect() {
    let mut s = Scenario::new(bolt_deck(), forest_elves_deck(), 6081);

    s.force_cards_in_hand(0, "Mountain", 2);
    s.force_cards_in_hand(0, "Lightning Bolt", 2);

    s.advance_to_active_step(0, StepKind::Main);
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));

    s.advance_to_active_step(1, StepKind::Main);
    s.force_card_in_hand(1, "Forest");
    s.force_card_in_hand(1, "Llanowar Elves");
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));
    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
    s.pass_priority();
    s.pass_priority();

    s.advance_to_active_step(0, StepKind::Main);
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));
    let elf = s.battlefield_permanents_named(1, "Llanowar Elves")[0];

    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
    assert!(s.choose_target(Target::Permanent(elf)));
    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
    assert!(s.choose_target(Target::Permanent(elf)));

    s.pass_priority();
    s.pass_priority();
    assert!(s
        .battlefield_permanents_named(1, "Llanowar Elves")
        .is_empty());
    s.assert_life(1, 20);

    s.pass_priority();
    s.pass_priority();
    s.assert_life(1, 20);
}

/// CR 608 + Counterspell text — Counterspell counters its legal stack target.
#[test]
fn cr_608_counterspell_moves_target_to_graveyard() {
    let mut s = Scenario::new(bolt_deck(), counterspell_deck(), 6082);

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
    let bolt = s
        .game()
        .state
        .zones
        .stack_order()
        .last()
        .copied()
        .expect("bolt should be on stack");
    s.pass_priority();

    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
    assert!(s.choose_target(Target::StackSpell(bolt)));

    s.pass_priority();
    s.pass_priority();

    assert_eq!(
        s.game().state.zones.zone_of(bolt),
        Some(ZoneType::Graveyard)
    );
    s.assert_life(1, 20);
}
