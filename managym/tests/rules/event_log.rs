use managym::{
    agent::action::ActionType,
    flow::{
        event::{DamageTarget, GameEvent},
        turn::StepKind,
    },
    state::game_object::{CardId, PlayerId, Target},
};

use super::helpers::*;

fn resolve_player_bolt(seed: u64) -> (Scenario, CardId, Vec<GameEvent>) {
    let mut s = Scenario::new(bolt_deck(), mountain_deck(), seed);
    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Mountain");
    s.force_card_in_hand(0, "Lightning Bolt");
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));
    let _ = s.drain_events();

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
    s.pass_priority();
    let events = s.drain_events();
    (s, bolt, events)
}

#[test]
fn event_log_records_zone_changes() {
    let (_s, bolt, events) = resolve_player_bolt(7001);

    assert!(events.contains(&GameEvent::CardMoved {
        card: bolt,
        from: managym::state::zone::ZoneType::Hand,
        to: managym::state::zone::ZoneType::Stack,
    }));
    assert!(events.contains(&GameEvent::CardMoved {
        card: bolt,
        from: managym::state::zone::ZoneType::Stack,
        to: managym::state::zone::ZoneType::Graveyard,
    }));
}

#[test]
fn event_log_records_damage() {
    let (_s, bolt, events) = resolve_player_bolt(7002);

    assert!(events.contains(&GameEvent::DamageDealt {
        source: Some(bolt),
        target: DamageTarget::Player(PlayerId(1)),
        amount: 3,
    }));
}

#[test]
fn event_log_records_life_change() {
    let (_s, _bolt, events) = resolve_player_bolt(7003);

    assert!(events.contains(&GameEvent::LifeChanged {
        player: PlayerId(1),
        old: 20,
        new: 17,
    }));
}

#[test]
fn event_log_records_counterspell() {
    let mut s = Scenario::new(bolt_deck(), counterspell_deck(), 7004);

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
    let _ = s.drain_events();

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
    let counterspell = s
        .game()
        .state
        .zones
        .stack_order()
        .last()
        .copied()
        .expect("counterspell should be on stack");
    s.pass_priority();
    s.pass_priority();
    let events = s.drain_events();

    assert!(events.contains(&GameEvent::SpellCountered {
        card: bolt,
        by: Some(counterspell),
    }));
}
