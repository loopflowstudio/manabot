use managym::{
    agent::action::ActionType,
    flow::{
        event::{DamageTarget, GameEvent},
        turn::StepKind,
    },
    state::{game_object::PlayerId, stack_object::StackObject, zone::ZoneType},
};

use super::helpers::*;

fn cast_llanowar_elves(
    seed: u64,
) -> (
    Scenario,
    managym::state::game_object::CardId,
    Vec<GameEvent>,
) {
    let mut s = Scenario::new(forest_elves_deck(), mountain_deck(), seed);
    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Forest");
    s.force_card_in_hand(0, "Llanowar Elves");
    let _ = s.drain_events();

    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));
    let _ = s.drain_events();
    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
    let elf = s
        .game()
        .state
        .stack_objects
        .last()
        .and_then(|stack_object| match stack_object {
            StackObject::Spell(spell) => Some(spell.card),
            _ => None,
        })
        .expect("elf should be on stack");
    s.pass_priority();
    s.pass_priority();

    let events = s.drain_events();
    (s, elf, events)
}

fn cast_manowar_to_trigger(seed: u64) -> Vec<GameEvent> {
    let mut s = Scenario::new(manowar_deck(), island_deck(), seed);
    let islands: Vec<_> = s
        .game()
        .state
        .cards
        .iter()
        .enumerate()
        .filter(|(_, card)| card.owner == PlayerId(0) && card.name == "Island")
        .map(|(idx, _)| managym::state::game_object::CardId(idx))
        .take(3)
        .collect();
    for island in islands {
        s.game_mut()
            .state
            .zones
            .move_card(island, PlayerId(0), ZoneType::Hand);
    }
    s.force_card_in_hand(0, "Man-o'-War");
    let _ = s.drain_events();

    for land_drop in 0..3 {
        s.advance_to_active_step(0, StepKind::Main);
        assert!(s.take_action_by_type(ActionType::PriorityPlayLand));
        if land_drop < 2 {
            s.pass_priority();
            s.pass_priority();
            let _ = s.drain_events();
        }
    }

    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
    s.pass_priority();
    s.pass_priority();
    let _ = s.drain_events();
    s.choose_target_named("Man-o'-War");
    s.drain_events()
}

#[test]
fn spell_events_are_observable_for_creature_spells() {
    let (_s, elf, events) = cast_llanowar_elves(7101);

    assert!(events.contains(&GameEvent::SpellCast {
        card: elf,
        target: None,
    }));
    assert!(events.contains(&GameEvent::SpellResolved { card: elf }));
    assert!(events.contains(&GameEvent::CardMoved {
        card: elf,
        from: Some(ZoneType::Hand),
        to: ZoneType::Stack,
        controller: PlayerId(0),
    }));
    assert!(events.contains(&GameEvent::CardMoved {
        card: elf,
        from: Some(ZoneType::Stack),
        to: ZoneType::Battlefield,
        controller: PlayerId(0),
    }));

    let spell_cast_index = events
        .iter()
        .position(|event| matches!(event, GameEvent::SpellCast { card, .. } if *card == elf))
        .expect("spell cast event should exist");
    let spell_resolved_index = events
        .iter()
        .position(|event| matches!(event, GameEvent::SpellResolved { card, .. } if *card == elf))
        .expect("spell resolved event should exist");
    assert!(spell_cast_index < spell_resolved_index);
}

#[test]
fn combat_damage_emits_damage_and_life_change_events() {
    let mut s = Scenario::new(forest_elves_deck(), forest_deck(), 7102);
    s.setup_attacker_ready();
    let attacker = s
        .battlefield_permanents_named(0, "Llanowar Elves")
        .into_iter()
        .next()
        .expect("attacker should exist");
    let _ = s.drain_events();

    s.declare_attack();
    let _ = s.drain_events();
    s.advance_to_active_step(0, StepKind::CombatDamage);
    let events = s.drain_events();

    // Find the attacker's card to use as source
    let attacker_card = s.game().state.permanents[attacker]
        .as_ref()
        .map(|p| p.card)
        .expect("attacker permanent should exist");

    assert!(events.contains(&GameEvent::DamageDealt {
        source: Some(attacker_card),
        target: DamageTarget::Player(PlayerId(1)),
        amount: 1,
    }));
    assert!(events.contains(&GameEvent::LifeChanged {
        player: PlayerId(1),
        old: 20,
        new: 19,
    }));
}

#[test]
fn triggered_abilities_emit_observable_trigger_events() {
    let events = cast_manowar_to_trigger(7103);

    assert!(events.iter().any(|event| {
        matches!(
            event,
            GameEvent::AbilityTriggered {
                source_card: _,
                controller: PlayerId(0),
            }
        )
    }));
}

#[test]
fn observation_event_window_resets_after_drain() {
    let (mut s, _elf, events) = cast_llanowar_elves(7104);
    assert!(!events.is_empty());
    assert!(s.drain_events().is_empty());
}
