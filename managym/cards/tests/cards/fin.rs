//! Final Fantasy card behavior through native casting, choices and zone moves.
use crate::support::{
    game::{five_color_setup, pin_top, Scenario},
    inspect::{counters_of, library_top, now, offered_cards, permanent_of, top_spell, zone_of},
    play::{cast, cast_card, command, resolve_top, settle, spend_colorless},
};
use managym::{
    agent::{
        action::Action,
        observation::Observation,
        observation_encoder::{encode, ObservationEncoderConfig},
        observation_layout::stack,
    },
    flow::{event::GameEvent, turn::StepKind},
    state::{
        card::Keyword,
        game_object::{ObjectId, PlayerId},
        permanent::CounterKind,
        zone::ZoneType,
    },
};

/// Mill the first card by Command, leaving the second on top for chapter I.
fn surveil(s: &mut Scenario, keep: &str) -> ObjectId {
    let cards = pin_top(s, &["Forest", keep]);
    resolve_top(s);
    assert_eq!(offered_cards(s), cards);
    command(
        s,
        |a| matches!(a, Action::SelectCard { object, .. } if *object == cards[0]),
    );
    command(s, |a| matches!(a, Action::Decline { .. }));
    assert_eq!(zone_of(s, cards[0]), Some(ZoneType::Graveyard));
    assert_eq!(library_top(s, 0), cards[1]);
    cards[1]
}

fn check_cast(s: &Scenario, from_graveyard: bool) {
    let obs = Observation::for_player(s.game(), PlayerId(1));
    assert_eq!(
        obs.stack_objects[0]
            .spell
            .as_ref()
            .unwrap()
            .cast_from_graveyard,
        from_graveyard
    );
    let encoded = encode(&obs, &ObservationEncoderConfig::default()).unwrap();
    assert_eq!(
        encoded.stack[stack::CAST_FROM_GRAVEYARD],
        f32::from(from_graveyard)
    );
}

#[test]
fn esper_origins_flashes_back_through_exile_and_finishes_its_saga() {
    let mut s = five_color_setup(&["Esper Origins", "Gray Ogre"]);
    let ally = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let opponent = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let spell = cast(&mut s, "Esper Origins");
    spend_colorless(&mut s);
    let spell = now(&s, spell);
    let card = s.game().state.card_of(spell).unwrap();
    assert_eq!(top_spell(&s).spent.total(), 2);
    check_cast(&s, false);
    surveil(&mut s, "Plains");
    settle(&mut s);
    s.assert_life(0, 22);
    assert_eq!(zone_of(&s, spell), Some(ZoneType::Graveyard));

    cast_card(&mut s, spell);
    spend_colorless(&mut s);
    let flashback = top_spell(&s).object;
    assert_eq!(top_spell(&s).spent.total(), 4);
    check_cast(&s, true);
    let revealed = surveil(&mut s, "Gray Ogre");
    let definition = s.game().state[revealed].definition_id;
    settle(&mut s);
    s.assert_life(0, 24);
    let entered = s.game().state.object_of(card);
    let saga = permanent_of(&s, entered).unwrap();
    let moves: Vec<_> = s
        .game()
        .state
        .events
        .iter()
        .filter_map(|event| match event {
            GameEvent::CardMoved {
                object_id,
                departed: Some(old),
                from,
                to,
                ..
            } if object_id.slot() == flashback.slot()
                && old.generation() >= flashback.generation() =>
            {
                Some((*old, *object_id, *from, *to))
            }
            _ => None,
        })
        .collect();
    assert_eq!(moves.len(), 2);
    let exile = moves[0].1;
    assert_eq!(
        moves,
        [
            (flashback, exile, Some(ZoneType::Stack), ZoneType::Exile),
            (exile, entered, Some(ZoneType::Exile), ZoneType::Battlefield)
        ]
    );
    assert!(s.game().state.object(exile).is_err());
    assert_eq!(s.game().state.card_of(entered), Some(card));
    assert_eq!(s.game().state[entered].name, "Summon: Esper Maduin");
    assert_eq!(counters_of(&s, saga, CounterKind::Finality), 1);
    assert_eq!(counters_of(&s, saga, CounterKind::Lore), 1);
    assert_eq!(zone_of(&s, revealed), Some(ZoneType::Hand));
    let seen = Observation::for_player(s.game(), PlayerId(1));
    assert_eq!(seen.opponent.known_hand.get(&definition.0), Some(&1));
    let (slot, projected) = seen
        .opponent_permanents
        .iter()
        .enumerate()
        .find(|(_, p)| p.characteristics.name == "Summon: Esper Maduin")
        .unwrap();
    assert_eq!(projected.mana_value(), 2);
    assert_eq!(projected.characteristics.mana_cost, Default::default());
    let encoded = encode(&seen, &ObservationEncoderConfig::default()).unwrap();
    use managym::agent::observation_layout::permanent;
    assert_eq!(
        encoded.opponent_permanents[slot * permanent::WIDTH + permanent::MANA_VALUE],
        0.2
    );
    s.game_mut().transform_permanent(saga);
    assert_eq!(s.game().state[entered].name, "Summon: Esper Maduin");
    assert_eq!(
        s.game().state.permanents[saga].as_ref().unwrap().transforms,
        0
    );

    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    assert_eq!(counters_of(&s, saga, CounterKind::Lore), 2);
    settle(&mut s);
    assert_eq!(s.game().state.players[0].mana_pool.mana[4], 2);
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    settle(&mut s);
    assert_eq!(zone_of(&s, entered), Some(ZoneType::Exile));
    assert_eq!(
        s.game().state[s.game().state.object_of(card)].name,
        "Esper Origins"
    );
    assert_eq!(s.game().characteristics(ally).pt(), (4, 4));
    assert!(s
        .game()
        .characteristics(ally)
        .keywords
        .has(Keyword::Trample));
    assert_eq!(s.game().characteristics(opponent).pt(), (2, 2));
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().characteristics(ally).pt(), (2, 2));
    assert!(!s
        .game()
        .characteristics(ally)
        .keywords
        .has(Keyword::Trample));
}

#[test]
fn esper_origins_reveals_a_sorcery_and_stays_exiled_on_an_ordinary_return() {
    let mut s = five_color_setup(&["Esper Origins"]);
    // A creature-entry replacement must inspect the Saga face while it is still in exile.
    let land = s.force_permanent_on_battlefield(1, "Plains");
    let land_object = crate::support::inspect::card_of(&s, land);
    s.game_mut().state[land_object].statics.push(
        managym::state::card::StaticAbility::EntersTapped(
            managym::state::predicate::CardPredicate::creature(),
        ),
    );
    let grave = crate::support::game::plant_in_graveyard(&mut s, 0, "Esper Origins");
    cast_card(&mut s, grave);
    spend_colorless(&mut s);
    let top = surveil(&mut s, "Esper Origins");
    settle(&mut s);
    assert_eq!(library_top(&s, 0), top);
    let seen = Observation::for_player(s.game(), PlayerId(1));
    assert_eq!(
        seen.opponent.known_library.get(&0),
        Some(&s.game().state[top].definition_id.0)
    );

    let saga = now(&s, grave);
    assert!(crate::support::inspect::tapped(
        &s,
        permanent_of(&s, saga).unwrap()
    ));
    let exiled = s.game_mut().move_object(saga, ZoneType::Exile);
    assert_eq!(
        s.game_mut().move_object(exiled, ZoneType::Battlefield),
        exiled
    );
    assert_eq!(zone_of(&s, exiled), Some(ZoneType::Exile));
}
