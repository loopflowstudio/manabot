//! Final Fantasy card behavior through native casting, choices and zone moves.
use crate::support::{
    game::{
        clear_summoning_sickness, five_color_setup, move_card, pin_top, plant_in_graveyard,
        set_counters, Scenario,
    },
    inspect::{
        counters_of, library_top, now, offered_cards, permanent, permanent_of, top_spell, zone_of,
    },
    play::{
        attack_with, attack_with_group, cast, cast_card, command, resolve_everything, resolve_top,
        settle, spend_colorless,
    },
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

#[test]
fn cecil_transforms_in_place_under_a_pending_bolt_and_protects_other_attackers() {
    use managym::state::game_object::Target;
    let mut s = five_color_setup(&[
        "Cecil, Dark Knight",
        "Trusty Boomerang",
        "Lightning Bolt",
        "Rocky Rebuke",
        "Gray Ogre",
        "Wind Drake",
    ]);
    let cecil = s.force_permanent_on_battlefield(0, "Cecil, Dark Knight");
    let equipment = s.force_permanent_on_battlefield(0, "Trusty Boomerang");
    let ally = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let victim = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.force_permanent_on_battlefield(0, "Wind Drake");
    let source = s.game().permanent_object(cecil).unwrap();
    s.game_mut().state.permanents[equipment]
        .as_mut()
        .unwrap()
        .attached_to = Some(source);
    s.game_mut().state.permanents[cecil]
        .as_mut()
        .unwrap()
        .tapped = true;
    set_counters(&mut s, cecil, CounterKind::PlusOnePlusOne, 1);
    s.game_mut().state.players[0].life = 13;
    cast(&mut s, "Lightning Bolt");
    assert!(s.choose_target(Target::Permanent(cecil)));
    if s.action_space().player == Some(PlayerId(1)) {
        s.pass_priority();
    }
    cast(&mut s, "Rocky Rebuke");
    assert!(s.choose_target(Target::Permanent(cecil)));
    assert!(s.choose_target(Target::Permanent(victim)));
    spend_colorless(&mut s);
    resolve_top(&mut s);
    assert_eq!(s.life(0), 13);
    let before = Observation::for_player(s.game(), PlayerId(1));
    assert_eq!(before.continuation[0].trigger_damage, 3);
    use managym::{
        agent::{observation_encoder::FactKind, observation_layout::fact},
        cards::tokens::{name_value, TokenKind},
    };
    let damage_tokens = [
        (TokenKind::Text as u16, name_value("DamageEvent::Amount")),
        (TokenKind::Integer as u16, 3),
    ];
    let program = &before.continuation[0].effect_execution;
    let rank = program
        .windows(2)
        .position(|pair| pair == damage_tokens)
        .unwrap();
    let encoded = encode(&before, &ObservationEncoderConfig::default()).unwrap();
    assert!(encoded
        .facts
        .chunks_exact(fact::WIDTH)
        .any(
            |row| row[fact::KIND] == FactKind::ContinuationProgram as u8 as f32
                && row[fact::VALUE] == TokenKind::Integer as u16 as f32
                && row[fact::VALUE_2] == (rank + 1) as f32
                && row[fact::DETAIL_HIGH + 1] == 3.0
        ));
    let front = before
        .opponent_permanents
        .iter()
        .find(|p| p.characteristics.name == "Cecil, Dark Knight")
        .unwrap();
    resolve_top(&mut s);
    assert_eq!(s.life(0), 10);
    let after = Observation::for_player(s.game(), PlayerId(1));
    let back = after
        .opponent_permanents
        .iter()
        .find(|p| p.characteristics.name == "Cecil, Redeemed Paladin")
        .unwrap();
    assert_eq!((back.id, back.object_id), (front.id, front.object_id));
    assert_eq!(s.game().permanent_object(cecil), Some(source));
    assert_eq!(permanent(&s, equipment).attached_to, Some(source));
    assert_eq!(back.plus1_counters, 1);
    assert!(!back.tapped);
    assert!(back.characteristics.keywords.lifelink);
    resolve_top(&mut s);
    assert_eq!(permanent(&s, cecil).damage, 3);
    for id in [cecil, ally] {
        clear_summoning_sickness(&mut s, id);
    }
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with_group(&mut s, &[cecil, ally]);
    settle(&mut s);
    let viewer = Observation::for_player(s.game(), PlayerId(1));
    for name in ["Gray Ogre", "Wind Drake", "Cecil, Redeemed Paladin"] {
        let row = viewer
            .opponent_permanents
            .iter()
            .find(|p| p.characteristics.name == name)
            .unwrap();
        assert_eq!(
            row.characteristics.keywords.indestructible,
            name == "Gray Ogre"
        );
    }
    s.advance_to_active_step(1, StepKind::Upkeep);
    assert!(!s
        .game()
        .characteristics(ally)
        .keywords
        .has(Keyword::Indestructible));
}

#[test]
fn cecil_uses_combat_damage_already_dealt_and_stays_a_knight_above_half_life() {
    let mut s = five_color_setup(&["Cecil, Dark Knight"]);
    let cecil = s.force_permanent_on_battlefield(0, "Cecil, Dark Knight");
    clear_summoning_sickness(&mut s, cecil);
    s.game_mut().state.players[0].life = 13;
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, cecil);
    s.advance_to_active_step(0, StepKind::CombatDamage);
    assert_eq!(s.life(1), 18);
    assert_eq!(s.life(0), 13);
    // The pending trigger remembers the hit even if Cecil's power changes.
    set_counters(&mut s, cecil, CounterKind::PlusOnePlusOne, 1);
    resolve_top(&mut s);
    assert_eq!(s.life(0), 11);
    let view = Observation::for_player(s.game(), PlayerId(1));
    let knight = view
        .opponent_permanents
        .iter()
        .find(|p| p.characteristics.name == "Cecil, Dark Knight")
        .unwrap();
    assert!(knight.tapped && knight.characteristics.keywords.deathtouch);
}

#[test]
fn cecil_trampling_over_a_lethal_blocker_leaves_one_damage_trigger() {
    let mut s = five_color_setup(&["Cecil, Dark Knight", "Craw Wurm"]);
    let cecil = s.force_permanent_on_battlefield(0, "Cecil, Dark Knight");
    s.force_permanent_on_battlefield(1, "Craw Wurm");
    clear_summoning_sickness(&mut s, cecil);
    set_counters(&mut s, cecil, CounterKind::Trample, 1);
    s.game_mut().state.players[0].life = 12;
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, cecil);
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    s.declare_block();
    s.advance_to_active_step(0, StepKind::CombatDamage);
    assert_eq!(s.life(1), 19);
    assert!(s.game().state.permanents[cecil].is_none());
    assert_eq!(s.game().state.stack_objects.len(), 1);
    resolve_top(&mut s);
    assert_eq!(s.life(0), 10);
    assert!(s
        .battlefield_permanents_named(0, "Cecil, Redeemed Paladin")
        .is_empty());
}

#[test]
fn fang_draws_and_loses_life_once_when_cards_leave_its_controllers_graveyard() {
    let mut s = five_color_setup(&["Fang, Fearless l'Cie", "Gray Ogre", "Lightning Bolt"]);
    s.force_permanent_on_battlefield(0, "Fang, Fearless l'Cie");
    let opponent = plant_in_graveyard(&mut s, 1, "Mountain");
    let first = plant_in_graveyard(&mut s, 0, "Gray Ogre");
    let second = plant_in_graveyard(&mut s, 0, "Lightning Bolt");
    let draw = library_top(&s, 0);
    move_card(&mut s, opponent, ZoneType::Exile);
    assert!(s.game().state.pending_triggers.is_empty());
    move_card(&mut s, first, ZoneType::Exile);
    move_card(&mut s, second, ZoneType::Exile);
    resolve_everything(&mut s);
    s.assert_life(0, 19);
    assert_eq!(zone_of(&s, draw), Some(ZoneType::Hand));
}

#[test]
fn fang_returning_from_the_graveyard_sees_only_later_departures() {
    let name = "Fang, Fearless l'Cie";
    let mut s = five_color_setup(&[name, "Gray Ogre"]);
    let fang = plant_in_graveyard(&mut s, 0, name);
    let ogre = plant_in_graveyard(&mut s, 0, "Gray Ogre");
    move_card(&mut s, fang, ZoneType::Battlefield);
    assert!(s.game().state.pending_triggers.is_empty());
    move_card(&mut s, ogre, ZoneType::Hand);
    resolve_everything(&mut s);
    s.assert_life(0, 19);
}
