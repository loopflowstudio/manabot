//! Final Fantasy card behavior through native casting, choices and zone moves.
use crate::support::{
    game::{
        clear_summoning_sickness, five_color_setup, move_card, move_to_hand, pin_top,
        plant_in_graveyard, set_counters, Scenario,
    },
    inspect::{
        counters_of, library, library_top, now, offered_cards, permanent, permanent_of, top_spell,
        zone_of,
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
        game_object::{CardId, ObjectId, PermanentId, PlayerId},
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

#[test]
fn fang_token_copy_leaves_the_library_order_intact() {
    use crate::support::play::target_opponent;
    use managym::agent::action::ActionSpaceKind;

    let mut s = five_color_setup(&[FANG, "Clone Legion", "Lost Days"]);
    let original = s.force_permanent_on_battlefield(1, FANG);
    cast(&mut s, "Clone Legion");
    target_opponent(&mut s);
    spend_colorless(&mut s);
    resolve_top(&mut s);
    let copies = s.battlefield_permanents_named(0, FANG);
    assert_eq!(copies.len(), 1);
    let token = copies[0];
    assert!(!s.game().state[permanent(&s, token).object].is_card());

    let spell = move_to_hand(&mut s, "Lost Days");
    let before = library(&s, 0);
    bottom_with_lost_days(&mut s, spell, token);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
    assert_eq!(library(&s, 0), before);
    assert!(s.game().permanent_object(token).is_none());
    assert!(s.game().permanent_object(original).is_some());
    assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 1);
}

const VANILLE: &str = "Vanille, Cheerful l'Cie";
const FANG: &str = "Fang, Fearless l'Cie";
const RAGNAROK: &str = "Ragnarok, Divine Deliverance";

fn reach_meld_payment(s: &mut Scenario) {
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    s.set_player_mana_pool(0, "20WWWWUUUUBBBBRRRRGGGG");
    resolve_top(s);
}

fn vanille_first_main(cards: &[&str]) -> Scenario {
    let mut s = five_color_setup(&[&[VANILLE, FANG][..], cards].concat());
    s.force_permanent_on_battlefield(0, VANILLE);
    s.force_permanent_on_battlefield(0, FANG);
    reach_meld_payment(&mut s);
    s
}

fn pay_for_meld(s: &mut Scenario) {
    command(s, |a| matches!(a, Action::PayCost { .. }));
    spend_colorless(s);
    if !offered_cards(s).is_empty() {
        command(s, |a| matches!(a, Action::SelectCard { .. }));
    }
}

fn form_ragnarok(cards: &[&str]) -> Scenario {
    let mut s = vanille_first_main(cards);
    pay_for_meld(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, RAGNAROK).len(), 1);
    s
}

#[test]
fn vanille_exiles_a_namesake_but_only_melds_the_original_cards() {
    use managym::state::target::Target;
    let mut s = five_color_setup(&[VANILLE, FANG, "Clone"]);
    let vanille = s.force_permanent_on_battlefield(0, VANILLE);
    let vanille = s
        .game()
        .state
        .card_of(s.game().permanent_object(vanille).unwrap())
        .unwrap();
    let fang = s.force_permanent_on_battlefield(1, FANG);
    let clone = cast(&mut s, "Clone");
    spend_colorless(&mut s);
    resolve_top(&mut s);
    assert!(s.choose_target(Target::Permanent(fang)));
    settle(&mut s);
    reach_meld_payment(&mut s);
    pay_for_meld(&mut s);
    assert!(s.battlefield_permanents_named(0, RAGNAROK).is_empty());
    let state = &s.game().state;
    for card in [vanille, state.card_id(clone).unwrap()] {
        assert_eq!(
            state.zones.zone_of(state.object_of(card)),
            Some(ZoneType::Exile)
        );
    }
}

#[test]
fn vanille_can_decline_meld() {
    let mut s = vanille_first_main(&[]);
    command(&mut s, |a| matches!(a, Action::Decline { .. }));
    for name in [VANILLE, FANG] {
        assert_eq!(s.battlefield_permanents_named(0, name).len(), 1);
    }
}

fn bottom_with_lost_days(s: &mut Scenario, spell: ObjectId, target: PermanentId) {
    use crate::support::play::pass_until;
    use managym::{agent::action::ActionSpaceKind, state::target::Target};
    cast_card(s, spell);
    assert!(s.choose_target(Target::Permanent(target)));
    spend_colorless(s);
    pass_until(s, ActionSpaceKind::Modal);
    command(s, |a| matches!(a, Action::ChooseMode { mode: 1, .. }));
}

fn ragnarok_parts(s: &Scenario) -> (PermanentId, ObjectId, [CardId; 2]) {
    let permanent = s.battlefield_permanents_named(0, RAGNAROK)[0];
    let object = s.game().permanent_object(permanent).unwrap();
    let state = &s.game().state;
    let cards = [
        state.card_of(object).unwrap(),
        state.zones.melded_card(object).unwrap(),
    ];
    (permanent, object, cards)
}

fn restored_meld_orders(
    s: &Scenario,
) -> impl Iterator<Item = (usize, managym::Env, managym::decision::Command)> {
    let snapshot = s.snapshot();
    let frame = s.game().semantic_decision_frame().unwrap();
    assert_eq!(frame.offers.len(), 2);
    (0..2).map(move |first| {
        let mut env = snapshot.restore(Default::default()).unwrap();
        assert_eq!(
            env.fixture_game_mut()
                .semantic_decision_frame()
                .unwrap()
                .offers,
            frame.offers
        );
        let cmd = managym::decision::Command {
            command_id: format!("order-{first}"),
            expected_revision: frame.revision,
            offer_id: frame.offers[first].id.0,
            answers: vec![],
            object_preconditions: vec![],
        };
        (first, env, cmd)
    })
}

#[test]
fn ragnarok_departure_waits_for_its_owner_before_creating_the_clue() {
    let mut s = form_ragnarok(&["Lost Days"]);
    let owner = PlayerId(0);
    let (permanent, object, cards) = ragnarok_parts(&s);
    let observed = Observation::for_player(s.game(), PlayerId(1));
    let ragnarok = observed
        .opponent_permanents
        .iter()
        .find(|p| p.characteristics.name == RAGNAROK)
        .unwrap();
    assert_eq!(ragnarok.mana_value(), 7);
    let state = &s.game().state;
    assert!(cards.iter().all(|card| state.object_of(*card) == object));
    // Leave a single unrelated library card after positioning Lost Days in hand.
    let spell = move_to_hand(&mut s, "Lost Days");
    let library_cards = library(&s, 0);
    let unrelated = s.game().state.card_of(library_cards[0]).unwrap();
    for &card in &library_cards[1..] {
        s.game_mut().move_object(card, ZoneType::Exile);
    }
    bottom_with_lost_days(&mut s, spell, permanent);
    assert_eq!(s.game().permanent_object(permanent), Some(object));
    assert_eq!(library(&s, 0), vec![library_cards[0]]);
    assert!(s.battlefield_permanents_named(0, "Clue").is_empty());
    assert_eq!(s.action_space().player, Some(owner));
    for (first, mut env, cmd) in restored_meld_orders(&s) {
        use managym::search_state::witness;
        let fork = env.fixture_game_mut();
        *s.game_mut() = fork.clone();
        let before = witness(s.game());
        fork.execute_semantic_command(&cmd).unwrap();
        assert_eq!(witness(s.game()), before);
        s.game_mut().execute_semantic_command(&cmd).unwrap();
        assert_eq!(witness(s.game()), witness(fork));
        let history = s.game().state.events.iter().cloned().collect::<Vec<_>>();
        let ordered = library(&s, 0);
        assert!(s.game_mut().execute_semantic_command(&cmd).is_err());
        assert!(s.game().state.events.iter().eq(&history));
        assert_eq!(library(&s, 0), ordered);
        assert_eq!(witness(s.game()), witness(fork));
        assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 1);
        for expected in [unrelated, cards[first], cards[1 - first]] {
            assert_eq!(s.game().state.card_of(library_top(&s, 0)), Some(expected));
            s.game_mut().draw_cards(owner, 1);
        }
        assert!(library(&s, 0).is_empty());
    }
}

#[test]
fn ragnarok_exile_timestamps_are_chosen_by_the_exiling_opponent() {
    use crate::support::game::force_new_card_in_hand;
    use managym::state::target::Target;

    let mut s = form_ragnarok(&["Epic Downfall"]);
    let (permanent, object, cards) = ragnarok_parts(&s);
    s.advance_to_active_step(1, StepKind::Main);
    s.set_player_mana_pool(1, "1B");
    let spell = force_new_card_in_hand(&mut s, 1, "Epic Downfall");
    cast_card(&mut s, spell);
    assert!(s.choose_target(Target::Permanent(permanent)));
    spend_colorless(&mut s);
    resolve_top(&mut s);
    assert_eq!(s.action_space().player, Some(PlayerId(1)));
    for (first, mut env, cmd) in restored_meld_orders(&s) {
        *s.game_mut() = env.fixture_game_mut().clone();
        assert_eq!(s.game().permanent_object(permanent), Some(object));
        let exile = |s: &Scenario| {
            s.game()
                .state
                .zones
                .zone_objects(ZoneType::Exile, PlayerId(0))
                .to_vec()
        };
        let mut expected = exile(&s);
        s.game_mut().execute_semantic_command(&cmd).unwrap();
        let state = &s.game().state;
        assert_eq!(s.game().permanent_object(permanent), None);
        for index in [first, 1 - first] {
            let arrival = state.object_of(cards[index]);
            assert_ne!(arrival, object);
            assert_eq!(state.card_of(arrival), Some(cards[index]));
            expected.push(arrival);
        }
        assert_eq!(exile(&s), expected);
        let history = state.events.iter().cloned().collect::<Vec<_>>();
        assert!(s.game_mut().execute_semantic_command(&cmd).is_err());
        assert_eq!(exile(&s), expected);
        assert!(s.game().state.events.iter().eq(&history));
    }
}

#[test]
fn ragnarok_airbend_orders_both_cast_grants_before_the_lesson_draw() {
    use crate::support::{game::force_new_card_in_hand, play::castable};
    use managym::state::{mana::ManaCost, target::Target};

    let mut s = form_ragnarok(&["Airbending Lesson"]);
    let (permanent, object, cards) = ragnarok_parts(&s);
    let owner = PlayerId(0);
    let caster = PlayerId(1);
    let hand = |s: &Scenario| {
        s.game()
            .state
            .zones
            .zone_objects(ZoneType::Hand, caster)
            .len()
    };
    s.advance_to_active_step(1, StepKind::Main);
    s.set_player_mana_pool(1, "2W");
    let lesson = force_new_card_in_hand(&mut s, 1, "Airbending Lesson");
    cast_card(&mut s, lesson);
    assert!(s.choose_target(Target::Permanent(permanent)));
    spend_colorless(&mut s);
    let before = hand(&s);
    let top = s.game().state.card_of(library_top(&s, 1)).unwrap();
    resolve_top(&mut s);
    assert_eq!(s.action_space().player, Some(caster));
    for (first, mut env, cmd) in restored_meld_orders(&s) {
        *s.game_mut() = env.fixture_game_mut().clone();
        assert_eq!(s.game().permanent_object(permanent), Some(object));
        assert_eq!(hand(&s), before);
        assert!(s.game().state.cast_permissions.exile_grants.is_empty());
        s.game_mut().execute_semantic_command(&cmd).unwrap();
        let state = &s.game().state;
        let arrivals = [first, 1 - first].map(|i| state.object_of(cards[i]));
        assert_eq!(
            state.zones.zone_objects(ZoneType::Exile, owner).to_vec(),
            arrivals
        );
        assert!(state
            .zones
            .contains(state.object_of(top), ZoneType::Hand, caster));
        assert_eq!(hand(&s), before + 1);
        let grants = state.cast_permissions.clone();
        assert_eq!(grants.exile_grants.len(), 2);
        for arrival in arrivals {
            assert!(grants.exile_grants.iter().any(|g| g.object == arrival
                && g.player == owner
                && g.cost == Some(ManaCost::parse("{2}"))));
        }
        let history = state.events.iter().cloned().collect::<Vec<_>>();
        assert!(s.game_mut().execute_semantic_command(&cmd).is_err());
        assert!(s.game().state.events.iter().eq(&history));
        assert_eq!(s.game().state.cast_permissions, grants);
        assert_eq!(hand(&s), before + 1);

        s.advance_to_active_step(0, StepKind::Main);
        s.set_player_mana_pool(0, "4");
        assert!(castable(&mut s, arrivals[1]));
        cast_card(&mut s, arrivals[0]);
        spend_colorless(&mut s);
        assert_eq!(s.game().state.players[0].mana_pool.total(), 2);
        resolve_everything(&mut s);
        cast_card(&mut s, arrivals[1]);
        spend_colorless(&mut s);
        assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
        for (arrival, zone) in arrivals
            .into_iter()
            .zip([ZoneType::Battlefield, ZoneType::Stack])
        {
            assert_eq!(zone_of(&s, arrival), Some(zone));
        }
    }
}

#[test]
fn ragnarok_token_copy_departs_without_ordering_or_moving_either_card() {
    use managym::{agent::action::ActionSpaceKind, state::target::Target};
    let mut s = form_ragnarok(&["Lost Days", "Ember Island Production"]);
    let original = s.battlefield_permanents_named(0, RAGNAROK)[0];
    cast(&mut s, "Ember Island Production");
    command(&mut s, |a| matches!(a, Action::ChooseMode { mode: 0, .. }));
    assert!(s.choose_target(Target::Permanent(original)));
    settle(&mut s);
    let token = s
        .battlefield_permanents_named(0, RAGNAROK)
        .into_iter()
        .find(|id| *id != original)
        .unwrap();
    let spell = move_to_hand(&mut s, "Lost Days");
    let before = library(&s, 0);
    let observed = Observation::for_player(s.game(), PlayerId(1));
    let copied = observed
        .opponent_permanents
        .iter()
        .find(|p| p.is_token && p.characteristics.name == RAGNAROK)
        .unwrap();
    assert_eq!(copied.mana_value(), 0);
    bottom_with_lost_days(&mut s, spell, token);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
    assert_eq!(library(&s, 0), before);
    assert_eq!(s.battlefield_permanents_named(0, RAGNAROK), vec![original]);
    assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 1);
}

#[test]
fn ragnarok_lethal_damage_orders_two_arrivals_and_resolves_one_death_trigger() {
    use managym::{agent::observation::EventType, state::target::Target};

    let mut s = form_ragnarok(&["Blasphemous Act", "Gray Ogre"]);
    let (permanent, object, cards) = ragnarok_parts(&s);
    let state = &s.game().state;
    let entity = state.permanents[permanent].as_ref().unwrap().id;
    let land = s.force_permanent_on_battlefield(1, "Island");
    let island = s.game().permanent_object(land).unwrap();
    let ogre = plant_in_graveyard(&mut s, 0, "Gray Ogre");
    cast(&mut s, "Blasphemous Act");
    spend_colorless(&mut s);
    resolve_top(&mut s);
    assert_eq!(s.action_space().player, Some(PlayerId(0)));
    for (first, mut env, cmd) in restored_meld_orders(&s) {
        let game = env.fixture_game_mut();
        assert_eq!(game.permanent_object(permanent), Some(object));
        let mut expected = game
            .state
            .zones
            .zone_objects(ZoneType::Graveyard, PlayerId(0))
            .to_vec();
        let (_, obs, ..) = env.step_semantic_command(&cmd).unwrap();
        *s.game_mut() = env.fixture_game_mut().clone();
        let state = &s.game().state;
        // The graveyard stores its top card last.
        for index in [1 - first, first] {
            let arrival = state.object_of(cards[index]);
            assert_ne!(arrival, object);
            assert_eq!(state.card_of(arrival), Some(cards[index]));
            expected.push(arrival);
        }
        assert_eq!(
            state
                .zones
                .zone_objects(ZoneType::Graveyard, PlayerId(0))
                .to_vec(),
            expected
        );
        let deaths: Vec<_> = obs
            .recent_events
            .iter()
            .filter(|event| event.event_type == EventType::PermanentsDied as i32)
            .map(|event| (event.source_id, event.source_incarnation, event.amount))
            .collect();
        assert_eq!(deaths, [(entity.0 as i32, object.generation() as i32, 1)]);
        assert!(s.choose_target(Target::Permanent(land)));
        assert!(s.choose_target(Target::GraveyardCard(ogre)));
        assert_eq!(s.game().state.stack_objects.len(), 1);
        resolve_top(&mut s);
        assert_eq!(zone_of(&s, island), Some(ZoneType::Graveyard));
        assert!(permanent_of(&s, ogre).is_some());
        assert!(s.game().state.stack_objects.is_empty());
    }
}

/// Day of Black Sun at X=0 reads each battlefield form's mana value: a
/// manifested Gray Ogre is a face-down 2/2 with mana value 0 (CR 708.2);
/// transformed Cecil keeps its front face's 3 (CR 712.8e) and Ragnarok its
/// two cards' total (CR 712.8g).
#[test]
fn day_of_black_sun_at_zero_reads_manifest_transform_and_meld_mana_values() {
    use crate::support::{inspect::library_card, play::choose_mode};
    let mut s = form_ragnarok(&["Day of Black Sun", "Cecil, Dark Knight", "Gray Ogre"]);
    settle(&mut s);
    let ragnarok = s.battlefield_permanents_named(0, RAGNAROK)[0];
    let cecil = s.force_permanent_on_battlefield(0, "Cecil, Dark Knight");
    s.game_mut().transform_permanent(cecil);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let hidden = library_card(&s, 1, "Gray Ogre");
    let manifest = s.game_mut().manifest(hidden, PlayerId(1));
    let manifest = permanent_of(&s, manifest).unwrap();

    s.set_player_mana_pool(0, "BB");
    cast(&mut s, "Day of Black Sun");
    choose_mode(&mut s, 0);
    settle(&mut s);

    let state = &s.game().state;
    assert!(state.permanents[manifest].is_none(), "mana value 0");
    for (survivor, why) in [
        (ogre, "face-up Gray Ogre"),
        (cecil, "Cecil's front face"),
        (ragnarok, "both melded cards"),
    ] {
        assert!(state.permanents[survivor].is_some(), "{why}");
        assert!(!s.game().characteristics(survivor).abilities_lost, "{why}");
    }
    assert_eq!(
        s.game().characteristics(cecil).name,
        "Cecil, Redeemed Paladin"
    );
}

/// The battlefield object of `permanent`.
fn shown(s: &Scenario, permanent: PermanentId) -> managym::state::game_object::Object {
    s.game().state[s.game().permanent_object(permanent).unwrap()].clone()
}

fn mana_value_seen(s: &Scenario, permanent: PermanentId) -> u8 {
    let id = s.game().state.permanents[permanent].as_ref().unwrap().id.0 as i32;
    let view = Observation::for_player(s.game(), PlayerId(1));
    view.agent_permanents
        .iter()
        .chain(&view.opponent_permanents)
        .find(|row| row.id == id)
        .unwrap()
        .mana_value()
}

#[test]
fn clone_of_either_cecil_face_is_single_faced() {
    use managym::state::game_object::Target;
    let mut s = five_color_setup(&["Cecil, Dark Knight", "Clone"]);
    // The opponent's Cecil keeps the legend rule out of the way.
    let cecil = s.force_permanent_on_battlefield(1, "Cecil, Dark Knight");
    let clone_cecil = |s: &mut Scenario| {
        s.set_player_mana_pool(0, "3U");
        let clone = cast(s, "Clone");
        spend_colorless(s);
        resolve_top(s);
        assert!(s.choose_target(Target::Permanent(cecil)));
        settle(s);
        permanent_of(s, clone).expect("entered")
    };
    // CR 712.9: a single-faced card copying a front face can't transform.
    let front = clone_cecil(&mut s);
    assert_eq!(shown(&s, front).name, "Cecil, Dark Knight");
    s.game_mut().transform_permanent(front);
    assert_eq!(shown(&s, front).name, "Cecil, Dark Knight");
    assert_eq!(permanent(&s, front).transforms, 0);
    assert_eq!(mana_value_seen(&s, front), 1);

    s.game_mut().transform_permanent(cecil);
    assert_eq!(shown(&s, cecil).name, "Cecil, Redeemed Paladin");
    assert_eq!(mana_value_seen(&s, cecil), 1);
    // Copying the back face copies no front face: mana value 0 (CR 712.8e)
    // and nothing to transform back to.
    let back = clone_cecil(&mut s);
    assert_eq!(shown(&s, back).name, "Cecil, Redeemed Paladin");
    assert!(!shown(&s, back).is_transformed());
    assert_eq!(mana_value_seen(&s, back), 0);
    s.game_mut().transform_permanent(back);
    assert_eq!(shown(&s, back).name, "Cecil, Redeemed Paladin");
    assert_eq!(permanent(&s, back).transforms, 0);
}

/// A transformed token copy of Cecil shows the Paladin while its copy effect
/// names the front face. Under closed decklists, with the copied Cecil back
/// in its owner's hand, an opponent whose own deck has no Cecil still gets
/// both definitions in its capture.
#[test]
fn a_transformed_cecil_token_copy_captures_its_copied_definition() {
    use crate::support::game::deck;
    use managym::{
        cards::content_pack::default_content_pack,
        state::{game_object::Target, player::DecklistDisclosure},
    };
    let mut s = Scenario::new(
        deck(&[
            ("Island", 32),
            ("Cecil, Dark Knight", 4),
            ("Ember Island Production", 4),
        ]),
        deck(&[("Plains", 40)]),
        161,
    );
    s.advance_to_active_step(0, StepKind::Main);
    s.game_mut().scenario_clear_hand(PlayerId(0));
    s.game_mut().state.disclosure = DecklistDisclosure::Closed;
    let cecil = s.force_permanent_on_battlefield(0, "Cecil, Dark Knight");
    // Mode 0 makes a nonlegendary Hero copy, which survives the legend rule.
    s.set_player_mana_pool(0, "3UU");
    cast(&mut s, "Ember Island Production");
    command(&mut s, |a| matches!(a, Action::ChooseMode { mode: 0, .. }));
    assert!(s.choose_target(Target::Permanent(cecil)));
    spend_colorless(&mut s);
    settle(&mut s);
    let token = s
        .battlefield_permanents_named(0, "Cecil, Dark Knight")
        .into_iter()
        .find(|id| *id != cecil)
        .expect("token copy");
    let original = permanent(&s, cecil).object;
    move_card(&mut s, original, ZoneType::Hand);
    s.game_mut().transform_permanent(token);
    assert_eq!(shown(&s, token).name, "Cecil, Redeemed Paladin");
    s.pass_priority();
    let viewer = s.action_space().player.unwrap();
    assert_eq!(viewer, PlayerId(1));
    let capture = s.game().capture_decision(viewer, 0).unwrap();
    let content = default_content_pack();
    for name in ["Cecil, Dark Knight", "Cecil, Redeemed Paladin"] {
        let id = content.definition_id(name).unwrap();
        assert!(capture.definitions.contains_key(&id.0), "{name}");
    }
}

#[test]
fn copied_cecil_tokens_transform_with_their_exceptions_once() {
    use managym::{cards::content_pack::default_content_pack, state::game_object::Target};
    let mut s = five_color_setup(&[
        "Cecil, Dark Knight",
        "Ember Island Production",
        "Electroduplicate",
    ]);
    let cecil = s.force_permanent_on_battlefield(0, "Cecil, Dark Knight");
    let tokens = |s: &Scenario, besides: &[PermanentId]| {
        s.battlefield_permanents_named(0, "Cecil, Dark Knight")
            .into_iter()
            .find(|id| !besides.contains(id))
            .expect("token copy")
    };
    s.set_player_mana_pool(0, "3UU");
    cast(&mut s, "Ember Island Production");
    command(&mut s, |a| matches!(a, Action::ChooseMode { mode: 0, .. }));
    assert!(s.choose_target(Target::Permanent(cecil)));
    spend_colorless(&mut s);
    settle(&mut s);
    let hero = tokens(&s, &[cecil]);
    // A copy of the nonlegendary Hero survives the legend rule and adds
    // haste and its end-step sacrifice after the Hero's exceptions.
    s.set_player_mana_pool(0, "2R");
    cast(&mut s, "Electroduplicate");
    assert!(s.choose_target(Target::Permanent(hero)));
    spend_colorless(&mut s);
    settle(&mut s);
    let token = tokens(&s, &[cecil, hero]);
    let content = default_content_pack();
    let printed = |name: &str| {
        let id = content.definition_id(name).unwrap();
        content.definition(id).unwrap().printed.abilities.len()
    };
    let front_id = shown(&s, cecil).definition_id;
    let check = |s: &Scenario, name: &str| {
        let object = shown(s, token);
        assert_eq!(object.name, name);
        assert_eq!(object.abilities.len(), printed(name) + 1);
        assert_eq!(object.subtypes.iter().filter(|t| *t == "Hero").count(), 1);
        assert!(!object.is_legendary());
        let characteristics = s.game().characteristics(token);
        assert_eq!(characteristics.pt(), (4, 4));
        assert!(characteristics.keywords.has(Keyword::Haste));
        let copy = object.copy().expect("copy effect");
        assert_eq!((copy.of, copy.exceptions.len()), (front_id, 5));
        assert_eq!(mana_value_seen(s, token), 1);
    };
    check(&s, "Cecil, Dark Knight");
    // CR 707.8a: a copy of a transforming permanent is a transforming token,
    // and each face shows the raw face with every exception applied once.
    for name in [
        "Cecil, Redeemed Paladin",
        "Cecil, Dark Knight",
        "Cecil, Redeemed Paladin",
    ] {
        s.game_mut().transform_permanent(token);
        check(&s, name);
    }
    assert_eq!(permanent(&s, token).transforms, 3);
}

/// Azula's opponent exiles Ragnarok, so they order its two cards (CR 713.8)
/// while Azula's controller receives a cast grant for each arrival before
/// the graveyard choice. Ragnarok stays assembled and ungranted while the
/// order is pending, and the spent order Command is rejected.
#[test]
fn azula_ragnarok_exile_is_ordered_by_the_exiling_owner_and_grants_both_cards() {
    use crate::support::game::force_new_card_in_hand;
    use managym::{
        agent::action::ActionSpaceKind,
        search_state::witness,
        state::{mana::ManaCost, target::Target},
    };

    const AZULA: &str = "Azula, Cunning Usurper";
    let mut s = form_ragnarok(&[AZULA, "Lost Days"]);
    let (permanent, object, cards) = ragnarok_parts(&s);
    let (owner, caster) = (PlayerId(0), PlayerId(1));
    let spell = plant_in_graveyard(&mut s, 0, "Lost Days");
    s.advance_to_active_step(1, StepKind::Main);
    s.set_player_mana_pool(1, "2UBB");
    let azula = force_new_card_in_hand(&mut s, 1, AZULA);
    cast_card(&mut s, azula);
    spend_colorless(&mut s);
    for _ in 0..10 {
        match s.action_space().kind {
            ActionSpaceKind::LookAndSelect => break,
            ActionSpaceKind::ChooseTarget => {
                assert!(s.choose_target(Target::Player(owner)));
            }
            _ => s.pass_priority(),
        }
    }
    assert_eq!(s.action_space().player, Some(owner));
    command(
        &mut s,
        |a| matches!(a, Action::SelectCard { object: o, .. } if *o == object),
    );
    assert_eq!(s.action_space().player, Some(owner));
    for (first, mut env, cmd) in restored_meld_orders(&s) {
        *s.game_mut() = env.fixture_game_mut().clone();
        assert_eq!(s.game().permanent_object(permanent), Some(object));
        assert!(s.game().state.cast_permissions.exile_grants.is_empty());
        s.game_mut().execute_semantic_command(&cmd).unwrap();
        let state = &s.game().state;
        let arrivals = [first, 1 - first].map(|i| state.object_of(cards[i]));
        assert_eq!(
            state.zones.zone_objects(ZoneType::Exile, owner).to_vec(),
            arrivals
        );
        // Both grants precede the graveyard choice, which the owner makes.
        assert_eq!(s.action_space().player, Some(owner));
        assert!(offered_cards(&s).contains(&now(&s, spell)));
        let grants = state.cast_permissions.clone();
        assert_eq!(grants.exile_grants.len(), 2);
        for arrival in arrivals {
            let grant = grants
                .exile_grants
                .iter()
                .find(|g| g.object == arrival)
                .expect("granted arrival");
            assert_eq!(grant.player, caster);
            assert!(grant.flash && grant.any_mana);
            assert_eq!(grant.cost, None::<ManaCost>);
        }
        // The grant names Azula by the viewer id her permanent row carries.
        let azula_id = grants
            .exile_grants
            .iter()
            .find_map(|grant| grant.source)
            .and_then(|source| s.game().lookup_current_permanent(source).ok())
            .map(|id| s.game().state.permanents[id].as_ref().unwrap().id.0 as i32)
            .expect("Azula on the battlefield");
        let view = Observation::for_player(s.game(), caster);
        assert!(view.agent_permanents.iter().any(|row| row.id == azula_id));
        let sources: Vec<_> = view
            .agent_cards
            .iter()
            .chain(&view.opponent_cards)
            .filter_map(|card| card.cast_permission.as_ref())
            .map(|permission| permission.source_permanent_id)
            .collect();
        assert_eq!(sources, [Some(azula_id); 2]);
        let before = witness(s.game());
        assert!(s.game_mut().execute_semantic_command(&cmd).is_err());
        assert_eq!(witness(s.game()), before);
        assert_eq!(s.game().state.cast_permissions, grants);
    }
}
