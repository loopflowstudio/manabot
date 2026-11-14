use managym::state::{
    game_object::{Incarnation, ObjectId, ObjectLookupError, ObjectRef, PlayerId},
    zone::ZoneType,
};

use crate::support::game::*;
use crate::support::inspect::owned_card;

#[test]
fn object_ref_leave_and_reenter_preserves_entity_but_invalidates_incarnation() {
    let mut s = Scenario::new(
        deck(&[("Mountain", 24), ("Gray Ogre", 16)]),
        deck(&[("Mountain", 40)]),
        400_701,
    );
    let permanent = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let object = s.game().state.permanents[permanent]
        .as_ref()
        .expect("forced permanent")
        .object;
    let old_presentation = s.game().state.permanents[permanent]
        .as_ref()
        .expect("forced permanent")
        .id;
    let old_ref = s
        .game()
        .current_object_ref(object)
        .expect("battlefield object ref");
    assert_eq!(s.game().lookup_current_permanent(old_ref), Ok(permanent));

    s.game_mut().move_object(object, ZoneType::Hand);
    assert_eq!(
        s.game().lookup_current_permanent(old_ref),
        Err(ObjectLookupError::StaleIncarnation)
    );
    let hand_ref = s
        .game()
        .current_object_ref(object)
        .expect("hand object ref");
    assert_eq!(
        s.game().lookup_current_permanent(hand_ref),
        Err(ObjectLookupError::WrongZone)
    );
    assert_eq!(
        s.game().lookup_current_permanent(ObjectRef {
            entity: ObjectId(usize::MAX),
            incarnation: Incarnation::INITIAL,
        }),
        Err(ObjectLookupError::MissingEntity)
    );
    let lki = s
        .game()
        .object_lki(old_ref)
        .expect("departing battlefield LKI")
        .clone();
    assert_eq!(lki.object_ref, old_ref);
    assert_eq!(lki.object, object);
    assert_eq!(lki.from_zone, ZoneType::Battlefield);
    assert_eq!(lki.owner, PlayerId(0));
    assert_eq!(lki.controller, PlayerId(0));
    assert_eq!(lki.presentation_id, old_presentation);
    assert_eq!(
        s.game()
            .object_lki_definition(old_ref)
            .expect("LKI definition in shared content pack")
            .name,
        "Gray Ogre"
    );

    let returned = s.game_mut().move_object(object, ZoneType::Battlefield);
    let new_ref = s
        .game()
        .current_object_ref(object)
        .expect("re-entered object ref");
    assert_eq!(returned, new_ref);
    let new_permanent = s.game().state.object_to_permanent[object].expect("re-entered permanent");
    let new_presentation = s.game().state.permanents[new_permanent]
        .as_ref()
        .expect("re-entered permanent")
        .id;

    assert_eq!(new_ref.entity, old_ref.entity);
    assert!(new_ref.incarnation > old_ref.incarnation);
    assert_ne!(new_presentation, old_presentation);
    assert_eq!(
        s.game().lookup_current_permanent(old_ref),
        Err(ObjectLookupError::StaleIncarnation)
    );
    assert_eq!(
        s.game().lookup_current_permanent(new_ref),
        Ok(new_permanent)
    );

    let branch = s.game().clone();
    assert_eq!(branch.current_object_ref(object), Some(new_ref));
    assert_eq!(branch.object_lki(old_ref), Some(&lki));
    assert_eq!(
        branch
            .object_lki_definition(old_ref)
            .expect("cloned LKI resolves through shared content")
            .name,
        "Gray Ogre"
    );
}

#[test]
fn object_ref_same_zone_move_is_not_a_new_rules_object() {
    let mut s = Scenario::new(
        deck_of("Mountain", "Gray Ogre"),
        deck(&[("Mountain", 40)]),
        400_702,
    );
    let permanent = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let object = s.game().state.permanents[permanent]
        .as_ref()
        .expect("forced permanent")
        .object;
    let before = s.game().current_object_ref(object).expect("object ref");
    let events_before = s.game().state.events.len();

    let returned = s.game_mut().move_object(object, ZoneType::Battlefield);

    assert_eq!(returned, before);
    assert_eq!(s.game().current_object_ref(object), Some(before));
    assert_eq!(s.game().state.object_to_permanent[object], Some(permanent));
    assert_eq!(s.game().state.events.len(), events_before);
}

#[test]
fn cr_400_7_death_trigger_keeps_source_lki_across_reentry() {
    let mut s = Scenario::new(
        deck(&[
            ("Island", 12),
            ("Dragonfly Swarm", 8),
            ("It'll Quench Ya!", 20),
        ]),
        deck(&[("Mountain", 40)]),
        400_703,
    );
    let lesson = owned_card(&s, 0, "It'll Quench Ya!");
    s.game_mut().move_object(lesson, ZoneType::Graveyard);
    let swarm = s.force_permanent_on_battlefield(0, "Dragonfly Swarm");
    let swarm_card = s.game().state.permanents[swarm]
        .as_ref()
        .expect("swarm permanent")
        .object;
    let old_ref = s
        .game()
        .current_object_ref(swarm_card)
        .expect("swarm object ref");
    let old_presentation = s.game().state.permanents[swarm]
        .as_ref()
        .expect("swarm permanent")
        .id;
    let hand_before = s.zone_size(0, ZoneType::Hand);

    s.game_mut().move_object(swarm_card, ZoneType::Graveyard);
    let pending = s
        .game()
        .state
        .pending_triggers
        .first()
        .expect("Dragonfly death trigger");
    assert_eq!(pending.source_ref, Some(old_ref));
    let source_lki = pending.source_lki.clone().expect("trigger source LKI");
    assert_eq!(source_lki.object_ref, old_ref);
    assert_eq!(source_lki.controller, PlayerId(0));
    assert_eq!(source_lki.presentation_id, old_presentation);
    assert_eq!(
        source_lki.definition_id,
        s.game().state.objects[swarm_card].definition_id
    );

    s.game_mut().move_object(swarm_card, ZoneType::Battlefield);
    let new_ref = s
        .game()
        .current_object_ref(swarm_card)
        .expect("re-entered swarm ref");
    assert_eq!(new_ref.entity, old_ref.entity);
    assert_ne!(new_ref.incarnation, old_ref.incarnation);
    assert_eq!(s.game().state.pending_triggers[0].source_ref, Some(old_ref));

    for _ in 0..8 {
        if s.game().state.pending_triggers.is_empty() && s.game().state.stack_objects.is_empty() {
            break;
        }
        s.pass_priority();
    }
    assert!(s.game().state.pending_triggers.is_empty());
    assert!(s.game().state.stack_objects.is_empty());
    assert_eq!(s.zone_size(0, ZoneType::Hand), hand_before + 1);
    assert_eq!(
        s.game().state.zones.zone_of(swarm_card),
        Some(ZoneType::Battlefield)
    );
}

/// A checked card id survives moves and copy effects; token and spell-copy
/// rows never pass the card boundary, even when they have a card's definition.
#[test]
fn card_ids_narrow_kind_and_keep_numeric_wire_identity() {
    use managym::state::game_object::{ObjectId, ObjectKind, ObjectVec};

    let mut s = Scenario::new(
        deck(&[("Mountain", 24), ("Gray Ogre", 16)]),
        deck(&[("Island", 40)]),
        400_708,
    );
    let object = owned_card(&s, 0, "Gray Ogre");
    let card = s.game().state.card_id(object).expect("deck card");
    assert_eq!(ObjectId::from(card), object);
    assert_eq!(
        serde_json::to_value(card).unwrap(),
        serde_json::to_value(object).unwrap()
    );
    assert!(s.game().state.card_id(ObjectId(usize::MAX)).is_none());

    let template = s.game().state.objects[card].clone();
    let mut objects = ObjectVec(vec![
        template.clone(),
        template.clone().with_kind(ObjectKind::Token),
        template.with_kind(ObjectKind::SpellCopy),
    ]);
    let checked = objects.card_id(ObjectId(0)).unwrap();
    assert!(objects.card_id(ObjectId(1)).is_none());
    assert!(objects.card_id(ObjectId(2)).is_none());
    let token = objects[ObjectId(1)].clone();
    objects[checked].become_copy_of(&token, &[]);
    assert_eq!(objects.card_id(ObjectId(0)), Some(checked));

    for zone in [
        ZoneType::Hand,
        ZoneType::Graveyard,
        ZoneType::Exile,
        ZoneType::Library,
    ] {
        s.game_mut().move_object(card, zone);
        assert_eq!(s.game().state.card_id(object), Some(card));
        assert!(s
            .game()
            .state
            .zone_cards(zone, PlayerId(0))
            .any(|found| found == card));
    }
}

/// Ordered zones keep their physical order and RNG consumption. Card-only
/// readers exclude visitors, while observations and moves still see them.
#[test]
fn card_zone_readers_preserve_transient_token_occupants() {
    use managym::agent::observation::Observation;
    use managym::state::game_object::ObjectId;
    use rand::seq::SliceRandom;
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    for zone in [
        ZoneType::Hand,
        ZoneType::Library,
        ZoneType::Graveyard,
        ZoneType::Exile,
    ] {
        let mut s = Scenario::new(deck(&[("Mountain", 40)]), deck(&[("Island", 40)]), 400_709);
        let player = PlayerId(0);
        let permanent = s
            .game_mut()
            .create_token("token.spirit", player, false)
            .unwrap();
        let token = s.game().state.permanents[permanent]
            .as_ref()
            .unwrap()
            .object;
        s.game_mut().move_object(token, zone);
        let before = s.game().state.zones.zone_objects(zone, player).to_vec();
        assert_eq!(before.last(), Some(&token));
        let cards: Vec<_> = s.game().state.zone_cards(zone, player).collect();
        assert!(!cards.iter().any(|card| ObjectId::from(*card) == token));
        assert_eq!(cards.len() + 1, before.len());
        assert_eq!(
            s.game().state.zones.zone_objects(zone, player).to_vec(),
            before
        );
        assert!(s.game().state.card_id(token).is_none());
        if zone == ZoneType::Hand {
            let observation = Observation::for_player(s.game(), player);
            assert!(observation
                .agent_cards
                .iter()
                .any(|row| row.id == s.game().state.objects[token].id.0 as i32));
        }
        if zone == ZoneType::Library {
            let mut drawn = s.game().clone();
            let history = drawn.state.turn.history[player.0].cards_drawn;
            drawn.draw_cards(player, 1);
            assert_eq!(drawn.state.zones.zone_of(token), Some(ZoneType::Hand));
            assert_eq!(drawn.state.turn.history[player.0].cards_drawn, history + 1);
            assert_eq!(
                drawn.state.zones.size(ZoneType::Library, player),
                before.len() - 1
            );
            assert_eq!(
                drawn.state.zones.top(ZoneType::Library, player),
                before.get(before.len() - 2).copied()
            );
            assert!(drawn
                .state
                .zone_cards(ZoneType::Hand, player)
                .all(|card| ObjectId::from(card) != token));
        }
        let mut expected = before;
        let mut expected_rng = ChaCha8Rng::seed_from_u64(400_710);
        let mut actual_rng = expected_rng.clone();
        expected.shuffle(&mut expected_rng);
        s.game_mut()
            .state
            .zones
            .shuffle(zone, player, &mut actual_rng);
        assert_eq!(actual_rng.next_u64(), expected_rng.next_u64());
        assert_eq!(
            s.game().state.zones.zone_objects(zone, player).to_vec(),
            expected
        );
    }
}

/// Typed occupants retain the old ordered numeric wire representation through
/// placement, reorder, hidden-world assignment, and removal of visitors.
#[test]
fn typed_card_zones_preserve_wire_order_and_hidden_reassignment() {
    use managym::state::game_object::{ObjectId, ObjectKind, ObjectVec};
    use managym::state::zone::ZoneManager;
    use rand::{seq::SliceRandom, RngCore, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    let s = Scenario::new(deck(&[("Mountain", 40)]), deck(&[("Island", 40)]), 400_711);
    let template = s.game().state.objects[ObjectId(0)].clone();
    let objects = ObjectVec(vec![
        template.clone(),
        template.clone().with_kind(ObjectKind::Token),
        template.clone(),
        template.with_kind(ObjectKind::SpellCopy),
    ]);
    let player = PlayerId(0);
    let mut zones = ZoneManager::default();
    let mut expected = vec![ObjectId(0), ObjectId(1), ObjectId(2), ObjectId(3)];
    for object in &expected {
        zones.move_object(&objects, *object, player, ZoneType::Library);
    }
    assert_eq!(
        zones
            .zone_objects(ZoneType::Library, player)
            .cards(&objects)
            .map(ObjectId::from)
            .collect::<Vec<_>>(),
        vec![ObjectId(0), ObjectId(2)]
    );

    // The legacy serializer emitted these fields, in this order, with raw ids.
    assert_eq!(
        String::from_utf8(serde_json::to_vec(&zones).unwrap()).unwrap(),
        r#"{"library":[[0,1,2,3],[]],"hand":[[],[]],"battlefield":[[],[]],"graveyard":[[],[]],"stack":[[],[]],"exile":[[],[]],"command":[[],[]],"card_zones":["Library","Library","Library","Library"]}"#
    );

    zones.reposition(ZoneType::Library, player, ObjectId(1), 0);
    expected.remove(1);
    expected.insert(0, ObjectId(1));
    assert_eq!(
        zones.zone_objects(ZoneType::Library, player).to_vec(),
        expected
    );
    let mut actual_rng = ChaCha8Rng::seed_from_u64(400_712);
    let mut expected_rng = actual_rng.clone();
    expected.sort_unstable_by_key(|object| object.0);
    expected.shuffle(&mut expected_rng);
    zones.shuffle_canonical(ZoneType::Library, player, &mut actual_rng);
    assert_eq!(
        zones.zone_objects(ZoneType::Library, player).to_vec(),
        expected
    );
    assert_eq!(actual_rng.next_u64(), expected_rng.next_u64());

    let new_hand = vec![ObjectId(3), ObjectId(0), ObjectId(1)];
    zones.reassign_hidden(&objects, player, new_hand.clone(), vec![ObjectId(2)]);
    assert_eq!(
        zones.zone_objects(ZoneType::Hand, player).to_vec(),
        new_hand
    );
    assert_eq!(
        zones
            .zone_objects(ZoneType::Hand, player)
            .cards(&objects)
            .map(ObjectId::from)
            .collect::<Vec<_>>(),
        vec![ObjectId(0)]
    );
    for object in new_hand {
        assert_eq!(zones.zone_of(object), Some(ZoneType::Hand));
    }
    zones.remove_object(ObjectId(3));
    zones.remove_object(ObjectId(1));
    assert_eq!(
        zones.zone_objects(ZoneType::Hand, player).to_vec(),
        vec![ObjectId(0)]
    );
    assert_eq!(zones.zone_of(ObjectId(1)), None);
    assert_eq!(zones.zone_of(ObjectId(3)), None);
}

/// A physical card retains its identity and printing while its appearances
/// change; the current-object bridge rejects departed appearances.
#[test]
fn cr_400_7_physical_card_survives_appearances_and_copy_effects() {
    let mut s = Scenario::new(
        deck(&[("Mountain", 24), ("Gray Ogre", 16)]),
        deck(&[("Island", 40)]),
        400_712,
    );
    let permanent = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let object = s.game().state.permanents[permanent]
        .as_ref()
        .unwrap()
        .object;
    let appearance = s.game().current_object_ref(object).unwrap();
    let card = s.game().state.card_of(appearance).unwrap();
    let physical = s.game().state.cards[card];
    assert_eq!(physical.owner, PlayerId(0));
    assert_eq!(
        physical.definition,
        s.game().state.objects[object].definition_id
    );
    assert_eq!(physical.public_id, s.game().state.objects[object].id);
    assert_eq!(s.game().state.object_of(card), Some(appearance));
    assert_eq!(
        s.game().state.object(appearance).unwrap().id,
        physical.public_id
    );

    let other = owned_card(&s, 1, "Island");
    let template = s.game().state.objects[other].clone();
    s.game_mut().state.objects[object].become_copy_of(&template, &[]);
    assert_ne!(
        s.game().state.objects[object].definition_id,
        physical.definition
    );
    assert_eq!(s.game().state.cards[card], physical);
    assert_eq!(s.game().state.card_of(appearance), Some(card));

    s.game_mut().move_object(object, ZoneType::Exile);
    let exiled = s.game().state.object_of(card).unwrap();
    assert_ne!(exiled, appearance);
    assert_eq!(s.game().state.card_of(appearance), None);
    assert_eq!(
        s.game().state.object(appearance).err(),
        Some(ObjectLookupError::StaleIncarnation)
    );
    assert_eq!(s.game().state.card_of(exiled), Some(card));
    assert_eq!(s.game().state.cards[card], physical);
    assert_eq!(
        s.game().state.objects[object].definition_id,
        physical.definition
    );

    s.game_mut().move_object(object, ZoneType::Battlefield);
    let returned = s.game().state.object_of(card).unwrap();
    assert_ne!(returned, exiled);
    assert_eq!(s.game().state.card_of(exiled), None);
    assert_eq!(s.game().state.card_of(returned), Some(card));
    let fork = s.game().clone();
    assert_eq!(fork.state.cards[card], physical);
    assert_eq!(fork.state.object_of(card), Some(returned));
}

#[test]
fn cr_400_7_token_creation_does_not_add_a_physical_card() {
    let mut s = Scenario::new(deck(&[("Mountain", 40)]), deck(&[("Island", 40)]), 400_713);
    let roster_size = s.game().state.cards.len();
    let permanent = s
        .game_mut()
        .create_token("token.spirit", PlayerId(0), false)
        .unwrap();
    let object = s.game().state.permanents[permanent]
        .as_ref()
        .unwrap()
        .object;
    let appearance = s.game().current_object_ref(object).unwrap();
    assert_eq!(s.game().state.card_of(appearance), None);
    assert_eq!(s.game().state.cards.len(), roster_size);
    s.game_mut().move_object(object, ZoneType::Graveyard);
    assert_eq!(s.game().state.card_of(appearance), None);
    assert_eq!(
        s.game()
            .state
            .card_of(s.game().current_object_ref(object).unwrap()),
        None
    );
    assert_eq!(s.game().state.cards.len(), roster_size);
}

/// A copied view is state of one appearance even outside the battlefield.
/// Rebuilding from the retained printing also preserves fixture edits.
#[test]
fn cr_400_7_non_battlefield_move_discards_old_copy_values() {
    let mut s = Scenario::new(
        deck(&[("Mountain", 24), ("Gray Ogre", 16)]),
        deck(&[("Island", 40)]),
        400_715,
    );
    let object = owned_card(&s, 0, "Gray Ogre");
    s.game_mut().move_object(object, ZoneType::Hand);
    s.game_mut().state.objects[object].power = Some(5);
    let appearance = s.game().current_object_ref(object).unwrap();
    let card = s.game().state.card_of(appearance).unwrap();
    let template = s.game().state.objects[owned_card(&s, 1, "Island")].clone();
    s.game_mut().state.objects[object].become_copy_of(&template, &[]);
    assert_eq!(s.game().state.object(appearance).unwrap().name, "Island");

    s.game_mut().move_object(object, ZoneType::Library);
    assert_eq!(
        s.game().state.object(appearance).err(),
        Some(ObjectLookupError::StaleIncarnation)
    );
    let current = s.game().state.object_of(card).unwrap();
    let new_object = s.game().state.object(current).unwrap();
    assert_eq!(new_object.name, "Gray Ogre");
    assert_eq!(new_object.power, Some(5));
    assert!(new_object.copy.is_none());
    assert_eq!(
        new_object.definition_id,
        s.game().state.cards[card].definition
    );
}

#[test]
fn cr_400_7_checked_object_lookup_rejects_a_token_that_ceased_to_exist() {
    let mut s = Scenario::new(deck(&[("Mountain", 40)]), deck(&[("Island", 40)]), 400_716);
    let token = s
        .game_mut()
        .create_token("token.spirit", PlayerId(0), false)
        .unwrap();
    let object = s.game().state.permanents[token].as_ref().unwrap().object;
    s.game_mut().move_object(object, ZoneType::Graveyard);
    let appearance = s.game().current_object_ref(object).unwrap();
    assert!(s.game().state.object(appearance).is_ok());
    s.advance_to_active_step(0, managym::flow::turn::StepKind::End);
    assert_eq!(
        s.game().state.object(appearance).err(),
        Some(ObjectLookupError::StaleIncarnation)
    );
}

/// CR 608.3a, 111.12: a permanent spell and its copy each get a new
/// appearance on resolution; only the original has a physical card. The
/// synthetic trigger isolates copying a permanent spell from card authoring.
#[test]
fn cr_400_7_permanent_spell_and_copy_mint_distinct_appearances() {
    use crate::support::play::{cast, resolve_top, settle, spend_colorless};
    use managym::state::{
        ability::{Ability, Actor, Effect, TriggerCondition},
        game_object::ObjectKind,
        predicate::CardPredicate,
        stack_object::StackObject,
    };

    let mut s = five_color_setup(&["Gray Ogre"]);
    let source = s.force_permanent_on_battlefield(0, "Mountain");
    let source_object = s.game().state.permanents[source].as_ref().unwrap().object;
    s.game_mut().state.objects[source_object]
        .abilities
        .push(Ability::Triggered {
            condition: TriggerCondition::casts(Actor::You, CardPredicate::default()),
            intervening_if: None,
            effects: vec![Effect::CopyTriggeringObject],
        });
    let roster_size = s.game().state.cards.len();
    let original = cast(&mut s, "Gray Ogre");
    spend_colorless(&mut s);
    // A synthetic announced X isolates provenance from any one X spell.
    for stack in &mut s.game_mut().state.stack_objects {
        if let StackObject::Spell(spell) = stack {
            if spell.object == original {
                spell.x = 7;
            }
        }
    }
    let spell_appearance = s.game().current_object_ref(original).unwrap();
    let card = s.game().state.card_of(spell_appearance).unwrap();
    let spent = s
        .game()
        .state
        .stack_objects
        .iter()
        .find_map(|stack| match stack {
            StackObject::Spell(spell) if spell.object == original => Some(spell.spent),
            _ => None,
        })
        .unwrap();
    resolve_top(&mut s);
    let copied = s
        .game()
        .state
        .stack_objects
        .iter()
        .find_map(|stack| match stack {
            StackObject::Spell(spell)
                if s.game().state.objects[spell.object].kind() == ObjectKind::SpellCopy =>
            {
                Some(spell.object)
            }
            _ => None,
        })
        .expect("copy on the stack");
    let copy_appearance = s.game().current_object_ref(copied).unwrap();
    assert_eq!(s.game().state.card_of(copy_appearance), None);
    resolve_top(&mut s);
    let token_appearance = s.game().current_object_ref(copied).unwrap();
    assert_ne!(token_appearance, copy_appearance);
    assert_eq!(
        s.game().state.object(copy_appearance).err(),
        Some(ObjectLookupError::StaleIncarnation)
    );
    assert_eq!(
        s.game().state.object(token_appearance).unwrap().kind(),
        ObjectKind::Token
    );
    assert_eq!(s.game().state.card_of(token_appearance), None);
    assert_eq!(s.game().state.cards.len(), roster_size);

    settle(&mut s);
    let permanent_appearance = s.game().state.object_of(card).unwrap();
    assert_ne!(permanent_appearance, spell_appearance);
    assert_eq!(
        s.game().state.object(spell_appearance).err(),
        Some(ObjectLookupError::StaleIncarnation)
    );
    let permanent = s
        .game()
        .lookup_current_permanent(permanent_appearance)
        .unwrap();
    let entered = s.game().state.permanents[permanent].as_ref().unwrap();
    assert!(entered.was_cast);
    assert_eq!(entered.cast_x, 7);
    assert_eq!(entered.cast_spent, spent);

    s.game_mut().move_object(original, ZoneType::Graveyard);
    s.game_mut().move_object(original, ZoneType::Battlefield);
    let returned = s
        .game()
        .lookup_current_permanent(s.game().state.object_of(card).unwrap())
        .unwrap();
    let returned = s.game().state.permanents[returned].as_ref().unwrap();
    assert!(!returned.was_cast);
    assert_eq!(returned.cast_x, 0);
    assert!(returned.cast_spent.is_empty());
}

#[test]
fn cr_400_7_outside_card_has_no_appearance_until_it_enters_the_game() {
    let mut s = Scenario::with_sideboards(
        deck(&[("Mountain", 40)]),
        deck(&[("Island", 40)]),
        deck(&[("Gray Ogre", 2)]),
        deck(&[("Forest", 1)]),
        400_714,
    );
    assert_eq!(s.game().state.cards.len(), 83);
    let card = s.game().state.players[0].sideboard[0];
    let physical = s.game().state.cards[card];
    assert_eq!(physical.owner, PlayerId(0));
    assert_eq!(s.game().state.object_of(card), None);
    assert_eq!(
        s.game().state.cards[s.game().state.players[1].sideboard[0]].owner,
        PlayerId(1)
    );
    s.game_mut().move_object(card, ZoneType::Hand);
    let appearance = s.game().state.object_of(card).unwrap();
    assert_eq!(s.game().state.card_of(appearance), Some(card));
    assert_eq!(s.game().state.cards[card], physical);
    assert_eq!(s.game().state.cards.len(), 83);
}
