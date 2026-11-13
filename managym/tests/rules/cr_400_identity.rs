use managym::state::{
    game_object::{EntityId, Incarnation, ObjectLookupError, ObjectRef, PlayerId},
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
            entity: EntityId(usize::MAX),
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

    s.game_mut().move_object(object, ZoneType::Battlefield);
    let new_ref = s
        .game()
        .current_object_ref(object)
        .expect("re-entered object ref");
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

    s.game_mut().move_object(object, ZoneType::Battlefield);

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
    objects[ObjectId(2)].resolve_into_token();
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
        s.game_mut().move_object(object, zone);
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
