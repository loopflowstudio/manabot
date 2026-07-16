use std::{collections::BTreeMap, sync::Arc};

use managym::state::{
    game_object::{CardId, EntityId, Incarnation, ObjectLookupError, ObjectRef, PlayerId},
    zone::ZoneType,
};

use super::helpers::*;

fn card_named(s: &Scenario, player: usize, name: &str) -> CardId {
    let index = s
        .game()
        .state
        .cards
        .iter()
        .position(|card| card.owner == PlayerId(player) && card.name == name)
        .unwrap_or_else(|| panic!("missing {name} for player {player}"));
    CardId(index)
}

fn deck(entries: &[(&str, usize)]) -> BTreeMap<String, usize> {
    entries
        .iter()
        .map(|(name, count)| ((*name).to_string(), *count))
        .collect()
}

#[test]
fn object_ref_leave_and_reenter_preserves_entity_but_invalidates_incarnation() {
    let mut s = Scenario::new(ogre_deck(), mountain_deck(), 400_701);
    let permanent = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let card = s.game().state.permanents[permanent]
        .as_ref()
        .expect("forced permanent")
        .card;
    let old_presentation = s.game().state.permanents[permanent]
        .as_ref()
        .expect("forced permanent")
        .id;
    let old_ref = s
        .game()
        .current_object_ref(card)
        .expect("battlefield object ref");
    assert_eq!(s.game().lookup_current_permanent(old_ref), Ok(permanent));

    s.game_mut().move_card(card, ZoneType::Hand);
    assert_eq!(
        s.game().lookup_current_permanent(old_ref),
        Err(ObjectLookupError::StaleIncarnation)
    );
    let hand_ref = s.game().current_object_ref(card).expect("hand object ref");
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
    let lki = *s
        .game()
        .object_lki(old_ref)
        .expect("departing battlefield LKI");
    assert_eq!(lki.object_ref, old_ref);
    assert_eq!(lki.card, card);
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

    s.game_mut().move_card(card, ZoneType::Battlefield);
    let new_ref = s
        .game()
        .current_object_ref(card)
        .expect("re-entered object ref");
    let new_permanent = s.game().state.card_to_permanent[card].expect("re-entered permanent");
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
    assert!(Arc::ptr_eq(&s.game().state.content, &branch.state.content));
    assert_eq!(branch.current_object_ref(card), Some(new_ref));
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
    let mut s = Scenario::new(ogre_deck(), mountain_deck(), 400_702);
    let permanent = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let card = s.game().state.permanents[permanent]
        .as_ref()
        .expect("forced permanent")
        .card;
    let before = s.game().current_object_ref(card).expect("object ref");
    let events_before = s.game().state.events.len();

    s.game_mut().move_card(card, ZoneType::Battlefield);

    assert_eq!(s.game().current_object_ref(card), Some(before));
    assert_eq!(s.game().state.card_to_permanent[card], Some(permanent));
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
        mountain_deck(),
        400_703,
    );
    let lesson = card_named(&s, 0, "It'll Quench Ya!");
    s.game_mut().move_card(lesson, ZoneType::Graveyard);
    let swarm = s.force_permanent_on_battlefield(0, "Dragonfly Swarm");
    let swarm_card = s.game().state.permanents[swarm]
        .as_ref()
        .expect("swarm permanent")
        .card;
    let old_ref = s
        .game()
        .current_object_ref(swarm_card)
        .expect("swarm object ref");
    let old_presentation = s.game().state.permanents[swarm]
        .as_ref()
        .expect("swarm permanent")
        .id;
    let hand_before = s.zone_size(0, ZoneType::Hand);

    s.game_mut().move_card(swarm_card, ZoneType::Graveyard);
    let pending = s
        .game()
        .state
        .pending_triggers
        .first()
        .expect("Dragonfly death trigger");
    assert_eq!(pending.source_ref, Some(old_ref));
    let source_lki = pending.source_lki.expect("trigger source LKI");
    assert_eq!(source_lki.object_ref, old_ref);
    assert_eq!(source_lki.controller, PlayerId(0));
    assert_eq!(source_lki.presentation_id, old_presentation);
    assert_eq!(
        source_lki.definition_id,
        s.game().state.cards[swarm_card].definition_id
    );

    s.game_mut().move_card(swarm_card, ZoneType::Battlefield);
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
