use std::collections::BTreeMap;

use managym::{
    agent::action::{ActionSpaceKind, ActionType},
    flow::{
        event::GameEvent,
        trigger::{DelayedTrigger, DelayedTriggerKind},
        turn::StepKind,
    },
    state::{
        ability::{Ability, Effect, TriggerCondition, TriggerSubject},
        game_object::{CardId, ObjectRef, PlayerId},
        predicate::CardPredicate,
        stack_object::StackObject,
        zone::ZoneType,
    },
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

fn dragonfly_sba_scenario(seed: u64) -> (Scenario, ObjectRef) {
    let deck = BTreeMap::from([
        ("Dragonfly Swarm".to_string(), 1),
        ("Firebending Lesson".to_string(), 1),
        ("Island".to_string(), 6),
    ]);
    let mut s = Scenario::new(deck, mountain_deck(), seed);
    s.advance_to_active_step(0, StepKind::Main);

    let lesson = card_named(&s, 0, "Firebending Lesson");
    s.game_mut().move_card(lesson, ZoneType::Graveyard);
    let swarm = s.force_permanent_on_battlefield(0, "Dragonfly Swarm");
    let swarm_ref = s
        .game()
        .permanent_object_ref(swarm)
        .expect("Dragonfly object ref");

    // The death trigger's draw will attempt to draw from an empty library.
    let library = s
        .game()
        .state
        .zones
        .zone_cards(ZoneType::Library, PlayerId(0))
        .to_vec();
    for card in library {
        s.game_mut().state.zones.remove_card(card, PlayerId(0));
    }
    s.game_mut().state.permanents[swarm]
        .as_mut()
        .expect("Dragonfly permanent")
        .damage = 3;

    (s, swarm_ref)
}

/// CR 704.5a — A player with 0 or less life loses the game as a state-based action.
#[test]
fn cr_704_5a_player_loses_at_zero_life() {
    let mut s = Scenario::new(mountain_deck(), mountain_deck(), 121);

    s.advance_to_active_step(0, StepKind::Main);
    s.game_mut().state.players[0].life = 0;

    s.pass_priority();

    s.assert_game_over();
    s.assert_winner(1);
}

/// CR 704.5b — A player who attempted to draw from an empty library loses.
#[test]
fn cr_704_5b_empty_library_draw_loses_game() {
    let mut s = Scenario::new(empty_deck(), mountain_deck(), 122);

    s.advance_until(
        |s| s.game().is_game_over(),
        "game should end from empty library draw".to_string(),
    );

    s.assert_game_over();
    s.assert_winner(1);
}

/// CR 704.5g — Creature with lethal damage is destroyed.
#[test]
fn cr_704_5g_lethal_damage_destroys_creature() {
    let mut s = Scenario::new(forest_elves_deck(), mountain_deck(), 123);

    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Forest");
    s.force_card_in_hand(0, "Llanowar Elves");
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));
    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
    s.pass_priority();
    s.pass_priority();

    let elf = s
        .battlefield_permanents_named(0, "Llanowar Elves")
        .into_iter()
        .next()
        .expect("elf should be on battlefield");
    s.game_mut().state.permanents[elf]
        .as_mut()
        .expect("permanent should exist")
        .damage = 1;

    s.pass_priority();
    s.pass_priority();

    assert!(s
        .battlefield_permanents_named(0, "Llanowar Elves")
        .is_empty());
    assert!(s.zone_size(0, ZoneType::Graveyard) >= 1);
}

/// CR 704.3, 603.3, 117.5 — an SBA death is committed, its trigger is
/// collected and stacked, and the trigger's later empty-library draw causes a
/// fresh SBA before priority can be granted.
#[test]
fn cr_704_3_sba_trigger_sba_chain_reaches_fixpoint_before_priority() {
    let (mut s, swarm_ref) = dragonfly_sba_scenario(704_301);

    s.pass_priority();

    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
    assert_eq!(s.game().state.stack_objects.len(), 1);
    let StackObject::TriggeredAbility(triggered) = &s.game().state.stack_objects[0] else {
        panic!("Dragonfly death trigger should be on the stack");
    };
    assert_eq!(triggered.source_ref, Some(swarm_ref));
    assert!(s.game().state.pending_events.is_empty());
    assert!(s.game().state.pending_triggers.is_empty());
    assert!(s.game().state.priority.sba_done);

    s.pass_priority();
    s.pass_priority();

    s.assert_game_over();
    s.assert_winner(1);
    assert!(s.game().state.players[0].drew_when_empty);
    assert!(s.game().state.pending_events.is_empty());
}

/// CR 704.3, 704.5a — all applicable loss SBAs are applied in the same check;
/// if both players lose, the engine reports a draw rather than choosing by
/// player iteration order.
#[test]
fn cr_704_3_simultaneous_terminal_conditions_are_a_draw() {
    let mut s = Scenario::new(mountain_deck(), mountain_deck(), 704_302);
    s.advance_to_active_step(0, StepKind::Main);
    s.game_mut().state.players[0].life = 0;
    s.game_mut().state.players[1].life = 0;

    s.pass_priority();

    s.assert_game_over();
    assert_eq!(s.game().winner_index(), None);
    assert!(!s.game().state.players[0].alive);
    assert!(!s.game().state.players[1].alive);
    assert_eq!(s.action_space().kind, ActionSpaceKind::GameOver);
}

/// CR 704.3, 704.5f, 704.5g — independently applicable SBAs are one batch.
/// The external committed-event stream is serialized in exact-object order so
/// equivalent replays do not depend on storage traversal accidents.
#[test]
fn cr_704_3_multiple_simultaneous_sbas_commit_in_object_order() {
    let mut s = Scenario::new(ogre_deck(), mountain_deck(), 704_303);
    s.advance_to_active_step(0, StepKind::Main);
    let land = s.force_permanent_on_battlefield(0, "Mountain");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");

    s.game_mut().state.permanents[land]
        .as_mut()
        .expect("animated land")
        .animated = true;
    s.game_mut().state.permanents[ogre]
        .as_mut()
        .expect("damaged ogre")
        .damage = 2;

    let mut expected = [
        s.game().permanent_object_ref(land).expect("land ref"),
        s.game().permanent_object_ref(ogre).expect("ogre ref"),
    ];
    expected.sort();
    let _ = s.game_mut().drain_events();

    s.pass_priority();

    let committed: Vec<_> = s
        .game()
        .state
        .events
        .iter()
        .filter_map(|event| match event {
            GameEvent::CardMoved {
                card,
                from: Some(ZoneType::Battlefield),
                to: ZoneType::Graveyard,
                ..
            } => Some(*card),
            _ => None,
        })
        .collect();
    let expected_cards: Vec<_> = expected
        .iter()
        .map(|object_ref| CardId::from(object_ref.entity))
        .collect();
    assert_eq!(committed, expected_cards);
    assert!(expected
        .iter()
        .all(|object_ref| s.game().object_lki(*object_ref).is_some()));
    assert!(s.game().state.pending_events.is_empty());
}

/// CR 603.6c, 704.3 — objects leaving simultaneously use the same pre-SBA
/// trigger-source snapshot. Each creature therefore sees the other creature
/// die even though committed facts have a deterministic serialization order.
#[test]
fn cr_704_3_simultaneous_departures_share_trigger_source_snapshot() {
    let mut s = Scenario::new(ogre_deck(), mountain_deck(), 704_306);
    s.advance_to_active_step(0, StepKind::Main);
    let first = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let second = s.force_permanent_on_battlefield(0, "Gray Ogre");
    assert_ne!(first, second);

    let ability = Ability::Triggered {
        condition: TriggerCondition::Dies {
            subject: TriggerSubject::AnotherYouControl(CardPredicate::creature()),
        },
        effects: vec![Effect::GainLife { amount: 1 }],
    };
    let mut expected_sources = Vec::new();
    for permanent_id in [first, second] {
        let object_ref = s
            .game()
            .permanent_object_ref(permanent_id)
            .expect("ogre ref");
        expected_sources.push(object_ref);
        let card = s.game().state.permanents[permanent_id]
            .as_ref()
            .expect("ogre")
            .card;
        s.game_mut().state.cards[card].abilities = vec![ability.clone()];
        s.game_mut().state.permanents[permanent_id]
            .as_mut()
            .expect("ogre")
            .damage = 2;
    }

    s.pass_priority();

    let mut stacked_sources: Vec<_> = s
        .game()
        .state
        .stack_objects
        .iter()
        .filter_map(|object| match object {
            StackObject::TriggeredAbility(triggered) => triggered.source_ref,
            _ => None,
        })
        .collect();
    expected_sources.sort();
    stacked_sources.sort();
    assert_eq!(stacked_sources, expected_sources);
    assert!(s.game().state.pending_triggers.is_empty());
    assert!(s.game().state.pending_events.is_empty());
}

/// CR 400.7, 603.7 — a delayed trigger watches one exact object. A stale
/// watcher must not fire for a later incarnation of the same physical card.
#[test]
fn cr_400_7_stale_delayed_trigger_reference_is_rejected() {
    let mut s = Scenario::new(ogre_deck(), mountain_deck(), 704_304);
    s.advance_to_active_step(0, StepKind::Main);
    let old_permanent = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let card = s.game().state.permanents[old_permanent]
        .as_ref()
        .expect("old ogre")
        .card;
    let old_ref = s
        .game()
        .permanent_object_ref(old_permanent)
        .expect("old ref");

    s.game_mut().move_card(card, ZoneType::Hand);
    s.game_mut().move_card(card, ZoneType::Battlefield);
    let new_permanent = s.game().state.card_to_permanent[card].expect("new permanent");
    let new_ref = s
        .game()
        .permanent_object_ref(new_permanent)
        .expect("new ref");
    assert_ne!(old_ref, new_ref);

    s.game_mut().state.delayed_triggers.push(DelayedTrigger {
        watched: old_ref,
        controller: PlayerId(0),
        kind: DelayedTriggerKind::ReturnToBattlefieldTapped,
    });
    s.game_mut().state.permanents[new_permanent]
        .as_mut()
        .expect("new ogre")
        .damage = 2;

    s.pass_priority();

    assert_eq!(
        s.game().state.zones.zone_of(card),
        Some(ZoneType::Graveyard)
    );
    assert!(s.game().state.delayed_triggers.is_empty());
    assert!(s.game().state.pending_triggers.is_empty());
    assert!(s.game().state.stack_objects.is_empty());
}

/// CR 117.5, 704.3 — independently constructed games with the same seed and
/// commands cross the stabilization boundary with identical authoritative
/// hashes, event order, trigger order, and terminal result.
#[test]
fn cr_117_5_equivalent_seeded_stabilization_replay_is_identical() {
    let (mut left, _) = dragonfly_sba_scenario(704_305);
    let (mut right, _) = dragonfly_sba_scenario(704_305);

    assert_eq!(
        left.game().state.deterministic_hash(),
        right.game().state.deterministic_hash()
    );
    for step in 0..3 {
        left.pass_priority();
        right.pass_priority();
        assert_eq!(
            left.game().state.deterministic_hash(),
            right.game().state.deterministic_hash(),
            "stabilization replay diverged after command {step}"
        );
        assert_eq!(left.game().state.events, right.game().state.events);
        assert_eq!(left.action_space(), right.action_space());
    }
    left.assert_game_over();
    right.assert_game_over();
    assert_eq!(left.game().winner_index(), right.game().winner_index());
}
