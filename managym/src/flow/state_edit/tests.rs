use super::{GameStateEdit::*, PermanentDestination::*};
use crate::{
    agent::policy_history::{HistoryReference, HISTORY_LIMIT},
    flow::{
        combat::{AttackTarget, CombatState},
        game::Game,
        trigger::ExileLink,
    },
    state::{
        ability::AbilitySet,
        effect::{Affected, EffectDuration, Modification},
        game_object::{PermanentId, PlayerId},
        permanent::CounterKind,
        player::PlayerConfig,
        zone::ZoneType,
    },
};

const HERO: PlayerId = PlayerId(0);
const OPPONENT: PlayerId = PlayerId(1);

fn game() -> Game {
    let deck = [
        ("Mountain", 44),
        ("Gray Ogre", 8),
        ("Pacifism", 1),
        ("Gleaming Barrier", 1),
        ("Liliana, Dreadhorde General", 1),
    ]
    .map(|(name, count)| (name.to_owned(), count))
    .into_iter()
    .collect::<std::collections::BTreeMap<_, _>>();
    let mut game = Game::new(
        vec![
            PlayerConfig::new("hero", deck.clone()),
            PlayerConfig::new("opponent", deck),
        ],
        7,
        false,
    );
    game.keep_opening_hands();
    game
}

fn put_on_board(game: &mut Game, owner: PlayerId, name: &str) -> PermanentId {
    let card = [ZoneType::Library, ZoneType::Hand]
        .into_iter()
        .flat_map(|zone| game.state.zones.zone_objects(zone, owner).iter())
        .copied()
        .find(|&object| game.state[object].name == name)
        .expect(name);
    let arrival = game.move_object(card, ZoneType::Battlefield);
    game.lookup_current_permanent(arrival).unwrap()
}

fn observed_id(game: &Game, id: PermanentId) -> i32 {
    game.state.permanents[id].as_ref().unwrap().id.0 as i32
}

#[test]
fn quiet_graveyard_move_keeps_a_full_history_without_firing_death_triggers() {
    let mut game = game();
    for _ in 0..HISTORY_LIMIT {
        put_on_board(&mut game, HERO, "Mountain");
    }
    let barrier = put_on_board(&mut game, HERO, "Gleaming Barrier");
    game.process_game_events();
    let permanent = observed_id(&game, barrier);
    let object = game.permanent_object(barrier).unwrap();
    let card = game.state.card_of(object).unwrap();
    let context = game.observed_object_id(object).0 as i32;
    let history = game.policy_history.clone();
    assert_eq!(history.len(), HISTORY_LIMIT);
    let HistoryReference::Card {
        context_id, zone, ..
    } = history.back().unwrap().source
    else {
        panic!("expected battlefield arrival");
    };
    assert_eq!(
        (context_id, zone),
        (Some(context), Some(ZoneType::Battlefield))
    );
    let events = game.state.events.len();

    game.apply_game_state_edits(&[MovePermanent {
        permanent,
        to: Graveyard,
    }])
    .unwrap();

    let arrival = game.state.object_of(card);
    assert_ne!(arrival, object);
    assert!(game.state.live(object).is_none());
    assert!(game.permanent_object(barrier).is_none());
    assert!(game
        .state
        .zones
        .contains(arrival, ZoneType::Graveyard, HERO));
    assert_eq!(game.state.events.len(), events);
    assert_eq!(game.state.turn.history[HERO.0].creatures_died, 0);
    assert!(game.state.pending_triggers.is_empty());
    assert!(game.state.stack_objects.is_empty());
    assert_eq!(game.policy_history.len(), HISTORY_LIMIT);
    let last = HISTORY_LIMIT - 1;
    for index in 0..last {
        assert_eq!(game.policy_history[index], history[index]);
    }
    let mut expected = history[last].clone();
    if let HistoryReference::Card { context_id, .. } = &mut expected.source {
        *context_id = None;
    }
    assert_eq!(game.policy_history[last], expected);
}

#[test]
fn quiet_move_preserves_exile_cards_and_defers_an_unattached_aura() {
    let mut game = game();
    let source = put_on_board(&mut game, HERO, "Gleaming Barrier");
    let host = game.permanent_object(source).unwrap();
    let aura = put_on_board(&mut game, OPPONENT, "Pacifism");
    let aura_card = game
        .state
        .card_of(game.permanent_object(aura).unwrap())
        .unwrap();
    game.state.permanents[aura].as_mut().unwrap().attached_to = Some(host);
    let ogre = put_on_board(&mut game, OPPONENT, "Gray Ogre");
    let exiled = game.move_object(game.permanent_object(ogre).unwrap(), ZoneType::Exile);
    let ability_set = AbilitySet::Printed(game.state[host].definition_id);
    game.state.exile_links.push(ExileLink {
        source: host,
        ability_set,
        exiled_object: exiled,
        returns: true,
    });

    game.apply_game_state_edits(&[MovePermanent {
        permanent: observed_id(&game, source),
        to: Exile,
    }])
    .unwrap();

    assert!(game.state.exile_links.is_empty());
    assert!(game.state.zones.contains(exiled, ZoneType::Exile, OPPONENT));
    assert!(game.permanent_object(aura).is_some());
    game.perform_state_based_actions();
    let aura_arrival = game.state.object_of(aura_card);
    assert!(game
        .state
        .zones
        .contains(aura_arrival, ZoneType::Graveyard, OPPONENT));
}

#[test]
fn quiet_move_retires_combatants_and_defenders() {
    let mut game = game();
    let walker = put_on_board(&mut game, OPPONENT, "Liliana, Dreadhorde General");
    let defender = game.permanent_object(walker).unwrap();
    game.state.combat = Some(Box::<CombatState>::default());
    let attacking = |game: &mut Game, target| {
        game.create_token_attacking("token.soldier", HERO, target)
            .unwrap()
    };
    let unblocked = attacking(&mut game, AttackTarget::Permanent(defender));
    let blocked = attacking(&mut game, AttackTarget::Player(OPPONENT));
    let removed = attacking(&mut game, AttackTarget::Player(OPPONENT));
    let blocker = game.create_token("token.soldier", OPPONENT).unwrap();
    let removed_object = game.permanent_object(removed).unwrap();
    let blocker_object = game.permanent_object(blocker).unwrap();
    game.declare_blocker(blocker, Some(blocked)).unwrap();
    let life = game.state.players[OPPONENT.0].life;

    game.apply_game_state_edits(&[walker, removed, blocker].map(|id| MovePermanent {
        permanent: observed_id(&game, id),
        to: LibraryBottom,
    }))
    .unwrap();

    let combat = game.state.combat.as_ref().unwrap();
    assert_eq!(combat.attackers().collect::<Vec<_>>(), [unblocked, blocked]);
    assert_eq!(combat.defender(unblocked), None);
    assert!(combat.is_blocked(blocked));
    assert!(combat.attacker_to_blockers[&blocked].is_empty());
    assert!(game.state.live(removed_object).is_none());
    assert!(game.state.live(blocker_object).is_none());
    game.resolve_combat_damage();
    assert_eq!(game.state.players[OPPONENT.0].life, life);
}

#[test]
fn library_bottom_keeps_existing_knowledge_and_pins_the_card() {
    let mut game = game();
    let ogre = put_on_board(&mut game, OPPONENT, "Gray Ogre");
    let top = game.reveal_library_top(OPPONENT).unwrap();
    let card = game
        .state
        .card_of(game.permanent_object(ogre).unwrap())
        .unwrap();

    game.apply_game_state_edits(&[MovePermanent {
        permanent: observed_id(&game, ogre),
        to: LibraryBottom,
    }])
    .unwrap();
    game.shuffle_unseen_libraries(19);

    let library = game.state.zones.zone_objects(ZoneType::Library, OPPONENT);
    let arrival = game.state.object_of(card);
    assert_eq!(library.iter().next(), Some(&arrival));
    assert_eq!(library.iter().last(), Some(&top));
    let facts = &game.state.players[OPPONENT.0].known_library.public;
    assert_eq!(facts.get(&0), Some(&game.state[top].definition_id));
    assert_eq!(
        facts.get(&(library.len() - 1)),
        Some(&game.state[arrival].definition_id)
    );
    game.assert_library_knowledge();
}

#[test]
fn scalar_edits_preserve_turn_facts_and_validate_the_entire_batch() {
    let mut game = game();
    let ogre = put_on_board(&mut game, HERO, "Gray Ogre");
    let object = game.permanent_object(ogre).unwrap();
    let permanent = observed_id(&game, ogre);
    game.state.tap(ogre);
    game.state.change_counters(ogre, CounterKind::Stun, 1);
    game.state.mark_damage(ogre, 1, true);
    game.state.mark_dealt_damage(ogre);
    let history = game.policy_history.clone();
    let before = game.state.deterministic_hash();
    let life = SetLife {
        player: HERO.0,
        life: 5,
    };
    let remove = MovePermanent {
        permanent,
        to: Graveyard,
    };
    let damage = SetDamage {
        permanent,
        damage: 0,
    };

    for edits in [
        vec![life, remove, damage],
        vec![life, remove, remove],
        vec![
            life,
            SetDamage {
                permanent,
                damage: -1,
            },
        ],
        vec![
            life,
            SetTapped {
                permanent: -1,
                tapped: true,
            },
        ],
        vec![life, SetLife { player: 2, life: 5 }],
    ] {
        assert!(game.apply_game_state_edits(&edits).is_err());
        assert_eq!(game.state.deterministic_hash(), before);
    }

    game.apply_game_state_edits(&[
        life,
        damage,
        SetTapped {
            permanent,
            tapped: false,
        },
        SetCounters {
            permanent,
            counter: CounterKind::Trample,
            count: 1,
        },
        SetCounters {
            permanent,
            counter: CounterKind::PlusOnePlusOne,
            count: i32::MAX,
        },
    ])
    .unwrap();

    let row = game.state.permanents[ogre].as_ref().unwrap();
    assert_eq!(row.object, object);
    assert_eq!(row.counter_count(CounterKind::Stun), 1);
    assert_eq!(row.damage, 0);
    assert!(!row.tapped);
    assert!(!row.deathtouch_damage);
    assert!(row.dealt_damage_this_turn);
    assert!(row
        .keyword_counter_timestamps
        .contains_key(&CounterKind::Trample));
    assert_eq!(game.characteristics(ogre).power, i32::MAX);
    assert_eq!(game.state.players[HERO.0].life, 5);
    assert_eq!(game.state.turn.history[HERO.0].life_lost, 0);
    assert_eq!(game.policy_history, history);

    game.apply_continuous(
        None,
        OPPONENT,
        [Affected::Object(object)],
        Modification::PhasedOut,
        EffectDuration::UntilYourNextTurn,
    );
    assert!(game
        .apply_game_state_edits(&[remove])
        .unwrap_err()
        .0
        .contains("phased out"));
}
