//! CR 104.4b loop fingerprint: what a repeating mandatory loop may vary
//! from cycle to cycle, and what it must not.

use std::collections::BTreeMap;

use crate::{
    flow::game::Game,
    state::{
        game_object::PublicId, permanent::CounterKind, player::PlayerConfig,
        stack_object::StackObject,
    },
};

fn game() -> Game {
    let deck = BTreeMap::from([
        ("Gray Ogre".into(), 12),
        ("Lightning Bolt".into(), 4),
        ("Mountain".into(), 24),
    ]);
    let mut game = Game::new(
        vec![
            PlayerConfig::new("p0", deck.clone()),
            PlayerConfig::new("p1", deck),
        ],
        1044,
        false,
    );
    game.keep_opening_hands();
    game
}

/// Poison is a total the watch tracks itself; other player counters are
/// used up by a chain and stay.
#[test]
fn poison_is_left_to_the_watch_and_other_player_counters_stay() {
    let mut game = game();
    let base = game.state.loop_fingerprint();
    game.state.players[1]
        .counters
        .change(CounterKind::Poison, 3);
    assert_eq!(game.state.loop_fingerprint(), base);
    game.state.players[1]
        .counters
        .change(CounterKind::Charge, 1);
    assert_ne!(game.state.loop_fingerprint(), base);
}

/// A stack object's id is its allocation order, renumbered each cycle.
#[test]
fn stack_object_ids_are_not_part_of_the_loop_state() {
    let mut game = game();
    let bolt = game
        .state
        .latest_objects()
        .find(|(_, object)| object.name == "Lightning Bolt")
        .unwrap()
        .0;
    let controller = game.state[bolt].owner;
    game.push_spell_to_stack(bolt, controller, Vec::new(), Vec::new(), Default::default());
    let base = game.state.loop_fingerprint();
    let StackObject::Spell(spell) = &mut game.state.stack_objects[0] else {
        unreachable!("a spell was pushed");
    };
    spell.id = PublicId(spell.id.0 + 1_000);
    assert_eq!(game.state.loop_fingerprint(), base);
}

/// Only the count of an ability's uses saturates; which ability it is
/// stays exact however large its identities.
#[test]
fn ability_counts_saturate_but_their_sources_stay_distinct() {
    let mut game = game();
    // The last Ogres allocated, whose identities exceed the saturation.
    let ogres: Vec<_> = game
        .state
        .latest_objects()
        .filter(|(_, object)| object.name == "Gray Ogre")
        .map(|(id, object)| (id, object.definition_id))
        .collect();
    let [.., (first, definition), (second, _)] = ogres[..] else {
        unreachable!("the deck holds Ogres");
    };
    let fingerprint_with = |game: &mut Game, object, count| {
        game.state.turn.ability_resolutions_this_turn =
            BTreeMap::from([((object, definition, 0), count)]);
        game.state.loop_fingerprint()
    };
    let many = fingerprint_with(&mut game, first, 20);
    assert_eq!(fingerprint_with(&mut game, first, 30), many);
    assert_ne!(fingerprint_with(&mut game, first, 3), many);
    assert_ne!(fingerprint_with(&mut game, second, 20), many);
}
