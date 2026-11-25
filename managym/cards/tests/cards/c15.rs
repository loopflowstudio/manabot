//! Tests for cards filed in `src/c15.rs`, in card order.

use crate::support::{
    game::{five_color_setup, pin_top},
    inspect::tapped,
    play::{cast, choose_mode, settle},
};
use managym::state::{game_object::PlayerId, zone::ZoneType};

/// Dread Summons has each player mill X cards as one move, and its caster
/// creates a tapped Zombie for each creature card among them, whoever's
/// graveyard it went to.
#[test]
fn dread_summons_makes_a_zombie_for_each_creature_card_milled() {
    let mut s = five_color_setup(&["Dread Summons", "Gray Ogre"]);
    pin_top(&mut s, &["Gray Ogre", "Plains", "Gray Ogre"]);
    let opponent_creatures = s
        .game()
        .state
        .zones
        .zone_objects(ZoneType::Library, PlayerId(1))
        .iter()
        .rev()
        .take(3)
        .filter(|object| s.game().state[**object].name == "Gray Ogre")
        .count();
    cast(&mut s, "Dread Summons");
    choose_mode(&mut s, 3);
    settle(&mut s);
    for player in [0, 1] {
        assert_eq!(
            s.game()
                .state
                .zones
                .size(ZoneType::Graveyard, PlayerId(player)),
            3 + usize::from(player == 0),
            "player {player} milled three; the spell is in its owner's graveyard"
        );
    }
    let zombies = s.battlefield_permanents_named(0, "Zombie");
    assert_eq!(zombies.len(), 2 + opponent_creatures);
    assert!(zombies.iter().all(|zombie| tapped(&s, *zombie)));
    assert!(s.battlefield_permanents_named(1, "Zombie").is_empty());
}
