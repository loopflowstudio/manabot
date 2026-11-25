//! Tests for cards filed in `src/cmd.rs`, in card order.

use crate::support::{
    game::{five_color_setup, plant_in_graveyard},
    inspect::{counters, zone_of},
    play::{activate, settle},
};
use managym::state::{game_object::Target, zone::ZoneType};

/// Scavenging Ooze exiles a card from any graveyard; only a creature card
/// grows it and gains its controller a life.
#[test]
fn scavenging_ooze_grows_only_for_an_exiled_creature_card() {
    let mut s = five_color_setup(&["Scavenging Ooze", "Gray Ogre"]);
    let ooze = s.force_permanent_on_battlefield(0, "Scavenging Ooze");
    let ogre = plant_in_graveyard(&mut s, 1, "Gray Ogre");
    let plains = plant_in_graveyard(&mut s, 1, "Plains");
    for (card, grown) in [(plains, 0), (ogre, 1)] {
        activate(&mut s, ooze, 0);
        assert!(s.choose_target(Target::GraveyardCard(card)));
        settle(&mut s);
        assert_eq!(zone_of(&s, card), Some(ZoneType::Exile));
        assert_eq!(counters(&s, ooze), grown);
        s.assert_life(0, 20 + grown);
    }
}
