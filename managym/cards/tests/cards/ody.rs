//! Tests for cards filed in `src/ody.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    inspect::{library_card, offered_cards, zone_of},
    play::{cast_owned, select_card, settle},
};
use managym::state::zone::ZoneType;

#[test]
fn entomb_puts_any_chosen_library_card_into_the_graveyard() {
    let mut s = five_color_setup(&["Entomb", "Gray Ogre"]);
    let ogre = library_card(&s, 0, "Gray Ogre");
    cast_owned(&mut s, "Entomb");
    for _ in 0..4 {
        if offered_cards(&s).contains(&ogre) {
            break;
        }
        s.pass_priority();
    }
    assert!(offered_cards(&s).contains(&ogre));
    let library = s.zone_size(0, ZoneType::Library);
    select_card(&mut s, ogre);
    settle(&mut s);
    assert_eq!(zone_of(&s, ogre), Some(ZoneType::Graveyard));
    s.assert_zone_size(0, ZoneType::Library, library - 1);
}
