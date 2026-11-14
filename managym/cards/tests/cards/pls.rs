//! Tests for cards filed in `src/pls.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    inspect::{library_card, offered_cards, zone_of},
    play::{cast_owned, select_card, settle},
};
use managym::state::zone::ZoneType;

#[test]
fn eladamri_s_call_finds_only_a_creature() {
    let mut s = five_color_setup(&["Eladamri's Call", "Gray Ogre", "Sol Ring"]);
    let ogre = library_card(&s, 0, "Gray Ogre");
    let ring = library_card(&s, 0, "Sol Ring");
    cast_owned(&mut s, "Eladamri's Call");
    for _ in 0..4 {
        if offered_cards(&s).contains(&ogre) {
            break;
        }
        s.pass_priority();
    }
    assert!(!offered_cards(&s).contains(&ring));
    select_card(&mut s, ogre);
    settle(&mut s);
    assert_eq!(zone_of(&s, ogre), Some(ZoneType::Hand));
}
