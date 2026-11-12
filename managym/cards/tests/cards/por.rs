//! Tests for cards filed in `src/por.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    inspect::{library_card, library_top, offered_cards},
    play::{cast_owned, select_card, settle},
};

#[test]
fn cruel_tutor_puts_the_found_card_on_top_and_costs_two_life() {
    let mut s = five_color_setup(&["Cruel Tutor", "Gray Ogre"]);
    let ogre = library_card(&s, 0, "Gray Ogre");
    cast_owned(&mut s, "Cruel Tutor");
    for _ in 0..4 {
        if offered_cards(&s).contains(&ogre) {
            break;
        }
        s.pass_priority();
    }
    select_card(&mut s, ogre);
    settle(&mut s);
    assert_eq!(library_top(&s, 0), ogre);
    s.assert_life(0, 18);
}
