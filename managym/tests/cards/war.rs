//! Tests for cards filed in `src/cardsets/war.rs`, in card order.

use crate::support::{
    game::{five_color_setup, until_end_of_turn},
    inspect::permanent_named,
    play::{
        activate_first, play_and_settle, select_card, settle_targeting_opponent, target_opponent,
    },
};
use managym::state::{card::Keywords, effect::Modification};

#[test]
fn mayhem_keeps_granted_lifelink_after_it_is_sacrificed() {
    let mut s = five_color_setup(&["Mayhem Devil", "Weaponize the Monsters"]);
    let card = play_and_settle(&mut s, "Mayhem Devil");
    play_and_settle(&mut s, "Weaponize the Monsters");
    let devil = permanent_named(&s, "Mayhem Devil");
    // Model a resolved until-end-of-turn grant before the sacrifice.
    until_end_of_turn(
        &mut s,
        devil,
        Modification::AddKeywords(Keywords {
            lifelink: true,
            ..Default::default()
        }),
    );
    let weapon = permanent_named(&s, "Weaponize the Monsters");
    activate_first(&mut s, weapon);
    target_opponent(&mut s);
    select_card(&mut s, card);
    settle_targeting_opponent(&mut s);
    assert_eq!(s.game().state.players[0].life, 21);
    assert_eq!(s.game().state.players[1].life, 17);
}
