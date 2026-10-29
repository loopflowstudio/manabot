//! Tests for cards filed in `src/cardsets/war.rs`, grouped by the helper family they
//! drive play with (`tests/support/`).

use crate::support::elemental::{permanent, setup};
use crate::support::sacrifice::{
    activate, enter, select, settle_targeting_opponent, target_opponent,
};

#[test]
fn mayhem_keeps_granted_lifelink_after_it_is_sacrificed() {
    let mut s = setup(&["Mayhem Devil", "Weaponize the Monsters"]);
    let card = enter(&mut s, "Mayhem Devil");
    enter(&mut s, "Weaponize the Monsters");
    let devil = permanent(&s, "Mayhem Devil");
    // Model a resolved until-end-of-turn grant before the sacrifice.
    s.game_mut().state.permanents[devil]
        .as_mut()
        .unwrap()
        .temp_keywords
        .lifelink = true;
    let weapon = permanent(&s, "Weaponize the Monsters");
    activate(&mut s, weapon);
    target_opponent(&mut s);
    select(&mut s, card);
    settle_targeting_opponent(&mut s);
    assert_eq!(s.game().state.players[0].life, 21);
    assert_eq!(s.game().state.players[1].life, 17);
}
