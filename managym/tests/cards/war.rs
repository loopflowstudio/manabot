//! War of the Spark (`src/cardsets/war.rs`).
use crate::support::{
    activate_any, enter_or_cast, permanent, select, settle_targeting_opponent, setup,
    target_opponent,
};

#[test]
fn mayhem_keeps_granted_lifelink_after_it_is_sacrificed() {
    let mut s = setup(&["Mayhem Devil", "Weaponize the Monsters"]);
    let card = enter_or_cast(&mut s, "Mayhem Devil");
    enter_or_cast(&mut s, "Weaponize the Monsters");
    let devil = permanent(&s, "Mayhem Devil");
    // Model a resolved until-end-of-turn grant before the sacrifice.
    s.game_mut().state.permanents[devil]
        .as_mut()
        .unwrap()
        .temp_keywords
        .lifelink = true;
    let weapon = permanent(&s, "Weaponize the Monsters");
    activate_any(&mut s, weapon);
    target_opponent(&mut s);
    select(&mut s, card);
    settle_targeting_opponent(&mut s);
    assert_eq!(s.game().state.players[0].life, 21);
    assert_eq!(s.game().state.players[1].life, 17);
}
