//! Tests for cards filed in `src/kld.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    inspect::tapped,
    play::{cast_owned, settle},
};

#[test]
fn dramatic_reversal_untaps_your_nonland_permanents_only() {
    let mut s = five_color_setup(&["Dramatic Reversal", "Gray Ogre", "Sol Ring"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let ring = s.force_permanent_on_battlefield(0, "Sol Ring");
    let island = s.force_permanent_on_battlefield(0, "Island");
    let foe = s.force_permanent_on_battlefield(1, "Gray Ogre");
    for id in [ogre, ring, island, foe] {
        s.game_mut().state.permanents[id].as_mut().unwrap().tapped = true;
    }
    s.game_mut().scenario_refresh_priority().unwrap();
    cast_owned(&mut s, "Dramatic Reversal");
    settle(&mut s);
    assert!(!tapped(&s, ogre));
    assert!(!tapped(&s, ring));
    assert!(tapped(&s, island));
    assert!(tapped(&s, foe));
}
