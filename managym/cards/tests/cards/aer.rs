//! Tests for cards filed in `src/aer.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    inspect::keywords,
    play::{cast_owned, settle},
};
use managym::state::card::Keyword;

#[test]
fn heroic_intervention_protects_only_your_permanents() {
    let mut s = five_color_setup(&["Heroic Intervention", "Gray Ogre", "Sol Ring"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let ring = s.force_permanent_on_battlefield(0, "Sol Ring");
    let foe = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.game_mut().scenario_refresh_priority().unwrap();
    cast_owned(&mut s, "Heroic Intervention");
    settle(&mut s);
    for id in [ogre, ring] {
        assert!(keywords(&s, id).has(Keyword::Hexproof));
        assert!(keywords(&s, id).has(Keyword::Indestructible));
    }
    assert!(!keywords(&s, foe).has(Keyword::Hexproof));
}
