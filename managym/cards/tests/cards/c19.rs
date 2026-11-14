//! Tests for cards filed in `src/c19.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    play::{cast, settle},
};

#[test]
fn dockside_extortionist_makes_a_treasure_per_opponent_artifact_and_enchantment() {
    let mut s = five_color_setup(&[
        "Dockside Extortionist",
        "Meteorite",
        "Beastmaster Ascension",
    ]);
    s.force_permanent_on_battlefield(1, "Meteorite");
    s.force_permanent_on_battlefield(1, "Meteorite");
    s.force_permanent_on_battlefield(1, "Beastmaster Ascension");
    // Your own artifacts do not count.
    s.force_permanent_on_battlefield(0, "Meteorite");
    cast(&mut s, "Dockside Extortionist");
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Treasure").len(), 3);
    assert!(s.battlefield_permanents_named(1, "Treasure").is_empty());
}
