//! Tests for cards filed in `src/m13.rs`, in card order.

use crate::support::{
    game::{clear_summoning_sickness, five_color_setup},
    play::{activate, settle},
};

/// Krenko counts every Goblin its controller controls, itself included, as
/// the ability resolves: with one other Goblin it makes two, then four.
#[test]
fn krenko_makes_a_goblin_for_each_goblin_you_control() {
    let mut s = five_color_setup(&["Krenko, Mob Boss", "Fanatical Firebrand"]);
    let krenko = s.force_permanent_on_battlefield(0, "Krenko, Mob Boss");
    s.force_permanent_on_battlefield(0, "Fanatical Firebrand");
    s.force_permanent_on_battlefield(1, "Fanatical Firebrand");
    clear_summoning_sickness(&mut s, krenko);
    activate(&mut s, krenko, 0);
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Goblin").len(), 2);
    s.game_mut().state.permanents[krenko]
        .as_mut()
        .unwrap()
        .tapped = false;
    activate(&mut s, krenko, 0);
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Goblin").len(), 6);
}
