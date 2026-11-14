//! Tests for cards filed in `src/shm.rs`, in card order.

use crate::support::{
    game::{clear_summoning_sickness, five_color_setup},
    play::{activate, settle},
};

/// The first ability makes a green-white Elf Warrior; the second copies each
/// creature token its controller has, and only tokens.
#[test]
fn rhys_the_redeemed_makes_elf_warriors_and_copies_each_creature_token() {
    let mut s = five_color_setup(&["Rhys the Redeemed", "Gray Ogre"]);
    let rhys = s.force_permanent_on_battlefield(0, "Rhys the Redeemed");
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    clear_summoning_sickness(&mut s, rhys);
    activate(&mut s, rhys, 0);
    settle(&mut s);
    let elves = s.battlefield_permanents_named(0, "Elf Warrior");
    assert_eq!(elves.len(), 1);
    let elf = s.game().state.permanents[elves[0]].as_ref().unwrap().object;
    let colors = s.game().characteristics(elves[0]).colors.len();
    assert!(!s.game().state[elf].is_card());
    assert_eq!(colors, 2);

    s.game_mut().state.permanents[rhys].as_mut().unwrap().tapped = false;
    activate(&mut s, rhys, 1);
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Elf Warrior").len(), 2);
    assert_eq!(s.battlefield_permanents_named(0, "Gray Ogre").len(), 1);
    assert_eq!(
        s.battlefield_permanents_named(0, "Rhys the Redeemed").len(),
        1
    );
}
