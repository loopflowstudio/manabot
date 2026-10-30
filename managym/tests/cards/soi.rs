//! Tests for cards filed in `src/cardsets/soi.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    inspect::{assert_choice_contract, library_card, permanent_named},
    play::{activate_first, pass_until, play_and_settle, select_card, settle},
};
use managym::agent::action::ActionSpaceKind;

#[test]
fn fetch_land_counts_the_found_land_and_triggers_tracker_twice() {
    let mut s = five_color_setup(&["Fabled Passage", "Tireless Tracker"]);
    for _ in 0..3 {
        s.force_permanent_on_battlefield(0, "Forest");
    }
    play_and_settle(&mut s, "Tireless Tracker");
    play_and_settle(&mut s, "Fabled Passage");
    s.pass_priority();
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 1);
    let fetch = permanent_named(&s, "Fabled Passage");
    activate_first(&mut s, fetch);
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    assert_choice_contract(&s);
    let found = library_card(&s, 0, "Forest");
    select_card(&mut s, found);
    settle(&mut s);
    let id = s.game().state.card_to_permanent[found].unwrap();
    assert!(!s.game().state.permanents[id].as_ref().unwrap().tapped);
    assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 2);
}
