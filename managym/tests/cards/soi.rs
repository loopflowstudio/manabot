//! Tests for cards filed in `src/cardsets/soi.rs`, grouped by the helper family they
//! drive play with (`tests/support/`).

use crate::support::elemental::{card, pass_until, permanent, settle, setup};
use crate::support::sacrifice::{activate, enter, select};
use managym::agent::action::ActionSpaceKind;

#[test]
fn fetch_land_counts_the_found_land_and_triggers_tracker_twice() {
    let mut s = setup(&["Fabled Passage", "Tireless Tracker"]);
    for _ in 0..3 {
        s.force_permanent_on_battlefield(0, "Forest");
    }
    enter(&mut s, "Tireless Tracker");
    enter(&mut s, "Fabled Passage");
    s.pass_priority();
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 1);
    let fetch = permanent(&s, "Fabled Passage");
    activate(&mut s, fetch);
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    crate::support::elemental::assert_choice_contract(&s);
    let found = card(&s, 0, "Forest");
    select(&mut s, found);
    settle(&mut s);
    let id = s.game().state.card_to_permanent[found].unwrap();
    assert!(!s.game().state.permanents[id].as_ref().unwrap().tapped);
    assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 2);
}
