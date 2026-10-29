//! Throne of Eldraine (`src/cardsets/eld.rs`).
use crate::support::{
    activate_any, assert_choice_contract, enter_or_cast, library_card, pass_until, permanent,
    select, settle, setup,
};
use managym::agent::action::ActionSpaceKind;

#[test]
fn fetch_lands_have_distinct_entry_and_life_rules() {
    for (name, tapped, life) in [("Fabled Passage", true, 20), ("Prismatic Vista", false, 19)] {
        let mut s = setup(&[name]);
        enter_or_cast(&mut s, name);
        let source = permanent(&s, name);
        activate_any(&mut s, source);
        pass_until(&mut s, ActionSpaceKind::LookAndSelect);
        let found = library_card(&s, 0, "Forest");
        select(&mut s, found);
        settle(&mut s);
        let id = s.game().state.card_to_permanent[found].unwrap();
        assert_eq!(
            s.game().state.permanents[id].as_ref().unwrap().tapped,
            tapped
        );
        assert_eq!(s.game().state.players[0].life, life);
        assert!(s.battlefield_permanents_named(0, name).is_empty());
    }
}

#[test]
fn fetch_land_counts_the_found_land_and_triggers_tracker_twice() {
    let mut s = setup(&["Fabled Passage", "Tireless Tracker"]);
    for _ in 0..3 {
        s.force_permanent_on_battlefield(0, "Forest");
    }
    enter_or_cast(&mut s, "Tireless Tracker");
    enter_or_cast(&mut s, "Fabled Passage");
    s.pass_priority();
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 1);
    let fetch = permanent(&s, "Fabled Passage");
    activate_any(&mut s, fetch);
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    assert_choice_contract(&s);
    let found = library_card(&s, 0, "Forest");
    select(&mut s, found);
    settle(&mut s);
    let id = s.game().state.card_to_permanent[found].unwrap();
    assert!(!s.game().state.permanents[id].as_ref().unwrap().tapped);
    assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 2);
}
