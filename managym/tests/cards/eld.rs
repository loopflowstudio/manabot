//! Tests for cards filed in `src/cardsets/eld.rs`, grouped by the helper family they
//! drive play with (`tests/support/`).

use crate::support::elemental::{card, pass_until, permanent, settle, setup};
use crate::support::sacrifice::{activate, enter, select};
use managym::agent::action::ActionSpaceKind;

#[test]
fn fetch_lands_have_distinct_entry_and_life_rules() {
    for (name, tapped, life) in [("Fabled Passage", true, 20), ("Prismatic Vista", false, 19)] {
        let mut s = setup(&[name]);
        enter(&mut s, name);
        let source = permanent(&s, name);
        activate(&mut s, source);
        pass_until(&mut s, ActionSpaceKind::LookAndSelect);
        let found = card(&s, 0, "Forest");
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
