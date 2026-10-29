//! Tests for cards filed in `src/cardsets/eld.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    inspect::{library_card, permanent_named},
    play::{activate_first, pass_until, play_and_settle, select_card, settle},
};
use managym::agent::action::ActionSpaceKind;

#[test]
fn fetch_lands_have_distinct_entry_and_life_rules() {
    for (name, tapped, life) in [("Fabled Passage", true, 20), ("Prismatic Vista", false, 19)] {
        let mut s = five_color_setup(&[name]);
        play_and_settle(&mut s, name);
        let source = permanent_named(&s, name);
        activate_first(&mut s, source);
        pass_until(&mut s, ActionSpaceKind::LookAndSelect);
        let found = library_card(&s, 0, "Forest");
        select_card(&mut s, found);
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
