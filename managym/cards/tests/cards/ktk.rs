//! Tests for cards filed in `src/ktk.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    play::{cast, settle, spend_colorless, spend_named},
};
use managym::agent::action::ActionSpaceKind;

#[test]
fn prowess_resolves_above_the_noncreature_spell() {
    let mut s = five_color_setup(&["Monastery Swiftspear", "Abandon Attachments"]);
    let swift = s.force_permanent_on_battlefield(0, "Monastery Swiftspear");
    cast(&mut s, "Abandon Attachments");
    spend_named(&mut s, |label| label == "Spend {U} (expires at step end)");
    spend_colorless(&mut s);
    s.pass_priority();
    s.pass_priority();
    assert_eq!(s.game().characteristics(swift).power, 2);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
    assert_eq!(s.game().state.stack_objects.len(), 1);
    settle(&mut s);
}
