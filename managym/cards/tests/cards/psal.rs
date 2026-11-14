//! Tests for cards filed in `src/psal.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    inspect::{offered_cards, zone_of},
    play::{cast, choose, pass_until, settle, spend_colorless},
};
use managym::{
    agent::action::{Action, ActionSpaceKind},
    state::zone::ZoneType,
};

/// Only Auras are offered, never a second card with a chosen name; the
/// third choice ends the search and all three go to hand.
#[test]
fn three_dreams_finds_up_to_three_differently_named_auras() {
    let auras = ["Imprisoned in the Moon", "Honest Work", "Avatar Destiny"];
    let mut names = vec!["Three Dreams"];
    names.extend(auras);
    let mut s = five_color_setup(&names);
    cast(&mut s, "Three Dreams");
    spend_colorless(&mut s);
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    let mut chosen = Vec::new();
    for name in auras {
        let offered = offered_cards(&s);
        assert!(offered.iter().all(|object| {
            let object = &s.game().state[*object];
            object.has_subtype("Aura") && !chosen.iter().any(|(_, n)| *n == object.name)
        }));
        let object = *offered
            .iter()
            .find(|object| s.game().state[**object].name == name)
            .unwrap();
        choose(
            &mut s,
            |action| matches!(action, Action::SelectCard { object: c, .. } if *c == object),
        );
        chosen.push((object, name.to_string()));
    }
    settle(&mut s);
    for (object, _) in &chosen {
        assert_eq!(zone_of(&s, *object), Some(ZoneType::Hand));
    }
}
