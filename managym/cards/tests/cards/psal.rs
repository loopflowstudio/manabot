//! Tests for cards filed in `src/psal.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    inspect::{offered_cards, zone_of},
    play::{cast, choose, pass_until, settle},
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
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    let mut chosen = Vec::new();
    for name in auras {
        let offered = offered_cards(&s);
        assert!(offered.iter().all(|card| {
            let card = &s.game().state.cards[*card];
            card.has_subtype("Aura") && !chosen.iter().any(|(_, n)| *n == card.name)
        }));
        let card = *offered
            .iter()
            .find(|card| s.game().state.cards[**card].name == name)
            .unwrap();
        choose(
            &mut s,
            |action| matches!(action, Action::SelectCard { card: c, .. } if *c == card),
        );
        chosen.push((card, name.to_string()));
    }
    settle(&mut s);
    for (card, _) in &chosen {
        assert_eq!(zone_of(&s, *card), Some(ZoneType::Hand));
    }
}
