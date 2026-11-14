//! Tests for cards filed in `src/mid.rs`, in card order.

use crate::support::inspect::zone_of;
use crate::support::{
    game::five_color_setup,
    play::{cast, settle},
};
use managym::state::zone::ZoneType;

#[test]
fn consider_surveils_one_then_draws() {
    let mut s = five_color_setup(&["Consider"]);
    cast(&mut s, "Consider");
    let hand = s.zone_size(0, ZoneType::Hand);
    let library = s.zone_size(0, ZoneType::Library);
    settle(&mut s);
    s.assert_zone_size(0, ZoneType::Hand, hand + 1);
    let milled = s.zone_size(0, ZoneType::Graveyard) - 1;
    assert!(milled <= 1);
    s.assert_zone_size(0, ZoneType::Library, library - 1 - milled);
}

#[test]
fn join_the_dance_makes_two_humans_and_flashes_back_for_two_more() {
    let mut s = five_color_setup(&["Join the Dance"]);
    let dance = cast(&mut s, "Join the Dance");
    settle(&mut s);
    let humans = s.battlefield_permanents_named(0, "Human");
    assert_eq!(humans.len(), 2);
    let view = s.game().characteristics(humans[0]);
    assert_eq!((view.power, view.toughness), (1, 1));
    assert_eq!(zone_of(&s, dance), Some(ZoneType::Graveyard));
    crate::support::play::cast_card(&mut s, dance);
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Human").len(), 4);
    assert_eq!(zone_of(&s, dance), Some(ZoneType::Exile));
}
