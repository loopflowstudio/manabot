//! Tests for cards filed in `src/mid.rs`, in card order.

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
