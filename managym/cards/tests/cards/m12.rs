//! Tests for cards filed in `src/m12.rs`, in card order.

use crate::support::{
    game::{five_color_setup, plant_in_graveyard},
    play::{cast_owned, settle},
};
use managym::state::zone::ZoneType;

/// Either player's graveyard counts, read before drawing.
#[test]
fn visions_of_beyond_draws_three_when_any_graveyard_has_twenty_cards() {
    for (planted, draws) in [(19, 1), (20, 3)] {
        let mut s = five_color_setup(&["Visions of Beyond"]);
        for index in 0..planted {
            plant_in_graveyard(&mut s, 1, ["Island", "Swamp"][index % 2]);
        }
        cast_owned(&mut s, "Visions of Beyond");
        let hand = s.zone_size(0, ZoneType::Hand);
        settle(&mut s);
        s.assert_zone_size(0, ZoneType::Hand, hand + draws);
    }
}
