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

/// The entering creature deals the damage, so its lifelink applies.
#[test]
fn warstorm_surge_has_the_entering_creature_deal_its_power() {
    use crate::support::play::{cast, settle_targeting_opponent};
    let mut s = five_color_setup(&["Warstorm Surge", "Aardvark Sloth"]);
    s.force_permanent_on_battlefield(0, "Warstorm Surge");
    cast(&mut s, "Aardvark Sloth");
    settle_targeting_opponent(&mut s);
    s.assert_life(1, 17);
    s.assert_life(0, 23);
}
