//! Tests for cards filed in `src/ori.rs`, in card order.

use crate::support::{game::five_color_setup, play::enter_and_settle};
use managym::state::zone::ZoneType;

#[test]
fn elemental_bond_draws_for_a_creature_with_power_three_or_more() {
    let mut s = five_color_setup(&["Elemental Bond", "Gray Ogre", "Craw Wurm"]);
    s.force_permanent_on_battlefield(0, "Elemental Bond");
    s.game_mut().scenario_refresh_priority().unwrap();
    let hand = s.zone_size(0, ZoneType::Hand);
    enter_and_settle(&mut s, "Gray Ogre");
    s.assert_zone_size(0, ZoneType::Hand, hand);
    enter_and_settle(&mut s, "Craw Wurm");
    s.assert_zone_size(0, ZoneType::Hand, hand + 1);
}
