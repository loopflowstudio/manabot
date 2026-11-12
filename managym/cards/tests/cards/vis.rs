//! Tests for cards filed in `src/vis.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    play::{cast_owned, choose_mode, settle},
};
use managym::state::zone::ZoneType;

#[test]
fn prosperity_has_each_player_draw_x() {
    let mut s = five_color_setup(&["Prosperity"]);
    s.set_player_mana_pool(0, "2U");
    cast_owned(&mut s, "Prosperity");
    let hands = [
        s.zone_size(0, ZoneType::Hand),
        s.zone_size(1, ZoneType::Hand),
    ];
    choose_mode(&mut s, 2);
    settle(&mut s);
    s.assert_zone_size(0, ZoneType::Hand, hands[0] - 1 + 2);
    s.assert_zone_size(1, ZoneType::Hand, hands[1] + 2);
}
