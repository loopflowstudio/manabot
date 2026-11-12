//! Tests for cards filed in `src/mh1.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    play::{activate, choose_mode, settle},
};
use managym::state::{mana::Mana, zone::ZoneType};

#[test]
fn sunbaked_canyon_pays_life_for_red_or_white_and_cycles_itself() {
    let mut s = five_color_setup(&["Sunbaked Canyon"]);
    let canyon = s.force_permanent_on_battlefield(0, "Sunbaked Canyon");
    s.set_player_mana_pool(0, "");
    s.game_mut().scenario_refresh_priority().unwrap();
    activate(&mut s, canyon, 0);
    choose_mode(&mut s, 1);
    assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("W"));
    s.assert_life(0, 19);

    let mut s = five_color_setup(&["Sunbaked Canyon"]);
    let canyon = s.force_permanent_on_battlefield(0, "Sunbaked Canyon");
    s.game_mut().scenario_refresh_priority().unwrap();
    let hand = s.zone_size(0, ZoneType::Hand);
    activate(&mut s, canyon, 1);
    settle(&mut s);
    s.assert_zone_size(0, ZoneType::Hand, hand + 1);
    assert!(s.game().state.permanents[canyon].is_none());
    s.assert_life(0, 20);
}
