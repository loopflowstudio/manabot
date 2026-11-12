//! Tests for cards filed in `src/m15.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    play::{activate, cast_owned, choose_mode, pass_until, settle},
};
use managym::{
    agent::action::ActionSpaceKind,
    state::game_object::{PlayerId, Target},
};

#[test]
fn meteorite_deals_two_on_entry_and_taps_for_any_color() {
    let mut s = five_color_setup(&["Meteorite"]);
    let object = cast_owned(&mut s, "Meteorite");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    settle(&mut s);
    s.assert_life(1, 18);
    let meteorite = s.game().state.object_to_permanent[object].unwrap();
    s.set_player_mana_pool(0, "");
    s.game_mut().scenario_refresh_priority().unwrap();
    activate(&mut s, meteorite, 0);
    choose_mode(&mut s, 2);
    assert_eq!(
        s.game().state.players[0].mana_pool,
        managym::state::mana::Mana::parse("B")
    );
}
