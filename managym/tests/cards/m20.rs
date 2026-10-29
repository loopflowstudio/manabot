//! Tests for cards filed in `src/cardsets/m20.rs`, grouped by the helper family they
//! drive play with (`tests/support/`).

use crate::support::elemental::{cast, settle, setup};
use managym::{
    agent::action::Action,
    state::{game_object::Target, zone::ZoneType},
};

#[test]
fn spectral_sailor_draw_ability_survives_the_sources_destruction() {
    let mut s = setup(&["Spectral Sailor", "Lightning Strike"]);
    let sailor = s.force_permanent_on_battlefield(0, "Spectral Sailor");
    s.game_mut().scenario_refresh_priority().unwrap();
    let index = s
        .action_space()
        .actions
        .iter()
        .position(|a| matches!(a,Action::ActivateAbility{permanent,..} if *permanent==sailor))
        .unwrap();
    s.step_action(index);
    cast(&mut s, "Lightning Strike");
    assert!(s.choose_target(Target::Permanent(sailor)));
    settle(&mut s);
    assert!(s.game().state.permanents[sailor].is_none());
    assert_eq!(s.zone_size(0, ZoneType::Hand), 1);
}
