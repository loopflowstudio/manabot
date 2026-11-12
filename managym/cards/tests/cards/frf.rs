//! Tests for cards filed in `src/frf.rs`, in card order.

use crate::support::{
    game::{clear_summoning_sickness, five_color_setup},
    inspect::{can_activate, permanent},
    play::{activate, settle},
};
use managym::{
    flow::turn::StepKind,
    state::{game_object::PlayerId, zone::ZoneType},
};

#[test]
fn humble_defector_draws_two_then_its_target_opponent_gains_control_only_on_your_turn() {
    let mut s = five_color_setup(&["Humble Defector"]);
    let defector = s.force_permanent_on_battlefield(0, "Humble Defector");
    clear_summoning_sickness(&mut s, defector);
    s.game_mut().scenario_refresh_priority().unwrap();
    let hand = s.zone_size(0, ZoneType::Hand);
    activate(&mut s, defector, 0);
    settle(&mut s);
    s.assert_zone_size(0, ZoneType::Hand, hand + 2);
    assert_eq!(permanent(&s, defector).controller, PlayerId(1));
    s.advance_to_active_step(0, StepKind::Upkeep);
    while s.action_space().player != Some(PlayerId(1)) {
        s.advance_default_action();
    }
    assert!(
        !can_activate(&s, defector),
        "only during its controller's turn"
    );
}
