//! Tests for cards filed in `src/jou.rs`, in card order.

use crate::support::{
    game::Scenario,
    play::{choose, pass_until, settle, tapped_dual_land_pays_either_color},
};
use managym::{
    agent::action::{Action, ActionSpaceKind},
    state::{game_object::PlayerId, zone::ZoneType},
};

/// Journey into Nyx's two Temples enter tapped, scry 1 as they enter (the
/// top card stays or goes to the bottom, as chosen) and pay either of their
/// two colors, and no other.
#[test]
fn temples_enter_tapped_scry_one_and_supply_either_color() {
    for (name, colors) in [("Temple of Epiphany", "UR"), ("Temple of Malady", "BG")] {
        for bottom in [false, true] {
            tapped_dual_land_pays_either_color(name, colors, |s, _| {
                pass_until(s, ActionSpaceKind::Scry);
                let library = |s: &Scenario| {
                    s.game()
                        .state
                        .zones
                        .zone_objects(ZoneType::Library, PlayerId(0))
                        .to_vec()
                };
                let before = library(s);
                let top = *before.last().unwrap();
                assert_eq!(s.action_space().actions.len(), 2);
                choose(
                    s,
                    |a| matches!(a, Action::ScryCard { to_bottom, .. } if *to_bottom == bottom),
                );
                settle(s);
                let after = library(s);
                assert_eq!(after.len(), before.len());
                if bottom {
                    assert_eq!(after[0], top, "{name}");
                } else {
                    assert_eq!(after, before, "{name}");
                }
            });
        }
    }
}
