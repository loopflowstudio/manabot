//! Tests for cards filed in `src/bng.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    play::{cast, settle},
};
use managym::state::game_object::Target;

#[test]
fn searing_blood_hits_the_controller_only_when_the_creature_dies() {
    for (victim, life) in [("Gray Ogre", 17), ("Agent of Treachery", 20)] {
        let mut s = five_color_setup(&["Searing Blood", victim]);
        let creature = s.force_permanent_on_battlefield(1, victim);
        cast(&mut s, "Searing Blood");
        assert!(s.choose_target(Target::Permanent(creature)));
        settle(&mut s);
        s.assert_life(1, life);
        s.assert_life(0, 20);
    }
}

/// Born of the Gods' three Temples enter tapped, scry 1 as they enter (the
/// top card stays or goes to the bottom, as chosen) and pay either of their
/// two colors, and no other.
#[test]
fn temples_enter_tapped_scry_one_and_supply_either_color() {
    use crate::support::{
        game::Scenario,
        play::{choose, pass_until, tapped_dual_land_pays_either_color},
    };
    use managym::agent::action::{Action, ActionSpaceKind};
    use managym::state::{game_object::PlayerId, zone::ZoneType};
    for (name, colors) in [
        ("Temple of Enlightenment", "WU"),
        ("Temple of Malice", "BR"),
        ("Temple of Plenty", "GW"),
    ] {
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
