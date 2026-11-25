//! Tests for cards filed in `src/arb.rs`, in card order.

use crate::support::{
    game::{five_color_setup, move_card},
    play::{cast, settle},
};
use managym::state::{game_object::Target, zone::ZoneType};

/// Maelstrom Pulse destroys its target and every other permanent with the
/// same name, on either side, at once. A target gone before resolution
/// leaves the spell without a legal target, and nothing is destroyed.
#[test]
fn maelstrom_pulse_destroys_the_target_and_its_namesakes() {
    for target_leaves in [false, true] {
        let mut s = five_color_setup(&["Maelstrom Pulse", "Gray Ogre", "Hill Giant"]);
        let target = s.force_permanent_on_battlefield(1, "Gray Ogre");
        let namesakes = [
            s.force_permanent_on_battlefield(1, "Gray Ogre"),
            s.force_permanent_on_battlefield(0, "Gray Ogre"),
        ];
        let other = s.force_permanent_on_battlefield(1, "Hill Giant");
        cast(&mut s, "Maelstrom Pulse");
        assert!(s.choose_target(Target::Permanent(target)));
        if target_leaves {
            let object = s.game().permanent_object(target).unwrap();
            move_card(&mut s, object, ZoneType::Hand);
        }
        settle(&mut s);
        for namesake in namesakes {
            assert_eq!(
                s.game().state.permanents[namesake].is_some(),
                target_leaves,
                "namesakes survive only a fizzle"
            );
        }
        assert!(s.game().state.permanents[other].is_some());
    }
}
