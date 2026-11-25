//! Tests for cards filed in `src/tmt.rs`, in card order.

use crate::support::{
    game::{five_color_setup, move_card},
    inspect::counters,
};
use managym::{flow::turn::StepKind, state::zone::ZoneType};

/// Michelangelo's disappear trigger counts any permanent, a land included,
/// that left the battlefield under its controller's control this turn; an
/// opponent's departure does not count.
#[test]
fn michelangelo_grows_after_your_permanent_left_this_turn() {
    for (leaver, mine, grows) in [
        (None, true, false),
        (Some("Mountain"), true, true),
        (Some("Gray Ogre"), true, true),
        (Some("Gray Ogre"), false, false),
    ] {
        let mut s = five_color_setup(&["Michelangelo, Game Master", "Gray Ogre"]);
        let michelangelo = s.force_permanent_on_battlefield(0, "Michelangelo, Game Master");
        if let Some(name) = leaver {
            let seat = if mine { 0 } else { 1 };
            let leaving = s.force_permanent_on_battlefield(seat, name);
            let leaving = s.game().state.permanents[leaving].as_ref().unwrap().object;
            move_card(&mut s, leaving, ZoneType::Graveyard);
        }
        s.advance_to_active_step(0, StepKind::Cleanup);
        assert_eq!(
            counters(&s, michelangelo),
            i32::from(grows),
            "{leaver:?}, mine {mine}"
        );
    }
}
