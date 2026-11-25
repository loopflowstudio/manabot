//! Tests for cards filed in `src/sok.rs`.
use crate::support::{
    game::{five_color_setup, force_new_card_in_hand},
    play::{cast_card, settle, target_opponent},
};
use managym::{agent::action::ActionSpaceKind, state::game_object::PlayerId};

/// The exact life total is read as the spell resolves (CR 608.2c), not as
/// it is cast.
#[test]
fn hidetsugus_second_rite_deals_ten_only_at_exactly_ten_on_resolution() {
    for (cast_at, resolve_at, after) in [(10, 10, 0), (10, 9, 9), (10, 11, 11), (12, 10, 0)] {
        let mut s = five_color_setup(&["Hidetsugu's Second Rite"]);
        s.game_mut().scenario_set_life(PlayerId(1), cast_at);
        let rite = force_new_card_in_hand(&mut s, 0, "Hidetsugu's Second Rite");
        cast_card(&mut s, rite);
        target_opponent(&mut s);
        while s.action_space().kind != ActionSpaceKind::Priority {
            s.step_action(0);
        }
        assert_eq!(s.game().state.stack_objects.len(), 1);
        s.game_mut().scenario_set_life(PlayerId(1), resolve_at);
        s.pass_priority();
        s.pass_priority();
        s.assert_life(1, after);
        if after == 0 {
            s.assert_winner(0);
        } else {
            settle(&mut s);
        }
    }
}
