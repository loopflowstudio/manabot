//! Tests for cards first printed in Scars of Mirrodin.
use crate::support::{
    game::{five_color_setup, force_new_card_in_hand},
    play::{cast, cast_card, choose_mode, settle},
};
use managym::flow::turn::StepKind;

#[test]
fn exsanguinate_gains_the_life_actually_lost() {
    for (protected, x, lost) in [(false, 3, 3), (true, 3, 0), (false, 0, 0)] {
        let mut s = five_color_setup(&["Exsanguinate", "Teferi's Protection"]);
        if protected {
            cast(&mut s, "Teferi's Protection");
            settle(&mut s);
        }
        s.advance_to_active_step(1, StepKind::Main);
        s.set_player_mana_pool(1, "3BB");
        let spell = force_new_card_in_hand(&mut s, 1, "Exsanguinate");
        cast_card(&mut s, spell);
        choose_mode(&mut s, x);
        settle(&mut s);
        s.assert_life(0, 20 - lost);
        s.assert_life(1, 20 + lost);
    }
}
