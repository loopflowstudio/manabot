//! Tests for cards filed in `src/m19.rs`, in card order.

use crate::support::{game::five_color_setup, inspect::pt_modifications, play::enter_and_settle};
use managym::flow::turn::StepKind;

/// Leonin Vanguard's combat trigger checks for three creatures as it
/// triggers and resolves: with two it does nothing; with three it gets
/// +1/+1 until end of turn and its controller gains 1 life.
#[test]
fn leonin_vanguard_grows_and_gains_life_with_three_creatures() {
    for (others, bonus, life) in [(1, (0, 0), 20), (2, (1, 1), 21)] {
        let mut s = five_color_setup(&["Leonin Vanguard", "Gray Ogre"]);
        let vanguard = s.force_permanent_on_battlefield(0, "Leonin Vanguard");
        for _ in 0..others {
            s.force_permanent_on_battlefield(0, "Gray Ogre");
        }
        s.force_permanent_on_battlefield(1, "Gray Ogre");
        s.advance_to_active_step(0, StepKind::DeclareAttackers);
        assert_eq!(pt_modifications(&s, vanguard), bonus, "{others} others");
        s.assert_life(0, life);
    }
}

/// Volley Veteran deals damage equal to its controller's Goblins, itself
/// included, to target creature an opponent controls.
#[test]
fn volley_veteran_deals_damage_equal_to_your_goblins() {
    use crate::support::inspect::permanent;
    let mut s = five_color_setup(&["Volley Veteran", "Fanatical Firebrand", "Hill Giant"]);
    s.force_permanent_on_battlefield(0, "Fanatical Firebrand");
    let giant = s.force_permanent_on_battlefield(1, "Hill Giant");
    enter_and_settle(&mut s, "Volley Veteran");
    assert_eq!(permanent(&s, giant).damage, 2);
}
