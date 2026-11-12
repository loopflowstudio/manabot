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
