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
    for (name, colors) in [
        ("Temple of Enlightenment", "WU"),
        ("Temple of Malice", "BR"),
        ("Temple of Plenty", "GW"),
    ] {
        crate::support::play::temple_scries_one_and_pays_either_color(name, colors);
    }
}
