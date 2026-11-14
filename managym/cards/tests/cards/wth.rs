//! Tests for cards filed in `src/wth.rs`, in card order.

use crate::support::inspect::permanent_of;
use crate::support::{
    game::five_color_setup,
    play::{attack_with, cast_owned, pass_until, settle},
};
use managym::{agent::action::ActionSpaceKind, flow::turn::StepKind};

#[test]
fn fervor_lets_a_new_creature_attack_at_once() {
    let mut s = five_color_setup(&["Fervor", "Gray Ogre"]);
    cast_owned(&mut s, "Fervor");
    settle(&mut s);
    let object = cast_owned(&mut s, "Gray Ogre");
    settle(&mut s);
    let ogre = permanent_of(&s, object).unwrap();
    pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
    attack_with(&mut s, ogre);
    s.advance_to_step(StepKind::EndOfCombat);
    s.assert_life(1, 18);
}
