use crate::support::{
    game::five_color_setup,
    inspect::{counters, permanent_named},
    play::{cast_owned, enter_and_settle, pass_until, play_and_settle, settle, spend_colorless},
};
use managym::{agent::action::ActionSpaceKind, state::game_object::Target};

#[test]
fn token_producer_primeval_bounty() {
    let mut s = five_color_setup(&["Primeval Bounty", "Gray Ogre", "Sol Ring"]);
    enter_and_settle(&mut s, "Primeval Bounty");
    play_and_settle(&mut s, "Gray Ogre");
    let beast = permanent_named(&s, "Beast");
    assert_eq!(s.game().characteristics(beast).pt(), (3, 3));
    cast_owned(&mut s, "Sol Ring");
    spend_colorless(&mut s);
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(beast)));
    settle(&mut s);
    assert_eq!(counters(&s, beast), 3);
    enter_and_settle(&mut s, "Forest");
    s.assert_life(0, 23);
    assert_eq!(s.battlefield_permanents_named(0, "Beast"), [beast]);
}
