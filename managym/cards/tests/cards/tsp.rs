//! Time Spiral card behavior.

#[test]
fn foundations_spell_cancel() {
    use crate::support::game::five_color_setup;
    use crate::support::inspect::{now, zone_of};
    use crate::support::play::{cast, settle, spend_colorless};
    use managym::state::game_object::Target;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Cancel", "Gray Ogre"]);
    let victim = cast(&mut s, "Gray Ogre");
    spend_colorless(&mut s);
    cast(&mut s, "Cancel");
    assert!(s.choose_target(Target::StackSpell(now(&s, victim))));
    settle(&mut s);
    assert_eq!(zone_of(&s, victim), Some(ZoneType::Graveyard));
}
