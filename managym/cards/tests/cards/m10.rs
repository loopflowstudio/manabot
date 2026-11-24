//! Magic 2010 card behavior.

#[test]
fn foundations_spell_essence_scatter() {
    use crate::support::game::{five_color_setup, move_to_hand};
    use crate::support::inspect::{now, zone_of};
    use crate::support::play::{cast, castable, settle, spend_colorless};
    use managym::state::game_object::Target;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Essence Scatter", "Gray Ogre", "Sol Ring"]);
    let counter = move_to_hand(&mut s, "Essence Scatter");
    cast(&mut s, "Sol Ring");
    spend_colorless(&mut s);
    assert!(!castable(&mut s, counter));
    settle(&mut s);
    let victim = cast(&mut s, "Gray Ogre");
    spend_colorless(&mut s);
    crate::support::play::cast_card(&mut s, counter);
    assert!(s.choose_target(Target::StackSpell(now(&s, victim))));
    settle(&mut s);
    assert_eq!(zone_of(&s, victim), Some(ZoneType::Graveyard));
}
