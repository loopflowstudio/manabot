//! Hour of Devastation card behavior.

#[test]
fn foundations_spell_abrade() {
    use crate::support::game::five_color_setup;
    use crate::support::inspect::{card_of, zone_of};
    use crate::support::play::{cast, choose_mode, settle};
    use managym::state::game_object::Target;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Abrade", "Sol Ring", "Hill Giant"]);
    let subject = s.force_permanent_on_battlefield(1, "Sol Ring");
    let object = card_of(&s, subject);
    let creature = s.force_permanent_on_battlefield(1, "Hill Giant");
    let killed = card_of(&s, creature);
    cast(&mut s, "Abrade");
    choose_mode(&mut s, 1);
    assert!(s.choose_target(Target::Permanent(subject)));
    settle(&mut s);
    assert_eq!(zone_of(&s, object), Some(ZoneType::Graveyard));
    cast(&mut s, "Abrade");
    choose_mode(&mut s, 0);
    assert!(s.choose_target(Target::Permanent(creature)));
    settle(&mut s);
    assert_eq!(zone_of(&s, killed), Some(ZoneType::Graveyard));
}
