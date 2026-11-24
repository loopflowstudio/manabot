#[test]
fn printed_creature_bishops_soldier() {
    use managym::state::card::Keyword;
    crate::printed_creature_combat("Bishop's Soldier", 2, 2, &[Keyword::Lifelink]);
}

#[test]
fn foundations_spell_rivers_rebuke() {
    use crate::support::game::five_color_setup;
    use crate::support::inspect::{card_of, zone_of};
    use crate::support::play::{cast, settle};
    use managym::state::game_object::{PlayerId, Target};
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["River's Rebuke", "Sol Ring", "Island", "Gray Ogre"]);
    let artifact = s.force_permanent_on_battlefield(1, "Sol Ring");
    let land = s.force_permanent_on_battlefield(1, "Island");
    let mine = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let object = card_of(&s, artifact);
    cast(&mut s, "River's Rebuke");
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    settle(&mut s);
    assert_eq!(zone_of(&s, object), Some(ZoneType::Hand));
    assert_eq!(s.game().characteristics(mine).pt(), (2, 2));
    assert!(s.game().state.permanents[land].is_some());
}

#[test]
fn foundations_spell_dive_down() {
    use crate::support::game::five_color_setup;
    use crate::support::play::{cast, settle};
    use managym::flow::turn::StepKind;
    use managym::state::card::Keyword;
    use managym::state::game_object::Target;
    let mut s = five_color_setup(&["Dive Down", "Hill Giant"]);
    let subject = s.force_permanent_on_battlefield(0, "Hill Giant");
    cast(&mut s, "Dive Down");
    assert!(s.choose_target(Target::Permanent(subject)));
    settle(&mut s);
    assert_eq!(s.game().characteristics(subject).pt(), (3, 6));
    assert!(s
        .game()
        .characteristics(subject)
        .keywords
        .has(Keyword::Hexproof));
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().characteristics(subject).pt(), (3, 3));
    assert!(!s
        .game()
        .characteristics(subject)
        .keywords
        .has(Keyword::Hexproof));
}
