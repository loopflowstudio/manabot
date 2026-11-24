//! Streets of New Capenna card behavior.

#[test]
fn foundations_5d_involuntary_employment() {
    use crate::support::game::five_color_setup;
    use crate::support::inspect::{keywords, permanent, tapped};
    use crate::support::play::{cast, settle};
    use managym::flow::turn::StepKind;
    use managym::state::card::Keyword;
    use managym::state::game_object::{PlayerId, Target};

    let mut s = five_color_setup(&["Involuntary Employment", "Gray Ogre"]);
    let id = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.game_mut().state.permanents[id].as_mut().unwrap().tapped = true;
    cast(&mut s, "Involuntary Employment");
    assert!(s.choose_target(Target::Permanent(id)));
    settle(&mut s);
    assert!(!tapped(&s, id));
    assert_eq!(permanent(&s, id).controller, PlayerId(0));
    assert!(keywords(&s, id).has(Keyword::Haste));
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(permanent(&s, id).controller, PlayerId(1));
    assert!(!keywords(&s, id).has(Keyword::Haste));
    assert_eq!(s.battlefield_permanents_named(0, "Treasure").len(), 1);
}

#[test]
fn foundations_5f_inspiring_overseer() {
    use crate::support::game::five_color_setup;
    use crate::support::play::enter_and_settle;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Inspiring Overseer"]);
    let hand = s.zone_size(0, ZoneType::Hand);
    enter_and_settle(&mut s, "Inspiring Overseer");
    s.assert_life(0, 21);
    s.assert_zone_size(0, ZoneType::Hand, hand + 1);
}
