//! Shards of Alara card behavior.

#[test]
fn foundations_5d_death_baron() {
    use crate::support::game::{five_color_setup, move_card, until_end_of_turn};
    use crate::support::inspect::{card_of, keywords};
    use managym::state::card::Keyword;
    use managym::state::effect::Modification;
    use managym::state::zone::ZoneType;

    let mut s = five_color_setup(&["Death Baron", "Hungry Ghoul", "Gray Ogre"]);
    let zombie = s.force_permanent_on_battlefield(0, "Hungry Ghoul");
    let enemy = s.force_permanent_on_battlefield(1, "Hungry Ghoul");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let baron = s.force_permanent_on_battlefield(0, "Death Baron");
    assert_eq!(s.game().characteristics(baron).pt(), (2, 2));
    assert_eq!(s.game().characteristics(zombie).pt(), (3, 3));
    until_end_of_turn(
        &mut s,
        zombie,
        Modification::AddSubtypes(vec!["Skeleton".into()]),
    );
    until_end_of_turn(
        &mut s,
        baron,
        Modification::AddSubtypes(vec!["Skeleton".into()]),
    );
    assert_eq!(s.game().characteristics(zombie).pt(), (3, 3));
    assert_eq!(s.game().characteristics(baron).pt(), (3, 3));
    assert!(keywords(&s, zombie).has(Keyword::Deathtouch));
    assert!(!keywords(&s, enemy).has(Keyword::Deathtouch));
    assert!(!keywords(&s, ogre).has(Keyword::Deathtouch));
    let source = card_of(&s, baron);
    move_card(&mut s, source, ZoneType::Graveyard);
    assert_eq!(s.game().characteristics(zombie).pt(), (2, 2));
    assert!(!keywords(&s, zombie).has(Keyword::Deathtouch));
}
