#[test]
fn printed_creature_tajuru_pathwarden() {
    use managym::state::card::Keyword;
    crate::printed_creature_combat(
        "Tajuru Pathwarden",
        5,
        4,
        &[Keyword::Vigilance, Keyword::Trample],
    );
}

#[test]
fn foundations_spell_make_a_stand() {
    use crate::support::game::five_color_setup;
    use crate::support::play::{cast, settle};
    use managym::flow::turn::StepKind;
    use managym::state::card::Keyword;
    let mut s = five_color_setup(&["Make a Stand", "Hill Giant"]);
    let subject = s.force_permanent_on_battlefield(0, "Hill Giant");
    cast(&mut s, "Make a Stand");
    settle(&mut s);
    assert_eq!(s.game().characteristics(subject).pt(), (4, 3));
    assert!(s
        .game()
        .characteristics(subject)
        .keywords
        .has(Keyword::Indestructible));
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().characteristics(subject).pt(), (3, 3));
    assert!(!s
        .game()
        .characteristics(subject)
        .keywords
        .has(Keyword::Indestructible));
}

#[test]
fn foundations_grant_untamed_hunger() {
    use crate::support::game::{five_color_setup, move_card};
    use crate::support::inspect::{card_of, keywords, offered_targets, permanent_named};
    use crate::support::play::{cast, settle};
    use managym::state::card::Keyword;
    use managym::state::game_object::Target;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Untamed Hunger", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let enemy = s.force_permanent_on_battlefield(1, "Gray Ogre");
    cast(&mut s, "Untamed Hunger");
    assert!(offered_targets(&s).contains(&Target::Permanent(enemy)));
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    let source = permanent_named(&s, "Untamed Hunger");
    assert!(keywords(&s, ogre).has(Keyword::Menace));
    assert!(!keywords(&s, enemy).has(Keyword::Menace));
    assert_eq!(s.game().characteristics(ogre).pt(), (4, 3));
    let card = card_of(&s, source);
    move_card(&mut s, card, ZoneType::Graveyard);
    assert!(!keywords(&s, ogre).has(Keyword::Menace));
    assert_eq!(s.game().characteristics(ogre).pt(), (2, 2));
}
