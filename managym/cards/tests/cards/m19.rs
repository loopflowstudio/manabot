#[test]
fn token_producer_heroic_reinforcements() {
    use crate::support::{game::five_color_setup, play::play_and_settle};
    let mut s = five_color_setup(&["Heroic Reinforcements", "Gray Ogre"]);
    let earlier = s.force_permanent_on_battlefield(0, "Gray Ogre");
    play_and_settle(&mut s, "Heroic Reinforcements");
    assert_eq!(s.game().characteristics(earlier).pt(), (3, 3));
    assert!(s
        .game()
        .characteristics(earlier)
        .keywords
        .has(managym::state::card::Keyword::Haste));
    crate::assert_creature_tokens(
        &s,
        "Soldier",
        2,
        (2, 2),
        &[managym::state::mana::Color::White],
        &[managym::state::card::Keyword::Haste],
    );
    let late = s.force_permanent_on_battlefield(0, "Gray Ogre");
    assert_eq!(s.game().characteristics(late).pt(), (2, 2));
    assert!(!s
        .game()
        .characteristics(late)
        .keywords
        .has(managym::state::card::Keyword::Haste));
    s.advance_to_active_step(1, managym::flow::turn::StepKind::Main);
    for id in s.battlefield_permanents_named(0, "Soldier") {
        let face = s.game().characteristics(id);
        assert_eq!(face.pt(), (1, 1));
        assert!(!face.keywords.has(managym::state::card::Keyword::Haste));
    }
}

#[test]
fn token_producer_lathliss_dragon_queen() {
    use crate::support::{game::five_color_setup, play::enter_and_settle};
    let mut s = five_color_setup(&["Lathliss, Dragon Queen", "Shivan Dragon"]);
    enter_and_settle(&mut s, "Lathliss, Dragon Queen");
    enter_and_settle(&mut s, "Shivan Dragon");
    crate::assert_creature_tokens(
        &s,
        "Dragon",
        1,
        (5, 5),
        &[managym::state::mana::Color::Red],
        &[managym::state::card::Keyword::Flying],
    );
    let queen = crate::support::inspect::permanent_named(&s, "Lathliss, Dragon Queen");
    crate::support::play::activate(&mut s, queen, 0);
    crate::support::play::settle(&mut s);
    assert_eq!(s.game().characteristics(queen).pt(), (7, 6));
    for name in ["Shivan Dragon", "Dragon"] {
        let dragon = crate::support::inspect::permanent_named(&s, name);
        assert_eq!(s.game().characteristics(dragon).pt(), (6, 5));
    }
}

#[test]
fn printed_creature_gigantosaurus() {
    crate::printed_creature_combat("Gigantosaurus", 10, 10, &[]);
}
