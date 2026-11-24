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
