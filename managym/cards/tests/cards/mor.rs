//! Morningtide card behavior.

#[test]
fn foundations_spell_negate() {
    crate::check_counterspell("Negate", "Sol Ring", "Gray Ogre");
}

#[test]
fn foundations_5e_kindled_fury() {
    use managym::state::card::Keyword;
    crate::assert_targeted_pump(
        "Kindled Fury",
        "Gray Ogre",
        0,
        (3, 2),
        &[Keyword::FirstStrike],
        0,
        0,
    );
}
