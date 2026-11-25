//! Phyrexia: All Will Be One.

#[test]
fn foundations_5e_offer_immortality() {
    use managym::state::card::Keyword;
    crate::assert_targeted_pump(
        "Offer Immortality",
        "Gray Ogre",
        0,
        (2, 2),
        &[Keyword::Deathtouch, Keyword::Indestructible],
        0,
        0,
    );
}
