//! Scourge card behavior.

#[test]
fn foundations_5d_fierce_empath() {
    crate::assert_optional_revealed_search(
        "Fierce Empath",
        &["Shivan Dragon", "Rune-Scarred Demon"],
        &["Gray Ogre", "Blasphemous Edict"],
    );
}
