//! Morningtide card behavior.

#[test]
fn foundations_spell_negate() {
    crate::check_counterspell("Negate", "Sol Ring", "Gray Ogre");
}
