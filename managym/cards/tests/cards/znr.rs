//! Zendikar Rising behavior.
#[test]
fn kicked_nullpriest_returns_only_creatures() {
    super::check_kicked_return("Nullpriest of Oblivion", "4BB", "Gray Ogre", "Sol Ring");
}
