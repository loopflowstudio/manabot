//! Modern Horizons 2 card behavior.

#[test]
fn foundations_5d_vile_entomber() {
    use managym::state::zone::ZoneType;
    crate::assert_required_library_search("Vile Entomber", ZoneType::Graveyard);
}
