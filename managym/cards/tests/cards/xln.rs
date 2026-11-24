#[test]
fn printed_creature_bishops_soldier() {
    use managym::state::card::Keyword;
    crate::printed_creature_combat(
        "Bishop's Soldier",
        2,
        2,
        managym::state::card::Keywords::of(&[Keyword::Lifelink]),
    );
}
