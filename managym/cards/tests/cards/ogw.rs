#[test]
fn printed_creature_tajuru_pathwarden() {
    use managym::state::card::Keyword;
    crate::printed_creature_combat(
        "Tajuru Pathwarden",
        5,
        4,
        managym::state::card::Keywords::of(&[Keyword::Vigilance, Keyword::Trample]),
    );
}
