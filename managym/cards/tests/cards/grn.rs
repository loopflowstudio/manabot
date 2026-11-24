#[test]
fn printed_creature_swiftblade_vindicator() {
    use managym::state::card::Keyword;
    crate::printed_creature_combat(
        "Swiftblade Vindicator",
        1,
        1,
        &[Keyword::Vigilance, Keyword::Trample, Keyword::DoubleStrike],
    );
}
