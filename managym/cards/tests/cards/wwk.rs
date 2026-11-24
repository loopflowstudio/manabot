//! Worldwake card behavior.

#[test]
fn foundations_grant_basilisk_collar() {
    use managym::state::card::Keyword;
    crate::check_attachment_grant(
        "Basilisk Collar",
        Some("2"),
        (2, 2),
        &[Keyword::Deathtouch, Keyword::Lifelink],
    );
}
