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

#[test]
fn foundations_5f_pulse_tracker() {
    use managym::flow::turn::StepKind;
    let (mut s, ..) = crate::attack_and_resolve("Pulse Tracker");
    assert_eq!((s.life(0), s.life(1)), (20, 19));
    s.advance_to_active_step(0, StepKind::PostcombatMain);
    assert_eq!((s.life(0), s.life(1)), (20, 18));
}
