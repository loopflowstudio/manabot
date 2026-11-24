#[test]
fn token_producer_maalfeld_twins() {
    use crate::support::{game::five_color_setup, play::enter_and_settle};
    let mut s = five_color_setup(&["Maalfeld Twins"]);
    let source = enter_and_settle(&mut s, "Maalfeld Twins");
    crate::support::play::move_and_settle(
        &mut s,
        source,
        managym::state::zone::ZoneType::Graveyard,
    );
    crate::assert_creature_tokens(
        &s,
        "Zombie",
        2,
        (2, 2),
        &[managym::state::mana::Color::Black],
        &[],
    );
}
