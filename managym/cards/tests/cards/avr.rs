#[test]
fn token_producer_maalfeld_twins() {
    use crate::support::{
        game::five_color_setup,
        play::{enter_and_settle, settle},
    };
    let mut s = five_color_setup(&["Maalfeld Twins"]);
    let source = enter_and_settle(&mut s, "Maalfeld Twins");
    crate::support::game::move_card(&mut s, source, managym::state::zone::ZoneType::Graveyard);
    s.game_mut().scenario_refresh_priority().unwrap();
    s.pass_priority();
    settle(&mut s);
    crate::assert_creature_tokens(
        &s,
        "Zombie",
        2,
        (2, 2),
        &[managym::state::mana::Color::Black],
        &[],
    );
}
