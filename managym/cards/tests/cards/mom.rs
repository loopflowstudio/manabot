//! March of the Machine card behavior.

#[test]
fn foundations_5d_wary_thespian() {
    use crate::support::game::{five_color_setup, move_card, put_onto_battlefield};
    use crate::support::inspect::card_of;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Wary Thespian"]);
    let id = put_onto_battlefield(&mut s, 0, "Wary Thespian");
    crate::assert_surveil_one(&mut s);
    let card = card_of(&s, id);
    move_card(&mut s, card, ZoneType::Graveyard);
    s.game_mut().scenario_refresh_priority().unwrap();
    crate::assert_surveil_one(&mut s);
}
