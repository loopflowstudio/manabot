use std::collections::BTreeMap;

use managym::{state::player::PlayerConfig, Game};

use super::helpers::*;

/// CR 103.4, 103.5 — Each player shuffles and draws seven cards for opening hands.
#[test]
fn cr_103_5_opening_hands_have_seven_cards() {
    let s = Scenario::new(mountain_deck(), forest_deck(), 7);

    s.assert_zone_size(0, managym::state::zone::ZoneType::Hand, 7);
    s.assert_zone_size(1, managym::state::zone::ZoneType::Hand, 7);
}

/// CR 103.3 — Players start the game at 20 life.
#[test]
fn cr_103_3_players_start_at_twenty_life() {
    let s = Scenario::new(mountain_deck(), forest_deck(), 13);

    s.assert_life(0, 20);
    s.assert_life(1, 20);
}

/// Negative: this engine stage supports exactly two players.
#[test]
#[should_panic(expected = "game supports exactly two players")]
fn cr_103_negative_rejects_non_two_player_game() {
    let p0 = PlayerConfig::new("solo", BTreeMap::from([("Mountain".to_string(), 40)]));
    let _ = Game::new(vec![p0], 1, false);
}
