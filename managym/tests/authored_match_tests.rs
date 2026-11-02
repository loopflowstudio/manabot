// Authored-match proof: the selected product decks play to terminal on their authored pack.

use managym::{
    cards::setups::{authored_pack, AuthoredPack},
    flow::event::GameEvent,
    state::player::PlayerConfig,
    Game,
};

fn authored_configs(pack: &AuthoredPack) -> Vec<PlayerConfig> {
    vec![
        pack.player_config("UR Lessons", "ur_lessons")
            .expect("UR setup"),
        pack.player_config("GW Allies", "gw_allies")
            .expect("GW setup"),
    ]
}

fn jeong_configs(pack: &AuthoredPack) -> Vec<PlayerConfig> {
    vec![
        pack.player_config("UR Lessons", "ur_lessons")
            .expect("UR setup"),
        pack.player_config("GW Allies — Jeong", "gw_allies_jeong")
            .expect("GW setup"),
    ]
}

#[test]
fn exact_authored_match_runs_to_a_winner_on_its_authored_pack() {
    let pack = authored_pack("two_deck").unwrap();
    let mut game = Game::new(authored_configs(pack), 0, true);
    assert_eq!(game.state.content.pack_key(), Some(pack.pack_key.as_str()));
    assert_eq!(
        game.state.content.content_digest(),
        pack.content.content_digest()
    );

    let winner = game
        .random_playout(100_000, None)
        .expect("compiled authored match runs");
    assert!(game.is_game_over(), "authored match must reach terminal");
    assert!(winner.is_some(), "authored match must have a winner");

    let events = game.drain_events();
    let resolved_spells = events
        .iter()
        .filter(|event| matches!(event, GameEvent::SpellResolved { .. }))
        .count();
    let triggered_abilities = events
        .iter()
        .filter(|event| matches!(event, GameEvent::AbilityTriggered { .. }))
        .count();
    let damage = events
        .iter()
        .filter(|event| matches!(event, GameEvent::DamageDealt { .. }))
        .count();
    assert!(resolved_spells > 0, "compiled spell programs must resolve");
    assert!(
        triggered_abilities > 0,
        "compiled triggered programs must resolve"
    );
    assert!(damage > 0, "compiled damage programs must affect the world");
}

#[test]
fn jeong_increment_selects_its_exact_pack_in_either_seat_order() {
    let pack = authored_pack("jeong_increment").unwrap();
    let configs = jeong_configs(pack);

    for player_configs in [configs.clone(), configs.into_iter().rev().collect()] {
        let game = Game::new(player_configs, 23, true);
        assert_eq!(
            game.state.content.pack_key(),
            Some("tla-jeong-increment-v1")
        );
        assert_eq!(
            game.state.content.content_digest(),
            pack.content.content_digest()
        );
    }
}

#[test]
fn tampered_jeong_deck_does_not_select_an_authored_pack() {
    let mut configs = jeong_configs(authored_pack("jeong_increment").unwrap());
    let deck = &mut configs[1].deck.main;
    deck.remove("Jeong Jeong's Deserters");
    *deck.entry("Plains".to_owned()).or_default() += 1;

    let game = Game::new(configs, 23, true);
    assert!(game.state.content.pack_key().is_none());
}

#[test]
fn non_authored_decks_keep_the_general_content_pack() {
    use std::collections::BTreeMap;

    let deck = BTreeMap::from([("Mountain".to_owned(), 20), ("Gray Ogre".to_owned(), 20)]);
    let game = Game::new(
        vec![
            PlayerConfig::new("left", deck.clone()),
            PlayerConfig::new("right", deck),
        ],
        7,
        true,
    );

    assert!(game.state.content.pack_key().is_none());
}

#[test]
fn elemental_validation_rosters_select_authored_content_and_play_to_terminal() {
    let pack = authored_pack("elemental").unwrap();
    let configs = vec![
        pack.player_config("left", "elemental_left").unwrap(),
        pack.player_config("right", "elemental_right").unwrap(),
    ];
    for configs in [configs.clone(), configs.into_iter().rev().collect()] {
        let mut game = Game::new(configs, 161, true);
        assert_eq!(game.state.content.pack_key(), Some(pack.pack_key.as_str()));
        assert!(game.random_playout(100_000, None).unwrap().is_some());
        assert!(game.is_game_over());
    }
}
