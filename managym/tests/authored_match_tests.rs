// Authored-match proof: the product decks play to terminal on the one catalog.

use std::sync::Arc;

use managym::{
    cards::{
        content_pack::default_content_pack,
        setups::{authored_setup, AuthoredSetup},
    },
    flow::event::GameEvent,
    state::player::PlayerConfig,
    Game,
};

fn authored_configs(pack: &AuthoredSetup) -> Vec<PlayerConfig> {
    vec![
        pack.player_config("UR Lessons", "ur_lessons")
            .expect("UR setup"),
        pack.player_config("GW Allies", "gw_allies")
            .expect("GW setup"),
    ]
}

fn jeong_configs(pack: &AuthoredSetup) -> Vec<PlayerConfig> {
    vec![
        pack.player_config("UR Lessons", "ur_lessons")
            .expect("UR setup"),
        pack.player_config("GW Allies — Jeong", "gw_allies_jeong")
            .expect("GW setup"),
    ]
}

#[test]
fn exact_authored_match_runs_to_a_winner_on_the_catalog() {
    let pack = authored_setup("two_deck").unwrap();
    let mut game = Game::new(authored_configs(pack), 0, true);
    assert!(Arc::ptr_eq(&game.state.content, &default_content_pack()));

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
fn jeong_increment_runs_on_the_catalog_in_either_seat_order() {
    let configs = jeong_configs(authored_setup("jeong_increment").unwrap());
    for player_configs in [configs.clone(), configs.into_iter().rev().collect()] {
        let game = Game::new(player_configs, 23, true);
        assert!(Arc::ptr_eq(&game.state.content, &default_content_pack()));
    }
}

#[test]
fn elemental_validation_rosters_play_to_terminal() {
    let pack = authored_setup("elemental").unwrap();
    let configs = vec![
        pack.player_config("left", "elemental_left").unwrap(),
        pack.player_config("right", "elemental_right").unwrap(),
    ];
    for configs in [configs.clone(), configs.into_iter().rev().collect()] {
        let mut game = Game::new(configs, 161, true);
        assert!(game.random_playout(100_000, None).unwrap().is_some());
        assert!(game.is_game_over());
    }
}
