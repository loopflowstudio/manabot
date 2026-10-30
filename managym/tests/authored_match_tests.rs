// Authored-match proof: the selected product decks play to terminal on compiled semantics.

use managym::{flow::event::GameEvent, semantic::SemanticPack, state::player::PlayerConfig, Game};

fn authored_configs(pack: &SemanticPack) -> Vec<PlayerConfig> {
    vec![
        pack.player_config("UR Lessons", "ur_lessons")
            .expect("UR setup"),
        pack.player_config("GW Allies", "gw_allies")
            .expect("GW setup"),
    ]
}

fn jeong_configs(pack: &SemanticPack) -> Vec<PlayerConfig> {
    vec![
        pack.player_config("UR Lessons", "ur_lessons")
            .expect("UR setup"),
        pack.player_config("GW Allies — Jeong", "gw_allies_jeong")
            .expect("GW setup"),
    ]
}

#[test]
fn exact_authored_match_runs_to_a_winner_on_compiled_semantics() {
    let semantic = SemanticPack::current("two_deck").expect("checked-in semantic IR parses");
    let mut game = Game::new(authored_configs(&semantic), 0, true);
    let provenance = game
        .state
        .content
        .compiled_semantics()
        .expect("exact authored decks select compiled semantics");

    assert_eq!(provenance.pack_key, semantic.pack_key);
    assert_eq!(provenance.ir_hash, semantic.ir_hash);
    assert_eq!(game.state.content.len(), semantic.definitions.len());

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
    let semantic = SemanticPack::current("jeong_increment").expect("checked-in Jeong IR parses");
    let configs = jeong_configs(&semantic);

    for player_configs in [configs.clone(), configs.into_iter().rev().collect()] {
        let game = Game::new(player_configs, 23, true);
        let provenance = game
            .state
            .content
            .compiled_semantics()
            .expect("exact Jeong decks select compiled semantics");
        assert_eq!(provenance.pack_key, "tla-jeong-increment-v1");
        assert_eq!(provenance.ir_hash, semantic.ir_hash);
        assert_eq!(provenance.source_hash, semantic.source_hash);
    }
}

#[test]
fn tampered_jeong_deck_does_not_select_a_compiled_pack() {
    let semantic = SemanticPack::current("jeong_increment").expect("checked-in Jeong IR parses");
    let mut configs = jeong_configs(&semantic);
    let deck = &mut configs[1].deck.main;
    deck.remove("Jeong Jeong's Deserters");
    *deck.entry("Plains".to_owned()).or_default() += 1;

    let game = Game::new(configs, 23, true);
    assert!(game.state.content.compiled_semantics().is_none());
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

    assert!(game.state.content.compiled_semantics().is_none());
}

#[test]
fn elemental_validation_rosters_select_compiled_content_and_play_to_terminal() {
    let semantic = SemanticPack::current("elemental").unwrap();
    let configs = vec![
        semantic.player_config("left", "elemental_left").unwrap(),
        semantic.player_config("right", "elemental_right").unwrap(),
    ];
    for configs in [configs.clone(), configs.into_iter().rev().collect()] {
        let mut game = Game::new(configs, 161, true);
        assert_eq!(
            game.state.content.compiled_semantics().unwrap().ir_hash,
            semantic.ir_hash
        );
        assert!(game.random_playout(100_000, None).unwrap().is_some());
        assert!(game.is_game_over());
    }
}
