// Authored-match proof: the selected product decks play to terminal on compiled semantics.

use managym::{
    cardsets::alpha::ContentPack, flow::event::GameEvent, semantic::SemanticPack,
    state::player::PlayerConfig, Game,
};

fn authored_configs(pack: &SemanticPack) -> Vec<PlayerConfig> {
    vec![
        PlayerConfig::new(
            "UR Lessons",
            pack.decklist("ur_lessons").expect("UR deck compiles"),
        ),
        PlayerConfig::new(
            "GW Allies",
            pack.decklist("gw_allies").expect("GW deck compiles"),
        ),
    ]
}

#[test]
fn exact_authored_match_runs_to_a_winner_on_compiled_semantics() {
    let semantic = SemanticPack::two_deck().expect("checked-in semantic IR parses");
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
fn compiled_pack_matches_the_reviewed_reference_behavior() {
    let semantic = SemanticPack::two_deck().expect("checked-in semantic IR parses");
    let compiled = semantic
        .compile_content_pack()
        .expect("semantic IR lowers into live content");
    let reference = ContentPack::default();

    for semantic_definition in &semantic.definitions {
        let compiled_id = compiled
            .definition_id(&semantic_definition.registry_name)
            .expect("compiled definition exists");
        let reference_id = reference
            .definition_id(&semantic_definition.registry_name)
            .expect("reviewed reference definition exists");
        let mut compiled_definition = compiled
            .definition(compiled_id)
            .expect("compiled definition")
            .clone();
        let mut reference_definition = reference
            .definition(reference_id)
            .expect("reference definition")
            .clone();
        if reference_definition.targeting.is_empty() {
            compiled_definition.targeting.clear();
        }
        compiled_definition
            .abilities
            .sort_by_key(|ability| format!("{ability:?}"));
        reference_definition
            .abilities
            .sort_by_key(|ability| format!("{ability:?}"));
        assert_eq!(
            &compiled_definition, &reference_definition,
            "compiled behavior drifted for {}",
            semantic_definition.registry_name
        );
    }
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
