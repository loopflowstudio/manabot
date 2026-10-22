// Full Elemental admission: exact rosters, complete native coherence and bounded games.
use managym::{cardsets::alpha::ContentPack, semantic::SemanticPack, Game};

#[test]
fn selected_world_compiles_every_card_without_reduced_native_rules() {
    for semantic in [
        SemanticPack::elemental_world().unwrap(),
        SemanticPack::optional_payments().unwrap(),
    ] {
        assert_eq!(semantic.definitions.len(), 100);
        assert_eq!(semantic.decks.len(), 10);
        let compiled = semantic.compile_content_pack().unwrap();
        let native = ContentPack::default();
        for definition in &semantic.definitions {
            let name = &definition.registry_name;
            let mut actual = compiled
                .definition(compiled.definition_id(name).unwrap())
                .unwrap()
                .clone();
            let mut expected = native
                .definition(native.definition_id(name).unwrap())
                .unwrap()
                .clone();
            if expected.targeting.is_empty() {
                actual.targeting.clear();
            }
            actual.abilities.sort_by_key(|a| format!("{a:?}"));
            expected.abilities.sort_by_key(|a| format!("{a:?}"));
            assert_eq!(actual, expected, "compiled rules drift for {name}");
        }
    }
}

#[test]
fn every_pair_and_mirror_uses_the_selected_world_and_completes() {
    let semantic = SemanticPack::optional_payments().unwrap();
    for (left, first) in semantic.decks.iter().enumerate() {
        for (right, second) in semantic.decks.iter().enumerate() {
            let mut configs = vec![
                semantic.player_config("left", &first.key).unwrap(),
                semantic.player_config("right", &second.key).unwrap(),
            ];
            for config in &mut configs {
                config.content_pack = Some(semantic.pack_key.clone());
            }
            let seed = 16400 + (left * 10 + right) as u64;
            let mut game = Game::new(configs, seed, true);
            assert_eq!(
                game.state.content.compiled_semantics().unwrap().ir_hash,
                semantic.ir_hash
            );
            let winner = game
                .random_playout(30_000, None)
                .unwrap_or_else(|e| panic!("{} vs {} seed {seed}: {e:?}", first.key, second.key));
            assert!(
                game.is_game_over(),
                "cap: {} vs {} seed {seed}",
                first.key,
                second.key
            );
            assert!(
                winner.is_some(),
                "no winner: {} vs {} seed {seed}",
                first.key,
                second.key
            );
        }
    }
}
