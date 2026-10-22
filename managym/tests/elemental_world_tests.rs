// Full Elemental admission: exact rosters, complete native coherence and bounded games.
use managym::{cardsets::alpha::ContentPack, semantic::SemanticPack, Game};

#[test]
fn retained_professor_cast_prefix_can_enter_and_finish_mana_payment() {
    // Native failure retained by MTG-175: one Forest, off-color lands and
    // Prophetic Prism can pay 2GG, provided Prism preserves the Forest.
    let pack = SemanticPack::elemental_world().unwrap();
    let mut configs = vec![
        pack.player_config("arena-seat-0", "elemental_wb").unwrap(),
        pack.player_config("arena-seat-1", "elemental_ug").unwrap(),
    ];
    for config in &mut configs {
        config.content_pack = Some(pack.pack_key.clone());
    }
    let mut env = managym::agent::env::Env::new(2165384, true, false, false);
    env.reset(configs).unwrap();
    let prefix = [
        2, 2, 0, 0, 4, 3, 3, 5, 2, 0, 2, 2, 2, 6, 2, 3, 1, 2, 1, 0, 4, 0, 0, 8, 0, 0, 1, 4, 6, 2,
        2, 5, 1, 1,
    ];
    for (step, action) in prefix.into_iter().enumerate() {
        env.step(action)
            .unwrap_or_else(|error| panic!("prefix {step}: {error:?}"));
    }
    env.step(4).expect("offered Professor cast enters payment");
    assert!(env
        .random_playout(2165384, 30_000)
        .expect("payment and subsequent offered actions execute")
        .is_some());
    assert!(env.is_game_over());
}

#[test]
fn selected_world_compiles_every_card_without_reduced_native_rules() {
    let semantic = SemanticPack::elemental_world().unwrap();
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

#[test]
fn every_pair_and_mirror_uses_the_selected_world_and_completes() {
    let semantic = SemanticPack::elemental_world().unwrap();
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
