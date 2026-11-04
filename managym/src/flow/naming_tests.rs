//! Naming candidates are public vocabulary, not game objects or effect programs.
//! The offline Scryfall fixture extends only this controlled decision's vocabulary.

use std::{
    collections::{BTreeMap, VecDeque},
    sync::Arc,
};

use crate::{
    agent::{
        observation::Observation,
        observation_encoder::{encode, ObservationEncoderConfig},
    },
    cards::{
        content_pack::default_content_pack,
        names::{NameCandidate, NameVocabulary},
    },
    decision::Command,
    flow::{
        decision::{Decision, EffectFrame, FrameFinalize},
        game::Game,
    },
    search_state::witness,
    state::{
        ability::Effect,
        game_object::{PlayerId, Target},
        player::PlayerConfig,
    },
};

fn vocabulary() -> Arc<NameVocabulary> {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/fixtures/naming_vocabulary.json")).unwrap();
    let pack = default_content_pack();
    let mut entries: BTreeMap<String, Option<crate::state::card::CardDefId>> = fixture["names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| (entry["name"].as_str().unwrap().to_owned(), None))
        .collect();
    // Keep the installed universe, including real back faces, and its learning joins.
    for entry in pack.name_candidates().iter() {
        entries.insert(entry.name.clone(), entry.definition);
    }
    NameVocabulary::new(
        entries
            .into_iter()
            .map(|(name, definition)| NameCandidate { name, definition })
            .collect(),
    )
}

fn naming_game(names: Arc<NameVocabulary>) -> Game {
    let deck = BTreeMap::from([("Gray Ogre".into(), 12), ("Mountain".into(), 28)]);
    let mut game = Game::new(
        vec![
            PlayerConfig::new("p0", deck.clone()),
            PlayerConfig::new("p1", deck),
        ],
        243,
        false,
    );
    game.run_frame(EffectFrame {
        remembered: vec![],
        inherited_from: None,
        source_transforms: 0,
        source: None,
        source_ref: None,
        controller: PlayerId(0),
        resolutions_this_turn: 0,
        kicked: false,
        x: 0,
        excess_damage: 0,
        this_way: vec![],
        moved: vec![],
        chosen: None,
        created: None,
        targets: vec![],
        active_requirement: 0,
        context_target: Some(Target::Player(PlayerId(1))),
        triggering_player: None,
        grantor: None,
        ability: None,
        queue: VecDeque::from([Effect::NameAndExileFromTargetPlayer { count: 4 }]),
        finalize: FrameFinalize::None,
    });
    let Decision::NameCard {
        names: installed, ..
    } = &mut game.state.suspended_decision.as_mut().unwrap().decision
    else {
        panic!("naming decision")
    };
    *installed = names;
    game.publish_action_space(game.suspended_decision_action_space().unwrap());
    game
}

fn command(game: &Game, index: usize) -> Command {
    let frame = game.semantic_decision_frame().unwrap();
    Command {
        command_id: format!("name.{index}"),
        expected_revision: frame.revision,
        offer_id: frame.offers[index].id.0,
        answers: vec![],
        object_preconditions: vec![],
    }
}

#[test]
fn catalog_naming_is_complete_encoded_and_forkable_beyond_512() {
    let names = vocabulary();
    assert!(names.len() > 600);
    let game = naming_game(Arc::clone(&names));
    let root = witness(&game);
    let obs = Observation::for_player(&game, PlayerId(0));
    assert_eq!(obs.action_space.actions.len(), names.len());
    let config = ObservationEncoderConfig {
        policy_input_version: 8,
        max_actions: names.len(),
        max_facts: 128,
        ..Default::default()
    };
    let encoded = encode(&obs, &config).unwrap();
    assert_eq!(
        encoded.actions_valid.iter().sum::<f32>(),
        names.len() as f32
    );
    assert!(encode(&obs, &ObservationEncoderConfig::default()).is_err());
    assert!(Observation::for_player(&game, PlayerId(1))
        .action_space
        .actions
        .is_empty());
    let back = names
        .iter()
        .position(|entry| entry.name == "Fire Lord Sozin")
        .unwrap();
    let ogre = names
        .iter()
        .position(|entry| entry.name == "Gray Ogre")
        .unwrap();
    for index in [0, 511, 512, names.len() - 1, back, ogre] {
        assert_eq!(encoded.name_candidates[2 * index], (index + 1) as f32);
        assert_eq!(
            obs.action_space.actions[index].description.as_deref(),
            Some(format!("Name {}", names.get(index).unwrap().name).as_str())
        );
        let mut branch = game.clone();
        let Decision::NameCard { names: cloned, .. } =
            &branch.state.suspended_decision.as_ref().unwrap().decision
        else {
            panic!()
        };
        assert!(Arc::ptr_eq(&names, cloned));
        let choose = command(&branch, index);
        branch.execute_semantic_command(&choose).unwrap();
        let Decision::SearchNamed {
            name, remaining, ..
        } = &branch.state.suspended_decision.as_ref().unwrap().decision
        else {
            panic!("named continuation")
        };
        assert_eq!(name, &names.get(index).unwrap().name);
        assert_eq!(*remaining, 4);
        let searched = witness(&branch);
        assert!(branch.execute_semantic_command(&choose).is_err());
        assert_eq!(witness(&branch), searched);
        let mut replay = game.clone();
        replay.execute_semantic_command(&choose).unwrap();
        assert_eq!(witness(&replay), searched);
        assert_eq!(witness(&game), root);
        // Fail-to-find remains an explicit choice, even when no definition exists.
        let stop = branch
            .semantic_decision_frame()
            .unwrap()
            .offers
            .iter()
            .position(|o| o.label == "Stop searching and shuffle")
            .unwrap();
        branch
            .execute_semantic_command(&command(&branch, stop))
            .unwrap();
        assert!(branch.state.suspended_decision.is_none());
    }
    // Vocabulary evidence is a digest, not a per-fork serialization of every name.
    assert!(
        serde_json::to_vec(&game.state.suspended_decision.as_ref().unwrap().decision)
            .unwrap()
            .len()
            < 200
    );
}

#[test]
fn installed_naming_vocabulary_preserves_the_exact_real_catalog() {
    let pack = default_content_pack();
    let old: std::collections::BTreeSet<_> = pack
        .definition_entries()
        .filter(|(id, definition)| {
            !definition.is_token && (!definition.key.is_empty() || pack.front_face(*id).is_some())
        })
        .map(|(_, definition)| definition.name.as_str())
        .collect();
    let names = pack.name_candidates();
    assert_eq!(
        old.into_iter().collect::<Vec<_>>(),
        names.iter().map(|c| c.name.as_str()).collect::<Vec<_>>()
    );
    assert!(Arc::ptr_eq(&names, &pack.name_candidates()));
}

#[cfg(feature = "benchmark")]
#[test]
fn catalog_naming_cost_attribution() {
    use crate::agent::observation_encoder::encode_into;
    use std::time::Instant;
    fn measure<T>(label: &str, mut operation: impl FnMut() -> T) {
        let started = Instant::now();
        let result = operation();
        let micros = started.elapsed().as_secs_f64() * 1e6;
        std::hint::black_box(&result);
        drop(result);
        crate::benchmark_allocations::start();
        let result = operation();
        let (allocations, bytes) = crate::benchmark_allocations::stop();
        std::hint::black_box(result);
        println!(
            "{}",
            serde_json::json!({"operation": label, "microseconds": micros, "allocations": allocations, "allocated_bytes": bytes, "samples": 1, "counting_during_timing": false})
        );
    }
    measure("catalog_setup", || {
        let pack = crate::cards::content_pack::ContentPack::default();
        pack.name_candidates()
    });
    measure("validated_large_vocabulary_setup", vocabulary);
    let names = vocabulary();
    let game = naming_game(Arc::clone(&names));
    measure("naming_decision", || Decision::NameCard {
        player: PlayerId(0),
        count: 4,
        names: Arc::clone(&names),
    });
    measure("naming_action_space", || {
        game.suspended_decision_action_space()
    });
    measure("naming_observation", || {
        Observation::for_player(&game, PlayerId(0))
    });
    let obs = Observation::for_player(&game, PlayerId(0));
    let config = ObservationEncoderConfig {
        policy_input_version: 8,
        max_actions: names.len(),
        max_facts: 128,
        ..Default::default()
    };
    let mut buffers = encode(&obs, &config).unwrap();
    measure("naming_encode_into", || {
        encode_into(&obs, &config, buffers.buffers_mut()).unwrap()
    });
    measure("naming_search_fork", || game.clone());
    let deck = BTreeMap::from([("Mountain".into(), 40)]);
    let ordinary = Game::new(
        vec![
            PlayerConfig::new("p0", deck.clone()),
            PlayerConfig::new("p1", deck),
        ],
        243,
        true,
    );
    measure("ordinary_search_fork", || ordinary.clone());
    measure("ordinary_fork_and_step", || {
        let mut branch = ordinary.clone();
        branch.step(0).unwrap();
        branch
    });
}
