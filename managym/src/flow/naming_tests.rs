//! Naming candidates are public vocabulary, not game objects or effect programs.
//! The offline Scryfall fixture extends only this controlled decision's vocabulary.

use std::{collections::BTreeMap, sync::Arc};

use crate::{
    agent::{
        observation::Observation,
        observation_encoder::{encode, ObservationEncoderConfig},
    },
    cards::{
        content_pack::default_content_pack,
        names::{NameOption, NameScope, NAME_BRANCHING},
    },
    decision::Command,
    flow::{
        decision::{Decision, EffectFrame},
        game::Game,
        scenario::naming_fixture_vocabulary,
    },
    search_state::witness,
    state::{
        ability::Effect,
        game_object::{PlayerId, Target},
        player::PlayerConfig,
        zone::ZoneType,
    },
};

fn naming_game() -> Game {
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
        context_target: Some(Target::Player(PlayerId(1))),
        ..EffectFrame::new(
            PlayerId(0),
            vec![Effect::NameAndExileFromTargetPlayer { count: 4 }],
        )
    });
    game.scenario_naming_vocabulary().unwrap();
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

/// The offer whose published group span contains vocabulary row `index`, or
/// the leaf offer naming it. Navigation reads only the public observation.
fn offer_toward(game: &Game, index: usize) -> (usize, bool) {
    let obs = Observation::for_player(game, PlayerId(0));
    obs.action_space
        .actions
        .iter()
        .enumerate()
        .find_map(
            |(offer, action)| match (action.name_group, action.name_index) {
                (Some([first, last]), _) if (first..=last).contains(&index) => Some((offer, false)),
                (_, Some(name)) if name == index => Some((offer, true)),
                _ => None,
            },
        )
        .expect("every name is reachable from every prompt on its path")
}

fn assert_no_naming_effect(before: &Game, after: &Game) {
    // Group navigation narrows the prompt; it searches, reveals and exiles nothing.
    assert!(matches!(
        after.state.suspended_decision.as_ref().unwrap().decision,
        Decision::NameCard { .. }
    ));
    for player in [PlayerId(0), PlayerId(1)] {
        for zone in [
            ZoneType::Library,
            ZoneType::Hand,
            ZoneType::Graveyard,
            ZoneType::Exile,
        ] {
            assert_eq!(
                before.state.zones.zone_cards(zone, player),
                after.state.zones.zone_cards(zone, player)
            );
        }
    }
    assert_eq!(before.state.events.len(), after.state.events.len());
}

#[test]
fn catalog_naming_is_a_complete_bounded_hierarchy_beyond_512() {
    let names = naming_fixture_vocabulary();
    assert!(names.len() > 600);
    let game = naming_game();
    let root = witness(&game);
    let obs = Observation::for_player(&game, PlayerId(0));
    // 928 names: ceil(928 / 64) = 15 balanced groups of 61 or 62 names.
    let groups: Vec<[usize; 2]> = obs
        .action_space
        .actions
        .iter()
        .map(|action| action.name_group.expect("root offers groups"))
        .collect();
    assert_eq!(groups.len(), names.len().div_ceil(NAME_BRANCHING));
    assert_eq!(groups[0][0], 0);
    assert_eq!(groups.last().unwrap()[1], names.len() - 1);
    for pair in groups.windows(2) {
        assert_eq!(pair[0][1] + 1, pair[1][0], "groups tile the vocabulary");
    }
    let sizes: Vec<usize> = groups.iter().map(|[a, b]| b - a + 1).collect();
    assert!(sizes.iter().max().unwrap() - sizes.iter().min().unwrap() <= 1);
    assert!(sizes.iter().all(|size| *size <= NAME_BRANCHING));
    assert_eq!(
        obs.action_space.actions[0].description.as_deref(),
        Some(
            format!(
                "Names {} through {}",
                names.get(0).unwrap().name,
                names.get(groups[0][1]).unwrap().name
            )
            .as_str()
        )
    );
    assert!(Observation::for_player(&game, PlayerId(1))
        .action_space
        .actions
        .is_empty());

    // Policy inputs: v9 encodes group spans at ordinary capacity; earlier
    // inputs see anonymous modal choices.
    let v9 = ObservationEncoderConfig {
        policy_input_version: 9,
        max_actions: NAME_BRANCHING,
        ..Default::default()
    };
    let encoded = encode(&obs, &v9).unwrap();
    assert_eq!(encoded.name_candidates.len(), 4 * NAME_BRANCHING);
    for (row, [first, last]) in groups.iter().enumerate() {
        assert_eq!(
            encoded.name_candidates[4 * row..4 * row + 4],
            [0.0, 0.0, (first + 1) as f32, (last + 1) as f32]
        );
    }
    let v8 = ObservationEncoderConfig {
        policy_input_version: 8,
        ..v9
    };
    assert!(encode(&obs, &v8).unwrap().name_candidates.is_empty());
    assert!(encode(&obs, &ObservationEncoderConfig::default()).is_ok());

    let back = names
        .iter()
        .position(|entry| entry.name == "Fire Lord Sozin")
        .unwrap();
    let ogre = names
        .iter()
        .position(|entry| entry.name == "Gray Ogre")
        .unwrap();
    for index in [0, 511, 512, names.len() - 1, back, ogre] {
        let mut branch = game.clone();
        let mut path = vec![];
        loop {
            let (offer, leaf) = offer_toward(&branch, index);
            let choose = command(&branch, offer);
            path.push(choose.clone());
            if leaf {
                let leaf_obs = Observation::for_player(&branch, PlayerId(0));
                assert!(leaf_obs.action_space.actions.len() <= NAME_BRANCHING);
                assert_eq!(
                    leaf_obs.action_space.actions[offer].description.as_deref(),
                    Some(format!("Name {}", names.get(index).unwrap().name).as_str())
                );
                let leaf_encoded = encode(&leaf_obs, &v9).unwrap();
                assert_eq!(leaf_encoded.name_candidates[4 * offer], (index + 1) as f32);
                branch.execute_semantic_command(&choose).unwrap();
                break;
            }
            let before = branch.clone();
            branch.execute_semantic_command(&choose).unwrap();
            assert_no_naming_effect(&before, &branch);
            // The earlier prompt's Command is stale once the scope narrows.
            assert!(branch.execute_semantic_command(&choose).is_err());
        }
        assert_eq!(path.len(), 2, "928 names need one group level");
        let Decision::NameCard { names: cloned, .. } =
            &game.state.suspended_decision.as_ref().unwrap().decision
        else {
            panic!()
        };
        assert!(Arc::ptr_eq(&names, cloned));
        let Decision::SearchNamed {
            name, remaining, ..
        } = &branch.state.suspended_decision.as_ref().unwrap().decision
        else {
            panic!("named continuation")
        };
        assert_eq!(name, &names.get(index).unwrap().name);
        assert_eq!(*remaining, 4);
        let searched = witness(&branch);
        for stale in &path {
            assert!(branch.execute_semantic_command(stale).is_err());
        }
        assert_eq!(witness(&branch), searched);
        let mut replay = game.clone();
        for step in &path {
            replay.execute_semantic_command(step).unwrap();
        }
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
            < 240
    );
}

#[test]
fn naming_groups_partition_completely_at_every_depth() {
    // A small branching bound over the real 928-name vocabulary forces a deep
    // tree; play uses NAME_BRANCHING. Every name has exactly one path, every
    // prompt is bounded and sibling sizes differ by at most one.
    let names = naming_fixture_vocabulary();
    for branching in [2, 4, 7, NAME_BRANCHING] {
        let mut reached = vec![0usize; names.len()];
        let mut deepest = 0;
        let mut pending = vec![(names.root(), 0usize)];
        while let Some((scope, depth)) = pending.pop() {
            let options = scope.option_count_with(branching);
            assert!(options >= 1 && options <= branching);
            if scope.len() <= branching {
                deepest = deepest.max(depth);
                for count in &mut reached[scope.start..scope.end] {
                    *count += 1;
                }
                continue;
            }
            let children: Vec<NameScope> = (0..options)
                .map(|index| match scope.option_with(index, branching) {
                    Some(NameOption::Group(child)) => child,
                    other => panic!("group prompt offered {other:?}"),
                })
                .collect();
            assert!(scope.option_with(options, branching).is_none());
            assert!(options >= 2, "a group prompt always has a real choice");
            let sizes: Vec<usize> = children.iter().map(NameScope::len).collect();
            assert!(sizes.iter().max().unwrap() - sizes.iter().min().unwrap() <= 1);
            pending.extend(children.into_iter().map(|child| (child, depth + 1)));
        }
        assert!(reached.iter().all(|count| *count == 1));
        // Minimal depth: the smallest d with branching^(d+1) >= len.
        let mut capacity = branching;
        let mut minimal = 0;
        while capacity < names.len() {
            capacity *= branching;
            minimal += 1;
        }
        assert_eq!(deepest, minimal, "branching {branching}");
    }
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
    measure(
        "validated_large_vocabulary_shared_handle",
        naming_fixture_vocabulary,
    );
    let names = naming_fixture_vocabulary();
    let game = naming_game();
    measure("naming_decision", || Decision::NameCard {
        player: PlayerId(0),
        count: 4,
        scope: names.root(),
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
        policy_input_version: 9,
        max_actions: NAME_BRANCHING,
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
