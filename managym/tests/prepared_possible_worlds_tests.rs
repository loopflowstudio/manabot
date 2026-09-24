// prepared_possible_worlds_tests.rs
// Regression proof for bounded possible-world batches over one live root.

use managym::state::{game_object::PlayerId, zone::ZoneType};
use managym::{semantic::SemanticPack, Env, PlayerConfig};

fn authored_configs() -> Vec<PlayerConfig> {
    let semantic = SemanticPack::two_deck().expect("checked-in semantic IR parses");
    vec![
        PlayerConfig::new(
            "UR Lessons",
            semantic.decklist("ur_lessons").expect("UR deck compiles"),
        ),
        PlayerConfig::new(
            "GW Allies",
            semantic.decklist("gw_allies").expect("GW deck compiles"),
        ),
    ]
}

fn authored_env() -> Env {
    let mut env = Env::new(0, true, false, false);
    env.reset(authored_configs())
        .expect("authored match resets");
    env
}

#[test]
fn prepared_batches_enumerate_once_preserve_order_and_fail_closed() {
    let mut root = authored_env();
    let projection = root.possible_world_space(0).expect("canonical space");
    assert!(projection.worlds.len() >= 3);
    assert_eq!(
        projection
            .worlds
            .iter()
            .map(|row| row.index)
            .collect::<Vec<_>>(),
        (0..projection.worlds.len()).collect::<Vec<_>>()
    );
    assert_eq!(
        projection
            .worlds
            .iter()
            .map(|row| row.weight.parse::<u128>().expect("exact weight"))
            .sum::<u128>(),
        projection
            .total_weight
            .parse::<u128>()
            .expect("total weight")
    );

    let scalar = [
        root.materialize_possible_world(0, &projection.identity, 0, 19, false)
            .expect("scalar row zero"),
        root.materialize_possible_world(0, &projection.identity, 1, 23, false)
            .expect("scalar row one"),
        root.materialize_possible_world(0, &projection.identity, 2, 29, false)
            .expect("scalar row two"),
    ];
    let count_before_prepare = root.possible_world_space_construction_count();
    let prepared = root
        .prepare_possible_world_materializer(0, &projection.identity, 2)
        .expect("prepared materializer");
    assert_eq!(prepared.construction_count(), 1);
    assert_eq!(prepared.space_identity(), projection.identity);
    assert_eq!(prepared.support_size(), projection.worlds.len());
    assert_eq!(
        root.possible_world_space_construction_count(),
        count_before_prepare + 1
    );
    let count_after_prepare = root.possible_world_space_construction_count();

    let mut first = prepared
        .materialize_indexes(&root, &[2, 0], &[29, 19], false)
        .expect("non-monotonic batch");
    let second = prepared
        .materialize_indexes(&root, &[1], &[23], false)
        .expect("second batch");
    assert_eq!(
        first[0].state_digest().unwrap(),
        scalar[2].state_digest().unwrap()
    );
    assert_eq!(
        first[1].state_digest().unwrap(),
        scalar[0].state_digest().unwrap()
    );
    assert_eq!(
        second[0].state_digest().unwrap(),
        scalar[1].state_digest().unwrap()
    );
    let repeated = prepared
        .materialize_indexes(&root, &[1], &[23], false)
        .unwrap();
    assert_eq!(
        repeated[0].state_digest().unwrap(),
        scalar[1].state_digest().unwrap()
    );
    assert_eq!(
        root.possible_world_space_construction_count(),
        count_after_prepare,
        "batch calls must not reconstruct the canonical space"
    );

    let root_digest = root.state_digest().unwrap();
    let sibling_digest = first[1].state_digest().unwrap();
    first[0].step(0).expect("isolated sibling advances");
    assert_eq!(root.state_digest().unwrap(), root_digest);
    assert_eq!(first[1].state_digest().unwrap(), sibling_digest);

    for (indexes, seeds, expected) in [
        (vec![], vec![], "must not be empty"),
        (vec![0], vec![], "equal length"),
        (vec![0, 1, 2], vec![1, 2, 3], "exceeds maximum"),
        (
            vec![projection.worlds.len()],
            vec![1],
            "outside support size",
        ),
    ] {
        let error = prepared
            .materialize_indexes(&root, &indexes, &seeds, false)
            .expect_err("invalid batch must fail before returning branches");
        assert!(error.0.contains(expected), "unexpected error: {}", error.0);
    }

    root.step(0).expect("advance the live root");
    let error = prepared
        .materialize_indexes(&root, &[0], &[19], false)
        .expect_err("stale root must fail between batches");
    assert!(error.0.contains("source identity is stale"));
}

#[test]
fn prepared_rejects_invalid_construction_and_changed_hidden_pool() {
    let empty = Env::new(0, true, false, false);
    assert!(empty
        .prepare_possible_world_materializer(0, "identity", 1)
        .unwrap_err()
        .0
        .contains("before reset"));
    let root = authored_env();
    let projection = root.possible_world_space(0).unwrap();
    for (viewer, identity, maximum, message) in [
        (2, projection.identity.as_str(), 1, "out of range"),
        (0, "", 1, "must not be empty"),
        (0, projection.identity.as_str(), 0, "must be positive"),
        (0, "wrong", 1, "identity mismatch"),
    ] {
        assert!(root
            .prepare_possible_world_materializer(viewer, identity, maximum)
            .unwrap_err()
            .0
            .contains(message));
    }
    let root = managym::Game::new(authored_configs(), 0, true);
    let space = managym::possible_worlds::PossibleWorldSpace::for_viewer(&root, PlayerId(0));
    let mut changed = root.clone();
    let card = changed
        .state
        .zones
        .zone_cards(ZoneType::Library, PlayerId(1))[0];
    changed.state.cards[card].name = "changed hidden pool".into();
    // The viewer cannot see this change; only the additional pool check catches it.
    assert_eq!(
        serde_json::to_value(changed.semantic_observation(PlayerId(0)).unwrap()).unwrap(),
        serde_json::to_value(root.semantic_observation(PlayerId(0)).unwrap()).unwrap()
    );
    assert!(matches!(
        space.materialize_indexes(
            &changed,
            &[0],
            &[1],
            managym::possible_worlds::MaterializeMode::PreserveViewerRoot,
            1
        ),
        Err(managym::possible_worlds::MaterializeError::StaleSource)
    ));

    let mut changed = root.clone();
    changed.state.players[0].life -= 1;
    assert!(matches!(
        space.materialize_indexes(
            &changed,
            &[0],
            &[1],
            managym::possible_worlds::MaterializeMode::PreserveViewerRoot,
            1
        ),
        Err(managym::possible_worlds::MaterializeError::StaleSource)
    ));
}

#[test]
fn prepared_matches_scalar_for_both_modes_and_deterministic_seeds() {
    let root = authored_env();
    let projection = root.possible_world_space(1).unwrap();
    let prepared = root
        .prepare_possible_world_materializer(1, &projection.identity, 2)
        .unwrap();
    for refresh in [false, true] {
        let branches = prepared
            .materialize_indexes(&root, &[1, 0], &[19, 23], refresh)
            .unwrap();
        for (branch, (index, seed)) in branches.iter().zip([(1, 19), (0, 23)]) {
            let scalar = root
                .materialize_possible_world(1, &projection.identity, index, seed, refresh)
                .unwrap();
            assert_eq!(
                branch.state_digest().unwrap(),
                scalar.state_digest().unwrap()
            );
            assert_eq!(
                serde_json::to_value(branch.semantic_observation(1).unwrap()).unwrap(),
                serde_json::to_value(scalar.semantic_observation(1).unwrap()).unwrap()
            );
            assert_eq!(
                serde_json::to_value(branch.semantic_decision_frame().unwrap()).unwrap(),
                serde_json::to_value(scalar.semantic_decision_frame().unwrap()).unwrap()
            );
            assert_eq!(
                branch.semantic_event_cursor().unwrap(),
                scalar.semantic_event_cursor().unwrap()
            );
        }
    }
    let seeds = prepared
        .materialize_indexes(&root, &[0, 0], &[19, 20], true)
        .unwrap();
    assert_ne!(
        seeds[0].state_digest().unwrap(),
        seeds[1].state_digest().unwrap()
    );
    assert_eq!(
        serde_json::to_value(seeds[0].semantic_observation(1).unwrap()).unwrap(),
        serde_json::to_value(seeds[1].semantic_observation(1).unwrap()).unwrap()
    );
}
