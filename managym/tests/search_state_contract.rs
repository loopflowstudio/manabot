use std::sync::Arc;

use managym::{
    agent::observation::Observation,
    benchmark::build_fixture,
    search_state::{
        verify_branch_contract, verify_seeded_trace_equivalence, BranchDriver, FullCloneDriver,
    },
    state::{
        game_object::{ObjectLookupError, PermanentId, PlayerId},
        zone::ZoneType,
    },
};

#[test]
fn full_clone_passes_the_representation_neutral_branch_contract() {
    let (root, _) = build_fixture("interactive-midgame-48-v1").expect("midgame contract fixture");
    let viewer = root.agent_player();
    let receipt = verify_branch_contract(&FullCloneDriver, &root, viewer, 0x1975eed, 2_000)
        .expect("full-clone reference must pass the branch contract");

    assert_ne!(receipt.root_hash, receipt.nested_hash);
    assert!(!receipt.replay_final_hash.is_empty());
    assert!(receipt.replay_steps > 0);
}

#[test]
fn independently_allocated_roots_have_equal_snapshots_and_seeded_traces() {
    let (mut first, _) = build_fixture("interactive-heavy-80-v1").expect("first heavy fixture");
    let (mut second, _) = build_fixture("interactive-heavy-80-v1").expect("second heavy fixture");
    second.state.content = Arc::new((*second.state.content).clone());
    assert!(
        !Arc::ptr_eq(&first.state.content, &second.state.content),
        "the proof needs independently allocated immutable definition packs"
    );

    let driver = FullCloneDriver;
    assert_eq!(driver.snapshot(&first), driver.snapshot(&second));
    let viewer = first.agent_player();
    let receipt =
        verify_seeded_trace_equivalence(&driver, &mut first, &mut second, viewer, 0x197c0de, 2_000)
            .expect("equivalent roots must remain equal through a seeded trace");

    assert!(receipt.compared_steps > 0);
    assert_eq!(driver.snapshot(&first), driver.snapshot(&second));
    assert_eq!(receipt.final_hash, driver.snapshot(&first).hash);
}

#[test]
fn fixed_viewer_projections_hide_opposing_hands_and_private_choices() {
    let (root, _) = build_fixture("interactive-midgame-48-v1").expect("midgame visibility fixture");
    let actor = root.agent_player();
    let other = PlayerId((actor.0 + 1) % 2);

    let actor_view = Observation::for_player(&root, actor);
    let other_view = Observation::for_player(&root, other);
    assert_eq!(
        actor_view.action_space.actions.len(),
        root.action_space()
            .expect("root action space")
            .actions
            .len()
    );
    assert!(other_view.action_space.actions.is_empty());
    assert!(other_view.action_space.focus.is_empty());

    for viewer in [PlayerId(0), PlayerId(1)] {
        let projection = Observation::for_player(&root, viewer);
        assert!(projection
            .agent_cards
            .iter()
            .any(|card| card.zone == ZoneType::Hand));
        assert!(projection
            .opponent_cards
            .iter()
            .all(|card| { card.zone != ZoneType::Hand && card.zone != ZoneType::Library }));
    }
}

#[test]
fn hidden_worlds_change_authority_without_changing_the_viewers_known_state() {
    let (root, _) =
        build_fixture("interactive-midgame-48-v1").expect("midgame hidden-world fixture");
    let driver = FullCloneDriver;
    let viewer = PlayerId(0);
    let opponent = PlayerId(1);
    let mut first = driver.fork_exact(&root);
    driver.determinize(&mut first, viewer, 1);
    let first_snapshot = driver.snapshot(&first);

    let second_snapshot = (2_u64..64)
        .find_map(|seed| {
            let mut candidate = driver.fork_exact(&root);
            driver.determinize(&mut candidate, viewer, seed);
            let snapshot = driver.snapshot(&candidate);
            (snapshot.hash != first_snapshot.hash
                && snapshot.viewer_projections[viewer.0]
                    == first_snapshot.viewer_projections[viewer.0]
                && snapshot.viewer_projections[opponent.0]
                    != first_snapshot.viewer_projections[opponent.0])
                .then_some(snapshot)
        })
        .expect("fixed seeds must produce a distinct opponent hidden hand");

    assert_ne!(first_snapshot.hash, second_snapshot.hash);
    assert_eq!(
        first_snapshot.viewer_hashes[viewer.0],
        second_snapshot.viewer_hashes[viewer.0]
    );
    assert_ne!(
        first_snapshot.viewer_hashes[opponent.0],
        second_snapshot.viewer_hashes[opponent.0]
    );
    for projection in &first_snapshot.viewer_projections {
        let json = std::str::from_utf8(projection).expect("projection JSON is UTF-8");
        assert!(!json.contains("semantic_hash"));
        assert!(!json.contains(&first_snapshot.hash));
    }
}

#[test]
fn stale_object_refs_are_rejected_and_identity_rolls_back_exactly() {
    let (root, _) = build_fixture("interactive-heavy-80-v1").expect("heavy identity fixture");
    let driver = FullCloneDriver;
    let root_snapshot = driver.snapshot(&root);
    let permanent = PermanentId(
        root.state
            .permanents
            .iter()
            .position(Option::is_some)
            .expect("fixture has a live permanent"),
    );
    let card = root.state.permanents[permanent]
        .as_ref()
        .expect("live permanent")
        .card;
    let old_ref = root.current_object_ref(card).expect("current object ref");
    assert_eq!(root.lookup_current_permanent(old_ref), Ok(permanent));

    let mut branch = driver.fork_exact(&root);
    let mark = driver.mark(&mut branch);
    branch.move_card(card, ZoneType::Graveyard);
    assert_eq!(
        branch.lookup_current_permanent(old_ref),
        Err(ObjectLookupError::StaleIncarnation)
    );
    assert_eq!(
        branch
            .object_lki(old_ref)
            .expect("departing object has LKI")
            .object_ref,
        old_ref
    );

    branch.move_card(card, ZoneType::Battlefield);
    let new_ref = branch
        .current_object_ref(card)
        .expect("re-entered object ref");
    assert_eq!(new_ref.entity, old_ref.entity);
    assert!(new_ref.incarnation > old_ref.incarnation);
    assert_eq!(
        branch.lookup_current_permanent(old_ref),
        Err(ObjectLookupError::StaleIncarnation)
    );
    assert!(branch.lookup_current_permanent(new_ref).is_ok());
    assert_ne!(driver.snapshot(&branch), root_snapshot);

    driver.rollback(&mut branch, mark);
    assert_eq!(driver.snapshot(&branch), root_snapshot);
    assert_eq!(branch.lookup_current_permanent(old_ref), Ok(permanent));
    assert_eq!(
        branch.lookup_current_permanent(new_ref),
        Err(ObjectLookupError::StaleIncarnation)
    );
}
