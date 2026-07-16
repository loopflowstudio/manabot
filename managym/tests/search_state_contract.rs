use std::sync::Arc;

use managym::{
    agent::observation::Observation,
    benchmark::build_fixture,
    search_state::{
        verify_branch_contract, verify_seeded_trace_equivalence, BranchDriver, ClonePlusUndoDriver,
        FullCloneDriver,
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

/// The undo driver must satisfy the identical, unchanged contract on both
/// fixtures and produce byte-identical receipts to the full-clone reference.
/// Equal receipts are the load-bearing claim: a rollback that restored only the
/// fields one fixture happens to touch would diverge here.
#[test]
fn clone_plus_undo_matches_the_full_clone_reference_receipts() {
    for fixture in ["interactive-midgame-48-v1", "interactive-heavy-80-v1"] {
        for trace_seed in [0x1975eed_u64, 0x5eed, 0xc10e] {
            let (root, _) = build_fixture(fixture).expect("contract fixture");
            let viewer = root.agent_player();
            let reference =
                verify_branch_contract(&FullCloneDriver, &root, viewer, trace_seed, 2_000)
                    .unwrap_or_else(|error| {
                        panic!("full clone {fixture}/{trace_seed:#x}: {error}")
                    });
            let undo = verify_branch_contract(
                &ClonePlusUndoDriver::default(),
                &root,
                viewer,
                trace_seed,
                2_000,
            )
            .unwrap_or_else(|error| panic!("clone plus undo {fixture}/{trace_seed:#x}: {error}"));

            assert_eq!(
                reference, undo,
                "{fixture}/{trace_seed:#x} receipts must be identical"
            );
            assert_ne!(undo.root_hash, undo.nested_hash);
            assert!(undo.replay_steps > 0);
        }
    }
}

/// Walk deep into a game and, at every decision, exhaustively mark/apply/
/// rollback each legal action to a nested depth, asserting the state returns
/// exactly to the pre-mark witness that the full-clone reference produces.
///
/// The two-level contract only nests from the root, where the fixture sits on a
/// plain priority decision. This drives combat, targeting, triggers, and token
/// creation far from the root, so an unjournaled field on any of those paths
/// surfaces here rather than as benchmark checksum drift.
#[test]
fn clone_plus_undo_rolls_back_exactly_at_every_depth_along_a_deep_trace() {
    fn probe(driver: &ClonePlusUndoDriver, branch: &mut managym::Game, depth: usize, seed: u64) {
        if depth == 0 {
            return;
        }
        let before = driver.witness(branch);
        if before.diagnostics.terminal {
            return;
        }
        for action_index in 0..before.legal_surface.action_count {
            let mark = driver.mark(branch);
            let applied = driver.apply(
                branch,
                managym::search_state::BenchCommand {
                    action_index,
                    expected_state_hash: Some(before.authority.hash.clone()),
                    expected_action_hash: Some(before.legal_surface.hash.clone()),
                },
            );
            if applied.is_ok() {
                probe(driver, branch, depth - 1, mix(seed, action_index as u64));
            }
            driver.rollback(branch, mark);
            assert_eq!(
                driver.witness(branch),
                before,
                "depth {depth} action {action_index} did not roll back exactly"
            );
        }
    }

    fn mix(seed: u64, value: u64) -> u64 {
        let mut hash = seed ^ value.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        hash ^= hash >> 33;
        hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
        hash ^= hash >> 33;
        hash
    }

    for fixture in ["interactive-midgame-48-v1", "interactive-heavy-80-v1"] {
        let (root, _) = build_fixture(fixture).expect("deep trace fixture");
        let driver = ClonePlusUndoDriver::default();
        let reference = FullCloneDriver;
        let mut branch = driver.fork_exact(&root);
        let mut ground_truth = reference.fork_exact(&root);
        let mut seed = 0x51a7e_u64;

        // Advance along a shared trace, probing rollback at each decision.
        for step in 0..40 {
            let here = driver.witness(&branch);
            assert_eq!(
                here,
                reference.witness(&ground_truth),
                "{fixture} diverged from the full-clone reference at step {step}"
            );
            if here.diagnostics.terminal || here.legal_surface.action_count == 0 {
                break;
            }

            probe(&driver, &mut branch, 3, seed);
            assert_eq!(
                driver.witness(&branch),
                here,
                "{fixture} probing perturbed the trace at step {step}"
            );

            seed = mix(seed, step);
            let action_index = (seed as usize) % here.legal_surface.action_count;
            let command = managym::search_state::BenchCommand {
                action_index,
                expected_state_hash: Some(here.authority.hash.clone()),
                expected_action_hash: Some(here.legal_surface.hash.clone()),
            };
            driver
                .apply(&mut branch, command.clone())
                .expect("undo driver advances the trace");
            reference
                .apply(&mut ground_truth, command)
                .expect("reference advances the trace");
        }
    }
}

/// Rollback must reverse token allocation: a token push grows `cards`,
/// `permanents`, `card_to_permanent`, `object_incarnations`, and the ID
/// watermark. A driver that only journals overwritten combat scalars leaves
/// the appended slots and the bumped watermark behind and fails here.
#[test]
fn clone_plus_undo_rolls_back_token_allocation_and_the_id_watermark() {
    let (root, _) = build_fixture("interactive-heavy-80-v1").expect("heavy allocation fixture");
    let driver = ClonePlusUndoDriver::default();
    let root_witness = driver.witness(&root);
    let mut branch = driver.fork_exact(&root);

    let cards_before = branch.state.cards.len();
    let permanents_before = branch.state.permanents.len();
    let watermark_before = branch.state.id_gen.watermark();

    let mark = driver.mark(&mut branch);
    let controller = branch.agent_player();
    // Ally and Clue are the admitted pack's real token definitions; passing a
    // nontoken card name trips a debug-only assertion in `create_token`.
    branch.create_token("Ally", controller, false);
    branch.create_token("Clue", controller, false);

    assert!(
        branch.state.cards.len() > cards_before,
        "tokens must allocate"
    );
    assert!(branch.state.permanents.len() > permanents_before);
    assert!(branch.state.id_gen.watermark() > watermark_before);
    assert_ne!(driver.witness(&branch), root_witness);

    driver.rollback(&mut branch, mark);

    assert_eq!(
        branch.state.cards.len(),
        cards_before,
        "cards must truncate"
    );
    assert_eq!(branch.state.permanents.len(), permanents_before);
    assert_eq!(
        branch.state.id_gen.watermark(),
        watermark_before,
        "the allocation watermark must roll back so re-forked IDs are reused"
    );
    assert_eq!(driver.witness(&branch), root_witness);
}

/// Rollback must restore zone order and the reverse card->zone index, not just
/// zone membership counts. Shuffling and moving cards perturbs both.
#[test]
fn clone_plus_undo_rolls_back_zone_order_and_reverse_indices() {
    let (root, _) = build_fixture("interactive-heavy-80-v1").expect("heavy zone fixture");
    let driver = ClonePlusUndoDriver::default();
    let root_witness = driver.witness(&root);
    let mut branch = driver.fork_exact(&root);
    let mark = driver.mark(&mut branch);

    let player = branch.agent_player();
    let library_before = branch
        .state
        .zones
        .zone_cards(ZoneType::Library, player)
        .to_vec();
    assert!(
        library_before.len() > 2,
        "fixture needs a nontrivial library"
    );

    branch.draw_cards(player, 3);
    let library_after = branch
        .state
        .zones
        .zone_cards(ZoneType::Library, player)
        .to_vec();
    assert_ne!(
        library_before, library_after,
        "drawing must change zone order"
    );
    assert_ne!(driver.witness(&branch), root_witness);

    driver.rollback(&mut branch, mark);

    assert_eq!(
        branch.state.zones.zone_cards(ZoneType::Library, player),
        library_before.as_slice(),
        "library order must be restored exactly, not just its size"
    );
    for card in &library_before {
        assert_eq!(
            branch.state.zones.zone_of(*card),
            Some(ZoneType::Library),
            "the reverse card->zone index must roll back"
        );
    }
    assert_eq!(driver.witness(&branch), root_witness);
}

/// Sibling sequential branches must be exactly isolated: rolling back one
/// branch may not perturb a fork taken from the same root, and the surfaced
/// action space must be restored along with the rules state.
#[test]
fn clone_plus_undo_isolates_siblings_and_restores_the_action_space() {
    let (root, _) = build_fixture("interactive-midgame-48-v1").expect("midgame sibling fixture");
    let driver = ClonePlusUndoDriver::default();
    let root_witness = driver.witness(&root);
    let action_count = root_witness.legal_surface.action_count;
    assert!(
        action_count > 1,
        "fixture needs sibling-distinguishing actions"
    );

    let mut branch = driver.fork_exact(&root);
    let sibling = driver.fork_exact(&root);

    // Walk every legal action forward and back; each must land on the root.
    for action_index in 0..action_count {
        let mark = driver.mark(&mut branch);
        driver
            .apply(
                &mut branch,
                managym::search_state::BenchCommand {
                    action_index,
                    expected_state_hash: Some(root_witness.authority.hash.clone()),
                    expected_action_hash: Some(root_witness.legal_surface.hash.clone()),
                },
            )
            .expect("legal action applies");
        driver.rollback(&mut branch, mark);

        assert_eq!(
            driver.witness(&branch),
            root_witness,
            "action {action_index} did not roll back exactly"
        );
        assert_eq!(
            driver.witness(&sibling),
            root_witness,
            "action {action_index} perturbed a sibling fork"
        );
        assert_eq!(driver.witness(&root), root_witness, "root was perturbed");
    }
}

#[test]
fn independently_allocated_roots_have_equal_witnesses_and_seeded_traces() {
    let (mut first, _) = build_fixture("interactive-heavy-80-v1").expect("first heavy fixture");
    let (mut second, _) = build_fixture("interactive-heavy-80-v1").expect("second heavy fixture");
    second.state.content = Arc::new((*second.state.content).clone());
    assert!(
        !Arc::ptr_eq(&first.state.content, &second.state.content),
        "the proof needs independently allocated immutable definition packs"
    );

    let driver = FullCloneDriver;
    assert_eq!(driver.witness(&first), driver.witness(&second));
    let viewer = first.agent_player();
    let receipt =
        verify_seeded_trace_equivalence(&driver, &mut first, &mut second, viewer, 0x197c0de, 2_000)
            .expect("equivalent roots must remain equal through a seeded trace");

    assert!(receipt.compared_steps > 0);
    assert_eq!(driver.witness(&first), driver.witness(&second));
    assert_eq!(receipt.final_hash, driver.witness(&first).authority.hash);
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
    let first_witness = driver.witness(&first);

    let second_witness = (2_u64..64)
        .find_map(|seed| {
            let mut candidate = driver.fork_exact(&root);
            driver.determinize(&mut candidate, viewer, seed);
            let witness = driver.witness(&candidate);
            (witness.authority.hash != first_witness.authority.hash
                && witness.viewers[viewer.0] == first_witness.viewers[viewer.0]
                && witness.viewers[opponent.0] != first_witness.viewers[opponent.0])
                .then_some(witness)
        })
        .expect("fixed seeds must produce a distinct opponent hidden hand");

    assert_ne!(first_witness.authority.hash, second_witness.authority.hash);
    assert_eq!(
        first_witness.viewers[viewer.0].hash,
        second_witness.viewers[viewer.0].hash
    );
    assert_ne!(
        first_witness.viewers[opponent.0].hash,
        second_witness.viewers[opponent.0].hash
    );
    for projection in &first_witness.viewers {
        let json = std::str::from_utf8(&projection.bytes).expect("projection JSON is UTF-8");
        assert!(!json.contains("semantic_hash"));
        assert!(!json.contains(&first_witness.authority.hash));
    }
}

#[test]
fn stale_object_refs_are_rejected_and_identity_rolls_back_exactly() {
    let (root, _) = build_fixture("interactive-heavy-80-v1").expect("heavy identity fixture");
    let driver = FullCloneDriver;
    let root_witness = driver.witness(&root);
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
    assert_ne!(driver.witness(&branch), root_witness);

    driver.rollback(&mut branch, mark);
    assert_eq!(driver.witness(&branch), root_witness);
    assert_eq!(branch.lookup_current_permanent(old_ref), Ok(permanent));
    assert_eq!(
        branch.lookup_current_permanent(new_ref),
        Err(ObjectLookupError::StaleIncarnation)
    );
}
