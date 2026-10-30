//! Tests for cards filed in `src/cardsets/stx.rs`, in card order.

use crate::support::{
    game::{clear_hand, deck, five_color_setup, force_lands, Scenario},
    inspect::{hand, library, library_card, permanent, stack_spell_card},
    play::{cast, cast_and_resolve, cast_only, pass_until, settle},
};
use managym::{
    agent::{
        action::{Action, ActionSpaceKind, ActionType},
        observation::Observation,
        structured_offer::PublicCommitment,
    },
    decision::Command,
    flow::turn::StepKind,
    possible_worlds::{viewer_observation, MaterializeMode, PossibleWorldSpace},
    state::{
        game_object::{PermanentId, PlayerId, Target},
        zone::ZoneType,
    },
};
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn divide_by_zero_bounces_spell_from_stack() {
    let mut s = Scenario::new(
        deck(&[("Island", 24), ("Divide by Zero", 16)]),
        deck(&[("Mountain", 24), ("Gray Ogre", 16)]),
        61,
    );
    force_lands(&mut s, 0, "Island", 3);
    force_lands(&mut s, 1, "Mountain", 3);
    clear_hand(&mut s, 0);
    clear_hand(&mut s, 1);
    s.force_card_in_hand(0, "Divide by Zero");
    s.force_card_in_hand(1, "Gray Ogre");
    s.advance_to_active_step(1, StepKind::Main);
    cast_only(&mut s); // P1 casts Gray Ogre
    s.pass_priority();
    cast_only(&mut s); // P0 responds with Divide by Zero
    let ogre = stack_spell_card(&s, "Gray Ogre");
    assert!(s.choose_target(Target::StackSpell(ogre)));
    s.pass_priority();
    s.pass_priority();

    // The Ogre spell was bounced to its owner's hand, not countered.
    assert_eq!(s.game().state.zones.zone_of(ogre), Some(ZoneType::Hand));
    assert_eq!(s.zone_size(1, ZoneType::Graveyard), 0);
    // Learn rider.
    assert_eq!(s.action_space().kind, ActionSpaceKind::Learn);
    assert!(s.take_action_by_type(ActionType::DeclineChoice));
    assert_eq!(s.game().state.stack_objects.len(), 0);
}

#[test]
fn divide_by_zero_bounces_permanent_but_not_lands() {
    let mut s = Scenario::new(
        deck(&[("Island", 24), ("Divide by Zero", 16)]),
        deck(&[("Mountain", 24), ("Gray Ogre", 16)]),
        62,
    );
    force_lands(&mut s, 0, "Island", 3);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    clear_hand(&mut s, 0);
    clear_hand(&mut s, 1);
    s.force_card_in_hand(0, "Divide by Zero");
    s.advance_to_active_step(0, StepKind::Main);
    cast_only(&mut s);

    // Lands (mana value 0) are not legal targets; the Ogre is.
    let space = s.action_space().clone();
    assert_eq!(space.kind, ActionSpaceKind::ChooseTarget);
    assert_eq!(space.actions.len(), 1, "only the Gray Ogre is targetable");
    assert!(s.choose_target(Target::Permanent(ogre)));
    s.pass_priority();
    s.pass_priority();
    assert_eq!(s.battlefield_permanents_named(1, "Gray Ogre").len(), 0);
    assert_eq!(s.zone_size(1, ZoneType::Hand), 1);
    assert!(s.take_action_by_type(ActionType::DeclineChoice));
}

#[test]
fn targeted_combat_tricks_apply_through_play() {
    for name in ["Enthusiastic Study", "Octopus Form", "Fists of Flame"] {
        let mut s = five_color_setup(&[name, "Gnarled Professor"]);
        let target = s.force_permanent_on_battlefield(0, "Gnarled Professor");
        cast(&mut s, name);
        assert!(s.choose_target(Target::Permanent(target)));
        settle(&mut s);
        let data = Observation::for_player(s.game(), PlayerId(0));
        let p = data
            .agent_permanents
            .iter()
            .find(|p| p.id == s.game().state.permanents[target].as_ref().unwrap().id.0 as i32)
            .unwrap();
        if name == "Octopus Form" {
            assert!(p.keywords.hexproof);
            assert_eq!(s.game().effective_power(target), 6);
        } else {
            assert!(p.keywords.trample);
            assert!(s.game().effective_power(target) > 5);
        }
        if name == "Enthusiastic Study" {
            assert_eq!(s.game().effective_power(target), 8);
        }
    }
}

#[test]
fn eyetwitch_and_poxbearer_deaths_keep_their_different_rewards() {
    for name in ["Eyetwitch", "Pretending Poxbearers", "Knowledge Seeker"] {
        let mut s = five_color_setup(&[name, "Lightning Strike"]);
        let target = s.force_permanent_on_battlefield(0, name);
        cast(&mut s, "Lightning Strike");
        assert!(s.choose_target(Target::Permanent(target)));
        settle(&mut s);
        assert!(s.game().state.permanents[target].is_none());
        match name {
            "Pretending Poxbearers" => {
                assert_eq!(s.battlefield_permanents_named(0, "Ally").len(), 1)
            }
            "Knowledge Seeker" => {
                assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 1)
            }
            _ => assert_eq!(s.battlefield_permanents_named(0, "Ally").len(), 0),
        }
    }
}

#[test]
fn library_search_is_private_enters_tapped_and_continues_to_learn() {
    for (name, forest) in [("Field Trip", true), ("Shared Roots", false)] {
        let mut s = five_color_setup(&[name]);
        cast(&mut s, name);
        s.pass_priority();
        s.pass_priority();
        assert_eq!(s.action_space().kind, ActionSpaceKind::LookAndSelect);
        let owner = Observation::for_player(s.game(), PlayerId(0));
        let other = Observation::for_player(s.game(), PlayerId(1));
        assert!(owner
            .agent_cards
            .iter()
            .any(|c| c.zone == ZoneType::Library));
        assert!(other
            .opponent_cards
            .iter()
            .all(|c| c.zone != ZoneType::Library));
        let selected = s
            .action_space()
            .actions
            .iter()
            .find_map(|a| {
                if let Action::SelectCard { card, .. } = a {
                    Some(*card)
                } else {
                    None
                }
            })
            .unwrap();
        if forest {
            assert_eq!(s.game().state.cards[selected].name, "Forest");
        }
        let index = s
            .action_space()
            .actions
            .iter()
            .position(|a| matches!(a,Action::SelectCard {card,..} if *card==selected))
            .unwrap();
        s.step_action(index);
        let id = s.game().state.card_to_permanent[selected].unwrap();
        assert!(s.game().state.permanents[id].as_ref().unwrap().tapped);
        assert_eq!(
            s.action_space().kind,
            if forest {
                ActionSpaceKind::Learn
            } else {
                ActionSpaceKind::Priority
            }
        );
        settle(&mut s);
    }
}

#[test]
fn search_selection_or_decline_and_shuffle_restore_exactly_from_fork() {
    use managym::search_state::witness;
    let mut s = five_color_setup(&["Field Trip"]);
    cast(&mut s, "Field Trip");
    s.pass_priority();
    s.pass_priority();
    for select in [false, true] {
        let root = witness(s.game());
        let mut branch = s.game().clone();
        let mark = branch.clone();
        let index = branch
            .action_space()
            .unwrap()
            .actions
            .iter()
            .position(|a| {
                if select {
                    matches!(a, Action::SelectCard { .. })
                } else {
                    matches!(a, Action::Decline { .. })
                }
            })
            .unwrap();
        branch.step(index).unwrap();
        assert_ne!(witness(&branch), root);
        branch = mark;
        assert_eq!(witness(&branch), root);
        assert_eq!(witness(s.game()), root);
    }
}

#[test]
fn resampled_private_search_choices_still_name_library_cards() {
    let mut s = five_color_setup(&["Field Trip"]);
    cast(&mut s, "Field Trip");
    s.pass_priority();
    s.pass_priority();
    for viewer in [PlayerId(0), PlayerId(1)] {
        for seed in 0..5 {
            let mut branch = s.game().clone();
            branch.determinize(viewer, seed);
            let expected: BTreeSet<_> = branch
                .state
                .zones
                .zone_cards(ZoneType::Library, PlayerId(0))
                .iter()
                .filter(|card| branch.state.cards[**card].name == "Forest")
                .map(|card| branch.state.cards[*card].id.0 as i32)
                .collect();
            let offered: BTreeSet<_> = branch
                .action_space()
                .unwrap()
                .actions
                .iter()
                .filter_map(|action| match action {
                    Action::SelectCard { card, .. } => Some(branch.state.cards[*card].id.0 as i32),
                    _ => None,
                })
                .collect();
            assert_eq!(offered, expected);
            let owner = Observation::for_player(&branch, PlayerId(0));
            let visible = owner
                .agent_cards
                .iter()
                .filter(|card| card.zone == ZoneType::Library)
                .map(|card| card.id)
                .collect::<BTreeSet<_>>();
            assert_eq!(visible, expected);
        }
    }
}

#[test]
fn library_search_semantic_command_replays_and_rejects_reuse_without_mutation() {
    use managym::decision::Command;
    use managym::search_state::witness;
    let mut s = five_color_setup(&["Field Trip"]);
    cast(&mut s, "Field Trip");
    s.pass_priority();
    s.pass_priority();
    let mut replay = s.game().clone();
    let frame = s.game().semantic_decision_frame().unwrap();
    let command = Command {
        command_id: "elemental-search".into(),
        expected_revision: frame.revision,
        offer_id: frame.offers[0].id.0,
        answers: vec![],
        object_preconditions: vec![],
    };
    s.game_mut().execute_semantic_command(&command).unwrap();
    replay.execute_semantic_command(&command).unwrap();
    assert_eq!(witness(s.game()), witness(&replay));
    let after = witness(s.game());
    assert!(s.game_mut().execute_semantic_command(&command).is_err());
    assert_eq!(witness(s.game()), after);
}

#[test]
fn search_possible_worlds_refresh_private_candidates_without_exposing_library_cards() {
    use managym::possible_worlds::{MaterializeMode, PossibleWorldSpace};
    let mut s = five_color_setup(&["Field Trip"]);
    let hand = library_card(&s, 0, "Forest");
    s.game_mut().move_card(hand, ZoneType::Hand);
    cast(&mut s, "Field Trip");
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    let space = PossibleWorldSpace::for_viewer(s.game(), PlayerId(1));
    for index in 0..space.worlds().len() {
        let branch = space
            .materialize_index(
                s.game(),
                index,
                161,
                MaterializeMode::RefreshOpponentCommitment,
            )
            .unwrap();
        let observed = Observation::for_player(&branch, PlayerId(1));
        assert!(observed
            .opponent_cards
            .iter()
            .all(|card| card.zone != ZoneType::Library && card.zone != ZoneType::Hand));
        for action in &branch.action_space().unwrap().actions {
            if let Action::SelectCard { card, .. } = action {
                assert_eq!(branch.state.zones.zone_of(*card), Some(ZoneType::Library));
                assert_eq!(branch.state.cards[*card].name, "Forest");
            }
        }
    }
}

#[test]
fn igneous_inspiration_deals_three_then_learns() {
    let mut s = Scenario::new(
        deck(&[("Mountain", 24), ("Igneous Inspiration", 16)]),
        deck(&[("Plains", 40)]),
        53,
    );
    force_lands(&mut s, 0, "Mountain", 3);
    clear_hand(&mut s, 0);
    s.force_card_in_hand(0, "Igneous Inspiration");
    s.advance_to_active_step(0, StepKind::Main);
    cast_only(&mut s);
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    s.pass_priority();
    s.pass_priority();
    s.assert_life(1, 17);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Learn);
    assert!(s.take_action_by_type(ActionType::DeclineChoice));
}

#[test]
fn pop_quiz_draws_then_learn_discard_draws() {
    let mut s = Scenario::new(
        deck(&[("Island", 24), ("Pop Quiz", 16)]),
        deck(&[("Plains", 40)]),
        51,
    );
    force_lands(&mut s, 0, "Island", 3);
    clear_hand(&mut s, 0);
    s.force_card_in_hand(0, "Pop Quiz");
    s.advance_to_active_step(0, StepKind::Main);
    cast_and_resolve(&mut s);

    // "Draw a card" resolved, then the learn decision surfaces.
    let space = s.action_space();
    assert_eq!(space.kind, ActionSpaceKind::Learn);
    assert_eq!(space.player, Some(PlayerId(0)));
    // One action per hand card plus a decline.
    let hand_size = hand(&s, 0).len();
    assert_eq!(space.actions.len(), hand_size + 1);
    assert_eq!(
        space.actions.last().map(Action::action_type),
        Some(ActionType::DeclineChoice)
    );

    assert!(s.take_action_by_type(ActionType::LearnDiscard));
    // Discarded one, drew one: hand size unchanged; graveyard has the
    // discarded card + Pop Quiz.
    assert_eq!(hand(&s, 0).len(), hand_size);
    assert_eq!(s.zone_size(0, ZoneType::Graveyard), 2);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
}

#[test]
fn pop_quiz_learn_can_be_declined() {
    let mut s = Scenario::new(
        deck(&[("Island", 24), ("Pop Quiz", 16)]),
        deck(&[("Plains", 40)]),
        52,
    );
    force_lands(&mut s, 0, "Island", 3);
    clear_hand(&mut s, 0);
    s.force_card_in_hand(0, "Pop Quiz");
    s.advance_to_active_step(0, StepKind::Main);
    cast_and_resolve(&mut s);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Learn);
    let hand_size = hand(&s, 0).len();
    assert!(s.take_action_by_type(ActionType::DeclineChoice));
    assert_eq!(hand(&s, 0).len(), hand_size, "no discard, no draw");
    assert_eq!(s.zone_size(0, ZoneType::Graveyard), 1); // just Pop Quiz
}

#[test]
fn discard_then_draw_public_commitment_is_viewer_safe_and_authoritative() {
    let mut s = Scenario::new(
        deck(&[("Island", 24), ("Pop Quiz", 16)]),
        deck(&[("Plains", 40)]),
        52,
    );
    force_lands(&mut s, 0, "Island", 3);
    clear_hand(&mut s, 0);
    s.force_card_in_hand(0, "Pop Quiz");
    s.advance_to_active_step(0, StepKind::Main);
    cast_and_resolve(&mut s);

    let root = s.game().clone();
    let frame = root.semantic_decision_frame().expect("discard frame");
    let non_actor = root
        .semantic_observation(PlayerId(1))
        .expect("non-actor Observation");
    assert!(non_actor.decision.is_none());
    assert!(non_actor.viewer_state["opponent_cards"]
        .as_array()
        .expect("opponent cards")
        .iter()
        .all(|card| card["zone"].as_i64() != Some(ZoneType::Hand as i64)));

    let (discard_id, discarded_card) = frame
        .offers
        .iter()
        .find_map(|offer| match &offer.public_commitment {
            Some(PublicCommitment::Discard { card }) => Some((offer.id.0, card.clone())),
            _ => None,
        })
        .expect("one hand card has a discard commitment");
    let decline_id = frame
        .offers
        .iter()
        .find(|offer| offer.public_commitment == Some(PublicCommitment::DeclineLearn))
        .expect("decline commitment")
        .id
        .0;

    let mut discard_branch = root.clone();
    let discard = discard_branch
        .execute_semantic_command(&Command {
            command_id: "public-commitment-discard".to_string(),
            expected_revision: frame.revision,
            offer_id: discard_id,
            answers: Vec::new(),
            object_preconditions: Vec::new(),
        })
        .expect("discard command executes");
    assert_eq!(
        discard.receipt.public_commitment,
        Some(PublicCommitment::Discard {
            card: discarded_card
        })
    );

    let mut decline_branch = root;
    let decline = decline_branch
        .execute_semantic_command(&Command {
            command_id: "public-commitment-decline".to_string(),
            expected_revision: frame.revision,
            offer_id: decline_id,
            answers: Vec::new(),
            object_preconditions: Vec::new(),
        })
        .expect("decline command executes");
    assert_eq!(
        decline.receipt.public_commitment,
        Some(PublicCommitment::DeclineLearn)
    );
}

#[test]
fn discard_public_commitment_materialization_rebuilds_hypothetical_hand() {
    let mut s = Scenario::new(
        deck(&[("Island", 24), ("Pop Quiz", 16)]),
        deck(&[("Plains", 40)]),
        53,
    );
    force_lands(&mut s, 0, "Island", 3);
    clear_hand(&mut s, 0);
    s.force_card_in_hand(0, "Pop Quiz");
    s.advance_to_active_step(0, StepKind::Main);
    cast_and_resolve(&mut s);

    let root = s.game().clone();
    let root_witness = root.state.deterministic_hash();
    let root_cursor = root.semantic_event_cursor();
    let root_observation = viewer_observation(&root, PlayerId(1));
    let space = PossibleWorldSpace::for_viewer(&root, PlayerId(1));
    let world_index = space
        .worlds()
        .iter()
        .position(|world| world.hand.get("Island").copied().unwrap_or(0) > 0)
        .expect("one hypothesis contains Island");
    let world = &space.worlds()[world_index];
    let mut branch = space
        .materialize_index(
            &root,
            world_index,
            907,
            MaterializeMode::RefreshOpponentCommitment,
        )
        .expect("discard prompt materializes");
    let frame = branch.semantic_decision_frame().expect("refreshed frame");
    let mut commitments = BTreeMap::new();
    for offer in &frame.offers {
        if let Some(PublicCommitment::Discard { card }) = &offer.public_commitment {
            *commitments.entry(card.clone()).or_insert(0_u32) += 1;
        }
    }
    assert_eq!(commitments, world.hand);

    let island = frame
        .offers
        .iter()
        .find(|offer| {
            offer.public_commitment
                == Some(PublicCommitment::Discard {
                    card: "Island".to_string(),
                })
        })
        .expect("hypothetical Island offer");
    let transition = branch
        .execute_semantic_command(&Command {
            command_id: "materialized-public-commitment-discard".to_string(),
            expected_revision: frame.revision,
            offer_id: island.id.0,
            answers: Vec::new(),
            object_preconditions: Vec::new(),
        })
        .expect("hypothetical discard executes");
    assert_eq!(
        transition.receipt.public_commitment,
        Some(PublicCommitment::Discard {
            card: "Island".to_string()
        })
    );
    assert_eq!(root.state.deterministic_hash(), root_witness);
    assert_eq!(root.semantic_event_cursor(), root_cursor);
    assert_eq!(viewer_observation(&root, PlayerId(1)), root_observation);
}

#[test]
fn pop_quiz_learn_retrieves_owned_lesson_and_finalizes_once() {
    for lesson in [
        "Firebending Lesson",
        "It'll Quench Ya!",
        "Accumulate Wisdom",
        "Yip Yip!",
        "Fancy Footwork",
    ] {
        let mut s = Scenario::with_sideboards(
            deck(&[("Island", 24), ("Pop Quiz", 16)]),
            deck(&[("Plains", 40)]),
            deck(&[(lesson, 1)]),
            BTreeMap::new(),
            51,
        );
        force_lands(&mut s, 0, "Island", 3);
        clear_hand(&mut s, 0);
        s.force_card_in_hand(0, "Pop Quiz");
        s.advance_to_active_step(0, StepKind::Main);
        cast_and_resolve(&mut s);
        assert_eq!(s.action_space().kind, ActionSpaceKind::Learn);
        let card = s.game().state.players[0].sideboard[0];
        let library_before = library(&s, 0);
        let hand_before = hand(&s, 0).len();
        s.drain_events();
        assert!(s.take_action_by_type(ActionType::LearnTakeLesson));
        assert!(hand(&s, 0).contains(&card));
        assert_eq!(hand(&s, 0).len(), hand_before + 1);
        assert_eq!(library(&s, 0), library_before);
        assert_eq!(s.zone_size(0, ZoneType::Graveyard), 1);
        assert!(s.game().state.stack_objects.is_empty());
        let events = s.drain_events();
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(
                    event,
                    managym::flow::event::GameEvent::SpellResolved { .. }
                ))
                .count(),
            1
        );
        assert!(!events
            .iter()
            .any(|event| matches!(event, managym::flow::event::GameEvent::CardDrawn { .. })));
    }
}

#[test]
fn professors_enter_and_offer_learn_before_priority_returns() {
    for name in ["Professor of Symbology", "Gnarled Professor"] {
        let mut s = five_color_setup(&[name]);
        cast(&mut s, name);
        pass_until(&mut s, ActionSpaceKind::Learn);
        assert_eq!(s.battlefield_permanents_named(0, name).len(), 1);
        settle(&mut s);
    }
}

/// Waterfall Aerialist with ward on player 0's battlefield; player 1 casts
/// Lightning Bolt at it and the ward trigger resolves to its pay-or-not.
fn ward_setup(seed: u64, p1_mountains: usize) -> (Scenario, PermanentId) {
    let mut s = Scenario::new(
        deck(&[
            ("Island", 12),
            ("Mountain", 12),
            ("Waterfall Aerialist", 8),
            ("Lightning Bolt", 8),
        ]),
        deck(&[("Mountain", 24), ("Lightning Bolt", 16)]),
        seed,
    );
    let aerialist = s.force_permanent_on_battlefield(0, "Waterfall Aerialist");
    force_lands(&mut s, 1, "Mountain", p1_mountains);
    clear_hand(&mut s, 0);
    clear_hand(&mut s, 1);
    s.force_card_in_hand(1, "Lightning Bolt");
    s.advance_to_active_step(1, StepKind::Main);
    cast_only(&mut s); // P1 casts Lightning Bolt
    s.choose_target_named("Waterfall Aerialist");
    // Ward trigger goes on the stack above the bolt; let it resolve.
    s.pass_priority();
    s.pass_priority();
    (s, aerialist)
}

#[test]
fn ward_asks_spell_controller_and_counters_on_decline() {
    let (mut s, aerialist) = ward_setup(101, 3);
    let space = s.action_space().clone();
    assert_eq!(space.kind, ActionSpaceKind::PayOrNot);
    assert_eq!(
        space.player,
        Some(PlayerId(1)),
        "the bolt's controller pays"
    );
    assert!(s.take_action_by_type(ActionType::DeclineChoice));

    // The bolt was countered.
    assert_eq!(s.game().state.stack_objects.len(), 0);
    assert_eq!(s.zone_size(1, ZoneType::Graveyard), 1);
    assert_eq!(permanent(&s, aerialist).damage, 0);
}

#[test]
fn ward_paid_lets_the_spell_resolve() {
    let (mut s, _aerialist) = ward_setup(102, 3);
    assert_eq!(s.action_space().kind, ActionSpaceKind::PayOrNot);
    assert!(s.take_action_by_type(ActionType::PayCost));
    s.pass_priority();
    s.pass_priority();
    // Bolt resolved: 3 damage kills the 3/1 Aerialist (lethal SBA).
    assert_eq!(
        s.battlefield_permanents_named(0, "Waterfall Aerialist")
            .len(),
        0
    );
    assert_eq!(s.zone_size(0, ZoneType::Graveyard), 1);
}

#[test]
fn ward_does_not_trigger_for_controllers_own_spell() {
    let mut s = Scenario::new(
        deck(&[
            ("Island", 12),
            ("Mountain", 12),
            ("Waterfall Aerialist", 8),
            ("Lightning Bolt", 8),
        ]),
        deck(&[("Plains", 40)]),
        103,
    );
    let _aerialist = s.force_permanent_on_battlefield(0, "Waterfall Aerialist");
    force_lands(&mut s, 0, "Mountain", 1);
    clear_hand(&mut s, 0);
    s.force_card_in_hand(0, "Lightning Bolt");
    s.advance_to_active_step(0, StepKind::Main);
    cast_only(&mut s);
    s.choose_target_named("Waterfall Aerialist");
    s.pass_priority();
    s.pass_priority();
    // No ward pay-or-not; the bolt resolved directly and killed the 3/1.
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
    assert_eq!(
        s.battlefield_permanents_named(0, "Waterfall Aerialist")
            .len(),
        0
    );
}
