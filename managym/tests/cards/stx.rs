//! Strixhaven: School of Mages (`src/cardsets/stx.rs`).
use crate::{
    helpers::Scenario,
    support::{
        cast, cast_only, clear_hand, deck, force_lands, library_card, pass_until, permanent_state,
        settle, setup,
    },
};
use managym::{
    agent::{
        action::{Action, ActionSpaceKind, ActionType},
        observation::Observation,
    },
    flow::turn::StepKind,
    state::{
        game_object::{PermanentId, PlayerId, Target},
        zone::ZoneType,
    },
};
use std::collections::BTreeSet;

// ---------------------------------------------------------------------------
// Ward (Waterfall Aerialist) — triggered PAY_OR_NOT on being targeted.
// ---------------------------------------------------------------------------

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
    assert_eq!(permanent_state(&s, aerialist).damage, 0);
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

#[test]
fn professors_enter_and_offer_learn_before_priority_returns() {
    for name in ["Professor of Symbology", "Gnarled Professor"] {
        let mut s = setup(&[name]);
        cast(&mut s, name);
        pass_until(&mut s, ActionSpaceKind::Learn);
        assert_eq!(s.battlefield_permanents_named(0, name).len(), 1);
        settle(&mut s);
    }
}

#[test]
fn library_search_is_private_enters_tapped_and_continues_to_learn() {
    for (name, forest) in [("Field Trip", true), ("Shared Roots", false)] {
        let mut s = setup(&[name]);
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
fn search_selection_or_decline_and_shuffle_restore_exactly_under_undo() {
    use managym::search_state::{BenchCommand, BranchDriver, ClonePlusUndoDriver};
    let mut s = setup(&["Field Trip"]);
    cast(&mut s, "Field Trip");
    s.pass_priority();
    s.pass_priority();
    for select in [false, true] {
        let driver = ClonePlusUndoDriver::default();
        let root = driver.witness(s.game());
        let mut branch = driver.fork_exact(s.game());
        let mark = driver.mark(&mut branch);
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
        driver
            .apply(
                &mut branch,
                BenchCommand {
                    action_index: index,
                    expected_action_hash: None,
                    expected_state_hash: None,
                },
            )
            .unwrap();
        assert_ne!(driver.witness(&branch), root);
        driver.rollback(&mut branch, mark);
        assert_eq!(driver.witness(&branch), root);
        assert_eq!(driver.witness(s.game()), root);
    }
}

#[test]
fn targeted_combat_tricks_apply_through_play() {
    for name in ["Enthusiastic Study", "Octopus Form", "Fists of Flame"] {
        let mut s = setup(&[name, "Gnarled Professor"]);
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
        let mut s = setup(&[name, "Lightning Strike"]);
        let target = s.force_permanent_on_battlefield(0, name);
        cast(&mut s, "Lightning Strike");
        assert!(s.choose_target(Target::Permanent(target)));
        settle(&mut s);
        assert!(s.game().state.permanents[target].is_none());
        match name {
            "Pretending Poxbearers" => {
                assert_eq!(s.battlefield_permanents_named(0, "Ally").len(), 1)
            }
            "Knowledge Seeker" => assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 1),
            _ => assert_eq!(s.battlefield_permanents_named(0, "Ally").len(), 0),
        }
    }
}

#[test]
fn resampled_private_search_choices_still_name_library_cards() {
    let mut s = setup(&["Field Trip"]);
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
    let mut s = setup(&["Field Trip"]);
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
    let mut s = setup(&["Field Trip"]);
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
