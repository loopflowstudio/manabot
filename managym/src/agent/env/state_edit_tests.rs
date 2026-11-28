use super::*;
use crate::{
    agent::action::{Action, ActionSpace, ActionSpaceKind},
    flow::{
        combat::{AttackTarget, CombatState},
        state_edit::PermanentDestination,
        turn::StepKind,
    },
    state::{game_object::PermanentId, permanent::CounterKind},
};

fn game(skip_trivial: bool) -> Env {
    let mut env = Env::new(42, skip_trivial, false).with_decision_memory_include_own(true);
    let deck = [
        ("Mountain".into(), 28),
        ("Gray Ogre".into(), 8),
        ("Hill Giant".into(), 4),
    ];
    env.reset(
        ["a", "b"]
            .map(|name| PlayerConfig::new(name, BTreeMap::from(deck.clone())))
            .to_vec(),
    )
    .unwrap();
    env
}

fn observed_id(game: &Game, permanent: PermanentId) -> i32 {
    game.state.permanents[permanent].as_ref().unwrap().id.0 as i32
}

fn command(frame: &DecisionFrame) -> SemanticCommand {
    SemanticCommand {
        command_id: format!("edit-test-{}", frame.revision),
        expected_revision: frame.revision,
        offer_id: frame.offers[0].id.0,
        answers: vec![],
        object_preconditions: frame
            .object_candidates
            .first()
            .cloned()
            .into_iter()
            .collect(),
    }
}

#[test]
fn game_state_edits_replay_a_cold_branch_and_its_next_command() {
    let mut root = game(true);
    for _ in 0..32 {
        let game = root.game.as_ref().unwrap();
        let space = game.action_space().unwrap();
        if space.kind == ActionSpaceKind::DeclareAttacker {
            break;
        }
        let index = space
            .actions
            .iter()
            .enumerate()
            .min_by_key(|(_, action)| match action {
                Action::PlayLand { .. } => 0,
                Action::CastSpell { .. } => 1,
                Action::PassPriority { .. } => 2,
                _ => 3,
            })
            .unwrap()
            .0;
        root.step(index as i64).unwrap();
    }
    let game = root.game.as_ref().unwrap();
    assert_eq!(
        game.action_space().unwrap().kind,
        ActionSpaceKind::DeclareAttacker
    );
    let owner = game.active_player();
    let lands: Vec<_> = game
        .battlefield_permanents(owner)
        .into_iter()
        .filter(|&id| game.characteristics(id).types.is_land())
        .collect();
    let original_frame = root.semantic_decision_frame().unwrap();
    let bottom_card = game
        .state
        .card_of(game.permanent_object(lands[0]).unwrap())
        .unwrap();
    let edits: Arc<[_]> = [
        GameStateEdit::MovePermanent {
            permanent: observed_id(game, lands[0]),
            to: PermanentDestination::LibraryBottom,
        },
        GameStateEdit::SetCounters {
            permanent: observed_id(game, lands[1]),
            counter: CounterKind::Stun,
            count: 2,
        },
        GameStateEdit::SetLife { player: 1, life: 5 },
    ]
    .into();
    let original = root.snapshot().unwrap();
    let original_replay = original.to_replay_json().unwrap();
    let empty = root.edit_game_state([]).unwrap();
    assert_eq!(
        empty,
        root.observation_for_player(original_frame.actor as usize)
            .unwrap()
    );
    let invalid = [edits[0], edits[0]];
    assert!(root.edit_game_state(invalid).is_err());
    assert_eq!(
        root.snapshot().unwrap().to_replay_json().unwrap(),
        original_replay
    );
    let options = RestoreOptions {
        game_state_edits: Some(edits.clone()),
        ..Default::default()
    };
    let mut direct = original.restore(Default::default()).unwrap();
    direct.edit_game_state(edits).unwrap();
    let direct_replay = direct.snapshot().unwrap().to_replay_json().unwrap();

    for mode in [
        options.mode,
        BranchMode::ShuffleLibraries { seed: 19 },
        BranchMode::Determinize {
            viewer: owner.0,
            seed: 19,
        },
    ] {
        let mut branch = original
            .restore(RestoreOptions {
                mode,
                ..options.clone()
            })
            .unwrap();
        if matches!(mode, BranchMode::Exact { .. }) {
            assert_eq!(
                branch.snapshot().unwrap().to_replay_json().unwrap(),
                direct_replay
            );
        }
        let game = branch.game.as_ref().unwrap();
        let bottom = game.state.zones.zone_objects(ZoneType::Library, owner)[0];
        assert_eq!(game.state[bottom].name, "Mountain");
        if !matches!(mode, BranchMode::Determinize { .. }) {
            assert_eq!(bottom, game.state.object_of(bottom_card));
        }
        assert_ne!(branch.state_digest().unwrap(), original.state_digest());
        assert!(branch
            .execute_semantic_command(&command(&original_frame))
            .unwrap_err()
            .0
            .contains("stale"));
        let frame = branch.semantic_decision_frame().unwrap();
        let current = command(&frame);
        assert!(!current.object_preconditions.is_empty());
        let snapshot = branch.snapshot().unwrap();
        let raw = snapshot.to_replay_json().unwrap();
        let json: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(json["interventions"][0]["op"], "game_state_edits");
        assert_eq!(json["interventions"][0]["edits"][1]["counter"], "stun");
        let rebuilt = GameSnapshot::from_replay_json(&raw).unwrap();
        assert_eq!(rebuilt.state_digest(), snapshot.state_digest());
        let mut cold = rebuilt.restore(Default::default()).unwrap();
        let viewer = frame.actor as usize;
        assert_eq!(
            cold.observation_for_player(viewer).unwrap(),
            branch.observation_for_player(viewer).unwrap()
        );
        cold.execute_semantic_command(&current).unwrap();
        branch.execute_semantic_command(&current).unwrap();
        assert_eq!(
            cold.selected_witness().unwrap(),
            branch.selected_witness().unwrap()
        );
        let continued = branch.snapshot().unwrap();
        let rebuilt = GameSnapshot::from_replay_json(&continued.to_replay_json().unwrap()).unwrap();
        assert_eq!(rebuilt.state_digest(), continued.state_digest());
    }
    assert_eq!(
        root.snapshot().unwrap().to_replay_json().unwrap(),
        original_replay
    );
}

#[test]
fn game_state_edits_refresh_priority_without_running_sbas() {
    let mut env = game(true);
    let edit = [GameStateEdit::SetLife { player: 0, life: 0 }];
    assert!(env.edit_game_state(edit).is_err());
    while env.game.as_ref().unwrap().state.opening().is_some() {
        env.step(0).unwrap();
    }
    let before = env.semantic_decision_frame().unwrap();
    let obs = env.edit_game_state(edit).unwrap();
    let after = env.semantic_decision_frame().unwrap();
    assert_eq!(after.revision, before.revision + 1);
    assert_eq!(
        obs.action_space.action_space_type,
        ActionSpaceKind::Priority
    );
    assert_eq!(obs.agent.life, 0);
    assert!(!env.is_game_over());
    let pass = env
        .game
        .as_ref()
        .unwrap()
        .action_space()
        .unwrap()
        .actions
        .iter()
        .position(|action| matches!(action, Action::PassPriority { .. }))
        .unwrap();
    env.step(pass as i64).unwrap();
    assert!(env.is_game_over());
}

#[test]
fn game_state_edits_keep_the_focal_blocker_and_only_prompt_new_declarers() {
    let mut env = game(false);
    let game = env.game.as_mut().unwrap();
    game.keep_opening_hands();
    while game.state.turn.current_step_kind() != StepKind::DeclareBlockers {
        game.state.turn.advance_step();
    }
    game.state.set_step_initialized(true);
    let active = game.active_player();
    let defender = game.non_active_player();
    let departing = game
        .scenario_force_battlefield(active, "Gray Ogre", true)
        .unwrap();
    let remaining = game
        .scenario_force_battlefield(active, "Hill Giant", true)
        .unwrap();
    let committed = game
        .scenario_force_battlefield(defender, "Gray Ogre", true)
        .unwrap();
    let focal = game
        .scenario_force_battlefield(defender, "Gray Ogre", true)
        .unwrap();
    let tapped = game
        .scenario_force_battlefield(defender, "Hill Giant", true)
        .unwrap();
    game.state.permanents[tapped].as_mut().unwrap().tapped = true;
    for attacker in [departing, remaining] {
        game.state.permanents[attacker].as_mut().unwrap().attacking = true;
    }
    game.state.combat = Some(Box::new(CombatState {
        attack_targets: [departing, remaining]
            .map(|id| (id, Some(AttackTarget::Player(defender))))
            .to_vec(),
        attacker_to_blockers: BTreeMap::from([(departing, vec![committed])]),
        blocked: vec![departing],
        ..Default::default()
    }));
    game.publish_action_space(ActionSpace {
        player: Some(defender),
        kind: ActionSpaceKind::DeclareBlocker,
        actions: game.block_actions(defender, focal),
        focus: Vec::new(),
    });
    let [departing_id, focal_id, tapped_id] =
        [departing, focal, tapped].map(|id| observed_id(game, id));
    let move_attacker = GameStateEdit::MovePermanent {
        permanent: departing_id,
        to: PermanentDestination::Exile,
    };
    let frame_before = env.semantic_decision_frame().unwrap();
    let before = env.snapshot().unwrap().to_replay_json().unwrap();
    let error = env
        .edit_game_state([
            move_attacker,
            GameStateEdit::SetTapped {
                permanent: focal_id,
                tapped: true,
            },
        ])
        .unwrap_err();
    assert!(error.0.contains("current declarer ineligible"));
    assert_eq!(
        env.semantic_decision_frame().unwrap().fingerprint,
        frame_before.fingerprint
    );
    assert_eq!(env.snapshot().unwrap().to_replay_json().unwrap(), before);
    for edits in [
        vec![
            move_attacker,
            GameStateEdit::SetTapped {
                permanent: tapped_id,
                tapped: false,
            },
        ],
        vec![GameStateEdit::SetLife {
            player: defender.0,
            life: 12,
        }],
    ] {
        env.edit_game_state(edits).unwrap();
        let game = env.game.as_ref().unwrap();
        assert_eq!(
            game.state.combat.as_ref().unwrap().blockers_to_declare,
            vec![tapped]
        );
        assert_eq!(
            game.action_space().unwrap().actions,
            [
                Action::DeclareBlocker {
                    player: defender,
                    blocker: focal,
                    attacker: Some(remaining)
                },
                Action::DeclareBlocker {
                    player: defender,
                    blocker: focal,
                    attacker: None
                },
            ]
        );
    }
    env.step(1).unwrap();
    let game = env.game.as_ref().unwrap();
    assert_eq!(
        game.action_space().unwrap().kind,
        ActionSpaceKind::DeclareBlocker
    );
    assert!(game.action_space().unwrap().actions.iter().all(|action| {
        matches!(action, Action::DeclareBlocker { blocker, .. } if *blocker == tapped)
    }));
}

#[test]
fn game_state_edits_clear_pre_edit_decline_suppression_without_reprompting() {
    let mut env = game(false);
    let game = env.game.as_mut().unwrap();
    game.keep_opening_hands();
    while game.state.turn.current_step_kind() != StepKind::DeclareBlockers {
        game.state.turn.advance_step();
    }
    game.state.set_step_initialized(true);
    let active = game.active_player();
    let defender = game.non_active_player();
    let attacker = game
        .scenario_force_battlefield(active, "Hill Giant", true)
        .unwrap();
    let declined = game
        .scenario_force_battlefield(defender, "Gray Ogre", true)
        .unwrap();
    let remaining_ogre = game
        .scenario_force_battlefield(defender, "Gray Ogre", true)
        .unwrap();
    let focal = game
        .scenario_force_battlefield(defender, "Hill Giant", true)
        .unwrap();
    game.state.permanents[attacker].as_mut().unwrap().attacking = true;
    game.state.combat = Some(Box::new(CombatState {
        attack_targets: vec![(attacker, Some(AttackTarget::Player(defender)))],
        blockers_to_declare: vec![remaining_ogre],
        declined_blockers: vec![declined],
        ..Default::default()
    }));
    game.publish_action_space(ActionSpace {
        player: Some(defender),
        kind: ActionSpaceKind::DeclareBlocker,
        actions: game.block_actions(defender, focal),
        focus: Vec::new(),
    });
    env.edit_game_state([GameStateEdit::SetLife {
        player: defender.0,
        life: 3,
    }])
    .unwrap();
    let combat = env.game.as_ref().unwrap().state.combat.as_ref().unwrap();
    assert!(combat.declined_blockers.is_empty());
    assert!(combat.detached_declarers.contains(&declined));
    assert_eq!(combat.blockers_to_declare, vec![remaining_ogre]);
    // Decline focal so remaining_ogre becomes the next prompt.
    let decline_focal = env
        .game
        .as_ref()
        .unwrap()
        .action_space()
        .unwrap()
        .actions
        .iter()
        .position(|action| {
            matches!(
                action,
                Action::DeclareBlocker {
                    blocker,
                    attacker: None,
                    ..
                } if *blocker == focal
            )
        })
        .unwrap();
    env.step(decline_focal as i64).unwrap();
    let space = env.game.as_ref().unwrap().action_space().unwrap();
    assert_eq!(space.kind, ActionSpaceKind::DeclareBlocker);
    // remaining_ogre must still be allowed to chump-block at 3 life, and declined is never reprompted.
    assert!(space.actions.iter().any(|action| {
        matches!(
            action,
            Action::DeclareBlocker {
                blocker,
                attacker: Some(target),
                ..
            } if *blocker == remaining_ogre && *target == attacker
        )
    }));
}
