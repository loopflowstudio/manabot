use crate::support::game::{deck_of, until_end_of_turn, Scenario};
use managym::{
    agent::{
        action::{Action, ActionSpaceKind},
        structured_offer::{ChoiceAnswer, ChoiceStep, OfferSubmission, StructuredOfferSet},
    },
    flow::{declaration::DeclarationAssignments, turn::StepKind},
    state::{
        card::{Keyword, Keywords},
        effect::Modification,
        game_object::{PermanentId, PlayerId},
    },
    Game,
};

fn blockers(lure: bool, count: usize) -> (Game, Vec<PermanentId>) {
    let mut scenario = Scenario::new(
        deck_of("Mountain", "Boggart Brute"),
        deck_of("Mountain", "Gray Ogre"),
        364,
    );
    scenario.advance_to_active_step(0, StepKind::Main);
    let mut attackers = Vec::new();
    for _ in 0..2 {
        let attacker = scenario.force_permanent_on_battlefield(0, "Boggart Brute");
        scenario.game_mut().state.permanents[attacker]
            .as_mut()
            .unwrap()
            .summoning_sick = false;
        if lure {
            until_end_of_turn(
                &mut scenario,
                attacker,
                Modification::MustBeBlockedByAllAble,
            );
        }
        attackers.push(attacker);
    }
    for _ in 0..count {
        scenario.force_permanent_on_battlefield(1, "Gray Ogre");
    }
    scenario.advance_to_active_step(0, StepKind::DeclareAttackers);
    scenario.declare_attack();
    scenario.declare_attack();
    scenario.advance_to_active_step(0, StepKind::DeclareBlockers);
    (scenario.game().clone(), attackers)
}

#[test]
fn combat_declarations_menace_and_maximum_one_allow_only_an_empty_block() {
    let mut scenario = Scenario::new(
        deck_of("Mountain", "Flopsie, Bumi's Buddy"),
        deck_of("Mountain", "Gray Ogre"),
        365,
    );
    scenario.advance_to_active_step(0, StepKind::Main);
    let attacker = scenario.force_permanent_on_battlefield(0, "Flopsie, Bumi's Buddy");
    scenario.game_mut().state.permanents[attacker]
        .as_mut()
        .unwrap()
        .summoning_sick = false;
    until_end_of_turn(
        &mut scenario,
        attacker,
        Modification::AddKeywords(Keywords::of(&[Keyword::Menace])),
    );
    until_end_of_turn(
        &mut scenario,
        attacker,
        Modification::MustBeBlockedByAllAble,
    );
    for _ in 0..2 {
        scenario.force_permanent_on_battlefield(1, "Gray Ogre");
    }
    scenario.advance_to_active_step(0, StepKind::DeclareAttackers);
    scenario.declare_attack();
    scenario.advance_to_active_step(0, StepKind::DeclareBlockers);
    let mut game = scenario.game().clone();
    assert!(matches!(
        game.action_space().unwrap().actions.as_slice(),
        [Action::DeclareBlocker { attacker: None, .. }]
    ));
    let offers = game.compound_offers().unwrap();
    // Pair eligibility remains visible, but neither a singleton nor a pair is
    // a legal complete declaration under both restrictions (CR 509.1b–c).
    for choices in [[Some(0), None], [Some(0), Some(0)]] {
        let before = format!("{game:?}");
        assert!(game
            .apply_offer_submission(&offers, &submission(&offers, &choices))
            .is_err());
        assert_eq!(format!("{game:?}"), before);
    }
    let transaction = game
        .prepare_declaration(&offers.decode(&submission(&offers, &[None, None])).unwrap())
        .unwrap();
    assert_eq!(
        transaction.completion.assignments,
        DeclarationAssignments::Blockers {
            assignments: vec![]
        }
    );
}

fn submission(offers: &StructuredOfferSet, choices: &[Option<usize>]) -> OfferSubmission {
    let offer = &offers.projection().offers[0];
    assert_eq!(offer.choices.len(), choices.len());
    OfferSubmission {
        offer_id: offer.id,
        answers: offer
            .choices
            .iter()
            .zip(choices)
            .map(|(choice, selected)| {
                let ChoiceStep::Select {
                    role, candidates, ..
                } = choice;
                ChoiceAnswer::Candidates {
                    role: *role,
                    candidates: selected
                        .map(|index| vec![candidates.initial.as_ref().unwrap()[index].id])
                        .unwrap_or_default(),
                }
            })
            .collect(),
    }
}

#[test]
fn combat_declarations_rejects_singletons_without_changing_the_root() {
    let (mut game, _) = blockers(false, 2);
    let offers = game.compound_offers().unwrap();
    let invalid = submission(&offers, &[Some(0), Some(1)]);
    let before = format!("{game:?}");
    assert!(game.apply_offer_submission(&offers, &invalid).is_err());
    assert_eq!(format!("{game:?}"), before);
    let valid = submission(&offers, &[Some(0), Some(0)]);
    let transaction = game
        .prepare_declaration(&offers.decode(&valid).unwrap())
        .unwrap();
    let DeclarationAssignments::Blockers { assignments } = &transaction.completion.assignments
    else {
        panic!("block completion");
    };
    assert_eq!(assignments.len(), 2);
    assert_eq!(assignments[0].1, assignments[1].1);
    assert_ne!(
        assignments[0].0, assignments[1].0,
        "same-name blockers remain distinct"
    );
}

#[test]
fn combat_declarations_two_menace_lures_offer_both_complete_alternatives() {
    let (mut game, attackers) = blockers(true, 2);
    game.skip_trivial = true;
    let actions = &game.action_space().unwrap().actions;
    assert_eq!(actions.len(), 2);
    for attacker in attackers {
        assert!(actions.iter().any(|action| matches!(action, Action::DeclareBlocker { attacker: Some(target), .. } if *target == attacker)));
    }
    let offers = game.compound_offers().unwrap();
    for choice in 0..2 {
        let submit = submission(&offers, &[Some(choice), Some(choice)]);
        let mut transaction = game
            .prepare_declaration(&offers.decode(&submit).unwrap())
            .unwrap();
        let mut replay = game.clone();
        for child in &transaction.children {
            let actual = replay.execute_semantic_command(&child.command).unwrap();
            assert_eq!(
                serde_json::to_value(actual).unwrap(),
                serde_json::to_value(&child.transition).unwrap()
            );
        }
        transaction.game.take_observation_events();
        assert_eq!(
            transaction.game.state.deterministic_hash(),
            replay.state.deterministic_hash()
        );
        assert_eq!(
            managym::search_state::witness(&transaction.game),
            managym::search_state::witness(&replay)
        );
    }
    assert!(game
        .prepare_declaration(&offers.decode(&submission(&offers, &[None, None])).unwrap())
        .is_err());
}

#[test]
fn combat_declarations_lone_blocker_can_decline_two_menace_lures() {
    let (mut game, _) = blockers(true, 1);
    assert_eq!(
        game.action_space().unwrap().kind,
        ActionSpaceKind::DeclareBlocker
    );
    assert!(matches!(
        game.action_space().unwrap().actions.as_slice(),
        [Action::DeclareBlocker {
            player: PlayerId(1),
            attacker: None,
            ..
        }]
    ));
    let offers = game.compound_offers().unwrap();
    let submit = submission(&offers, &[None]);
    game.apply_offer_submission(&offers, &submit).unwrap();
}

#[test]
fn combat_declarations_bind_both_blocker_and_attacker_incarnations() {
    use managym::{agent::structured_offer::AtomicCommand, state::zone::ZoneType};
    for replace_attacker in [false, true] {
        let (mut game, _) = blockers(false, 2);
        let offers = game.compound_offers().unwrap();
        let command = offers
            .decode(&submission(&offers, &[Some(0), Some(0)]))
            .unwrap();
        let AtomicCommand::DeclareBlockers { assignments, .. } = &command else {
            panic!("block declaration");
        };
        let old = if replace_attacker {
            assignments[0].attacker.unwrap()
        } else {
            assignments[0].blocker
        };
        let successor = game.move_object(old, ZoneType::Graveyard);
        assert_ne!(old, successor);
        let before = format!("{game:?}");
        assert!(game.apply_atomic_command(&command).is_err());
        assert_eq!(format!("{game:?}"), before);
    }
}

#[test]
fn combat_declarations_rejects_the_retired_generic_sequence_executor() {
    use managym::agent::structured_offer::{AtomicCommand, StructuredOfferError};
    let (mut game, _) = blockers(false, 2);
    let offers = game.compound_offers().unwrap();
    let typed = offers
        .decode(&submission(&offers, &[Some(0), Some(0)]))
        .unwrap();
    let AtomicCommand::DeclareBlockers { binding, .. } = typed else {
        panic!("block declaration");
    };
    let partial = AtomicCommand::DeclarationSequence {
        binding,
        kind: ActionSpaceKind::DeclareBlocker,
        actions: vec![game.action_space().unwrap().actions[0].clone()],
    };
    let before = format!("{game:?}");
    assert_eq!(
        game.apply_atomic_command(&partial),
        Err(StructuredOfferError::WrongDecision)
    );
    assert_eq!(format!("{game:?}"), before);
}

#[test]
fn combat_declarations_dedicated_surface_binds_complete_plans_and_fixed_prefixes() {
    use managym::{
        experience::SubjectRef,
        flow::declaration::{DeclarationAnswer, DeclarationPlan},
    };
    let (mut game, _) = blockers(false, 2);
    assert!(game.declaration_projection(PlayerId(0)).unwrap().is_none());
    let projection = game.declaration_projection(PlayerId(1)).unwrap().unwrap();
    assert_eq!(projection.actor, 1);
    assert_eq!(projection.subjects.len(), 2);
    assert!(projection
        .subjects
        .iter()
        .all(|row| row.object.label == "Gray Ogre" && row.power == 2));
    assert!(projection
        .destinations
        .iter()
        .all(|row| row.label == "Boggart Brute"));
    let plan = DeclarationPlan {
        binding: projection.binding.clone(),
        assignments: projection
            .subjects
            .iter()
            .map(|row| {
                let SubjectRef::Object { id } = row.object.subject else {
                    panic!("creature");
                };
                DeclarationAnswer {
                    subject: id,
                    destination: managym::experience::RequiredNullable(Some(
                        projection.destinations[0].subject,
                    )),
                }
            })
            .collect(),
    };
    let before = format!("{game:?}");
    game.validate_declaration_plan(&plan).unwrap();
    let transaction = game.prepare_declaration_plan(&plan).unwrap();
    assert_eq!(
        format!("{game:?}"),
        before,
        "validation and preparation are read-only"
    );
    let mut duplicate = plan.clone();
    duplicate.assignments[1] = duplicate.assignments[0].clone();
    assert!(game.validate_declaration_plan(&duplicate).is_err());
    let mut incomplete = plan.clone();
    incomplete.assignments.pop();
    assert!(game.validate_declaration_plan(&incomplete).is_err());
    game.execute_semantic_command(&transaction.children[0].command)
        .unwrap();
    assert!(
        game.validate_declaration_plan(&plan).is_err(),
        "old binding is stale"
    );
    let continuation = game.declaration_projection(PlayerId(1)).unwrap().unwrap();
    assert_eq!(
        continuation.subjects.iter().filter(|row| row.fixed).count(),
        1
    );
    let fixed = continuation.subjects.iter().find(|row| row.fixed).unwrap();
    assert_eq!(fixed.assigned_to, Some(projection.destinations[0].subject));
    assert!(fixed.destinations.is_empty());
    let suffix = DeclarationPlan {
        binding: continuation.binding,
        assignments: continuation
            .subjects
            .iter()
            .filter(|row| !row.fixed)
            .map(|row| {
                let SubjectRef::Object { id } = row.object.subject else {
                    panic!("creature");
                };
                DeclarationAnswer {
                    subject: id,
                    destination: managym::experience::RequiredNullable(Some(
                        projection.destinations[0].subject,
                    )),
                }
            })
            .collect(),
    };
    let actual = game.prepare_declaration_plan(&suffix).unwrap();
    assert_eq!(actual.completion, transaction.completion);
    assert_eq!(actual.children.len(), 1);
}
