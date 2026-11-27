use std::collections::{BTreeMap, BTreeSet};

use managym::{
    agent::{
        action::{Action, ActionSpaceKind},
        decision_surface::{
            AtomicCommand, BoundTarget, Candidate, CandidateId, CandidateSource, CandidateSourceId,
            CandidateValue, ChoiceAnswer, ChoiceStep, DecisionOption, DecisionOptionId,
            DecisionProjection, DecisionSubmission, DecisionSurfaceError, DecisionVerb,
            ObjectRenderId, PromptKind, RoleId, SubjectRef,
        },
        observation::Observation,
    },
    flow::turn::StepKind,
    state::{
        game_object::{ObjectId, PlayerId, Target},
        stack_object::StackObject,
        target::Target as ActionTarget,
    },
    Game,
};

use crate::support::game::*;

fn bolt_priority_scenario() -> Scenario {
    let mut scenario = Scenario::new(
        deck_of("Mountain", "Lightning Bolt"),
        deck_of("Mountain", "Gray Ogre"),
        181,
    );
    scenario.advance_to_active_step(0, StepKind::Main);
    scenario.game_mut().scenario_clear_hand(PlayerId(0));
    scenario.game_mut().scenario_clear_hand(PlayerId(1));
    scenario.force_card_in_hand(0, "Lightning Bolt");
    scenario.force_permanent_on_battlefield(0, "Mountain");
    scenario.force_permanent_on_battlefield(1, "Gray Ogre");
    scenario
        .game_mut()
        .scenario_refresh_priority()
        .expect("arranged priority space should refresh");
    scenario
}

fn cast_offer(set: &managym::agent::decision_surface::DecisionSurface) -> &DecisionOption {
    set.projection()
        .offers
        .iter()
        .find(|offer| offer.verb == DecisionVerb::Cast)
        .expect("Bolt cast offer")
}

fn pass_offer(set: &managym::agent::decision_surface::DecisionSurface) -> &DecisionOption {
    set.projection()
        .offers
        .iter()
        .find(|offer| offer.verb == DecisionVerb::PassPriority)
        .expect("pass-priority offer")
}

fn select_step(offer: &DecisionOption) -> (RoleId, &[Candidate]) {
    let [ChoiceStep::Select {
        role,
        candidates,
        min,
        max,
        ..
    }] = offer.choices.as_slice()
    else {
        panic!("expected one select choice")
    };
    assert_eq!((*min, *max), (1, 1));
    (
        *role,
        candidates.initial.as_deref().expect("initial candidates"),
    )
}

fn player_candidate(candidates: &[Candidate], player: u8) -> CandidateId {
    candidates
        .iter()
        .find_map(|candidate| match candidate.value {
            CandidateValue::Subject {
                subject: SubjectRef::Player { id },
            } if id == player => Some(candidate.id),
            _ => None,
        })
        .expect("player candidate")
}

fn pass_priority(game: &mut Game) {
    let index = game
        .action_space()
        .expect("active action space")
        .actions
        .iter()
        .position(|action| matches!(action, Action::PassPriority { .. }))
        .expect("pass action");
    game.step(index).expect("pass should succeed");
}

fn legacy_cast_at_player(game: &mut Game, object: ObjectId, player: PlayerId) {
    let cast_index = game
        .action_space()
        .expect("priority action space")
        .actions
        .iter()
        .position(
            |action| matches!(action, Action::CastSpell { object: legal, .. } if *legal == object),
        )
        .expect("legacy cast action");
    game.step(cast_index).expect("legacy cast declaration");

    assert_eq!(
        game.action_space().map(|space| space.kind),
        Some(ActionSpaceKind::ChooseTarget)
    );
    let target_index = game
        .action_space()
        .expect("target action space")
        .actions
        .iter()
        .position(|action| {
            matches!(
                action,
                Action::ChooseTarget {
                    target: ActionTarget::Player(legal),
                    ..
                } if *legal == player
            )
        })
        .expect("legacy target action");
    game.step(target_index).expect("legacy target declaration");
}

fn assert_equivalent_surface(structured: &Game, legacy: &Game) {
    assert_eq!(structured.action_space(), legacy.action_space());
    assert_eq!(structured.state.stack_objects, legacy.state.stack_objects);
    assert_eq!(structured.state.events, legacy.state.events);
    assert_eq!(structured.state.pending_events, legacy.state.pending_events);
    assert_eq!(
        structured.state.observation_events,
        legacy.state.observation_events
    );
    assert_eq!(
        Observation::new(structured, &[]).to_json(),
        Observation::new(legacy, &[]).to_json()
    );
}

#[test]
fn structured_offer_bolt_cast_is_atomic_and_legacy_equivalent() {
    let root = bolt_priority_scenario().game().clone();
    let set = root
        .structured_priority_offers()
        .expect("structured priority offers");

    assert_eq!(set.projection().actor, 0);
    assert_eq!(set.projection().kind, PromptKind::Priority);
    assert_eq!(set.projection().offers.len(), 2);

    let offer = cast_offer(&set);
    assert_eq!(offer.label, "Cast Lightning Bolt");
    let (role, candidates) = select_step(offer);
    assert_eq!(candidates.len(), 3, "two players plus Gray Ogre");
    assert_eq!(
        candidates
            .iter()
            .filter(|candidate| candidate.label == "Gray Ogre")
            .count(),
        1
    );

    let submission = DecisionSubmission {
        offer_id: offer.id,
        answers: vec![ChoiceAnswer::Candidates {
            role,
            candidates: vec![player_candidate(candidates, 1)],
        }],
    };
    let command = set
        .decode(&submission)
        .expect("offered target should decode");
    let AtomicCommand::CastSpell {
        object, targets, ..
    } = &command
    else {
        panic!("expected cast command")
    };
    assert_eq!(targets, &[BoundTarget::Player(PlayerId(1))]);

    let mut structured = root.clone();
    let mut legacy = root;
    assert!(!structured
        .apply_offer_submission(&set, &submission)
        .expect("atomic cast should apply"));
    legacy_cast_at_player(&mut legacy, *object, PlayerId(1));

    assert!(structured.pending_choice.is_none());
    let Some(StackObject::Spell(spell)) = structured.state.stack_objects.last() else {
        panic!("Bolt should be on the stack")
    };
    // The spell is the object the offered hand card became (CR 400.7g).
    assert_eq!(
        structured.state.card_id(spell.object),
        structured.state.card_id(*object)
    );
    assert_ne!(spell.object, *object);
    assert_eq!(spell.targets, vec![Target::Player(PlayerId(1))]);
    assert_equivalent_surface(&structured, &legacy);

    pass_priority(&mut structured);
    pass_priority(&mut structured);
    pass_priority(&mut legacy);
    pass_priority(&mut legacy);
    assert_eq!(structured.state.players[1].life, 17);
    assert_eq!(legacy.state.players[1].life, 17);
    assert_equivalent_surface(&structured, &legacy);
}

#[test]
fn structured_offer_pass_is_legacy_equivalent() {
    let root = bolt_priority_scenario().game().clone();
    let set = root
        .structured_priority_offers()
        .expect("structured priority offers");
    let submission = DecisionSubmission {
        offer_id: pass_offer(&set).id,
        answers: Vec::new(),
    };

    let mut structured = root.clone();
    let mut legacy = root;
    structured
        .apply_offer_submission(&set, &submission)
        .expect("atomic pass should apply");
    pass_priority(&mut legacy);
    assert_equivalent_surface(&structured, &legacy);
}

#[test]
fn structured_offer_candidates_are_uncapped_past_legacy_tensor_width() {
    let mut scenario = Scenario::new(
        deck_of("Mountain", "Lightning Bolt"),
        BTreeMap::from([("Gray Ogre".to_string(), 40)]),
        182,
    );
    scenario.advance_to_active_step(0, StepKind::Main);
    scenario.game_mut().scenario_clear_hand(PlayerId(0));
    scenario.force_card_in_hand(0, "Lightning Bolt");
    scenario.force_permanent_on_battlefield(0, "Mountain");
    for count in 0..33 {
        let perm = scenario.force_permanent_on_battlefield(1, "Gray Ogre");
        if count > 0 {
            scenario.game_mut().state.permanents[perm]
                .as_mut()
                .unwrap()
                .counters
                .change(
                    managym::state::permanent::CounterKind::PlusOnePlusOne,
                    count,
                );
        }
    }
    scenario
        .game_mut()
        .scenario_refresh_priority()
        .expect("large priority space should refresh");

    let set = scenario
        .game()
        .structured_priority_offers()
        .expect("structured priority offers");
    let (_, candidates) = select_step(cast_offer(&set));
    assert_eq!(
        candidates.len(),
        35,
        "33 distinguishable creatures plus both players"
    );
    assert!(candidates.len() > 32);
    assert_eq!(
        candidates
            .iter()
            .map(|candidate| candidate.id)
            .collect::<BTreeSet<_>>()
            .len(),
        candidates.len()
    );
}

#[test]
fn structured_offer_deduplicates_identical_target_candidates() {
    let mut scenario = Scenario::new(
        deck_of("Mountain", "Lightning Bolt"),
        BTreeMap::from([("Gray Ogre".to_string(), 40)]),
        182,
    );
    scenario.advance_to_active_step(0, StepKind::Main);
    scenario.game_mut().scenario_clear_hand(PlayerId(0));
    scenario.force_card_in_hand(0, "Lightning Bolt");
    scenario.force_permanent_on_battlefield(0, "Mountain");
    for _ in 0..5 {
        scenario.force_permanent_on_battlefield(1, "Gray Ogre");
    }
    scenario
        .game_mut()
        .scenario_refresh_priority()
        .expect("priority space should refresh");

    let set = scenario
        .game()
        .structured_priority_offers()
        .expect("structured priority offers");
    let (_, candidates) = select_step(cast_offer(&set));
    assert_eq!(
        candidates.len(),
        3,
        "5 identical Gray Ogres deduplicate to 1 candidate plus both players"
    );
    assert_eq!(
        candidates
            .iter()
            .filter(|candidate| candidate.label == "Gray Ogre")
            .count(),
        1
    );
}

#[test]
fn structured_offer_rejects_fabricated_and_stale_ids_without_mutation() {
    let mut root = bolt_priority_scenario().game().clone();
    let set = root
        .structured_priority_offers()
        .expect("structured priority offers");
    let offer = cast_offer(&set);
    let (role, candidates) = select_step(offer);

    assert_eq!(
        set.decode(&DecisionSubmission {
            offer_id: DecisionOptionId(999),
            answers: Vec::new(),
        }),
        Err(DecisionSurfaceError::UnknownOffer(DecisionOptionId(999)))
    );
    assert_eq!(
        set.decode(&DecisionSubmission {
            offer_id: offer.id,
            answers: vec![ChoiceAnswer::Candidates {
                role,
                candidates: vec![CandidateId(999)],
            }],
        }),
        Err(DecisionSurfaceError::UnknownCandidate(CandidateId(999)))
    );
    assert_eq!(
        set.decode(&DecisionSubmission {
            offer_id: offer.id,
            answers: vec![
                ChoiceAnswer::Candidates {
                    role,
                    candidates: vec![player_candidate(candidates, 1)],
                },
                ChoiceAnswer::Candidates {
                    role,
                    candidates: vec![player_candidate(candidates, 0)],
                },
            ],
        }),
        Err(DecisionSurfaceError::DuplicateRole(role))
    );

    let submission = DecisionSubmission {
        offer_id: offer.id,
        answers: vec![ChoiceAnswer::Candidates {
            role,
            candidates: vec![player_candidate(candidates, 1)],
        }],
    };

    // Re-publishing the same legal priority decision makes the old offer set
    // stale even though every public offer, candidate, and legacy action still
    // looks identical. The binding is to the exact decision, not its shape.
    root.scenario_refresh_priority()
        .expect("same priority decision should refresh");
    let replacement = root
        .structured_priority_offers()
        .expect("replacement structured priority offers");
    assert_eq!(replacement.projection(), set.projection());
    let before = format!("{root:?}");
    assert!(matches!(
        root.apply_offer_submission(&set, &submission),
        Err(DecisionSurfaceError::StaleOrIllegal(_))
    ));
    assert_eq!(format!("{root:?}"), before);
}

#[test]
fn structured_offer_game_fixture_matches_typed_wire_shape() {
    let projection = DecisionProjection {
        actor: 0,
        kind: PromptKind::Priority,
        offers: vec![
            DecisionOption {
                id: DecisionOptionId(0),
                actor: 0,
                verb: DecisionVerb::Cast,
                public_commitment: None,
                source: Some(SubjectRef::Object {
                    id: ObjectRenderId {
                        entity: 31,
                        incarnation: 0,
                    },
                }),
                label: "Cast Lightning Bolt".to_string(),
                help: None,
                choices: vec![ChoiceStep::Select {
                    role: RoleId(1),
                    label: "Target".to_string(),
                    candidates: CandidateSource {
                        id: CandidateSourceId(0),
                        depends_on: Vec::new(),
                        initial: Some(vec![
                            fixture_candidate(0, SubjectRef::Player { id: 0 }, "Hero"),
                            fixture_candidate(1, SubjectRef::Player { id: 1 }, "Villain"),
                            fixture_candidate(
                                2,
                                SubjectRef::Object {
                                    id: ObjectRenderId {
                                        entity: 77,
                                        incarnation: 0,
                                    },
                                },
                                "Gray Ogre",
                            ),
                        ]),
                    },
                    min: 1,
                    max: 1,
                    ordered: false,
                    distinct: true,
                }],
                confirm_label: "Cast".to_string(),
            },
            DecisionOption {
                id: DecisionOptionId(1),
                actor: 0,
                verb: DecisionVerb::PassPriority,
                public_commitment: None,
                source: None,
                label: "Pass priority".to_string(),
                help: None,
                choices: Vec::new(),
                confirm_label: "Pass".to_string(),
            },
        ],
    };

    let fixture = include_str!("../fixtures/structured_priority_bolt_offer.json");
    let fixture_value: serde_json::Value =
        serde_json::from_str(fixture).expect("fixture JSON should parse");
    assert_eq!(
        serde_json::to_value(&projection).expect("projection should serialize"),
        fixture_value
    );
    assert_eq!(
        serde_json::from_str::<DecisionProjection>(fixture).expect("fixture should deserialize"),
        projection
    );
}

fn fixture_candidate(id: u32, subject: SubjectRef, label: &str) -> Candidate {
    Candidate {
        id: CandidateId(id),
        value: CandidateValue::Subject { subject },
        label: label.to_string(),
        help: None,
        preview: None,
    }
}

#[test]
fn compound_blockers_bind_roles_and_stop_at_priority() {
    let mut scenario = Scenario::new(
        BTreeMap::from([("Mountain".to_string(), 20), ("Hill Giant".to_string(), 20)]),
        deck_of("Mountain", "Gray Ogre"),
        991,
    );
    scenario.advance_to_active_step(0, StepKind::Main);
    let attacker = scenario.force_permanent_on_battlefield(0, "Hill Giant");
    scenario.game_mut().state.permanents[attacker]
        .as_mut()
        .unwrap()
        .summoning_sick = false;
    for _ in 0..3 {
        scenario.force_permanent_on_battlefield(1, "Gray Ogre");
    }
    scenario.advance_to_active_step(0, StepKind::DeclareAttackers);
    scenario.declare_attack();
    scenario.advance_to_active_step(0, StepKind::DeclareBlockers);
    let root = scenario.game().clone();
    let offers = root.compound_offers().unwrap();
    let offer = &offers.projection().offers[0];
    assert_eq!(offer.verb, DecisionVerb::DeclareBlockers);
    // 3 identical Gray Ogres facing 1 Hill Giant fold into 1 count-based role (counts 1..=3).
    assert_eq!(offer.choices.len(), 1);
    let ChoiceStep::Select {
        role,
        candidates,
        min,
        max,
        ..
    } = &offer.choices[0];
    assert_eq!((*min, *max), (0, 1));
    let initial = candidates.initial.as_ref().unwrap();
    assert_eq!(initial.len(), 3);
    assert_eq!(initial[0].label, "1 × Gray Ogre → Hill Giant");
    assert_eq!(initial[1].label, "2 × Gray Ogre → Hill Giant");
    assert_eq!(initial[2].label, "3 × Gray Ogre → Hill Giant");
    let submission = DecisionSubmission {
        offer_id: offer.id,
        answers: vec![ChoiceAnswer::Candidates {
            role: *role,
            candidates: vec![initial[2].id],
        }],
    };
    let mut applied = root.clone();
    applied
        .apply_offer_submission(&offers, &submission)
        .unwrap();
    assert_eq!(
        applied.action_space().unwrap().kind,
        ActionSpaceKind::Priority
    );
    let commands = root.compound_commands(&offers, &submission).unwrap();
    assert_eq!(commands.len(), 3);
    assert_eq!(
        root.action_space().unwrap().kind,
        ActionSpaceKind::DeclareBlocker
    );
    assert!(matches!(
        applied.apply_offer_submission(&offers, &submission),
        Err(DecisionSurfaceError::StaleOrIllegal(_))
    ));
    let mut invalid = submission.clone();
    invalid.answers.push(invalid.answers[0].clone());
    assert!(matches!(
        offers.decode(&invalid),
        Err(DecisionSurfaceError::DuplicateRole(_))
    ));
    invalid.answers = vec![ChoiceAnswer::Candidates {
        role: RoleId(0),
        candidates: vec![CandidateId(999)],
    }];
    assert_eq!(
        offers.decode(&invalid),
        Err(DecisionSurfaceError::UnknownCandidate(CandidateId(999)))
    );
}

#[test]
fn compound_waterbend_subset_is_order_independent_and_forced_suffix_safe() {
    for skip_trivial in [false, true] {
        let mut scenario = Scenario::new(
            BTreeMap::from([
                ("Forest".to_string(), 16),
                ("Water Tribe Rallier".to_string(), 16),
                ("Gray Ogre".to_string(), 8),
            ]),
            deck_of("Mountain", "Gray Ogre"),
            992,
        );
        scenario.advance_to_active_step(0, StepKind::Main);
        scenario.game_mut().skip_trivial = skip_trivial;
        scenario.force_permanent_on_battlefield(0, "Water Tribe Rallier");
        for _ in 0..4 {
            scenario.force_permanent_on_battlefield(0, "Gray Ogre");
        }
        scenario.game_mut().scenario_refresh_priority().unwrap();
        assert!(scenario
            .take_action_by_type(managym::agent::action::ActionType::PriorityActivateAbility));
        let root = scenario.game().clone();
        let offers = root.compound_offers().unwrap();
        let offer = &offers.projection().offers[0];
        assert_eq!(offer.verb, DecisionVerb::PayWaterbend);
        // 1 Water Tribe Rallier + 4 identical Gray Ogres fold into 2 forced roles.
        assert_eq!(offer.choices.len(), 2);
        let mut answers = Vec::new();
        for ChoiceStep::Select {
            role,
            min,
            max,
            candidates,
            ..
        } in &offer.choices
        {
            assert_eq!((*min, *max), (1, 1));
            let initial = candidates.initial.as_ref().unwrap();
            assert_eq!(initial.len(), 1);
            answers.push(ChoiceAnswer::Candidates {
                role: *role,
                candidates: vec![initial[0].id],
            });
        }
        let submission = DecisionSubmission {
            offer_id: offer.id,
            answers: answers.clone(),
        };
        let commands = root.compound_commands(&offers, &submission).unwrap();
        // With skip_trivial enabled, tapping Rallier leaves 4 identical Gray
        // Ogres whose folded 1-action prompts auto-complete as forced suffix taps.
        assert_eq!(commands.len(), if skip_trivial { 1 } else { 5 });
        answers.reverse();
        let reversed = DecisionSubmission {
            offer_id: offer.id,
            answers,
        };
        assert_eq!(
            serde_json::to_value(root.compound_commands(&offers, &reversed).unwrap()).unwrap(),
            serde_json::to_value(commands).unwrap()
        );
        let mut applied = root.clone();
        applied
            .apply_offer_submission(&offers, &submission)
            .unwrap();
        assert!(!matches!(
            applied.action_space().map(|space| space.kind),
            Some(ActionSpaceKind::Waterbend)
        ));
        if skip_trivial {
            // Native auto-pass may resolve the paid ability; lowering stops
            // before answering the newly revealed library-card decision.
            assert_eq!(
                applied.action_space().unwrap().kind,
                ActionSpaceKind::LookAndSelect
            );
        } else {
            assert_eq!(applied.state.stack_objects.len(), 1);
        }
    }
}

#[test]
fn structured_compound_menace_retains_sequential_support() {
    let mut scenario = Scenario::new(
        deck_of("Mountain", "Boggart Brute"),
        deck_of("Mountain", "Gray Ogre"),
        993,
    );
    scenario.advance_to_active_step(0, StepKind::Main);
    let attacker = scenario.force_permanent_on_battlefield(0, "Boggart Brute");
    scenario.game_mut().state.permanents[attacker]
        .as_mut()
        .unwrap()
        .summoning_sick = false;
    for _ in 0..2 {
        scenario.force_permanent_on_battlefield(1, "Gray Ogre");
    }
    scenario.advance_to_active_step(0, StepKind::DeclareAttackers);
    scenario.declare_attack();
    scenario.advance_to_active_step(0, StepKind::DeclareBlockers);
    let root = scenario.game();
    let offers = root.compound_offers().unwrap();
    assert_eq!(
        offers.projection().offers.len(),
        root.action_space().unwrap().actions.len()
    );
    assert!(offers
        .projection()
        .offers
        .iter()
        .all(|offer| offer.verb == DecisionVerb::DeclareBlocker && offer.choices.is_empty()));
    for offer in &offers.projection().offers {
        assert_eq!(
            root.compound_commands(
                &offers,
                &DecisionSubmission {
                    offer_id: offer.id,
                    answers: vec![]
                }
            )
            .unwrap()
            .len(),
            1
        );
    }
}

#[test]
fn structured_compound_flying_attack_has_optionless_blocker_roles() {
    let mut scenario = Scenario::new(
        BTreeMap::from([("Island".to_string(), 20), ("Wind Drake".to_string(), 20)]),
        deck_of("Mountain", "Gray Ogre"),
        994,
    );
    scenario.advance_to_active_step(0, StepKind::Main);
    let attacker = scenario.force_permanent_on_battlefield(0, "Wind Drake");
    scenario.game_mut().state.permanents[attacker]
        .as_mut()
        .unwrap()
        .summoning_sick = false;
    for _ in 0..2 {
        scenario.force_permanent_on_battlefield(1, "Gray Ogre");
    }
    scenario.advance_to_active_step(0, StepKind::DeclareAttackers);
    scenario.declare_attack();
    scenario.advance_to_active_step(0, StepKind::DeclareBlockers);
    let root = scenario.game();
    let offers = root.compound_offers().unwrap();
    let offer = &offers.projection().offers[0];
    assert_eq!(offer.verb, DecisionVerb::DeclareBlockers);
    let mut answers = Vec::new();
    for ChoiceStep::Select {
        role,
        min,
        max,
        candidates,
        ..
    } in &offer.choices
    {
        assert_eq!((*min, *max), (0, 0));
        assert!(candidates.initial.as_ref().unwrap().is_empty());
        answers.push(ChoiceAnswer::Candidates {
            role: *role,
            candidates: vec![],
        });
    }
    assert_eq!(
        root.compound_commands(
            &offers,
            &DecisionSubmission {
                offer_id: offer.id,
                answers
            }
        )
        .unwrap()
        .len(),
        2
    );
}

#[test]
fn structured_compound_waterbend_effects_or_competing_mana_stay_sequential() {
    for excluded in ["Llanowar Elves", "Badgermole Cub"] {
        let mut scenario = Scenario::new(
            BTreeMap::from([
                ("Forest".to_string(), 12),
                ("Water Tribe Rallier".to_string(), 12),
                ("Gray Ogre".to_string(), 8),
                (excluded.to_string(), 8),
            ]),
            deck_of("Mountain", "Gray Ogre"),
            995,
        );
        scenario.advance_to_active_step(0, StepKind::Main);
        scenario.force_permanent_on_battlefield(0, "Water Tribe Rallier");
        scenario.force_permanent_on_battlefield(0, excluded);
        for _ in 0..4 {
            scenario.force_permanent_on_battlefield(0, "Gray Ogre");
        }
        scenario.game_mut().scenario_refresh_priority().unwrap();
        assert!(scenario
            .take_action_by_type(managym::agent::action::ActionType::PriorityActivateAbility));
        let root = scenario.game();
        let offers = root.compound_offers().unwrap();
        assert_eq!(
            offers.projection().offers.len(),
            root.action_space().unwrap().actions.len()
        );
        assert!(offers
            .projection()
            .offers
            .iter()
            .all(|offer| offer.verb != DecisionVerb::PayWaterbend && offer.choices.is_empty()));
    }
}

#[test]
fn compound_blockers_interleaved_identical_blockers_lower_in_engine_order() {
    for skip_trivial in [false, true] {
        // 1 attacker (count-role path): defenders enter as [Gray Ogre, Hill Giant, Gray Ogre]
        // so identical blockers are interleaved around a distinct blocker in engine order.
        let mut scenario = Scenario::new(
            BTreeMap::from([("Forest".to_string(), 20), ("Craw Wurm".to_string(), 20)]),
            BTreeMap::from([
                ("Mountain".to_string(), 20),
                ("Gray Ogre".to_string(), 10),
                ("Hill Giant".to_string(), 10),
            ]),
            996,
        );
        scenario.advance_to_active_step(0, StepKind::Main);
        scenario.game_mut().skip_trivial = skip_trivial;
        let attacker = scenario.force_permanent_on_battlefield(0, "Craw Wurm");
        scenario.game_mut().state.permanents[attacker]
            .as_mut()
            .unwrap()
            .summoning_sick = false;
        let ogre1 = scenario.force_permanent_on_battlefield(1, "Gray Ogre");
        let giant = scenario.force_permanent_on_battlefield(1, "Hill Giant");
        let ogre2 = scenario.force_permanent_on_battlefield(1, "Gray Ogre");
        scenario.advance_to_active_step(0, StepKind::DeclareAttackers);
        scenario.declare_attack();
        scenario.advance_to_active_step(0, StepKind::DeclareBlockers);

        let root = scenario.game().clone();
        let combat = root.state.combat.as_ref().expect("combat active");
        let first = root
            .action_space()
            .unwrap()
            .actions
            .iter()
            .find_map(|a| match a {
                Action::DeclareBlocker { blocker, .. } => Some(*blocker),
                _ => None,
            })
            .unwrap();
        let engine_order: Vec<_> = std::iter::once(first)
            .chain(combat.blockers_to_declare.iter().rev().copied())
            .collect();
        assert_eq!(engine_order.len(), 3);
        // Verify the distinct blocker sits strictly between the two identical Gray Ogres.
        assert_eq!(engine_order[1], giant);
        assert!(
            (engine_order[0] == ogre1 && engine_order[2] == ogre2)
                || (engine_order[0] == ogre2 && engine_order[2] == ogre1)
        );

        let offers = root.compound_offers().unwrap();
        let offer = &offers.projection().offers[0];
        assert_eq!(offer.verb, DecisionVerb::DeclareBlockers);
        // Role 0: count role for [Ogre1, Ogre2]; Role 1: singleton role for Hill Giant.
        assert_eq!(offer.choices.len(), 2);
        let ChoiceStep::Select {
            role: ogre_role,
            candidates: ogre_cands,
            ..
        } = &offer.choices[0];
        let ChoiceStep::Select {
            role: giant_role,
            candidates: giant_cands,
            ..
        } = &offer.choices[1];
        let ogre_list = ogre_cands.initial.as_ref().unwrap();
        let giant_list = giant_cands.initial.as_ref().unwrap();
        assert_eq!(ogre_list.len(), 2);
        assert_eq!(ogre_list[0].label, "1 × Gray Ogre → Craw Wurm");
        assert_eq!(ogre_list[1].label, "2 × Gray Ogre → Craw Wurm");
        assert_eq!(giant_list.len(), 1);

        // Exhaust all 6 schema-valid combinations (0/1/2 Ogres × 0/1 Hill Giant).
        let events_from = root.state.observation_events.len();
        for ogre_count in 0..=2usize {
            for giant_blocks in [false, true] {
                let ogre_ans = if ogre_count == 0 {
                    vec![]
                } else {
                    vec![ogre_list[ogre_count - 1].id]
                };
                let giant_ans = if giant_blocks {
                    vec![giant_list[0].id]
                } else {
                    vec![]
                };
                let submission = DecisionSubmission {
                    offer_id: offer.id,
                    answers: vec![
                        ChoiceAnswer::Candidates {
                            role: *ogre_role,
                            candidates: ogre_ans,
                        },
                        ChoiceAnswer::Candidates {
                            role: *giant_role,
                            candidates: giant_ans,
                        },
                    ],
                };
                let mut applied = root.clone();
                applied
                    .apply_offer_submission(&offers, &submission)
                    .unwrap_or_else(|err| {
                        panic!(
                            "skip_trivial={skip_trivial}, ogre_count={ogre_count}, \
                             giant_blocks={giant_blocks} failed: {err}"
                        )
                    });
                assert_ne!(
                    applied.action_space().map(|space| space.kind),
                    Some(ActionSpaceKind::DeclareBlocker),
                    "skip_trivial={skip_trivial}, ogre_count={ogre_count}, \
                     giant_blocks={giant_blocks} left a blocker undeclared"
                );
                let expected_total = ogre_count + usize::from(giant_blocks);
                let actual_blockers: usize = applied.state.observation_events[events_from..]
                    .iter()
                    .find_map(|ev| match ev {
                        managym::flow::event::GameEvent::BlockersDeclared { assignments } => Some(
                            assignments
                                .iter()
                                .map(|(_, blockers)| blockers.len())
                                .sum::<usize>(),
                        ),
                        _ => None,
                    })
                    .unwrap_or(0);
                assert_eq!(actual_blockers, expected_total);
                let commands = root.compound_commands(&offers, &submission).unwrap();
                assert!(!commands.is_empty());
            }
        }

        // 2 attackers (per-blocker path): interleaved [Gray Ogre, Hill Giant, Gray Ogre]
        // facing 2 distinct attackers (Craw Wurm + Hill Giant).
        let mut two = Scenario::new(
            BTreeMap::from([
                ("Forest".to_string(), 15),
                ("Mountain".to_string(), 15),
                ("Craw Wurm".to_string(), 5),
                ("Hill Giant".to_string(), 5),
            ]),
            BTreeMap::from([
                ("Mountain".to_string(), 20),
                ("Gray Ogre".to_string(), 10),
                ("Hill Giant".to_string(), 10),
            ]),
            997,
        );
        two.advance_to_active_step(0, StepKind::Main);
        two.game_mut().skip_trivial = skip_trivial;
        let att0 = two.force_permanent_on_battlefield(0, "Craw Wurm");
        let att1 = two.force_permanent_on_battlefield(0, "Hill Giant");
        for att in [att0, att1] {
            two.game_mut().state.permanents[att]
                .as_mut()
                .unwrap()
                .summoning_sick = false;
        }
        let b_ogre1 = two.force_permanent_on_battlefield(1, "Gray Ogre");
        let b_giant = two.force_permanent_on_battlefield(1, "Hill Giant");
        let b_ogre2 = two.force_permanent_on_battlefield(1, "Gray Ogre");
        two.advance_to_active_step(0, StepKind::DeclareAttackers);
        two.declare_attack();
        two.declare_attack();
        two.advance_to_active_step(0, StepKind::DeclareBlockers);

        let root2 = two.game().clone();
        let combat2 = root2.state.combat.as_ref().expect("combat active");
        let first2 = root2
            .action_space()
            .unwrap()
            .actions
            .iter()
            .find_map(|a| match a {
                Action::DeclareBlocker { blocker, .. } => Some(*blocker),
                _ => None,
            })
            .unwrap();
        let engine_order2: Vec<_> = std::iter::once(first2)
            .chain(combat2.blockers_to_declare.iter().rev().copied())
            .collect();
        assert_eq!(engine_order2[1], b_giant);
        assert!(
            (engine_order2[0] == b_ogre1 && engine_order2[2] == b_ogre2)
                || (engine_order2[0] == b_ogre2 && engine_order2[2] == b_ogre1)
        );

        let offers2 = root2.compound_offers().unwrap();
        let offer2 = &offers2.projection().offers[0];
        assert_eq!(offer2.verb, DecisionVerb::DeclareBlockers);
        // Grouped roles: [Ogre1, Ogre2, Hill Giant].
        assert_eq!(offer2.choices.len(), 3);
        let role_cands: Vec<(RoleId, &[Candidate])> = offer2
            .choices
            .iter()
            .map(
                |ChoiceStep::Select {
                     role, candidates, ..
                 }| (*role, candidates.initial.as_deref().unwrap()),
            )
            .collect();

        // Test representative interleaved plans:
        // (Ogre1 -> att0, Ogre2 -> decline, Giant -> decline)
        // (Ogre1 -> decline, Ogre2 -> att0, Giant -> att1)
        // (Ogre1 -> att0, Ogre2 -> att1, Giant -> decline)
        // (all decline)
        let plans: [([Option<usize>; 3], usize); 4] = [
            ([Some(0), None, None], 1),
            ([None, Some(0), Some(1)], 2),
            ([Some(0), Some(1), None], 2),
            ([None, None, None], 0),
        ];
        let events_from2 = root2.state.observation_events.len();
        for (picks, expected_total) in plans {
            let answers = picks
                .iter()
                .enumerate()
                .map(|(idx, pick)| ChoiceAnswer::Candidates {
                    role: role_cands[idx].0,
                    candidates: pick
                        .map(|p| vec![role_cands[idx].1[p].id])
                        .unwrap_or_default(),
                })
                .collect();
            let submission = DecisionSubmission {
                offer_id: offer2.id,
                answers,
            };
            let mut applied = root2.clone();
            applied
                .apply_offer_submission(&offers2, &submission)
                .unwrap_or_else(|err| {
                    panic!("skip_trivial={skip_trivial}, picks={picks:?} failed: {err}")
                });
            assert_ne!(
                applied.action_space().map(|space| space.kind),
                Some(ActionSpaceKind::DeclareBlocker),
                "skip_trivial={skip_trivial}, picks={picks:?} left a blocker undeclared"
            );
            let total_blocks: usize = applied.state.observation_events[events_from2..]
                .iter()
                .find_map(|ev| match ev {
                    managym::flow::event::GameEvent::BlockersDeclared { assignments } => Some(
                        assignments
                            .iter()
                            .map(|(_, blockers)| blockers.len())
                            .sum::<usize>(),
                    ),
                    _ => None,
                })
                .unwrap_or(0);
            assert_eq!(total_blocks, expected_total);
        }
    }
}

#[test]
fn compound_waterbend_coupled_multi_class_falls_back_and_exact_cases_succeed() {
    // 1. Announced coupled multi-class Waterbend (Water Tribe Rallier already tapped,
    // 2 Gray Ogre + 2 Hill Giant, {3} floating -> Waterbend {5} needs min=2, max=4,
    // total_legal=4 across two multi-member classes). Independent roles would admit
    // illegal combinations (e.g. 0+0=0 < min 2), so compound_offers() falls back to
    // the folded sequential ActionSpace (1 WaterbendTap per class).
    let mut scenario = Scenario::new(
        BTreeMap::from([
            ("Forest".to_string(), 16),
            ("Water Tribe Rallier".to_string(), 8),
            ("Gray Ogre".to_string(), 8),
            ("Hill Giant".to_string(), 8),
        ]),
        deck_of("Mountain", "Gray Ogre"),
        998,
    );
    scenario.advance_to_active_step(0, StepKind::Main);
    let rallier = scenario.force_permanent_on_battlefield(0, "Water Tribe Rallier");
    scenario.game_mut().state.permanents[rallier]
        .as_mut()
        .unwrap()
        .tapped = true;
    for _ in 0..2 {
        scenario.force_permanent_on_battlefield(0, "Gray Ogre");
        scenario.force_permanent_on_battlefield(0, "Hill Giant");
    }
    scenario.game_mut().state.players[0]
        .mana_pool
        .add(&managym::state::mana::Mana::parse("GGG"), None);
    scenario.game_mut().scenario_refresh_priority().unwrap();
    assert!(
        scenario.take_action_by_type(managym::agent::action::ActionType::PriorityActivateAbility)
    );
    assert_eq!(scenario.action_space().kind, ActionSpaceKind::Waterbend);

    let root = scenario.game().clone();
    let offers = root.compound_offers().unwrap();
    assert_eq!(
        offers.projection().offers.len(),
        root.action_space().unwrap().actions.len()
    );
    let tap_offers: Vec<_> = offers
        .projection()
        .offers
        .iter()
        .filter(|offer| offer.verb == DecisionVerb::WaterbendTap)
        .collect();
    assert_eq!(
        tap_offers.len(),
        2,
        "folded sequential space offers 1 tap per equivalence class (Gray Ogre, Hill Giant)"
    );
    assert!(offers
        .projection()
        .offers
        .iter()
        .all(|offer| offer.verb != DecisionVerb::PayWaterbend && offer.choices.is_empty()));
    // Every projected offer decodes and applies cleanly (RaggedPolicyDecoder contract).
    for offer in &tap_offers {
        let mut applied = root.clone();
        applied
            .apply_offer_submission(
                &offers,
                &DecisionSubmission {
                    offer_id: offer.id,
                    answers: vec![],
                },
            )
            .unwrap();
        assert_eq!(
            applied.action_space().unwrap().kind,
            ActionSpaceKind::Waterbend
        );
    }

    // 2. Single-class count role (Rallier tapped, 4 Gray Ogres, {3} floating ->
    // Waterbend {5} needs min=2, max=4 in a single class): 1 Count role where every
    // schema-valid submission decodes and applies cleanly.
    let mut single = Scenario::new(
        BTreeMap::from([
            ("Forest".to_string(), 16),
            ("Water Tribe Rallier".to_string(), 8),
            ("Gray Ogre".to_string(), 16),
        ]),
        deck_of("Mountain", "Gray Ogre"),
        999,
    );
    single.advance_to_active_step(0, StepKind::Main);
    let s_rallier = single.force_permanent_on_battlefield(0, "Water Tribe Rallier");
    single.game_mut().state.permanents[s_rallier]
        .as_mut()
        .unwrap()
        .tapped = true;
    for _ in 0..4 {
        single.force_permanent_on_battlefield(0, "Gray Ogre");
    }
    single.game_mut().state.players[0]
        .mana_pool
        .add(&managym::state::mana::Mana::parse("GGG"), None);
    single.game_mut().scenario_refresh_priority().unwrap();
    assert!(single.take_action_by_type(managym::agent::action::ActionType::PriorityActivateAbility));
    assert_eq!(single.action_space().kind, ActionSpaceKind::Waterbend);

    let single_root = single.game().clone();
    let single_offers = single_root.compound_offers().unwrap();
    let single_offer = &single_offers.projection().offers[0];
    assert_eq!(single_offer.verb, DecisionVerb::PayWaterbend);
    assert_eq!(single_offer.choices.len(), 1);
    let ChoiceStep::Select {
        role,
        min,
        max,
        candidates,
        ..
    } = &single_offer.choices[0];
    assert_eq!((*min, *max), (1, 1));
    let list = candidates.initial.as_ref().unwrap();
    assert_eq!(list.len(), 3);
    assert_eq!(list[0].label, "2 × Gray Ogre");
    assert_eq!(list[1].label, "3 × Gray Ogre");
    assert_eq!(list[2].label, "4 × Gray Ogre");
    for cand in list {
        let mut applied = single_root.clone();
        applied
            .apply_offer_submission(
                &single_offers,
                &DecisionSubmission {
                    offer_id: single_offer.id,
                    answers: vec![ChoiceAnswer::Candidates {
                        role: *role,
                        candidates: vec![cand.id],
                    }],
                },
            )
            .unwrap();
        assert_ne!(
            applied.action_space().map(|space| space.kind),
            Some(ActionSpaceKind::Waterbend)
        );
    }

    // 3. Resolving Waterbend {2} (Waterbending Lesson with 2 Gray Ogre + 2 Hill Giant):
    // compound_offers() falls back cleanly to the folded sequential space (no WrongDecision).
    let mut lesson = Scenario::new(
        BTreeMap::from([
            ("Island".to_string(), 16),
            ("Waterbending Lesson".to_string(), 8),
            ("Gray Ogre".to_string(), 8),
            ("Hill Giant".to_string(), 8),
        ]),
        deck_of("Mountain", "Gray Ogre"),
        1000,
    );
    lesson.advance_to_active_step(0, StepKind::Main);
    lesson.game_mut().scenario_clear_hand(PlayerId(0));
    lesson.force_card_in_hand(0, "Waterbending Lesson");
    for _ in 0..2 {
        lesson.force_permanent_on_battlefield(0, "Gray Ogre");
        lesson.force_permanent_on_battlefield(0, "Hill Giant");
    }
    lesson.game_mut().state.players[0]
        .mana_pool
        .add(&managym::state::mana::Mana::parse("3U"), None);
    lesson.game_mut().scenario_refresh_priority().unwrap();
    assert!(lesson.take_action_by_type(managym::agent::action::ActionType::PriorityCastSpell));
    pass_priority(lesson.game_mut());
    pass_priority(lesson.game_mut());
    assert_eq!(lesson.action_space().kind, ActionSpaceKind::Waterbend);
    let lesson_offers = lesson.game().compound_offers().unwrap();
    assert_eq!(
        lesson_offers.projection().offers.len(),
        lesson.action_space().actions.len()
    );
    assert_eq!(
        lesson_offers
            .projection()
            .offers
            .iter()
            .filter(|o| o.verb == DecisionVerb::WaterbendTap)
            .count(),
        2
    );
}
