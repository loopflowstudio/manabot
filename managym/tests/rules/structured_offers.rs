use std::collections::{BTreeMap, BTreeSet};

use managym::{
    agent::{
        action::{Action, ActionSpaceKind},
        observation::Observation,
        structured_offer::{
            AtomicCommand, BoundTarget, Candidate, CandidateId, CandidateSource, CandidateSourceId,
            CandidateValue, ChoiceAnswer, ChoiceStep, InteractionOffer, ObjectRenderId, OfferId,
            OfferSubmission, OfferVerb, PromptKind, RoleId, StructuredOfferError,
            StructuredOfferProjection, SubjectRef,
        },
    },
    flow::turn::StepKind,
    state::{
        game_object::{CardId, PlayerId, Target},
        stack_object::StackObject,
        target::Target as ActionTarget,
    },
    Game,
};

use super::helpers::*;

fn bolt_priority_scenario() -> Scenario {
    let mut scenario = Scenario::new(bolt_deck(), ogre_deck(), 181);
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

fn cast_offer(set: &managym::agent::structured_offer::StructuredOfferSet) -> &InteractionOffer {
    set.projection()
        .offers
        .iter()
        .find(|offer| offer.verb == OfferVerb::Cast)
        .expect("Bolt cast offer")
}

fn pass_offer(set: &managym::agent::structured_offer::StructuredOfferSet) -> &InteractionOffer {
    set.projection()
        .offers
        .iter()
        .find(|offer| offer.verb == OfferVerb::PassPriority)
        .expect("pass-priority offer")
}

fn select_step(offer: &InteractionOffer) -> (RoleId, &[Candidate]) {
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

fn legacy_cast_at_player(game: &mut Game, card: CardId, player: PlayerId) {
    let cast_index = game
        .action_space()
        .expect("priority action space")
        .actions
        .iter()
        .position(
            |action| matches!(action, Action::CastSpell { card: legal, .. } if *legal == card),
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
    assert_eq!(structured.current_action_space, legacy.current_action_space);
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

    let submission = OfferSubmission {
        offer_id: offer.id,
        answers: vec![ChoiceAnswer::Candidates {
            role,
            candidates: vec![player_candidate(candidates, 1)],
        }],
    };
    let command = set
        .decode(&submission)
        .expect("offered target should decode");
    let AtomicCommand::CastSpell { card, targets, .. } = &command else {
        panic!("expected cast command")
    };
    assert_eq!(targets, &[BoundTarget::Player(PlayerId(1))]);

    let mut structured = root.clone();
    let mut legacy = root;
    assert!(!structured
        .apply_offer_submission(&set, &submission)
        .expect("atomic cast should apply"));
    legacy_cast_at_player(&mut legacy, *card, PlayerId(1));

    assert!(structured.pending_choice.is_none());
    let Some(StackObject::Spell(spell)) = structured.state.stack_objects.last() else {
        panic!("Bolt should be on the stack")
    };
    assert_eq!(spell.card, *card);
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
    let submission = OfferSubmission {
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
        bolt_deck(),
        BTreeMap::from([("Gray Ogre".to_string(), 40)]),
        182,
    );
    scenario.advance_to_active_step(0, StepKind::Main);
    scenario.game_mut().scenario_clear_hand(PlayerId(0));
    scenario.force_card_in_hand(0, "Lightning Bolt");
    scenario.force_permanent_on_battlefield(0, "Mountain");
    for _ in 0..33 {
        scenario.force_permanent_on_battlefield(1, "Gray Ogre");
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
    assert_eq!(candidates.len(), 35, "33 creatures plus both players");
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
fn structured_offer_rejects_fabricated_and_stale_ids_without_mutation() {
    let mut root = bolt_priority_scenario().game().clone();
    let set = root
        .structured_priority_offers()
        .expect("structured priority offers");
    let offer = cast_offer(&set);
    let (role, candidates) = select_step(offer);

    assert_eq!(
        set.decode(&OfferSubmission {
            offer_id: OfferId(999),
            answers: Vec::new(),
        }),
        Err(StructuredOfferError::UnknownOffer(OfferId(999)))
    );
    assert_eq!(
        set.decode(&OfferSubmission {
            offer_id: offer.id,
            answers: vec![ChoiceAnswer::Candidates {
                role,
                candidates: vec![CandidateId(999)],
            }],
        }),
        Err(StructuredOfferError::UnknownCandidate(CandidateId(999)))
    );
    assert_eq!(
        set.decode(&OfferSubmission {
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
        Err(StructuredOfferError::DuplicateRole(role))
    );

    let submission = OfferSubmission {
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
        Err(StructuredOfferError::StaleOrIllegal(_))
    ));
    assert_eq!(format!("{root:?}"), before);
}

#[test]
fn structured_offer_game_fixture_matches_typed_wire_shape() {
    let projection = StructuredOfferProjection {
        actor: 0,
        kind: PromptKind::Priority,
        offers: vec![
            InteractionOffer {
                id: OfferId(0),
                actor: 0,
                verb: OfferVerb::Cast,
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
            InteractionOffer {
                id: OfferId(1),
                actor: 0,
                verb: OfferVerb::PassPriority,
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
        serde_json::from_str::<StructuredOfferProjection>(fixture)
            .expect("fixture should deserialize"),
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
    let mut scenario = Scenario::new(ogre_deck(), ogre_deck(), 991);
    scenario.advance_to_active_step(0, StepKind::Main);
    let attacker = scenario.force_permanent_on_battlefield(0, "Gray Ogre");
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
    assert_eq!(offer.verb, OfferVerb::DeclareBlockers);
    assert_eq!(offer.choices.len(), 3);
    let mut answers = Vec::new();
    for ChoiceStep::Select {
        role, candidates, ..
    } in &offer.choices
    {
        answers.push(ChoiceAnswer::Candidates {
            role: *role,
            candidates: vec![candidates.initial.as_ref().unwrap()[0].id],
        });
    }
    let submission = OfferSubmission {
        offer_id: offer.id,
        answers,
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
        Err(StructuredOfferError::StaleOrIllegal(_))
    ));
    let mut invalid = submission.clone();
    invalid.answers.push(invalid.answers[0].clone());
    assert!(matches!(
        offers.decode(&invalid),
        Err(StructuredOfferError::DuplicateRole(_))
    ));
    invalid.answers = vec![ChoiceAnswer::Candidates {
        role: RoleId(0),
        candidates: vec![CandidateId(999)],
    }];
    assert_eq!(
        offers.decode(&invalid),
        Err(StructuredOfferError::UnknownCandidate(CandidateId(999)))
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
            ogre_deck(),
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
        assert_eq!(offer.verb, OfferVerb::PayWaterbend);
        let ChoiceStep::Select {
            role,
            min,
            max,
            candidates,
            ..
        } = &offer.choices[0];
        assert_eq!((*min, *max), (5, 5));
        let mut selected: Vec<_> = candidates
            .initial
            .as_ref()
            .unwrap()
            .iter()
            .map(|candidate| candidate.id)
            .collect();
        let submission = OfferSubmission {
            offer_id: offer.id,
            answers: vec![ChoiceAnswer::Candidates {
                role: *role,
                candidates: selected.clone(),
            }],
        };
        let commands = root.compound_commands(&offers, &submission).unwrap();
        assert_eq!(commands.len(), if skip_trivial { 4 } else { 5 });
        selected.reverse();
        let reversed = OfferSubmission {
            offer_id: offer.id,
            answers: vec![ChoiceAnswer::Candidates {
                role: *role,
                candidates: selected,
            }],
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
    let mut scenario = Scenario::new(boggart_brute_deck(), ogre_deck(), 993);
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
        .all(|offer| offer.verb == OfferVerb::DeclareBlocker && offer.choices.is_empty()));
    for offer in &offers.projection().offers {
        assert_eq!(
            root.compound_commands(
                &offers,
                &OfferSubmission {
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
        ogre_deck(),
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
    assert_eq!(offer.verb, OfferVerb::DeclareBlockers);
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
            &OfferSubmission {
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
            ogre_deck(),
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
            .all(|offer| offer.verb != OfferVerb::PayWaterbend && offer.choices.is_empty()));
    }
}
