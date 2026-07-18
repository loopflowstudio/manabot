use std::collections::{BTreeMap, BTreeSet};

use managym::{
    agent::{
        action::{Action, ActionSpaceKind},
        observation::Observation,
        observation_encoder::ObservationEncoderConfig,
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
    assert!(candidates.len() > ObservationEncoderConfig::default().max_actions);
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
