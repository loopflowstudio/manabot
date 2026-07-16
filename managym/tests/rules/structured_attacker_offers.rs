use std::collections::{BTreeMap, BTreeSet};

use managym::{
    agent::{
        action::{Action, ActionSpaceKind},
        observation::Observation,
        observation_encoder::ObservationEncoderConfig,
        structured_offer::{
            Candidate, CandidateId, CandidateSource, CandidateSourceId, CandidateValue,
            ChoiceAnswer, ChoiceStep, InteractionOffer, ObjectRenderId, OfferId, OfferSubmission,
            OfferVerb, PromptKind, RoleId, StructuredOfferError, StructuredOfferProjection,
            SubjectRef,
        },
    },
    benchmark::snapshot,
    flow::turn::StepKind,
    state::game_object::PermanentId,
    Game,
};

use super::helpers::Scenario;

fn ur_lessons_deck() -> BTreeMap<String, usize> {
    BTreeMap::from([
        ("Island".to_string(), 9),
        ("Mountain".to_string(), 8),
        ("Tiger-Seal".to_string(), 2),
        ("Otter-Penguin".to_string(), 2),
        ("Fire Nation Cadets".to_string(), 2),
        ("First-Time Flyer".to_string(), 2),
        ("Forecasting Fortune Teller".to_string(), 1),
        ("Dragonfly Swarm".to_string(), 1),
        ("Firebending Lesson".to_string(), 4),
        ("Igneous Inspiration".to_string(), 2),
        ("Pop Quiz".to_string(), 2),
        ("Divide by Zero".to_string(), 2),
        ("It'll Quench Ya!".to_string(), 2),
        ("Accumulate Wisdom".to_string(), 2),
    ])
}

fn gw_allies_deck() -> BTreeMap<String, usize> {
    BTreeMap::from([
        ("Plains".to_string(), 9),
        ("Forest".to_string(), 8),
        ("Water Tribe Rallier".to_string(), 2),
        ("Invasion Reinforcements".to_string(), 2),
        ("Compassionate Healer".to_string(), 2),
        ("Earth Kingdom Jailer".to_string(), 2),
        ("White Lotus Reinforcements".to_string(), 2),
        ("Earth King's Lieutenant".to_string(), 2),
        ("Kyoshi Warriors".to_string(), 2),
        ("Badgermole Cub".to_string(), 2),
        ("Suki, Kyoshi Warrior".to_string(), 1),
        ("South Pole Voyager".to_string(), 1),
        ("Allies at Last".to_string(), 2),
        ("Yip Yip!".to_string(), 1),
        ("Fancy Footwork".to_string(), 2),
    ])
}

fn arranged_attack_scenario(
    active_deck: BTreeMap<String, usize>,
    defending_deck: BTreeMap<String, usize>,
    attacker_names: &[&str],
    seed: u64,
) -> Scenario {
    let mut scenario = Scenario::new(active_deck, defending_deck, seed);
    scenario.advance_to_active_step(0, StepKind::Main);
    let attackers: Vec<PermanentId> = attacker_names
        .iter()
        .map(|name| scenario.force_permanent_on_battlefield(0, name))
        .collect();
    for attacker in attackers {
        let permanent = scenario.game_mut().state.permanents[attacker]
            .as_mut()
            .expect("arranged attacker should exist");
        permanent.summoning_sick = false;
        permanent.tapped = false;
    }
    scenario.advance_to_active_step(0, StepKind::DeclareAttackers);
    assert_eq!(
        scenario.action_space().kind,
        ActionSpaceKind::DeclareAttacker
    );
    scenario
}

fn attacker_offer(set: &managym::agent::structured_offer::StructuredOfferSet) -> &InteractionOffer {
    let projection = set.projection();
    assert_eq!(projection.kind, PromptKind::DeclareAttackers);
    assert_eq!(projection.offers.len(), 1);
    let offer = &projection.offers[0];
    assert_eq!(offer.actor, projection.actor);
    assert_eq!(offer.verb, OfferVerb::DeclareAttackers);
    offer
}

fn attacker_candidates(offer: &InteractionOffer) -> (RoleId, &[Candidate]) {
    let [ChoiceStep::Select {
        role,
        candidates,
        min,
        max,
        ordered,
        distinct,
        ..
    }] = offer.choices.as_slice()
    else {
        panic!("attacker offer should contain one select step")
    };
    let candidates = candidates.initial.as_deref().expect("initial candidates");
    assert_eq!(*min, 0);
    assert_eq!(usize::from(*max), candidates.len());
    assert!(!ordered);
    assert!(distinct);
    (*role, candidates)
}

fn subject_entity(candidate: &Candidate) -> u32 {
    let CandidateValue::Subject {
        subject: SubjectRef::Object { id },
    } = candidate.value
    else {
        panic!("attacker candidate should name a permanent")
    };
    id.entity
}

fn submission_for_mask(
    offer: &InteractionOffer,
    role: RoleId,
    candidates: &[Candidate],
    mask: usize,
) -> (OfferSubmission, BTreeSet<u32>) {
    let mut selected_candidates = Vec::new();
    let mut selected_entities = BTreeSet::new();
    for (index, candidate) in candidates.iter().enumerate() {
        if mask & (1 << index) != 0 {
            selected_candidates.push(candidate.id);
            selected_entities.insert(subject_entity(candidate));
        }
    }
    (
        OfferSubmission {
            offer_id: offer.id,
            answers: vec![ChoiceAnswer::Candidates {
                role,
                candidates: selected_candidates,
            }],
        },
        selected_entities,
    )
}

/// Translate one complete structured declaration back through the preserved
/// fixed-action ABI. This deliberately uses only the public legacy prompts;
/// it is the independent oracle for the new atomic command path.
fn apply_legacy_attacker_adapter(game: &mut Game, selected_entities: &BTreeSet<u32>) {
    let mut seen_entities = BTreeSet::new();
    while game
        .action_space()
        .is_some_and(|space| space.kind == ActionSpaceKind::DeclareAttacker)
    {
        let space = game.action_space().expect("legacy attacker prompt");
        let permanent_id = space
            .actions
            .iter()
            .find_map(|action| match action {
                Action::DeclareAttacker { permanent, .. } => Some(*permanent),
                _ => None,
            })
            .expect("legacy attacker prompt should name a permanent");
        let entity = game.state.permanents[permanent_id]
            .as_ref()
            .expect("legacy attacker should exist")
            .id
            .0;
        seen_entities.insert(entity);
        let attack = selected_entities.contains(&entity);
        let action_index = space
            .actions
            .iter()
            .position(|action| {
                matches!(
                    action,
                    Action::DeclareAttacker {
                        permanent,
                        attack: offered,
                        ..
                    } if *permanent == permanent_id && *offered == attack
                )
            })
            .expect("legacy ABI should represent attack and decline");
        game.step(action_index)
            .expect("legacy attacker action should apply");
    }
    assert!(
        selected_entities.is_subset(&seen_entities),
        "structured selection must be representable by legacy prompts"
    );
}

fn assert_equivalent_surface(structured: &Game, legacy: &Game) {
    assert_eq!(snapshot(structured), snapshot(legacy));
}

#[test]
fn structured_attacker_offer_exhausts_two_deck_declarations_past_32() {
    let fixtures = [
        (
            gw_allies_deck(),
            ur_lessons_deck(),
            vec![
                "Water Tribe Rallier",
                "Water Tribe Rallier",
                "Invasion Reinforcements",
                "Invasion Reinforcements",
                "Compassionate Healer",
                "Compassionate Healer",
            ],
        ),
        (
            ur_lessons_deck(),
            gw_allies_deck(),
            vec![
                "Tiger-Seal",
                "Tiger-Seal",
                "Otter-Penguin",
                "Otter-Penguin",
                "Fire Nation Cadets",
                "Fire Nation Cadets",
            ],
        ),
    ];

    for (fixture_index, (active, defending, attacker_names)) in fixtures.into_iter().enumerate() {
        let root = arranged_attack_scenario(
            active,
            defending,
            &attacker_names,
            188 + fixture_index as u64,
        )
        .game()
        .clone();
        let set = root.structured_offers().expect("structured attacker offer");
        let offer = attacker_offer(&set);
        let (role, candidates) = attacker_candidates(offer);
        assert_eq!(candidates.len(), 6);
        assert_eq!(
            candidates
                .iter()
                .map(|candidate| candidate.id)
                .collect::<BTreeSet<_>>()
                .len(),
            candidates.len()
        );

        let declaration_count = 1_usize << candidates.len();
        assert_eq!(declaration_count, 64);
        assert!(
            declaration_count > ObservationEncoderConfig::default().max_actions,
            "fixture must exceed the legacy tensor width"
        );
        assert_eq!(
            root.action_space().expect("legacy prompt").actions.len(),
            2,
            "the preserved ABI remains sequential instead of enumerating subsets"
        );

        // Six candidates encode all 64 subsets. Execute every one through the
        // atomic path and independently through the old binary prompt chain.
        for mask in 0..declaration_count {
            let (submission, selected_entities) =
                submission_for_mask(offer, role, candidates, mask);
            let mut structured = root.clone();
            let mut legacy = root.clone();
            structured
                .apply_offer_submission(&set, &submission)
                .expect("every projected declaration should be legal");
            apply_legacy_attacker_adapter(&mut legacy, &selected_entities);
            assert_equivalent_surface(&structured, &legacy);
        }
    }
}

#[test]
fn structured_attacker_offer_is_prompt_bound_and_rejects_bad_selections() {
    let mut root = arranged_attack_scenario(
        gw_allies_deck(),
        ur_lessons_deck(),
        &[
            "Water Tribe Rallier",
            "Invasion Reinforcements",
            "Compassionate Healer",
        ],
        190,
    )
    .game()
    .clone();
    let set = root.structured_attacker_offers().expect("attacker offer");
    let offer = attacker_offer(&set);
    let (role, candidates) = attacker_candidates(offer);

    assert_eq!(
        set.decode(&OfferSubmission {
            offer_id: offer.id,
            answers: Vec::new(),
        }),
        Err(StructuredOfferError::MissingAnswer(role))
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
            answers: vec![ChoiceAnswer::Candidates {
                role,
                candidates: vec![candidates[0].id, candidates[0].id],
            }],
        }),
        Err(StructuredOfferError::DuplicateCandidate(candidates[0].id))
    );

    let (empty_submission, _) = submission_for_mask(offer, role, candidates, 0);
    let legacy_decline = root
        .action_space()
        .expect("legacy prompt")
        .actions
        .iter()
        .position(|action| matches!(action, Action::DeclareAttacker { attack: false, .. }))
        .expect("legacy decline");
    root.step(legacy_decline)
        .expect("advance to a later attacker prompt");
    let before = format!("{root:?}");
    assert!(matches!(
        root.apply_offer_submission(&set, &empty_submission),
        Err(StructuredOfferError::StaleOrIllegal(_))
    ));
    assert_eq!(
        format!("{root:?}"),
        before,
        "rejection must not mutate state"
    );
}

#[test]
fn structured_attacker_adapter_matches_seeded_two_deck_traces() {
    for (case, (active, defending)) in [
        (gw_allies_deck(), ur_lessons_deck()),
        (ur_lessons_deck(), gw_allies_deck()),
    ]
    .into_iter()
    .enumerate()
    {
        for seed_offset in 0..2_u64 {
            let seed = 0x1880 + case as u64 * 16 + seed_offset;
            let mut structured = Game::new(
                vec![
                    managym::PlayerConfig::new("active", active.clone()),
                    managym::PlayerConfig::new("defending", defending.clone()),
                ],
                seed,
                false,
            );
            let mut legacy = structured.clone();
            let mut structured_trace = Vec::new();
            let mut legacy_trace = Vec::new();
            let mut checkpoints = 0_usize;

            while !structured.is_game_over() {
                assert_eq!(structured.is_game_over(), legacy.is_game_over());
                assert_eq!(structured.current_action_space, legacy.current_action_space);
                let space = structured.action_space().expect("live action space");

                if space.kind == ActionSpaceKind::DeclareAttacker {
                    let set = structured
                        .structured_offers()
                        .expect("structured attacker offer in acceptance game");
                    let offer = attacker_offer(&set);
                    let (role, candidates) = attacker_candidates(offer);
                    let mut mask = 0_usize;
                    for (index, candidate) in candidates.iter().enumerate() {
                        let bit = seed
                            .wrapping_add(checkpoints as u64 * 0x9e37)
                            .wrapping_add(u64::from(candidate.id.0) * 0x85eb)
                            .count_ones()
                            & 1;
                        if bit == 1 {
                            mask |= 1 << index;
                        }
                    }
                    let (submission, selected_entities) =
                        submission_for_mask(offer, role, candidates, mask);
                    structured
                        .apply_offer_submission(&set, &submission)
                        .expect("seeded structured declaration");
                    apply_legacy_attacker_adapter(&mut legacy, &selected_entities);
                    // The two engines took different ABI paths at this
                    // boundary, so compare the complete canonical state.
                    assert_equivalent_surface(&structured, &legacy);
                } else {
                    let action_count = space.actions.len();
                    assert!(action_count > 0);
                    let action =
                        seed.wrapping_add(checkpoints as u64 * 0x9e37_79b9)
                            .wrapping_mul(0xbf58_476d_1ce4_e5b9) as usize
                            % action_count;
                    structured
                        .step(action)
                        .expect("structured-side legacy step");
                    legacy.step(action).expect("legacy-side step");
                }

                structured_trace.push(
                    Observation::new(&structured, &structured.state.observation_events).to_json(),
                );
                legacy_trace
                    .push(Observation::new(&legacy, &legacy.state.observation_events).to_json());
                assert_eq!(structured_trace.last(), legacy_trace.last());
                assert_eq!(structured.current_action_space, legacy.current_action_space);
                assert_eq!(structured.state.events, legacy.state.events);
                assert_eq!(structured.state.pending_events, legacy.state.pending_events);
                assert_eq!(
                    structured.state.observation_events,
                    legacy.state.observation_events
                );
                checkpoints += 1;
                assert!(
                    checkpoints < 50_000,
                    "seed {seed}: acceptance game appears stuck"
                );
            }

            assert!(legacy.is_game_over());
            assert_eq!(structured.winner_index(), legacy.winner_index());
            assert_eq!(structured_trace, legacy_trace);
            assert_equivalent_surface(&structured, &legacy);
        }
    }
}

#[test]
fn structured_attacker_offer_fixture_matches_typed_wire_shape() {
    let labels = [
        "Water Tribe Rallier",
        "Water Tribe Rallier",
        "Invasion Reinforcements",
        "Invasion Reinforcements",
        "Compassionate Healer",
        "Compassionate Healer",
    ];
    let candidates = labels
        .iter()
        .enumerate()
        .map(|(index, label)| Candidate {
            id: CandidateId(index as u32),
            value: CandidateValue::Subject {
                subject: SubjectRef::Object {
                    id: ObjectRenderId {
                        entity: 201 + index as u32,
                        incarnation: 0,
                    },
                },
            },
            label: (*label).to_string(),
            help: None,
            preview: None,
        })
        .collect();
    let projection = StructuredOfferProjection {
        actor: 0,
        kind: PromptKind::DeclareAttackers,
        offers: vec![InteractionOffer {
            id: OfferId(0),
            actor: 0,
            verb: OfferVerb::DeclareAttackers,
            source: None,
            label: "Declare attackers".to_string(),
            help: None,
            choices: vec![ChoiceStep::Select {
                role: RoleId(1),
                label: "Attackers".to_string(),
                candidates: CandidateSource {
                    id: CandidateSourceId(0),
                    depends_on: Vec::new(),
                    initial: Some(candidates),
                },
                min: 0,
                max: 6,
                ordered: false,
                distinct: true,
            }],
            confirm_label: "Declare attackers".to_string(),
        }],
    };

    let fixture = include_str!("../fixtures/structured_declare_attackers_offer.json");
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
