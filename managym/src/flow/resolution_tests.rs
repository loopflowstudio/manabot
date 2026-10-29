//! Bounded CR 608.2b witnesses. Synthetic programs isolate resolution semantics;
//! they do not assert these combinations are reachable in an admitted deck.

use std::{collections::BTreeMap, sync::Arc};

use crate::{
    agent::action::Action,
    flow::{game::Game, undo::JournalStats},
    state::{
        ability::{Effect, TargetRequirement, TargetSpec},
        card::{ActivatedAbilityDefinition, CardType, Keywords},
        game_object::{CardId, PermanentId, PlayerId, Target},
        permanent::CounterKind,
        player::PlayerConfig,
        predicate::CardPredicate,
        stack_object::{ActivatedAbilityOnStack, StackObject, TriggeredAbilityOnStack},
        zone::ZoneType,
    },
};

#[derive(Clone, Copy, Debug)]
enum Entry {
    Spell,
    Activated,
    Triggered,
}
const ENTRIES: [Entry; 3] = [Entry::Spell, Entry::Activated, Entry::Triggered];

fn fixture(controller: PlayerId) -> (Game, PermanentId, PermanentId) {
    let deck = BTreeMap::from([
        ("Gray Ogre".into(), 12),
        ("Lightning Bolt".into(), 4),
        ("Mountain".into(), 24),
    ]);
    let mut game = Game::new(
        vec![
            PlayerConfig::new("p0", deck.clone()),
            PlayerConfig::new("p1", deck),
        ],
        608_2026,
        false,
    );
    let opponent = PlayerId(1 - controller.0);
    let first = game
        .scenario_force_battlefield(opponent, "Gray Ogre", true)
        .unwrap();
    let second = game
        .scenario_force_battlefield(opponent, "Gray Ogre", true)
        .unwrap();
    (game, first, second)
}

fn stack(
    game: &mut Game,
    entry: Entry,
    controller: PlayerId,
    mut effects: Vec<Effect>,
    targets: Vec<Target>,
    indices: Vec<usize>,
) {
    if matches!(entry, Entry::Spell) {
        let card = CardId(
            game.state
                .cards
                .iter()
                .position(|card| card.owner == controller && card.name == "Lightning Bolt")
                .unwrap(),
        );
        // Lightning Bolt is only a vessel: the synthetic clauses carry their
        // own requirements, so its printed targeting must not apply.
        game.state.cards[card].spell_effects = effects;
        game.state.cards[card].targeting = Vec::new();
        game.push_spell_to_stack(card, controller, targets, indices, false);
        return;
    }
    let source = game
        .scenario_force_battlefield(controller, "Gray Ogre", true)
        .unwrap();
    let source_card = game.state.permanents[source].as_ref().unwrap().card;
    let source_ref = game.permanent_object_ref(source).unwrap();
    let source_definition_id = game.state.cards[source_card].definition_id;
    let id = game.state.id_gen.next_id();
    let object = match entry {
        Entry::Activated => {
            // Activated announcements currently expose one clause. Multi-clause
            // cases below use the spell/trigger paths that can announce them.
            assert_eq!(effects.len(), 1);
            game.state.cards[source_card].activated_abilities = vec![ActivatedAbilityDefinition {
                effect: effects.remove(0),
                ..Default::default()
            }];
            StackObject::ActivatedAbility(ActivatedAbilityOnStack {
                id,
                controller,
                source_definition_id,
                source_card,
                source_ref,
                ability_index: 0,
                targets,
            })
        }
        Entry::Triggered => StackObject::TriggeredAbility(TriggeredAbilityOnStack {
            id,
            controller,
            source_card,
            source_ref: Some(source_ref),
            source_lki: None,
            source_definition_id,
            ability_index: 0,
            targets,
            target_req_indices: indices,
            context: None,
            inline_effects: Some(effects),
        }),
        Entry::Spell => unreachable!(),
    };
    game.push_to_stack(object);
}

fn clause(spec: TargetSpec, effects: Vec<Effect>) -> Effect {
    Effect::TargetClause {
        requirement: TargetRequirement::one(spec),
        effects,
    }
}

fn damage() -> Effect {
    Effect::DealDamage {
        amount: 1,
        target: TargetSpec::Creature,
    }
}

fn hexproof(game: &mut Game, id: PermanentId, value: bool) {
    game.state.permanents[id]
        .as_mut()
        .unwrap()
        .temp_keywords
        .hexproof = value;
}

fn pause() -> Effect {
    Effect::Modal {
        modes: vec![vec![], vec![Effect::GainLife { amount: 2 }]],
    }
}

#[test]
fn resolution_keeps_eligibility_after_hexproof_and_across_choice_clone_and_undo() {
    for controller in [PlayerId(0), PlayerId(1)] {
        for entry in ENTRIES {
            let (mut game, victim, _) = fixture(controller);
            stack(
                &mut game,
                entry,
                controller,
                vec![clause(
                    TargetSpec::Creature,
                    vec![
                        Effect::GrantKeywordsToTarget {
                            keywords: Keywords {
                                hexproof: true,
                                ..Default::default()
                            },
                            target: TargetSpec::Creature,
                        },
                        pause(),
                        damage(),
                    ],
                )],
                vec![Target::Permanent(victim)],
                vec![0],
            );
            game.resolve_top_of_stack();
            assert!(
                game.state.permanents[victim]
                    .as_ref()
                    .unwrap()
                    .temp_keywords
                    .hexproof
            );
            assert_eq!(game.state.permanents[victim].as_ref().unwrap().damage, 0);
            let space = game.suspended_decision_action_space().unwrap();
            let choice = space
                .actions
                .iter()
                .position(|a| matches!(a, Action::ChooseMode { mode: 0, .. }))
                .unwrap();
            game.publish_action_space(space);
            let captured = game.capture_decision(controller, 0).unwrap();
            let targets = &captured
                .frames
                .iter()
                .find(|f| f.kind == 5)
                .unwrap()
                .targets;
            assert_eq!(targets[0].legal_at_resolution_start, Some(true));
            assert!(game
                .continuation_frames(PlayerId(1 - controller.0))
                .iter()
                .all(|f| f.kind <= 3));
            let mut fork = game.clone();
            assert_eq!(
                fork.capture_decision(controller, 0).unwrap().digest,
                captured.digest
            );
            let before = game.state.deterministic_hash();
            game.enable_undo(Arc::new(JournalStats::default()), 608);
            let mark = game.undo_mark();
            game.step(choice).unwrap();
            assert_eq!(
                game.state.permanents[victim].as_ref().unwrap().damage,
                1,
                "{entry:?}"
            );
            assert!(game.state.suspended_decision.is_none());
            let after = game.state.deterministic_hash();
            fork.step(choice).unwrap();
            assert_eq!(fork.state.deterministic_hash(), after);
            game.undo_rollback(mark);
            assert_eq!(
                game.capture_decision(controller, 0).unwrap().digest,
                captured.digest
            );
            assert_eq!(game.state.deterministic_hash(), before);
            game.step(choice).unwrap();
            assert_eq!(game.state.deterministic_hash(), after);
        }
    }
}

#[test]
fn resolution_all_invalid_suppresses_every_effect_but_mixed_slots_remain_fixed() {
    for controller in [PlayerId(0), PlayerId(1)] {
        for entry in ENTRIES {
            for all_invalid in [false, true] {
                let (mut game, first, second) = fixture(controller);
                stack(
                    &mut game,
                    entry,
                    controller,
                    vec![Effect::TargetClause {
                        requirement: TargetRequirement::up_to(2, TargetSpec::Creature),
                        effects: vec![
                            pause(),
                            Effect::ForEachTarget {
                                effects: vec![damage()],
                            },
                            Effect::GainLife { amount: 7 },
                        ],
                    }],
                    vec![Target::Permanent(first), Target::Permanent(second)],
                    vec![0, 0],
                );
                hexproof(&mut game, first, true);
                hexproof(&mut game, second, all_invalid);
                game.resolve_top_of_stack();
                if !all_invalid {
                    // Becoming legal after entry cannot revive an illegal slot.
                    hexproof(&mut game, first, false);
                    game.execute_decision_action(&Action::ChooseMode {
                        player: controller,
                        mode: 0,
                    })
                    .unwrap();
                }
                assert!(game.state.suspended_decision.is_none());
                assert!(game.state.stack_objects.is_empty());
                assert_eq!(game.state.permanents[first].as_ref().unwrap().damage, 0);
                assert_eq!(
                    game.state.permanents[second].as_ref().unwrap().damage,
                    i32::from(!all_invalid)
                );
                assert_eq!(
                    game.state.players[controller.0].life,
                    if all_invalid { 20 } else { 27 }
                );
            }
        }
    }
}

#[test]
fn resolution_same_object_can_be_illegal_for_only_one_clause() {
    for controller in [PlayerId(0), PlayerId(1)] {
        for entry in [Entry::Spell, Entry::Triggered] {
            let (mut game, victim, _) = fixture(controller);
            let owner = PlayerId(1 - controller.0);
            stack(
                &mut game,
                entry,
                controller,
                vec![
                    clause(TargetSpec::CreatureOpponentControls, vec![damage()]),
                    clause(
                        TargetSpec::Creature,
                        vec![Effect::PutCounters {
                            count: 2,
                            kind: CounterKind::PlusOnePlusOne,
                            target: TargetSpec::Creature,
                        }],
                    ),
                ],
                vec![Target::Permanent(victim); 2],
                vec![0, 1],
            );
            // A response changes control: the same physical target remains legal
            // for the unrestricted clause. Outcome is specified independently.
            game.state.permanents[victim].as_mut().unwrap().controller = controller;
            game.resolve_top_of_stack();
            let permanent = game.state.permanents[victim].as_ref().unwrap();
            assert_eq!(permanent.damage, 0);
            assert_eq!(permanent.plus1_counters, 2);
            assert_eq!(game.state.cards[permanent.card].owner, owner);
        }
    }
}

#[test]
fn resolution_characteristics_changed_by_earlier_instruction_do_not_retarget() {
    for controller in [PlayerId(0), PlayerId(1)] {
        for entry in ENTRIES {
            let (mut game, _, _) = fixture(controller);
            let land = game
                .scenario_force_battlefield(controller, "Mountain", true)
                .unwrap();
            let spec = TargetSpec::PermanentYouControl {
                predicate: CardPredicate {
                    not_card_types: vec![CardType::Creature],
                    ..Default::default()
                },
            };
            stack(
                &mut game,
                entry,
                controller,
                vec![clause(
                    spec.clone(),
                    vec![
                        Effect::Earthbend {
                            count: 1,
                            target: spec.clone(),
                        },
                        Effect::PutCounters {
                            count: 2,
                            kind: CounterKind::PlusOnePlusOne,
                            target: spec,
                        },
                    ],
                )],
                vec![Target::Permanent(land)],
                vec![0],
            );
            game.resolve_top_of_stack();
            let permanent = game.state.permanents[land].as_ref().unwrap();
            assert!(permanent.animated);
            assert_eq!(permanent.plus1_counters, 3);
        }
    }
}

#[test]
fn resolution_zero_optional_targets_and_untargeted_effects_resolve() {
    for controller in [PlayerId(0), PlayerId(1)] {
        for entry in ENTRIES {
            for targeted in [false, true] {
                let (mut game, _, _) = fixture(controller);
                let effects = vec![if targeted {
                    Effect::TargetClause {
                        requirement: TargetRequirement::up_to(1, TargetSpec::Creature),
                        effects: vec![damage(), Effect::GainLife { amount: 7 }],
                    }
                } else {
                    Effect::GainLife { amount: 7 }
                }];
                stack(&mut game, entry, controller, effects, vec![], vec![]);
                game.resolve_top_of_stack();
                assert_eq!(game.state.players[controller.0].life, 27);
                assert!(game.state.stack_objects.is_empty());
            }
        }
    }
}

#[test]
fn resolution_does_not_follow_departed_permanent_or_graveyard_incarnation() {
    for controller in [PlayerId(0), PlayerId(1)] {
        for entry in ENTRIES {
            let (mut game, victim, _) = fixture(controller);
            let card = game.state.permanents[victim].as_ref().unwrap().card;
            stack(
                &mut game,
                entry,
                controller,
                vec![clause(
                    TargetSpec::Creature,
                    vec![
                        Effect::ReturnToHand {
                            target: TargetSpec::Creature,
                        },
                        pause(),
                        damage(),
                    ],
                )],
                vec![Target::Permanent(victim)],
                vec![0],
            );
            game.resolve_top_of_stack();
            assert_eq!(game.state.zones.zone_of(card), Some(ZoneType::Hand));
            // The first instruction moved the original object. A synthetic
            // reentry during the choice must not let the suffix find the card.
            game.move_card(card, ZoneType::Battlefield);
            let replacement = game.state.card_to_permanent[card].unwrap();
            assert_ne!(victim, replacement);
            let space = game.suspended_decision_action_space().unwrap();
            game.publish_action_space(space);
            let frames = game.continuation_frames(controller);
            let slot = &frames.iter().find(|f| f.kind == 5).unwrap().targets[0];
            assert_eq!(slot.legal_at_resolution_start, Some(true));
            assert_eq!(slot.reference.object, None);
            game.execute_decision_action(&Action::ChooseMode {
                player: controller,
                mode: 0,
            })
            .unwrap();
            assert_eq!(
                game.state.permanents[replacement].as_ref().unwrap().damage,
                0
            );

            let (mut game, _, _) = fixture(controller);
            let card = CardId(
                game.state
                    .cards
                    .iter()
                    .position(|c| c.owner == controller && c.name == "Mountain")
                    .unwrap(),
            );
            game.move_card(card, ZoneType::Graveyard);
            let original = game.current_object_ref(card).unwrap();
            let spec = TargetSpec::GraveyardCardYouOwn {
                predicate: CardPredicate::default(),
            };
            stack(
                &mut game,
                entry,
                controller,
                vec![clause(
                    spec.clone(),
                    vec![
                        Effect::ReturnGraveyardTarget {
                            target: spec.clone(),
                        },
                        pause(),
                        Effect::ReturnGraveyardTarget { target: spec },
                    ],
                )],
                vec![Target::GraveyardCard(original)],
                vec![0],
            );
            game.resolve_top_of_stack();
            assert_eq!(game.state.zones.zone_of(card), Some(ZoneType::Hand));
            game.move_card(card, ZoneType::Graveyard);
            assert_ne!(game.current_object_ref(card), Some(original));
            let space = game.suspended_decision_action_space().unwrap();
            game.publish_action_space(space);
            let frames = game.continuation_frames(controller);
            let slot = &frames.iter().find(|f| f.kind == 5).unwrap().targets[0];
            assert_eq!(slot.legal_at_resolution_start, Some(true));
            assert_eq!(slot.reference.object, None);
            game.execute_decision_action(&Action::ChooseMode {
                player: controller,
                mode: 0,
            })
            .unwrap();
            assert_eq!(game.state.zones.zone_of(card), Some(ZoneType::Graveyard));
        }
    }
}

#[test]
fn resolution_positional_power_damage_never_substitutes_an_illegal_victim() {
    for controller in [PlayerId(0), PlayerId(1)] {
        for invalid_victim in [false, true] {
            let (mut game, victim, _) = fixture(controller);
            let attacker = game
                .scenario_force_battlefield(controller, "Gray Ogre", true)
                .unwrap();
            // This is the existing Allies at Last instruction, with its explicit
            // two-clause targeting contract and untargeted suffix for observation.
            stack(
                &mut game,
                Entry::Spell,
                controller,
                vec![
                    Effect::TargetCreaturesDealPowerDamageToLastTarget,
                    Effect::GainLife { amount: 7 },
                ],
                vec![Target::Permanent(attacker), Target::Permanent(victim)],
                vec![0, 1],
            );
            let spell_card = match game.state.stack_objects.last().unwrap() {
                StackObject::Spell(spell) => spell.card,
                _ => unreachable!(),
            };
            game.state.cards[spell_card].targeting = vec![
                TargetRequirement::up_to(2, TargetSpec::CreatureYouControl),
                TargetRequirement::one(TargetSpec::CreatureOpponentControls),
            ];
            hexproof(&mut game, victim, invalid_victim);
            if !invalid_victim {
                game.state.permanents[attacker].as_mut().unwrap().controller =
                    PlayerId(1 - controller.0);
            }
            game.resolve_top_of_stack();
            assert_eq!(game.state.permanents[victim].as_ref().unwrap().damage, 0);
            assert_eq!(game.state.permanents[attacker].as_ref().unwrap().damage, 0);
            assert_eq!(game.state.players[controller.0].life, 27);
        }
    }
}

/// Identical present characteristics cannot reconstruct an earlier eligibility check.
/// Exported captures are portable evidence for the Python text/reference encoder proof.
#[test]
fn resolution_eligibility_distinguishes_identical_present_observations() {
    use crate::agent::{
        observation::Observation,
        observation_encoder::{encode, ObservationEncoderConfig},
    };
    let mut evidence = Vec::new();
    for controller in [PlayerId(0), PlayerId(1)] {
        let mut observations = Vec::new();
        let mut encodings = Vec::new();
        for legal in [true, false] {
            let (mut game, first, second) = fixture(controller);
            hexproof(&mut game, first, !legal);
            stack(
                &mut game,
                Entry::Spell,
                controller,
                vec![
                    clause(
                        TargetSpec::Creature,
                        vec![Effect::GrantKeywordsToTarget {
                            keywords: Keywords {
                                hexproof: true,
                                ..Default::default()
                            },
                            target: TargetSpec::Creature,
                        }],
                    ),
                    clause(TargetSpec::Creature, vec![pause(), damage()]),
                ],
                vec![Target::Permanent(first), Target::Permanent(second)],
                vec![0, 1],
            );
            // Public stack targets have not been evaluated yet.
            assert!(game
                .continuation_frames(controller)
                .iter()
                .flat_map(|f| &f.targets)
                .all(|t| t.legal_at_resolution_start.is_none()));
            game.resolve_top_of_stack();
            let space = game.suspended_decision_action_space().unwrap();
            game.publish_action_space(space);
            let obs = Observation::for_player(&game, controller);
            let frame = obs.continuation.iter().find(|f| f.kind == 5).unwrap();
            assert_eq!(
                frame
                    .targets
                    .iter()
                    .map(|t| (t.clause, t.legal_at_resolution_start))
                    .collect::<Vec<_>>(),
                vec![(0, Some(legal)), (1, Some(true))]
            );
            let capture = game.capture_decision(controller, 0).unwrap();
            let other = game.continuation_frames(PlayerId(1 - controller.0));
            assert!(other.iter().all(|f| f.kind <= 3));
            let native = encode(&obs, &ObservationEncoderConfig::default())
                .unwrap()
                .relations;
            let legacy = encode(
                &obs,
                &ObservationEncoderConfig {
                    policy_input_version: 2,
                    ..Default::default()
                },
            )
            .unwrap()
            .relations;
            for (row, old) in native
                .as_chunks::<33>()
                .0
                .iter()
                .zip(legacy.as_chunks::<32>().0)
            {
                assert_eq!(&row[..32], old);
            }
            let target_rows = native
                .as_chunks::<33>()
                .0
                .iter()
                .filter(|r| r[0] == 2.0 && r[3] == 5.0)
                .collect::<Vec<_>>();
            assert_eq!(target_rows.len(), 2);
            assert_eq!(target_rows[0][32], if legal { 1.0 } else { 2.0 });
            assert_eq!(target_rows[1][32], 1.0);
            evidence
                .push(serde_json::json!({"legal": legal, "capture": capture, "relations": native}));
            observations.push(obs.to_json());
            encodings.push(legacy);
        }
        assert_eq!(observations[0], observations[1]);
        assert_eq!(encodings[0], encodings[1]);
    }
    if let Ok(path) = std::env::var("RESOLUTION_ELIGIBILITY_EVIDENCE") {
        std::fs::write(path, serde_json::to_string_pretty(&evidence).unwrap()).unwrap();
    }
}
