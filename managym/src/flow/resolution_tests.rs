//! Bounded CR 608.2b witnesses. Synthetic programs isolate resolution semantics;
//! they do not assert these combinations are reachable in an admitted deck.

use crate::state::ability::LibraryEnd;
use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

use crate::{
    agent::action::Action,
    decision::Command,
    flow::{event::GameEvent, game::Game, turn::StepKind},
    state::{
        ability::{
            DelayedEvent, DelayedExpiry, DelayedFrequency, Effect, EffectValue, ObjectSelector,
            TargetRequirement, TargetSpec,
        },
        card::{ActivatedAbilityDefinition, CardType, Keyword, Keywords},
        effect::{EffectDuration, Modification},
        game_object::{ObjectId, ObjectKind, PermanentId, PlayerId, Target},
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
    game.keep_opening_hands();
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
        let object_id = game
            .state
            .latest_objects()
            .find(|(_, object)| object.owner == controller && object.name == "Lightning Bolt")
            .unwrap()
            .0;
        // Lightning Bolt is only a vessel: the synthetic clauses carry their
        // own requirements, so its printed targeting must not apply.
        game.state[object_id].spell_effects = effects;
        game.state[object_id].targeting = Vec::new();
        game.push_spell_to_stack(object_id, controller, targets, indices, Default::default());
        return;
    }
    let source = game
        .scenario_force_battlefield(controller, "Gray Ogre", true)
        .unwrap();
    let source_object = game.state.permanents[source].as_ref().unwrap().object;
    let source_ref = game.permanent_object(source).unwrap();
    let source_definition_id = game.state[source_object].definition_id;
    let id = game.state.id_gen.next_id();
    let object = match entry {
        Entry::Activated => {
            // Activated announcements currently expose one clause. Multi-clause
            // cases below use the spell/trigger paths that can announce them.
            assert_eq!(effects.len(), 1);
            game.state[source_object].activated_abilities = vec![ActivatedAbilityDefinition {
                effects: vec![effects.remove(0)],
                ..Default::default()
            }];
            StackObject::ActivatedAbility(ActivatedAbilityOnStack {
                effects: game.state[source_object].activated_abilities[0]
                    .effects
                    .clone(),
                program: false,
                source_transforms: 0,
                inherited_from: None,
                id,
                controller,
                source_definition_id,
                source_object,
                ability_index: 0,
                targets,
                target_req_indices: Vec::new(),
                attached: None,
                x: 0,
                granted_by: None,
                paid: Default::default(),
            })
        }
        Entry::Triggered => StackObject::TriggeredAbility(TriggeredAbilityOnStack {
            triggering_damage: 0,
            program: false,
            remembered: Vec::new(),
            source_transforms: 0,
            id,
            controller,
            source_identity: crate::state::game_object::AbilitySource::new(
                source_object,
                Some(source_ref),
            ),
            source_lki: None,
            source_definition_id,
            ability_index: 0,
            targets,
            target_req_indices: indices,
            context: None,
            player: None,
            inline_effects: Some(effects),
            x: 0,
            spent: Default::default(),
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
    Effect::deal_damage(1, ObjectSelector::Target(TargetSpec::Creature))
}

fn hexproof_keywords() -> Keywords {
    Keywords::of(&[Keyword::Hexproof])
}

/// Grant (or remove the grant of) hexproof until end of turn.
fn hexproof(game: &mut Game, id: PermanentId, value: bool) {
    let grant = Modification::AddKeywords(hexproof_keywords());
    let affected = game.affected_permanent(id).unwrap();
    if value {
        game.apply_continuous(
            None,
            PlayerId(0),
            [affected],
            grant,
            EffectDuration::EndOfTurn,
        );
    } else {
        game.state
            .effects
            .retain(|effect| effect.affected != affected || effect.modification != grant);
    }
}

fn pause() -> Effect {
    Effect::Modal {
        modes: vec![vec![], vec![Effect::gain_life(2)]],
    }
}

/// The v16 bytes keep the two names an activated ability's source had as a
/// row and an exact reference, both read from its one stored id.
#[test]
fn activated_ability_records_both_v16_source_projections_from_one_object() {
    let (mut game, _, _) = fixture(PlayerId(0));
    stack(
        &mut game,
        Entry::Activated,
        PlayerId(0),
        vec![Effect::gain_life(2)],
        Vec::new(),
        Vec::new(),
    );
    let Some(StackObject::ActivatedAbility(ability)) = game.state.stack_objects.last() else {
        panic!("activated ability on the stack");
    };
    let source = ability.source_object();
    let value = serde_json::to_value(ability).unwrap();
    assert_eq!(value["source_card"], serde_json::json!(source.slot()));
    assert_eq!(
        value["source_ref"],
        serde_json::json!({"entity": source.slot(), "incarnation": source.generation()})
    );
    let text = serde_json::to_string(ability).unwrap();
    let position = |key: &str| text.find(key).unwrap();
    assert!(position("\"source_card_registry_key\"") < position("\"source_card\""));
    assert!(position("\"source_card\"") < position("\"source_ref\""));
    assert!(position("\"source_ref\"") < position("\"ability_index\""));
}

#[test]
fn resolution_keeps_eligibility_after_hexproof_and_across_choice_clones() {
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
                        Effect::until_end_of_turn(
                            ObjectSelector::Target(TargetSpec::Creature),
                            Modification::AddKeywords(hexproof_keywords()),
                        ),
                        pause(),
                        damage(),
                    ],
                )],
                vec![Target::Permanent(victim)],
                vec![0],
            );
            game.resolve_top_of_stack();
            assert!(game.characteristics(victim).keywords.has(Keyword::Hexproof));
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
            let root = game.clone();
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
            assert_eq!(
                root.capture_decision(controller, 0).unwrap().digest,
                captured.digest
            );
            assert_eq!(root.state.deterministic_hash(), before);
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
                            Effect::gain_life(7),
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
                        vec![Effect::put_counters(
                            ObjectSelector::Target(TargetSpec::Creature),
                            CounterKind::PlusOnePlusOne,
                            2,
                        )],
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
            assert_eq!(permanent.counter_count(CounterKind::PlusOnePlusOne), 2);
            assert_eq!(game.state[permanent.object].owner, owner);
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
                            count: EffectValue::Fixed(1),
                            what: ObjectSelector::Target(spec.clone()),
                        },
                        Effect::put_counters(
                            ObjectSelector::Target(spec),
                            CounterKind::PlusOnePlusOne,
                            2,
                        ),
                    ],
                )],
                vec![Target::Permanent(land)],
                vec![0],
            );
            game.resolve_top_of_stack();
            let characteristics = game.characteristics(land);
            assert!(characteristics.is_animated);
            assert!(characteristics.is_creature());
            assert_eq!(characteristics.pt(), (3, 3));
            assert!(characteristics.keywords.has(Keyword::Haste));
            let permanent = game.state.permanents[land].as_ref().unwrap();
            assert_eq!(permanent.counter_count(CounterKind::PlusOnePlusOne), 3);
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
                        effects: vec![damage(), Effect::gain_life(7)],
                    }
                } else {
                    Effect::gain_life(7)
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
            let object_id = game.state.permanents[victim].as_ref().unwrap().object;
            stack(
                &mut game,
                entry,
                controller,
                vec![clause(
                    TargetSpec::Creature,
                    vec![
                        Effect::return_to_hand(ObjectSelector::Target(TargetSpec::Creature)),
                        pause(),
                        damage(),
                    ],
                )],
                vec![Target::Permanent(victim)],
                vec![0],
            );
            game.resolve_top_of_stack();
            assert_eq!(
                game.state.zones.zone_of(game.state.now(object_id)),
                Some(ZoneType::Hand)
            );
            // The first instruction moved the original object. A synthetic
            // reentry during the choice must not let the suffix find the card.
            let object_id = game.move_object(game.state.now(object_id), ZoneType::Battlefield);
            let replacement = game.state.permanent_of(object_id).unwrap();
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
            let object_id = game
                .state
                .latest_objects()
                .find(|(_, c)| c.owner == controller && c.name == "Mountain")
                .unwrap()
                .0;
            let object_id = game.move_object(object_id, ZoneType::Graveyard);
            let original = game.state.now(object_id);
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
                        Effect::return_to_hand(ObjectSelector::Target(spec.clone())),
                        pause(),
                        Effect::return_to_hand(ObjectSelector::Target(spec)),
                    ],
                )],
                vec![Target::GraveyardCard(original)],
                vec![0],
            );
            game.resolve_top_of_stack();
            assert_eq!(
                game.state.zones.zone_of(game.state.now(object_id)),
                Some(ZoneType::Hand)
            );
            let object_id = game.move_object(game.state.now(object_id), ZoneType::Graveyard);
            assert_ne!(game.state.live(object_id), Some(original));
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
                game.state.zones.zone_of(game.state.now(object_id)),
                Some(ZoneType::Graveyard)
            );
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
                    Effect::gain_life(7),
                ],
                vec![Target::Permanent(attacker), Target::Permanent(victim)],
                vec![0, 1],
            );
            let spell_card = match game.state.stack_objects.last().unwrap() {
                StackObject::Spell(spell) => spell.object,
                _ => unreachable!(),
            };
            game.state[spell_card].targeting = vec![
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
        for legal in [true, false] {
            let (mut game, first, second) = fixture(controller);
            stack(
                &mut game,
                Entry::Spell,
                controller,
                vec![
                    clause(
                        TargetSpec::Creature,
                        vec![Effect::until_end_of_turn(
                            ObjectSelector::Target(TargetSpec::Creature),
                            Modification::AddKeywords(hexproof_keywords()),
                        )],
                    ),
                    clause(TargetSpec::Creature, vec![pause(), damage()]),
                ],
                vec![Target::Permanent(first), Target::Permanent(second)],
                vec![0, 1],
            );
            if !legal {
                // The first target gains the same hexproof record in response,
                // so the present state matches the legal case's after its
                // grant resolves.
                let Some(StackObject::Spell(spell)) = game.state.stack_objects.last() else {
                    unreachable!()
                };
                let bolt = spell.object;
                let affected = game.affected_permanent(first).unwrap();
                game.apply_continuous(
                    Some(bolt),
                    controller,
                    [affected],
                    Modification::AddKeywords(hexproof_keywords()),
                    EffectDuration::EndOfTurn,
                );
            }
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
            let target_rows = native
                .as_chunks::<{ crate::agent::continuation::RELATION_DIM }>()
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
        }
        assert_eq!(observations[0], observations[1]);
    }
    if let Ok(path) = std::env::var("RESOLUTION_ELIGIBILITY_EVIDENCE") {
        std::fs::write(path, serde_json::to_string_pretty(&evidence).unwrap()).unwrap();
    }
}

/// Effect::CounterSpell counters spells only: given an ability as its
/// object (a trigger context), it leaves the ability on the stack.
#[test]
fn counter_spell_cannot_affect_an_ability() {
    let (mut game, first, _) = fixture(PlayerId(0));
    stack(
        &mut game,
        Entry::Activated,
        PlayerId(1),
        vec![damage()],
        vec![Target::Permanent(first)],
        vec![0],
    );
    let ability = game.state.stack_objects[0].id();
    let source = game
        .scenario_force_battlefield(PlayerId(0), "Gray Ogre", true)
        .unwrap();
    let source_object = game.state.permanents[source].as_ref().unwrap().object;
    let id = game.state.id_gen.next_id();
    game.push_to_stack(StackObject::TriggeredAbility(TriggeredAbilityOnStack {
        triggering_damage: 0,
        program: false,
        remembered: Vec::new(),
        source_transforms: 0,
        id,
        controller: PlayerId(0),
        source_identity: crate::state::game_object::AbilitySource::new(
            source_object,
            game.permanent_object(source),
        ),
        source_lki: None,
        source_definition_id: game.state[source_object].definition_id,
        ability_index: 0,
        targets: Vec::new(),
        target_req_indices: Vec::new(),
        context: Some(Target::StackAbility(ability)),
        player: None,
        inline_effects: Some(vec![Effect::CounterSpell {
            what: ObjectSelector::Target(TargetSpec::Spell),
            exile: false,
        }]),
        x: 0,
        spent: Default::default(),
    }));
    game.resolve_top_of_stack();
    assert_eq!(game.state.stack_objects.len(), 1);
    assert_eq!(game.state.stack_objects[0].id(), ability);
}

/// Effect::CounterAbility removes the targeted ability from the stack and
/// announces GameEvent::AbilityCountered; it offers no spell as a target.
#[test]
fn counter_ability_counters_an_ability_and_cannot_target_a_spell() {
    let (mut game, first, _) = fixture(PlayerId(0));
    stack(
        &mut game,
        Entry::Activated,
        PlayerId(1),
        vec![damage()],
        vec![Target::Permanent(first)],
        vec![0],
    );
    let ability = game.state.stack_objects[0].id();
    let spec = TargetSpec::Ability;
    assert_eq!(
        game.legal_targets_for_spec(&spec, PlayerId(0)),
        vec![Target::StackAbility(ability)]
    );
    stack(
        &mut game,
        Entry::Spell,
        PlayerId(0),
        vec![Effect::CounterAbility {
            what: ObjectSelector::Target(spec.clone()),
        }],
        vec![Target::StackAbility(ability)],
        vec![0],
    );
    let Some(StackObject::Spell(counter)) = game.state.stack_objects.last() else {
        panic!("the counter is a spell");
    };
    let counter = counter.object;
    assert!(!game.target_is_legal(Target::StackSpell(counter), &spec, PlayerId(0)));
    assert!(!game
        .legal_targets_for_spec(&spec, PlayerId(0))
        .contains(&Target::StackSpell(counter)));
    game.drain_events();
    game.resolve_top_of_stack();
    assert!(game.state.stack_objects.is_empty());
    assert!(game.drain_events().iter().any(|event| matches!(
        event,
        crate::flow::event::GameEvent::AbilityCountered { ability: countered, controller: PlayerId(1), by: Some(by), .. }
            if *countered == ability && *by == counter
    )));
}

/// Effect::CounterSpell cannot target an ability: no ability is offered or
/// legal for its spell target.
#[test]
fn counter_spell_cannot_target_an_ability() {
    let (mut game, first, _) = fixture(PlayerId(0));
    stack(
        &mut game,
        Entry::Triggered,
        PlayerId(1),
        vec![damage()],
        vec![Target::Permanent(first)],
        vec![0],
    );
    let ability = game.state.stack_objects[0].id();
    assert!(game
        .legal_targets_for_spec(&TargetSpec::Spell, PlayerId(0))
        .is_empty());
    assert!(!game.target_is_legal(
        Target::StackAbility(ability),
        &TargetSpec::Spell,
        PlayerId(0)
    ));
}

/// The selector is evaluated at creation; expiry and a response-time blink
/// cannot make its effects follow the physical card into another incarnation.
#[test]
fn delayed_capture_survives_placement_and_choice_forks_without_following_reentry() {
    for controller in [PlayerId(0), PlayerId(1)] {
        for leave_before_firing in [false, true] {
            let (mut game, subject, other) = fixture(controller);
            let subject_object = game.state.permanents[subject].as_ref().unwrap().object;
            let captured = game.permanent_object(subject).unwrap();
            stack(
                &mut game,
                Entry::Triggered,
                controller,
                vec![Effect::CreateDelayedTrigger {
                    event: DelayedEvent::EndStep { your_turn: false },
                    frequency: DelayedFrequency::Once,
                    expiry: DelayedExpiry::Never,
                    remember: Some(ObjectSelector::Target(TargetSpec::Creature)),
                    effects: vec![
                        pause(),
                        Effect::Tap {
                            what: ObjectSelector::Remembered,
                        },
                    ],
                }],
                vec![Target::Permanent(subject)],
                vec![0],
            );
            game.resolve_top_of_stack();
            assert_eq!(
                game.state.delayed_triggers[0].remembered[0].object,
                captured
            );
            let source = game.state.delayed_triggers[0].source_object();
            let mut different = game.clone();
            different.state.delayed_triggers[0].remembered[0].object =
                game.permanent_object(other).unwrap();
            assert_ne!(
                game.state.deterministic_hash(),
                different.state.deterministic_hash()
            );
            // Source departure does not discard the ability or its controller.
            game.move_object(source, ZoneType::Graveyard);
            if leave_before_firing {
                let subject_object = game.move_object(subject_object, ZoneType::Exile);
                game.move_object(subject_object, ZoneType::Battlefield);
            }
            game.emit(GameEvent::StepStarted {
                step: StepKind::End,
            });
            game.process_game_events();
            assert!(game.state.delayed_triggers.is_empty(), "one shot");
            assert_eq!(
                game.state.pending_triggers[0].remembered[0].object,
                captured
            );
            assert!(game.flush_triggers().is_none());
            let frame = game
                .continuation_frames(controller)
                .into_iter()
                .find(|f| f.kind == 3)
                .unwrap();
            assert_eq!(frame.remembered.len(), 1);
            assert!(frame.context.is_none(), "step events supply no object");
            game.resolve_top_of_stack();
            assert_eq!(
                game.state
                    .suspended_decision
                    .as_ref()
                    .unwrap()
                    .frame
                    .remembered[0]
                    .object,
                captured
            );
            if !leave_before_firing {
                let subject_object = game.move_object(subject_object, ZoneType::Exile);
                game.move_object(subject_object, ZoneType::Battlefield);
            }
            let new_subject = game
                .state
                .permanent_of(game.state.now(subject_object))
                .unwrap();
            assert_ne!(game.permanent_object(new_subject).unwrap(), captured);
            let space = game.suspended_decision_action_space().unwrap();
            assert_eq!(space.player, Some(controller));
            let choice = space
                .actions
                .iter()
                .position(|a| matches!(a, Action::ChooseMode { mode: 0, .. }))
                .unwrap();
            game.publish_action_space(space);
            let before = game.capture_decision(controller, 0).unwrap();
            let mut replay = game.clone();
            assert_eq!(
                replay.capture_decision(controller, 0).unwrap().digest,
                before.digest
            );
            game.step(choice).unwrap();
            replay.step(choice).unwrap();
            assert_eq!(
                game.state.deterministic_hash(),
                replay.state.deterministic_hash()
            );
            assert!(!game.state.permanents[new_subject].as_ref().unwrap().tapped);
        }
    }
}

#[test]
fn delayed_capture_keeps_event_context_and_original_controller_after_source_changes() {
    for controller in [PlayerId(0), PlayerId(1)] {
        let (mut game, subject, _) = fixture(controller);
        let remembered = game.permanent_object(subject).unwrap();
        stack(
            &mut game,
            Entry::Triggered,
            controller,
            vec![Effect::CreateDelayedTrigger {
                event: DelayedEvent::YouCastSpell(Box::default()),
                frequency: DelayedFrequency::Once,
                expiry: DelayedExpiry::ThisTurn,
                remember: Some(ObjectSelector::Target(TargetSpec::Creature)),
                effects: vec![
                    pause(),
                    Effect::Tap {
                        what: ObjectSelector::Remembered,
                    },
                    Effect::gain_life(3),
                ],
            }],
            vec![Target::Permanent(subject)],
            vec![0],
        );
        game.resolve_top_of_stack();
        let source = game.state.delayed_triggers[0].source_object();
        let source_permanent = game.state.permanent_of(game.state.now(source)).unwrap();
        game.change_control(source_permanent, PlayerId(1 - controller.0));
        game.move_object(source, ZoneType::Graveyard);
        stack(&mut game, Entry::Spell, controller, vec![], vec![], vec![]);
        let StackObject::Spell(spell) = game.state.stack_objects.last().unwrap() else {
            panic!("spell")
        };
        let cast = spell.object;
        game.emit(GameEvent::SpellCast {
            object_id: cast,
            targets: vec![],
            caster: controller,
            spend_before: 0,
            spend_after: 0,
        });
        game.process_game_events();
        assert!(game.state.delayed_triggers.is_empty());
        let pending = &game.state.pending_triggers[0];
        assert_eq!(pending.remembered[0].object, remembered);
        assert_eq!(pending.context, Some(Target::StackSpell(cast)));
        assert!(game.flush_triggers().is_none());
        game.resolve_top_of_stack();
        let suspended = game.state.suspended_decision.as_ref().unwrap();
        assert_eq!(suspended.frame.remembered[0].object, remembered);
        assert_eq!(
            suspended.frame.context_target,
            Some(Target::StackSpell(cast))
        );
        let space = game.suspended_decision_action_space().unwrap();
        assert_eq!(
            space.player,
            Some(controller),
            "source's later controller does not own this ability"
        );
        let choice = space
            .actions
            .iter()
            .position(|a| matches!(a, Action::ChooseMode { mode: 0, .. }))
            .unwrap();
        game.publish_action_space(space);
        let life = game.state.players[controller.0].life;
        game.step(choice).unwrap();
        assert!(game.state.permanents[subject].as_ref().unwrap().tapped);
        assert_eq!(game.state.players[controller.0].life, life + 3);
    }
}

#[test]
fn delayed_capture_projects_only_viewer_visible_incarnations() {
    use crate::agent::continuation::VisibleReference;

    let (mut game, subject, _) = fixture(PlayerId(0));
    let object_id = game.state.permanents[subject].as_ref().unwrap().object;
    stack(
        &mut game,
        Entry::Triggered,
        PlayerId(0),
        vec![
            Effect::return_to_hand(ObjectSelector::Target(TargetSpec::Creature)),
            Effect::CreateDelayedTrigger {
                event: DelayedEvent::EndStep { your_turn: false },
                frequency: DelayedFrequency::Once,
                expiry: DelayedExpiry::Never,
                remember: Some(ObjectSelector::Moved),
                effects: vec![Effect::exile(ObjectSelector::Remembered)],
            },
        ],
        vec![Target::Permanent(subject)],
        vec![0],
    );
    game.resolve_top_of_stack();
    let captured = game.state.now(object_id);
    assert_eq!(
        game.state.delayed_triggers[0].remembered[0].object,
        captured
    );
    for viewer in [PlayerId(0), PlayerId(1)] {
        let frame = game
            .continuation_frames(viewer)
            .into_iter()
            .find(|frame| frame.kind == 10)
            .unwrap();
        assert_eq!(frame.remembered.len(), 1);
        if viewer == PlayerId(0) {
            assert_eq!(frame.remembered[0], VisibleReference::default());
        } else {
            assert!(frame.remembered[0].object.is_some());
            assert_eq!(
                frame.remembered[0].definition,
                Some(game.state[game.state.now(object_id)].definition_id.0)
            );
        }
    }
}

fn completed_private_capture(player: PlayerId) -> Game {
    let (mut root, _, _) = fixture(player);
    let capture = Effect::CreateDelayedTrigger {
        event: DelayedEvent::EndStep { your_turn: false },
        frequency: DelayedFrequency::Once,
        expiry: DelayedExpiry::Never,
        remember: Some(ObjectSelector::Selected),
        effects: vec![pause(), Effect::exile(ObjectSelector::Remembered)],
    };
    stack(
        &mut root,
        Entry::Triggered,
        player,
        vec![Effect::SelectCards {
            from: ObjectSelector::Zone(ZoneType::Hand),
            predicate: CardPredicate::default(),
            min: 1,
            max: 1.into(),
            // Two occurrences alias one selected subject; they must receive one
            // joint assignment even when their execution lifetimes diverge.
            effects: vec![capture.clone(), capture],
            if_declined: Vec::new(),
        }],
        Vec::new(),
        Vec::new(),
    );
    root.resolve_top_of_stack();
    root.publish_action_space(root.suspended_decision_action_space().unwrap());
    let object = root.state.zones.zone_objects(ZoneType::Hand, player)[0];
    checked_command(&mut root, Action::SelectCard { player, object });
    assert!(root.state.suspended_decision.is_none());
    assert_eq!(root.state.delayed_triggers.len(), 2);
    root
}

#[test]
fn hidden_selection_completed_capture_stays_in_sampled_hand() {
    use crate::possible_worlds::PossibleWorldSpace;
    for player in [PlayerId(0), PlayerId(1)] {
        let mut root = completed_private_capture(player);
        let original = root.state.delayed_triggers[0].remembered[0].object;
        let mut stages = vec![root.clone()];
        root.emit(GameEvent::StepStarted {
            step: StepKind::End,
        });
        root.process_game_events();
        assert_eq!(root.state.pending_triggers.len(), 2);
        stages.push(root.clone());
        let order = root.flush_triggers().unwrap();
        root.publish_action_space(order);
        stages.push(root.clone());
        let trigger = root.state.trigger_placement.as_ref().unwrap().remaining[0].id;
        root.choose_trigger(player, trigger).unwrap();
        assert!(root
            .state
            .trigger_placement
            .as_ref()
            .unwrap()
            .placing
            .is_some());
        stages.push(root.clone());
        assert!(root.flush_triggers().is_none());
        stages.push(root.clone());
        root.resolve_top_of_stack();
        root.publish_action_space(root.suspended_decision_action_space().unwrap());
        stages.push(root);
        for (stage, root) in stages.iter().enumerate() {
            for viewer in [player, PlayerId(1 - player.0)] {
                let space = PossibleWorldSpace::for_viewer(root, viewer);
                let mut sampled = root.clone();
                sampled.determinize(viewer, 0).unwrap();
                let materialized = space.materialize(root, &space.worlds()[0], 0).unwrap();
                let before = root.continuation_frames(viewer);
                for branch in [sampled, materialized] {
                    let objects: Vec<_> = branch
                        .captured_bindings()
                        .map(|binding| binding.object)
                        .collect();
                    assert_eq!(objects.len(), 2, "stage {stage}");
                    assert_eq!(objects[0], objects[1], "aliases, stage {stage}");
                    let object = objects[0];
                    assert_eq!(branch.state.live(branch.state.now(object)), Some(object));
                    assert_eq!(
                        branch.state.zones.zone_of(branch.state.now(object)),
                        Some(ZoneType::Hand)
                    );
                    if viewer == player {
                        assert_eq!(object, original);
                    }
                    assert_eq!(
                        branch.continuation_frames(viewer),
                        before,
                        "private identity, stage {stage}"
                    );
                }
            }
        }
    }
}

#[test]
fn hidden_selection_capture_does_not_follow_a_later_incarnation() {
    use crate::possible_worlds::PossibleWorldSpace;
    let player = PlayerId(1);
    let viewer = PlayerId(0);
    let mut root = completed_private_capture(player);
    let original = root.state.delayed_triggers[0].remembered[0].object;
    let object_id = original;
    let object_id = root.move_object(object_id, ZoneType::Graveyard);
    let object_id = root.move_object(object_id, ZoneType::Hand);
    assert_ne!(root.state.live(root.state.now(object_id)), Some(original));
    assert!(!root.hidden_commitments(viewer).constrains());
    let space = PossibleWorldSpace::for_viewer(&root, viewer);
    let mut sampled = root.clone();
    sampled.determinize(viewer, 0).unwrap();
    let materialized = space.materialize(&root, &space.worlds()[0], 0).unwrap();
    for mut branch in [sampled, materialized] {
        assert_eq!(
            branch.state.delayed_triggers[0].remembered[0].object,
            original
        );
        branch.emit(GameEvent::StepStarted {
            step: StepKind::End,
        });
        branch.process_game_events();
        let order = branch.flush_triggers().unwrap();
        branch.publish_action_space(order);
        let trigger = branch.state.trigger_placement.as_ref().unwrap().remaining[0].id;
        branch.choose_trigger(player, trigger).unwrap();
        assert!(branch.flush_triggers().is_none());
        branch.resolve_top_of_stack();
        branch.publish_action_space(branch.suspended_decision_action_space().unwrap());
        let exile_count = branch.state.zones.size(ZoneType::Exile, player);
        checked_command(&mut branch, Action::ChooseMode { player, mode: 0 });
        assert_eq!(
            branch.state.zones.size(ZoneType::Exile, player),
            exile_count
        );
    }
}

#[test]
fn hidden_selection_moved_capture_uses_its_new_zone() {
    let player = PlayerId(1);
    let viewer = PlayerId(0);
    let (mut root, _, _) = fixture(player);
    let object_id = root.state.zones.zone_objects(ZoneType::Hand, player)[0];
    stack(
        &mut root,
        Entry::Triggered,
        player,
        vec![Effect::SelectCards {
            from: ObjectSelector::Zone(ZoneType::Hand),
            predicate: CardPredicate::default(),
            min: 1,
            max: 1.into(),
            if_declined: Vec::new(),
            effects: vec![
                Effect::put_in_library(ObjectSelector::Selected, LibraryEnd::Bottom),
                Effect::CreateDelayedTrigger {
                    event: DelayedEvent::EndStep { your_turn: false },
                    frequency: DelayedFrequency::Once,
                    expiry: DelayedExpiry::Never,
                    remember: Some(ObjectSelector::Moved),
                    effects: vec![Effect::exile(ObjectSelector::Remembered)],
                },
            ],
        }],
        Vec::new(),
        Vec::new(),
    );
    root.resolve_top_of_stack();
    root.publish_action_space(root.suspended_decision_action_space().unwrap());
    checked_command(
        &mut root,
        Action::SelectCard {
            player,
            object: object_id,
        },
    );
    assert_eq!(
        root.state.delayed_triggers[0].remembered[0]
            .selection
            .as_ref()
            .unwrap()
            .zone,
        ZoneType::Library
    );
    let space = crate::possible_worlds::PossibleWorldSpace::for_viewer(&root, viewer);
    let mut sampled = root.clone();
    sampled.determinize(viewer, 0).unwrap();
    for branch in [
        sampled,
        space.materialize(&root, &space.worlds()[0], 0).unwrap(),
    ] {
        let object = branch.state.delayed_triggers[0].remembered[0].object;
        assert_eq!(
            branch.state.zones.zone_of(branch.state.now(object)),
            Some(ZoneType::Library)
        );
    }
}

#[test]
fn hidden_selection_known_capture_pins_the_exact_copy_after_control_expires() {
    use crate::flow::turn::PlayerControl;
    use crate::state::ability::PlayerControlDuration;
    let player = PlayerId(1);
    let viewer = PlayerId(0);
    let mut root = completed_private_capture(player);
    // Re-capture through the rules instruction while the viewer controls this
    // player, then expire control. The known exact subject survives that expiry.
    root.state.turn.player_controls.push(PlayerControl {
        player,
        controller: viewer,
        duration: PlayerControlDuration::NextTurn,
        active: true,
    });
    let original = root.state.delayed_triggers[0].remembered[0].object;
    let mut frame = super::EffectFrame::new(player, Vec::new());
    frame.remembered = root.state.delayed_triggers[0].remembered.clone();
    let known = root.capture_bindings(&ObjectSelector::Remembered, &frame);
    for trigger in &mut root.state.delayed_triggers {
        trigger.remembered = known.clone();
    }
    root.state.turn.player_controls.clear();
    let before = root.state.deterministic_hash();
    let mut different = root.clone();
    different.state.delayed_triggers[0].remembered[0].known_to[viewer.0] = false;
    assert_ne!(
        before,
        different.state.deterministic_hash(),
        "knowledge is canonical state"
    );
    let space = crate::possible_worlds::PossibleWorldSpace::for_viewer(&root, viewer);
    let mut sampled = root.clone();
    sampled.determinize(viewer, 0).unwrap();
    for branch in [
        sampled,
        space.materialize(&root, &space.worlds()[0], 0).unwrap(),
    ] {
        assert_eq!(
            branch.state.delayed_triggers[0].remembered[0].object,
            original
        );
        assert_eq!(
            branch.state.zones.zone_of(branch.state.now(original)),
            Some(ZoneType::Hand)
        );
    }
}

#[test]
fn hidden_selection_from_remembered_keeps_decision_and_child_aliases() {
    let player = PlayerId(1);
    let viewer = PlayerId(0);
    let mut root = completed_private_capture(player);
    for trigger in &mut root.state.delayed_triggers {
        trigger.effects = vec![Effect::SelectCards {
            from: ObjectSelector::Remembered,
            predicate: CardPredicate::default(),
            min: 1,
            max: 1.into(),
            if_declined: Vec::new(),
            effects: vec![
                Effect::ForPlayer {
                    player: crate::state::ability::PlayerSelector::EachOpponent,
                    effects: vec![pause()],
                },
                Effect::exile(ObjectSelector::Selected),
            ],
        }];
    }
    root.emit(GameEvent::StepStarted {
        step: StepKind::End,
    });
    root.process_game_events();
    let order = root.flush_triggers().unwrap();
    root.publish_action_space(order);
    let trigger = root.state.trigger_placement.as_ref().unwrap().remaining[0].id;
    root.choose_trigger(player, trigger).unwrap();
    assert!(root.flush_triggers().is_none());
    root.resolve_top_of_stack();
    root.publish_action_space(root.suspended_decision_action_space().unwrap());
    let choosing = root.clone();
    let object_id = root
        .state
        .suspended_decision
        .as_ref()
        .unwrap()
        .frame
        .remembered[0]
        .object;
    checked_command(
        &mut root,
        Action::SelectCard {
            player,
            object: object_id,
        },
    );
    for (selected, source) in [(false, choosing), (true, root)] {
        let space = crate::possible_worlds::PossibleWorldSpace::for_viewer(&source, viewer);
        let mut sampled = source.clone();
        sampled.determinize(viewer, 0).unwrap();
        for mut branch in [
            sampled,
            space
                .materialize_index(
                    &source,
                    0,
                    0,
                    if selected {
                        crate::possible_worlds::MaterializeMode::PreserveViewerRoot
                    } else {
                        crate::possible_worlds::MaterializeMode::RefreshOpponentCommitment
                    },
                )
                .unwrap(),
        ] {
            let frame = &branch.state.suspended_decision.as_ref().unwrap().frame;
            let captured = frame.remembered[0].object;
            if selected {
                assert_eq!(frame.selected, vec![captured]);
            } else {
                let super::Decision::SelectCards { candidates, .. } =
                    &branch.state.suspended_decision.as_ref().unwrap().decision
                else {
                    panic!("selection")
                };
                assert_eq!(candidates, &vec![captured]);
                checked_command(
                    &mut branch,
                    Action::SelectCard {
                        player,
                        object: captured,
                    },
                );
            }
            checked_command(
                &mut branch,
                Action::ChooseMode {
                    player: viewer,
                    mode: 0,
                },
            );
            assert_eq!(
                branch.state.zones.zone_of(branch.state.now(captured)),
                Some(ZoneType::Exile)
            );
        }
    }
}

fn delay_life_gain(
    game: &mut Game,
    event: DelayedEvent,
    frequency: DelayedFrequency,
    expiry: DelayedExpiry,
) {
    stack(
        game,
        Entry::Triggered,
        PlayerId(0),
        vec![Effect::CreateDelayedTrigger {
            event,
            frequency,
            expiry,
            remember: None,
            effects: vec![Effect::gain_life(3)],
        }],
        vec![],
        vec![],
    );
    game.resolve_top_of_stack();
}

#[test]
fn copied_island_death_uses_departure_type_for_morbid_and_delayed_watchers() {
    let deck = BTreeMap::from([("Island".into(), 20), ("Gray Ogre".into(), 20)]);
    let mut game = Game::new(
        vec![
            PlayerConfig::new("p0", deck.clone()),
            PlayerConfig::new("p1", deck),
        ],
        418,
        false,
    );
    let ogre = game
        .scenario_force_battlefield(PlayerId(1), "Gray Ogre", true)
        .unwrap();
    let island = game
        .scenario_force_battlefield(PlayerId(1), "Island", true)
        .unwrap();
    let object = game.state.permanents[island].as_ref().unwrap().object;
    let original = game.state[game.state.permanents[ogre].as_ref().unwrap().object].clone();
    game.state[object].become_copy_of(&original, &[]);
    delay_life_gain(
        &mut game,
        DelayedEvent::CreatureDies,
        DelayedFrequency::Once,
        DelayedExpiry::ThisTurn,
    );
    for zone in [ZoneType::Exile, ZoneType::Graveyard] {
        let mut branch = game.clone();
        branch.move_object(object, zone);
        branch.process_game_events();
        assert!(branch.flush_triggers().is_none());
        let died = zone == ZoneType::Graveyard;
        assert_eq!(branch.state.turn.creature_died_this_turn, died);
        assert_eq!(branch.state.stack_objects.len(), usize::from(died));
        let hash = branch.state.deterministic_hash();
        branch.state.turn.creature_died_this_turn = !died;
        assert_ne!(branch.state.deterministic_hash(), hash);
    }
}

/// The same matcher composes with either firing policy and either expiry.
/// These bounded synthetic effects exercise the real resolver and occurrence queue.
#[test]
fn delayed_lifetime_composes_frequency_and_expiry_independently() {
    for event in [DelayedEvent::Upkeep, DelayedEvent::CreatureDies] {
        for frequency in [DelayedFrequency::Once, DelayedFrequency::EachMatch] {
            for expiry in [DelayedExpiry::ThisTurn, DelayedExpiry::Never] {
                let (mut game, first, second) = fixture(PlayerId(0));
                delay_life_gain(&mut game, event.clone(), frequency, expiry);
                let root = game.state.deterministic_hash();
                let mut changed = game.clone();
                changed.state.delayed_triggers[0].frequency = if frequency == DelayedFrequency::Once
                {
                    DelayedFrequency::EachMatch
                } else {
                    DelayedFrequency::Once
                };
                assert_ne!(root, changed.state.deterministic_hash());
                let mut changed = game.clone();
                changed.state.delayed_triggers[0].expiry = if expiry == DelayedExpiry::ThisTurn {
                    DelayedExpiry::Never
                } else {
                    DelayedExpiry::ThisTurn
                };
                assert_ne!(root, changed.state.deterministic_hash());
                let mut fork = game.clone();
                for current in [&mut game, &mut fork] {
                    let life = current.state.players[0].life;
                    for subject in [first, second] {
                        if event == DelayedEvent::CreatureDies {
                            let object = current.state.permanents[subject].as_ref().unwrap().object;
                            current.move_object(object, ZoneType::Graveyard);
                        } else {
                            current.emit(GameEvent::StepStarted {
                                step: StepKind::Upkeep,
                            });
                        }
                        current.process_game_events();
                        assert!(current.flush_triggers().is_none());
                        if !current.state.stack_objects.is_empty() {
                            current.resolve_top_of_stack();
                        }
                    }
                    let repeats = frequency == DelayedFrequency::EachMatch;
                    assert_eq!(
                        current.state.players[0].life,
                        life + if repeats { 6 } else { 3 }
                    );
                    assert_eq!(current.state.delayed_triggers.len(), usize::from(repeats));
                    current.end_turn_modifiers();
                    assert_eq!(
                        current.state.delayed_triggers.len(),
                        usize::from(repeats && expiry == DelayedExpiry::Never)
                    );
                }
                assert_eq!(
                    game.state.deterministic_hash(),
                    fork.state.deterministic_hash()
                );
                assert_ne!(root, game.state.deterministic_hash());
            }
        }
    }
}

#[test]
fn delayed_lifetime_no_match_expires_without_firing_next_turn() {
    for frequency in [DelayedFrequency::Once, DelayedFrequency::EachMatch] {
        for expiry in [DelayedExpiry::ThisTurn, DelayedExpiry::Never] {
            let (mut game, _, _) = fixture(PlayerId(0));
            delay_life_gain(
                &mut game,
                DelayedEvent::EndStep { your_turn: false },
                frequency,
                expiry,
            );
            game.end_turn_modifiers();
            game.emit(GameEvent::StepStarted {
                step: StepKind::End,
            });
            game.process_game_events();
            assert_eq!(
                game.state.pending_triggers.len(),
                usize::from(expiry == DelayedExpiry::Never)
            );
        }
    }
}

#[test]
fn delayed_lifetime_countered_first_occurrence_does_not_rearm() {
    let (mut game, _, _) = fixture(PlayerId(0));
    delay_life_gain(
        &mut game,
        DelayedEvent::Upkeep,
        DelayedFrequency::Once,
        DelayedExpiry::Never,
    );
    let life = game.state.players[0].life;
    game.emit(GameEvent::StepStarted {
        step: StepKind::Upkeep,
    });
    game.process_game_events();
    assert!(game.flush_triggers().is_none());
    let ability = game.state.stack_objects.last().unwrap().id();
    stack(
        &mut game,
        Entry::Spell,
        PlayerId(1),
        vec![Effect::CounterAbility {
            what: ObjectSelector::Target(TargetSpec::Ability),
        }],
        vec![Target::StackAbility(ability)],
        vec![0],
    );
    game.resolve_top_of_stack();
    assert!(game.state.stack_objects.is_empty());
    game.emit(GameEvent::StepStarted {
        step: StepKind::Upkeep,
    });
    game.process_game_events();
    assert!(game.state.pending_triggers.is_empty());
    assert!(game.state.delayed_triggers.is_empty());
    assert_eq!(game.state.players[0].life, life);
}

#[test]
fn delayed_lifetime_expiry_preserves_queued_and_stacked_occurrences() {
    for place_before_cleanup in [false, true] {
        let (mut game, _, _) = fixture(PlayerId(0));
        delay_life_gain(
            &mut game,
            DelayedEvent::EndStep { your_turn: false },
            DelayedFrequency::EachMatch,
            DelayedExpiry::ThisTurn,
        );
        let life = game.state.players[0].life;
        game.emit(GameEvent::StepStarted {
            step: StepKind::End,
        });
        game.process_game_events();
        assert_eq!(game.state.pending_triggers.len(), 1);
        if place_before_cleanup {
            assert!(game.flush_triggers().is_none());
        }
        game.end_turn_modifiers();
        assert!(game.state.delayed_triggers.is_empty());
        if !place_before_cleanup {
            assert!(game.flush_triggers().is_none());
        }
        game.resolve_top_of_stack();
        assert_eq!(game.state.players[0].life, life + 3);
    }
}

#[test]
fn delayed_lifetime_death_watchers_retire_on_exact_object_departure() {
    for frequency in [DelayedFrequency::Once, DelayedFrequency::EachMatch] {
        for destination in [ZoneType::Hand, ZoneType::Graveyard] {
            let (mut game, subject, _) = fixture(PlayerId(0));
            let object = game.state.permanents[subject].as_ref().unwrap().object;
            stack(
                &mut game,
                Entry::Triggered,
                PlayerId(0),
                vec![Effect::CreateDelayedTrigger {
                    event: DelayedEvent::Dies,
                    frequency,
                    expiry: DelayedExpiry::Never,
                    remember: Some(ObjectSelector::Target(TargetSpec::Creature)),
                    effects: vec![Effect::gain_life(3)],
                }],
                vec![Target::Permanent(subject)],
                vec![0],
            );
            game.resolve_top_of_stack();
            game.move_object(object, destination);
            game.process_game_events();
            assert!(
                game.state.delayed_triggers.is_empty(),
                "retired even when repeating"
            );
            let expected = usize::from(destination == ZoneType::Graveyard);
            assert_eq!(game.state.pending_triggers.len(), expected);
            let object = game.move_object(object, ZoneType::Battlefield);
            game.move_object(object, ZoneType::Graveyard);
            game.process_game_events();
            assert_eq!(
                game.state.pending_triggers.len(),
                expected,
                "new incarnation is not watched"
            );
        }
    }
}

#[test]
fn nested_iteration_retains_choices_scopes_and_finalizes_once() {
    for controller in [PlayerId(0), PlayerId(1)] {
        let (mut game, first, second) = fixture(controller);
        let choose_damage = Effect::Modal {
            modes: vec![
                vec![damage()],
                vec![Effect::put_counters(
                    ObjectSelector::Target(TargetSpec::Creature),
                    CounterKind::PlusOnePlusOne,
                    1,
                )],
            ],
        };
        stack(
            &mut game,
            Entry::Spell,
            controller,
            vec![Effect::TargetClause {
                requirement: TargetRequirement::up_to(2, TargetSpec::Creature),
                effects: vec![
                    Effect::ForEachTarget {
                        effects: vec![Effect::ForEachTarget {
                            effects: vec![choose_damage],
                        }],
                    },
                    Effect::gain_life(3),
                ],
            }],
            vec![Target::Permanent(first), Target::Permanent(second)],
            vec![0, 0],
        );
        game.resolve_top_of_stack();
        game.publish_action_space(game.suspended_decision_action_space().unwrap());
        let root = game.state.deterministic_hash();
        let frames = game.capture_decision(controller, 0).unwrap().frames;
        let scopes: Vec<_> = frames.iter().filter(|frame| frame.kind == 5).collect();
        assert_eq!(scopes.len(), 3, "root and both interrupted iterations");
        assert_eq!(scopes[0].targets.len(), 2);
        assert_eq!(scopes[1].targets.len(), 1);
        assert_eq!(game.state.stack_objects.len(), 1);
        for first_mode in [0, 1] {
            let mut branch = game.clone();
            for mode in [first_mode, 1 - first_mode] {
                checked_command(
                    &mut branch,
                    Action::ChooseMode {
                        player: controller,
                        mode,
                    },
                );
            }
            assert!(branch.state.suspended_decision.is_none());
            assert!(branch.state.stack_objects.is_empty());
            assert_eq!(branch.state.players[controller.0].life, 23);
            for (id, mode) in [(first, first_mode), (second, 1 - first_mode)] {
                let permanent = branch.state.permanents[id].as_ref().unwrap();
                assert_eq!(permanent.damage, i32::from(mode == 0));
                assert_eq!(
                    permanent.counter_count(CounterKind::PlusOnePlusOne),
                    i32::from(mode == 1)
                );
            }
            assert_eq!(
                branch
                    .state
                    .events
                    .iter()
                    .filter(|event| matches!(event, GameEvent::SpellResolved { .. }))
                    .count(),
                1
            );
            assert_eq!(game.state.deterministic_hash(), root, "retained fork root");
        }
    }
}

#[test]
fn nested_iteration_defers_state_based_actions_until_root_completion() {
    let controller = PlayerId(0);
    let (mut game, first, second) = fixture(controller);
    stack(
        &mut game,
        Entry::Spell,
        controller,
        vec![Effect::TargetClause {
            requirement: TargetRequirement::up_to(2, TargetSpec::Creature),
            effects: vec![Effect::ForEachTarget {
                effects: vec![
                    Effect::deal_damage(2, ObjectSelector::Target(TargetSpec::Creature)),
                    pause(),
                ],
            }],
        }],
        vec![Target::Permanent(first), Target::Permanent(second)],
        vec![0, 0],
    );
    game.resolve_top_of_stack();
    game.publish_action_space(game.suspended_decision_action_space().unwrap());
    assert_eq!(game.state.permanents[first].as_ref().unwrap().damage, 2);
    assert_eq!(game.state.permanents[second].as_ref().unwrap().damage, 0);
    game.step(0).unwrap();
    assert!(game.state.suspended_decision.is_some());
    assert_eq!(game.state.permanents[first].as_ref().unwrap().damage, 2);
    assert_eq!(game.state.permanents[second].as_ref().unwrap().damage, 2);
    assert!(!game
        .state
        .events
        .iter()
        .any(|event| matches!(event, GameEvent::SpellResolved { .. })));
    game.step(0).unwrap();
    assert!(game.state.suspended_decision.is_none());
    assert!(game.state.permanents[first].is_none());
    assert!(game.state.permanents[second].is_none());
    assert_eq!(
        game.state
            .events
            .iter()
            .filter(|event| matches!(event, GameEvent::SpellResolved { .. }))
            .count(),
        1
    );
}

/// Apply one ordinary bound Command, checking replay and stale rejection at
/// each choice boundary rather than calling a decision helper.
fn checked_command(game: &mut Game, action: Action) {
    let index = game
        .action_space()
        .unwrap()
        .actions
        .iter()
        .position(|candidate| *candidate == action)
        .expect("action must be a legal offer");
    let frame = game.semantic_decision_frame().unwrap();
    let command = Command {
        command_id: format!("choice-{}", frame.revision),
        expected_revision: frame.revision,
        offer_id: frame.offers[index].id.0,
        answers: vec![],
        object_preconditions: vec![],
    };
    let mut replay = game.clone();
    game.execute_semantic_command(&command).unwrap();
    replay.execute_semantic_command(&command).unwrap();
    assert_eq!(
        game.state.deterministic_hash(),
        replay.state.deterministic_hash()
    );
    let after = game.state.deterministic_hash();
    assert!(game.execute_semantic_command(&command).is_err());
    assert_eq!(game.state.deterministic_hash(), after);
}

#[test]
fn surveil_kept_order_retains_private_choices_forks_and_suffix() {
    use crate::agent::observation::Observation;

    for player in [PlayerId(0), PlayerId(1)] {
        let (mut root, _, _) = fixture(player);
        // More than two kept cards exposes an intermediate ordered prefix.
        let looked = root.library_top_objects(player, 4);
        let original_refs: Vec<_> = looked
            .iter()
            .map(|object| root.state.live(root.state.now(*object)))
            .collect();
        stack(
            &mut root,
            Entry::Spell,
            player,
            vec![
                Effect::Surveil { count: 4 },
                Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                },
            ],
            vec![],
            vec![],
        );
        root.resolve_top_of_stack();
        root.publish_action_space(root.suspended_decision_action_space().unwrap());
        checked_command(
            &mut root,
            Action::SelectCard {
                player,
                object: looked[0],
            },
        );
        assert_eq!(
            root.state.zones.zone_of(root.state.now(looked[0])),
            Some(ZoneType::Library),
            "selection is private until the partition commits"
        );
        checked_command(&mut root, Action::Decline { player });
        assert_eq!(
            root.state.zones.zone_of(root.state.now(looked[0])),
            Some(ZoneType::Graveyard)
        );
        let before = root.state.deterministic_hash();
        let hand = root.state.zones.size(ZoneType::Hand, player);
        let inspected = |game: &Game, viewer: PlayerId| {
            let observation = Observation::for_player(game, viewer);
            observation
                .agent_cards
                .into_iter()
                .chain(observation.opponent_cards)
                .filter(|object| object.zone == ZoneType::Library)
                .map(|object| object.id)
                .collect::<Vec<_>>()
        };
        assert_eq!(inspected(&root, player).len(), 3);
        assert!(inspected(&root, PlayerId(1 - player.0)).is_empty());
        assert_eq!(
            root.capture_decision(player, 0)
                .unwrap()
                .frames
                .last()
                .unwrap()
                .decision,
            27
        );
        assert_eq!(
            root.decision_context(player).unwrap().stage,
            "order_library_top"
        );
        for order in [
            [looked[3], looked[1], looked[2]],
            [looked[2], looked[3], looked[1]],
        ] {
            let mut branch = root.clone();
            for (index, object) in order.into_iter().enumerate() {
                assert!(!branch
                    .action_space()
                    .unwrap()
                    .actions
                    .iter()
                    .any(|action| matches!(action, Action::Decline { .. })));
                checked_command(&mut branch, Action::SelectCard { player, object });
                if index < 2 {
                    assert_eq!(
                        branch.state.zones.size(ZoneType::Hand, player),
                        hand,
                        "suffix waits for the complete order"
                    );
                    assert_eq!(branch.state.stack_objects.len(), 1);
                    assert_eq!(
                        inspected(&branch, player).len(),
                        3,
                        "chosen prefix stays visible"
                    );
                    assert!(inspected(&branch, PlayerId(1 - player.0)).is_empty());
                    let frames = branch.capture_decision(player, 0).unwrap().frames;
                    assert_eq!(frames.last().unwrap().mode, Some(index + 1));
                    let mut sampled = branch.clone();
                    sampled.determinize(player, 119).unwrap();
                    assert_eq!(
                        sampled.library_top_objects(player, 3),
                        branch.library_top_objects(player, 3)
                    );
                }
            }
            assert_eq!(
                branch.state.zones.zone_of(branch.state.now(order[0])),
                Some(ZoneType::Hand)
            );
            assert_eq!(branch.library_top_objects(player, 2), order[1..]);
            for object in &order[1..] {
                let index = looked
                    .iter()
                    .position(|candidate| candidate == object)
                    .unwrap();
                assert_eq!(
                    branch.state.live(branch.state.now(*object)),
                    original_refs[index],
                    "reorder is not a zone change"
                );
            }
            assert_eq!(branch.state.zones.size(ZoneType::Hand, player), hand + 1);
            assert!(branch.state.stack_objects.is_empty());
            assert!(branch.state.suspended_decision.is_none());
            assert_eq!(
                branch
                    .state
                    .events
                    .iter()
                    .filter(|event| matches!(event, GameEvent::SpellResolved { .. }))
                    .count(),
                1
            );
            assert_eq!(root.state.deterministic_hash(), before);
        }
    }
}

#[test]
fn cost_effect_completion_retains_output_and_parent_payment_across_choices() {
    use crate::flow::decision::{Decision, EffectFrame, ManaProduction, SuspendedResolution};
    use crate::flow::payment::{CostComponent, PaymentProgress};
    use crate::state::mana::{Mana, ManaCost};

    let player = PlayerId(0);
    let (mut game, _, _) = fixture(player);
    let source = game
        .scenario_force_battlefield(player, "Gray Ogre", true)
        .unwrap();
    let object = game.state.permanents[source].as_ref().unwrap().object;
    let source_ref = game.permanent_object(source).unwrap();
    let looked = game.library_top_objects(player, 2);
    let caller = SuspendedResolution {
        frame: EffectFrame::new(player, vec![Effect::gain_life(3)]),
        decision: Decision::ManaChoice {
            player,
            choices: vec![Mana::parse("U"), Mana::parse("B")],
            production: ManaProduction::Output {
                source: None,
                tapped: None,
            },
            resume: Some(Box::new(Decision::Payment(PaymentProgress::new(
                player,
                CostComponent::mana(ManaCost::parse("{1}")),
                vec![Effect::gain_life(2)],
                vec![],
            )))),
        },
    };
    let mut child = EffectFrame::new(
        player,
        vec![Effect::sacrifice_this(), Effect::Surveil { count: 2 }],
    );
    child.bind_source(Some(object), Some(source_ref));
    // A rules-level child choice isolates completion ownership. Entry
    // replacement suspension is a separate event-batch integration witness.
    game.run_before_decision(child, caller.clone());
    game.publish_action_space(game.suspended_decision_action_space().unwrap());
    assert_eq!(
        game.state.zones.zone_of(game.state.now(object)),
        Some(ZoneType::Graveyard)
    );
    let after_sacrifice = game.state.deterministic_hash();
    let frames = game.capture_decision(player, 0).unwrap().frames;
    let obligations: Vec<_> = frames
        .iter()
        .filter(|frame| frame.kind == 6)
        .map(|frame| frame.decision)
        .collect();
    assert_eq!(
        obligations,
        [1, 3, 22],
        "payment and output callers precede the active child"
    );
    assert_eq!(game.state.players[player.0].mana_pool.total(), 0);
    assert!(!game
        .action_space()
        .unwrap()
        .actions
        .iter()
        .any(|action| matches!(action, Action::ChooseMode { .. })));
    let mut fork = game.clone();
    checked_command(&mut fork, Action::Decline { player });
    for object in looked.iter().rev() {
        checked_command(
            &mut fork,
            Action::SelectCard {
                player,
                object: *object,
            },
        );
    }
    assert_eq!(fork.state.suspended_decision, Some(caller));
    assert_eq!(
        fork.state.players[player.0].life, 20,
        "parent suffix has not executed"
    );
    assert_eq!(
        fork.state.players[player.0].mana_pool.total(),
        0,
        "output remains an explicit choice"
    );
    checked_command(&mut fork, Action::ChooseMode { player, mode: 0 });
    assert_eq!(fork.state.players[player.0].mana_pool, Mana::parse("U"));
    checked_command(&mut fork, Action::PayCost { player });
    assert_eq!(fork.state.players[player.0].mana_pool.total(), 0);
    assert_eq!(
        fork.state.players[player.0].life, 25,
        "paid branch and parent suffix each ran once"
    );
    assert_eq!(fork.state.events.iter().filter(|event| matches!(event, GameEvent::PermanentSacrificed { object_id: sacrificed, .. } if *sacrificed == object)).count(), 1);
    assert_eq!(game.state.deterministic_hash(), after_sacrifice);
}

/// A player who ordered the top of their own library knows that order until
/// a draw, shuffle or move disturbs it: no sampled world may contradict it.
#[test]
fn own_ordered_library_top_survives_hidden_world_sampling() {
    for player in [PlayerId(0), PlayerId(1)] {
        let (mut game, _, _) = fixture(player);
        stack(
            &mut game,
            Entry::Spell,
            player,
            vec![Effect::Surveil { count: 4 }],
            vec![],
            vec![],
        );
        game.resolve_top_of_stack();
        game.publish_action_space(game.suspended_decision_action_space().unwrap());
        let observation = crate::agent::observation::Observation::for_player(&game, player);
        assert_eq!(
            observation
                .agent
                .unseen_counts
                .as_ref()
                .unwrap()
                .values()
                .sum::<u32>() as usize,
            game.state.zones.size(ZoneType::Library, player) - 4
        );
        checked_command(&mut game, Action::Decline { player });
        while game.state.suspended_decision.is_some() {
            let choice = game
                .action_space()
                .unwrap()
                .actions
                .iter()
                .rev()
                .find(|action| matches!(action, Action::SelectCard { .. }))
                .cloned()
                .expect("ordering offers a card");
            checked_command(&mut game, choice);
        }
        let definitions = |game: &Game| -> Vec<_> {
            game.library_top_objects(player, 4)
                .into_iter()
                .map(|object| game.state[object].definition_id)
                .collect()
        };
        let known = definitions(&game);
        game.refresh_priority_actions(player);
        let space = crate::possible_worlds::PossibleWorldSpace::for_viewer(&game, player);
        assert_eq!(
            crate::agent::observation::Observation::for_player(&game, player)
                .agent
                .known_library
                .len(),
            4
        );
        for seed in 0..64 {
            let mut sampled = game.clone();
            sampled.determinize(player, seed).unwrap();
            assert_eq!(
                definitions(&sampled),
                known,
                "seed {seed} sampled a library top its owner knows is impossible"
            );
            let materialized = space
                .materialize_index(
                    &game,
                    0,
                    seed,
                    crate::possible_worlds::MaterializeMode::PreserveViewerRoot,
                )
                .unwrap();
            assert_eq!(definitions(&materialized), known);
        }
        game.draw_cards(player, 1);
        for seed in 0..16 {
            let mut sampled = game.clone();
            sampled.determinize(player, seed).unwrap();
            assert_eq!(&definitions(&sampled)[..3], &known[1..]);
        }
        game.shuffle_library(player);
        assert!(game.state.players[player.0].known_library.is_empty());
    }
}

#[test]
fn retained_inspection_survives_nested_choices_and_hidden_world_sampling() {
    use crate::agent::observation::Observation;
    use crate::flow::decision::{Decision, EffectFrame, FrameCompletion, SuspendedResolution};

    for player in [PlayerId(0), PlayerId(1)] {
        let opponent = PlayerId(1 - player.0);
        let (mut game, _, _) = fixture(player);
        let looked = game.library_top_objects(player, 3);
        let caller = SuspendedResolution {
            frame: EffectFrame::new(player, vec![Effect::gain_life(2)]),
            decision: Decision::OrderLibrary {
                player,
                object_ids: looked.clone(),
                next: 0,
                end: LibraryEnd::Top,
            },
        };
        let mut parent = EffectFrame::new(opponent, vec![]);
        parent.completion = FrameCompletion::Decision(Box::new(caller.clone()));
        let mut child = EffectFrame::new(opponent, vec![pause()]);
        child.completion = FrameCompletion::Frame(Box::new(parent));
        game.run_frame(child);
        game.publish_action_space(game.suspended_decision_action_space().unwrap());
        let visible_library = |game: &Game, viewer| {
            let obs = Observation::for_player(game, viewer);
            obs.agent_cards
                .into_iter()
                .chain(obs.opponent_cards)
                .filter(|object| object.zone == ZoneType::Library)
                .map(|object| object.id)
                .collect::<Vec<_>>()
        };
        let expected = looked
            .iter()
            .map(|object| game.state[*object].id.0 as i32)
            .collect::<Vec<_>>();
        assert_eq!(visible_library(&game, player), expected);
        assert!(visible_library(&game, opponent).is_empty());
        let original = game.state.deterministic_hash();
        let mut sampled = game.clone();
        sampled.determinize(player, 241).unwrap();
        assert_eq!(sampled.library_top_objects(player, 3), looked);
        assert_eq!(visible_library(&sampled, player), expected);
        checked_command(
            &mut sampled,
            Action::ChooseMode {
                player: opponent,
                mode: 0,
            },
        );
        assert_eq!(sampled.state.suspended_decision, Some(caller));
        for object in looked.iter().rev() {
            checked_command(
                &mut sampled,
                Action::SelectCard {
                    player,
                    object: *object,
                },
            );
        }
        assert_eq!(
            sampled.library_top_objects(player, 3),
            looked.into_iter().rev().collect::<Vec<_>>()
        );
        assert_eq!(sampled.state.players[player.0].life, 22);
        assert!(sampled.state.suspended_decision.is_none());
        assert!(
            visible_library(&sampled, player).is_empty(),
            "permission ends with its owner"
        );
        assert_eq!(game.state.deterministic_hash(), original);
    }
}

#[test]
fn retained_search_permission_preserves_decision_membership_until_caller_finishes() {
    use crate::agent::observation::Observation;
    use crate::flow::decision::{Decision, EffectFrame, SuspendedResolution};

    for player in [PlayerId(0), PlayerId(1)] {
        let opponent = PlayerId(1 - player.0);
        let (mut game, _, _) = fixture(player);
        let hand = game
            .state
            .zones
            .zone_objects(ZoneType::Hand, opponent)
            .to_vec();
        let library = game
            .state
            .zones
            .zone_objects(ZoneType::Library, opponent)
            .to_vec();
        let caller = SuspendedResolution {
            frame: EffectFrame::new(player, vec![]),
            decision: Decision::SearchNamed {
                player,
                owner: opponent,
                name: "Mountain".into(),
                remaining: 1,
            },
        };
        game.run_before_decision(EffectFrame::new(opponent, vec![pause()]), caller);
        game.publish_action_space(game.suspended_decision_action_space().unwrap());
        assert!(game.searches_private_zones(player, opponent));
        let obs = Observation::for_player(&game, player);
        assert_eq!(
            obs.opponent_cards
                .iter()
                .filter(|object| object.zone == ZoneType::Hand)
                .count(),
            hand.len()
        );
        assert_eq!(
            obs.opponent_cards
                .iter()
                .filter(|object| object.zone == ZoneType::Library)
                .count(),
            library.len()
        );
        game.determinize(player, 242).unwrap();
        assert_eq!(
            game.state
                .zones
                .zone_objects(ZoneType::Hand, opponent)
                .to_vec(),
            hand
        );
        assert_eq!(
            game.state
                .zones
                .zone_objects(ZoneType::Library, opponent)
                .to_vec(),
            library
        );
        checked_command(
            &mut game,
            Action::ChooseMode {
                player: opponent,
                mode: 0,
            },
        );
        assert!(game.searches_private_zones(player, opponent));
        checked_command(&mut game, Action::Decline { player });
        assert!(!game.searches_private_zones(player, opponent));
        let obs = Observation::for_player(&game, player);
        assert!(obs
            .opponent_cards
            .iter()
            .all(|object| object.zone != ZoneType::Hand && object.zone != ZoneType::Library));
    }
}

#[test]
fn surveil_commits_the_partition_once_and_announces_only_after_kept_order() {
    for player in [PlayerId(0), PlayerId(1)] {
        let (mut game, _, _) = fixture(player);
        let objects = game.library_top_objects(player, 4);
        stack(
            &mut game,
            Entry::Spell,
            player,
            vec![Effect::Surveil { count: 4 }, Effect::gain_life(3)],
            vec![],
            vec![],
        );
        game.resolve_top_of_stack();
        game.publish_action_space(game.suspended_decision_action_space().unwrap());
        let surveils = |game: &Game| {
            game.state
                .events
                .iter()
                .filter(|event| matches!(event, GameEvent::Surveilled { .. }))
                .count()
        };
        for object in &objects[..2] {
            checked_command(
                &mut game,
                Action::SelectCard {
                    player,
                    object: *object,
                },
            );
            assert_eq!(game.library_top_objects(player, 4), objects);
            assert_eq!(surveils(&game), 0);
            assert_eq!(game.state.players[player.0].life, 20);
        }
        checked_command(&mut game, Action::Decline { player });
        assert!(objects[..2].iter().all(|object| game
            .state
            .zones
            .zone_of(game.state.now(*object))
            == Some(ZoneType::Graveyard)));
        assert_eq!(surveils(&game), 0, "ordering is still part of surveil");
        let events_before = game.state.events.len();
        for object in objects[2..].iter().rev() {
            checked_command(
                &mut game,
                Action::SelectCard {
                    player,
                    object: *object,
                },
            );
        }
        assert_eq!(surveils(&game), 1);
        assert_eq!(game.state.players[player.0].life, 23);
        assert!(
            matches!(game.state.events[events_before], GameEvent::Surveilled { player: who } if who == player)
        );
        for object in &objects[..2] {
            assert_eq!(
                game.state
                    .events
                    .iter()
                    .filter(|event| matches!(event,
                GameEvent::CardMoved { departed: Some(moved), to: ZoneType::Graveyard, .. } if moved == object))
                    .count(),
                1
            );
        }
    }
}

#[test]
fn library_keyword_completion_handles_empty_libraries_and_zero_counts() {
    for scry in [false, true] {
        for count in [0, 2] {
            let (mut game, _, _) = fixture(PlayerId(0));
            for object in game.library_top_objects(PlayerId(0), usize::MAX) {
                game.move_object(object, ZoneType::Graveyard);
            }
            stack(
                &mut game,
                Entry::Spell,
                PlayerId(0),
                vec![if scry {
                    Effect::Scry { count }
                } else {
                    Effect::Surveil { count }
                }],
                vec![],
                vec![],
            );
            game.resolve_top_of_stack();
            assert!(game.state.suspended_decision.is_none());
            assert_eq!(
                game.state
                    .events
                    .iter()
                    .filter(|event| matches!(
                        event,
                        GameEvent::Scried { .. } | GameEvent::Surveilled { .. }
                    ))
                    .count(),
                usize::from(count > 0)
            );
        }
    }
}

#[test]
fn destruction_count_survives_a_child_returning_one_of_the_destroyed_creatures() {
    use crate::state::ability::Scope;

    let player = PlayerId(0);
    let (mut game, first, _) = fixture(player);
    let victim = game.permanent_object(first).unwrap();
    stack(
        &mut game,
        Entry::Spell,
        player,
        vec![
            Effect::destroy(ObjectSelector::each(
                Scope::Battlefield,
                CardPredicate::creature(),
            )),
            Effect::SelectCards {
                from: ObjectSelector::Moved,
                predicate: CardPredicate::creature(),
                min: 1,
                max: EffectValue::Fixed(1),
                effects: vec![Effect::return_to_hand(ObjectSelector::Selected)],
                if_declined: vec![],
            },
            Effect::gain_life(ObjectSelector::Moved.count()),
        ],
        vec![],
        vec![],
    );
    game.resolve_top_of_stack();
    let choice = Action::SelectCard {
        player,
        object: game.state.now(victim),
    };
    game.publish_action_space(game.suspended_decision_action_space().unwrap());
    checked_command(&mut game, choice);
    assert_eq!(
        game.state.zones.zone_of(game.state.now(victim)),
        Some(ZoneType::Hand)
    );
    assert_eq!(game.state.players[player.0].life, 22);
}

#[test]
fn selected_cards_bind_exact_objects_and_keep_the_callers_mill_result() {
    for player in [PlayerId(0), PlayerId(1)] {
        let (mut game, _, _) = fixture(player);
        let milled = game.library_top_objects(player, 3);
        let select = |min, max, effects| Effect::SelectCards {
            from: ObjectSelector::CardsThisWay,
            predicate: CardPredicate::default(),
            min,
            max: EffectValue::Fixed(max),
            effects,
            if_declined: vec![],
        };
        stack(
            &mut game,
            Entry::Spell,
            player,
            vec![
                Effect::MillCards {
                    count: EffectValue::Fixed(3),
                    target: None,
                },
                select(
                    1,
                    2,
                    vec![
                        Effect::return_to_hand(ObjectSelector::Selected),
                        pause(),
                        // The selected graveyard incarnation has departed. This
                        // must not move the same physical card out of its hand.
                        Effect::exile(ObjectSelector::Selected),
                    ],
                ),
                select(0, 1, vec![Effect::return_to_hand(ObjectSelector::Selected)]),
                Effect::gain_life(3),
            ],
            vec![],
            vec![],
        );
        game.resolve_top_of_stack();
        // The selection offers the graveyard objects the milled cards became.
        let milled: Vec<_> = milled
            .iter()
            .map(|object| game.state.now(*object))
            .collect();
        game.publish_action_space(game.suspended_decision_action_space().unwrap());
        assert!(!game
            .action_space()
            .unwrap()
            .actions
            .iter()
            .any(|a| matches!(a, Action::Decline { .. })));
        checked_command(
            &mut game,
            Action::SelectCard {
                player,
                object: milled[0],
            },
        );
        assert_eq!(
            game.state.zones.zone_of(game.state.now(milled[0])),
            Some(ZoneType::Graveyard),
            "selection has not moved anything"
        );
        let root = game.state.deterministic_hash();
        for second in [None, Some(milled[1])] {
            let mut fork = game.clone();
            checked_command(
                &mut fork,
                second.map_or(Action::Decline { player }, |object| Action::SelectCard {
                    player,
                    object,
                }),
            );
            assert_eq!(
                fork.state.zones.zone_of(fork.state.now(milled[0])),
                Some(ZoneType::Hand)
            );
            assert_eq!(fork.state.players[player.0].life, 20, "parent suffix waits");
            checked_command(&mut fork, Action::ChooseMode { player, mode: 0 });
            assert_eq!(
                fork.state.zones.zone_of(fork.state.now(milled[0])),
                Some(ZoneType::Hand),
                "selected refs never follow a moved incarnation"
            );
            let offered: Vec<_> = fork
                .action_space()
                .unwrap()
                .actions
                .iter()
                .filter_map(|action| match action {
                    Action::SelectCard { object, .. } => Some(*object),
                    _ => None,
                })
                .collect();
            assert!(!offered.contains(&milled[0]));
            assert_eq!(offered.contains(&milled[1]), second.is_none());
            assert!(
                offered.contains(&milled[2]),
                "parent still owns its original mill result"
            );
            checked_command(
                &mut fork,
                Action::SelectCard {
                    player,
                    object: milled[2],
                },
            );
            assert_eq!(
                fork.state.zones.zone_of(fork.state.now(milled[2])),
                Some(ZoneType::Hand)
            );
            assert_eq!(fork.state.players[player.0].life, 23);
            assert!(fork.state.suspended_decision.is_none());
            assert_eq!(
                fork.state
                    .events
                    .iter()
                    .filter(|e| matches!(e, GameEvent::SpellResolved { .. }))
                    .count(),
                1
            );
            assert_eq!(game.state.deterministic_hash(), root);
        }
    }
}

#[test]
fn keyword_completion_captures_triggers_before_the_callers_suffix_removes_sources() {
    use crate::state::ability::{Ability, Actor, KeywordAction, TriggerCondition};

    let player = PlayerId(0);
    let (mut game, _, _) = fixture(player);
    stack(
        &mut game,
        Entry::Triggered,
        player,
        vec![Effect::Surveil { count: 1 }, Effect::sacrifice_this()],
        vec![],
        vec![],
    );
    let source = match game.state.stack_objects.last().unwrap() {
        StackObject::TriggeredAbility(ability) => ability.source_object(),
        _ => unreachable!(),
    };
    game.state[source].abilities = vec![Ability::Triggered {
        condition: TriggerCondition::Performs {
            who: Actor::You,
            any_of: vec![KeywordAction::Scry, KeywordAction::Surveil],
        },
        intervening_if: None,
        effects: vec![Effect::gain_life(1)],
    }];
    game.resolve_top_of_stack();
    game.publish_action_space(game.suspended_decision_action_space().unwrap());
    assert!(game.state.pending_triggers.is_empty());
    checked_command(&mut game, Action::Decline { player });
    assert_eq!(
        game.state.zones.zone_of(game.state.now(source)),
        Some(ZoneType::Graveyard)
    );
    let occurrences = game
        .state
        .pending_triggers
        .iter()
        .filter(|t| t.source_object() == source)
        .count()
        + game
            .state
            .stack_objects
            .iter()
            .filter(|object| {
                matches!(object,
            StackObject::TriggeredAbility(ability) if ability.source_object() == source)
            })
            .count();
    assert_eq!(occurrences, 1, "completion matched before source departure");
    assert_eq!(
        game.state.players[0].life, 20,
        "queued occurrence has not resolved"
    );
}

fn counter_replacement_drivers(game: &mut Game, player: PlayerId) {
    use crate::state::card::ReplacementEffect;
    for (multiplier, additional) in [(1, 1), (2, 0)] {
        let id = game
            .scenario_force_battlefield(player, "Mountain", true)
            .unwrap();
        let object = game.state.permanents[id].as_ref().unwrap().object;
        game.state[object].replacement_effects = vec![ReplacementEffect::ModifyCounters {
            effects_only: false,
            kind: Some(CounterKind::PlusOnePlusOne),
            predicate: CardPredicate::default(),
            multiplier,
            additional,
        }];
    }
}

#[test]
fn retained_batch_counter_order_is_affected_players_choice_and_commits_once() {
    for caster in [PlayerId(0), PlayerId(1)] {
        let affected = PlayerId(1 - caster.0);
        let (mut root, first, second) = fixture(caster);
        counter_replacement_drivers(&mut root, affected);
        stack(
            &mut root,
            Entry::Spell,
            caster,
            vec![
                Effect::put_counters(
                    ObjectSelector::Each {
                        of: crate::state::ability::Scope::OpponentsControl,
                        predicate: CardPredicate {
                            card_type: Some(CardType::Creature),
                            ..Default::default()
                        },
                        except: None,
                        max_mana_value: None,
                        exact_mana_value: None,
                    },
                    CounterKind::PlusOnePlusOne,
                    1,
                ),
                Effect::gain_life(7),
            ],
            vec![],
            vec![],
        );
        root.resolve_top_of_stack();
        root.publish_action_space(root.suspended_decision_action_space().unwrap());
        let before = root.state.deterministic_hash();
        for (mode, expected) in [(0, 4), (1, 3)] {
            let mut game = root.clone();
            assert_eq!(game.action_space().unwrap().player, Some(affected));
            checked_command(
                &mut game,
                Action::ChooseMode {
                    player: affected,
                    mode,
                },
            );
            for id in [first, second] {
                assert_eq!(
                    game.state.permanents[id]
                        .as_ref()
                        .unwrap()
                        .counter_count(CounterKind::PlusOnePlusOne),
                    0,
                    "all proposals wait until the whole batch is ready"
                );
            }
            assert_eq!(game.state.players[caster.0].life, 20);
            checked_command(
                &mut game,
                Action::ChooseMode {
                    player: affected,
                    mode,
                },
            );
            for id in [first, second] {
                assert_eq!(
                    game.state.permanents[id]
                        .as_ref()
                        .unwrap()
                        .counter_count(CounterKind::PlusOnePlusOne),
                    expected
                );
                assert_eq!(game.state.events.iter().filter(|event| matches!(event, GameEvent::CountersPut { permanent, count, .. } if *permanent == id && *count == expected)).count(), 1);
            }
            assert_eq!(game.state.players[caster.0].life, 27);
            assert_eq!(
                game.state
                    .events
                    .iter()
                    .filter(|event| matches!(event, GameEvent::SpellResolved { .. }))
                    .count(),
                1
            );
        }
        assert_eq!(root.state.deterministic_hash(), before);
    }
}

/// An exiled object whose own entry replacement enables the counter drivers.
fn exiled_counter_entry(game: &mut Game, player: PlayerId) -> ObjectId {
    use crate::state::card::ReplacementEffect;

    let object = game
        .state
        .latest_objects()
        .find(|(index, object)| {
            object.owner == player
                && object.name == "Gray Ogre"
                && game.state.zones.zone_of(game.state.now(*index)) != Some(ZoneType::Battlefield)
        })
        .unwrap()
        .0;
    let object = game.move_object(object, ZoneType::Exile);
    game.state[object].replacement_effects = vec![ReplacementEffect::EntersWithPlusOneCounters {
        count: 1,
        condition: None,
    }];
    object
}

#[test]
fn retained_batch_rechecks_new_entry_replacements_without_reapplying_identity() {
    use crate::flow::decision::{EffectFrame, FrameCompletion};
    let player = PlayerId(0);
    let (mut game, _, _) = fixture(player);
    counter_replacement_drivers(&mut game, player);
    let object = exiled_counter_entry(&mut game, player);
    let mut frame = EffectFrame::new(
        player,
        vec![Effect::return_to_battlefield(ObjectSelector::This, false)],
    );
    frame.bind_source(Some(object), game.state.live(game.state.now(object)));
    frame.completion = FrameCompletion::Continue;
    game.run_frame(frame);
    game.publish_action_space(game.suspended_decision_action_space().unwrap());
    assert_eq!(
        game.state.zones.zone_of(game.state.now(object)),
        Some(ZoneType::Exile)
    );
    assert_eq!(
        game.action_space().unwrap().actions.len(),
        2,
        "own entry created counters, making both battlefield replacements applicable"
    );
    checked_command(&mut game, Action::ChooseMode { player, mode: 0 });
    let id = game.state.permanent_of(game.state.now(object)).unwrap();
    assert_eq!(
        game.state.permanents[id]
            .as_ref()
            .unwrap()
            .counter_count(CounterKind::PlusOnePlusOne),
        4
    );
    assert!(
        game.state.suspended_decision.is_none(),
        "each identity applies once"
    );
}

#[test]
fn retained_batch_inspection_finishes_entry_before_more_selection_or_ordering() {
    use crate::flow::decision::{Decision, EffectFrame};

    let player = PlayerId(0);
    for max_select in [1, 2] {
        let (mut game, _, _) = fixture(player);
        counter_replacement_drivers(&mut game, player);
        let entrant = exiled_counter_entry(&mut game, player);
        let entrant = game.move_object(entrant, ZoneType::Library);
        let looked = game.library_top_objects(player, 4);
        assert_eq!(looked[0], entrant);
        game.run_frame(EffectFrame::new(
            player,
            vec![
                Effect::LookAndSelect {
                    look: EffectValue::Fixed(4),
                    predicate: CardPredicate::default(),
                    min_select: 0,
                    max_select,
                    reveal: false,
                    to_battlefield: true,
                    random_order: false,
                },
                Effect::gain_life(3),
            ],
        ));
        game.publish_action_space(game.suspended_decision_action_space().unwrap());
        checked_command(
            &mut game,
            Action::SelectCard {
                player,
                object: entrant,
            },
        );
        assert_eq!(
            game.state.zones.zone_of(game.state.now(entrant)),
            Some(ZoneType::Library)
        );
        assert!(matches!(
            game.state.suspended_decision.as_ref().unwrap().decision,
            Decision::OrderReplacement { .. }
        ));
        checked_command(&mut game, Action::ChooseMode { player, mode: 0 });
        assert_eq!(
            game.state.zones.zone_of(game.state.now(entrant)),
            Some(ZoneType::Battlefield)
        );
        if max_select == 2 {
            assert!(matches!(
                game.state.suspended_decision.as_ref().unwrap().decision,
                Decision::LookAndSelect { .. }
            ));
            checked_command(
                &mut game,
                Action::SelectCard {
                    player,
                    object: looked[1],
                },
            );
        }
        assert!(matches!(
            game.state.suspended_decision.as_ref().unwrap().decision,
            Decision::OrderLibrary {
                end: LibraryEnd::Bottom,
                ..
            }
        ));
        for object in looked[max_select..].iter().rev() {
            assert_eq!(game.state.players[player.0].life, 20, "suffix still waits");
            checked_command(
                &mut game,
                Action::SelectCard {
                    player,
                    object: *object,
                },
            );
        }
        assert_eq!(game.state.players[player.0].life, 23);
        assert!(game.state.suspended_decision.is_none());
        assert!(looked[..max_select].iter().all(|object| game
            .state
            .zones
            .zone_of(game.state.now(*object))
            == Some(ZoneType::Battlefield)));
    }
}

#[test]
fn retained_batch_linked_entry_suspends_actual_mana_cost_completion() {
    use crate::flow::decision::{Decision, EffectFrame, SuspendedResolution};
    use crate::flow::payment::{CostComponent, PaymentProgress};
    use crate::flow::trigger::ExileLink;
    use crate::state::mana::{Mana, ManaCost};
    let player = PlayerId(0);
    let (mut game, _, _) = fixture(player);
    counter_replacement_drivers(&mut game, player);
    let source = game
        .scenario_force_battlefield(player, "Gray Ogre", true)
        .unwrap();
    let source_object = game.state.permanents[source].as_ref().unwrap().object;
    let source_ref = game.permanent_object(source).unwrap();
    game.state[source_object].activated_abilities = vec![ActivatedAbilityDefinition {
        mana_cost: ManaCost::parse("{1}"),
        life_cost: 2,
        tap_source: true,
        sacrifice_source: true,
        effects: vec![Effect::Modal {
            modes: vec![
                vec![Effect::add_mana(Mana::parse("U"))],
                vec![Effect::add_mana(Mana::parse("B"))],
            ],
        }],
        ..Default::default()
    }];
    game.state[source_object].abilities = vec![crate::state::ability::Ability::Triggered {
        condition: crate::state::ability::TriggerCondition::BecomesTapped {
            subject: crate::state::ability::TriggerSubject::This,
        },
        effects: vec![Effect::gain_life(9)],
        intervening_if: None,
    }];
    let returned = exiled_counter_entry(&mut game, player);
    let returned_ref = game.state.now(returned);
    game.state.exile_links.push(ExileLink {
        source: source_ref,
        exiled_object: returned_ref,
        returns: true,
    });
    game.state.players[player.0].mana_pool = Mana::parse("R").into();
    game.state.suspended_decision = Some(SuspendedResolution {
        frame: EffectFrame::new(player, vec![Effect::gain_life(3)]),
        decision: Decision::Payment(PaymentProgress::new(
            player,
            CostComponent::mana(ManaCost::parse("{U}")),
            vec![Effect::gain_life(2)],
            vec![],
        )),
    });
    game.publish_action_space(game.suspended_decision_action_space().unwrap());
    checked_command(
        &mut game,
        Action::ActivateAbility {
            player,
            permanent: source,
            ability_index: 0,
        },
    );
    assert_eq!(
        game.state.zones.zone_of(game.state.now(source_object)),
        Some(ZoneType::Graveyard)
    );
    assert_eq!(
        game.state.zones.zone_of(game.state.now(returned)),
        Some(ZoneType::Exile)
    );
    assert_eq!(game.state.players[player.0].life, 18);
    assert_eq!(
        game.state.pending_triggers.len(),
        1,
        "the paid tap's trigger is retained before its source leaves"
    );
    assert!(
        game.state.stack_objects.is_empty(),
        "replacement choice does not place triggers"
    );

    assert_eq!(game.state.players[player.0].mana_pool.total(), 0);
    assert!(matches!(
        game.state.suspended_decision.as_ref().unwrap().decision,
        Decision::OrderReplacement { .. }
    ));
    checked_command(&mut game, Action::ChooseMode { player, mode: 0 });
    assert!(matches!(
        game.state.suspended_decision.as_ref().unwrap().decision,
        Decision::ManaChoice { .. }
    ));
    assert_eq!(
        game.state.players[player.0].mana_pool.total(),
        0,
        "output still needs its ordinary choice"
    );
    checked_command(&mut game, Action::ChooseMode { player, mode: 0 });
    checked_command(&mut game, Action::PayCost { player });
    assert_eq!(game.state.players[player.0].life, 23);
    assert_eq!(game.state.players[player.0].mana_pool.total(), 0);
    assert_eq!(game.state.events.iter().filter(|event| matches!(event, GameEvent::PermanentSacrificed { object_id: object, .. } if *object == source_object)).count(), 1);
    assert_eq!(game.state.events.iter().filter(|event| matches!(event, GameEvent::CardMoved { departed: Some(object), to: ZoneType::Battlefield, .. } if *object == returned_ref)).count(), 1);
}

#[test]
fn retained_batch_combat_defers_every_hit_until_all_replacements_are_chosen() {
    use crate::flow::combat::CombatState;
    use crate::state::card::ReplacementEffect;
    let defender = PlayerId(1);
    let (mut root, first, second) = fixture(defender);
    for effect in [
        ReplacementEffect::PreventDamageToController { amount: 1 },
        ReplacementEffect::DoubleDamageToController,
    ] {
        let id = root
            .scenario_force_battlefield(defender, "Mountain", true)
            .unwrap();
        let object = root.state.permanents[id].as_ref().unwrap().object;
        root.state[object].replacement_effects = vec![effect];
    }
    root.state.combat = Some(Box::new(CombatState {
        attackers: vec![first, second],
        ..Default::default()
    }));
    root.resolve_combat_damage();
    root.publish_action_space(root.suspended_decision_action_space().unwrap());
    for (mode, life) in [(0, 16), (1, 14)] {
        let mut game = root.clone();
        checked_command(
            &mut game,
            Action::ChooseMode {
                player: defender,
                mode,
            },
        );
        assert_eq!(game.state.players[defender.0].life, 20);
        checked_command(
            &mut game,
            Action::ChooseMode {
                player: defender,
                mode,
            },
        );
        assert_eq!(game.state.players[defender.0].life, life);
        let mut batches = game.state.events.iter().filter_map(|event| match event {
            GameEvent::CombatDamageToPlayers { hits } => Some(hits),
            _ => None,
        });
        let expected = [first, second].map(|id| crate::flow::event::CombatPlayerHit {
            source: game.permanent_object(id).unwrap(),
            recipient: defender,
            amount: (20 - life) / 2,
        });
        assert_eq!(batches.next().unwrap().as_slice(), expected);
        assert!(batches.next().is_none());
    }
}

#[test]
fn retained_batch_state_based_departure_retains_linked_entry_and_lki() {
    use crate::flow::{sba::SbaProgress, trigger::ExileLink};
    let player = PlayerId(0);
    let (mut game, source, _) = fixture(player);
    counter_replacement_drivers(&mut game, player);
    let source_object = game.state.permanents[source].as_ref().unwrap().object;
    let source_ref = game.permanent_object(source).unwrap();
    let returned = exiled_counter_entry(&mut game, player);
    let returned_ref = game.state.now(returned);
    game.state.exile_links.push(ExileLink {
        source: source_ref,
        exiled_object: returned_ref,
        returns: true,
    });
    game.state.permanents[source].as_mut().unwrap().damage = 2;
    assert_eq!(game.perform_state_based_actions(), SbaProgress::Pending);
    game.publish_action_space(game.suspended_decision_action_space().unwrap());
    assert_eq!(
        game.state.zones.zone_of(game.state.now(source_object)),
        Some(ZoneType::Graveyard)
    );
    assert_eq!(game.object_lki(source_ref).unwrap().power, 2);
    assert_eq!(
        game.state.zones.zone_of(game.state.now(returned)),
        Some(ZoneType::Exile)
    );
    checked_command(&mut game, Action::ChooseMode { player, mode: 1 });
    let id = game.state.permanent_of(game.state.now(returned)).unwrap();
    assert_eq!(
        game.state.permanents[id]
            .as_ref()
            .unwrap()
            .counter_count(CounterKind::PlusOnePlusOne),
        3
    );
    assert_eq!(game.perform_state_based_actions(), SbaProgress::Stable);
    assert_eq!(game.state.events.iter().filter(|event| matches!(event, GameEvent::CardMoved { departed: Some(object), from: Some(ZoneType::Battlefield), .. } if *object == source_ref)).count(), 1);
}

#[test]
fn hidden_selection_partial_discard_survives_both_samplers_and_divergent_commands() {
    use crate::{
        flow::decision::{Decision, EffectFrame},
        possible_worlds::{MaterializeMode, PossibleWorldSpace},
    };

    for player in [PlayerId(0), PlayerId(1)] {
        let viewer = PlayerId(1 - player.0);
        let (mut root, _, _) = fixture(player);
        let hand = root
            .state
            .zones
            .zone_objects(ZoneType::Hand, player)
            .to_vec();
        assert!(hand.len() >= 2);
        root.run_frame(EffectFrame::new(
            player,
            vec![Effect::discard_draw(2, 1), Effect::gain_life(3)],
        ));
        root.publish_action_space(root.suspended_decision_action_space().unwrap());
        checked_command(
            &mut root,
            Action::SelectCard {
                player,
                object: hand[0],
            },
        );
        assert_eq!(
            root.state.zones.size(ZoneType::Hand, player),
            hand.len(),
            "partial selection commits no zone moves"
        );
        let original = root.state.deterministic_hash();
        let mut known = root.clone();
        known.determinize(player, 241).unwrap();
        assert_eq!(
            known.state.suspended_decision, root.state.suspended_decision,
            "the choosing viewer retains exact choices"
        );
        let space = PossibleWorldSpace::for_viewer(&root, viewer);
        let mut materialized = space
            .materialize_index(&root, 0, 17, MaterializeMode::RefreshOpponentCommitment)
            .unwrap();
        let mut sampled = root.clone();
        sampled.determinize(viewer, 91).unwrap();
        for branch in [&mut materialized, &mut sampled] {
            let Decision::SelectCards {
                selected,
                candidates,
                ..
            } = &branch.state.suspended_decision.as_ref().unwrap().decision
            else {
                panic!("selection retained");
            };
            assert_eq!(selected.len(), 1, "a prior choice cannot disappear");
            assert!(candidates.iter().all(|object| branch.state.zones.contains(
                *object,
                ZoneType::Hand,
                player
            )));
            assert!(selected.iter().all(|object| candidates.contains(object)));
            let first = selected[0];
            let second = branch
                .action_space()
                .unwrap()
                .actions
                .iter()
                .find_map(|action| match action {
                    Action::SelectCard {
                        object: object_id, ..
                    } => Some(*object_id),
                    _ => None,
                })
                .unwrap();
            for choice in [
                Action::Decline { player },
                Action::SelectCard {
                    player,
                    object: second,
                },
            ] {
                let mut fork = branch.clone();
                let count = if matches!(choice, Action::Decline { .. }) {
                    1
                } else {
                    2
                };
                let graveyard = fork.state.zones.size(ZoneType::Graveyard, player);
                checked_command(&mut fork, choice);
                assert_eq!(
                    fork.state.zones.zone_of(fork.state.now(first)),
                    Some(ZoneType::Graveyard)
                );
                assert_eq!(
                    fork.state.zones.size(ZoneType::Graveyard, player),
                    graveyard + count
                );
                assert_eq!(
                    fork.state.zones.size(ZoneType::Hand, player),
                    hand.len(),
                    "draw once per committed discard"
                );
                assert_eq!(
                    fork.state.players[player.0].life, 23,
                    "one caller finalization"
                );
            }
        }
        assert_eq!(root.state.deterministic_hash(), original);
    }
}

#[test]
fn hidden_selection_child_and_saved_caller_share_one_sampled_binding() {
    use crate::{
        agent::observation::Observation,
        flow::decision::{Decision, EffectFrame, FrameCompletion, SuspendedResolution},
        possible_worlds::HiddenHandConstraints,
        state::ability::PlayerSelector,
    };
    let player = PlayerId(0);
    let viewer = PlayerId(1);
    let (mut root, _, _) = fixture(player);
    let hand = root
        .state
        .zones
        .zone_objects(ZoneType::Hand, player)
        .to_vec();
    root.run_frame(EffectFrame::new(
        player,
        vec![
            Effect::SelectCards {
                from: ObjectSelector::Zone(ZoneType::Hand),
                predicate: CardPredicate::default(),
                min: 1,
                max: 1.into(),
                effects: vec![
                    Effect::ForPlayer {
                        player: PlayerSelector::EachOpponent,
                        effects: vec![pause()],
                    },
                    Effect::move_to(ObjectSelector::Selected, ZoneType::Graveyard),
                ],
                if_declined: vec![],
            },
            Effect::gain_life(3),
        ],
    ));
    root.publish_action_space(root.suspended_decision_action_space().unwrap());
    checked_command(
        &mut root,
        Action::SelectCard {
            player,
            object: hand[0],
        },
    );
    // Retain the choice-bearing child beneath another scope and saved decision.
    let mut caller = root.state.suspended_decision_mut().take().unwrap();
    caller.frame.remembered = root.capture_bindings(&ObjectSelector::Selected, &caller.frame);
    let mut child = EffectFrame::new(viewer, vec![]);
    child.remembered = caller.frame.remembered.clone();
    child.selected = caller.frame.selected.clone();
    child.selected_from = caller.frame.selected_from.clone();
    child.completion = FrameCompletion::Decision(Box::new(caller));
    *root.state.suspended_decision_mut() = Some(SuspendedResolution {
        frame: child,
        decision: Decision::Modal {
            player: viewer,
            modes: vec![vec![], vec![Effect::gain_life(2)]],
        },
    });
    root.publish_action_space(root.suspended_decision_action_space().unwrap());
    let hidden_before = Observation::for_player(&root, viewer);
    let constraints = HiddenHandConstraints::for_viewer(&root, viewer);
    let compatible = BTreeMap::from([("Mountain".to_string(), hand.len() as u32)]);
    let materialized = constraints
        .materialize_hand(&root, &compatible, 19)
        .unwrap();
    let mut sampled = root.clone();
    sampled.determinize(viewer, 107).unwrap();
    for mut branch in [materialized, sampled] {
        let suspended = branch.state.suspended_decision.as_ref().unwrap();
        let FrameCompletion::Decision(caller) = &suspended.frame.completion else {
            panic!("caller retained");
        };
        assert_eq!(
            suspended.frame.selected, caller.frame.selected,
            "inherited bindings remap together"
        );
        assert_eq!(
            suspended.frame.remembered[0].object,
            caller.frame.selected[0]
        );
        assert_eq!(caller.frame.remembered[0].object, caller.frame.selected[0]);
        let selected = caller.frame.selected[0];
        assert!(branch
            .state
            .zones
            .contains(selected, ZoneType::Hand, player));
        let hidden_after = Observation::for_player(&branch, viewer);
        assert_eq!(
            hidden_after.opponent_cards, hidden_before.opponent_cards,
            "private selection membership is not observation data"
        );
        checked_command(
            &mut branch,
            Action::ChooseMode {
                player: viewer,
                mode: 0,
            },
        );
        checked_command(
            &mut branch,
            Action::ChooseMode {
                player: viewer,
                mode: 0,
            },
        );
        assert_eq!(
            branch.state.zones.zone_of(branch.state.now(selected)),
            Some(ZoneType::Graveyard)
        );
        assert_eq!(branch.state.players[player.0].life, 23);
    }
}

#[test]
fn hidden_selection_library_commitments_survive_sampling_without_revealing_order() {
    use crate::{
        agent::observation::Observation,
        flow::decision::{Decision, EffectFrame},
        possible_worlds::{MaterializeError, MaterializeMode, PossibleWorld, PossibleWorldSpace},
    };
    let player = PlayerId(0);
    let viewer = PlayerId(1);
    let (mut root, _, _) = fixture(player);
    // A restricted, partial search must leave at least its committed number
    // of eligible cards in the sampled library, even when its chooser is hidden.
    let bolts: Vec<_> = root
        .state
        .latest_objects()
        .filter(|(_, object)| object.owner == player && object.name == "Lightning Bolt")
        .map(|(index, _)| index)
        .collect();
    for object in &bolts {
        root.move_object(*object, ZoneType::Library);
    }
    while root.state.zones.size(ZoneType::Hand, player) < 7 {
        let object = root
            .state
            .zones
            .zone_objects(ZoneType::Library, player)
            .iter()
            .copied()
            .find(|object| root.state[*object].name == "Mountain")
            .unwrap();
        root.move_object(object, ZoneType::Hand);
    }
    root.run_frame(EffectFrame::new(
        player,
        vec![
            Effect::search_library(
                CardPredicate {
                    card_type: Some(CardType::Instant),
                    ..Default::default()
                },
                2.into(),
                vec![Effect::return_to_hand(ObjectSelector::Selected)],
            ),
            Effect::gain_life(3),
        ],
    ));
    root.publish_action_space(root.suspended_decision_action_space().unwrap());
    checked_command(
        &mut root,
        Action::SelectCard {
            player,
            object: bolts[0],
        },
    );
    let owner_before = Observation::for_player(&root, player);
    let mut known = root.clone();
    known.determinize(player, 13).unwrap();
    assert_eq!(
        Observation::for_player(&known, player).agent_cards,
        owner_before.agent_cards,
        "inspection reveals membership, never a shuffled draw order"
    );
    assert_eq!(
        known.state.suspended_decision,
        root.state.suspended_decision
    );
    let space = PossibleWorldSpace::for_viewer(&root, viewer);
    let hand_size = root.state.zones.size(ZoneType::Hand, player);
    let impossible = PossibleWorld {
        hand: BTreeMap::from([
            ("Lightning Bolt".into(), bolts.len() as u32),
            ("Mountain".into(), (hand_size - bolts.len()) as u32),
        ]),
        weight: 1,
    };
    assert!(
        matches!(
            space.materialize(&root, &impossible, 29),
            Err(MaterializeError::InconsistentWorld)
        ),
        "a world cannot erase a committed eligible selection"
    );
    assert!(
        space.worlds().iter().all(
            |world| world.hand.get("Lightning Bolt").copied().unwrap_or(0) < bolts.len() as u32
        ),
        "incompatible deals carry no probability mass"
    );
    let index = space
        .worlds()
        .iter()
        .position(|world| world.hand.get("Lightning Bolt").copied().unwrap_or(0) == 0)
        .unwrap();
    let materialized = space
        .materialize_index(&root, index, 31, MaterializeMode::RefreshOpponentCommitment)
        .unwrap();
    let mut sampled = root.clone();
    sampled.determinize(viewer, 7).unwrap();
    for mut branch in [sampled, materialized] {
        let Decision::SelectCards { selected, .. } =
            &branch.state.suspended_decision.as_ref().unwrap().decision
        else {
            panic!("search retained");
        };
        assert_eq!(selected.len(), 1);
        let selected = selected[0];
        assert_eq!(
            branch.state.zones.zone_of(branch.state.now(selected)),
            Some(ZoneType::Library)
        );
        assert_eq!(branch.state[selected].name, "Lightning Bolt");
        assert!(Observation::for_player(&branch, viewer)
            .opponent_cards
            .iter()
            .all(|object| object.zone != ZoneType::Library && object.zone != ZoneType::Hand));
        checked_command(&mut branch, Action::Decline { player });
        assert_eq!(
            branch.state.zones.zone_of(branch.state.now(selected)),
            Some(ZoneType::Hand)
        );
        assert_eq!(
            branch.state.players[player.0]
                .known_hand
                .get(&branch.state[branch.state.now(selected)].definition_id),
            Some(&1)
        );
        assert_eq!(branch.state.events.iter().filter(|event| matches!(event, GameEvent::LibrarySearched { player: owner } if *owner == player)).count(), 1);
        assert_eq!(branch.state.players[player.0].life, 23);
    }
}

#[test]
fn hidden_selection_merchant_bottoms_source_before_linked_return_and_shuffles_after() {
    use crate::{
        flow::{
            decision::{Decision, EffectFrame},
            trigger::ExileLink,
        },
        state::ability::Ability,
    };
    let player = PlayerId(0);
    let deck = BTreeMap::from([
        ("Mountain".into(), 24),
        ("Gray Ogre".into(), 12),
        ("Unlucky Cabbage Merchant".into(), 4),
    ]);
    let mut game = Game::new(
        vec![
            PlayerConfig::new("p0", deck.clone()),
            PlayerConfig::new("p1", deck),
        ],
        241,
        false,
    );
    game.keep_opening_hands();
    let merchant = game
        .scenario_force_battlefield(player, "Unlucky Cabbage Merchant", true)
        .unwrap();
    let source = game.state.permanents[merchant].as_ref().unwrap().object;
    let source_ref = game.permanent_object(merchant).unwrap();
    counter_replacement_drivers(&mut game, player);
    let returned = exiled_counter_entry(&mut game, player);
    let returned_ref = game.state.now(returned);
    // The real Merchant program is the consumer; this constructed link isolates
    // a source departure whose consequent entry asks a replacement question.
    game.state.exile_links.push(ExileLink {
        source: source_ref,
        exiled_object: returned_ref,
        returns: true,
    });
    let Ability::Triggered { effects, .. } = &game.state[source].abilities[1];
    let mut effects = effects.clone();
    effects.push(Effect::gain_life(3));
    let mut frame = EffectFrame::new(player, effects);
    frame.bind_source(Some(source), Some(source_ref));
    game.run_frame(frame);
    game.publish_action_space(game.suspended_decision_action_space().unwrap());
    checked_command(&mut game, Action::ChooseMode { player, mode: 0 });
    let library_before = game
        .state
        .zones
        .zone_objects(ZoneType::Library, player)
        .to_vec();
    checked_command(&mut game, Action::Decline { player });
    // The Merchant is a new object on the bottom of its library.
    let mut expected_library = vec![game.state.now(source)];
    expected_library.extend(library_before);
    assert!(matches!(
        game.state.suspended_decision.as_ref().unwrap().decision,
        Decision::OrderReplacement { .. }
    ));
    assert_eq!(
        game.state
            .zones
            .zone_objects(ZoneType::Library, player)
            .to_vec(),
        expected_library,
        "bottom placement has committed; the shuffle must wait for linked entry"
    );
    assert_eq!(
        game.state.zones.zone_of(game.state.now(returned)),
        Some(ZoneType::Exile)
    );
    assert_eq!(game.state.players[player.0].life, 20);
    assert!(
        !game
            .state
            .events
            .iter()
            .any(|event| matches!(event, GameEvent::LibrarySearched { .. })),
        "search completion waits for its whole child scope"
    );
    for mode in [0, 1] {
        let mut fork = game.clone();
        checked_command(&mut fork, Action::ChooseMode { player, mode });
        assert_eq!(
            fork.state.zones.zone_of(fork.state.now(returned)),
            Some(ZoneType::Battlefield)
        );
        assert_eq!(
            fork.state.zones.zone_of(fork.state.now(source)),
            Some(ZoneType::Library)
        );
        assert_eq!(fork.state.players[player.0].life, 23);
        assert_eq!(
            fork.state
                .events
                .iter()
                .filter(|event| matches!(event, GameEvent::LibrarySearched { .. }))
                .count(),
            1
        );
        assert_eq!(
            fork.state.zones.size(ZoneType::Library, player),
            expected_library.len()
        );
    }
}

#[test]
fn hidden_selection_public_entry_is_pinned_through_both_samplers() {
    use crate::{
        agent::observation::Observation,
        flow::decision::{Decision, EffectFrame},
        possible_worlds::{MaterializeMode, PossibleWorldSpace},
    };
    for zone in [ZoneType::Hand, ZoneType::Library] {
        let player = PlayerId(0);
        let viewer = PlayerId(1);
        let (mut root, _, _) = fixture(player);
        counter_replacement_drivers(&mut root, player);
        let entering = exiled_counter_entry(&mut root, player);
        let entering = root.move_object(entering, zone);
        root.run_frame(EffectFrame::new(
            player,
            vec![Effect::SelectCards {
                from: ObjectSelector::Zone(zone),
                predicate: CardPredicate::creature(),
                min: 0,
                max: 1.into(),
                effects: vec![Effect::return_to_battlefield(
                    ObjectSelector::Selected,
                    false,
                )],
                if_declined: vec![],
            }],
        ));
        root.publish_action_space(root.suspended_decision_action_space().unwrap());
        checked_command(
            &mut root,
            Action::SelectCard {
                player,
                object: entering,
            },
        );
        assert!(matches!(
            root.state.suspended_decision.as_ref().unwrap().decision,
            Decision::OrderReplacement { .. }
        ));
        assert!(
            Observation::for_player(&root, viewer)
                .opponent_cards
                .iter()
                .any(|object| object.id == root.state[entering].id.0 as i32 && object.zone == zone),
            "a precommit entrant is public during replacement choices"
        );
        let before = crate::search_state::witness(&root);
        for perspective in [player, viewer] {
            let bytes = Observation::for_player(&root, perspective).to_json();
            let expected = format!("{:x}", Sha256::digest(bytes.as_bytes()));
            assert_eq!(
                root.semantic_viewer_state_hash(perspective).unwrap(),
                expected
            );
            assert_eq!(
                root.semantic_observation(perspective)
                    .unwrap()
                    .identity
                    .viewer_state_hash,
                expected
            );
        }
        assert_eq!(crate::search_state::witness(&root), before);
        let space = PossibleWorldSpace::for_viewer(&root, viewer);
        let materialized = space
            .materialize_index(&root, 0, 31, MaterializeMode::RefreshOpponentCommitment)
            .unwrap();
        let mut sampled = root.clone();
        sampled.determinize(viewer, 11).unwrap();
        for mut branch in [sampled, materialized] {
            assert_eq!(
                branch.state.zones.zone_of(branch.state.now(entering)),
                Some(zone)
            );
            assert_eq!(
                branch
                    .state
                    .suspended_decision
                    .as_ref()
                    .unwrap()
                    .frame
                    .selected,
                root.state
                    .suspended_decision
                    .as_ref()
                    .unwrap()
                    .frame
                    .selected
            );
            checked_command(&mut branch, Action::ChooseMode { player, mode: 0 });
            assert_eq!(
                branch.state.zones.zone_of(branch.state.now(entering)),
                Some(ZoneType::Battlefield)
            );
        }
    }
}

#[test]
fn hidden_selection_foretold_measure_preserves_every_committed_slot() {
    use crate::{
        flow::{
            cast_permission::ExileCastGrant,
            decision::{Decision, EffectFrame},
        },
        possible_worlds::{MaterializeMode, PossibleWorldSpace},
        state::{card::StaticAbility, mana::ManaCost},
    };
    let player = PlayerId(0);
    let viewer = PlayerId(1);
    let (mut root, _, _) = fixture(player);
    let object_ids: Vec<_> = root
        .state
        .latest_objects()
        .filter(|(_, object_id)| object_id.owner == player)
        .map(|(index, _)| index)
        .collect();
    let bolts: Vec<_> = object_ids
        .iter()
        .copied()
        .filter(|object_id| root.state[*object_id].name == "Lightning Bolt")
        .take(3)
        .collect();
    let ogres: Vec<_> = object_ids
        .iter()
        .copied()
        .filter(|object_id| root.state[*object_id].name == "Gray Ogre")
        .take(2)
        .collect();
    for object_id in object_ids {
        root.move_object(object_id, ZoneType::Graveyard);
    }
    // Two foretold slots can consume either of two names. With two of the
    // three instants already selected in the library, only 14 of the 20
    // ordered fills remain compatible: 2 Ogre/Ogre and 12 Ogre/Bolt fills.
    for object_id in bolts.iter().chain(&ogres) {
        let current = root.state.now(*object_id);
        root.state[current]
            .statics
            .push(StaticAbility::Foretell(ManaCost::parse("{1}")));
        root.move_object(
            current,
            if bolts.contains(object_id) {
                ZoneType::Library
            } else {
                ZoneType::Exile
            },
        );
    }
    // The library and exile objects those cards are now.
    let current = |objects: &[ObjectId]| -> Vec<ObjectId> {
        objects
            .iter()
            .map(|object| root.state.now(*object))
            .collect()
    };
    let (bolts, ogres) = (current(&bolts), current(&ogres));
    for object_id in &ogres {
        let object = root.state.live(*object_id).unwrap();
        root.state
            .cast_permissions
            .exile_grants
            .push(ExileCastGrant {
                foretold: true,
                ..ExileCastGrant::new(object, player)
            });
    }
    // The constructed root starts with unknown library positions. The
    // visible setup moves above would otherwise disclose all three Bolts.
    assert_eq!(root.state.players[player.0].known_library.public.len(), 3);
    root.state.players[player.0].known_library.forget_order();
    root.run_frame(EffectFrame::new(
        player,
        vec![Effect::search_library(
            CardPredicate {
                card_type: Some(CardType::Instant),
                ..Default::default()
            },
            3.into(),
            vec![Effect::return_to_hand(ObjectSelector::Selected)],
        )],
    ));
    root.publish_action_space(root.suspended_decision_action_space().unwrap());
    for object_id in &bolts[..2] {
        checked_command(
            &mut root,
            Action::SelectCard {
                player,
                object: *object_id,
            },
        );
    }
    let space = PossibleWorldSpace::for_viewer(&root, viewer);
    assert_eq!(space.worlds().len(), 1);
    assert_eq!(
        space.total_weight(),
        14,
        "count compatible physical deals, not subset choices"
    );
    for seed in 0..8 {
        let exact = space
            .materialize_index(&root, 0, seed, MaterializeMode::RefreshOpponentCommitment)
            .unwrap();
        let mut sampled = root.clone();
        sampled.determinize(viewer, seed).unwrap();
        for mut branch in [exact, sampled] {
            let Decision::SelectCards { selected, .. } =
                &branch.state.suspended_decision.as_ref().unwrap().decision
            else {
                panic!("selection remains pending");
            };
            assert_eq!(selected.len(), 2);
            for object in selected {
                let object_id = *object;
                assert_eq!(branch.state[object_id].name, "Lightning Bolt");
                assert!(branch
                    .state
                    .zones
                    .contains(object_id, ZoneType::Library, player));
            }
            checked_command(&mut branch, Action::Decline { player });
            assert_eq!(branch.state.zones.size(ZoneType::Hand, player), 2);
            assert_eq!(branch.foretold_cards(player).count(), 2);
            assert!(branch
                .state
                .zones
                .zone_objects(ZoneType::Hand, player)
                .iter()
                .all(|object_id| branch.state[*object_id].name == "Lightning Bolt"));
        }
    }
}

#[test]
fn retained_creation_precedes_allocation_and_preserves_copy_entry_terms() {
    use crate::flow::decision::{EffectFrame, FrameCompletion};
    use crate::state::card::ReplacementEffect;
    let player = PlayerId(0);
    let (mut game, _, _) = fixture(player);
    let source = game
        .scenario_force_battlefield(player, "Gray Ogre", true)
        .unwrap();
    let object = game.state.permanents[source].as_ref().unwrap().object;
    game.state[object].replacement_effects = vec![ReplacementEffect::EntersWithPlusOneCounters {
        count: 1,
        condition: None,
    }];
    for multiplier in [2, 3] {
        let id = game
            .scenario_force_battlefield(player, "Mountain", true)
            .unwrap();
        let object = game.state.permanents[id].as_ref().unwrap().object;
        game.state[object].replacement_effects =
            vec![ReplacementEffect::MultiplyTokens { multiplier }];
    }
    counter_replacement_drivers(&mut game, player);
    let mut frame = EffectFrame::new(
        player,
        vec![
            Effect::CreateTokenCopy {
                of: ObjectSelector::This,
                exceptions: vec![Modification::SetBasePt {
                    power: Some(EffectValue::Fixed(4)),
                    toughness: Some(EffectValue::Fixed(5)),
                }],
            },
            Effect::gain_life(3),
        ],
    );
    frame.bind_source(Some(object), game.permanent_object(source));
    frame.completion = FrameCompletion::Continue;
    let cards_before = game.state.objects.len();
    game.run_frame(frame);
    game.publish_action_space(game.suspended_decision_action_space().unwrap());
    assert_eq!(
        game.state.objects.len(),
        cards_before,
        "no allocation before creation choice"
    );
    let measure = |game: &Game| {
        use crate::agent::observation::Observation;
        use crate::agent::observation_encoder::{encode, ObservationEncoderConfig};
        let obs = Observation::for_player(game, player);
        let config = ObservationEncoderConfig {
            max_facts: 4096,
            ..Default::default()
        };
        let encoded = encode(&obs, &config).unwrap();
        let facts = encoded
            .facts
            .as_chunks::<12>()
            .0
            .iter()
            .filter(|row| row[11] == 1.0)
            .count();
        let relations = encoded
            .relations
            .as_chunks::<49>()
            .0
            .iter()
            .filter(|row| row[31] == 1.0)
            .count();
        serde_json::json!({"facts":facts,"relations":relations,"default_overflow":facts > 512})
    };
    let mut occupancy = vec![measure(&game)];
    let root = crate::search_state::witness(&game);
    let mut replay = game.clone();
    for (branch_index, branch) in [&mut game, &mut replay].into_iter().enumerate() {
        checked_command(branch, Action::ChooseMode { player, mode: 0 });
        assert_eq!(branch.state.objects.len(), cards_before + 6);
        for _ in 0..6 {
            if branch_index == 0 {
                occupancy.push(measure(branch));
            }
            assert_eq!(branch.state.players[0].life, 20, "suffix waits for entries");
            assert!(branch
                .state
                .latest_objects()
                .skip(cards_before)
                .all(|(id, _)| branch.state.zones.zone_of(branch.state.now(id)).is_none()));
            checked_command(branch, Action::ChooseMode { player, mode: 0 });
        }
        let tokens: Vec<_> = branch
            .state
            .latest_objects()
            .skip(cards_before)
            .map(|(id, object)| {
                assert_eq!(object.kind(), ObjectKind::Token);
                branch.state.permanent_of(branch.state.now(id)).unwrap()
            })
            .collect();
        for token in tokens {
            let permanent = branch.state.permanents[token].as_ref().unwrap();
            assert_eq!(permanent.counter_count(CounterKind::PlusOnePlusOne), 4);
            assert_eq!(branch.characteristics(token).pt(), (8, 9));
        }
        assert_eq!(branch.state.players[0].life, 23);
    }
    assert_eq!(
        crate::search_state::witness(&game),
        crate::search_state::witness(&replay)
    );
    assert_ne!(crate::search_state::witness(&game), root);
    if let Ok(path) = std::env::var("TOKEN_CREATION_OCCUPANCY") {
        std::fs::write(path, serde_json::to_string_pretty(&occupancy).unwrap()).unwrap();
    }
}

#[test]
fn hidden_selection_library_inspections_preserve_known_prefixes_and_refresh_unknown_ones() {
    use crate::{
        flow::decision::EffectFrame,
        possible_worlds::{MaterializeMode, PossibleWorldSpace},
    };
    for effect in [
        Effect::Scry { count: 3 },
        Effect::Surveil { count: 3 },
        Effect::LookAndSelect {
            random_order: false,
            reveal: false,
            to_battlefield: false,
            look: 3.into(),
            min_select: 1,
            max_select: 1,
            predicate: CardPredicate::default(),
        },
    ] {
        let player = PlayerId(0);
        let viewer = PlayerId(1);
        let (mut root, _, _) = fixture(player);
        root.run_frame(EffectFrame::new(player, vec![effect]));
        root.publish_action_space(root.suspended_decision_action_space().unwrap());
        let before = root
            .state
            .suspended_decision
            .as_ref()
            .unwrap()
            .decision
            .library_inspection()
            .unwrap()
            .1
            .to_vec();
        let mut known = root.clone();
        known.determinize(player, 241).unwrap();
        assert_eq!(known.library_top_objects(player, 3), before);
        assert_eq!(
            known.state.suspended_decision,
            root.state.suspended_decision
        );
        let mut unknown = root.clone();
        unknown.determinize(viewer, 241).unwrap();
        let space = PossibleWorldSpace::for_viewer(&root, viewer);
        let materialized = space
            .materialize_index(&root, 0, 241, MaterializeMode::RefreshOpponentCommitment)
            .unwrap();
        for mut branch in [unknown, materialized] {
            let looked = branch
                .state
                .suspended_decision
                .as_ref()
                .unwrap()
                .decision
                .library_inspection()
                .unwrap()
                .1
                .to_vec();
            assert_eq!(looked.len(), 3);
            assert_eq!(looked, branch.library_top_objects(player, 3));
            assert_ne!(
                looked, before,
                "the unknown prefix is sampled, not secretly pinned"
            );
            assert!(looked.iter().all(|object| branch.state.zones.contains(
                *object,
                ZoneType::Library,
                player
            )));
            let observation = crate::agent::observation::Observation::for_player(&branch, viewer);
            assert!(!observation
                .agent_cards
                .iter()
                .chain(&observation.opponent_cards)
                .any(|object| object.zone == ZoneType::Library));
            let action = branch.action_space().unwrap().actions[0].clone();
            checked_command(&mut branch, action);
        }
    }
}

/// A pending free cast of the looked-at top card keeps that exact card for
/// its chooser, or the offered cast would name a card no longer on top.
#[test]
fn determinize_keeps_a_pending_library_top_cast_for_its_chooser() {
    use crate::flow::decision::EffectFrame;
    let player = PlayerId(0);
    let (mut root, _, _) = fixture(player);
    let castable = root
        .state
        .zones
        .zone_objects(ZoneType::Library, player)
        .iter()
        .copied()
        .find(|object| root.state[*object].types.is_creature())
        .unwrap();
    root.state
        .zones
        .reposition(ZoneType::Library, player, castable, usize::MAX);
    root.run_frame(EffectFrame::new(player, vec![Effect::CastLibraryTopFree]));
    root.publish_action_space(root.suspended_decision_action_space().unwrap());
    for seed in 0..16 {
        let mut branch = root.clone();
        branch.determinize(player, seed).unwrap();
        assert_eq!(branch.library_top_objects(player, 1), vec![castable]);
        assert_eq!(
            branch.state.suspended_decision,
            root.state.suspended_decision
        );
        let action = branch.action_space().unwrap().actions[0].clone();
        checked_command(&mut branch, action);
    }
}

/// Equal current choices can retain different work. These are constructed
/// positions, serialized for the Python model witness below.
#[test]
fn effect_execution_inputs_separate_results_iteration_and_completion() {
    use crate::agent::observation::Observation;
    use crate::agent::observation_encoder::{
        encode, ObservationEncodeError, ObservationEncoderConfig,
    };
    use crate::flow::decision::{
        Decision, EffectFrame, FrameCompletion, ResolutionTarget, SuspendedResolution,
        TargetIteration,
    };
    use crate::flow::proposed_event::{
        CounterProvenance, EventBinding, EventCause, ProposedEvent, TokenSource, ZoneEntry,
    };
    let player = PlayerId(0);
    let (base, first, second) = fixture(player);
    let mut evidence = Vec::new();
    for case in [
        "selected",
        "iteration",
        "completion",
        "counter_cause",
        "token_count",
        "departed_source",
    ] {
        let mut observations = Vec::new();
        let mut inputs = Vec::new();
        let mut private = Vec::new();
        for choice in 0..2 {
            let mut game = base.clone();
            let subject = if choice == 0 { first } else { second };
            let mut frame = EffectFrame::new(player, vec![Effect::gain_life(1)]);
            match case {
                "selected" => {
                    frame.selected.push(game.permanent_object(subject).unwrap());
                    frame.queue = vec![Effect::Tap {
                        what: ObjectSelector::Selected,
                    }]
                    .into();
                }
                "iteration" => {
                    frame.iteration = Some(TargetIteration {
                        remaining: vec![ResolutionTarget {
                            target: Target::Permanent(subject),
                            requirement: 0,
                            legal: true,
                        }]
                        .into(),
                        effects: vec![Effect::Tap {
                            what: ObjectSelector::Targets(TargetSpec::Creature),
                        }],
                    })
                }
                "completion" => {
                    frame.completion = FrameCompletion::Event {
                        event: if choice == 0 {
                            GameEvent::Scried { player }
                        } else {
                            GameEvent::Surveilled { player }
                        },
                        frame: Box::new(EffectFrame::new(player, vec![])),
                    }
                }
                "counter_cause" => {
                    frame.batch = Some(game.event_batch(
                        EventBinding::None,
                        [ProposedEvent::CounterChange {
                            provenance: CounterProvenance::new(if choice == 0 {
                                EventCause::Effect
                            } else {
                                EventCause::TurnAction
                            }),
                            target: game.permanent_object(first).unwrap(),
                            kind: CounterKind::PlusOnePlusOne,
                            delta: 1,
                            by: Some(player),
                        }],
                    ))
                }
                "token_count" => {
                    frame.batch = Some(game.event_batch(
                        EventBinding::Created { attacking: false },
                        [ProposedEvent::CreateTokens {
                            definition: Box::new(TokenSource::Registered("token.food".into())),
                            controller: player,
                            count: choice + 1,
                            entry: ZoneEntry::default(),
                        }],
                    ));
                }
                "departed_source" => {
                    let object = game.state.permanents[first].as_ref().unwrap().object;
                    let source = game.permanent_object(first).unwrap();
                    let object = game.move_object(object, ZoneType::Graveyard);
                    let mut lki = game.object_lki(source).unwrap().clone();
                    // Two possible histories with the same present board and
                    // offers, but a consequential captured power of 2 versus 3.
                    lki.power += choice;
                    game.record_object_lki(lki);
                    frame.bind_source(Some(object), Some(source));
                    frame.queue = vec![Effect::GainLife {
                        amount: EffectValue::SourcePower,
                    }]
                    .into();
                    frame.completion = FrameCompletion::Event {
                        event: GameEvent::Scried { player },
                        frame: Box::new(EffectFrame::new(player, vec![])),
                    };
                }
                _ => unreachable!(),
            }
            game.state.suspended_decision = Some(SuspendedResolution {
                frame,
                decision: Decision::Modal {
                    player,
                    modes: vec![vec![], vec![Effect::gain_life(2)]],
                },
            });
            game.publish_action_space(game.suspended_decision_action_space().unwrap());
            let obs = Observation::for_player(&game, player);
            let config = ObservationEncoderConfig::default();
            let input = encode(&obs, &config).unwrap();
            let used = input
                .facts
                .as_chunks::<12>()
                .0
                .iter()
                .filter(|row| row[11] == 1.0)
                .count();
            let relations = input
                .relations
                .as_chunks::<49>()
                .0
                .iter()
                .filter(|row| row[31] == 1.0)
                .count();
            assert!(matches!(
                encode(
                    &obs,
                    &ObservationEncoderConfig {
                        max_facts: 1,
                        ..config
                    }
                ),
                Err(ObservationEncodeError::Capacity { field: "facts", .. })
            ));
            let capture = game.capture_decision(player, 0).unwrap();
            evidence.push(serde_json::json!({"case":case,"choice":choice,"capture":capture,
                "facts": &input.facts[..used * 12], "relations": &input.relations[..relations * 49]}));
            assert_eq!(
                game.clone().capture_decision(player, 0).unwrap().digest,
                capture.digest
            );
            private.push(game.continuation_frames(PlayerId(1)));
            observations.push(obs.to_json());
            inputs.push(input);
        }
        assert_eq!(
            observations[0], observations[1],
            "same present offers: {case}"
        );
        assert_eq!(
            private[0], private[1],
            "opponent cannot inspect private execution: {case}"
        );
        assert!(
            inputs[0].facts != inputs[1].facts || inputs[0].relations != inputs[1].relations,
            "retained distinction: {case}"
        );
    }
    if let Ok(path) = std::env::var("EFFECT_EXECUTION_EVIDENCE") {
        std::fs::write(path, serde_json::to_string(&evidence).unwrap()).unwrap();
    }
}

/// A secret choice from inspected positions reveals the count moved, not the
/// position it came from. Knowledge hooks must consume that public scope,
/// rather than the true object's index in the library.
#[test]
fn hidden_knowledge_private_selection_does_not_disclose_departing_position() {
    use crate::{agent::observation::Observation, flow::decision::EffectFrame};

    for player in [PlayerId(0), PlayerId(1)] {
        let viewer = PlayerId(1 - player.0);
        let (mut root, _, _) = fixture(player);
        // Publicly place three cards on top before the private inspection.
        // A knowledge implementation must retain these exact positions.
        let mut visible = Vec::new();
        for name in ["Gray Ogre", "Lightning Bolt", "Mountain"] {
            let object = root
                .state
                .zones
                .zone_objects(ZoneType::Library, player)
                .iter()
                .copied()
                .find(|object| root.state[*object].name == name)
                .expect("fixture has each definition in the library");
            visible.push(root.move_object(object, ZoneType::Graveyard));
        }
        for object in visible {
            root.move_object(object, ZoneType::Library);
        }
        assert_eq!(root.state.players[player.0].known_library.public.len(), 3);
        root.run_frame(EffectFrame::new(
            player,
            vec![Effect::LookAndSelect {
                random_order: false,
                reveal: false,
                to_battlefield: false,
                look: 3.into(),
                min_select: 1,
                max_select: 2,
                predicate: CardPredicate::default(),
            }],
        ));
        root.publish_action_space(root.suspended_decision_action_space().unwrap());
        let inspected = root.library_top_objects(player, 3);
        let mut observations = Vec::new();
        for selected in &inspected {
            let mut branch = root.clone();
            checked_command(
                &mut branch,
                Action::SelectCard {
                    player,
                    object: *selected,
                },
            );
            assert!(branch.state.zones.contains(
                branch.state.zones.successor(*selected).unwrap(),
                ZoneType::Hand,
                player,
            ));
            assert!(branch.state.suspended_decision.is_some());
            observations.push(Observation::for_player(&branch, viewer).to_json());
        }
        assert_eq!(observations[0], observations[1]);
        assert_eq!(observations[1], observations[2]);
    }
}

#[test]
fn hidden_knowledge_public_positions_survive_both_samplers_and_private_refresh() {
    use crate::{agent::observation::Observation, possible_worlds::PossibleWorldSpace};
    for owner in [PlayerId(0), PlayerId(1)] {
        let viewer = PlayerId(1 - owner.0);
        let (mut game, _, _) = fixture(owner);
        let mut visible = Vec::new();
        for name in ["Lightning Bolt", "Mountain"] {
            let object = game
                .state
                .zones
                .zone_objects(ZoneType::Library, owner)
                .iter()
                .copied()
                .find(|object| game.state[*object].name == name)
                .unwrap();
            visible.push(game.move_object(object, ZoneType::Graveyard));
        }
        for object in visible {
            game.move_object(object, ZoneType::Library);
        }
        game.refresh_priority_actions(viewer);
        let original = Observation::for_player(&game, viewer);
        assert_eq!(original.opponent.known_library.len(), 2);
        let space = PossibleWorldSpace::for_viewer(&game, viewer);
        assert_eq!(space.projection().known_library.values().sum::<u32>(), 2);
        let before = game.state.deterministic_hash();
        for seed in 0..16 {
            let mut sampled = game.clone();
            sampled.determinize(viewer, seed).unwrap();
            assert_eq!(
                Observation::for_player(&sampled, viewer)
                    .opponent
                    .known_library,
                original.opponent.known_library
            );
            let knowledge = &sampled.state.players[owner.0].known_library;
            assert_eq!(knowledge.own, knowledge.public);
            sampled.determinize(owner, seed).unwrap();
            sampled.assert_library_knowledge();
        }
        for index in 0..space.worlds().len() {
            let mut branch = space
                .materialize_index(
                    &game,
                    index,
                    index as u64,
                    crate::possible_worlds::MaterializeMode::PreserveViewerRoot,
                )
                .unwrap();
            assert_eq!(
                Observation::for_player(&branch, viewer)
                    .opponent
                    .known_library,
                original.opponent.known_library
            );
            branch.determinize(owner, index as u64).unwrap();
            branch.assert_library_knowledge();
        }
        assert_eq!(game.state.deterministic_hash(), before);
        let definition = game.state[game.library_top_objects(owner, 1)[0]].definition_id;
        let count = game.state.players[owner.0]
            .known_hand
            .get(&definition)
            .copied()
            .unwrap_or(0);
        game.draw_cards(owner, 1);
        assert_eq!(
            game.state.players[owner.0].known_hand.get(&definition),
            Some(&(count + 1))
        );
        assert_eq!(
            Observation::for_player(&game, viewer)
                .opponent
                .known_library
                .len(),
            1
        );
        game.shuffle_library(owner);
        assert!(game.state.players[owner.0].known_library.is_empty());
        assert_eq!(
            game.state.players[owner.0].known_hand.get(&definition),
            Some(&(count + 1))
        );
    }
}

#[test]
fn hidden_knowledge_secret_hand_departure_weakens_each_minimum_without_identifying_it() {
    for owner in [PlayerId(0), PlayerId(1)] {
        let (mut game, _, _) = fixture(owner);
        game.remember_hand(owner);
        let before = game.state.players[owner.0].known_hand.clone();
        let hand = game
            .state
            .zones
            .zone_objects(ZoneType::Hand, owner)
            .to_vec();
        let mut views = Vec::new();
        for object in hand {
            let mut branch = game.clone();
            branch.move_object(object, ZoneType::Library);
            let expected: BTreeMap<_, _> = before
                .iter()
                .filter_map(|(definition, count)| (*count > 1).then_some((*definition, count - 1)))
                .collect();
            assert_eq!(branch.state.players[owner.0].known_hand, expected);
            views.push(
                crate::agent::observation::Observation::for_player(&branch, PlayerId(1 - owner.0))
                    .to_json(),
            );
        }
        assert!(views.windows(2).all(|pair| pair[0] == pair[1]));
    }
}

#[test]
fn hidden_knowledge_private_cap_keeps_complete_public_facts() {
    for owner in [PlayerId(0), PlayerId(1)] {
        let (mut game, _, _) = fixture(owner);
        let objects = game.library_top_objects(owner, 30);
        let public: BTreeMap<_, _> = objects
            .iter()
            .enumerate()
            .skip(10)
            .take(10)
            .map(|(position, object)| (position, game.state[*object].definition_id))
            .collect();
        let private: BTreeMap<_, _> = objects
            .iter()
            .enumerate()
            .skip(20)
            .take(10)
            .map(|(position, object)| (position, game.state[*object].definition_id))
            .collect();
        game.state.players[owner.0].known_library.public = public.clone();
        game.state.players[owner.0].known_library.own = private;
        game.remember_library_objects(owner, &objects[..1]);
        assert_eq!(game.state.players[owner.0].known_library.public, public);
        let facts = game.library_facts(owner, owner);
        assert_eq!(facts.len(), 20);
        assert!(public
            .iter()
            .all(|(position, definition)| facts.get(position) == Some(definition)));
        for seed in 0..16 {
            let mut branch = game.clone();
            branch.determinize(owner, seed).unwrap();
            branch.assert_library_knowledge();
            assert_eq!(branch.library_facts(owner, owner), facts);
        }
    }
}

#[test]
fn hidden_knowledge_public_prefix_survives_a_pending_inspection() {
    use crate::{
        agent::observation::Observation,
        flow::decision::EffectFrame,
        possible_worlds::{MaterializeMode, PossibleWorldSpace},
    };
    for (owner, reveal) in [PlayerId(0), PlayerId(1)]
        .into_iter()
        .flat_map(|owner| [false, true].map(|reveal| (owner, reveal)))
    {
        let viewer = PlayerId(1 - owner.0);
        let (mut game, _, _) = fixture(owner);
        let mut visible = Vec::new();
        for name in ["Gray Ogre", "Lightning Bolt", "Mountain"] {
            let object = game
                .state
                .zones
                .zone_objects(ZoneType::Library, owner)
                .iter()
                .copied()
                .find(|object| game.state[*object].name == name)
                .unwrap();
            visible.push(game.move_object(object, ZoneType::Graveyard));
        }
        for object in visible {
            game.move_object(object, ZoneType::Library);
        }
        game.run_frame(EffectFrame::new(
            owner,
            vec![if reveal {
                Effect::LookAndSelect {
                    look: 3.into(),
                    min_select: 1,
                    max_select: 2,
                    predicate: CardPredicate::default(),
                    reveal: true,
                    to_battlefield: false,
                    random_order: false,
                }
            } else {
                Effect::Surveil { count: 3 }
            }],
        ));
        game.publish_action_space(game.suspended_decision_action_space().unwrap());
        let known = Observation::for_player(&game, viewer)
            .opponent
            .known_library;
        assert_eq!(known.len(), 3);
        for seed in 0..16 {
            let mut branch = game.clone();
            branch.determinize(viewer, seed).unwrap();
            assert_eq!(
                Observation::for_player(&branch, viewer)
                    .opponent
                    .known_library,
                known
            );
        }
        let space = PossibleWorldSpace::for_viewer(&game, viewer);
        for index in 0..space.worlds().len() {
            let branch = space
                .materialize_index(
                    &game,
                    index,
                    index as u64,
                    MaterializeMode::RefreshOpponentCommitment,
                )
                .unwrap();
            assert_eq!(
                Observation::for_player(&branch, viewer)
                    .opponent
                    .known_library,
                known
            );
        }
    }
}

#[test]
fn hidden_knowledge_transient_departures_cannot_leave_false_hidden_zone_facts() {
    for owner in [PlayerId(0), PlayerId(1)] {
        for zone in [ZoneType::Library, ZoneType::Hand] {
            let (mut game, _, _) = fixture(owner);
            let inspected = game.library_top_objects(owner, 3);
            game.remember_library_objects(owner, &inspected);
            game.remember_hand(owner);
            let expected_hand = game.state.players[owner.0].known_hand.clone();
            let token = game.create_token("token.ally", owner, false).unwrap();
            let object = game.permanent_object(token).unwrap();
            game.move_object(object, zone);
            game.finish_state_based_batch();
            game.assert_library_knowledge();
            if zone == ZoneType::Library {
                assert_eq!(game.library_facts(owner, owner).len(), 3);
            }
            assert_eq!(game.state.players[owner.0].known_hand, expected_hand);
        }
    }
}

#[test]
fn hidden_knowledge_forced_single_bottom_is_remembered() {
    use crate::flow::decision::EffectFrame;
    for owner in [PlayerId(0), PlayerId(1)] {
        let (mut game, _, _) = fixture(owner);
        let top = game.library_top_objects(owner, 1)[0];
        let definition = game.state[top].definition_id;
        game.run_frame(EffectFrame::new(
            owner,
            vec![Effect::LookAndSelect {
                look: 1.into(),
                min_select: 0,
                max_select: 1,
                predicate: CardPredicate::default(),
                reveal: false,
                to_battlefield: false,
                random_order: false,
            }],
        ));
        game.publish_action_space(game.suspended_decision_action_space().unwrap());
        checked_command(&mut game, Action::Decline { player: owner });
        let bottom = game.state.zones.size(ZoneType::Library, owner) - 1;
        assert_eq!(
            game.library_facts(owner, owner).get(&bottom),
            Some(&definition)
        );
        assert!(game.library_facts(owner, PlayerId(1 - owner.0)).is_empty());
    }
}

#[test]
fn hidden_knowledge_search_reveals_do_not_displace_a_known_top() {
    use crate::flow::decision::EffectFrame;
    use crate::possible_worlds::{MaterializeMode, PossibleWorldSpace};
    use crate::state::ability::DistinctSearchEnd;
    for owner in [PlayerId(0), PlayerId(1)] {
        let viewer = PlayerId(1 - owner.0);
        let (mut game, _, _) = fixture(owner);
        let top = game.library_top_objects(owner, 1)[0];
        let definition = game.state[top].definition_id;
        game.state
            .known_library_mut(owner)
            .public
            .insert(0, definition);
        game.run_frame(EffectFrame::new(
            owner,
            vec![Effect::SearchDistinctNames {
                predicate: CardPredicate::default(),
                count: 2,
                end: DistinctSearchEnd::Hand,
            }],
        ));
        game.publish_action_space(game.suspended_decision_action_space().unwrap());
        let selected = game
            .state
            .zones
            .zone_objects(ZoneType::Library, owner)
            .iter()
            .copied()
            .find(|object| game.state[*object].definition_id != definition)
            .unwrap();
        checked_command(
            &mut game,
            Action::SelectCard {
                player: owner,
                object: selected,
            },
        );
        for seed in 0..8 {
            let mut branch = game.clone();
            branch.determinize(viewer, seed).unwrap();
            assert_eq!(
                branch.library_facts(owner, viewer).get(&0),
                Some(&definition)
            );
        }
        let space = PossibleWorldSpace::for_viewer(&game, viewer);
        space
            .materialize_index(&game, 0, 7, MaterializeMode::RefreshOpponentCommitment)
            .unwrap();
    }
}

/// Resolve through the printed ability, with the same stack placement and
/// counter operations that priority uses.
fn history_game(name: &str) -> (Game, PermanentId) {
    let deck = BTreeMap::from([(name.to_owned(), 12), ("Forest".to_owned(), 28)]);
    let mut game = Game::new(
        vec![
            PlayerConfig::new("p0", deck.clone()),
            PlayerConfig::new("p1", deck),
        ],
        608_2026,
        false,
    );
    game.keep_opening_hands();
    let permanent = game
        .scenario_force_battlefield(PlayerId(0), name, true)
        .unwrap();
    (game, permanent)
}

#[test]
fn cat_collector_two_gains_before_matching_create_one_cat() {
    let (mut game, _) = history_game("Cat Collector");
    let player = PlayerId(0);
    game.state.turn.active_player = player;
    assert_eq!(game.commit_life_change(player, 3), 3);
    assert_eq!(game.commit_life_change(player, 2), 2);
    let ordinals: Vec<_> = game
        .state
        .pending_events
        .iter()
        .filter_map(|event| match event {
            GameEvent::LifeChanged { gain_ordinal, .. } => Some(*gain_ordinal),
            _ => None,
        })
        .collect();
    assert_eq!(ordinals, [1, 2]);
    game.process_game_events();
    assert_eq!(game.state.turn.history[0].life_gains, 2);
    assert_eq!(game.state.pending_triggers.len(), 1);
    assert!(game.flush_triggers().is_none());
    game.resolve_top_of_stack();
    assert_eq!(
        game.iter_battlefield_permanents(player)
            .filter(|&id| game.state[game.permanent_object(id).unwrap()].name == "Cat")
            .count(),
        1
    );

    for (active, expected_triggers) in [(PlayerId(1), 0), (player, 1)] {
        while game.active_player() != active {
            game.state.turn.advance_step();
        }
        assert_eq!(game.state.turn.history[player.0].life_gains, 0);
        game.gain_life(player, 1);
        assert_eq!(game.state.pending_triggers.len(), expected_triggers);
    }
}

#[test]
fn venom_connoisseur_only_resolved_triggers_advance_their_source() {
    let (mut game, source) = history_game("Venom Connoisseur");
    let player = PlayerId(0);
    let original = game.permanent_object(source).unwrap();
    let trigger = |game: &mut Game| {
        game.create_token("token.ally", player, false);
        game.process_game_events();
        assert!(game.flush_triggers().is_none());
        game.state.stack_objects.last().unwrap().id()
    };
    let countered = trigger(&mut game);
    game.counter_ability(countered, None);
    assert!(game.state.turn.ability_resolutions_this_turn.is_empty());
    trigger(&mut game);
    game.resolve_top_of_stack();
    assert!(game
        .characteristics(source)
        .keywords
        .has(Keyword::Deathtouch));
    trigger(&mut game);
    let exiled = game.move_object(original, ZoneType::Exile);
    let returned = game.move_object(exiled, ZoneType::Battlefield);
    let countered = trigger(&mut game);
    game.counter_ability(countered, None);
    let counts = &game.state.turn.ability_resolutions_this_turn;
    assert_eq!(counts.len(), 1);
    assert_eq!(
        counts[&(original, game.state[returned].definition_id, 0)],
        1
    );
    trigger(&mut game);
    let observation = crate::agent::observation::Observation::for_player(&game, player);
    assert_eq!(
        observation
            .stack_objects
            .iter()
            .map(|row| row.ability_resolutions)
            .collect::<Vec<_>>(),
        [0, 1]
    );
    assert_ne!(
        observation.stack_objects[0].source_permanent_id,
        observation.stack_objects[1].source_permanent_id
    );
    game.resolve_top_of_stack();
    let ability = game.state[returned].definition_id;
    let count =
        |game: &Game, source| game.state.turn.ability_resolutions_this_turn[&(source, ability, 0)];
    assert_eq!((count(&game, original), count(&game, returned)), (1, 1));
    let ally = game
        .iter_battlefield_permanents(player)
        .find(|&id| game.state[game.permanent_object(id).unwrap()].name == "Ally")
        .unwrap();
    assert!(!game.characteristics(ally).keywords.has(Keyword::Deathtouch));
    game.resolve_top_of_stack();
    assert_eq!((count(&game, original), count(&game, returned)), (2, 1));
    assert!(game.characteristics(ally).keywords.has(Keyword::Deathtouch));
}

#[test]
fn separate_printed_abilities_on_one_source_each_resolve_for_the_first_time() {
    let (mut game, permanent) = history_game("Venom Connoisseur");
    let source = game.permanent_object(permanent).unwrap();
    let original = game.state[source].abilities[0].clone();
    game.state[source].abilities.push(original);
    game.create_token("token.ally", PlayerId(0), false);
    game.process_game_events();
    let choices = game
        .flush_triggers()
        .expect("two printed triggers to order");
    assert_eq!(choices.actions.len(), 2);
    game.publish_action_space(choices.clone());
    checked_command(&mut game, choices.actions[0].clone());
    while !game.state.stack_objects.is_empty() {
        game.resolve_top_of_stack();
    }
    let face = game.state[source].definition_id;
    let counts = &game.state.turn.ability_resolutions_this_turn;
    assert_eq!(counts.get(&(source, face, 0)), Some(&1));
    assert_eq!(counts.get(&(source, face, 1)), Some(&1));
}

/// CR 725.1a-f. A loss-of-toughness or lethal-damage SBA must happen
/// before cleanup can clear damage, even when a legend choice interrupts
/// that batch. Only the second of these identical death triggers survives.
#[test]
fn end_turn_resumes_its_state_based_check_before_cleanup() {
    use crate::agent::action::ActionSpaceKind;
    let player = PlayerId(0);
    let deck = BTreeMap::from([
        ("Curious Farm Animals".into(), 4),
        ("Fire Lord Azula".into(), 4),
        ("Lightning Bolt".into(), 4),
        ("Time Stop".into(), 4),
        ("Mountain".into(), 24),
    ]);
    let config = PlayerConfig::new("player", deck);
    let mut game = Game::new(vec![config.clone(), config], 725, false);
    for seat in [PlayerId(0), PlayerId(1)] {
        checked_command(
            &mut game,
            Action::ChooseMode {
                player: seat,
                mode: 0,
            },
        );
    }
    assert!(matches!(
        game.state.procedure,
        crate::flow::game::GameProcedure::Playing
    ));
    let step = game.state.turn.current_step_kind();
    let first = game
        .scenario_force_battlefield(player, "Curious Farm Animals", true)
        .unwrap();
    let second = game
        .scenario_force_battlefield(player, "Curious Farm Animals", true)
        .unwrap();
    let first_object = game.permanent_object(first).unwrap();
    game.move_object(first_object, ZoneType::Graveyard);
    assert_eq!(game.state.pending_triggers.len(), 1);
    game.state.permanents[second].as_mut().unwrap().damage = 1;

    let legends = [0, 1].map(|_| {
        game.scenario_force_battlefield(player, "Fire Lord Azula", true)
            .unwrap()
    });
    let spell = |game: &mut Game, name: &str| {
        let object = game
            .state
            .latest_objects()
            .find(|(_, o)| o.owner == player && o.name == name)
            .unwrap()
            .0;
        game.push_spell_to_stack(object, player, vec![], vec![], Default::default())
    };
    let bolt = spell(&mut game, "Lightning Bolt");
    let Some(Target::StackSpell(copy)) = game.copy_stack_object(Target::StackSpell(bolt), player)
    else {
        panic!("expected a spell copy");
    };
    spell(&mut game, "Time Stop");
    game.exile_top_cards_to_play(player, 1, crate::state::ability::PlayWindow::ThisTurn);
    let playable = game.state.cast_permissions.exile_grants[0].object;
    game.resolve_top_of_stack();
    game.publish_action_space(game.suspended_decision_action_space().unwrap());
    assert_eq!(
        game.action_space().unwrap().kind,
        ActionSpaceKind::LegendRule
    );
    assert_eq!(game.state.turn.current_step_kind(), step);
    assert!(game.state.stack_objects.is_empty());
    assert!(game.state.pending_triggers.is_empty());
    assert!(game
        .decision_context(player)
        .unwrap()
        .pending
        .iter()
        .any(|line| line.contains("turn is ending")));
    assert!(game.exile_grant_on(playable).is_some());
    let copy_in_exile = game.state.zones.successor(copy).unwrap();
    assert_eq!(
        game.state.zones.zone_of(copy_in_exile),
        Some(ZoneType::Exile)
    );

    let root = game.state.deterministic_hash();
    for keep in legends {
        let mut fork = game.clone();
        let keep_object = fork.permanent_object(keep).unwrap();
        checked_command(
            &mut fork,
            Action::SelectCard {
                player,
                object: keep_object,
            },
        );
        assert_eq!(fork.state.turn.current_step_kind(), StepKind::Cleanup);
        assert!(fork.state.permanents[keep].is_some());
        assert!(fork.state.permanents[second].is_none());
        assert_eq!(fork.state.zones.zone_of(copy_in_exile), None);
        assert!(
            fork.exile_grant_on(playable).is_none(),
            "this-turn play permission expires before cleanup priority"
        );
        assert_eq!(fork.state.stack_objects.len(), 1);
        assert_eq!(fork.action_space().unwrap().kind, ActionSpaceKind::Priority);
        fork.resolve_top_of_stack();
        assert_eq!(fork.state.players[player.0].life, 23);
        assert_eq!(game.state.deterministic_hash(), root);
    }
}

#[test]
fn snapshot_shuffle_keeps_inspections_and_random_bottom_placements_until_a_rules_shuffle() {
    use crate::flow::decision::EffectFrame;
    use rand::RngCore;
    for owner in [PlayerId(0), PlayerId(1)] {
        let (mut game, _, _) = fixture(owner);
        game.run_frame(EffectFrame::new(
            owner,
            vec![Effect::LookAndSelect {
                look: 12.into(),
                min_select: 0,
                max_select: 1,
                predicate: CardPredicate::default(),
                reveal: false,
                to_battlefield: false,
                random_order: true,
            }],
        ));
        game.publish_action_space(game.suspended_decision_action_space().unwrap());
        let inspected = game.library_top_objects(owner, 12);
        let space = game.current_action_space.clone();
        let mut sibling = game.clone();
        game.shuffle_unseen_libraries(101);
        sibling.shuffle_unseen_libraries(101);
        assert_eq!(game.library_top_objects(owner, 12), inspected);
        assert_eq!(game.current_action_space, space);
        checked_command(&mut game, Action::Decline { player: owner });
        checked_command(&mut sibling, Action::Decline { player: owner });
        let order = |game: &Game| {
            game.state
                .zones
                .zone_objects(ZoneType::Library, owner)
                .to_vec()
        };
        let before = order(&game);
        let mut sorted_bottom = before[..12].to_vec();
        sorted_bottom.sort();
        let mut sorted_inspected = inspected;
        sorted_inspected.sort();
        assert_eq!(sorted_bottom, sorted_inspected);
        assert!(game.state.players[owner.0].known_library.is_empty());
        let rng = game.state.rng.clone().next_u64();
        game.shuffle_unseen_libraries(77);
        sibling.shuffle_unseen_libraries(77);
        let after = order(&game);
        assert_eq!(after, order(&sibling));
        assert_eq!(&after[..12], &before[..12]);
        assert_ne!(after[12..], before[12..]);
        assert_eq!(game.state.rng.clone().next_u64(), rng);
        let drawn = game.state.card_of(*after.last().unwrap()).unwrap();
        game.draw_cards(owner, 1);
        assert!(game
            .state
            .zone_cards(ZoneType::Hand, owner)
            .any(|card| card == drawn));

        game.shuffle_library(owner);
        let fresh = order(&game);
        game.shuffle_unseen_libraries(77);
        let reshuffled = order(&game);
        assert!(fresh
            .iter()
            .zip(reshuffled)
            .any(|(old, new)| sorted_bottom.contains(old) && *old != new));
    }
}

#[test]
fn snapshot_shuffle_holds_the_searched_library_and_releases_private_facts_in_a_new_deal() {
    use crate::{flow::decision::EffectFrame, state::ability::DistinctSearchEnd};
    let owner = PlayerId(0);
    let viewer = PlayerId(1);
    let (mut game, _, _) = fixture(owner);
    let order = |game: &Game, seat| {
        game.state
            .zones
            .zone_objects(ZoneType::Library, seat)
            .to_vec()
    };
    game.remember_library_objects(owner, &order(&game, owner));
    game.run_frame(EffectFrame::new(
        viewer,
        vec![Effect::SearchDistinctNames {
            predicate: CardPredicate::default(),
            count: 2,
            end: DistinctSearchEnd::Hand,
        }],
    ));
    game.publish_action_space(game.suspended_decision_action_space().unwrap());
    let searched = order(&game, viewer);
    let frozen = order(&game, owner);
    game.shuffle_unseen_libraries(82);
    assert_eq!(order(&game, viewer), searched);
    assert_eq!(order(&game, owner), frozen);
    game.determinize(viewer, 35).unwrap();
    let before = order(&game, owner);
    game.shuffle_unseen_libraries(82);
    assert_ne!(order(&game, owner), before);
    let fixed = order(&game, viewer);
    game.shuffle_unseen_libraries(83);
    assert_eq!(order(&game, viewer), fixed);
}

#[test]
fn snapshot_shuffle_preserves_a_visible_top_after_the_look_permission_ends() {
    let owner = PlayerId(0);
    let mut game = Game::new(
        ["a", "b"]
            .map(|name| {
                PlayerConfig::new(
                    name,
                    BTreeMap::from([
                        ("Hakoda, Selfless Commander".into(), 1),
                        ("Mountain".into(), 32),
                    ]),
                )
            })
            .to_vec(),
        42,
        false,
    );
    let hakoda = game
        .scenario_force_battlefield(owner, "Hakoda, Selfless Commander", true)
        .unwrap();
    game.refresh_priority_actions(owner);
    let top = game.library_top_objects(owner, 1);
    game.move_object(game.permanent_object(hakoda).unwrap(), ZoneType::Graveyard);
    assert!(!game.looks_at_library_top(owner));
    // An unrelated bottom placement forgets bounded facts, including the top.
    let bottom = game.library_top_objects(owner, 2)[1];
    game.put_on_bottom_of_library(bottom);
    assert!(game.library_facts(owner, owner).is_empty());
    game.shuffle_unseen_libraries(7);
    assert_eq!(game.library_top_objects(owner, 1), top);
    assert_eq!(
        game.state.zones.zone_objects(ZoneType::Library, owner)[0],
        bottom
    );
}

#[test]
fn snapshot_constrained_determinize_retry_preserves_live_capture() {
    use crate::flow::decision::EffectFrame;
    let owner = PlayerId(0);
    let viewer = PlayerId(1);
    let deck = BTreeMap::from([("Mountain".into(), 24), ("Lightning Bolt".into(), 1)]);
    let mut game = Game::new(
        vec![
            PlayerConfig::new("a", deck.clone()),
            PlayerConfig::new("b", deck),
        ],
        42,
        true,
    );
    let has_bolt = |g: &Game| {
        g.state
            .zones
            .zone_objects(ZoneType::Hand, owner)
            .iter()
            .any(|object| g.state[*object].name == "Lightning Bolt")
    };
    if !has_bolt(&game) {
        game.scenario_force_card_in_hand(owner, "Lightning Bolt")
            .unwrap();
    }
    // The unconstrained first deal puts the only eligible card in the
    // library. A mandatory selection in the hand must therefore retry.
    let seed = (0..8)
        .find(|seed| {
            let mut ordinary = game.clone();
            ordinary.determinize(viewer, *seed).unwrap();
            !has_bolt(&ordinary)
        })
        .unwrap();
    game.run_frame(EffectFrame::new(
        owner,
        vec![Effect::SelectCards {
            from: ObjectSelector::Zone(ZoneType::Hand),
            predicate: CardPredicate {
                card_type: Some(CardType::Instant),
                ..Default::default()
            },
            min: 1,
            max: 1.into(),
            effects: Vec::new(),
            if_declined: Vec::new(),
        }],
    ));
    game.publish_action_space(game.suspended_decision_action_space().unwrap());
    game.decision_memory_enabled = true;
    game.decision_memory_include_own = true;
    game.determinize(viewer, seed).unwrap();
    assert!(has_bolt(&game));
    game.step(0).unwrap();
    game.step(0).unwrap();
    assert!(game.decision_memory.iter().all(|log| !log.is_empty()));
}

#[test]
fn snapshot_shuffle_preserves_a_declined_library_cast() {
    use crate::flow::decision::EffectFrame;
    let owner = PlayerId(1);
    let (mut game, _, _) = fixture(owner);
    let object = game
        .library_from_top(owner)
        .find(|id| game.state[*id].types.is_creature())
        .unwrap();
    game.state
        .zones
        .reposition(ZoneType::Library, owner, object, usize::MAX);
    game.run_frame(EffectFrame::new(owner, vec![Effect::CastLibraryTopFree]));
    game.publish_action_space(game.suspended_decision_action_space().unwrap());
    checked_command(&mut game, Action::Decline { player: owner });
    game.shuffle_unseen_libraries(12);
    assert_eq!(game.library_top_objects(owner, 1), [object]);
}
