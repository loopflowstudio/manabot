//! Bounded CR 608.2b witnesses. Synthetic programs isolate resolution semantics;
//! they do not assert these combinations are reachable in an admitted deck.

use crate::state::ability::LibraryEnd;
use std::collections::BTreeMap;

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
        game.push_spell_to_stack(card, controller, targets, indices, Default::default());
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
                effects: vec![effects.remove(0)],
                ..Default::default()
            }];
            StackObject::ActivatedAbility(ActivatedAbilityOnStack {
                source_transforms: 0,
                inherited_from: None,
                id,
                controller,
                source_definition_id,
                source_card,
                source_ref,
                ability_index: 0,
                targets,
                attached: None,
                x: 0,
                granted_by: None,
            })
        }
        Entry::Triggered => StackObject::TriggeredAbility(TriggeredAbilityOnStack {
            remembered: Vec::new(),
            source_transforms: 0,
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
            player: None,
            inline_effects: Some(effects),
            x: 0,
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
        amount: EffectValue::Fixed(1),
        what: ObjectSelector::Target(TargetSpec::Creature),
    }
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
            let card = game.state.permanents[victim].as_ref().unwrap().card;
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
                        Effect::return_to_hand(ObjectSelector::Target(spec.clone())),
                        pause(),
                        Effect::return_to_hand(ObjectSelector::Target(spec)),
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
                    Effect::gain_life(7),
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
                let bolt = spell.card;
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
                .as_chunks::<{ crate::agent::continuation::RELATION_DIM_V5 }>()
                .0
                .iter()
                .zip(legacy.as_chunks::<32>().0)
            {
                assert_eq!(&row[..32], old);
            }
            let target_rows = native
                .as_chunks::<{ crate::agent::continuation::RELATION_DIM_V5 }>()
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
    let source_card = game.state.permanents[source].as_ref().unwrap().card;
    let id = game.state.id_gen.next_id();
    game.push_to_stack(StackObject::TriggeredAbility(TriggeredAbilityOnStack {
        remembered: Vec::new(),
        source_transforms: 0,
        id,
        controller: PlayerId(0),
        source_card,
        source_ref: game.permanent_object_ref(source),
        source_lki: None,
        source_definition_id: game.state.cards[source_card].definition_id,
        ability_index: 0,
        targets: Vec::new(),
        target_req_indices: Vec::new(),
        context: Some(Target::StackAbility(ability)),
        player: None,
        inline_effects: Some(vec![Effect::CounterSpell {
            what: ObjectSelector::Target(TargetSpec::Spell),
        }]),
        x: 0,
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
    let counter = counter.card;
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
            let subject_card = game.state.permanents[subject].as_ref().unwrap().card;
            let captured = game.permanent_object_ref(subject).unwrap();
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
            assert_eq!(game.state.delayed_triggers[0].remembered, vec![captured]);
            let source = game.state.delayed_triggers[0].source_card;
            let mut different = game.clone();
            different.state.delayed_triggers[0].remembered =
                vec![game.permanent_object_ref(other).unwrap()];
            assert_ne!(
                game.state.deterministic_hash(),
                different.state.deterministic_hash()
            );
            // Source departure does not discard the ability or its controller.
            game.move_card(source, ZoneType::Graveyard);
            if leave_before_firing {
                game.move_card(subject_card, ZoneType::Exile);
                game.move_card(subject_card, ZoneType::Battlefield);
            }
            game.emit(GameEvent::StepStarted {
                step: StepKind::End,
            });
            game.process_game_events();
            assert!(game.state.delayed_triggers.is_empty(), "one shot");
            assert_eq!(game.state.pending_triggers[0].remembered, vec![captured]);
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
                    .remembered,
                vec![captured]
            );
            if !leave_before_firing {
                game.move_card(subject_card, ZoneType::Exile);
                game.move_card(subject_card, ZoneType::Battlefield);
            }
            let new_subject = game.state.card_to_permanent[subject_card].unwrap();
            assert_ne!(game.permanent_object_ref(new_subject).unwrap(), captured);
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
        let remembered = game.permanent_object_ref(subject).unwrap();
        stack(
            &mut game,
            Entry::Triggered,
            controller,
            vec![Effect::CreateDelayedTrigger {
                event: DelayedEvent::YouCastSpell(CardPredicate::default()),
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
        let source = game.state.delayed_triggers[0].source_card;
        let source_permanent = game.state.card_to_permanent[source].unwrap();
        game.change_control(source_permanent, PlayerId(1 - controller.0));
        game.move_card(source, ZoneType::Graveyard);
        stack(&mut game, Entry::Spell, controller, vec![], vec![], vec![]);
        let StackObject::Spell(spell) = game.state.stack_objects.last().unwrap() else {
            panic!("spell")
        };
        let cast = spell.card;
        game.emit(GameEvent::SpellCast {
            card: cast,
            targets: vec![],
        });
        game.process_game_events();
        assert!(game.state.delayed_triggers.is_empty());
        let pending = &game.state.pending_triggers[0];
        assert_eq!(pending.remembered, vec![remembered]);
        assert_eq!(pending.context, Some(Target::StackSpell(cast)));
        assert!(game.flush_triggers().is_none());
        game.resolve_top_of_stack();
        let suspended = game.state.suspended_decision.as_ref().unwrap();
        assert_eq!(suspended.frame.remembered, vec![remembered]);
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
    let card = game.state.permanents[subject].as_ref().unwrap().card;
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
    let captured = game.current_object_ref(card).unwrap();
    assert_eq!(game.state.delayed_triggers[0].remembered, vec![captured]);
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
                Some(game.state.cards[card].definition_id.0)
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
                            let card = current.state.permanents[subject].as_ref().unwrap().card;
                            current.move_card(card, ZoneType::Graveyard);
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
            let card = game.state.permanents[subject].as_ref().unwrap().card;
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
            game.move_card(card, destination);
            game.process_game_events();
            assert!(
                game.state.delayed_triggers.is_empty(),
                "retired even when repeating"
            );
            let expected = usize::from(destination == ZoneType::Graveyard);
            assert_eq!(game.state.pending_triggers.len(), expected);
            game.move_card(card, ZoneType::Battlefield);
            game.move_card(card, ZoneType::Graveyard);
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
        let looked = root.library_top(player, 4);
        let original_refs: Vec<_> = looked
            .iter()
            .map(|card| root.current_object_ref(*card))
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
                card: looked[0],
            },
        );
        assert_eq!(
            root.state.zones.zone_of(looked[0]),
            Some(ZoneType::Library),
            "selection is private until the partition commits"
        );
        checked_command(&mut root, Action::Decline { player });
        assert_eq!(
            root.state.zones.zone_of(looked[0]),
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
                .filter(|card| card.zone == ZoneType::Library)
                .map(|card| card.id)
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
            for (index, card) in order.into_iter().enumerate() {
                assert!(!branch
                    .action_space()
                    .unwrap()
                    .actions
                    .iter()
                    .any(|action| matches!(action, Action::Decline { .. })));
                checked_command(&mut branch, Action::SelectCard { player, card });
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
                        sampled.library_top(player, 3),
                        branch.library_top(player, 3)
                    );
                }
            }
            assert_eq!(branch.state.zones.zone_of(order[0]), Some(ZoneType::Hand));
            assert_eq!(branch.library_top(player, 2), order[1..]);
            for card in &order[1..] {
                let index = looked
                    .iter()
                    .position(|candidate| candidate == card)
                    .unwrap();
                assert_eq!(
                    branch.current_object_ref(*card),
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
    use crate::flow::decision::{Decision, EffectFrame, PaymentProgress, SuspendedResolution};
    use crate::state::ability::PaymentCost;
    use crate::state::mana::{Mana, ManaCost};

    let player = PlayerId(0);
    let (mut game, _, _) = fixture(player);
    let source = game
        .scenario_force_battlefield(player, "Gray Ogre", true)
        .unwrap();
    let card = game.state.permanents[source].as_ref().unwrap().card;
    let source_ref = game.permanent_object_ref(source).unwrap();
    let looked = game.library_top(player, 2);
    let caller = SuspendedResolution {
        frame: EffectFrame::new(player, vec![Effect::gain_life(3)]),
        decision: Decision::ManaChoice {
            player,
            choices: vec![Mana::parse("U"), Mana::parse("B")],
            resume: Some(Box::new(Decision::Payment(PaymentProgress::new(
                player,
                PaymentCost::Mana(ManaCost::parse("{1}")),
                vec![Effect::gain_life(2)],
                vec![],
            )))),
        },
    };
    let mut child = EffectFrame::new(
        player,
        vec![Effect::sacrifice_this(), Effect::Surveil { count: 2 }],
    );
    child.source = Some(card);
    child.source_ref = Some(source_ref);
    // A rules-level child choice isolates completion ownership. Entry
    // replacement suspension is a separate event-batch integration witness.
    game.run_before_decision(child, caller.clone());
    game.publish_action_space(game.suspended_decision_action_space().unwrap());
    assert_eq!(game.state.zones.zone_of(card), Some(ZoneType::Graveyard));
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
    for card in looked.iter().rev() {
        checked_command(
            &mut fork,
            Action::SelectCard {
                player,
                card: *card,
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
    assert_eq!(fork.state.events.iter().filter(|event| matches!(event, GameEvent::PermanentSacrificed { card: sacrificed, .. } if *sacrificed == card)).count(), 1);
    assert_eq!(game.state.deterministic_hash(), after_sacrifice);
}

#[test]
fn retained_inspection_survives_nested_choices_and_hidden_world_sampling() {
    use crate::agent::observation::Observation;
    use crate::flow::decision::{Decision, EffectFrame, FrameCompletion, SuspendedResolution};

    for player in [PlayerId(0), PlayerId(1)] {
        let opponent = PlayerId(1 - player.0);
        let (mut game, _, _) = fixture(player);
        let looked = game.library_top(player, 3);
        let caller = SuspendedResolution {
            frame: EffectFrame::new(player, vec![Effect::gain_life(2)]),
            decision: Decision::OrderLibrary {
                player,
                cards: looked.clone(),
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
                .filter(|card| card.zone == ZoneType::Library)
                .map(|card| card.id)
                .collect::<Vec<_>>()
        };
        let expected = looked
            .iter()
            .map(|card| game.state.cards[*card].id.0 as i32)
            .collect::<Vec<_>>();
        assert_eq!(visible_library(&game, player), expected);
        assert!(visible_library(&game, opponent).is_empty());
        let original = game.state.deterministic_hash();
        let mut sampled = game.clone();
        sampled.determinize(player, 241).unwrap();
        assert_eq!(sampled.library_top(player, 3), looked);
        assert_eq!(visible_library(&sampled, player), expected);
        checked_command(
            &mut sampled,
            Action::ChooseMode {
                player: opponent,
                mode: 0,
            },
        );
        assert_eq!(sampled.state.suspended_decision, Some(caller));
        for card in looked.iter().rev() {
            checked_command(
                &mut sampled,
                Action::SelectCard {
                    player,
                    card: *card,
                },
            );
        }
        assert_eq!(
            sampled.library_top(player, 3),
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
fn retained_search_permission_preserves_opponent_membership_until_caller_finishes() {
    use crate::agent::observation::Observation;
    use crate::flow::decision::{Decision, EffectFrame, SuspendedResolution};

    for player in [PlayerId(0), PlayerId(1)] {
        let opponent = PlayerId(1 - player.0);
        let (mut game, _, _) = fixture(player);
        let hand = game
            .state
            .zones
            .zone_cards(ZoneType::Hand, opponent)
            .to_vec();
        let library = game
            .state
            .zones
            .zone_cards(ZoneType::Library, opponent)
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
                .filter(|card| card.zone == ZoneType::Hand)
                .count(),
            hand.len()
        );
        assert_eq!(
            obs.opponent_cards
                .iter()
                .filter(|card| card.zone == ZoneType::Library)
                .count(),
            library.len()
        );
        game.determinize(player, 242).unwrap();
        assert_eq!(game.state.zones.zone_cards(ZoneType::Hand, opponent), &hand);
        assert_eq!(
            game.state.zones.zone_cards(ZoneType::Library, opponent),
            &library
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
            .all(|card| card.zone != ZoneType::Hand && card.zone != ZoneType::Library));
    }
}

#[test]
fn surveil_commits_the_partition_once_and_announces_only_after_kept_order() {
    for player in [PlayerId(0), PlayerId(1)] {
        let (mut game, _, _) = fixture(player);
        let cards = game.library_top(player, 4);
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
        for card in &cards[..2] {
            checked_command(
                &mut game,
                Action::SelectCard {
                    player,
                    card: *card,
                },
            );
            assert_eq!(game.library_top(player, 4), cards);
            assert_eq!(surveils(&game), 0);
            assert_eq!(game.state.players[player.0].life, 20);
        }
        checked_command(&mut game, Action::Decline { player });
        assert!(cards[..2]
            .iter()
            .all(|card| game.state.zones.zone_of(*card) == Some(ZoneType::Graveyard)));
        assert_eq!(surveils(&game), 0, "ordering is still part of surveil");
        let events_before = game.state.events.len();
        for card in cards[2..].iter().rev() {
            checked_command(
                &mut game,
                Action::SelectCard {
                    player,
                    card: *card,
                },
            );
        }
        assert_eq!(surveils(&game), 1);
        assert_eq!(game.state.players[player.0].life, 23);
        assert!(
            matches!(game.state.events[events_before], GameEvent::Surveilled { player: who } if who == player)
        );
        for card in &cards[..2] {
            assert_eq!(
                game.state
                    .events
                    .iter()
                    .filter(|event| matches!(event,
                GameEvent::CardMoved { card: moved, to: ZoneType::Graveyard, .. } if moved == card))
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
            for card in game.library_top(PlayerId(0), usize::MAX) {
                game.move_card(card, ZoneType::Graveyard);
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
fn selected_cards_bind_exact_objects_and_keep_the_callers_mill_result() {
    for player in [PlayerId(0), PlayerId(1)] {
        let (mut game, _, _) = fixture(player);
        let milled = game.library_top(player, 3);
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
                card: milled[0],
            },
        );
        assert_eq!(
            game.state.zones.zone_of(milled[0]),
            Some(ZoneType::Graveyard),
            "selection has not moved anything"
        );
        let root = game.state.deterministic_hash();
        for second in [None, Some(milled[1])] {
            let mut fork = game.clone();
            checked_command(
                &mut fork,
                second.map_or(Action::Decline { player }, |card| Action::SelectCard {
                    player,
                    card,
                }),
            );
            assert_eq!(fork.state.zones.zone_of(milled[0]), Some(ZoneType::Hand));
            assert_eq!(fork.state.players[player.0].life, 20, "parent suffix waits");
            checked_command(&mut fork, Action::ChooseMode { player, mode: 0 });
            assert_eq!(
                fork.state.zones.zone_of(milled[0]),
                Some(ZoneType::Hand),
                "selected refs never follow a moved incarnation"
            );
            let offered: Vec<_> = fork
                .action_space()
                .unwrap()
                .actions
                .iter()
                .filter_map(|action| match action {
                    Action::SelectCard { card, .. } => Some(*card),
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
                    card: milled[2],
                },
            );
            assert_eq!(fork.state.zones.zone_of(milled[2]), Some(ZoneType::Hand));
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
    use crate::state::ability::{Ability, TriggerCondition};

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
        StackObject::TriggeredAbility(ability) => ability.source_card,
        _ => unreachable!(),
    };
    game.state.cards[source].abilities = vec![Ability::Triggered {
        condition: TriggerCondition::YouScryOrSurveil,
        intervening_if: None,
        effects: vec![Effect::gain_life(1)],
    }];
    game.resolve_top_of_stack();
    game.publish_action_space(game.suspended_decision_action_space().unwrap());
    assert!(game.state.pending_triggers.is_empty());
    checked_command(&mut game, Action::Decline { player });
    assert_eq!(game.state.zones.zone_of(source), Some(ZoneType::Graveyard));
    let occurrences = game
        .state
        .pending_triggers
        .iter()
        .filter(|t| t.source_card == source)
        .count()
        + game
            .state
            .stack_objects
            .iter()
            .filter(|object| {
                matches!(object,
            StackObject::TriggeredAbility(ability) if ability.source_card == source)
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
        let card = game.state.permanents[id].as_ref().unwrap().card;
        game.state.cards[card].replacement_effects = vec![ReplacementEffect::ModifyCounters {
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
fn exiled_counter_entry(game: &mut Game, player: PlayerId) -> CardId {
    use crate::state::card::ReplacementEffect;

    let card = CardId(
        game.state
            .cards
            .iter()
            .enumerate()
            .find(|(index, card)| {
                card.owner == player
                    && card.name == "Gray Ogre"
                    && game.state.zones.zone_of(CardId(*index)) != Some(ZoneType::Battlefield)
            })
            .unwrap()
            .0,
    );
    game.move_card(card, ZoneType::Exile);
    game.state.cards[card].replacement_effects =
        vec![ReplacementEffect::EntersWithPlusOneCounters {
            count: 1,
            condition: None,
        }];
    card
}

#[test]
fn retained_batch_rechecks_new_entry_replacements_without_reapplying_identity() {
    use crate::flow::decision::{EffectFrame, FrameCompletion};
    let player = PlayerId(0);
    let (mut game, _, _) = fixture(player);
    counter_replacement_drivers(&mut game, player);
    let card = exiled_counter_entry(&mut game, player);
    let mut frame = EffectFrame::new(
        player,
        vec![Effect::return_to_battlefield(ObjectSelector::This, false)],
    );
    frame.source = Some(card);
    frame.source_ref = game.current_object_ref(card);
    frame.completion = FrameCompletion::Continue;
    game.run_frame(frame);
    game.publish_action_space(game.suspended_decision_action_space().unwrap());
    assert_eq!(game.state.zones.zone_of(card), Some(ZoneType::Exile));
    assert_eq!(
        game.action_space().unwrap().actions.len(),
        2,
        "own entry created counters, making both battlefield replacements applicable"
    );
    checked_command(&mut game, Action::ChooseMode { player, mode: 0 });
    let id = game.state.card_to_permanent[card].unwrap();
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
fn retained_batch_linked_entry_suspends_actual_mana_cost_completion() {
    use crate::flow::decision::{Decision, EffectFrame, PaymentProgress, SuspendedResolution};
    use crate::flow::trigger::ExileLink;
    use crate::state::ability::PaymentCost;
    use crate::state::mana::{Mana, ManaCost};
    let player = PlayerId(0);
    let (mut game, _, _) = fixture(player);
    counter_replacement_drivers(&mut game, player);
    let source = game
        .scenario_force_battlefield(player, "Gray Ogre", true)
        .unwrap();
    let source_card = game.state.permanents[source].as_ref().unwrap().card;
    let source_ref = game.permanent_object_ref(source).unwrap();
    game.state.cards[source_card].activated_abilities = vec![ActivatedAbilityDefinition {
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
    game.state.cards[source_card].abilities = vec![crate::state::ability::Ability::Triggered {
        condition: crate::state::ability::TriggerCondition::BecomesTapped {
            subject: crate::state::ability::TriggerSubject::This,
        },
        effects: vec![Effect::gain_life(9)],
        intervening_if: None,
    }];
    let returned = exiled_counter_entry(&mut game, player);
    game.state.exile_links.push(ExileLink {
        source: source_ref,
        exiled_card: returned,
        returns: true,
    });
    game.state.players[player.0].mana_pool = Mana::parse("R");
    game.state.suspended_decision = Some(SuspendedResolution {
        frame: EffectFrame::new(player, vec![Effect::gain_life(3)]),
        decision: Decision::Payment(PaymentProgress::new(
            player,
            PaymentCost::Mana(ManaCost::parse("{U}")),
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
        game.state.zones.zone_of(source_card),
        Some(ZoneType::Graveyard)
    );
    assert_eq!(game.state.zones.zone_of(returned), Some(ZoneType::Exile));
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
    assert_eq!(game.state.events.iter().filter(|event| matches!(event, GameEvent::PermanentSacrificed { card, .. } if *card == source_card)).count(), 1);
    assert_eq!(game.state.events.iter().filter(|event| matches!(event, GameEvent::CardMoved { card, to: ZoneType::Battlefield, .. } if *card == returned)).count(), 1);
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
        let card = root.state.permanents[id].as_ref().unwrap().card;
        root.state.cards[card].replacement_effects = vec![effect];
    }
    root.state.combat = Some(CombatState {
        attackers: vec![first, second],
        ..Default::default()
    });
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
        assert_eq!(game.state.events.iter().filter(|event| matches!(event, GameEvent::CombatDamageToPlayers { creatures } if creatures == &[first, second])).count(), 1);
    }
}

#[test]
fn retained_batch_state_based_departure_retains_linked_entry_and_lki() {
    use crate::flow::{sba::SbaProgress, trigger::ExileLink};
    let player = PlayerId(0);
    let (mut game, source, _) = fixture(player);
    counter_replacement_drivers(&mut game, player);
    let source_card = game.state.permanents[source].as_ref().unwrap().card;
    let source_ref = game.permanent_object_ref(source).unwrap();
    let returned = exiled_counter_entry(&mut game, player);
    game.state.exile_links.push(ExileLink {
        source: source_ref,
        exiled_card: returned,
        returns: true,
    });
    game.state.permanents[source].as_mut().unwrap().damage = 2;
    assert_eq!(game.perform_state_based_actions(), SbaProgress::Pending);
    game.publish_action_space(game.suspended_decision_action_space().unwrap());
    assert_eq!(
        game.state.zones.zone_of(source_card),
        Some(ZoneType::Graveyard)
    );
    assert_eq!(game.object_lki(source_ref).unwrap().power, 2);
    assert_eq!(game.state.zones.zone_of(returned), Some(ZoneType::Exile));
    checked_command(&mut game, Action::ChooseMode { player, mode: 1 });
    let id = game.state.card_to_permanent[returned].unwrap();
    assert_eq!(
        game.state.permanents[id]
            .as_ref()
            .unwrap()
            .counter_count(CounterKind::PlusOnePlusOne),
        3
    );
    assert_eq!(game.perform_state_based_actions(), SbaProgress::Stable);
    assert_eq!(game.state.events.iter().filter(|event| matches!(event, GameEvent::CardMoved { card, from: Some(ZoneType::Battlefield), .. } if *card == source_card)).count(), 1);
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
        let hand = root.state.zones.zone_cards(ZoneType::Hand, player).to_vec();
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
                card: hand[0],
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
                CardId::from(object.entity),
                ZoneType::Hand,
                player
            )));
            assert!(selected.iter().all(|object| candidates.contains(object)));
            let first = CardId::from(selected[0].entity);
            let second = branch
                .action_space()
                .unwrap()
                .actions
                .iter()
                .find_map(|action| match action {
                    Action::SelectCard { card, .. } => Some(*card),
                    _ => None,
                })
                .unwrap();
            for choice in [
                Action::Decline { player },
                Action::SelectCard {
                    player,
                    card: second,
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
                assert_eq!(fork.state.zones.zone_of(first), Some(ZoneType::Graveyard));
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
    let hand = root.state.zones.zone_cards(ZoneType::Hand, player).to_vec();
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
            card: hand[0],
        },
    );
    // Retain the choice-bearing child beneath another scope and saved decision.
    let mut caller = root.state.suspended_decision_mut().take().unwrap();
    caller.frame.remembered = caller.frame.selected.clone();
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
        assert_eq!(suspended.frame.remembered, caller.frame.selected);
        assert_eq!(caller.frame.remembered, caller.frame.selected);
        let selected = CardId::from(caller.frame.selected[0].entity);
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
            branch.state.zones.zone_of(selected),
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
        .cards
        .iter()
        .enumerate()
        .filter(|(_, card)| card.owner == player && card.name == "Lightning Bolt")
        .map(|(index, _)| CardId(index))
        .collect();
    for card in &bolts {
        root.move_card(*card, ZoneType::Library);
    }
    while root.state.zones.size(ZoneType::Hand, player) < 7 {
        let card = root
            .state
            .zones
            .zone_cards(ZoneType::Library, player)
            .iter()
            .copied()
            .find(|card| root.state.cards[*card].name == "Mountain")
            .unwrap();
        root.move_card(card, ZoneType::Hand);
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
            card: bolts[0],
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
        let selected = CardId::from(selected[0].entity);
        assert_eq!(
            branch.state.zones.zone_of(selected),
            Some(ZoneType::Library)
        );
        assert_eq!(branch.state.cards[selected].name, "Lightning Bolt");
        assert!(Observation::for_player(&branch, viewer)
            .opponent_cards
            .iter()
            .all(|card| card.zone != ZoneType::Library && card.zone != ZoneType::Hand));
        checked_command(&mut branch, Action::Decline { player });
        assert_eq!(branch.state.zones.zone_of(selected), Some(ZoneType::Hand));
        assert_eq!(
            branch.state.players[player.0]
                .known_hand
                .get(&branch.state.cards[selected].definition_id),
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
    let merchant = game
        .scenario_force_battlefield(player, "Unlucky Cabbage Merchant", true)
        .unwrap();
    let source = game.state.permanents[merchant].as_ref().unwrap().card;
    let source_ref = game.permanent_object_ref(merchant).unwrap();
    counter_replacement_drivers(&mut game, player);
    let returned = exiled_counter_entry(&mut game, player);
    // The real Merchant program is the consumer; this constructed link isolates
    // a source departure whose consequent entry asks a replacement question.
    game.state.exile_links.push(ExileLink {
        source: source_ref,
        exiled_card: returned,
        returns: true,
    });
    let Ability::Triggered { effects, .. } = &game.state.cards[source].abilities[1];
    let mut effects = effects.clone();
    effects.push(Effect::gain_life(3));
    let mut frame = EffectFrame::new(player, effects);
    frame.source = Some(source);
    frame.source_ref = Some(source_ref);
    game.run_frame(frame);
    game.publish_action_space(game.suspended_decision_action_space().unwrap());
    checked_command(&mut game, Action::ChooseMode { player, mode: 0 });
    let mut expected_library = vec![source];
    expected_library.extend(game.state.zones.zone_cards(ZoneType::Library, player));
    checked_command(&mut game, Action::Decline { player });
    assert!(matches!(
        game.state.suspended_decision.as_ref().unwrap().decision,
        Decision::OrderReplacement { .. }
    ));
    assert_eq!(
        game.state.zones.zone_cards(ZoneType::Library, player),
        &expected_library,
        "bottom placement has committed; the shuffle must wait for linked entry"
    );
    assert_eq!(game.state.zones.zone_of(returned), Some(ZoneType::Exile));
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
            fork.state.zones.zone_of(returned),
            Some(ZoneType::Battlefield)
        );
        assert_eq!(fork.state.zones.zone_of(source), Some(ZoneType::Library));
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
        root.move_card(entering, zone);
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
                card: entering,
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
                .any(|card| card.id == root.state.cards[entering].id.0 as i32 && card.zone == zone),
            "a precommit entrant is public during replacement choices"
        );
        let space = PossibleWorldSpace::for_viewer(&root, viewer);
        let materialized = space
            .materialize_index(&root, 0, 31, MaterializeMode::RefreshOpponentCommitment)
            .unwrap();
        let mut sampled = root.clone();
        sampled.determinize(viewer, 11).unwrap();
        for mut branch in [sampled, materialized] {
            assert_eq!(branch.state.zones.zone_of(entering), Some(zone));
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
                branch.state.zones.zone_of(entering),
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
    let cards: Vec<_> = root
        .state
        .cards
        .iter()
        .enumerate()
        .filter(|(_, card)| card.owner == player)
        .map(|(index, _)| CardId(index))
        .collect();
    let bolts: Vec<_> = cards
        .iter()
        .copied()
        .filter(|card| root.state.cards[*card].name == "Lightning Bolt")
        .take(3)
        .collect();
    let ogres: Vec<_> = cards
        .iter()
        .copied()
        .filter(|card| root.state.cards[*card].name == "Gray Ogre")
        .take(2)
        .collect();
    for card in cards {
        root.move_card(card, ZoneType::Graveyard);
    }
    // Two foretold slots can consume either of two names. With two of the
    // three instants already selected in the library, only 14 of the 20
    // ordered fills remain compatible: 2 Ogre/Ogre and 12 Ogre/Bolt fills.
    for card in bolts.iter().chain(&ogres) {
        root.state.cards[*card]
            .statics
            .push(StaticAbility::Foretell(ManaCost::parse("{1}")));
        root.move_card(
            *card,
            if bolts.contains(card) {
                ZoneType::Library
            } else {
                ZoneType::Exile
            },
        );
    }
    for card in &ogres {
        let object = root.current_object_ref(*card).unwrap();
        root.state
            .cast_permissions
            .exile_grants
            .push(ExileCastGrant {
                foretold: true,
                ..ExileCastGrant::new(object, player)
            });
    }
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
    for card in &bolts[..2] {
        checked_command(
            &mut root,
            Action::SelectCard {
                player,
                card: *card,
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
                let card = CardId::from(object.entity);
                assert_eq!(branch.state.cards[card].name, "Lightning Bolt");
                assert!(branch.state.zones.contains(card, ZoneType::Library, player));
            }
            checked_command(&mut branch, Action::Decline { player });
            assert_eq!(branch.state.zones.size(ZoneType::Hand, player), 2);
            assert_eq!(branch.foretold_cards(player).len(), 2);
            assert!(branch
                .state
                .zones
                .zone_cards(ZoneType::Hand, player)
                .iter()
                .all(|card| branch.state.cards[*card].name == "Lightning Bolt"));
        }
    }
}
