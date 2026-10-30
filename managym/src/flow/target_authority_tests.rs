//! Tiny rules witnesses for stack-target enumeration, acceptance and resolution.

use std::collections::BTreeMap;

use crate::{
    flow::{game::Game, trigger::PendingTrigger},
    state::{
        ability::{Effect, EffectValue, PlayerSelector, TargetRequirement, TargetSpec},
        card::Keywords,
        effect::{EffectDuration, Modification},
        game_object::{CardId, PermanentId, PlayerId, Target},
        player::PlayerConfig,
        predicate::CardPredicate,
        zone::ZoneType,
    },
};

/// Hexproof until end of turn, as a resolved spell would grant it.
fn grant_hexproof(game: &mut Game, id: PermanentId) {
    let affected = game.affected_permanent(id).unwrap();
    game.apply_continuous(
        None,
        PlayerId(0),
        [affected],
        Modification::AddKeywords(Keywords {
            hexproof: true,
            ..Default::default()
        }),
        EffectDuration::EndOfTurn,
    );
}

fn fixture() -> (Game, CardId, CardId, Target, Target) {
    let deck = BTreeMap::from([
        ("Gray Ogre".to_string(), 8),
        ("Lightning Bolt".to_string(), 8),
        ("Mountain".to_string(), 24),
    ]);
    let mut game = Game::new(
        vec![
            PlayerConfig::new("p0", deck.clone()),
            PlayerConfig::new("p1", deck),
        ],
        115_2026,
        false,
    );
    let ogre = game
        .scenario_force_battlefield(PlayerId(0), "Gray Ogre", true)
        .unwrap();
    let land = game
        .scenario_force_battlefield(PlayerId(0), "Mountain", true)
        .unwrap();
    let card_named = |name: &str| {
        CardId(
            game.state
                .cards
                .iter()
                .position(|card| card.owner == PlayerId(1) && card.name == name)
                .unwrap(),
        )
    };
    let bolt = card_named("Lightning Bolt");
    let ogre_spell = card_named("Gray Ogre");
    game.push_spell_to_stack(bolt, PlayerId(1), vec![], vec![], Default::default());
    game.push_spell_to_stack(ogre_spell, PlayerId(1), vec![], vec![], Default::default());
    (
        game,
        bolt,
        ogre_spell,
        Target::Permanent(ogre),
        Target::Permanent(land),
    )
}

fn trigger(source: CardId, effects: Vec<Effect>, order: u64) -> PendingTrigger {
    PendingTrigger {
        source_card: source,
        source_ref: None,
        source_lki: None,
        ability_index: 0,
        controller: PlayerId(0),
        targets: vec![],
        target_req_indices: vec![],
        requirement_index: 0,
        id: crate::flow::trigger::TriggerId(order),
        ordering_class: crate::flow::trigger::TriggerOrderingClass::Ordinary,
        context: None,
        player: None,
        chosen_modes: Vec::new(),
        inline_effects: Some(effects),
        x: 0,
    }
}

#[test]
fn stack_target_mana_value_boundaries_have_explicit_expected_choices() {
    let (game, bolt, ogre, permanent, land) = fixture();
    // CR 202.3: Bolt has mana value 1, Ogre 3, Mountain 0. CR 115.1 / 601.2c:
    // a restricted target must satisfy the restriction, including equality.
    // Preserve candidate order: newest spell first, then battlefield order.
    let cases = [
        (
            0,
            vec![
                Target::StackSpell(ogre),
                Target::StackSpell(bolt),
                permanent,
                land,
            ],
        ),
        (
            1,
            vec![
                Target::StackSpell(ogre),
                Target::StackSpell(bolt),
                permanent,
            ],
        ),
        (2, vec![Target::StackSpell(ogre), permanent]),
        (3, vec![Target::StackSpell(ogre), permanent]),
        (4, vec![]),
    ];
    for (minimum, expected) in cases {
        let spec = TargetSpec::SpellOrPermanent {
            min_mana_value: minimum,
        };
        assert_eq!(game.legal_targets_for_spec(&spec, PlayerId(0)), expected);
        for target in [
            Target::StackSpell(ogre),
            Target::StackSpell(bolt),
            permanent,
            land,
        ] {
            assert_eq!(
                game.target_is_legal(target, &spec, PlayerId(0)),
                expected.contains(&target)
            );
        }
    }
    assert_eq!(
        game.legal_targets_for_spec(&TargetSpec::Spell, PlayerId(0)),
        vec![Target::StackSpell(ogre), Target::StackSpell(bolt)]
    );
}

#[test]
fn player_and_graveyard_targets_preserve_scope_and_order() {
    let (mut game, bolt, ogre, permanent, _) = fixture();
    // CR 115.2: player choices do not admit cards, and graveyard choices do
    // not admit battlefield objects. Mixed choices keep players first.
    for (controller, opponent) in [(PlayerId(0), PlayerId(1)), (PlayerId(1), PlayerId(0))] {
        assert_eq!(
            game.legal_targets_for_spec(&TargetSpec::Player, controller),
            vec![Target::Player(PlayerId(0)), Target::Player(PlayerId(1))]
        );
        assert_eq!(
            game.legal_targets_for_spec(&TargetSpec::Opponent, controller),
            vec![Target::Player(opponent)]
        );
        assert_eq!(
            game.legal_targets_for_spec(&TargetSpec::CreatureOrPlayer, controller),
            vec![
                Target::Player(PlayerId(0)),
                Target::Player(PlayerId(1)),
                permanent
            ]
        );
    }
    game.move_card(bolt, ZoneType::Graveyard);
    game.move_card(ogre, ZoneType::Graveyard);
    let spec = TargetSpec::GraveyardCardYouOwn {
        predicate: CardPredicate::default(),
    };
    assert!(game.legal_targets_for_spec(&spec, PlayerId(0)).is_empty());
    assert_eq!(
        game.legal_targets_for_spec(&spec, PlayerId(1)),
        vec![
            Target::GraveyardCard(game.current_object_ref(bolt).unwrap()),
            Target::GraveyardCard(game.current_object_ref(ogre).unwrap()),
        ]
    );
    assert_eq!(
        game.legal_targets_for_spec(
            &TargetSpec::GraveyardCardYouOwn {
                predicate: CardPredicate::creature()
            },
            PlayerId(1)
        ),
        vec![Target::GraveyardCard(
            game.current_object_ref(ogre).unwrap()
        )]
    );
}

#[test]
fn restricted_trigger_rejects_low_value_spell_and_counters_legal_spell() {
    let (mut game, bolt, ogre, _, _) = fixture();
    game.state.pending_triggers.push(trigger(
        bolt,
        vec![Effect::CounterSpell {
            target: TargetSpec::SpellOrPermanent { min_mana_value: 2 },
        }],
        0,
    ));
    assert!(game.flush_triggers().is_some());
    assert!(game
        .choose_trigger_target(PlayerId(0), Target::StackSpell(bolt))
        .is_err());
    game.choose_trigger_target(PlayerId(0), Target::StackSpell(ogre))
        .unwrap();
    assert!(game.flush_triggers().is_none());
    game.resolve_top_of_stack();
    assert_eq!(game.state.zones.zone_of(ogre), Some(ZoneType::Graveyard));
    assert_eq!(game.state.zones.zone_of(bolt), Some(ZoneType::Stack));
}

#[test]
fn departed_only_target_prevents_even_untargeted_gain_life() {
    let (mut game, bolt, ogre, _, _) = fixture();
    game.state.pending_triggers.push(trigger(
        bolt,
        vec![
            Effect::CounterSpell {
                target: TargetSpec::SpellOrPermanent { min_mana_value: 2 },
            },
            Effect::GainLife { amount: 7 },
        ],
        0,
    ));
    assert!(game.flush_triggers().is_some());
    game.choose_trigger_target(PlayerId(0), Target::StackSpell(ogre))
        .unwrap();
    assert!(game.flush_triggers().is_none());
    game.move_card(ogre, ZoneType::Hand);
    // CR 608.2b: with every target illegal the whole ability fails to resolve,
    // including its untargeted instruction. This is more than skipping countering.
    game.resolve_top_of_stack();
    assert_eq!(game.state.players[0].life, 20);
    assert_eq!(game.state.zones.zone_of(ogre), Some(ZoneType::Hand));
}

#[test]
fn recast_spell_resolves_before_the_old_targeted_ability() {
    let (mut game, source, ogre, _, _) = fixture();
    let original = game.current_object_ref(ogre).unwrap();
    game.state.pending_triggers.push(trigger(
        source,
        vec![
            Effect::CounterSpell {
                target: TargetSpec::SpellOrPermanent { min_mana_value: 2 },
            },
            Effect::GainLife { amount: 7 },
        ],
        0,
    ));
    assert!(game.flush_triggers().is_some());
    game.choose_trigger_target(PlayerId(0), Target::StackSpell(ogre))
        .unwrap();
    assert!(game.flush_triggers().is_none());
    game.move_card(ogre, ZoneType::Hand);
    game.push_spell_to_stack(ogre, PlayerId(1), vec![], vec![], Default::default());
    assert_ne!(game.current_object_ref(ogre).unwrap(), original);

    // CR 405.2 / 608.2b: the recast object is above the older ability, so
    // resolves first. CardId-only targets remain an identity limitation;
    // this ordinary LIFO sequence does not expose a stale-target effect.
    game.resolve_top_of_stack();
    assert_eq!(game.state.zones.zone_of(ogre), Some(ZoneType::Battlefield));
    game.resolve_top_of_stack();
    assert_eq!(game.state.zones.zone_of(ogre), Some(ZoneType::Battlefield));
    assert_eq!(game.state.players[0].life, 20);
}

#[test]
fn trigger_target_distinctness_is_enforced_per_clause_at_submission() {
    let (mut game, source, _, first, second) = fixture();
    let spec = TargetSpec::Permanent {
        predicate: CardPredicate::default(),
    };
    game.state.pending_triggers.push(trigger(
        source,
        vec![
            Effect::TargetClause {
                requirement: TargetRequirement::up_to(2, spec.clone()),
                effects: vec![],
            },
            Effect::TargetClause {
                requirement: TargetRequirement::one(spec),
                effects: vec![],
            },
            Effect::GainLife { amount: 7 },
        ],
        0,
    ));
    assert!(game.flush_triggers().is_some());
    game.choose_trigger_target(PlayerId(0), first).unwrap();
    // CR 115.3: resubmitting the same target in one clause is illegal even
    // when bypassing the offered candidate list; rejection preserves progress.
    assert!(game.choose_trigger_target(PlayerId(0), first).is_err());
    game.choose_trigger_target(PlayerId(0), second).unwrap();
    // The next instance of "target" may choose the same object again.
    game.choose_trigger_target(PlayerId(0), first).unwrap();
    assert!(game.flush_triggers().is_none());
    game.resolve_top_of_stack();
    assert_eq!(game.state.players[0].life, 27);
}

#[test]
fn audit_same_controller_triggers_require_an_order_choice() {
    let (mut game, bolt, _, _, _) = fixture();
    game.state.pending_triggers = vec![
        trigger(bolt, vec![Effect::GainLife { amount: 2 }], 0),
        trigger(
            bolt,
            vec![Effect::DealDamageToPlayer {
                player: PlayerSelector::You,
                amount: EffectValue::Fixed(2),
            }],
            1,
        ),
    ];
    // At 2 life, the controller may stack damage below gain (survive at 2),
    // or gain below damage (lose at 0 before gain resolves): CR 603.3b,
    // 405.2, 117.5, 704.5a. An immediate fixed ordering cannot express both.
    game.state.players[0].life = 2;
    assert!(
        game.flush_triggers().is_some(),
        "must suspend for the controller's order choice"
    );
    assert!(
        game.state.stack_objects.len() == 2,
        "neither waiting trigger should be placed before choosing order"
    );
}

#[test]
fn simultaneous_trigger_orders_have_distinct_rules_outcomes() {
    for (gain_last, expected_life, expected_alive) in [(true, 2, true), (false, 0, false)] {
        let (mut game, source, _, _, _) = fixture();
        game.state.players[0].life = 2;
        let gain = vec![Effect::GainLife { amount: 2 }];
        let damage = vec![Effect::DealDamageToPlayer {
            player: PlayerSelector::You,
            amount: EffectValue::Fixed(2),
        }];
        game.state.pending_triggers = vec![trigger(source, gain, 0), trigger(source, damage, 1)];
        let choice = game.flush_triggers().unwrap();
        assert_eq!(choice.actions.len(), 2);
        game.choose_trigger(
            PlayerId(0),
            crate::flow::trigger::TriggerId(if gain_last { 1 } else { 0 }),
        )
        .unwrap();
        assert!(game.flush_triggers().is_none());
        game.resolve_top_of_stack();
        game.perform_state_based_actions();
        if game.state.players[0].alive {
            game.resolve_top_of_stack();
            game.perform_state_based_actions();
        }
        // CR 405.2: last placed resolves first. CR 704.5a: lose at zero
        // before the next ability resolves. Both placements must be choices.
        assert_eq!(game.state.players[0].life, expected_life);
        assert_eq!(game.state.players[0].alive, expected_alive);
    }
}

#[test]
fn audit_spell_or_permanent_must_respect_opponents_hexproof() {
    let (mut game, _, _, permanent, _) = fixture();
    let Target::Permanent(id) = permanent else {
        unreachable!()
    };
    grant_hexproof(&mut game, id);
    // CR 702.11b applies regardless of the wording used to target the object.
    // A creature-only restriction and a spell-or-permanent restriction must
    // both exclude this opposing Ogre; changing spec cannot bypass hexproof.
    assert!(!game.target_is_legal(permanent, &TargetSpec::Creature, PlayerId(1)));
    let spec = TargetSpec::SpellOrPermanent { min_mana_value: 3 };
    assert!(!game
        .legal_targets_for_spec(&spec, PlayerId(1))
        .contains(&permanent));
    assert!(!game.target_is_legal(permanent, &spec, PlayerId(1)));
}

#[test]
fn hexproof_target_wordings_preserve_controller_and_predicate_restrictions() {
    for owner in [PlayerId(0), PlayerId(1)] {
        let (mut game, _, _, _, _) = fixture();
        let id = game
            .scenario_force_battlefield(owner, "Gray Ogre", true)
            .unwrap();
        let target = Target::Permanent(id);
        grant_hexproof(&mut game, id);
        for controller in [PlayerId(0), PlayerId(1)] {
            let own = controller == owner;
            let cases = [
                (TargetSpec::Creature, own),
                (TargetSpec::CreatureOrPlayer, own),
                (TargetSpec::CreatureYouControl, own),
                (TargetSpec::CreatureOpponentControls, false),
                (
                    TargetSpec::Permanent {
                        predicate: CardPredicate::default(),
                    },
                    own,
                ),
                (
                    TargetSpec::PermanentYouControl {
                        predicate: CardPredicate::default(),
                    },
                    own,
                ),
                (
                    TargetSpec::PermanentOpponentControls {
                        predicate: CardPredicate::default(),
                    },
                    false,
                ),
                (TargetSpec::SpellOrPermanent { min_mana_value: 3 }, own),
                (TargetSpec::SpellOrPermanent { min_mana_value: 4 }, false),
                (TargetSpec::LandYouControl, false),
                (
                    TargetSpec::Permanent {
                        predicate: CardPredicate::subtype("Elf"),
                    },
                    false,
                ),
            ];
            for (spec, expected) in cases {
                assert_eq!(
                    game.target_is_legal(target, &spec, controller),
                    expected,
                    "{owner:?} {controller:?} {spec:?}"
                );
                assert_eq!(
                    game.legal_targets_for_spec(&spec, controller)
                        .contains(&target),
                    expected
                );
            }
        }
        let land = game
            .scenario_force_battlefield(owner, "Mountain", true)
            .unwrap();
        grant_hexproof(&mut game, land);
        assert!(game.target_is_legal(Target::Permanent(land), &TargetSpec::LandYouControl, owner));
        assert!(!game.target_is_legal(
            Target::Permanent(land),
            &TargetSpec::SpellOrPermanent { min_mana_value: 0 },
            PlayerId(1 - owner.0)
        ));
    }
}

#[test]
fn response_hexproof_rechecks_selected_targets_and_all_invalid_stops_ability() {
    for controller in [PlayerId(0), PlayerId(1)] {
        for mixed in [false, true] {
            let (mut game, source, _, _, _) = fixture();
            let opponent = PlayerId(1 - controller.0);
            let first = game
                .scenario_force_battlefield(opponent, "Gray Ogre", true)
                .unwrap();
            let second = game
                .scenario_force_battlefield(opponent, "Gray Ogre", true)
                .unwrap();
            let spec = TargetSpec::SpellOrPermanent { min_mana_value: 3 };
            let mut pending = trigger(
                source,
                vec![
                    Effect::TargetClause {
                        requirement: TargetRequirement::one(spec.clone()),
                        effects: vec![Effect::ReturnToHand {
                            target: spec.clone(),
                        }],
                    },
                    Effect::TargetClause {
                        requirement: TargetRequirement::one(spec.clone()),
                        effects: vec![Effect::ReturnToHand { target: spec }],
                    },
                    Effect::GainLife { amount: 7 },
                ],
                0,
            );
            pending.controller = controller;
            game.state.pending_triggers.push(pending);
            assert!(game.flush_triggers().is_some());
            game.choose_trigger_target(controller, Target::Permanent(first))
                .unwrap();
            game.choose_trigger_target(controller, Target::Permanent(second))
                .unwrap();
            assert!(game.flush_triggers().is_none());
            let second_card = game.state.permanents[second].as_ref().unwrap().card;
            // CR 608.2b: gaining hexproof after selection invalidates an opposing
            // target; remaining legal targets and untargeted instructions survive.
            grant_hexproof(&mut game, first);
            if !mixed {
                grant_hexproof(&mut game, second);
            }
            game.resolve_top_of_stack();
            assert!(game.state.permanents[first].is_some());
            assert_eq!(
                game.state.zones.zone_of(second_card),
                Some(if mixed {
                    ZoneType::Hand
                } else {
                    ZoneType::Battlefield
                })
            );
            assert_eq!(
                game.state.players[controller.0].life,
                if mixed { 27 } else { 20 }
            );
        }
    }
}

#[test]
fn spell_or_permanent_target_does_not_follow_a_reentered_card() {
    for controller in [PlayerId(0), PlayerId(1)] {
        let (mut game, source, _, _, _) = fixture();
        let owner = PlayerId(1 - controller.0);
        let id = game
            .scenario_force_battlefield(owner, "Gray Ogre", true)
            .unwrap();
        let card = game.state.permanents[id].as_ref().unwrap().card;
        let before = game.current_object_ref(card).unwrap();
        let spec = TargetSpec::SpellOrPermanent { min_mana_value: 3 };
        let mut pending = trigger(
            source,
            vec![
                Effect::ReturnToHand {
                    target: spec.clone(),
                },
                Effect::GainLife { amount: 7 },
            ],
            0,
        );
        pending.controller = controller;
        game.state.pending_triggers.push(pending);
        assert!(game.flush_triggers().is_some());
        game.choose_trigger_target(controller, Target::Permanent(id))
            .unwrap();
        assert!(game.flush_triggers().is_none());
        game.move_card(card, ZoneType::Hand);
        game.move_card(card, ZoneType::Battlefield);
        let replacement = game.state.card_to_permanent[card].unwrap();
        assert_ne!(game.current_object_ref(card).unwrap(), before);
        assert_ne!(replacement, id);
        assert!(!game.target_is_legal(Target::Permanent(id), &spec, controller));
        assert!(game.target_is_legal(Target::Permanent(replacement), &spec, controller));
        game.resolve_top_of_stack();
        assert_eq!(game.state.zones.zone_of(card), Some(ZoneType::Battlefield));
        assert_eq!(game.state.players[controller.0].life, 20);
    }
}

#[test]
fn trigger_order_all_three_occurrence_permutations_are_legal() {
    use crate::flow::trigger::TriggerId;
    for order in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let (mut game, source, ..) = fixture();
        game.state.pending_triggers = (0..3)
            .map(|id| {
                trigger(
                    source,
                    vec![Effect::GainLife {
                        amount: id as i32 + 1,
                    }],
                    id,
                )
            })
            .collect();
        for (position, id) in order[..2].iter().enumerate() {
            let space = game.flush_triggers().unwrap();
            assert_eq!(
                space.kind,
                crate::agent::action::ActionSpaceKind::OrderTriggers
            );
            assert_eq!(space.actions.len(), 3 - position);
            game.choose_trigger(PlayerId(0), TriggerId(*id)).unwrap();
        }
        assert!(game.flush_triggers().is_none());
        for id in order.iter().rev() {
            let before = game.state.players[0].life;
            game.resolve_top_of_stack();
            assert_eq!(game.state.players[0].life, before + *id as i32 + 1);
        }
    }
}

#[test]
fn trigger_order_two_passes_precede_controller_and_defer_new_arrivals() {
    use crate::flow::trigger::{TriggerId, TriggerOrderingClass};
    for active in [PlayerId(0), PlayerId(1)] {
        let (mut game, source, ..) = fixture();
        let ordinary = trigger(source, vec![Effect::GainLife { amount: 1 }], 0);
        let mut batch = Vec::new();
        for group in 0..4 {
            for id in 0..2 {
                let mut t = ordinary.clone();
                t.id = TriggerId(group * 2 + id);
                t.controller = if group % 2 == 0 {
                    active
                } else {
                    PlayerId(1 - active.0)
                };
                t.ordering_class = if group < 2 {
                    TriggerOrderingClass::Ordinary
                } else {
                    TriggerOrderingClass::AbilityTriggered
                };
                batch.push(t);
            }
        }
        game.state.trigger_placement = Some(crate::flow::trigger::TriggerPlacement {
            remaining: batch,
            active,
            group: 0,
            placing: None,
        });
        game.state
            .pending_triggers
            .push(trigger(source, vec![Effect::GainLife { amount: 9 }], 99));
        for group in 0..4 {
            let space = game.flush_triggers().unwrap();
            let player = if group % 2 == 0 {
                active
            } else {
                PlayerId(1 - active.0)
            };
            assert_eq!(space.player, Some(player));
            assert_eq!(space.actions.len(), 2);
            let hash = game.state.deterministic_hash();
            assert!(game
                .choose_trigger(PlayerId(1 - player.0), TriggerId(group * 2))
                .is_err());
            assert!(game.choose_trigger(player, TriggerId(99)).is_err());
            assert_eq!(game.state.deterministic_hash(), hash);
            game.choose_trigger(player, TriggerId(group * 2 + 1))
                .unwrap();
        }
        assert!(game.flush_triggers().is_none());
        assert_eq!(game.state.stack_objects.len(), 10);
        assert_eq!(game.state.pending_triggers.len(), 1);
    }
}

#[test]
fn trigger_order_large_menu_is_complete_and_empty_singleton_are_forced() {
    let (mut game, source, ..) = fixture();
    assert!(game.flush_triggers().is_none());
    game.state
        .pending_triggers
        .push(trigger(source, vec![Effect::GainLife { amount: 1 }], 0));
    assert!(game.flush_triggers().is_none());
    game.state.pending_triggers = (1..66)
        .map(|id| trigger(source, vec![Effect::GainLife { amount: 1 }], id))
        .collect();
    let space = game.flush_triggers().unwrap();
    assert_eq!(space.actions.len(), 65);
    game.current_action_space = Some(space);
    let observation = crate::agent::observation::Observation::new(&game, std::iter::empty());
    assert_eq!(observation.action_space.actions.len(), 65);
    assert!(crate::agent::observation_encoder::encode(&observation, &Default::default()).is_err());
    let larger = crate::agent::observation_encoder::ObservationEncoderConfig {
        max_actions: 65,
        ..Default::default()
    };
    assert!(matches!(
        crate::agent::observation_encoder::encode(&observation, &larger),
        Err(
            crate::agent::observation_encoder::ObservationEncodeError::Capacity {
                field: "continuation relations",
                ..
            }
        )
    ));
    assert!(
        crate::agent::observation::Observation::for_player(&game, PlayerId(1))
            .action_space
            .actions
            .is_empty()
    );
}

#[test]
fn trigger_order_same_source_joins_preserve_ability_and_causal_context() {
    use crate::agent::{observation::Observation, observation_encoder};
    let (mut game, source, other, ..) = fixture();
    let mut first = trigger(source, vec![Effect::GainLife { amount: 1 }], 900);
    first.source_ref = game.current_object_ref(source);
    first.context = Some(Target::StackSpell(source));
    let mut second = first.clone();
    second.id = crate::flow::trigger::TriggerId(901);
    second.ability_index = 1;
    let mut third = first.clone();
    third.id = crate::flow::trigger::TriggerId(902);
    third.context = Some(Target::StackSpell(other));
    game.state.pending_triggers = vec![first, second, third];
    game.current_action_space = game.flush_triggers();
    let observation = Observation::for_player(&game, PlayerId(0));
    let frames: Vec<_> = observation
        .action_space
        .actions
        .iter()
        .map(|action| &observation.continuation[action.continuation_index.unwrap()])
        .collect();
    assert_eq!(frames[0].source, frames[1].source);
    assert_eq!(frames[0].source, frames[2].source);
    assert_ne!(frames[0].ability, frames[1].ability);
    assert_eq!(frames[0].ability, frames[2].ability);
    assert_ne!(frames[0].context, frames[2].context);
    let encoded = observation_encoder::encode(&observation, &Default::default()).unwrap();
    // Occurrence counters bind Commands but are not policy features.
    for trigger in &mut game.state.trigger_placement.as_mut().unwrap().remaining {
        trigger.id.0 += 10_000;
    }
    game.current_action_space = game.flush_triggers();
    let renumbered = Observation::for_player(&game, PlayerId(0));
    assert_eq!(
        encoded,
        observation_encoder::encode(&renumbered, &Default::default()).unwrap()
    );
    let other_view = Observation::for_player(&game, PlayerId(1));
    assert!(other_view.continuation.iter().all(|frame| frame.kind <= 3));
}

#[test]
fn trigger_order_target_resume_and_nested_rollback_preserve_round() {
    use crate::agent::action::{Action, ActionSpaceKind};
    use crate::search_state::{witness, BenchCommand, BranchDriver, ClonePlusUndoDriver};
    let (mut game, source, ..) = fixture();
    game.state.pending_triggers = vec![
        trigger(
            source,
            vec![Effect::TargetClause {
                requirement: TargetRequirement::one(TargetSpec::Player),
                effects: vec![Effect::GainLife { amount: 1 }],
            }],
            0,
        ),
        trigger(source, vec![Effect::GainLife { amount: 2 }], 1),
        trigger(source, vec![Effect::GainLife { amount: 3 }], 2),
    ];
    game.state.pending_triggers[0].source_ref = game.current_object_ref(source);
    game.current_action_space = game.flush_triggers();
    let root = witness(&game);
    let driver = ClonePlusUndoDriver::default();
    let mut branch = driver.fork_exact(&game);
    let outer = driver.mark(&mut branch);
    driver
        .apply(
            &mut branch,
            BenchCommand {
                action_index: 0,
                expected_action_hash: None,
                expected_state_hash: None,
            },
        )
        .unwrap();
    assert_eq!(
        branch.action_space().unwrap().kind,
        ActionSpaceKind::ChooseTarget
    );
    assert_eq!(branch.state.stack_objects.len(), 2);
    let context = branch.decision_context(PlayerId(0)).unwrap();
    assert_eq!(context.stage, "trigger_targets");
    assert_eq!(context.source.0.unwrap().label, "Lightning Bolt");
    assert!(context
        .instruction
        .contains("Lightning Bolt — delayed ability:"));
    assert!(context.instruction.contains("You gain 1 life"));
    assert!(!branch
        .decision_context(PlayerId(1))
        .unwrap()
        .instruction
        .contains("Lightning Bolt"));
    let targeting = driver.witness(&branch);
    let inner = driver.mark(&mut branch);
    driver
        .apply(
            &mut branch,
            BenchCommand {
                action_index: 0,
                expected_action_hash: None,
                expected_state_hash: None,
            },
        )
        .unwrap();
    assert_eq!(
        branch.action_space().unwrap().kind,
        ActionSpaceKind::OrderTriggers
    );
    assert_eq!(branch.state.stack_objects.len(), 3);
    assert!(branch
        .action_space()
        .unwrap()
        .actions
        .iter()
        .all(|a| matches!(a, Action::ChooseTrigger { .. })));
    driver.rollback(&mut branch, inner);
    assert_eq!(driver.witness(&branch), targeting);
    driver.rollback(&mut branch, outer);
    assert_eq!(driver.witness(&branch), root);
    assert_eq!(witness(&game), root);
}

#[test]
fn trigger_order_descriptions_use_exact_ability_and_delayed_payload() {
    use crate::agent::observation::Observation;
    use crate::state::ability::{Ability, TriggerCondition, TriggerSubject};
    let (mut game, source, ..) = fixture();
    game.state.cards[source].abilities = vec![
        Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::Dies {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::DrawCards {
                count: EffectValue::Fixed(2),
            }],
        },
        Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::GainLife { amount: 3 }],
        },
    ];
    let mut dies = trigger(source, vec![], 0);
    dies.inline_effects = None;
    let mut enters = dies.clone();
    enters.id = crate::flow::trigger::TriggerId(1);
    enters.ability_index = 1;
    let mut delayed = trigger(source, vec![Effect::ReturnSourceToBattlefieldTapped], 2);
    delayed.ability_index = usize::MAX; // Inline effects do not reference a card slot.
    game.state.pending_triggers = vec![dies, enters, delayed];
    game.current_action_space = game.flush_triggers();
    let observation = Observation::for_player(&game, PlayerId(0));
    let labels: Vec<_> = observation
        .action_space
        .actions
        .iter()
        .map(|a| a.description.as_deref().unwrap())
        .collect();
    assert_eq!(labels, vec![
        "Put on stack next — Lightning Bolt — When this dies: Draw 2 cards",
        "Put on stack next — Lightning Bolt — When this enters the battlefield: You gain 3 life",
        "Put on stack next — Lightning Bolt — delayed ability: Return this card to the battlefield tapped",
    ]);
    assert!(labels
        .iter()
        .all(|label| !label.contains(&game.state.cards[source].text_box)));
    for id in 0..2 {
        game.choose_trigger(PlayerId(0), crate::flow::trigger::TriggerId(id))
            .unwrap();
        game.current_action_space = game.flush_triggers();
    }
    let stack = Observation::for_player(&game, PlayerId(1));
    let descriptions: Vec<_> = stack
        .stack_objects
        .iter()
        .filter_map(|s| s.description.as_deref())
        .collect();
    assert_eq!(
        descriptions,
        labels
            .iter()
            .rev()
            .map(|l| l.strip_prefix("Put on stack next — ").unwrap())
            .collect::<Vec<_>>()
    );
}
