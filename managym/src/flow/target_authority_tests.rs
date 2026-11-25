//! Exact event identities and stack targets: enumeration, acceptance and resolution.

use std::collections::BTreeMap;

use crate::{
    flow::{
        game::Game,
        trigger::{PendingTrigger, TriggerId},
    },
    state::{
        ability::{
            Effect, EffectValue, ObjectSelector, PlayerSelector, TargetRequirement, TargetSpec,
        },
        card::{Keyword, Keywords},
        effect::{EffectDuration, Modification},
        game_object::{AbilitySource, ObjectId, PermanentId, PlayerId, Target},
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
        Modification::AddKeywords(Keywords::of(&[Keyword::Hexproof])),
        EffectDuration::EndOfTurn,
    );
}

fn fixture() -> (Game, ObjectId, ObjectId, Target, Target) {
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
    game.keep_opening_hands();
    let ogre = game
        .scenario_force_battlefield(PlayerId(0), "Gray Ogre", true)
        .unwrap();
    let land = game
        .scenario_force_battlefield(PlayerId(0), "Mountain", true)
        .unwrap();
    let card_named = |name: &str| {
        game.state
            .latest_objects()
            .find(|(_, object)| object.owner == PlayerId(1) && object.name == name)
            .unwrap()
            .0
    };
    let bolt = card_named("Lightning Bolt");
    let ogre_spell = card_named("Gray Ogre");
    let bolt = game.push_spell_to_stack(bolt, PlayerId(1), vec![], vec![], Default::default());
    let ogre_spell =
        game.push_spell_to_stack(ogre_spell, PlayerId(1), vec![], vec![], Default::default());
    (
        game,
        bolt,
        ogre_spell,
        Target::Permanent(ogre),
        Target::Permanent(land),
    )
}

fn trigger(source: ObjectId, effects: Vec<Effect>, order: u64) -> PendingTrigger {
    PendingTrigger {
        inline_effects: Some(effects),
        ..PendingTrigger::new(
            TriggerId(order),
            PlayerId(0),
            AbilitySource::new(source, None),
            0,
        )
    }
}

/// The snapshot's single exact identity still projects the v16 `card` slot,
/// even after the source departs and its storage slot holds another object.
#[test]
fn trigger_source_snapshot_keeps_v16_bytes_after_departure() {
    use crate::flow::triggers::TriggerSourceSnapshot;

    let (mut game, _, _, Target::Permanent(permanent), _) = fixture() else {
        unreachable!()
    };
    let object = game.state.permanents[permanent].as_ref().unwrap().object;
    let object = game.move_object(object, ZoneType::Graveyard);
    let object = game.move_object(object, ZoneType::Battlefield);
    assert!(object.generation() > 0);
    let source = TriggerSourceSnapshot {
        lki: game.snapshot_current_permanent(object).unwrap(),
    };
    let expected = format!(
        "{{\"card\":{},\"controller\":0,\"lki\":{}}}",
        object.slot(),
        serde_json::to_string(&source.lki).unwrap()
    );
    assert_eq!(serde_json::to_string(&source).unwrap(), expected);

    game.move_object(object, ZoneType::Graveyard);
    assert!(game.state.object(source.lki.object).is_err());
    assert_eq!(source.lki.object, object);
    assert_eq!(serde_json::to_string(&source).unwrap(), expected);
}

#[test]
fn cr_400_zone_event_keeps_its_departure_after_a_later_death() {
    use crate::flow::event::GameEvent;

    let (mut game, _, _, Target::Permanent(first), _) = fixture() else {
        unreachable!()
    };
    let object = game.state.permanents[first].as_ref().unwrap().object;
    let original = game.state.now(object);
    let object = game.move_object(object, ZoneType::Graveyard);
    let event = game.state.events.iter().last().unwrap().clone();
    let GameEvent::CardMoved {
        object_id: became,
        departed,
        ..
    } = &event
    else {
        panic!("committed zone event")
    };
    assert_eq!(*departed, Some(original));
    assert_eq!(
        game.state.object_of(game.state.card_id(object).unwrap()),
        *became
    );
    assert_ne!(*became, original);
    assert_eq!(
        serde_json::to_value(&event).unwrap(),
        serde_json::json!({"CardMoved": {
            "card": object.slot(),
            "from": "Battlefield",
            "to": "Graveyard",
            "controller": 0
        }})
    );

    let object = game.move_object(object, ZoneType::Battlefield);
    let second = game.state.permanent_of(game.state.now(object)).unwrap();
    let object = game.move_object(object, ZoneType::Graveyard);
    assert_ne!(first, second);
    assert_eq!(
        game.object_lki(original).map(|lki| lki.permanent),
        Some(first)
    );
    let sources = game.trigger_ability_sources(&event);
    assert!(sources.iter().any(|source| source.lki.object == original));
    assert!(!sources.iter().any(|source| source.lki.permanent == second));
    assert!(game.state.object(*became).is_err());
    let fork = game.clone();
    assert_eq!(
        fork.object_lki(original).map(|lki| lki.permanent),
        Some(first)
    );

    let object = game.move_object(object, ZoneType::Hand);
    let hand_event = game.state.events.iter().last().unwrap();
    assert!(!game
        .trigger_ability_sources(hand_event)
        .iter()
        .any(|source| source.lki.object == object));
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
            game.legal_targets_for_spec(&TargetSpec::AnyTarget, controller),
            vec![
                Target::Player(PlayerId(0)),
                Target::Player(PlayerId(1)),
                permanent
            ]
        );
    }
    let bolt = game.move_object(bolt, ZoneType::Graveyard);
    let ogre = game.move_object(ogre, ZoneType::Graveyard);
    let spec = TargetSpec::GraveyardCardYouOwn {
        predicate: CardPredicate::default(),
    };
    assert!(game.legal_targets_for_spec(&spec, PlayerId(0)).is_empty());
    assert_eq!(
        game.legal_targets_for_spec(&spec, PlayerId(1)),
        vec![
            Target::GraveyardCard(game.state.now(bolt)),
            Target::GraveyardCard(game.state.now(ogre)),
        ]
    );
    assert_eq!(
        game.legal_targets_for_spec(
            &TargetSpec::GraveyardCardYouOwn {
                predicate: CardPredicate::creature()
            },
            PlayerId(1)
        ),
        vec![Target::GraveyardCard(game.state.now(ogre))]
    );
}

#[test]
fn restricted_trigger_rejects_low_value_spell_and_counters_legal_spell() {
    let (mut game, bolt, ogre, _, _) = fixture();
    game.state.pending_triggers.push(trigger(
        bolt,
        vec![Effect::CounterSpell {
            what: ObjectSelector::Target(TargetSpec::SpellOrPermanent { min_mana_value: 2 }),
            exile: false,
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
    assert_eq!(
        game.state.zones.zone_of(game.state.now(ogre)),
        Some(ZoneType::Graveyard)
    );
    assert_eq!(
        game.state.zones.zone_of(game.state.now(bolt)),
        Some(ZoneType::Stack)
    );
}

#[test]
fn departed_only_target_prevents_even_untargeted_gain_life() {
    let (mut game, bolt, ogre, _, _) = fixture();
    game.state.pending_triggers.push(trigger(
        bolt,
        vec![
            Effect::CounterSpell {
                what: ObjectSelector::Target(TargetSpec::SpellOrPermanent { min_mana_value: 2 }),
                exile: false,
            },
            Effect::gain_life(7),
        ],
        0,
    ));
    assert!(game.flush_triggers().is_some());
    game.choose_trigger_target(PlayerId(0), Target::StackSpell(ogre))
        .unwrap();
    assert!(game.flush_triggers().is_none());
    let ogre = game.move_object(ogre, ZoneType::Hand);
    // CR 608.2b: with every target illegal the whole ability fails to resolve,
    // including its untargeted instruction. This is more than skipping countering.
    game.resolve_top_of_stack();
    assert_eq!(game.state.players[0].life, 20);
    assert_eq!(
        game.state.zones.zone_of(game.state.now(ogre)),
        Some(ZoneType::Hand)
    );
}

#[test]
fn recast_spell_resolves_before_the_old_targeted_ability() {
    let (mut game, source, ogre, _, _) = fixture();
    let original = game.state.now(ogre);
    game.state.pending_triggers.push(trigger(
        source,
        vec![
            Effect::CounterSpell {
                what: ObjectSelector::Target(TargetSpec::SpellOrPermanent { min_mana_value: 2 }),
                exile: false,
            },
            Effect::gain_life(7),
        ],
        0,
    ));
    assert!(game.flush_triggers().is_some());
    game.choose_trigger_target(PlayerId(0), Target::StackSpell(ogre))
        .unwrap();
    assert!(game.flush_triggers().is_none());
    let ogre = game.move_object(ogre, ZoneType::Hand);
    game.push_spell_to_stack(ogre, PlayerId(1), vec![], vec![], Default::default());
    assert_ne!(game.state.now(ogre), original);

    // CR 405.2 / 608.2b: the recast object is above the older ability, so
    // resolves first. ObjectId-only targets remain an identity limitation;
    // this ordinary LIFO sequence does not expose a stale-target effect.
    game.resolve_top_of_stack();
    assert_eq!(
        game.state.zones.zone_of(game.state.now(ogre)),
        Some(ZoneType::Battlefield)
    );
    game.resolve_top_of_stack();
    assert_eq!(
        game.state.zones.zone_of(game.state.now(ogre)),
        Some(ZoneType::Battlefield)
    );
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
            Effect::gain_life(7),
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
        trigger(bolt, vec![Effect::gain_life(2)], 0),
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
        let gain = vec![Effect::gain_life(2)];
        let damage = vec![Effect::DealDamageToPlayer {
            player: PlayerSelector::You,
            amount: EffectValue::Fixed(2),
        }];
        game.state.pending_triggers = vec![trigger(source, gain, 0), trigger(source, damage, 1)];
        let choice = game.flush_triggers().unwrap();
        assert_eq!(choice.actions.len(), 2);
        game.choose_trigger(PlayerId(0), TriggerId(if gain_last { 1 } else { 0 }))
            .unwrap();
        assert!(game.flush_triggers().is_none());
        game.resolve_top_of_stack();
        game.perform_state_based_actions();
        if game.state.loss(0).is_none() {
            game.resolve_top_of_stack();
            game.perform_state_based_actions();
        }
        // CR 405.2: last placed resolves first. CR 704.5a: lose at zero
        // before the next ability resolves. Both placements must be choices.
        assert_eq!(game.state.players[0].life, expected_life);
        assert_eq!(game.state.loss(0).is_none(), expected_alive);
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
                (TargetSpec::AnyTarget, own),
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
                        effects: vec![Effect::return_to_hand(ObjectSelector::Target(spec.clone()))],
                    },
                    Effect::TargetClause {
                        requirement: TargetRequirement::one(spec.clone()),
                        effects: vec![Effect::return_to_hand(ObjectSelector::Target(spec))],
                    },
                    Effect::gain_life(7),
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
            let second_card = game.state.permanents[second].as_ref().unwrap().object;
            // CR 608.2b: gaining hexproof after selection invalidates an opposing
            // target; remaining legal targets and untargeted instructions survive.
            grant_hexproof(&mut game, first);
            if !mixed {
                grant_hexproof(&mut game, second);
            }
            game.resolve_top_of_stack();
            assert!(game.state.permanents[first].is_some());
            assert_eq!(
                game.state.zones.zone_of(game.state.now(second_card)),
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
        let object = game.state.permanents[id].as_ref().unwrap().object;
        let before = game.state.now(object);
        let spec = TargetSpec::SpellOrPermanent { min_mana_value: 3 };
        let mut pending = trigger(
            source,
            vec![
                Effect::return_to_hand(ObjectSelector::Target(spec.clone())),
                Effect::gain_life(7),
            ],
            0,
        );
        pending.controller = controller;
        game.state.pending_triggers.push(pending);
        assert!(game.flush_triggers().is_some());
        game.choose_trigger_target(controller, Target::Permanent(id))
            .unwrap();
        assert!(game.flush_triggers().is_none());
        let object = game.move_object(object, ZoneType::Hand);
        let object = game.move_object(object, ZoneType::Battlefield);
        let replacement = game.state.permanent_of(game.state.now(object)).unwrap();
        assert_ne!(game.state.now(object), before);
        assert_ne!(replacement, id);
        assert!(!game.target_is_legal(Target::Permanent(id), &spec, controller));
        assert!(game.target_is_legal(Target::Permanent(replacement), &spec, controller));
        game.resolve_top_of_stack();
        assert_eq!(
            game.state.zones.zone_of(game.state.now(object)),
            Some(ZoneType::Battlefield)
        );
        assert_eq!(game.state.players[controller.0].life, 20);
    }
}

#[test]
fn trigger_order_all_three_occurrence_permutations_are_legal() {
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
            .map(|id| trigger(source, vec![Effect::gain_life(id as i32 + 1)], id))
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
    use crate::flow::trigger::TriggerOrderingClass;
    for active in [PlayerId(0), PlayerId(1)] {
        let (mut game, source, ..) = fixture();
        let ordinary = trigger(source, vec![Effect::gain_life(1)], 0);
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
        game.state.trigger_placement = Some(Box::new(crate::flow::trigger::TriggerPlacement {
            remaining: batch,
            active,
            group: 0,
            placing: None,
        }));
        game.state
            .pending_triggers
            .push(trigger(source, vec![Effect::gain_life(9)], 99));
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
        .push(trigger(source, vec![Effect::gain_life(1)], 0));
    assert!(game.flush_triggers().is_none());
    game.state.pending_triggers = (1..66)
        .map(|id| trigger(source, vec![Effect::gain_life(1)], id))
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
    let mut first = trigger(source, vec![Effect::gain_life(1)], 900);
    first.source_identity = AbilitySource::new(
        first.source_object(),
        game.state.live(game.state.now(source)),
    );
    first.context = Some(Target::StackSpell(source));
    let mut second = first.clone();
    second.id = TriggerId(901);
    second.ability_index = 1;
    let mut third = first.clone();
    third.id = TriggerId(902);
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
fn trigger_order_target_resume_and_nested_fork_preserve_round() {
    use crate::agent::action::{Action, ActionSpaceKind};
    use crate::search_state::witness;
    let (mut game, source, ..) = fixture();
    game.state.pending_triggers = vec![
        trigger(
            source,
            vec![Effect::TargetClause {
                requirement: TargetRequirement::one(TargetSpec::Player),
                effects: vec![Effect::gain_life(1)],
            }],
            0,
        ),
        trigger(source, vec![Effect::gain_life(2)], 1),
        trigger(source, vec![Effect::gain_life(3)], 2),
    ];
    game.state.pending_triggers[0].source_identity =
        AbilitySource::new(source, game.state.live(game.state.now(source)));
    game.current_action_space = game.flush_triggers();
    let root = witness(&game);
    let mut branch = game.clone();
    let outer = branch.clone();
    branch.step(0).unwrap();
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
    let targeting = witness(&branch);
    let inner = branch.clone();
    branch.step(0).unwrap();
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
    branch = inner;
    assert_eq!(witness(&branch), targeting);
    branch = outer;
    assert_eq!(witness(&branch), root);
    assert_eq!(witness(&game), root);
}

#[test]
fn trigger_order_descriptions_use_exact_ability_and_delayed_payload() {
    use crate::agent::observation::Observation;
    use crate::state::ability::{Ability, TriggerCondition, TriggerSubject};
    let (mut game, source, ..) = fixture();
    game.state[source].abilities = vec![
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
            effects: vec![Effect::gain_life(3)],
        },
    ];
    let mut dies = trigger(source, vec![], 0);
    dies.inline_effects = None;
    let mut enters = dies.clone();
    enters.id = TriggerId(1);
    enters.ability_index = 1;
    let mut delayed = trigger(
        source,
        vec![Effect::Move {
            unrevealed: false,
            library_end: None,
            what: ObjectSelector::This,
            to: ZoneType::Battlefield,
            tapped: true,
            transformed: false,
            under: Some(PlayerSelector::Owner),
            with_counter: None,
            linked: None,
        }],
        2,
    );
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
        .all(|label| !label.contains(&game.state[source].text_box)));
    for id in 0..2 {
        game.choose_trigger(PlayerId(0), TriggerId(id)).unwrap();
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

/// A departure trigger matches the face the source had when it left play.
#[test]
fn retained_departure_predicate_uses_departing_copy_after_reentry() {
    use crate::flow::triggers::TriggerSourceSnapshot;
    use crate::state::card::{CardType, StaticAbility};
    let (mut game, bolt, _, Target::Permanent(ogre), Target::Permanent(land)) = fixture() else {
        unreachable!()
    };
    let object = game.state.permanents[ogre].as_ref().unwrap().object;
    let grantor = game.state.permanents[land].as_ref().unwrap().object;
    game.state[grantor]
        .statics
        .push(StaticAbility::TriggersAdditionalTime(CardPredicate {
            card_type: Some(CardType::Land),
            ..Default::default()
        }));
    let printed = TriggerSourceSnapshot {
        lki: game.snapshot_current_permanent(object).unwrap(),
    };
    assert_eq!(game.additional_trigger_times(&printed), 0);

    let template = game.state[grantor].clone();
    game.state[object].become_copy_of(&template, &[]);
    let source = TriggerSourceSnapshot {
        lki: game.snapshot_current_permanent(object).unwrap(),
    };
    let graveyard = game.move_object(object, ZoneType::Graveyard);
    assert_eq!(game.additional_trigger_times(&source), 1);
    let reentered = game.move_object(graveyard, ZoneType::Battlefield);
    let template = game.state[bolt].clone();
    game.state[reentered].become_copy_of(&template, &[]);
    assert_eq!(game.additional_trigger_times(&source), 1);
    assert_eq!(game.state.source_base(object).1.name, "Gray Ogre");
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| &game.state[object])).is_err()
    );
}

#[test]
fn retained_departure_modes_targets_and_description_use_source_face() {
    use crate::agent::observation::Observation;
    use crate::state::ability::{Ability, Actor, TriggerCondition};
    use crate::state::stack_object::StackObject;
    let (mut game, bolt, ogre_spell, Target::Permanent(ogre), land) = fixture() else {
        unreachable!()
    };
    let object = game.state.permanents[ogre].as_ref().unwrap().object;
    let effects = vec![Effect::TriggerModes {
        modes: vec![vec![Effect::deal_damage(
            2,
            ObjectSelector::Target(TargetSpec::SpellOrPermanent { min_mana_value: 0 }),
        )]],
        up_to: None,
        unchosen: false,
    }];
    game.state[object].abilities = vec![Ability::Triggered {
        condition: TriggerCondition::casts(Actor::You, CardPredicate::default()),
        intervening_if: None,
        effects,
    }];
    let mut pending = trigger(object, vec![], 200);
    pending.inline_effects = None;
    pending.source_identity = AbilitySource::new(object, Some(object));
    pending.source_lki = game.snapshot_current_permanent(object);
    let graveyard = game.move_object(object, ZoneType::Graveyard);
    let current = game.move_object(graveyard, ZoneType::Battlefield);
    let template = game.state[bolt].clone();
    game.state[current].become_copy_of(&template, &[]);

    let trigger_id = pending.id;
    game.state.trigger_placement = Some(Box::new(crate::flow::trigger::TriggerPlacement {
        remaining: vec![pending.clone()],
        active: PlayerId(0),
        group: 0,
        placing: None,
    }));
    let label = game
        .action_description(&crate::agent::action::Action::ChooseTrigger {
            player: PlayerId(0),
            trigger_id,
        })
        .unwrap();
    assert!(label.contains("Gray Ogre"));
    // Make the choice through placement, using the departed source's face.
    game.choose_trigger(PlayerId(0), trigger_id).unwrap();
    game.current_action_space = game.flush_triggers();
    let context = game.decision_context(PlayerId(0)).unwrap();
    assert_eq!(context.stage, "trigger_mode");
    assert!(context.instruction.contains("Gray Ogre"));
    assert_eq!(game.action_space().unwrap().actions.len(), 1);
    game.choose_trigger_mode(PlayerId(0), 0).unwrap();
    game.current_action_space = game.flush_triggers();
    let context = game.decision_context(PlayerId(0)).unwrap();
    assert_eq!(context.stage, "trigger_targets");
    assert!(context.instruction.contains("Gray Ogre"));
    game.choose_trigger_target(PlayerId(0), Target::StackSpell(bolt))
        .unwrap();
    assert!(game.flush_triggers().is_none());
    let stacked = game.state.stack_objects.last().unwrap();
    assert_eq!(
        game.retarget_candidates(stacked.as_target(), 0),
        [
            Target::StackSpell(ogre_spell),
            land,
            Target::Permanent(game.state.permanent_of(current).unwrap())
        ],
    );
    let StackObject::TriggeredAbility(stacked) = stacked else {
        unreachable!()
    };
    assert_eq!(stacked.source_object(), object);
    assert_eq!(stacked.source_ref(), Some(object));
    let id = stacked.id.0 as i32;
    let description = Observation::for_player(&game, PlayerId(0))
        .stack_objects
        .iter()
        .find(|row| row.stack_object_id == id)
        .unwrap()
        .description
        .clone()
        .unwrap();
    assert!(description.contains("Gray Ogre"));
}

#[test]
fn retained_departure_spell_delayed_and_child_reflexive_keep_one_source() {
    use crate::flow::{
        decision::EffectFrame,
        trigger::{DelayedTrigger, DelayedTriggerEvent},
    };
    use crate::state::ability::{DelayedExpiry, DelayedFrequency};
    let (mut game, bolt, _, _, _) = fixture();
    let source = AbilitySource::new(bolt, Some(bolt));
    let delayed = DelayedTrigger {
        remembered: vec![],
        event: DelayedTriggerEvent::Upkeep,
        controller: PlayerId(1),
        source_identity: source,
        effects: crate::flow::trigger::DelayedEffects::new(vec![Effect::GainLife {
            amount: EffectValue::Fixed(3),
        }]),
        frequency: DelayedFrequency::Once,
        expiry: DelayedExpiry::Never,
    };
    game.move_object(bolt, ZoneType::Graveyard);
    game.fire_delayed_trigger(&delayed, None);
    let pending = game.state.pending_triggers.last().unwrap();
    assert_eq!(pending.source_identity, source);
    game.place_triggered_ability_on_stack(pending.clone());
    let life = game.state.players[1].life;
    game.resolve_top_of_stack();
    assert_eq!(game.state.players[1].life, life + 3);
    let mut frame = EffectFrame::new(
        PlayerId(1),
        vec![Effect::WhenYouDo {
            effects: vec![Effect::GainLife {
                amount: EffectValue::Fixed(1),
            }],
        }],
    );
    frame.source_identity = Some(source);
    game.run_frame(frame);
    assert_eq!(
        game.state.pending_triggers.last().unwrap().source_identity,
        source
    );
}

#[test]
fn pending_source_keeps_v16_field_order_and_nullable_reference() {
    let (mut game, source, _, _, _) = fixture();
    let mut pending = trigger(source, vec![], 42);
    for captured in [false, true] {
        pending.source_identity = AbilitySource::new(source, captured.then_some(source));
        let reference = if captured {
            format!(
                "{{\"entity\":{},\"incarnation\":{}}}",
                source.slot(),
                source.generation()
            )
        } else {
            "null".into()
        };
        let expected = format!("{{\"source_card\":{},\"targets\":[],\"target_req_indices\":[],\"requirement_index\":0,\"source_ref\":{},\"source_lki\":null,\"ability_index\":0,\"controller\":0,\"id\":42,\"ordering_class\":\"Ordinary\",\"context\":null,\"inline_effects\":[]}}", source.slot(), reference);
        assert_eq!(serde_json::to_string(&pending).unwrap(), expected);
    }
    let successor = game.move_object(source, ZoneType::Graveyard);
    let identity = AbilitySource::new(successor, Some(source));
    assert_eq!(identity.object(), source);
    let other = game
        .state
        .zones
        .zone_objects(ZoneType::Library, PlayerId(0))[0];
    assert!(std::panic::catch_unwind(|| AbilitySource::new(other, Some(source))).is_err());
}

#[test]
fn retained_departure_recorded_card_and_spell_predicates_ignore_later_copy() {
    use crate::flow::decision::EffectFrame;
    use crate::state::{ability::Condition, card::CardType};
    let (mut game, bolt, ogre, _, _) = fixture();
    let later = game.move_object(bolt, ZoneType::Graveyard);
    let template = game.state[ogre].clone();
    game.state[later].become_copy_of(&template, &[]);
    let predicate = CardPredicate {
        card_type: Some(CardType::Instant),
        ..Default::default()
    };
    let mut frame = EffectFrame::new(PlayerId(1), vec![]);
    frame.this_way = vec![bolt];
    frame.context_target = Some(Target::StackSpell(bolt));
    assert!(game.check_condition(
        &Condition::CardsThisWayMatch(predicate.clone()),
        PlayerId(1),
        Some(&frame)
    ));
    assert!(game.check_condition(
        &Condition::TriggeringSpellMatches(predicate.clone()),
        PlayerId(1),
        Some(&frame)
    ));
    assert!(game.recorded_object_matches_predicate(bolt, &predicate));
    assert!(!game.object_matches_predicate(later, &predicate));
}

/// "That share a creature type" holds for every clause target: a trigger
/// placement rejects a non-sharing second target, and retargeting offers
/// only replacements sharing a type with the clause's other targets.
#[test]
fn shared_creature_type_clause_binds_trigger_targets_and_retargeting() {
    let (mut game, source, _, Target::Permanent(ogre), _) = fixture() else {
        unreachable!()
    };
    let force_ogre = |game: &mut Game| {
        game.scenario_force_battlefield(PlayerId(0), "Gray Ogre", true)
            .unwrap()
    };
    let second = force_ogre(&mut game);
    let third = force_ogre(&mut game);
    let giant = force_ogre(&mut game);
    let giant_object = game.state.permanents[giant].as_ref().unwrap().object;
    game.state[giant_object].subtypes = vec!["Giant".into()];
    game.state.pending_triggers.push(trigger(
        source,
        vec![
            Effect::TargetClause {
                requirement: TargetRequirement {
                    shares_creature_type: true,
                    min: 2,
                    max: 2,
                    ..TargetRequirement::one(TargetSpec::Creature)
                },
                effects: vec![],
            },
            Effect::gain_life(1),
        ],
        0,
    ));
    assert!(game.flush_triggers().is_some());
    game.choose_trigger_target(PlayerId(0), Target::Permanent(ogre))
        .unwrap();
    assert!(game
        .choose_trigger_target(PlayerId(0), Target::Permanent(giant))
        .is_err());
    game.choose_trigger_target(PlayerId(0), Target::Permanent(second))
        .unwrap();
    assert!(game.flush_triggers().is_none());
    let stacked = game.state.stack_objects.last().unwrap().as_target();
    assert_eq!(
        game.retarget_candidates(stacked, 1),
        [Target::Permanent(third)]
    );
}

/// New targets are judged as one assignment (CR 115.7e). A copy of
/// "4 damage to any target and 3 to another target" may swap its two
/// targets, which a check of each slot against the others' old targets
/// would reject; once the first slot takes the second's object, the second
/// must change. "Change the target" keeps the other slots, so it never
/// offers an object another slot holds.
#[test]
fn copy_targets_change_as_one_assignment() {
    use crate::{flow::retarget::RetargetStep, state::ability::TargetRelation};
    let (mut game, source, _, Target::Permanent(a), _) = fixture() else {
        unreachable!()
    };
    let b = game
        .scenario_force_battlefield(PlayerId(1), "Gray Ogre", true)
        .unwrap();
    let (a, b) = (Target::Permanent(a), Target::Permanent(b));
    game.state.pending_triggers.push(trigger(
        source,
        vec![
            Effect::TargetClause {
                requirement: TargetRequirement::one(TargetSpec::Creature),
                effects: vec![],
            },
            Effect::TargetClause {
                requirement: TargetRequirement {
                    relation: Some(TargetRelation::DistinctFromEarlier),
                    ..TargetRequirement::up_to(1, TargetSpec::Creature)
                },
                effects: vec![],
            },
        ],
        0,
    ));
    assert!(game.flush_triggers().is_some());
    game.choose_trigger_target(PlayerId(0), a).unwrap();
    // The second clause no longer offers the first clause's target.
    assert_eq!(
        game.flush_triggers().unwrap().actions,
        [
            crate::agent::action::Action::ChooseTarget {
                player: PlayerId(0),
                target: b,
            },
            crate::agent::action::Action::Decline {
                player: PlayerId(0)
            },
        ]
    );
    assert!(game.choose_trigger_target(PlayerId(0), a).is_err());
    game.choose_trigger_target(PlayerId(0), b).unwrap();
    assert!(game.flush_triggers().is_none());
    let original = game.state.stack_objects.last().unwrap().as_target();

    // "Change the target" of one slot: the other slot's object is taken.
    assert_eq!(game.retarget_offer(original, &[], false), (vec![], false));

    let copy = game.copy_stack_object(original, PlayerId(0)).unwrap();
    assert_eq!(game.retarget_offer(copy, &[], true), (vec![b], true));
    assert_eq!(game.retarget_offer(copy, &[b], true), (vec![a], false));
    assert_eq!(game.retarget_offer(copy, &[a], true), (vec![], true));
    let Some(RetargetStep::Decide(_)) = game.next_retarget(PlayerId(0), copy, vec![b], true) else {
        panic!("the swapped slot still decides");
    };
    // Publishing writes both targets at once.
    assert!(game
        .next_copy_target(PlayerId(0), copy, vec![b, a])
        .is_none());
    let copied = &game.state.stack_objects[game.stack_index_of(copy).unwrap()];
    assert_eq!(copied.targets(), [b, a]);
}

#[test]
fn graveyard_target_rechecks_live_source_power_for_retarget_and_resolution() {
    use crate::state::{game_object::AbilitySource, permanent::CounterKind};
    let (mut game, _, _, Target::Permanent(ogre), _) = fixture() else {
        unreachable!()
    };
    let player = PlayerId(0);
    let source = game.permanent_object(ogre).unwrap();
    let mut bury_ogre = || {
        let id = game
            .scenario_force_battlefield(player, "Gray Ogre", true)
            .unwrap();
        game.move_object(game.permanent_object(id).unwrap(), ZoneType::Graveyard)
    };
    let first = bury_ogre();
    let second = bury_ogre();
    game.state.permanents[ogre]
        .as_mut()
        .unwrap()
        .change_counters(CounterKind::PlusOnePlusOne, 1);
    let spec = TargetSpec::GraveyardCardYouOwnWithinSourcePower {
        predicate: CardPredicate::creature(),
    };
    let mut pending = trigger(
        source,
        vec![Effect::return_to_battlefield(
            ObjectSelector::Target(spec),
            false,
        )],
        0,
    );
    pending.source_identity = AbilitySource::new(source, Some(source));
    game.state.pending_triggers.push(pending);
    assert_eq!(game.flush_triggers().unwrap().actions.len(), 2);
    game.choose_trigger_target(player, Target::GraveyardCard(first))
        .unwrap();
    assert!(game.flush_triggers().is_none());
    let stack = game.state.stack_objects.last().unwrap().as_target();
    assert!(game.next_retarget(player, stack, 0, true).is_some());
    assert_eq!(
        game.retarget_candidates(stack, 0),
        vec![Target::GraveyardCard(second)]
    );
    game.state.permanents[ogre]
        .as_mut()
        .unwrap()
        .change_counters(CounterKind::PlusOnePlusOne, -1);
    assert!(game.retarget_candidates(stack, 0).is_empty());
    game.resolve_top_of_stack();
    assert_eq!(game.state.zones.zone_of(first), Some(ZoneType::Graveyard));
}

#[test]
fn exude_toxin_target_threshold_uses_the_announced_x() {
    use crate::{flow::game::CastChoices, state::card::SpellProfile};
    let deck = BTreeMap::from([("Scavenger Regent".to_string(), 20)]);
    let mut game = Game::new(
        vec![
            PlayerConfig::new("a", deck.clone()),
            PlayerConfig::new("b", deck),
        ],
        42,
        false,
    );
    for x in [0, 1] {
        let card = game
            .state
            .zones
            .zone_objects(ZoneType::Library, PlayerId(0))[0];
        let spell = game.push_spell_to_stack(
            card,
            PlayerId(0),
            vec![],
            vec![],
            CastChoices {
                profile: SpellProfile::Omen,
                x,
                ..Default::default()
            },
        );
        let spec = TargetSpec::SpellOrPermanent { min_mana_value: 3 };
        let target = Target::StackSpell(spell);
        assert_eq!(
            game.legal_targets_for_spec(&spec, PlayerId(1))
                .contains(&target),
            x == 1
        );
        assert_eq!(game.target_is_legal(target, &spec, PlayerId(1)), x == 1);
    }
}
