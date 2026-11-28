//! Constructed boards where one payment step changes the next step's rules.

use super::{counter_replacement_mana_setup, only_card_in_main, payment_game, step_action};
use crate::{
    agent::action::Action,
    flow::{
        decision::{Decision, EffectFrame},
        game::Game,
        payment::{CostComponent, PaymentProgress},
    },
    state::{
        ability::{Actor, Condition, Effect, EffectValue, PlayerSelector},
        card::{
            ActivatedAbilityDefinition, AdditionalCost, DamageRecipient, DamageSource,
            DamageTransform, Keyword, Keywords, ReplacementEffect, StaticAbility,
            StaticKeywordGrant, StaticScope, TapCost,
        },
        effect::{EffectDuration, Modification},
        game_object::{ObjectId, PermanentId, PlayerId},
        mana::{Mana, ManaCost},
        permanent::CounterKind,
        player::ManaFloats,
        predicate::CardPredicate,
        zone::ZoneType,
    },
};

const PAYER: PlayerId = PlayerId(0);

fn put(game: &mut Game, name: &str) -> PermanentId {
    game.scenario_force_battlefield(PAYER, name, true).unwrap()
}

fn activate(permanent: PermanentId, ability_index: usize) -> Action {
    Action::ActivateAbility {
        player: PAYER,
        permanent,
        ability_index,
    }
}

fn offer(game: &Game, action: &Action) -> bool {
    game.action_space().unwrap().actions.contains(action)
}

fn pay_during_resolution(game: &mut Game, mana: &str, life: u32) {
    game.mana_floats[PAYER.0] = ManaFloats::All;
    let mut owed = vec![CostComponent::mana(ManaCost::parse(mana))];
    if life != 0 {
        owed.push(CostComponent::Life(life));
    }
    let mut payment = PaymentProgress::new(PAYER, owed, Vec::new(), Vec::new());
    payment.committed = true;
    game.continue_resolution(
        EffectFrame::new(PAYER, Vec::new()),
        Some(Decision::Payment(payment)),
    );
    game.tick().unwrap();
}

fn choose_mana(game: &mut Game, output: &str) {
    let Decision::ManaChoice { choices, .. } =
        &game.state.suspended_decision.as_ref().unwrap().decision
    else {
        panic!("expected a mana choice");
    };
    let output = Mana::parse(output);
    let mode = choices.iter().position(|choice| *choice == output).unwrap();
    step_action(
        game,
        Action::ChooseMode {
            player: PAYER,
            mode,
        },
    )
    .unwrap();
}

fn finish_payment(game: &mut Game) {
    if game.state.suspended_decision.is_some() || game.pending_choice.is_some() {
        step_action(game, Action::PayCost { player: PAYER }).unwrap();
    }
    assert!(game.state.suspended_decision.is_none());
    assert!(game.pending_choice.is_none());
}

fn cast_offer(game: &Game, spell: ObjectId) -> Option<usize> {
    game.action_space()
        .unwrap()
        .actions
        .iter()
        .position(|action| matches!(action, Action::CastSpell { object, .. } if *object == spell))
}

#[test]
fn coupled_payer_replacement_requires_the_completing_order() {
    let output = Effect::If {
        condition: Condition::ControlsAtLeast {
            count: 1,
            distinct_names: false,
            predicate: CardPredicate {
                min_power: Some(6),
                ..CardPredicate::creature()
            },
        },
        then: vec![Effect::add_mana(Mana::parse("GG"))],
        otherwise: vec![Effect::add_mana(Mana::parse("G"))],
    };
    for skip_trivial in [false, true] {
        let (mut env, ogre) = counter_replacement_mana_setup(output.clone());
        let game = env.fixture_game_mut();
        let source = game.permanent_object(ogre).unwrap();
        // Retain a priority choice after payment so the spell stays on the
        // stack for inspection with trivial-decision skipping enabled.
        game.state[source]
            .activated_abilities
            .push(ActivatedAbilityDefinition {
                sacrifice_source: true,
                effects: vec![Effect::gain_life(1)],
                ..Default::default()
            });
        let spell = only_card_in_main(game, PAYER, "Gray Ogre");
        game.state[spell].mana_cost = Some(ManaCost::parse("GG"));
        game.scenario_refresh_priority().unwrap();
        game.step(cast_offer(game, spell).unwrap()).unwrap();
        game.skip_trivial = skip_trivial;
        step_action(game, activate(ogre, 0)).unwrap();

        if !skip_trivial {
            let Decision::OrderReplacement { candidates, .. } =
                &game.state.suspended_decision.as_ref().unwrap().decision
            else {
                panic!("expected counter replacement order");
            };
            let caradora = candidates
                .iter()
                .position(|candidate| {
                    matches!(
                        candidate.effect,
                        ReplacementEffect::ModifyCounters { additional: 1, .. }
                    )
                })
                .unwrap();
            assert_eq!(candidates.len(), 2);
            assert_eq!(
                game.action_space().unwrap().actions,
                [Action::ChooseMode {
                    player: PAYER,
                    mode: caradora
                }]
            );
            let pending = game.state.deterministic_hash();
            let err = game
                .execute_decision_action(&Action::ChooseMode {
                    player: PAYER,
                    mode: 1 - caradora,
                })
                .unwrap_err();
            assert!(err.0.contains("cannot complete the payment"));
            assert_eq!(game.state.deterministic_hash(), pending);
            step_action(
                game,
                Action::ChooseMode {
                    player: PAYER,
                    mode: caradora,
                },
            )
            .unwrap();
            choose_mana(game, "GG");
            finish_payment(game);
        }
        assert_eq!(
            game.state.permanents[ogre]
                .as_ref()
                .unwrap()
                .counter_count(CounterKind::PlusOnePlusOne),
            4
        );
        assert_eq!(game.state.stack_objects.len(), 1);
    }
}

#[test]
fn coupled_opponent_replacement_requires_every_order_to_complete() {
    let mut game = payment_game(352, [("Forest", 37), ("Gray Ogre", 3)]);
    game.keep_opening_hands();
    game.mana_floats[PAYER.0] = ManaFloats::All;
    let damage = put(&mut game, "Gray Ogre");
    let reader = put(&mut game, "Gray Ogre");
    let damage_object = game.permanent_object(damage).unwrap();
    game.state[damage_object].activated_abilities = vec![ActivatedAbilityDefinition {
        tap_source: true,
        effects: vec![
            Effect::add_mana(Mana::parse("R")),
            Effect::DealDamageToPlayer {
                player: PlayerSelector::EachOpponent,
                amount: EffectValue::Fixed(1),
            },
        ],
        ..Default::default()
    }];
    game.state[damage_object].replacement_effects = [
        DamageTransform::Double,
        DamageTransform::add(EffectValue::Fixed(1)),
    ]
    .into_iter()
    .map(|transform| ReplacementEffect::ModifyDamage {
        source: DamageSource::this(),
        recipient: DamageRecipient::any(Actor::Opponent),
        combat: Some(false),
        transform,
    })
    .collect();
    let reader_object = game.permanent_object(reader).unwrap();
    let green_for_damage = |amount| Effect::If {
        condition: Condition::OpponentLostLifeThisTurnAtLeast { amount },
        then: vec![Effect::add_mana(Mana::parse("GG"))],
        otherwise: vec![Effect::add_mana(Mana::parse("G"))],
    };
    game.state[reader_object].activated_abilities = vec![ActivatedAbilityDefinition {
        tap_source: true,
        effects: vec![green_for_damage(4)],
        ..Default::default()
    }];

    let spell = only_card_in_main(&mut game, PAYER, "Gray Ogre");
    game.state[spell].mana_cost = Some(ManaCost::parse("RGG"));
    game.scenario_refresh_priority().unwrap();
    assert!(cast_offer(&game, spell).is_none());

    // Once either damage total suffices, the cast can begin. The opponent
    // retains both choices even though the payer is still paying.
    game.state[reader_object].activated_abilities[0].effects[0] = green_for_damage(3);
    game.scenario_refresh_priority().unwrap();
    game.step(cast_offer(&game, spell).unwrap()).unwrap();
    step_action(&mut game, activate(damage, 0)).unwrap();
    let opponent = PlayerId(1);
    assert_eq!(
        game.action_space().unwrap().actions,
        [
            Action::ChooseMode {
                player: opponent,
                mode: 0
            },
            Action::ChooseMode {
                player: opponent,
                mode: 1
            },
        ]
    );
    step_action(
        &mut game,
        Action::ChooseMode {
            player: opponent,
            mode: 0,
        },
    )
    .unwrap();
    choose_mana(&mut game, "R");
    step_action(&mut game, activate(reader, 0)).unwrap();
    choose_mana(&mut game, "GG");
    finish_payment(&mut game);
    assert_eq!(game.state.stack_objects.len(), 1);
}

#[test]
fn coupled_delve_counts_only_sacrifices_that_reach_the_graveyard() {
    for exile in [false, true] {
        let mut game = payment_game(353, [("Forest", 38), ("Gray Ogre", 2)]);
        game.keep_opening_hands();
        let source = put(&mut game, "Gray Ogre");
        let object = game.permanent_object(source).unwrap();
        game.state[object].activated_abilities = vec![ActivatedAbilityDefinition {
            tap_source: true,
            sacrifice_source: true,
            effects: vec![Effect::add_mana(Mana::parse("G"))],
            ..Default::default()
        }];
        if exile {
            game.apply_continuous(
                None,
                PAYER,
                [game.affected_permanent(source).unwrap()],
                Modification::ExileIfWouldDie,
                EffectDuration::Indefinite,
            );
        }
        let spell = only_card_in_main(&mut game, PAYER, "Gray Ogre");
        game.state[spell].mana_cost = Some(ManaCost::parse("1G"));
        game.state[spell]
            .additional_costs
            .push(AdditionalCost::Delve);
        game.scenario_refresh_priority().unwrap();
        let cast = cast_offer(&game, spell);
        if exile {
            assert!(cast.is_none());
            continue;
        }
        game.step(cast.unwrap()).unwrap();
        step_action(&mut game, activate(source, 0)).unwrap();
        choose_mana(&mut game, "G");
        let sacrificed = game.state.zones.zone_objects(ZoneType::Graveyard, PAYER)[0];
        step_action(
            &mut game,
            Action::SelectCard {
                player: PAYER,
                object: sacrificed,
            },
        )
        .unwrap();
        finish_payment(&mut game);
        assert_eq!(game.state.stack_objects.len(), 1);
    }
}

#[test]
fn coupled_mana_uses_haste_before_removing_its_counter() {
    let mut game = payment_game(354, [("Forest", 39), ("Gray Ogre", 1)]);
    game.keep_opening_hands();
    let source = game
        .scenario_force_battlefield(PAYER, "Gray Ogre", false)
        .unwrap();
    let object = game.permanent_object(source).unwrap();
    let kind = CounterKind::PlusOnePlusOne;
    game.state.change_counters(source, kind, 1);
    game.state[object]
        .statics
        .push(StaticAbility::KeywordGrant(StaticKeywordGrant {
            scope: StaticScope::This,
            condition: Some(Condition::SourceCountersAtLeast { kind, count: 1 }),
            keywords: Keywords::of(&[Keyword::Haste]),
        }));
    game.state[object].activated_abilities = vec![
        ActivatedAbilityDefinition {
            tap_source: true,
            effects: vec![Effect::add_mana(Mana::parse("G"))],
            ..Default::default()
        },
        ActivatedAbilityDefinition {
            remove_counters: Some((kind, 1)),
            effects: vec![Effect::add_mana(Mana::parse("R"))],
            ..Default::default()
        },
    ];
    pay_during_resolution(&mut game, "GR", 0);

    assert!(offer(&game, &activate(source, 0)));
    assert!(!offer(&game, &activate(source, 1)));
    step_action(&mut game, activate(source, 0)).unwrap();
    choose_mana(&mut game, "G");
    step_action(&mut game, activate(source, 1)).unwrap();
    choose_mana(&mut game, "R");
    finish_payment(&mut game);
}

#[test]
fn coupled_counted_tap_reads_counters_removed_by_mana() {
    let mut game = payment_game(355, [("Forest", 38), ("Gray Ogre", 2)]);
    game.keep_opening_hands();
    let counted = put(&mut game, "Gray Ogre");
    let payer = put(&mut game, "Gray Ogre");
    let counted_object = game.permanent_object(counted).unwrap();
    let payer_object = game.permanent_object(payer).unwrap();
    let kind = CounterKind::PlusOnePlusOne;
    game.state.change_counters(payer, kind, 1);
    game.state[counted_object].activated_abilities = vec![ActivatedAbilityDefinition {
        tap_cost: Some(TapCost {
            count: 1,
            predicate: CardPredicate {
                with_counter: true,
                ..CardPredicate::creature()
            },
        }),
        effects: vec![Effect::add_mana(Mana::parse("G"))],
        ..Default::default()
    }];
    game.state[payer_object].activated_abilities = vec![ActivatedAbilityDefinition {
        remove_counters: Some((kind, 1)),
        effects: vec![Effect::add_mana(Mana::parse("R"))],
        ..Default::default()
    }];
    pay_during_resolution(&mut game, "GR", 0);

    assert!(offer(&game, &activate(counted, 0)));
    assert!(!offer(&game, &activate(payer, 0)));
    step_action(&mut game, activate(counted, 0)).unwrap();
    step_action(
        &mut game,
        Action::SelectCard {
            player: PAYER,
            object: payer_object,
        },
    )
    .unwrap();
    choose_mana(&mut game, "G");
    step_action(&mut game, activate(payer, 0)).unwrap();
    choose_mana(&mut game, "R");
    finish_payment(&mut game);
}

#[test]
fn coupled_fee_discount_reads_counters_removed_by_another_activation() {
    let mut game = payment_game(
        356,
        [
            ("Forest", 37),
            ("Gray Ogre", 1),
            ("Mountain", 1),
            ("White Lotus Tile", 1),
        ],
    );
    game.keep_opening_hands();
    let discount = put(&mut game, "White Lotus Tile");
    let source = put(&mut game, "Gray Ogre");
    let mountain = put(&mut game, "Mountain");
    let discount_object = game.permanent_object(discount).unwrap();
    let source_object = game.permanent_object(source).unwrap();
    let kind = CounterKind::PlusOnePlusOne;
    game.state.change_counters(source, kind, 1);
    game.state[discount_object].activated_abilities.clear();
    game.state[discount_object]
        .statics
        .push(StaticAbility::ActivatedAbilitiesCostLess {
            predicate: CardPredicate {
                with_counter: true,
                ..CardPredicate::creature()
            },
            generic: 1,
        });
    game.state[source_object].activated_abilities = vec![
        ActivatedAbilityDefinition {
            mana_cost: ManaCost::parse("1R"),
            tap_source: true,
            effects: vec![Effect::add_mana(Mana::parse("GG"))],
            ..Default::default()
        },
        ActivatedAbilityDefinition {
            remove_counters: Some((kind, 1)),
            effects: vec![Effect::add_mana(Mana::parse("B"))],
            ..Default::default()
        },
    ];
    pay_during_resolution(&mut game, "GGB", 0);
    assert!(offer(&game, &activate(source, 0)));
    assert!(!offer(&game, &activate(source, 1)));

    step_action(&mut game, activate(source, 0)).unwrap();
    step_action(&mut game, Action::PayCost { player: PAYER }).unwrap();
    assert!(game.state.permanents[mountain].as_ref().unwrap().tapped);
    choose_mana(&mut game, "GG");
    step_action(&mut game, activate(source, 1)).unwrap();
    choose_mana(&mut game, "B");
    finish_payment(&mut game);
}

#[test]
fn coupled_damage_prevention_is_needed_before_sacrifice() {
    let mut game = payment_game(
        357,
        [("Forest", 38), ("Gray Ogre", 1), ("Tarnished Citadel", 1)],
    );
    game.keep_opening_hands();
    game.scenario_set_life(PAYER, 3);
    let protection = put(&mut game, "Gray Ogre");
    let citadel = put(&mut game, "Tarnished Citadel");
    let object = game.permanent_object(protection).unwrap();
    game.state[object].replacement_effects = vec![ReplacementEffect::ModifyDamage {
        source: DamageSource::any(Actor::AnyPlayer),
        recipient: DamageRecipient::any(Actor::You),
        combat: Some(false),
        transform: DamageTransform::PreventAll,
    }];
    game.state[object].activated_abilities = vec![ActivatedAbilityDefinition {
        tap_source: true,
        sacrifice_source: true,
        effects: vec![Effect::add_mana(Mana::parse("G"))],
        ..Default::default()
    }];
    pay_during_resolution(&mut game, "GR", 2);

    assert!(offer(&game, &activate(citadel, 0)));
    assert!(!offer(&game, &activate(protection, 0)));
    step_action(&mut game, activate(citadel, 0)).unwrap();
    choose_mana(&mut game, "R");
    step_action(&mut game, activate(protection, 0)).unwrap();
    choose_mana(&mut game, "G");
    finish_payment(&mut game);
    assert_eq!(game.state.players[PAYER.0].life, 1);
}

#[test]
fn coupled_audience_needs_the_counter_until_its_tap() {
    let mut game = payment_game(
        358,
        [("Forest", 38), ("Gray Ogre", 1), ("Raucous Audience", 1)],
    );
    game.keep_opening_hands();
    let ogre = put(&mut game, "Gray Ogre");
    let audience = put(&mut game, "Raucous Audience");
    let kind = CounterKind::PlusOnePlusOne;
    game.state.change_counters(ogre, kind, 2);
    let object = game.permanent_object(ogre).unwrap();
    game.state[object].activated_abilities = vec![ActivatedAbilityDefinition {
        remove_counters: Some((kind, 1)),
        effects: vec![Effect::add_mana(Mana::parse("R"))],
        ..Default::default()
    }];
    pay_during_resolution(&mut game, "GGR", 0);
    assert!(offer(&game, &activate(audience, 0)));
    assert!(!offer(&game, &activate(ogre, 0)));
    step_action(&mut game, activate(audience, 0)).unwrap();
    choose_mana(&mut game, "GG");
    step_action(&mut game, activate(ogre, 0)).unwrap();
    choose_mana(&mut game, "R");
    finish_payment(&mut game);
}

#[test]
fn coupled_sacrifice_history_counts_a_treasure_token() {
    let mut game = payment_game(359, [("Forest", 39), ("Gray Ogre", 1)]);
    game.keep_opening_hands();
    let ogre = put(&mut game, "Gray Ogre");
    let object = game.permanent_object(ogre).unwrap();
    game.state[object].activated_abilities = vec![ActivatedAbilityDefinition {
        tap_source: true,
        effects: vec![Effect::If {
            condition: Condition::SacrificedThisTurnAtLeast { count: 1 },
            then: vec![Effect::add_mana(Mana::parse("GG"))],
            otherwise: vec![Effect::add_mana(Mana::parse("G"))],
        }],
        ..Default::default()
    }];
    game.run_frame(EffectFrame::new(
        PAYER,
        vec![Effect::create_tokens("token.treasure", 1)],
    ));
    let treasure = game
        .iter_battlefield_permanents(PAYER)
        .find(|id| *id != ogre)
        .unwrap();
    pay_during_resolution(&mut game, "GGR", 0);

    assert!(!offer(&game, &activate(ogre, 0)));
    step_action(&mut game, activate(treasure, 0)).unwrap();
    choose_mana(&mut game, "R");
    step_action(&mut game, activate(ogre, 0)).unwrap();
    choose_mana(&mut game, "GG");
    finish_payment(&mut game);
}

#[test]
fn coupled_tile_and_kyoshi_count_creatures_before_sacrifice() {
    for name in ["White Lotus Tile", "The Legend of Kyoshi"] {
        let mut game = payment_game(360, [("Forest", 37), ("Gray Ogre", 2), (name, 1)]);
        game.keep_opening_hands();
        let reader = put(&mut game, name);
        let sacrifice = put(&mut game, "Gray Ogre");
        let object = game.permanent_object(sacrifice).unwrap();
        game.state[object].activated_abilities = vec![ActivatedAbilityDefinition {
            tap_source: true,
            sacrifice_source: true,
            effects: vec![Effect::add_mana(Mana::parse("R"))],
            ..Default::default()
        }];
        // A second Ogre supplies the Tile's shared type. Kyoshi needs a
        // stronger creature for sacrificing the Ogre to change her output.
        let (mana, cost) = if name == "White Lotus Tile" {
            put(&mut game, "Gray Ogre");
            ("GG", "GGR")
        } else {
            game.transform_permanent(reader);
            game.state
                .change_counters(sacrifice, CounterKind::PlusOnePlusOne, 4);
            ("GGGGGG", "GGGGGGR")
        };
        pay_during_resolution(&mut game, cost, 0);

        assert!(offer(&game, &activate(reader, 0)));
        assert!(!offer(&game, &activate(sacrifice, 0)));
        step_action(&mut game, activate(reader, 0)).unwrap();
        choose_mana(&mut game, mana);
        step_action(&mut game, activate(sacrifice, 0)).unwrap();
        choose_mana(&mut game, "R");
        finish_payment(&mut game);
    }
}
