//! Constructed positions for manual mana activations, their callers and their costs.
use crate::{
    agent::action::{Action, AgentError},
    flow::{
        decision::{
            Decision, EffectFrame, ManaProduction, ManaUpfrontActivation, SuspendedResolution,
        },
        game::{ActivationSource, Game, PendingChoice},
        mana::Obligation,
        payment::{CostComponent, Operation, Payment, PaymentProgress},
        turn::StepKind,
    },
    state::{
        ability::{Actor, Condition, Effect, EffectValue, ObjectSelector, PlayerSelector},
        card::{
            ActivatedAbilityDefinition, CardType, DamageRecipient, DamageSource, DamageTransform,
            Keyword, ManaAbility, ReplacementEffect, StaticAbility, StaticScope,
        },
        effect::{EffectDuration, Modification},
        game_object::{PermanentId, PlayerId},
        mana::{Color, Mana, ManaCost},
        permanent::{CounterKind, EntryChoice},
        player::{ManaFloats, PlayerConfig},
        predicate::CardPredicate,
        zone::ZoneType,
    },
    Env,
};
use std::sync::Arc;

fn payment_game<const N: usize>(seed: u64, deck: [(&str, usize); N]) -> Game {
    Game::new(
        vec![
            PlayerConfig::new(
                "payer",
                deck.into_iter()
                    .map(|(name, count)| (name.into(), count))
                    .collect(),
            ),
            PlayerConfig::new("opponent", [("Forest".into(), 40)].into()),
        ],
        seed,
        false,
    )
}

fn announce_activation(game: &mut Game, player: PlayerId, source: PermanentId) {
    let operation = Operation::Activation {
        source: ActivationSource::Permanent(source),
        ability_index: 0,
        targets: vec![],
        target_req_indices: vec![],
        target_divisions: vec![],
        x: 0,
    };
    let payment = Payment {
        owed: game.cost_components(player, &operation),
        operation,
        receipt: Default::default(),
    };
    game.pending_choice = Some(PendingChoice::Pay {
        player,
        payment: Box::new(payment),
    });
}

fn action_index(game: &Game, action: Action) -> usize {
    let actions = &game.action_space().unwrap().actions;
    actions
        .iter()
        .position(|offer| *offer == action)
        .unwrap_or_else(|| panic!("missing {action:?} in {actions:?}"))
}

fn step_action(game: &mut Game, action: Action) -> Result<bool, AgentError> {
    game.step(action_index(game, action))
}

fn cast_only_card(game: &mut Game, player: PlayerId, name: &str) {
    game.scenario_clear_hand(player);
    game.scenario_force_card_in_hand(player, name).unwrap();
    let spell = game.state.zones.zone_objects(ZoneType::Hand, player)[0];
    game.tick().unwrap();
    for _ in 0..4 {
        if game.state.turn.current_step_kind() == StepKind::Main {
            break;
        }
        let holder = game.action_space().unwrap().player.unwrap();
        step_action(game, Action::PassPriority { player: holder }).unwrap();
    }
    assert_eq!(game.state.turn.current_step_kind(), StepKind::Main);
    let cast = game
        .action_space()
        .unwrap()
        .actions
        .iter()
        .position(|action| matches!(action, Action::CastSpell { object, .. } if *object == spell))
        .unwrap_or_else(|| panic!("missing cast for {name}"));
    game.step(cast).unwrap();
}

#[test]
fn mana_upfront_fee_can_remove_the_activated_ability() {
    let player = PlayerId(0);
    let deck = [("Mountain".into(), 39), ("Gray Ogre".into(), 1)].into();
    let mut env = Env::new(340, false, false, false);
    env.reset(vec![
        PlayerConfig::new("payer", deck),
        PlayerConfig::new("opponent", [("Mountain".into(), 40)].into()),
    ])
    .unwrap();
    let game = env.fixture_game_mut();
    game.keep_opening_hands();
    game.mana_floats[player.0] = ManaFloats::All;
    let source = game
        .scenario_force_battlefield(player, "Gray Ogre", true)
        .unwrap();
    game.scenario_force_battlefield(player, "Mountain", true)
        .unwrap();
    let cost = ManaCost::parse("{1}{R}");
    let grant = StaticAbility::continuous(
        StaticScope::This,
        Some(Condition::UnspentManaAtLeast { count: 1 }),
        vec![Modification::GrantActivated(Box::new(
            ActivatedAbilityDefinition {
                mana_cost: cost,
                tap_source: true,
                effects: vec![Effect::add_mana(Mana::parse("BB"))],
                ..Default::default()
            },
        ))],
    );
    game.apply_continuous(
        None,
        player,
        [game.affected_permanent(source).unwrap()],
        Modification::GrantStatic(Box::new(grant)),
        EffectDuration::Indefinite,
    );
    *game.state.mana_pool_mut(player) = Mana::parse("R").into();
    game.activate_mana_ability(player, source, 0).unwrap();
    assert!(matches!(
        game.state.suspended_decision.as_ref().unwrap().decision,
        Decision::ManaUpfront { permanent, ability_index: 0, .. } if permanent == source
    ));
    game.tick().unwrap();
    let pay = action_index(game, Action::PayCost { player });
    let announced = game.state.deterministic_hash();
    let mut fork = game.clone();
    let Decision::ManaUpfront { activation, .. } = &mut fork
        .state
        .suspended_decision_mut()
        .as_mut()
        .unwrap()
        .decision
    else {
        unreachable!();
    };
    activation.output = Effect::add_mana(Mana::parse("B"));
    assert_ne!(fork.state.deterministic_hash(), announced);

    let mut restored = env.snapshot().unwrap().restore(Default::default()).unwrap();
    assert_eq!(
        restored.fixture_game_mut().state.deterministic_hash(),
        announced
    );
    for branch in [&mut env, &mut restored] {
        branch.step(pay as i64).unwrap();
        let game = branch.fixture_game_mut();
        assert_eq!(game.pooled_mana(player).total(), 0);
        assert!(game.characteristics(source).activated_abilities.is_empty());
        assert!(game.state.permanents[source].as_ref().unwrap().tapped);
        assert!(matches!(
            &game.state.suspended_decision.as_ref().unwrap().decision,
            Decision::ManaChoice { choices, .. } if choices == &[Mana::parse("BB")]
        ));
        assert_eq!(
            game.action_space().unwrap().actions,
            [Action::ChooseMode { player, mode: 0 }]
        );
        branch.step(0).unwrap();
        assert_eq!(
            branch.fixture_game_mut().pooled_mana(player),
            Mana::parse("BB")
        );
    }
    assert_eq!(
        env.selected_witness().unwrap(),
        restored.selected_witness().unwrap()
    );
}

#[test]
fn mana_upfront_dynamic_output_reads_the_board_after_the_fee() {
    let player = PlayerId(0);
    let mut game = payment_game(
        341,
        [
            ("Forest", 37),
            ("The Great Henge", 1),
            ("Angel of Vitality", 1),
            ("Gray Ogre", 1),
        ],
    );
    game.keep_opening_hands();
    game.mana_floats[player.0] = ManaFloats::All;
    game.scenario_set_life(player, 22);
    // Keep an extra float available so the restored caller still offers
    // PayCost after the Henge has supplied the full fee.
    game.scenario_force_battlefield(player, "Forest", true)
        .unwrap();
    let henge = game
        .scenario_force_battlefield(player, "The Great Henge", true)
        .unwrap();
    game.scenario_force_battlefield(player, "Angel of Vitality", true)
        .unwrap();
    let source = game
        .scenario_force_battlefield(player, "Gray Ogre", true)
        .unwrap();
    let object = game.permanent_object(source).unwrap();
    let powerful = CardPredicate {
        min_power: Some(4),
        ..CardPredicate::creature()
    };
    game.state[object].activated_abilities = vec![ActivatedAbilityDefinition {
        mana_cost: ManaCost::parse("1G"),
        cost_reduction_per: Some(powerful.clone()),
        tap_source: true,
        effects: vec![Effect::If {
            condition: Condition::ControlsAtLeast {
                count: 1,
                distinct_names: false,
                predicate: powerful,
            },
            then: vec![Effect::add_mana(Mana::parse("GG"))],
            otherwise: vec![Effect::add_mana(Mana::parse("G"))],
        }],
        ..Default::default()
    }];
    assert_eq!(
        game.characteristics(source).activated_abilities[0].mana_choices(),
        Some(vec![Mana::parse("G")])
    );
    game.activate_mana_ability(player, source, 0).unwrap();
    game.tick().unwrap();

    step_action(
        &mut game,
        Action::ActivateAbility {
            player,
            permanent: henge,
            ability_index: 0,
        },
    )
    .unwrap();
    step_action(&mut game, Action::ChooseMode { player, mode: 0 }).unwrap();
    assert_eq!(game.state.players[0].life, 25);
    assert_eq!(
        game.characteristics(source).activated_abilities[0]
            .mana_cost
            .cost,
        ManaCost::parse("G").cost
    );
    step_action(&mut game, Action::PayCost { player }).unwrap();
    assert_eq!(game.pooled_mana(player).total(), 0);
    assert!(matches!(
        &game.state.suspended_decision.as_ref().unwrap().decision,
        Decision::ManaChoice { choices, .. } if choices == &[Mana::parse("GG")]
    ));
    step_action(&mut game, Action::ChooseMode { player, mode: 0 }).unwrap();
    assert_eq!(game.pooled_mana(player), Mana::parse("GG"));
}

#[test]
fn fixed_mana_suffix_reads_life_through_damage_replacement() {
    let player = PlayerId(0);
    let mut game = payment_game(
        343,
        [
            ("Forest", 37),
            ("Tarnished Citadel", 1),
            ("The Great Henge", 1),
            ("Gray Ogre", 1),
        ],
    );
    game.keep_opening_hands();
    game.scenario_set_life(player, 4);
    let citadel = game
        .scenario_force_battlefield(player, "Tarnished Citadel", true)
        .unwrap();
    let henge = game
        .scenario_force_battlefield(player, "The Great Henge", true)
        .unwrap();
    let source = game
        .scenario_force_battlefield(player, "Gray Ogre", true)
        .unwrap();
    let object = game.permanent_object(source).unwrap();
    game.state[object].replacement_effects = vec![ReplacementEffect::ModifyDamage {
        source: DamageSource::any(Actor::You),
        recipient: DamageRecipient::any(Actor::You),
        combat: Some(false),
        transform: DamageTransform::add(EffectValue::LifeGainedThisTurn(PlayerSelector::You)),
    }];
    game.state[object].activated_abilities = vec![ActivatedAbilityDefinition {
        mana_cost: ManaCost::parse("GGR"),
        life_cost: 2,
        effects: vec![Effect::gain_life(1)],
        ..Default::default()
    }];
    announce_activation(&mut game, player, source);
    let delta = |game: &Game, permanent| {
        game.payment_model(player, None)
            .sources
            .iter()
            .find(|source| source.permanent == permanent)
            .unwrap()
            .abilities[0]
            .life_delta
    };
    // Both root probes write only life. The replacement reads life gained
    // this turn through the event path, outside the layer board's reads.
    assert_eq!(delta(&game, citadel), -3);
    assert_eq!(delta(&game, henge), 2);
    let activate = |permanent| Action::ActivateAbility {
        player,
        permanent,
        ability_index: 0,
    };
    game.tick().unwrap();
    let mut henge_first = game.clone();
    // The frozen estimate admits Henge first. That gain changes the
    // Citadel's damage from 3 to 5, leaving only 1 life for the owed fee.
    let error = step_action(&mut henge_first, activate(henge)).unwrap_err();
    assert!(error.0.contains("empty non-terminal action space"));
    assert_eq!(henge_first.state.players[0].life, 6);
    assert_eq!(delta(&henge_first, citadel), -5);

    // The same costs complete if Citadel deals 3 before Henge gains 2.
    step_action(&mut game, activate(citadel)).unwrap();
    assert_eq!(game.state.players[0].life, 1);
    step_action(&mut game, Action::ChooseMode { player, mode: 3 }).unwrap();
    step_action(&mut game, activate(henge)).unwrap();
    step_action(&mut game, Action::ChooseMode { player, mode: 0 }).unwrap();
    assert_eq!(game.state.players[0].life, 1);
    assert!(game.pending_choice.is_none());
    assert_eq!(game.state.stack_objects.len(), 1);
}

#[test]
fn mana_trigger_reads_keyword_changed_by_another_activation() {
    let player = PlayerId(0);
    let mut game = payment_game(
        344,
        [
            ("Forest", 37),
            ("Mountain", 1),
            ("Heartbeat of Spring", 1),
            ("Gray Ogre", 1),
        ],
    );
    game.keep_opening_hands();
    let mountain = game
        .scenario_force_battlefield(player, "Mountain", true)
        .unwrap();
    let heartbeat = game
        .scenario_force_battlefield(player, "Heartbeat of Spring", true)
        .unwrap();
    let ogre = game
        .scenario_force_battlefield(player, "Gray Ogre", true)
        .unwrap();
    let mountain_object = game.permanent_object(mountain).unwrap();
    let heartbeat_object = game.permanent_object(heartbeat).unwrap();
    let ogre_object = game.permanent_object(ogre).unwrap();
    game.state[heartbeat_object].triggered_mana_abilities[0]
        .predicate
        .keyword = Some(Keyword::Trample);
    game.state[mountain_object].activated_abilities = vec![ActivatedAbilityDefinition {
        remove_counters: Some((CounterKind::Trample, 1)),
        effects: vec![Effect::add_mana(Mana::parse("G"))],
        ..Default::default()
    }];
    game.state[ogre_object].activated_abilities = vec![ActivatedAbilityDefinition {
        mana_cost: ManaCost::parse("GRR"),
        effects: vec![Effect::gain_life(1)],
        ..Default::default()
    }];
    game.state
        .change_counters(mountain, CounterKind::Trample, 1);
    announce_activation(&mut game, player, ogre);

    let tap = Action::TapForMana {
        player,
        permanent: mountain,
    };
    let remove_counter = Action::ActivateAbility {
        player,
        permanent: mountain,
        ability_index: 0,
    };

    game.tick().unwrap();
    let mut counter_first = game.clone();
    // The frozen witness counts Heartbeat's second {R} even after it has
    // spent the counter that makes the Mountain match Heartbeat's predicate.
    let error = step_action(&mut counter_first, remove_counter.clone()).unwrap_err();
    assert!(error.0.contains("empty non-terminal action space"));
    assert!(!counter_first
        .characteristics(mountain)
        .keywords
        .has(Keyword::Trample));

    // Tapping first keeps the trigger and funds the same announced activation.
    step_action(&mut game, tap).unwrap();
    assert_eq!(
        game.pending_payment(player).unwrap().receipt.mana().len(),
        2
    );
    step_action(&mut game, remove_counter).unwrap();
    step_action(&mut game, Action::ChooseMode { player, mode: 0 }).unwrap();
    assert!(game.pending_choice.is_none());
    assert_eq!(game.state.stack_objects.len(), 1);
}

#[test]
fn fixed_mana_tap_can_open_a_bonus_choice() {
    let player = PlayerId(0);
    let mut game = payment_game(
        348,
        [
            ("Island", 37),
            ("Mountain", 1),
            ("Heartbeat of Spring", 1),
            ("Sokka, Bold Boomeranger", 1),
        ],
    );
    game.keep_opening_hands();
    game.mana_floats[player.0] = ManaFloats::All;
    let mountain = game
        .scenario_force_battlefield(player, "Mountain", true)
        .unwrap();
    let heartbeat = game
        .scenario_force_battlefield(player, "Heartbeat of Spring", true)
        .unwrap();
    let mountain_object = game.permanent_object(mountain).unwrap();
    let heartbeat_object = game.permanent_object(heartbeat).unwrap();
    game.state[mountain_object].mana_abilities = vec![ManaAbility {
        mana: Mana::parse("UR"),
    }];
    game.state[heartbeat_object].triggered_mana_abilities[0]
        .predicate
        .tapped = true;

    cast_only_card(&mut game, player, "Sokka, Bold Boomeranger");
    action_index(
        &game,
        Action::TapForMana {
            player,
            permanent: mountain,
        },
    );

    let model = game.payment_model(player, None);
    let source = model
        .sources
        .iter()
        .find(|source| source.permanent == mountain)
        .unwrap();
    let produced = Mana::parse("UR");
    assert_eq!(source.fixed.as_ref(), Some(&produced));

    // Follow production to the bonus boundary. The tap makes Heartbeat
    // applicable, and the resolver returns a decision to retain.
    let mut fork = game.clone();
    assert!(fork.tap_permanent(mountain, true));
    let origin = fork.mana_origin(mountain_object);
    fork.state
        .mana_pool_mut(player)
        .add(&produced, Some(origin));
    let tapped = fork.tapped_for_mana(mountain);
    let bonuses = fork.production_bonuses(player, &tapped);
    let decision =
        fork.resolve_mana_bonuses(player, mountain_object, &produced, bonuses, None, None);
    assert!(matches!(
        decision.as_deref(),
        Some(Decision::ManaChoice {
            player: chooser,
            choices,
            production: ManaProduction::Bonus { source, .. },
            ..
        }) if *chooser == player && *source == heartbeat_object
            && choices == &[Mana::parse("U"), Mana::parse("R")]
    ));
}

#[test]
fn repeatable_mana_suffix_uses_the_stack() {
    let player = PlayerId(0);
    let mut game = payment_game(345, [("Gray Ogre", 1), ("Forest", 39)]);
    game.keep_opening_hands();
    game.mana_floats[player.0] = ManaFloats::All;
    let permanent = game
        .scenario_force_battlefield(player, "Gray Ogre", true)
        .unwrap();
    let object = game.permanent_object(permanent).unwrap();
    game.state[object].activated_abilities = vec![ActivatedAbilityDefinition {
        tap_source: true,
        effects: vec![
            Effect::add_mana(Mana::parse("G")),
            Effect::create_tokens("token.clue", 1),
            Effect::untap(ObjectSelector::This),
        ],
        ..Default::default()
    }];
    assert!(!game.characteristics(permanent).activated_abilities[0].is_mana_ability());
    assert!(!game.can_pay_with_mana_abilities(player, &ManaCost::parse("U"), None));
    game.tick().unwrap();

    let activate = Action::ActivateAbility {
        player,
        permanent,
        ability_index: 0,
    };
    assert!(!step_action(&mut game, activate).unwrap());
    assert_eq!(game.state.stack_objects.len(), 1);
    assert_eq!(game.pooled_mana(player), Mana::default());
    assert!(game.state.permanents[permanent].as_ref().unwrap().tapped);
}

#[test]
fn counter_cost_cannot_be_replenished_by_another_counter_only_mana_ability() {
    let green = || vec![Effect::add_mana(Mana::parse("G"))];
    let plus = (CounterKind::PlusOnePlusOne, 1);
    let charge = (CounterKind::Charge, 1);
    for (remove, add) in [(None, plus), (Some(plus), charge), (Some(charge), plus)] {
        let ability = ActivatedAbilityDefinition {
            remove_counters: remove,
            add_counters: Some(add),
            effects: green(),
            ..Default::default()
        };
        assert!(!ability.is_mana_ability());
        assert!(ability.mana_choices().is_none());
    }
    let ability = ActivatedAbilityDefinition {
        remove_counters: Some(plus),
        effects: green(),
        ..Default::default()
    };
    assert!(ability.is_mana_ability());
}

fn counter_replacement_mana_setup(output: Effect) -> (Env, PermanentId) {
    let player = PlayerId(0);
    let caradora = "Caradora, Heart of Alacria";
    let mut env = Env::new(347, false, false, false);
    env.reset(vec![
        PlayerConfig::new(
            "payer",
            [
                ("Forest".into(), 35),
                (caradora.into(), 1),
                ("Doubling Season".into(), 1),
                ("Gray Ogre".into(), 2),
                ("White Lotus Tile".into(), 1),
            ]
            .into(),
        ),
        PlayerConfig::new("opponent", [("Forest".into(), 40)].into()),
    ])
    .unwrap();
    let game = env.fixture_game_mut();
    game.keep_opening_hands();
    game.scenario_force_battlefield(player, caradora, true)
        .unwrap();
    let season = game
        .scenario_force_battlefield(player, "Doubling Season", true)
        .unwrap();
    let season = game.permanent_object(season).unwrap();
    // Let Doubling Season apply to costs too, so changing the order
    // changes the counter total.
    for effect in &mut game.state[season].replacement_effects {
        if let ReplacementEffect::ModifyCounters { effects_only, .. } = effect {
            *effects_only = false;
        }
    }
    let permanent = game
        .scenario_force_battlefield(player, "Gray Ogre", true)
        .unwrap();
    let object = game.permanent_object(permanent).unwrap();
    game.state[object].activated_abilities = vec![ActivatedAbilityDefinition {
        tap_source: true,
        add_counters: Some((CounterKind::PlusOnePlusOne, 1)),
        effects: vec![output],
        ..Default::default()
    }];
    game.mana_floats[player.0] = ManaFloats::All;
    (env, permanent)
}

#[test]
fn mana_counter_cost_can_require_replacement_order() {
    let player = PlayerId(0);
    let fixed = Effect::add_mana(Mana::parse("G"));
    let dynamic = Effect::If {
        condition: Condition::ControlsAtLeast {
            count: 1,
            distinct_names: false,
            predicate: CardPredicate {
                min_power: Some(6),
                ..CardPredicate::creature()
            },
        },
        then: vec![Effect::add_mana(Mana::parse("GG"))],
        otherwise: vec![fixed.clone()],
    };
    for (output, depends_on_counters) in [(fixed, false), (dynamic, true)] {
        let (mut env, permanent) = counter_replacement_mana_setup(output);
        let game = env.fixture_game_mut();
        game.tick().unwrap();
        let activate = action_index(
            game,
            Action::ActivateAbility {
                player,
                permanent,
                ability_index: 0,
            },
        );
        env.step(activate as i64).unwrap();

        let game = env.fixture_game_mut();
        assert!(matches!(
            &game.state.suspended_decision.as_ref().unwrap().decision,
            Decision::OrderReplacement { player: chooser, candidates }
                if *chooser == player && candidates.len() == 2
        ));
        assert_eq!(
            game.action_space().unwrap().actions,
            [
                Action::ChooseMode { player, mode: 0 },
                Action::ChooseMode { player, mode: 1 },
            ]
        );
        assert_eq!(game.pooled_mana(player), Mana::default());
        let pending = game.state.deterministic_hash();
        let snapshot = env.snapshot().unwrap();
        let mut totals = Vec::new();
        for mode in [0, 1] {
            let mut restored = snapshot.restore(Default::default()).unwrap();
            let game = restored.fixture_game_mut();
            assert_eq!(game.state.deterministic_hash(), pending);
            let source = game.state.permanents[permanent].as_ref().unwrap();
            assert!(source.tapped);
            assert_eq!(source.counter_count(CounterKind::PlusOnePlusOne), 0);

            restored.step(mode).unwrap();
            let game = restored.fixture_game_mut();
            let counters = game.state.permanents[permanent]
                .as_ref()
                .unwrap()
                .counter_count(CounterKind::PlusOnePlusOne);
            totals.push(counters);
            let mana = if depends_on_counters && counters == 4 {
                Mana::parse("GG")
            } else {
                Mana::parse("G")
            };
            assert!(matches!(
                &game.state.suspended_decision.as_ref().unwrap().decision,
                Decision::ManaChoice { choices, .. } if choices == std::slice::from_ref(&mana)
            ));
            assert_eq!(game.pooled_mana(player), Mana::default());
            restored.step(0).unwrap();
            assert_eq!(restored.fixture_game_mut().pooled_mana(player), mana);
            if mode == 0 {
                env.step(mode).unwrap();
                env.step(0).unwrap();
                assert_eq!(
                    env.selected_witness().unwrap(),
                    restored.selected_witness().unwrap()
                );
            }
        }
        totals.sort_unstable();
        assert_eq!(totals, [3, 4]);
    }
}

#[test]
fn mana_sacrifice_cost_commits_before_counting_output() {
    let player = PlayerId(0);
    for chosen_color in [false, true] {
        let mut game = payment_game(348, [("Forest", 38), ("Gray Ogre", 2)]);
        game.keep_opening_hands();
        game.mana_floats[player.0] = ManaFloats::All;
        let permanent = game
            .scenario_force_battlefield(player, "Gray Ogre", true)
            .unwrap();
        game.scenario_force_battlefield(player, "Gray Ogre", true)
            .unwrap();
        let source = game.state.permanents[permanent].as_mut().unwrap();
        source.chosen = chosen_color.then(|| Arc::new(EntryChoice::Color(Color::Red)));
        let object = source.object;
        let amount = EffectValue::BattlefieldMatching {
            predicate: CardPredicate::creature(),
        };
        let output = if chosen_color {
            Effect::AddOneColorManaValue {
                amount,
                chosen_color: true,
                creature_purposes: Vec::new(),
            }
        } else {
            Effect::AddMana {
                mana: Mana::parse("G"),
                amount,
                until_end_of_combat: false,
            }
        };
        let counter = CounterKind::PlusOnePlusOne;
        game.state[object].activated_abilities = vec![ActivatedAbilityDefinition {
            tap_source: true,
            add_counters: Some((counter, 1)),
            sacrifice_source: true,
            effects: vec![output],
            ..Default::default()
        }];
        game.tick().unwrap();
        step_action(
            &mut game,
            Action::ActivateAbility {
                player,
                permanent,
                ability_index: 0,
            },
        )
        .unwrap();
        assert!(game.permanent_object(permanent).is_none());
        assert_eq!(game.object_lki(object).unwrap().counters.get(counter), 1);
        let mana = Mana::parse(if chosen_color { "R" } else { "G" });
        assert!(matches!(
            &game.state.suspended_decision.as_ref().unwrap().decision,
            Decision::ManaChoice { choices, .. } if choices == std::slice::from_ref(&mana)
        ));
        step_action(&mut game, Action::ChooseMode { player, mode: 0 }).unwrap();
        assert_eq!(game.pooled_mana(player), mana);
    }
}

#[test]
fn mana_counter_cost_preserves_enclosing_payment() {
    let player = PlayerId(0);
    for resolving in [false, true] {
        let (mut env, ogre) = counter_replacement_mana_setup(Effect::add_mana(Mana::parse("G")));
        let game = env.fixture_game_mut();
        let ogre_object = game.permanent_object(ogre).unwrap();
        game.state[ogre_object].activated_abilities[0]
            .effects
            .push(Effect::DealDamageToPlayer {
                player: PlayerSelector::You,
                amount: EffectValue::Fixed(1),
            });
        game.state[ogre_object].replacement_effects =
            [DamageTransform::Double, DamageTransform::PreventAll]
                .into_iter()
                .map(|transform| ReplacementEffect::ModifyDamage {
                    source: DamageSource::this(),
                    recipient: DamageRecipient::any(Actor::You),
                    combat: Some(false),
                    transform,
                })
                .collect();
        let outer = game
            .scenario_force_battlefield(player, "White Lotus Tile", true)
            .unwrap();
        let source = game.permanent_object(outer).unwrap();
        let counter = CounterKind::Charge;
        game.state.change_counters(outer, counter, 1);
        game.state[source].activated_abilities = vec![
            ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("G"),
                remove_counters: Some((counter, 1)),
                effects: vec![Effect::add_mana(Mana::parse("RRR"))],
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                remove_counters: Some((counter, 1)),
                effects: vec![Effect::add_mana(Mana::parse("B"))],
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                sacrifice_source: true,
                effects: vec![Effect::add_mana(Mana::parse("B"))],
                ..Default::default()
            },
        ];
        if resolving {
            let mut payment = PaymentProgress::new(
                player,
                vec![
                    CostComponent::mana(ManaCost::parse("2R")),
                    CostComponent::Life(2),
                ],
                Vec::new(),
                Vec::new(),
            );
            payment.committed = true;
            game.continue_resolution(
                EffectFrame::new(player, Vec::new()),
                Some(Decision::Payment(payment)),
            );
            game.tick().unwrap();
        } else {
            cast_only_card(game, player, "Gray Ogre");
        }

        for permanent in [outer, ogre] {
            step_action(
                game,
                Action::ActivateAbility {
                    player,
                    permanent,
                    ability_index: 0,
                },
            )
            .unwrap();
        }
        // The counter batch retains the unpaid caller. The damage suffix
        // also retains that caller through the now-computed ManaChoice.
        for has_output in [false, true] {
            let suspended = game.state.suspended_decision.as_ref().unwrap();
            assert!(matches!(
                suspended.decision,
                Decision::OrderReplacement { .. }
            ));
            assert_eq!(
                suspended
                    .decisions()
                    .any(|decision| matches!(decision, Decision::ManaChoice { .. })),
                has_output,
            );
            let model = game.payment_model(player, None);
            assert!(model
                .sources
                .iter()
                .find(|source| source.permanent == outer)
                .unwrap()
                .abilities
                .is_empty());
            let obligations = game.payment_obligations(player);
            assert!(
                matches!(&obligations[..3], [
                    Obligation::Pay { cost: fee, .. },
                    Obligation::Output { choices, .. },
                    Obligation::Pay { cost: cast, .. }
                ] if fee.cost == ManaCost::parse("G").cost
                    && choices.as_ref() == [Mana::parse("RRR")]
                    && cast.cost == ManaCost::parse("2R").cost
                ),
                "{obligations:?}"
            );
            assert_eq!(
                &obligations[3..],
                if resolving {
                    &[Obligation::PayLife(2)][..]
                } else {
                    &[]
                },
            );
            step_action(game, Action::ChooseMode { player, mode: 0 }).unwrap();
        }

        for _ in 0..2 {
            step_action(game, Action::ChooseMode { player, mode: 0 }).unwrap();
        }
        if resolving {
            step_action(game, Action::PayCost { player }).unwrap();
        }
        assert_eq!(game.pooled_mana(player), Mana::default());
        assert_eq!(
            game.state.players[player.0].life,
            if resolving { 18 } else { 20 }
        );
        assert_eq!(game.state.stack_objects.len(), usize::from(!resolving));
    }
}

#[test]
fn declinable_ward_with_life_allows_floating_mana() {
    let player = PlayerId(0);
    let mut game = payment_game(349, [("Forest", 39), ("Tarnished Citadel", 1)]);
    game.keep_opening_hands();
    game.mana_floats[player.0] = ManaFloats::All;
    let citadel = game
        .scenario_force_battlefield(player, "Tarnished Citadel", true)
        .unwrap();
    let forest = game
        .scenario_force_battlefield(player, "Forest", true)
        .unwrap();
    game.scenario_set_life(player, 4);
    game.state
        .mana_pool_mut(player)
        .add(&Mana::parse("2"), None);
    game.continue_resolution(
        EffectFrame::new(player, Vec::new()),
        Some(Decision::Payment(PaymentProgress::new(
            player,
            vec![
                CostComponent::mana(ManaCost::parse("2")),
                CostComponent::Life(2),
            ],
            Vec::new(),
            Vec::new(),
        ))),
    );
    game.tick().unwrap();
    assert!(game
        .action_space()
        .unwrap()
        .actions
        .contains(&Action::PayCost { player }));
    assert_eq!(game.saved_obligations(player), [Obligation::PayLife(2)]);

    step_action(
        &mut game,
        Action::ActivateAbility {
            player,
            permanent: citadel,
            ability_index: 0,
        },
    )
    .unwrap();
    step_action(&mut game, Action::ChooseMode { player, mode: 1 }).unwrap();

    step_action(
        &mut game,
        Action::TapForMana {
            player,
            permanent: forest,
        },
    )
    .unwrap();
    assert!(!game
        .action_space()
        .unwrap()
        .actions
        .contains(&Action::PayCost { player }));
    step_action(&mut game, Action::Decline { player }).unwrap();
    assert!(game.state.suspended_decision.is_none());
    assert_eq!(game.state.players[player.0].life, 1);
    assert_eq!(game.pooled_mana(player), Mana::parse("2UG"));
}

#[test]
fn pending_mana_activation_reserves_its_counter() {
    let player = PlayerId(0);
    let mut game = payment_game(346, [("Forest", 40)]);
    let permanent = game
        .scenario_force_battlefield(player, "Forest", true)
        .unwrap();
    let object = game.permanent_object(permanent).unwrap();
    let kind = CounterKind::Charge;
    let activation = ActivatedAbilityDefinition {
        mana_cost: ManaCost::parse("U"),
        remove_counters: Some((kind, 1)),
        effects: vec![Effect::add_mana(Mana::parse("GG"))],
        ..Default::default()
    };
    game.state[object].activated_abilities = vec![
        activation.clone(),
        ActivatedAbilityDefinition {
            mana_cost: ManaCost::default(),
            remove_counters: Some((kind, 1)),
            effects: vec![Effect::add_mana(Mana::parse("U"))],
            ..Default::default()
        },
        ActivatedAbilityDefinition {
            sacrifice_source: true,
            effects: vec![Effect::add_mana(Mana::parse("U"))],
            ..Default::default()
        },
    ];
    game.state.change_counters(permanent, kind, 1);
    assert!(!game.can_activate_mana_ability(player, permanent, 0));
    *game.state.suspended_decision_mut() = Some(SuspendedResolution {
        frame: EffectFrame::new(player, Vec::new()),
        decision: Decision::ManaUpfront {
            player,
            permanent,
            ability_index: 0,
            tap_remaining: 0,
            activation: Box::new(ManaUpfrontActivation::new(
                activation.mana_cost.clone(),
                &activation,
            )),
            resume: None,
        },
    });
    assert!(game.payment_model(player, None).sources[0]
        .abilities
        .is_empty());
    assert!(!game.can_activate_mana_ability(player, permanent, 1));
    assert!(!game.can_activate_mana_ability(player, permanent, 2));
    game.state.change_counters(permanent, kind, 1);
    assert!(game.can_activate_mana_ability(player, permanent, 1));
    assert!(!game.can_activate_mana_ability(player, permanent, 2));
}

#[test]
fn sacrificing_treasure_changes_battlefield_mana_conditions() {
    let player = PlayerId(0);
    let mut game = payment_game(342, [("Gray Ogre", 1), ("Forest", 39)]);
    game.keep_opening_hands();
    let treasure = game.create_token("token.treasure", player).unwrap();
    let source = game
        .scenario_force_battlefield(player, "Gray Ogre", true)
        .unwrap();
    let object = game.permanent_object(source).unwrap();
    game.state[object].activated_abilities = vec![ActivatedAbilityDefinition {
        condition: Some(Condition::ControlsAtLeast {
            count: 1,
            distinct_names: false,
            predicate: CardPredicate::card_type(CardType::Artifact),
        }),
        tap_source: true,
        effects: vec![Effect::add_mana(Mana::parse("G"))],
        ..Default::default()
    }];
    let offers_mana = |game: &Game| {
        let _scan = game.payment_scan();
        game.can_activate_mana_ability(player, source, 0)
    };
    assert!(offers_mana(&game));
    game.activate_mana_ability(player, treasure, 0).unwrap();
    assert!(!offers_mana(&game));
}
