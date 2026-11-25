use super::{
    tap_obligations, ManaActivation, ManaSource, Obligation, PaymentModel, PaymentSearch,
    PaymentSearchState, PaymentStep, SourcePolicy, Taps,
};
use crate::cards::content_pack::ContentPack;
use crate::state::{
    game_object::{ObjectId, PermanentId},
    mana::{Mana, ManaCost, ManaRestriction, SpendEffect, SpendingContext},
    payment::How,
};
use std::sync::Arc;

/// `cost` with `split` of its generic that any artifact or creature may tap
/// to pay.
fn waterbend(cost: &ManaCost, split: u8) -> Vec<Obligation> {
    tap_obligations(cost, 0, split)
}

#[test]
fn production_totals_preserve_first_modes_classes_and_trigger_choices() {
    use crate::state::card::TriggeredManaOutput;

    let restricted = Mana::restricted_to(ManaRestriction::only(["Ally"], []), [0, 1, 0, 0, 0, 0]);
    let outputs = vec![
        Mana::parse("UR"),
        restricted.clone(),
        Mana::default(),
        Mana::parse("UR"),
        Mana::parse("G"),
        restricted,
    ];
    for bonuses in [
        vec![],
        vec![TriggeredManaOutput::Fixed(Mana::parse("G"))],
        vec![TriggeredManaOutput::OneProducedType],
        vec![
            TriggeredManaOutput::OneProducedType,
            TriggeredManaOutput::Fixed(Mana::parse("U")),
            TriggeredManaOutput::OneProducedType,
        ],
    ] {
        let mut expected = Vec::new();
        for output in &outputs {
            for bonus in super::bonus_alternatives(output, &bonuses) {
                let mut sum = output.clone();
                sum.add(&bonus);
                if !expected.contains(&sum) {
                    expected.push(sum);
                }
            }
        }
        assert_eq!(
            super::production_alternatives(outputs.clone(), &bonuses),
            expected
        );
    }
    assert_eq!(
        super::production_alternatives(Vec::new(), &[]),
        Vec::<Mana>::new()
    );
}

/// Synthetic announced life fee: the same obligation moves from
/// Waterbend to spending, while a mana caller has its own Output fee.
#[test]
fn announced_payment_reserves_life_across_nested_callers() {
    use crate::flow::{
        decision::{Decision, EffectFrame, SpendCompletion, SuspendedResolution},
        game::{ActivationSource, Game, PendingChoice},
        payment::{CostComponent, Operation, Payment},
    };
    use crate::state::{
        card::ActivatedAbilityDefinition, game_object::PlayerId, player::PlayerConfig,
    };
    use std::collections::BTreeMap;

    let player = PlayerId(0);
    let deck = BTreeMap::from([("Gray Ogre".into(), 8), ("Sunbaked Canyon".into(), 8)]);
    let mut game = Game::new(
        vec![
            PlayerConfig::new("p0", deck.clone()),
            PlayerConfig::new("p1", deck),
        ],
        350,
        true,
    );
    game.keep_opening_hands();
    let source = game
        .scenario_force_battlefield(player, "Gray Ogre", true)
        .unwrap();
    let canyon = game
        .scenario_force_battlefield(player, "Sunbaked Canyon", true)
        .unwrap();
    let object = game.state.permanents[source].as_ref().unwrap().object;
    game.state[object].activated_abilities = vec![ActivatedAbilityDefinition {
        mana_cost: ManaCost::parse("1"),
        waterbend: true,
        tap_source: true,
        life_cost: 2,
        ..Default::default()
    }];
    let operation = Operation::Activation {
        source: ActivationSource::Permanent(source),
        ability_index: 0,
        targets: vec![],
        x: 0,
    };
    let payment = Payment {
        owed: game.cost_components(player, &operation),
        operation,
        receipt: Default::default(),
    };
    assert_eq!(payment.help().waterbend, 1);
    assert_eq!(
        payment.owed[1..],
        [CostComponent::Life(2), CostComponent::Source(How::Tapped)]
    );
    for life in [2, 4] {
        game.scenario_set_life(player, life);
        game.pending_choice = Some(PendingChoice::Pay {
            player,
            payment: Box::new(payment.clone()),
        });
        assert_eq!(game.can_activate_mana_ability(player, canyon, 0), life == 4);
        // The life fee remains reserved after the announcement moves
        // into spending, including when a nested upfront owns that caller.
        game.pending_choice = None;
        let announced = Decision::ManaSpend {
            player,
            cost: ManaCost::parse("1"),
            spent: Default::default(),
            completion: Box::new(SpendCompletion::Announced(payment.clone())),
        };
        for decision in [
            announced.clone(),
            Decision::ManaUpfront {
                player,
                permanent: canyon,
                ability_index: 0,
                tap_remaining: 0,
                resume: Some(Box::new(announced)),
            },
        ] {
            *game.state.suspended_decision_mut() = Some(SuspendedResolution {
                frame: EffectFrame::new(player, vec![]),
                decision,
            });
            let model = game.payment_model(player, None);
            assert_eq!(model.life, (life - 2) as u32);
            let source = model
                .sources
                .iter()
                .find(|s| s.permanent == source)
                .unwrap();
            assert!(!source.waterbend);
            let payable = model.payable(
                Mana::default(),
                game.payment_obligations(player),
                SourcePolicy::All,
            );
            assert_eq!(payable, life == 4);
        }
        *game.state.suspended_decision_mut() = None;
    }
    // Execute the nearby payable case: Canyon's fee commits before its
    // color choice; the announced fee is still owed, exactly once.
    game.pending_choice = Some(PendingChoice::Pay {
        player,
        payment: Box::new(payment),
    });
    game.activate_mana_ability(player, canyon, 0).unwrap();
    assert_eq!(game.state.players[0].life, 3);
    assert!(game.state.permanents[canyon].as_ref().unwrap().tapped);
    assert!(game.mana_choice_completes_payment(player, &Mana::parse("R")));
    let mut fork = game.clone();
    for candidate in [&mut game, &mut fork] {
        candidate
            .execute_action(&crate::agent::action::Action::ChooseMode { player, mode: 0 })
            .unwrap();
        candidate.finish_action_step();
        assert!(candidate.pending_choice.is_none());
        assert!(candidate.state.suspended_decision.is_none());
        assert_eq!(candidate.state.players[0].life, 1);
        assert_eq!(candidate.state.players[0].mana_pool, Mana::default());
        assert!(candidate.state.permanents[source].as_ref().unwrap().tapped);
        assert_eq!(candidate.state.stack_objects.len(), 0);
    }
    assert_eq!(
        crate::search_state::witness(&game),
        crate::search_state::witness(&fork)
    );
}

#[test]
fn smart_tap_prunes_surplus_and_preserves_mana_options() {
    use crate::flow::game::Game;
    use crate::state::{game_object::PlayerId, player::PlayerConfig};
    use std::collections::BTreeMap;

    let player = PlayerId(0);
    let board = |lands: &[&str], creature: Option<&str>, hand: Option<&str>| {
        let deck = BTreeMap::from([
            ("Mountain".into(), 8),
            ("Forest".into(), 8),
            ("Lightning Bolt".into(), 4),
            ("Llanowar Elves".into(), 4),
            ("Shivan Dragon".into(), 4),
        ]);
        let mut game = Game::new(
            vec![
                PlayerConfig::new("p0", deck.clone()),
                PlayerConfig::new("p1", deck),
            ],
            347,
            true,
        );
        game.scenario_clear_hand(player);
        let lands: Vec<PermanentId> = lands
            .iter()
            .map(|land| game.scenario_force_battlefield(player, land, true).unwrap())
            .collect();
        if let Some(creature) = creature {
            game.scenario_force_battlefield(player, creature, true)
                .unwrap();
        }
        if let Some(card) = hand {
            game.scenario_force_card_in_hand(player, card).unwrap();
        }
        (game, lands)
    };
    // The lands tapped by paying `cost`, by position.
    let tapped = |game: &Game, lands: &[PermanentId], cost: &str| {
        let mut game = game.clone();
        game.produce_mana(player, &ManaCost::parse(cost)).unwrap();
        let tapped: Vec<usize> = (0..lands.len())
            .filter(|index| {
                game.state.permanents[lands[*index]]
                    .as_ref()
                    .unwrap()
                    .tapped
            })
            .collect();
        tapped
    };

    // The first witness remains in source order.
    let (game, lands) = board(
        &["Forest", "Mountain", "Forest"],
        None,
        Some("Lightning Bolt"),
    );
    let model = game.payment_model(player, None);
    let witness = model
        .plan(Mana::default(), vec![pay("1G")], SourcePolicy::FixedOnly)
        .unwrap();
    assert_eq!(
        witness,
        [
            PaymentStep::Fixed(lands[0]),
            PaymentStep::Fixed(lands[1]),
            PaymentStep::Spend
        ]
    );
    // The chosen payment keeps the Mountain for Lightning Bolt.
    assert_eq!(tapped(&game, &lands, "1G"), vec![0, 2]);
    let probe = game.payment_followups(player, &ManaCost::parse("1G"), &model);
    assert_eq!(
        (probe.followups.len(), probe.hand, probe.instant),
        (1, 1, 1)
    );

    // Mountain, Mountain, Forest for {1}{G}: two taps instead of three.
    let (game, lands) = board(&["Mountain", "Mountain", "Forest"], None, None);
    assert_eq!(tapped(&game, &lands, "1G"), vec![0, 2]);

    // Shivan Dragon's {R} ability is a follow-up like a card in hand.
    let (game, lands) = board(
        &["Forest", "Mountain", "Forest"],
        Some("Shivan Dragon"),
        None,
    );
    let model = game.payment_model(player, None);
    let probe = game.payment_followups(player, &ManaCost::parse("1G"), &model);
    assert_eq!((probe.followups.len(), probe.hand), (1, 0));
    assert_eq!(probe.tapping, [None]);
    assert_eq!(tapped(&game, &lands, "1G"), [0, 2]);

    // One of each kept either way: the card in hand beats the ability.
    let (game, lands) = board(
        &["Forest", "Mountain"],
        Some("Shivan Dragon"),
        Some("Llanowar Elves"),
    );
    assert_eq!(tapped(&game, &lands, "1"), [1]);
}

/// The kind of a source breaks ties among payments that keep the same
/// follow-ups: a basic land pays before Treetop Village or Llanowar
/// Elves, which pay when that keeps a card in hand castable.
#[test]
fn smart_tap_prefers_basic_lands_until_a_follow_up_needs_them() {
    use super::autotapper::Reluctance;
    use crate::flow::game::Game;
    use crate::state::{game_object::PlayerId, player::PlayerConfig};
    use std::collections::BTreeMap;

    let player = PlayerId(0);
    let board = |sources: &[&str], hand: Option<&str>| {
        let deck = BTreeMap::from([
            ("Mountain".into(), 8),
            ("Forest".into(), 8),
            ("Lightning Bolt".into(), 4),
            ("Llanowar Elves".into(), 4),
            ("Treetop Village".into(), 4),
            ("Sol Ring".into(), 4),
            ("Secret Tunnel".into(), 4),
        ]);
        let mut game = Game::new(
            vec![
                PlayerConfig::new("p0", deck.clone()),
                PlayerConfig::new("p1", deck),
            ],
            347,
            true,
        );
        game.scenario_clear_hand(player);
        let sources: Vec<PermanentId> = sources
            .iter()
            .map(|name| game.scenario_force_battlefield(player, name, true).unwrap())
            .collect();
        if let Some(card) = hand {
            game.scenario_force_card_in_hand(player, card).unwrap();
        }
        (game, sources)
    };
    // The sources tapped by paying `cost`, by position.
    let tapped = |game: &Game, sources: &[PermanentId], cost: &str| {
        let mut game = game.clone();
        game.produce_mana(player, &ManaCost::parse(cost)).unwrap();
        (0..sources.len())
            .filter(|index| {
                game.state.permanents[sources[*index]]
                    .as_ref()
                    .unwrap()
                    .tapped
            })
            .collect::<Vec<usize>>()
    };

    let (game, sources) = board(
        &[
            "Llanowar Elves",
            "Sol Ring",
            "Treetop Village",
            "Secret Tunnel",
            "Forest",
        ],
        None,
    );
    let model = game.payment_model(player, None);
    assert_eq!(
        game.source_reluctance(&model),
        [
            Reluctance::Creature,
            Reluctance::Permanent,
            Reluctance::UtilityLand,
            Reluctance::UtilityLand,
            Reluctance::BasicLand,
        ]
    );
    // Tap the Forest first, then the
    // colorless Secret Tunnel ahead of the green Treetop Village.
    assert_eq!(tapped(&game, &sources, "1"), [4]);
    assert_eq!(tapped(&game, &sources, "2"), [3, 4]);
    assert_eq!(tapped(&game, &sources, "3"), [2, 3, 4]);

    // A generic pip with nothing in hand: the Mountain. With Lightning
    // Bolt in hand the Mountain stays for it and the other source pays.
    for other in ["Treetop Village", "Llanowar Elves"] {
        for order in [[other, "Mountain"], ["Mountain", other]] {
            let mountain = usize::from(order[1] == "Mountain");
            let (game, sources) = board(&order, None);
            assert_eq!(tapped(&game, &sources, "1"), [mountain]);
            let (game, sources) = board(&order, Some("Lightning Bolt"));
            assert_eq!(tapped(&game, &sources, "1"), [1 - mountain]);
        }
    }

    // Mana floating in the pool is spent before a source is tapped.
    let (mut game, sources) = board(&["Forest", "Forest", "Forest"], None);
    game.state
        .mana_pool_mut(player)
        .add(&Mana::parse("G"), None);
    assert_eq!(tapped(&game, &sources, "1G"), [0]);
    assert_eq!(tapped(&game, &sources, "G"), Vec::<usize>::new());
    game.state
        .mana_pool_mut(player)
        .add(&Mana::parse("G"), None);
    assert_eq!(tapped(&game, &sources, "1G"), Vec::<usize>::new());
}

thread_local! {
    /// Questions this thread answered by searching rather than directly;
    /// the debug parity searches beside direct answers are not counted.
    pub(in crate::flow::mana) static SEARCHED: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

pub(in crate::flow::mana) fn tapping_ability(outputs: Vec<Mana>) -> ManaActivation {
    ManaActivation {
        tap_source: true,
        tap_count: 0,
        tap_candidates: Arc::new(Vec::new()),
        limited: false,
        sacrifice_source: false,
        ability_index: 0,
        cost: ManaCost::default(),
        outputs: outputs.into(),
        life_cost: 0,
    }
}

// Payment models take distinct appearances from the ordinary allocator.
fn payment_object(index: usize) -> ObjectId {
    static OBJECTS: std::sync::LazyLock<Vec<ObjectId>> = std::sync::LazyLock::new(|| {
        let mut zones = crate::state::zone::ZoneManager::default();
        (0..128).map(|_| zones.allocate()).collect()
    });
    OBJECTS[index]
}

pub(in crate::flow::mana) fn source(
    index: usize,
    fixed: Option<&str>,
    abilities: Vec<ManaActivation>,
) -> ManaSource {
    ManaSource {
        permanent: PermanentId(index),
        object: payment_object(index),
        untapped: true,
        fixed: fixed.map(Mana::parse),
        abilities,
        keyword: false,
        waterbend: false,
    }
}

fn forests(count: usize) -> PaymentModel<'static> {
    PaymentModel {
        cache: None,
        sources: Arc::new(
            (0..count)
                .map(|index| source(index, Some("G"), Vec::new()))
                .collect::<Vec<_>>(),
        ),
        life: 20,
    }
}

fn pay(text: &str) -> Obligation {
    Obligation::mana(ManaCost::parse(text))
}

fn search_state(model: &PaymentModel, obligations: Vec<Obligation>) -> PaymentSearchState {
    PaymentSearchState {
        resources: model.resources(),
        pool: Mana::default(),
        life: model.life,
        obligations,
        prefix: None,
    }
}

/// Compare full concrete witnesses with the identity-keyed DFS on small
/// boards; on large boards fail during exploration, before a stall can grow.
/// A search holds up to `GROWTH` identity states before it classifies its
/// sources, so a smaller limit binds from there; zero asserts that the
/// answer needs no state expansion.
fn bounded_search(
    model: &PaymentModel,
    state: PaymentSearchState,
    policy: SourcePolicy,
    limit: usize,
    compare: bool,
) -> Option<Vec<PaymentStep>> {
    let mut search = PaymentSearch::new(&state.obligations);
    search.max_states = Some(if limit == 0 {
        0
    } else {
        limit.max(PaymentSearch::GROWTH)
    });
    let path = model.witness(state.clone(), policy, &mut search);
    if compare {
        // No classes: identity keys throughout.
        let mut identities = PaymentSearch {
            groups: Some(Vec::new()),
            gross_bound_only: true,
            max_states: Some(20_000),
            ..Default::default()
        };
        let original = model.witness(state, policy, &mut identities);
        assert_eq!(path, original);
    }
    path
}

#[test]
fn payment_shared_outputs_keep_cached_witness_after_source_change() {
    let cache = std::cell::RefCell::new(super::PaymentCache::default());
    let mut model = PaymentModel {
        cache: Some(&cache),
        sources: Arc::new(vec![modal_source(0, &["U", "R"], "")]),
        life: 20,
    };
    let original = model.sources.clone();
    let blue = model.plan(Mana::default(), vec![pay("U")], SourcePolicy::All);
    assert!(blue.is_some());
    let ability = &mut Arc::make_mut(&mut model.sources)[0].abilities[0];
    let mut updated = ability.outputs.to_vec();
    updated[0] = Mana::parse("G");
    ability.outputs = updated.into();
    assert!(model
        .plan(Mana::default(), vec![pay("U")], SourcePolicy::All)
        .is_none());
    assert!(model
        .plan(Mana::default(), vec![pay("G")], SourcePolicy::All)
        .is_some());
    model.sources = original;
    assert_eq!(
        model.plan(Mana::default(), vec![pay("U")], SourcePolicy::All),
        blue
    );
}

#[test]
fn payment_cache_keeps_selected_taps_and_exact_mutations() {
    let cache = std::cell::RefCell::new(super::PaymentCache::default());
    let mut model = PaymentModel {
        cache: Some(&cache),
        sources: Arc::new(vec![source(0, Some("U"), vec![]), source(1, None, vec![])]),
        life: 3,
    };
    let obligations = vec![
        Obligation::Tap {
            candidates: vec![PermanentId(0), PermanentId(1)].into(),
            remaining: 1,
        },
        pay("U"),
    ];
    // These queries have identical unmodified obligations and models.
    // Only the selected tap differs; one spends the sole blue producer.
    for _ in 0..2 {
        assert!(model
            .plan_after_tap(Mana::default(), obligations.clone(), PermanentId(0))
            .is_none());
        assert_eq!(
            model.plan_after_tap(Mana::default(), obligations.clone(), PermanentId(1)),
            Some(vec![PaymentStep::Fixed(PermanentId(0)), PaymentStep::Spend])
        );
    }
    assert_eq!(cache.borrow().entries.len(), 2);
    for mutation in 0..6 {
        match mutation {
            0 => Arc::make_mut(&mut model.sources)[0].untapped = false,
            1 => {
                Arc::make_mut(&mut model.sources)[0].fixed = Some(Mana::restricted_to(
                    ManaRestriction::only([], ["Ally"]),
                    [0, 1, 0, 0, 0, 0],
                ))
            }
            2 => Arc::make_mut(&mut model.sources)[0].untapped = true,
            3 => model.life = 0,
            4 => Arc::make_mut(&mut model.sources).reverse(),
            _ => model.life = 3,
        }
        for pool in [Mana::default(), Mana::parse("U")] {
            for tail in [
                vec![],
                vec![pay("1")],
                vec![Obligation::Output {
                    choices: vec![Mana::parse("U")].into(),
                    life_cost: 1,
                }],
            ] {
                let mut owed = vec![pay("U")];
                owed.extend(tail);
                for policy in [
                    SourcePolicy::All,
                    SourcePolicy::FixedOnly,
                    SourcePolicy::AutomaticPrefix,
                ] {
                    let expected = model.search(pool.clone(), owed.clone(), policy);
                    for _ in 0..2 {
                        assert_eq!(model.plan(pool.clone(), owed.clone(), policy), expected);
                    }
                }
            }
        }
    }
    assert_eq!(cache.borrow().entries.len(), 64);
    let fork = cache.borrow().clone();
    cache.borrow_mut().entries.clear();
    assert_eq!(fork.entries.len(), 64);
    // FIFO retention remains bounded across many distinct life queries.
    for life in 0..100 {
        model.life = life;
        model.plan(Mana::default(), vec![pay("U"), pay("G")], SourcePolicy::All);
    }
    let cache = cache.borrow();
    assert_eq!(cache.entries.len(), 64);
    assert_eq!(cache.entries.front().unwrap().life, 36);
    assert_eq!(cache.entries.back().unwrap().life, 99);
}

#[test]
fn direct_payment_matches_the_search_on_fixed_sources() {
    use rand::{seq::SliceRandom, Rng, SeedableRng};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(339);
    let lesson = || ManaRestriction::spell(["Lesson"]);
    // Empty and unusable outputs are worth nothing to a non-Lesson cost,
    // which is where a waterbend tap beats a mana tap.
    let outputs = [
        None,
        Some(Mana::default()),
        Some(Mana::parse("G")),
        Some(Mana::parse("U")),
        Some(Mana::parse("UG")),
        Some(Mana::parse("C")),
        Some(Mana::restricted_to(lesson(), [0, 1, 0, 0, 0, 0])),
        Some(Mana::restricted_to(lesson(), [0, 0, 0, 0, 1, 1])),
    ];
    let pools = [
        Mana::default(),
        Mana::parse("U"),
        Mana::parse("GC"),
        Mana::restricted_to(lesson(), [0, 1, 0, 0, 1, 0]),
    ];
    let costs = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "U",
        "G",
        "UU",
        "1U",
        "2U",
        "2G",
        "3G",
        "UG",
        "{U/G}",
        "1{U/G}{U/G}",
        "C",
        "3UG",
    ];
    let purposes = [
        SpendingContext::default(),
        SpendingContext::cast(["Lesson"], []),
    ];
    let (mut direct, mut payable, mut waterbent) = (0, 0, 0);
    for _ in 0..20_000 {
        let mut model = PaymentModel {
            cache: None,
            sources: Arc::new(
                (0..rng.gen_range(0..7))
                    .map(|index| {
                        let untapped = rng.gen_bool(0.85);
                        ManaSource {
                            permanent: PermanentId(index),
                            object: payment_object(index),
                            untapped,
                            fixed: outputs.choose(&mut rng).unwrap().clone(),
                            abilities: Vec::new(),
                            keyword: false,
                            waterbend: untapped && rng.gen_bool(0.4),
                        }
                    })
                    .collect::<Vec<_>>(),
            ),
            life: 20,
        };
        let cost = ManaCost {
            purpose: purposes.choose(&mut rng).unwrap().clone(),
            ..ManaCost::parse(costs.choose(&mut rng).unwrap())
        };
        let split = if rng.gen_bool(0.5) {
            cost.generic().min(3)
        } else {
            rng.gen_range(0..=cost.generic().min(3))
        };
        let mut policy = SourcePolicy::All;
        if split == 0 && rng.gen_bool(0.3) {
            // Automatic production ignores abilities, so they stay direct.
            policy = SourcePolicy::FixedOnly;
            if let Some(source) = Arc::make_mut(&mut model.sources).first_mut() {
                source.abilities = vec![tapping_ability(vec![Mana::parse("U")])];
            }
        }
        let obligations = waterbend(&cost, split);
        let pool = pools.choose(&mut rng).unwrap().clone();
        let question = model
            .direct(&obligations, policy)
            .expect("fixed sources and mana costs are direct");
        let searched = model.search(pool.clone(), obligations.clone(), policy);
        assert_eq!(model.direct_witness(pool.clone(), &question), searched);
        assert_eq!(model.direct_payable(&pool, &question), searched.is_some());
        direct += 1;
        if let Some(path) = searched {
            payable += 1;
            waterbent += usize::from(
                path.iter()
                    .any(|step| matches!(step, PaymentStep::Waterbend(_))),
            );
        }
    }
    // The comparison is vacuous unless both answers and waterbend taps occur.
    assert!(payable > direct / 4 && payable < 3 * direct / 4);
    assert!(
        waterbent > direct / 50,
        "{waterbent} of {direct} ({payable} payable)"
    );
}

#[test]
fn direct_payment_leaves_searched_sources_and_obligations_to_the_search() {
    let cost = ManaCost::parse("1U");
    let mut model = forests(3);
    assert!(model
        .direct(&[pay("1U")], SourcePolicy::AutomaticPrefix)
        .is_none());
    assert!(model
        .direct(&[pay("1"), pay("U")], SourcePolicy::All)
        .is_none());
    assert!(model
        .direct(&waterbend(&cost, 1), SourcePolicy::FixedOnly)
        .is_none());
    let counted = Obligation::Tap {
        candidates: vec![PermanentId(0)].into(),
        remaining: 1,
    };
    assert!(model
        .direct(&[counted, pay("U")], SourcePolicy::All)
        .is_none());
    let output = Obligation::Output {
        choices: vec![Mana::parse("U")].into(),
        life_cost: 0,
    };
    assert!(model
        .direct(&[pay("1"), output], SourcePolicy::All)
        .is_none());
    // A waterbend split spent for another purpose is not one sum.
    let mut other = waterbend(&cost, 1);
    if let Obligation::Pay { cost, .. } = &mut other[1] {
        cost.purpose = SpendingContext::cast(["Lesson"], []);
    }
    assert!(model.direct(&other, SourcePolicy::All).is_none());
    assert!(model
        .direct(&waterbend(&cost, 1), SourcePolicy::All)
        .is_some());

    // A modal land chooses its color and a filter has an upfront cost.
    let modal = tapping_ability(vec![Mana::parse("U"), Mana::parse("R")]);
    let mut filter = tapping_ability(vec![Mana::parse("U")]);
    filter.cost = ManaCost::parse("1");
    for ability in [modal, filter] {
        Arc::make_mut(&mut model.sources).push(source(3, None, vec![ability]));
        assert!(model.direct(&[pay("1U")], SourcePolicy::All).is_none());
        assert!(model
            .direct(&[pay("1U")], SourcePolicy::FixedOnly)
            .is_some());
        // The search still answers, through `plan` and `payable` alike.
        assert!(model.payable(Mana::default(), vec![pay("1U")], SourcePolicy::All));
        let path = model
            .plan(Mana::default(), vec![pay("1U")], SourcePolicy::All)
            .unwrap();
        assert!(path
            .iter()
            .any(|step| matches!(step, PaymentStep::Activate(PermanentId(3), 0))));
        Arc::make_mut(&mut model.sources).pop();
    }
}

#[test]
fn a_payment_scan_keeps_sources_until_its_last_guard_ends() {
    use crate::{
        flow::game::Game,
        state::{game_object::PlayerId, player::PlayerConfig},
    };
    let deck: std::collections::BTreeMap<String, usize> = [("Mountain".into(), 40)].into();
    let mut game = Game::new(
        vec![
            PlayerConfig::new("a", deck.clone()),
            PlayerConfig::new("b", deck),
        ],
        339,
        true,
    );
    game.play();
    let player = PlayerId(0);
    let kept = |game: &Game| game.payment_sources.borrow().sources[player.0].clone();

    // Outside a scan every question builds its own sources.
    let unshared = game.payment_model(player, None).sources;
    assert_eq!(kept(&game), None);
    {
        let _outer = game.payment_scan();
        {
            let _inner = game.payment_scan();
            assert_eq!(game.payment_model(player, None).sources, unshared);
        }
        // The first question's sources outlive the nested computation,
        // and a later question reads them.
        assert_eq!(kept(&game), Some(unshared.clone()));
        assert_eq!(game.payment_model(player, None).sources, unshared);
        // A clone can be written to, so it starts with nothing shared.
        assert_eq!(kept(&game.clone()), None);
    }
    assert_eq!(kept(&game), None);
    game.payment_cache.borrow_mut().entries.clear();
    let binding = crate::search_state::witness(&game);
    game.payment_model(player, None).plan(
        Mana::default(),
        vec![pay("U"), pay("")],
        SourcePolicy::All,
    );
    assert_eq!(game.payment_cache.borrow().entries.len(), 1);
    let fork = game.clone();
    assert_eq!(crate::search_state::witness(&fork), binding);
    assert_eq!(fork.payment_cache.borrow().entries.len(), 1);
    assert_eq!(
        fork.payment_model(player, None).plan(
            Mana::parse("U"),
            vec![pay("U"), pay("")],
            SourcePolicy::All
        ),
        Some(vec![PaymentStep::Spend, PaymentStep::Spend])
    );
    assert_eq!(fork.payment_cache.borrow().entries.len(), 2);
    assert_eq!(game.payment_cache.borrow().entries.len(), 1);
    assert_eq!(crate::search_state::witness(&game), binding);
}

#[test]
fn direct_payment_taps_in_source_order_and_keeps_waterbend_taps_it_needs() {
    // Source order, not usefulness: the Forest ahead of the Islands is tapped.
    let model = PaymentModel {
        cache: None,
        sources: Arc::new(vec![
            source(0, Some("G"), Vec::new()),
            source(1, Some("U"), Vec::new()),
            source(2, Some("U"), Vec::new()),
        ]),
        life: 20,
    };
    let path = model
        .plan(Mana::default(), vec![pay("U")], SourcePolicy::All)
        .unwrap();
    assert_eq!(
        path,
        vec![
            PaymentStep::Fixed(PermanentId(0)),
            PaymentStep::Fixed(PermanentId(1)),
            PaymentStep::Spend,
        ]
    );
    assert!(!model.payable(Mana::default(), vec![pay("UUU")], SourcePolicy::All));

    // {1}{U} with one generic waterbendable: the creature covers the
    // split, so the Island pays {U} and nothing is tapped for {1}.
    let mut creature = source(0, None, Vec::new());
    creature.waterbend = true;
    let model = PaymentModel {
        cache: None,
        sources: Arc::new(vec![creature, source(1, Some("U"), Vec::new())]),
        life: 20,
    };
    let cost = ManaCost::parse("1U");
    let path = model
        .plan(Mana::default(), waterbend(&cost, 1), SourcePolicy::All)
        .unwrap();
    assert_eq!(
        path,
        vec![
            PaymentStep::Fixed(PermanentId(1)),
            PaymentStep::Spend,
            PaymentStep::Waterbend(PermanentId(0)),
            PaymentStep::Spend,
        ]
    );
    assert!(!model.payable(Mana::default(), waterbend(&cost, 0), SourcePolicy::All));
}

#[test]
fn payment_search_bounds_identical_land_subsets_in_automatic_prefixes() {
    for count in [8, 24] {
        let model = forests(count);
        let mut state = search_state(&model, vec![pay(&count.to_string())]);
        state.prefix = Some(count - 1);
        assert!(bounded_search(
            &model,
            state,
            SourcePolicy::AutomaticPrefix,
            count,
            count == 8
        )
        .is_none());
        let path = model
            .plan(
                Mana::default(),
                vec![pay(&count.to_string())],
                SourcePolicy::AutomaticPrefix,
            )
            .unwrap();
        assert_eq!(
            path[..count],
            (0..count)
                .map(|index| PaymentStep::Fixed(PermanentId(index)))
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn payment_search_bounds_modal_and_restricted_bottlenecks() {
    for count in [8, 24] {
        let mut model = forests(count);
        // One Herbalist-shaped source: any color, or two restricted mana.
        Arc::make_mut(&mut model.sources).push(ManaSource {
            permanent: PermanentId(count),
            object: payment_object(count),
            untapped: true,
            fixed: None,
            keyword: false,
            waterbend: true,
            abilities: vec![tapping_ability(vec![
                Mana::parse("U"),
                Mana::parse("R"),
                Mana::restricted_to(ManaRestriction::spell(["Lesson"]), [0, 2, 0, 0, 0, 0]),
            ])],
        });
        let state = search_state(&model, vec![pay("U"), pay("U")]);
        assert!(bounded_search(
            &model,
            state,
            SourcePolicy::All,
            20 * (count + 1),
            count == 8
        )
        .is_none());
        // Fixed-only probes must not enumerate the lands trying to reach
        // a color available solely from a visible modal activation.
        let state = search_state(&model, vec![pay("U")]);
        assert!(bounded_search(
            &model,
            state,
            SourcePolicy::FixedOnly,
            count + 1,
            count == 8
        )
        .is_none());
    }
}

#[test]
fn payment_search_bounds_counted_taps_competing_with_mana() {
    for count in [8, 24] {
        let model = forests(count);
        let state = search_state(
            &model,
            vec![
                Obligation::Tap {
                    candidates: model
                        .sources
                        .iter()
                        .map(|source| source.permanent)
                        .collect::<Vec<_>>()
                        .into(),
                    remaining: count / 2,
                },
                pay(&(count / 2 + 1).to_string()),
            ],
        );
        assert!(bounded_search(&model, state, SourcePolicy::All, count, count == 8).is_none());
    }
}

#[test]
fn payment_search_bounds_seedless_nested_filters() {
    for count in [6, 24] {
        let mut model = forests(count);
        let mut filter = tapping_ability(vec![Mana::parse("U")]);
        filter.cost = ManaCost::parse("U");
        for index in count..count + 2 {
            Arc::make_mut(&mut model.sources).push(source(index, None, vec![filter.clone()]));
        }
        let state = search_state(&model, vec![pay("U")]);
        assert!(bounded_search(
            &model,
            state,
            SourcePolicy::All,
            10 * (count + 1),
            count == 6
        )
        .is_none());
    }
}

#[test]
fn payment_search_preserves_witnesses_and_counted_tap_membership() {
    let mut model = forests(4);
    let mut sacrifice = tapping_ability(vec![Mana::parse("U"), Mana::parse("R")]);
    sacrifice.tap_source = false;
    sacrifice.sacrifice_source = true;
    for source in Arc::make_mut(&mut model.sources) {
        source.abilities.push(sacrifice.clone());
        source.waterbend = true;
    }
    // Only the first land can pay a later counted tap. A failed branch
    // spending it must not hide the successful branch spending land #1.
    let obligations = vec![
        pay("1"),
        Obligation::Tap {
            candidates: vec![PermanentId(0)].into(),
            remaining: 1,
        },
        pay("U"),
    ];
    let state = search_state(&model, obligations);
    let path = bounded_search(&model, state, SourcePolicy::All, 1000, true).unwrap();
    assert!(matches!(path[0], PaymentStep::Fixed(PermanentId(1))));
    // Vary initial resources, colors and prefix policy; compare the
    // complete first witness, not only feasibility or a final mana total.
    for mask in 0..16 {
        for text in ["2", "GU", "UU", "UR", "5"] {
            for policy in [
                SourcePolicy::All,
                SourcePolicy::FixedOnly,
                SourcePolicy::AutomaticPrefix,
            ] {
                let mut state = search_state(&model, vec![pay(text)]);
                for index in 0..4 {
                    state.resources.untapped[index] = mask & (1 << index) == 0;
                }
                if policy == SourcePolicy::AutomaticPrefix {
                    state.prefix = Some(2);
                }
                bounded_search(&model, state, policy, 2000, true);
            }
        }
    }
}

#[test]
fn payment_search_keeps_future_tap_membership_and_limited_uses() {
    let mut model = forests(4);
    let mut filter = tapping_ability(vec![Mana::parse("U")]);
    filter.cost = ManaCost::parse("1");
    filter.tap_count = 1;
    filter.tap_candidates = vec![PermanentId(0)].into();
    Arc::make_mut(&mut model.sources).push(source(4, None, vec![filter]));
    let state = search_state(&model, vec![pay("U")]);
    let path = bounded_search(&model, state, SourcePolicy::All, 200, true).unwrap();
    assert!(matches!(path[0], PaymentStep::Fixed(PermanentId(1))));

    let mut model = forests(3);
    let mut limited = tapping_ability(vec![Mana::parse("U")]);
    limited.tap_source = false;
    limited.limited = true;
    limited.life_cost = 1;
    for source in Arc::make_mut(&mut model.sources) {
        source.abilities.push(limited.clone());
    }
    for mask in 0..8 {
        let mut state = search_state(&model, vec![pay("UU")]);
        state.life = 1; // Output life costs, too, survive canonicalization.
        for index in 0..3 {
            if mask & (1 << index) != 0 {
                state.resources.use_ability(index, 0);
            }
            state.resources.present[index] = index != 2;
        }
        assert!(bounded_search(&model, state.clone(), SourcePolicy::All, 200, true).is_none());
        state.life = 2;
        bounded_search(&model, state, SourcePolicy::All, 200, true);
    }
}

#[test]
fn payment_search_bounds_repeat_activations_sharing_counted_taps() {
    for count in [6, 24] {
        let mut model = forests(count);
        // Two Cabbage Merchant-shaped abilities share the same Food taps.
        for source in Arc::make_mut(&mut model.sources) {
            source.fixed = None;
        }
        let mut counted = tapping_ability(vec![Mana::parse("U")]);
        counted.tap_source = false;
        counted.tap_count = 2;
        counted.tap_candidates = model
            .sources
            .iter()
            .map(|source| source.permanent)
            .collect::<Vec<_>>()
            .into();
        for index in count..count + 2 {
            Arc::make_mut(&mut model.sources).push(source(index, None, vec![counted.clone()]));
        }
        let state = search_state(&model, vec![pay(&"U".repeat(count / 2 + 1))]);
        assert!(bounded_search(&model, state, SourcePolicy::All, 1000, count == 6).is_none());
        let state = search_state(&model, vec![pay(&"U".repeat(count / 2))]);
        assert!(bounded_search(&model, state, SourcePolicy::All, 1000, count == 6).is_some());
    }
}

/// A single tap cannot both produce mana and help waterbend. Modal and
/// restricted outputs still contribute only what this cost can spend.
#[test]
fn payment_bound_does_not_search_subsets_of_alternative_taps() {
    let sources = (0..4)
        .map(|index| ManaSource {
            permanent: PermanentId(index),
            object: payment_object(index),
            untapped: true,
            fixed: Some(Mana::parse("G")),
            abilities: vec![
                tapping_ability(vec![Mana::parse("U"), Mana::parse("R")]),
                tapping_ability(vec![Mana::restricted_to(
                    ManaRestriction::spell(["Lesson"]),
                    [0, 0, 0, 0, 2, 0],
                )]),
            ],
            keyword: false,
            waterbend: true,
        })
        .collect();
    let model = PaymentModel {
        cache: None,
        sources: Arc::new(sources),
        life: 20,
    };
    let mut visited = PaymentSearch::default();
    let state = PaymentSearchState {
        resources: model.resources(),
        pool: Mana::default(),
        life: 20,
        obligations: vec![
            Obligation::mana(ManaCost::default()),
            Obligation::Pay {
                cost: ManaCost::parse("5"),
                taps: Taps::Any,
            },
        ],
        prefix: None,
    };
    assert!(model
        .witness(state, SourcePolicy::All, &mut visited)
        .is_none());
    // Count explored states instead of asserting a host-dependent duration.
    assert!(visited.visited.is_empty());

    let path = model
        .plan(
            Mana::default(),
            vec![Obligation::Pay {
                cost: ManaCost::parse("4"),
                taps: Taps::Any,
            }],
            SourcePolicy::All,
        )
        .unwrap();
    assert_eq!(path.len(), 5);
    for (index, step) in path[..4].iter().enumerate() {
        assert!(matches!(step, PaymentStep::Fixed(id) if *id == PermanentId(index)));
    }
    assert!(matches!(path[4], PaymentStep::Spend));
}

/// Tapping and then sacrificing one object are independent resources.
#[test]
fn payment_bound_keeps_non_tapping_production_after_a_tap() {
    let mut sacrifice = tapping_ability(vec![Mana::parse("U")]);
    sacrifice.tap_source = false;
    sacrifice.sacrifice_source = true;
    let model = PaymentModel {
        cache: None,
        sources: Arc::new(vec![ManaSource {
            permanent: PermanentId(0),
            object: payment_object(0),
            untapped: true,
            fixed: Some(Mana::parse("G")),
            abilities: vec![sacrifice],
            keyword: false,
            waterbend: true,
        }]),
        life: 20,
    };
    let path = model
        .plan(
            Mana::default(),
            vec![Obligation::mana(ManaCost::parse("GU"))],
            SourcePolicy::All,
        )
        .unwrap();
    assert!(matches!(path[0], PaymentStep::Fixed(PermanentId(0))));
    assert!(matches!(path[1], PaymentStep::Activate(PermanentId(0), 0)));
}

#[test]
fn payment_bound_counts_restricted_mana_for_each_pending_purpose() {
    let mut pool = Mana::restricted_to(ManaRestriction::spell(["Lesson"]), [0, 1, 0, 0, 0, 0]);
    pool.add(&Mana::restricted_to(
        ManaRestriction::spell(["Shrine"]),
        [0, 1, 0, 0, 0, 0],
    ));
    let model = PaymentModel {
        cache: None,
        sources: Arc::new(Vec::new()),
        life: 20,
    };
    let obligations = ["Lesson", "Shrine"]
        .map(|subtype| {
            Obligation::mana(ManaCost {
                purpose: SpendingContext::cast([subtype], []),
                ..ManaCost::parse("U")
            })
        })
        .to_vec();
    assert!(model.plan(pool, obligations, SourcePolicy::All).is_some());
}

#[test]
fn payment_bound_counts_spend_effect_and_combat_classes() {
    let mut pool = Mana::of_class(
        SpendEffect::CreatureEntersWithPlusOneCounter {
            without_subtype: "Human".into(),
        }
        .into(),
        [0, 1, 0, 0, 0, 0],
    );
    pool.add(&Mana::parse("R").with_expiry(crate::state::mana::ManaExpiry::EndOfCombat));
    let model = PaymentModel {
        cache: None,
        sources: Arc::new(Vec::new()),
        life: 20,
    };
    assert!(model
        .plan(pool, vec![pay("U"), pay("R")], SourcePolicy::All)
        .is_some());
}

#[test]
fn funded_final_payment_needs_no_unit_search() {
    let model = forests(1);
    let mut pool = Mana::parse("U");
    pool.add(&Mana::parse("R").with_expiry(crate::state::mana::ManaExpiry::EndOfCombat));
    pool.add(&Mana::of_class(
        SpendEffect::CreatureEntersWithPlusOneCounter {
            without_subtype: "Human".into(),
        }
        .into(),
        [2, 0, 0, 0, 0, 0],
    ));
    // Hybrid assignment must backtrack (R for U/R leaves U for U/B),
    // while the generic pip can use the rider. No source needs tapping.
    for policy in [
        SourcePolicy::All,
        SourcePolicy::FixedOnly,
        SourcePolicy::AutomaticPrefix,
    ] {
        let mut state = search_state(&model, vec![pay("{1}{U/R}{U/B}")]);
        state.pool = pool.clone();
        state.prefix = Some(0);
        assert_eq!(
            bounded_search(&model, state, policy, 0, false),
            Some(vec![PaymentStep::Spend])
        );
    }
}

/// Restricted mana can finance a future activation even when it cannot
/// pay the current debt. That activation's output covers the debt instead.
#[test]
fn payment_bound_allows_restricted_mana_to_fund_a_filter() {
    let mut filter = tapping_ability(vec![Mana::parse("U")]);
    filter.cost = ManaCost {
        purpose: SpendingContext::ability(["Ally"]),
        ..ManaCost::parse("1")
    };
    let model = PaymentModel {
        cache: None,
        sources: Arc::new(vec![source(0, None, vec![filter])]),
        life: 20,
    };
    let pool = Mana::restricted_to(ManaRestriction::only([], ["Ally"]), [0, 0, 0, 0, 1, 0]);
    let path = model
        .plan(
            pool,
            vec![Obligation::mana(ManaCost::parse("U"))],
            SourcePolicy::All,
        )
        .unwrap();
    assert!(matches!(path[0], PaymentStep::Activate(PermanentId(0), 0)));
    assert!(matches!(path[1], PaymentStep::Spend));
    assert!(matches!(path[2], PaymentStep::Output(0)));
}

#[test]
fn automatic_prefix_lower_bound_keeps_the_first_witness() {
    let mut model = forests(4);
    Arc::make_mut(&mut model.sources).push(modal_source(4, &["U", "R"], "1"));
    Arc::make_mut(&mut model.sources)[0].fixed = Some(Mana::parse("UU"));
    Arc::make_mut(&mut model.sources)[1].waterbend = true;
    for tapped in [false, true] {
        Arc::make_mut(&mut model.sources)[0].untapped = !tapped;
        for pool in [Mana::default(), Mana::parse("U"), Mana::parse("2")] {
            for cost in ["U", "3U", "5U", "R", "6"] {
                for obligations in [
                    vec![pay(cost)],
                    waterbend(&ManaCost::parse(cost), 2),
                    vec![pay(cost), pay("G")],
                ] {
                    let reference = (0..=model.sources.len()).find_map(|budget| {
                        let mut state = search_state(&model, obligations.clone());
                        state.pool = pool.clone();
                        state.prefix = Some(budget);
                        let mut search = PaymentSearch::new(&obligations);
                        model.witness(state, SourcePolicy::AutomaticPrefix, &mut search)
                    });
                    assert_eq!(
                        model.search(pool.clone(), obligations, SourcePolicy::AutomaticPrefix),
                        reference
                    );
                }
            }
        }
    }
}

#[test]
fn limited_mana_uses_preserve_grouped_witnesses() {
    let abilities = ["U", "R"].map(|output| ManaActivation {
        tap_source: false,
        limited: true,
        ..tapping_ability(vec![Mana::parse(output)])
    });
    let model = PaymentModel {
        cache: None,
        sources: Arc::new(
            (0..3)
                .map(|index| source(index, None, abilities.to_vec()))
                .collect(),
        ),
        life: 20,
    };
    for mask in 0..64 {
        let mut state = search_state(&model, vec![pay("UR")]);
        for source in 0..3 {
            for ability in 0..2 {
                if mask & (1 << (source * 2 + ability)) != 0 {
                    state.resources.use_ability(source, ability);
                }
            }
        }
        let mut identities = PaymentSearch {
            groups: Some(vec![]),
            max_states: Some(2000),
            ..Default::default()
        };
        let reference = model.witness(state.clone(), SourcePolicy::All, &mut identities);
        let mut grouped = PaymentSearch::new(&state.obligations);
        grouped.groups = Some(grouped.groups(&model));
        grouped.max_states = Some(2000);
        assert_eq!(
            model.witness(state, SourcePolicy::All, &mut grouped),
            reference
        );
    }
}

#[test]
fn projected_output_bound_keeps_class_maxima_and_saturation() {
    let lesson = ManaRestriction::spell(["Lesson"]);
    let ally = ManaRestriction::spell(["Ally"]);
    let outputs = [
        Mana::parse("UU"),
        Mana::restricted_to(lesson.clone(), [0, 3, 0, 0, 0, 0]),
        Mana::restricted_to(lesson, [0, 1, 0, 4, 0, 0]),
        Mana::restricted_to(ally, [0, 5, 0, 0, 0, 0]),
        Mana::of_class(
            crate::state::mana::ManaClass {
                expiry: crate::state::mana::ManaExpiry::EndOfCombat,
                restriction: None,
                effect: None,
            },
            [0, 250, 0, 0, 0, 0],
        ),
    ];
    for purpose in [
        SpendingContext::Resolving,
        SpendingContext::cast(["Lesson"], []),
        SpendingContext::cast(["Ally"], []),
    ] {
        for text in ["U", "7U", "R", "{U/R}R", "255"] {
            let cost = ManaCost {
                purpose: purpose.clone(),
                ..ManaCost::parse(text)
            };
            for count in 0..=outputs.len() {
                let modes = &outputs[..count];
                let mut reference = Mana::default();
                for mode in modes {
                    reference.max_with(mode);
                }
                let (upper, max_usable) =
                    super::optimistic_supply(modes, |class| class.allows(&cost.purpose));
                let projected = Mana {
                    mana: upper,
                    classes: None,
                };
                assert_eq!(
                    max_usable,
                    modes
                        .iter()
                        .map(|mode| u64::from(mode.usable_total(&cost)))
                        .max()
                        .unwrap_or(0)
                );
                assert_eq!(projected.can_pay(&cost), reference.can_pay(&cost));
                // Adding repeated outputs preserves the same saturation
                // even when their allowed classes differ.
                for mode in &outputs {
                    let (mode_upper, _) =
                        super::optimistic_supply(std::slice::from_ref(mode), |class| {
                            class.allows(&cost.purpose)
                        });
                    let mut combined = projected.clone();
                    combined.add(&Mana {
                        mana: mode_upper,
                        classes: None,
                    });
                    let mut single_reference = reference.clone();
                    single_reference.add(mode);
                    assert_eq!(combined.can_pay(&cost), single_reference.can_pay(&cost));
                }
            }
        }
    }
}

#[test]
fn forced_payment_transitions_keep_complete_witnesses() {
    let model = forests(1);
    let obligations = vec![
        Obligation::Tap {
            candidates: vec![PermanentId(0)].into(),
            remaining: 0,
        },
        pay(""),
        Obligation::Output {
            choices: vec![Mana::parse("U")].into(),
            life_cost: 2,
        },
        pay("1U"),
    ];
    assert_eq!(
        model.plan(Mana::default(), obligations.clone(), SourcePolicy::All),
        Some(vec![
            PaymentStep::Spend,
            PaymentStep::Output(0),
            PaymentStep::Fixed(PermanentId(0)),
            PaymentStep::Spend
        ])
    );
    let mut poor = model;
    poor.life = 1;
    assert!(poor
        .plan(Mana::default(), obligations, SourcePolicy::All)
        .is_none());
}

#[test]
fn payment_net_bound_counts_filter_fees_and_saved_outputs() {
    for output in ["U", "UU", "UUU"] {
        for fee in ["1", "2", "3"] {
            let model = PaymentModel {
                cache: None,
                sources: Arc::new(vec![
                    modal_source(0, &[output], fee),
                    modal_source(1, &["R"], "U"),
                ]),
                life: 3,
            };
            for pool in [Mana::default(), Mana::parse("G"), Mana::parse("GG")] {
                for saved in [false, true] {
                    for cost in ["U", "UU", "UR", "2U"] {
                        let mut obligations = vec![];
                        if saved {
                            obligations.push(Obligation::Output {
                                choices: vec![Mana::parse("G")].into(),
                                life_cost: 1,
                            });
                        }
                        obligations.push(pay(cost));
                        let mut state = search_state(&model, obligations);
                        state.pool = pool.clone();
                        bounded_search(&model, state, SourcePolicy::All, 200, true);
                    }
                }
            }
        }
    }
    let model = PaymentModel {
        cache: None,
        sources: Arc::new(vec![modal_source(0, &["UU"], "2")]),
        life: 3,
    };
    let mut state = search_state(&model, vec![pay("UU")]);
    state.pool = Mana::parse("G");
    assert!(bounded_search(&model, state, SourcePolicy::All, 0, true).is_none());
}

fn modal_source(index: usize, outputs: &[&str], cost: &str) -> ManaSource {
    let mut ability = tapping_ability(outputs.iter().map(|text| Mana::parse(text)).collect());
    ability.cost = ManaCost::parse(cost);
    source(index, None, vec![ability])
}

/// The same search without the color-set bound, as a reference.
fn total_bound_search(
    model: &PaymentModel,
    pool: &Mana,
    obligations: &[Obligation],
) -> Option<Vec<PaymentStep>> {
    let mut reference = PaymentSearch::new(obligations);
    reference.total_bound_only = true;
    reference.gross_bound_only = true;
    let mut state = search_state(model, obligations.to_vec());
    state.pool = pool.clone();
    model.witness(state, SourcePolicy::All, &mut reference)
}

/// One source that makes U or R cannot pay {U}{R}, however many Forests
/// stand beside it: the root is rejected without expanding a state.
#[test]
fn payment_color_bound_rejects_unpayable_colors_without_searching() {
    const ANY: [&str; 5] = ["W", "U", "B", "R", "G"];
    for count in [8, 24] {
        let mut dual = forests(count);
        Arc::make_mut(&mut dual.sources).push(modal_source(count, &["U", "R"], ""));
        let mut filter = forests(count);
        Arc::make_mut(&mut filter.sources).push(modal_source(count, &ANY, "1"));
        let mut any = forests(count);
        Arc::make_mut(&mut any.sources).push(modal_source(count, &ANY, ""));
        for (model, obligations) in [
            (&dual, vec![pay("UR")]),
            (&dual, vec![pay("{U/R}R")]),
            (&filter, vec![pay("UR")]),
            (&any, vec![pay("U"), pay("U")]),
            (&any, vec![pay("1U"), pay("R")]),
            (&any, vec![pay("CU")]),
        ] {
            let state = search_state(model, obligations.clone());
            assert!(
                bounded_search(model, state, SourcePolicy::All, 0, false).is_none(),
                "{obligations:?}"
            );
            if count == 8 {
                assert!(total_bound_search(model, &Mana::default(), &obligations).is_none());
            }
        }
    }
}

/// Payable costs that meet the bound with nothing to spare keep their
/// first witness: hybrid pips, restricted mana for its purpose, a modal
/// two-mana output and a paid filter.
#[test]
fn payment_color_bound_keeps_payable_costs_at_the_bound() {
    let lesson = |text: &str| {
        Obligation::mana(ManaCost {
            purpose: SpendingContext::cast(["Lesson"], []),
            ..ManaCost::parse(text)
        })
    };
    let mut herbalist = modal_source(9, &["U", "R"], "");
    let mut outputs = herbalist.abilities[0].outputs.to_vec();
    outputs.push(Mana::restricted_to(
        ManaRestriction::spell(["Lesson"]),
        [0, 2, 0, 0, 0, 0],
    ));
    herbalist.abilities[0].outputs = outputs.into();
    let cases: Vec<(Vec<ManaSource>, Mana, Vec<Obligation>)> = vec![
        // Two U/R sources and a Forest pay two hybrid pips and green.
        (
            vec![
                modal_source(8, &["U", "R"], ""),
                modal_source(9, &["U", "R"], ""),
            ],
            Mana::default(),
            vec![pay("{U/R}{U/R}G")],
        ),
        // The hybrid pip takes the color the colored pip leaves.
        (
            vec![modal_source(9, &["U", "R"], "")],
            Mana::parse("R"),
            vec![pay("{U/R}R")],
        ),
        // Lesson-only mana in the pool pays a Lesson's blue pip.
        (
            vec![modal_source(9, &["U", "R"], "")],
            Mana::restricted_to(ManaRestriction::spell(["Lesson"]), [0, 1, 0, 0, 0, 0]),
            vec![lesson("UR")],
        ),
        // One tap for two Lesson mana pays both blue pips of a Lesson.
        (
            vec![herbalist.clone()],
            Mana::default(),
            vec![lesson("UUG")],
        ),
        // A filter paid by one Forest makes the only blue mana.
        (
            vec![modal_source(9, &["W", "U", "B", "R", "G"], "1")],
            Mana::default(),
            vec![pay("UG")],
        ),
        // A nested filter cost and its parent share the supply.
        (
            vec![modal_source(9, &["W", "U", "B", "R", "G"], "G")],
            Mana::default(),
            vec![pay("G"), pay("U")],
        ),
    ];
    for (extra, pool, obligations) in cases {
        let mut model = forests(2);
        Arc::make_mut(&mut model.sources).extend(extra);
        let reference = total_bound_search(&model, &pool, &obligations);
        assert!(reference.is_some(), "{obligations:?}");
        let mut state = search_state(&model, obligations.clone());
        state.pool = pool;
        assert_eq!(
            bounded_search(&model, state, SourcePolicy::All, 200, false),
            reference,
            "{obligations:?}"
        );
    }
    // The same two Lesson mana cannot pay a spell that is not a Lesson.
    let mut model = forests(8);
    Arc::make_mut(&mut model.sources).push(herbalist);
    let state = search_state(&model, vec![pay("UUG")]);
    assert!(bounded_search(&model, state, SourcePolicy::All, 0, true).is_none());
}

/// On every small board of a varied palette, the bound returns exactly
/// the reference search's first witness: it rejects nothing payable.
#[test]
fn payment_color_bound_matches_the_search_without_it() {
    let lesson_two = Mana::restricted_to(ManaRestriction::spell(["Lesson"]), [0, 2, 0, 0, 0, 0]);
    let palette = |kind: usize, index: usize| -> ManaSource {
        let mut source = modal_source(index, &[], "");
        match kind {
            0 => source.abilities.clear(),
            1 => source.abilities[0].outputs = vec![Mana::parse("U"), Mana::parse("R")].into(),
            2 => {
                source.abilities[0].outputs =
                    ["W", "U", "B", "R", "G"].map(Mana::parse).to_vec().into();
                source.abilities[0].cost = ManaCost::parse("1");
            }
            3 => {
                source.abilities[0].outputs =
                    vec![Mana::parse("U"), Mana::parse("G"), lesson_two.clone()].into();
                source.waterbend = true;
            }
            4 => {
                // A repeatable-per-turn ability that does not tap.
                source.abilities[0].tap_source = false;
                source.abilities[0].limited = true;
                source.abilities[0].outputs = vec![Mana::parse("R"), Mana::parse("C")].into();
            }
            _ => source.abilities[0].outputs = vec![Mana::parse("UR")].into(),
        }
        source.fixed = match kind {
            0 => Some(Mana::parse("G")),
            4 => Some(Mana::parse("U")),
            _ => None,
        };
        source
    };
    let purposes = [
        SpendingContext::default(),
        SpendingContext::cast(["Lesson"], []),
    ];
    let costs = [
        vec!["UR"],
        vec!["UG"],
        vec!["{U/R}R"],
        vec!["{U/R}{U/G}"],
        vec!["UU1"],
        vec!["CU"],
        vec!["{U/R}{U/R}G"],
        vec!["2UR"],
        vec!["U", "U"],
        vec!["1", "UR"],
        vec!["G", "{U/R}"],
    ];
    let pools = [Mana::default(), Mana::parse("R"), lesson_two.clone()];
    let (mut payable, mut rejected) = (0, 0);
    // Every multiset of up to four sources from the six kinds.
    for code in 0..7_usize.pow(4) {
        let kinds = (0..4)
            .map(|slot| code / 7_usize.pow(slot) % 7)
            .collect::<Vec<_>>();
        if kinds.windows(2).any(|pair| pair[0] > pair[1]) {
            continue;
        }
        let model = PaymentModel {
            cache: None,
            sources: Arc::new(
                kinds
                    .iter()
                    .enumerate()
                    .filter(|(_, kind)| **kind < 6)
                    .map(|(index, kind)| palette(*kind, index))
                    .collect::<Vec<_>>(),
            ),
            life: 20,
        };
        for cost in &costs {
            for purpose in &purposes {
                for pool in &pools {
                    let mut obligations = cost
                        .iter()
                        .map(|text| Obligation::Pay {
                            cost: ManaCost {
                                purpose: purpose.clone(),
                                ..ManaCost::parse(text)
                            },
                            taps: if cost.len() == 1 {
                                Taps::Any
                            } else {
                                Taps::None
                            },
                        })
                        .collect::<Vec<_>>();
                    if cost.len() == 2 {
                        // A started activation: its cost, its output, the parent.
                        obligations.insert(
                            1,
                            Obligation::Output {
                                choices: vec![Mana::parse("U"), Mana::parse("R")].into(),
                                life_cost: 0,
                            },
                        );
                    }
                    let reference = total_bound_search(&model, pool, &obligations);
                    let mut state = search_state(&model, obligations.clone());
                    state.pool = pool.clone();
                    let mut search = PaymentSearch::new(&obligations);
                    let path = model.witness(state, SourcePolicy::All, &mut search);
                    assert_eq!(path, reference, "{kinds:?} {cost:?} {pool:?}");
                    payable += usize::from(path.is_some());
                    rejected += usize::from(path.is_none() && search.visited.is_empty());
                }
            }
        }
    }
    assert!(payable > 1_000 && rejected > 1_000, "{payable} {rejected}");
}

#[test]
fn registered_mana_activations_consume_finite_payment_resources() {
    for definition in ContentPack::default().definitions() {
        for ability in &definition.activated_abilities {
            if ability.mana_choices().is_some() {
                assert!(
                    ability.tap_source
                        || ability.sacrifice_source
                        || ability.is_use_limited()
                        || ability.tap_cost.as_ref().is_some_and(|cost| cost.count > 0),
                    "{} has no finite payment resource cost",
                    definition.name
                );
                assert!(
                    ability.sacrifice.is_none() && !ability.waterbend,
                    "{} needs its additional upfront cost represented",
                    definition.name
                );
            }
        }
    }
}

#[test]
fn purpose_projected_groups_collapse_identically_usable_restrictions_and_preserve_riders() {
    use crate::state::card::CardType;
    use crate::state::mana::{ManaClass, ManaExpiry, SpendPredicate};
    use std::collections::BTreeSet;

    const TYPES: [&str; 16] = [
        "Elf", "Goblin", "Ally", "Dragon", "Human", "Soldier", "Wizard", "Cat", "Angel", "Demon",
        "Spirit", "Zombie", "Beast", "Bird", "Warrior", "Knight",
    ];
    let courtyard = |index: usize, subtype: &str| -> ManaSource {
        let restriction = ManaRestriction::only_predicates(BTreeSet::from([
            SpendPredicate::CastCreatureSubtype(subtype.into()),
            SpendPredicate::ActivateCreatureSubtype(subtype.into()),
        ]));
        let outputs = crate::state::mana::Color::COLORS
            .into_iter()
            .map(|color| {
                let mut amounts = [0; 6];
                amounts[color as usize] = 1;
                Mana::restricted_to(restriction.clone(), amounts)
            })
            .collect();
        source(index, Some("C"), vec![tapping_ability(outputs)])
    };

    // On a non-creature spell with no creature filter, all 16 distinct-type
    // Courtyards behave identically and collapse into one group.
    let model = PaymentModel {
        cache: None,
        sources: Arc::new(
            TYPES
                .iter()
                .enumerate()
                .map(|(i, ty)| courtyard(i, ty))
                .collect(),
        ),
        life: 20,
    };
    let noncreature = Obligation::mana(ManaCost {
        purpose: SpendingContext::cast([], [CardType::Instant]),
        ..ManaCost::parse("UU")
    });
    let mut search = PaymentSearch::new(std::slice::from_ref(&noncreature));
    assert_eq!(
        search.groups_for(&model, std::slice::from_ref(&noncreature)),
        vec![(0..16).collect::<Vec<_>>()]
    );

    // On a Cat creature spell, Cat stays separate while the other 15 collapse.
    let cat_spell = Obligation::mana(ManaCost {
        purpose: SpendingContext::cast(["Cat"], [CardType::Creature]),
        ..ManaCost::parse("1WW")
    });
    let mut cat_search = PaymentSearch::new(std::slice::from_ref(&cat_spell));
    let cat_groups = cat_search.groups_for(&model, std::slice::from_ref(&cat_spell));
    assert_eq!(cat_groups.len(), 1);
    assert_eq!(cat_groups[0].len(), 15);
    assert!(!cat_groups[0].contains(&7));

    // With a Changeling filter ({1} -> any color), all 16 distinct-type
    // Courtyards can pay the filter; verify exact first witness parity on 4
    // Courtyards and bounded states on 16 Courtyards.
    for count in [4_usize, 16] {
        let mut sources: Vec<ManaSource> = TYPES[..count]
            .iter()
            .enumerate()
            .map(|(i, ty)| courtyard(i, ty))
            .collect();
        let mut mascot = tapping_ability(
            crate::state::mana::Color::COLORS
                .into_iter()
                .map(Mana::single)
                .collect(),
        );
        mascot.tap_source = false;
        mascot.limited = true;
        mascot.cost = ManaCost {
            purpose: SpendingContext::Activate {
                zone: crate::state::zone::ZoneType::Battlefield,
                source: crate::state::mana::SpendingObject::new([], true, true),
            },
            ..ManaCost::parse("1")
        };
        sources.push(source(count, None, vec![mascot]));
        let model = PaymentModel {
            cache: None,
            sources: Arc::new(sources),
            life: 20,
        };
        // Unpayable double colored pip for a non-creature spell (only 1 Mascot
        // can filter color; Courtyards produce only restricted color + {C}).
        let state = search_state(&model, vec![noncreature.clone()]);
        assert!(bounded_search(&model, state, SourcePolicy::All, 200, count == 4).is_none());
        // Payable single colored pip + generic on a Cat spell.
        if count >= 8 {
            let state = search_state(&model, vec![cat_spell.clone()]);
            assert!(bounded_search(&model, state, SourcePolicy::All, 200, false).is_some());
        }
    }

    // SpendEffect and ManaExpiry are preserved and never collapsed together.
    let mut rider_sources = vec![
        source(
            0,
            Some("G"),
            vec![tapping_ability(vec![Mana::of_class(
                ManaClass {
                    expiry: ManaExpiry::Step,
                    restriction: None,
                    effect: Some(SpendEffect::CreatureEntersWithPlusOneCounter {
                        without_subtype: "Human".into(),
                    }),
                },
                [0, 1, 0, 0, 0, 0],
            )])],
        ),
        source(
            1,
            Some("G"),
            vec![tapping_ability(vec![Mana::of_class(
                ManaClass {
                    expiry: ManaExpiry::EndOfCombat,
                    restriction: None,
                    effect: None,
                },
                [0, 1, 0, 0, 0, 0],
            )])],
        ),
        source(2, Some("G"), vec![tapping_ability(vec![Mana::parse("U")])]),
    ];
    let rider_model = PaymentModel {
        cache: None,
        sources: Arc::new(std::mem::take(&mut rider_sources)),
        life: 20,
    };
    let mut rider_search = PaymentSearch::new(&[pay("U")]);
    assert!(rider_search.groups(&rider_model).is_empty());
}

#[test]
fn payment_resources_bitmasks_and_overflow_preserve_order_and_hash() {
    use std::hash::{Hash, Hasher};

    let hash_of = |res: &super::PaymentResources| -> u64 {
        let mut h = super::FastHasher::default();
        res.hash(&mut h);
        h.finish()
    };

    let mut a = super::PaymentResources {
        present: vec![true, true],
        untapped: vec![true, true],
        used_masks: Vec::new(),
        used_overflow: None,
    };
    assert!(!a.is_used(0, 0));
    assert!(!a.is_used(0, 63));
    assert!(!a.is_used(0, 64));
    assert_eq!(a.cmp_source(0, 1), std::cmp::Ordering::Equal);

    a.use_ability(0, 0);
    a.use_ability(0, 63);
    assert!(a.is_used(0, 0));
    assert!(a.is_used(0, 63));
    assert!(!a.is_used(0, 1));
    assert!(!a.is_used(1, 0));
    assert_eq!(a.cmp_source(0, 1), std::cmp::Ordering::Greater);

    let mut b = a.clone();
    b.use_ability(0, 63); // idempotent
    assert_eq!(a, b);
    assert_eq!(hash_of(&a), hash_of(&b));

    // Overflow fallback for ability positions >= 64.
    a.use_ability(1, 64);
    a.use_ability(0, 64);
    a.use_ability(0, 70);
    assert!(a.is_used(1, 64));
    assert!(a.is_used(0, 64));
    assert!(a.is_used(0, 70));
    assert!(!a.is_used(1, 70));
    a.use_ability(1, 0);
    a.use_ability(1, 63);
    assert_eq!(a.cmp_source(0, 1), std::cmp::Ordering::Greater);
    a.use_ability(1, 70);
    assert_eq!(a.cmp_source(0, 1), std::cmp::Ordering::Equal);
}

#[test]
fn mana_ability_actions_deduplicates_equivalent_sources_and_keeps_convoke_colors_distinct() {
    use crate::agent::action::Action;
    use crate::flow::game::{Game, PendingChoice};
    use crate::flow::payment::{AnnouncedSpell, Operation, Payment};
    use crate::state::{
        card::{ActivatedAbilityDefinition, CardType},
        game_object::PlayerId,
        permanent::EntryChoice,
        player::PlayerConfig,
        zone::ZoneType,
    };
    use std::collections::BTreeMap;

    let player = PlayerId(0);
    let deck = BTreeMap::from([
        ("Secluded Courtyard".into(), 8),
        ("Three Tree Mascot".into(), 8),
        ("Forest".into(), 4),
        ("Helpful Hunter".into(), 4),
        ("Pillar Launch".into(), 4),
    ]);
    let mut game = Game::new(
        vec![
            PlayerConfig::new("p0", deck.clone()),
            PlayerConfig::new("p1", deck),
        ],
        431,
        true,
    );
    game.keep_opening_hands();
    game.scenario_clear_hand(player);

    let mut courtyards = Vec::new();
    for ty in ["Elf", "Goblin", "Cat", "Dragon", "Human", "Soldier"] {
        let id = game
            .scenario_force_battlefield(player, "Secluded Courtyard", true)
            .unwrap();
        game.state.permanents[id].as_mut().unwrap().chosen =
            Some(Arc::new(EntryChoice::CreatureType(ty.into())));
        courtyards.push(id);
    }
    let mut mascots = Vec::new();
    for _ in 0..4 {
        mascots.push(
            game.scenario_force_battlefield(player, "Three Tree Mascot", true)
                .unwrap(),
        );
    }
    game.scenario_force_card_in_hand(player, "Helpful Hunter")
        .unwrap();
    let hunter_card = game.state.zones.zone_objects(ZoneType::Hand, player)[0];
    game.payment_cache.borrow_mut().entries.clear();

    // Casting Helpful Hunter ({1}{W}, Cat creature): only the Cat Courtyard
    // can pay {W} directly; all 6 Courtyards and 4 Mascots are legal because
    // Mascot can filter and Cat Courtyard can pay, and deduplication must
    // preserve the exact action list while running one search per class.
    let operation = Operation::Spell(AnnouncedSpell {
        object: hunter_card,
        cast: Default::default(),
        mode: None,
        targets: vec![],
        target_req_indices: vec![],
        target_shares: vec![],
    });
    let payment = Payment {
        owed: game.cost_components(player, &operation),
        operation,
        receipt: Default::default(),
    };
    game.pending_choice = Some(PendingChoice::Pay {
        player,
        payment: Box::new(payment),
    });

    SEARCHED.with(|s| s.set(0));
    let offered = game.mana_ability_actions(player);
    let searches = SEARCHED.with(|s| s.get());
    let expected: Vec<Action> = courtyards
        .iter()
        .chain(&mascots)
        .map(|&permanent| Action::ActivateAbility {
            player,
            permanent,
            ability_index: 0,
        })
        .collect();
    assert_eq!(offered, expected);
    // 3 equivalence classes: Cat Courtyard, the other 5 Courtyards, 4 Mascots.
    assert!(searches >= 3);
    game.pending_choice = None;

    // Convoke distinction: two creatures with identical `{T}: Add {C}`
    // abilities but different colors (Green vs Red) paying a `{1}{G}` convoke
    // cost. Only tapping the Red creature leaves the Green creature untapped
    // to convoke `{G}`.
    let deck = BTreeMap::from([
        ("Gray Ogre".into(), 4),
        ("Guardian of Vitu-Ghazi".into(), 4),
    ]);
    let mut convoke_game = Game::new(
        vec![
            PlayerConfig::new("p0", deck.clone()),
            PlayerConfig::new("p1", deck),
        ],
        432,
        true,
    );
    convoke_game.keep_opening_hands();
    convoke_game.scenario_clear_hand(player);
    let tap_for_c = ActivatedAbilityDefinition {
        tap_source: true,
        effects: vec![crate::state::ability::Effect::add_mana(Mana::parse("C"))],
        ..Default::default()
    };
    let green = convoke_game
        .scenario_force_battlefield(player, "Gray Ogre", true)
        .unwrap();
    let green_obj = convoke_game.state.permanents[green]
        .as_ref()
        .unwrap()
        .object;
    convoke_game.state[green_obj].mana_cost = Some(ManaCost::parse("G"));
    convoke_game.state[green_obj].activated_abilities = vec![tap_for_c.clone()];

    let red = convoke_game
        .scenario_force_battlefield(player, "Gray Ogre", true)
        .unwrap();
    let red_obj = convoke_game.state.permanents[red].as_ref().unwrap().object;
    convoke_game.state[red_obj].mana_cost = Some(ManaCost::parse("R"));
    convoke_game.state[red_obj].activated_abilities = vec![tap_for_c];

    convoke_game
        .scenario_force_card_in_hand(player, "Guardian of Vitu-Ghazi")
        .unwrap();
    let spell = convoke_game
        .state
        .zones
        .zone_objects(ZoneType::Hand, player)[0];
    convoke_game.state[spell].mana_cost = Some(ManaCost::parse("1G"));
    convoke_game.state[spell].types = crate::state::card::CardTypes::new([CardType::Creature]);
    let op = Operation::Spell(AnnouncedSpell {
        object: spell,
        cast: Default::default(),
        mode: None,
        targets: vec![],
        target_req_indices: vec![],
        target_shares: vec![],
    });
    let pay = Payment {
        owed: convoke_game.cost_components(player, &op),
        operation: op,
        receipt: Default::default(),
    };
    convoke_game.pending_choice = Some(PendingChoice::Pay {
        player,
        payment: Box::new(pay),
    });
    assert_eq!(
        convoke_game.mana_ability_actions(player),
        vec![Action::ActivateAbility {
            player,
            permanent: red,
            ability_index: 0,
        }]
    );
}
