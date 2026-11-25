//! Source choices and automatic completion share the pending payment.

use managym::state::player::ManaFloats;
use std::collections::BTreeMap;

use managym::{
    agent::action::{Action, ActionType},
    state::{
        game_object::{PermanentId, PlayerId},
        mana::{Color, Mana},
        payment::PaymentReceipt,
        player::{ManaPayment, PlayerConfig},
        stack_object::StackObject,
    },
    Game,
};

fn game(lands: &[&str]) -> Game {
    game_with(lands, &[("Mountain", 20), ("Forest", 12), ("Gray Ogre", 8)])
}

fn game_with(lands: &[&str], deck: &[(&str, usize)]) -> Game {
    position(lands, deck, "Gray Ogre")
}

fn position(lands: &[&str], deck: &[(&str, usize)], spell: &str) -> Game {
    let deck: BTreeMap<String, usize> = deck
        .iter()
        .copied()
        .map(|(name, count)| (name.to_string(), count))
        .collect();
    let mut game = Game::new(
        vec![
            PlayerConfig::new("p0", deck.clone()),
            PlayerConfig::new("p1", deck),
        ],
        347,
        true,
    );
    for _ in 0..2 {
        game.step(0).unwrap(); // Both seats keep.
    }
    let player = PlayerId(0);
    game.scenario_clear_hand(player);
    game.scenario_force_card_in_hand(player, spell).unwrap();
    for land in lands {
        game.scenario_force_battlefield(player, land, true).unwrap();
    }
    game.scenario_refresh_priority().unwrap();
    game
}

fn cast(game: &mut Game) {
    let space = game.action_space().expect("priority").clone();
    let index = space
        .actions
        .iter()
        .position(|action| action.action_type() == ActionType::PriorityCastSpell)
        .expect("the spell is castable");
    game.step(index).unwrap();
}

fn name(game: &Game, permanent: PermanentId) -> String {
    let object = game.state.permanents[permanent].as_ref().unwrap().object;
    game.state[object].name.clone()
}

/// The tap offers of the pending payment, as (action index, land name).
fn taps(game: &Game) -> Vec<(usize, String)> {
    game.action_space()
        .expect("payment decision")
        .actions
        .iter()
        .enumerate()
        .filter_map(|(index, action)| match action {
            Action::TapForMana { permanent, .. } => Some((index, name(game, *permanent))),
            _ => None,
        })
        .collect()
}

/// Untapped lands; the Ogre may already have resolved.
fn untapped(game: &Game) -> Vec<String> {
    let mut names: Vec<String> = game
        .state
        .permanents
        .iter()
        .flatten()
        .filter(|permanent| !permanent.tapped)
        .map(|permanent| game.state[permanent.object].name.clone())
        .filter(|land| land != "Gray Ogre")
        .collect();
    names.sort();
    names
}

#[test]
fn per_source_taps_choose_what_stays_untapped_and_pay_cost_finishes() {
    // {2}{R} from three Mountains and a Forest: every land can pay a pip and
    // one will be left, so all four are offered beside PayCost.
    let lands = ["Mountain", "Mountain", "Mountain", "Forest"];
    let mut by_hand = game(&lands);
    cast(&mut by_hand);
    let offers = taps(&by_hand);
    assert_eq!(offers.len(), 4);
    let space = by_hand.action_space().unwrap();
    assert_eq!(space.player, Some(PlayerId(0)));
    assert!(space
        .actions
        .iter()
        .any(|action| matches!(action, Action::PayCost { .. })));

    // Tap the Forest: two Mountains are still needed, and which one stays
    // is still a choice. The Forest is gone from the offers.
    let forest = offers.iter().find(|(_, land)| land == "Forest").unwrap().0;
    by_hand.step(forest).unwrap();
    let offers = taps(&by_hand);
    assert_eq!(offers.len(), 3);
    assert!(offers.iter().all(|(_, land)| land == "Mountain"));
    by_hand.step(offers[0].0).unwrap();
    assert_eq!(taps(&by_hand).len(), 2);
    // The last needed tap completes the payment without a further decision.
    let last = taps(&by_hand)[0].0;
    by_hand.step(last).unwrap();
    assert!(by_hand.pending_choice.is_none());
    assert_eq!(untapped(&by_hand), ["Mountain"]);

    // PayCost at once is the automatic payer: Mountains first, in order.
    let mut automatic = game(&lands);
    cast(&mut automatic);
    let pay = automatic
        .action_space()
        .unwrap()
        .actions
        .iter()
        .position(|action| matches!(action, Action::PayCost { .. }))
        .unwrap();
    automatic.step(pay).unwrap();
    assert_eq!(untapped(&automatic), ["Forest"]);
}

fn payment_position(spell: &str, sources: &[&str]) -> Game {
    let mut deck = vec![("Mountain", 20), (spell, 8)];
    for source in sources {
        if !deck.iter().any(|(name, _)| name == source) {
            deck.push((source, 8));
        }
    }
    position(sources, &deck, spell)
}

fn payment_labels(game: &Game) -> Vec<String> {
    game.semantic_decision_frame()
        .unwrap()
        .offers
        .into_iter()
        .map(|offer| offer.label)
        .collect()
}

#[test]
fn exact_dual_land_payments_cast_without_a_source_or_color_prompt() {
    for lands in [
        ["Gruul Guildgate", "Forest", "Forest"],
        ["Gruul Guildgate", "Mountain", "Forest"],
        ["Gruul Guildgate"; 3],
    ] {
        let mut game = payment_position("Gray Ogre", &lands);
        cast(&mut game);
        assert!(game.pending_choice.is_none(), "{lands:?}");
        assert!(untapped(&game).is_empty(), "{lands:?}");
        assert_eq!(spell_receipt(&game, "Gray Ogre").total(), 3);
        assert_eq!(game.state.players[0].mana_pool.total(), 0);
    }
}

#[test]
fn a_spare_source_keeps_every_payment_offer_including_the_dual() {
    let mut game = payment_position(
        "Gray Ogre",
        &["Gruul Guildgate", "Mountain", "Mountain", "Forest"],
    );
    cast(&mut game);
    let labels = payment_labels(&game);
    assert_eq!(labels.len(), 5, "{labels:?}");
    assert_eq!(labels[0], "Tap Gruul Guildgate for {R}/{G}");
    assert_eq!(taps(&game).len(), 3);
    game.step(0).unwrap();
    // Red leaves two generic pips, green leaves a red pip and a generic.
    assert_eq!(payment_labels(&game), ["Add {R}", "Add {G}"]);
}

#[test]
fn automatic_payment_with_a_dual_leaves_a_surplus_plains_untapped() {
    let mut game = payment_position("Badgermole Cub", &["Plains", "Plains", "Kyoshi Village"]);
    cast(&mut game);
    pay_cost(&mut game);
    assert!(game.pending_choice.is_none());
    assert_eq!(game.state.players[0].mana_pool.total(), 0);
    assert_eq!(
        untapped(&game)
            .iter()
            .filter(|name| name.as_str() == "Plains")
            .count(),
        1
    );
    assert!(!untapped(&game).contains(&"Kyoshi Village".to_string()));
}

#[test]
fn a_dual_paying_generic_mana_skips_only_equivalent_colors() {
    let mut game = payment_position(
        "Gray Ogre",
        &["Kyoshi Village", "Plains", "Mountain", "Island"],
    );
    cast(&mut game);
    tap(&mut game, "Mountain");
    let labels = payment_labels(&game);
    let dual = labels
        .iter()
        .position(|label| label == "Tap Kyoshi Village for {G}/{W}")
        .unwrap();
    game.step(dual).unwrap();
    assert_eq!(owed(&game), "{1}");
    assert_eq!(taps(&game).len(), 2);
    assert_eq!(committed(&game).len(), 2);
}

#[test]
fn two_forced_completions_tapping_different_lands_keep_the_color_modal() {
    let mut game = payment_position(
        "Anthem of Champions",
        &["Kyoshi Village", "Forest", "Plains"],
    );
    cast(&mut game);
    let dual = payment_labels(&game)
        .iter()
        .position(|label| label == "Tap Kyoshi Village for {G}/{W}")
        .unwrap();
    game.step(dual).unwrap();
    assert_eq!(payment_labels(&game), ["Add {G}", "Add {W}"]);
    for (mode, kept) in [(0, "Forest"), (1, "Plains")] {
        let mut branch = game.clone();
        branch.step(mode).unwrap();
        assert!(branch.pending_choice.is_none());
        assert!(untapped(&branch).contains(&kept.to_string()));
        assert_eq!(branch.state.players[0].mana_pool.total(), 0);
    }
}

#[test]
fn mana_abilities_on_one_permanent_only_fund_one_tap() {
    for source in ["Jasmine Dragon Tea Shop", "Kyoshi Village"] {
        let mut game = payment_position("Compassionate Healer", &[source, "Plains"]);
        cast(&mut game);
        assert!(game.pending_choice.is_none());
        for permanent in game.state.permanents.iter().flatten() {
            if [source, "Plains"].contains(&game.state[permanent.object].name.as_str()) {
                assert!(permanent.tapped);
            }
        }
    }
    let mut game = payment_position("Badgermole Cub", &["Animal Attendant", "Plains"]);
    cast(&mut game);
    assert!(game.pending_choice.is_none());
    // The spend effect is part of the receipt whichever color was chosen.
    assert_eq!(spell_receipt(&game, "Badgermole Cub").entry_counters(), 1);
}

#[test]
fn turning_a_manifest_face_up_pays_with_a_dual_and_a_basic() {
    let player = PlayerId(0);
    let mut game = payment_position("Compassionate Healer", &["Kyoshi Village", "Plains"]);
    let card = game
        .state
        .zones
        .zone_objects(managym::state::zone::ZoneType::Hand, player)
        .iter()
        .copied()
        .next()
        .unwrap();
    let face_down = game.manifest(card, player);
    game.scenario_refresh_priority().unwrap();
    let index = game
        .action_space()
        .unwrap()
        .actions
        .iter()
        .position(|a| matches!(a, Action::TurnFaceUp { object, .. } if *object == face_down))
        .unwrap();
    game.step(index).unwrap();
    assert!(!game.state[face_down].is_face_down());
    assert_eq!(game.state.players[0].mana_pool.total(), 0);
    assert_eq!(untapped(&game), ["Compassionate Healer"]);
}

#[test]
fn ward_keeps_pay_or_decline_and_keeps_source_choices_when_they_matter() {
    let player = PlayerId(0);
    let deck = [
        ("Mountain", 20),
        ("Lightning Bolt", 8),
        ("Kyoshi Village", 4),
        ("Plains", 8),
        ("Waterfall Aerialist", 4),
        ("Sol Ring", 4),
    ];
    for other in ["Plains", "Sol Ring"] {
        let mut game = position(&["Kyoshi Village", other], &deck, "Lightning Bolt");
        let target = game
            .scenario_force_battlefield(PlayerId(1), "Waterfall Aerialist", true)
            .unwrap();
        game.state.players[0].mana_pool = Mana::parse("R").into();
        game.scenario_refresh_priority().unwrap();
        cast(&mut game);
        let index = game.action_space().unwrap().actions.iter().position(|a| matches!(a, Action::ChooseTarget { target: managym::state::game_object::Target::Permanent(p), .. } if *p == target)).unwrap();
        game.step(index).unwrap();
        let actions = &game.action_space().unwrap().actions;
        if other == "Sol Ring" {
            assert!(actions
                .iter()
                .any(|a| matches!(a, Action::ActivateAbility { .. })));
        } else {
            assert_eq!(
                actions,
                &[Action::PayCost { player }, Action::Decline { player }]
            );
            let mut decline = game.clone();
            decline.step(1).unwrap();
            assert!(untapped(&decline).contains(&"Kyoshi Village".into()));
            game.step(0).unwrap();
            assert!(!untapped(&game).contains(&"Kyoshi Village".into()));
        }
    }
}

#[test]
fn granted_mana_does_not_double_count_a_forest_and_treasure_stays_explicit() {
    let deck = [
        ("Forest", 20),
        ("Great Divide Guide", 4),
        ("Firebrand Archer", 8),
    ];
    let mut game = position(&["Forest", "Forest"], &deck, "Firebrand Archer");
    game.scenario_force_battlefield(PlayerId(0), "Great Divide Guide", false)
        .unwrap();
    game.scenario_refresh_priority().unwrap();
    let mut with_treasure = game.clone();
    let treasure = with_treasure
        .create_token("token.treasure", PlayerId(0), false)
        .unwrap();
    with_treasure.scenario_refresh_priority().unwrap();
    cast(&mut game);
    assert!(game.pending_choice.is_none());
    cast(&mut with_treasure);
    assert!(with_treasure
        .action_space()
        .unwrap()
        .actions
        .iter()
        .any(|a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == treasure)));
}

#[test]
fn fourteen_lands_with_granted_color_choices_stay_under_the_policy_action_capacity() {
    let deck = [
        ("Forest", 24),
        ("Great Divide Guide", 4),
        ("Firebrand Archer", 8),
    ];
    let mut game = position(&["Forest"; 14], &deck, "Firebrand Archer");
    game.scenario_force_battlefield(PlayerId(0), "Great Divide Guide", false)
        .unwrap();
    game.scenario_refresh_priority().unwrap();
    cast(&mut game);
    let labels = payment_labels(&game);
    assert_eq!(
        labels
            .iter()
            .filter(|s| s.as_str() == "Tap Forest for any color")
            .count(),
        14
    );
    assert_eq!(taps(&game).len(), 14);
    assert_eq!(labels.len(), 29);
}

#[test]
fn paying_a_convoke_spell_entirely_in_mana_charges_its_generic_cost_once() {
    let mut game = payment_position("Guardian of Vitu-Ghazi", &["Kyoshi Village", "Forest"]);
    game.state.players[0].mana_pool = Mana::parse("6").into();
    game.scenario_refresh_priority().unwrap();
    cast(&mut game);
    assert!(game.pending_choice.is_none());
    assert_eq!(spell_receipt(&game, "Guardian of Vitu-Ghazi").total(), 8);
    assert_eq!(game.state.players[0].mana_pool.total(), 0);
}

#[test]
fn lesson_mana_uses_two_units_only_when_the_cost_needs_both() {
    let mut exact = payment_position("Energybending", &["Hermitic Herbalist"]);
    cast(&mut exact);
    assert!(exact.pending_choice.is_none());
    assert_eq!(exact.state.players[0].mana_pool.total(), 0);
    assert!(!untapped(&exact).contains(&"Hermitic Herbalist".into()));

    let mut spare = payment_position("Boomerang Basics", &["Hermitic Herbalist"]);
    cast(&mut spare);
    let labels = payment_labels(&spare);
    assert!(
        labels.contains(&"Tap Hermitic Herbalist for any color".into()),
        "{labels:?}"
    );
    assert!(
        labels.contains(&"Tap Hermitic Herbalist for 2 mana (Lesson spells only)".into()),
        "{labels:?}"
    );
}

#[test]
fn converge_keeps_a_color_choice_with_exact_sources() {
    let mut game = payment_position(
        "Uncle's Musings",
        &["Kyoshi Village", "Forest", "Forest", "Forest"],
    );
    cast(&mut game);
    let labels = payment_labels(&game);
    let dual = labels
        .iter()
        .position(|label| label == "Tap Kyoshi Village for {G}/{W}")
        .unwrap();
    game.step(dual).unwrap();
    assert_eq!(payment_labels(&game), ["Add {G}", "Add {W}"]);
}

#[test]
fn funded_and_all_sources_needed_payments_remain_trivial() {
    let mut exact = game(&["Mountain", "Mountain", "Mountain"]);
    cast(&mut exact);
    assert!(exact.pending_choice.is_none());
    assert!(untapped(&exact).is_empty());

    let lands = ["Mountain", "Mountain", "Mountain", "Forest"];
    let mut funded = game(&lands);
    funded.state.players[0].mana_pool = Mana::parse("2R").into();
    cast(&mut funded);
    assert!(funded.pending_choice.is_none());
    assert_eq!(
        untapped(&funded),
        ["Forest", "Mountain", "Mountain", "Mountain"]
    );
}

#[test]
fn priority_taps_append_each_source_and_float_mana_for_a_cast() {
    let lands = ["Mountain", "Forest", "Forest"];
    let mut game = game(&lands);
    let ordinary = game.action_space().unwrap().actions.clone();
    game.mana_floats = [ManaFloats::Off, ManaFloats::All];
    game.scenario_refresh_priority().unwrap();
    assert_eq!(game.action_space().unwrap().actions, ordinary);

    game.mana_floats = [ManaFloats::All, ManaFloats::Off];
    game.scenario_refresh_priority().unwrap();
    assert_eq!(
        &game.action_space().unwrap().actions[..ordinary.len()],
        ordinary
    );
    let offers = taps(&game);
    assert_eq!(
        offers,
        [
            (ordinary.len(), "Mountain".into()),
            (ordinary.len() + 1, "Forest".into()),
            (ordinary.len() + 2, "Forest".into())
        ]
    );
    game.step(offers[1].0).unwrap();
    assert_eq!(game.state.players[0].mana_pool, Mana::parse("G"));
    assert_eq!(game.action_space().unwrap().player, Some(PlayerId(0)));
    assert_eq!(taps(&game).len(), 2);
    cast(&mut game);
    pay_cost(&mut game);
    assert!(game.pending_choice.is_none());
    assert!(untapped(&game).is_empty());
    assert_eq!(game.state.players[0].mana_pool, Mana::default());
}

#[test]
fn funded_cast_can_tap_a_surplus_forest_before_paying() {
    let mut game = game(&["Forest", "Mountain"]);
    game.state.players[0].mana_pool = Mana::parse("2R").into();
    game.mana_floats = [ManaFloats::All, ManaFloats::Off];
    game.scenario_refresh_priority().unwrap();
    cast(&mut game);
    assert!(matches!(
        game.action_space().unwrap().actions[0],
        Action::PayCost { .. }
    ));
    let forest = taps(&game)
        .into_iter()
        .find(|(_, name)| name == "Forest")
        .unwrap()
        .0;
    game.step(forest).unwrap();
    // The Forest's {G} could pay {1} that floating colorless could pay
    // instead, so it waits with the floating mana for the spend choice.
    assert_eq!(*game.state.players[0].mana_pool.mana(), Mana::parse("2RG"));
    assert_eq!(owed(&game), "{2}{R}");
    assert!(committed(&game).is_empty());
    pay_cost(&mut game);
    // Generic spending can ask which floating unit to keep.
    while game
        .action_space()
        .unwrap()
        .actions
        .iter()
        .all(|a| matches!(a, Action::ChooseMode { .. }))
    {
        game.step(0).unwrap();
    }
    assert!(game.pending_choice.is_none());
    assert_eq!(game.state.players[0].mana_pool.total(), 1);
    assert_eq!(untapped(&game), ["Mountain"]);
}

/// The mana the pending decision says is still owed.
fn owed(game: &Game) -> String {
    game.decision_context(PlayerId(0))
        .and_then(|context| context.mana_cost.0)
        .expect("a mana cost is being paid")
}

/// What the pending decision says is committed to the cost, in order.
fn committed(game: &Game) -> Vec<String> {
    game.decision_context(PlayerId(0))
        .map(|context| context.paid.into_iter().map(|paid| paid.phrase).collect())
        .unwrap_or_default()
}

fn tap(game: &mut Game, land: &str) {
    let index = taps(game)
        .into_iter()
        .find(|(_, name)| name == land)
        .unwrap_or_else(|| panic!("{land} is offered"))
        .0;
    game.step(index).unwrap();
}

/// CR 601.2g-h: mana a payer makes while paying a cost pays toward it as it
/// is made. The owed cost ticks down, the unit joins the receipt with its
/// source and the decision names it, as a convoke tap does.
#[test]
fn mana_made_while_paying_commits_to_the_cost() {
    let mut game = game(&["Mountain", "Mountain", "Forest", "Forest"]);
    cast(&mut game);
    assert_eq!(owed(&game), "{2}{R}");
    let labels: Vec<_> = game
        .semantic_decision_frame()
        .unwrap()
        .offers
        .iter()
        .map(|offer| offer.label.clone())
        .collect();
    assert!(
        labels.contains(&"Tap Forest for {G}".to_string()),
        "{labels:?}"
    );
    tap(&mut game, "Forest");
    assert_eq!(owed(&game), "{1}{R}");
    assert_eq!(committed(&game), ["{G} from Forest"]);
    assert_eq!(game.state.players[0].mana_pool, Mana::default());
    // A red unit pays the red pip, which leaves the most that can pay the rest.
    tap(&mut game, "Mountain");
    assert_eq!(owed(&game), "{1}");
    assert_eq!(committed(&game), ["{G} from Forest", "{R} from Mountain"]);

    // A fork at the last tap finishes once each way.
    for last in ["Mountain", "Forest"] {
        let mut fork = game.clone();
        tap(&mut fork, last);
        assert!(fork.pending_choice.is_none());
        let receipt = spell_receipt(&fork, "Gray Ogre");
        let colors: Vec<u8> = receipt.mana().iter().map(|unit| unit.color).collect();
        let third = if last == "Forest" { GREEN } else { RED };
        assert_eq!(colors, [GREEN, RED, third]);
        assert_eq!(untapped(&fork).len(), 1);
        assert_eq!(fork.state.players[0].mana_pool, Mana::default());
        assert_eq!(owed(&game), "{1}");
    }
}

#[test]
fn required_sol_ring_can_leave_a_mountain_or_floating_mana() {
    let mut game = payment_position("Gray Ogre", &["Sol Ring", "Mountain", "Mountain"]);
    cast(&mut game);
    assert_eq!(taps(&game).len(), 3);
    game.skip_trivial = false; // Inspect priority before the floating mana empties.

    let mut exact = game.clone();
    pay_cost(&mut exact);
    assert_eq!(spell_receipt(&exact, "Gray Ogre").total(), 3);
    assert_eq!(untapped(&exact), ["Mountain"]);
    assert_eq!(exact.state.players[0].mana_pool.total(), 0);

    tap(&mut game, "Mountain");
    tap(&mut game, "Mountain");
    if game.pending_choice.is_some() {
        pay_cost(&mut game);
    }
    assert_eq!(spell_receipt(&game, "Gray Ogre").total(), 3);
    assert!(untapped(&game).is_empty());
    assert_eq!(game.state.players[0].mana_pool, Mana::parse("C"));
}

/// A source that makes more than is owed pays what is owed; the rest floats.
#[test]
fn surplus_of_mana_made_while_paying_floats() {
    let deck = [
        ("Mountain", 20),
        ("Forest", 11),
        ("Sol Ring", 1),
        ("Gray Ogre", 8),
    ];
    let mut game = game_with(&["Mountain", "Forest", "Sol Ring", "Mountain"], &deck);
    // Floats keep priority a real choice after the cast, so the pool is seen.
    game.mana_floats = [ManaFloats::All, ManaFloats::Off];
    game.scenario_refresh_priority().unwrap();
    cast(&mut game);
    tap(&mut game, "Mountain");
    tap(&mut game, "Forest");
    assert_eq!(owed(&game), "{1}");
    // The Sol Ring's {C} paid the last {1}: the Ogre is cast, the surplus
    // floats.
    tap(&mut game, "Sol Ring");
    assert!(game.pending_choice.is_none());
    assert_eq!(untapped(&game), ["Mountain"]);
    assert_eq!(
        spell_receipt(&game, "Gray Ogre").amounts(),
        [0, 0, 0, 1, 1, 1]
    );
    assert_eq!(game.state.players[0].mana_pool, Mana::parse("C"));
}

/// Mana that can pay nothing still owed stays in the pool, and the cost
/// does not move.
#[test]
fn mana_made_while_paying_that_cannot_pay_floats() {
    // A spare Mountain keeps priority a real choice after the cast.
    let mut game = game(&["Mountain", "Mountain", "Forest", "Forest", "Forest"]);
    game.mana_floats = [ManaFloats::All, ManaFloats::Off];
    game.scenario_refresh_priority().unwrap();
    cast(&mut game);
    tap(&mut game, "Forest");
    tap(&mut game, "Forest");
    assert_eq!(owed(&game), "{R}");
    tap(&mut game, "Forest");
    assert_eq!(owed(&game), "{R}");
    assert_eq!(game.state.players[0].mana_pool, Mana::parse("G"));
    assert_eq!(committed(&game).len(), 2);
    tap(&mut game, "Mountain");
    assert!(game.pending_choice.is_none());
    assert_eq!(
        spell_receipt(&game, "Gray Ogre").amounts(),
        [0, 0, 0, 1, 2, 0]
    );
    assert_eq!(game.state.players[0].mana_pool, Mana::parse("G"));
}

/// A payment ends when no mana is owed, however it was paid: the last unit
/// here comes from an activated mana ability while another one is still
/// untapped beside two lands, floats off, and the Ogre is cast with no
/// "Pay the cost".
#[test]
fn a_payment_ends_when_its_last_unit_comes_from_a_mana_ability() {
    let deck = [
        ("Mountain", 20),
        ("Forest", 10),
        ("Meteorite", 2),
        ("Gray Ogre", 8),
    ];
    let lands = [
        "Mountain",
        "Mountain",
        "Forest",
        "Forest",
        "Meteorite",
        "Meteorite",
    ];
    let mut game = game_with(&lands, &deck);
    cast(&mut game);
    tap(&mut game, "Mountain");
    tap(&mut game, "Forest");
    assert_eq!(owed(&game), "{1}");
    let activate = game
        .action_space()
        .unwrap()
        .actions
        .iter()
        .position(|action| matches!(action, Action::ActivateAbility { .. }))
        .expect("a Meteorite's mana ability is offered");
    game.step(activate).unwrap();
    while game.pending_choice.is_some() {
        // The Meteorite's color: any pays {1}.
        let space = game.action_space().unwrap();
        assert!(
            space
                .actions
                .iter()
                .all(|action| matches!(action, Action::ChooseMode { .. })),
            "only the Meteorite's color is chosen: {space:?}"
        );
        game.step(0).unwrap();
    }
    assert!(game.pending_choice.is_none());
    assert_eq!(untapped(&game), ["Forest", "Meteorite", "Mountain"]);
    assert_eq!(
        spell_receipt(&game, "Gray Ogre")
            .amounts()
            .iter()
            .sum::<u8>(),
        3
    );
    assert_eq!(game.state.players[0].mana_pool.total(), 0);
}

fn pay_cost(game: &mut Game) {
    let index = game
        .action_space()
        .unwrap()
        .actions
        .iter()
        .position(|a| matches!(a, Action::PayCost { .. }))
        .unwrap();
    game.step(index).unwrap();
}

#[test]
fn surplus_sources_follow_the_existing_payment_actions() {
    let mut off = game(&["Mountain", "Mountain", "Forest"]);
    off.state.players[0].mana_pool = Mana::parse("2").into();
    let mut on = off.clone();
    on.mana_floats = [ManaFloats::All, ManaFloats::Off];
    let mut usable = off.clone();
    usable.mana_floats = [ManaFloats::Usable, ManaFloats::Off];
    cast(&mut off);
    cast(&mut on);
    cast(&mut usable);
    assert_eq!(
        on.action_space().unwrap().actions,
        usable.action_space().unwrap().actions
    );
    let ordinary = &off.action_space().unwrap().actions;
    assert_eq!(taps(&off).len(), 2);
    assert_eq!(
        &on.action_space().unwrap().actions[..ordinary.len()],
        ordinary
    );
    assert_eq!(taps(&on).last(), Some(&(ordinary.len(), "Forest".into())));
    assert_eq!(on.action_space().unwrap().actions.len(), ordinary.len() + 1);
}

#[test]
fn usable_priority_requires_a_spend_or_a_trigger_on_that_source() {
    use managym::state::ability::{Ability, Effect, TriggerCondition, TriggerSubject};
    let mut game = game(&["Mountain", "Forest", "Forest"]);
    game.mana_floats = [ManaFloats::Usable, ManaFloats::Off];
    game.scenario_refresh_priority().unwrap();
    assert_eq!(taps(&game).len(), 3); // Ogre costs {2}{R}.
    game.scenario_clear_hand(PlayerId(0));
    game.scenario_refresh_priority().unwrap();
    assert!(taps(&game).is_empty());

    let mountain = game
        .state
        .permanents
        .iter()
        .flatten()
        .map(|permanent| permanent.object)
        .find(|object| game.state[*object].name == "Mountain")
        .unwrap();
    game.state[mountain].abilities.push(Ability::Triggered {
        condition: TriggerCondition::TappedForMana {
            subject: TriggerSubject::This,
        },
        intervening_if: None,
        effects: vec![Effect::gain_life(1)],
    });
    game.scenario_refresh_priority().unwrap();
    let offers = taps(&game);
    assert_eq!(offers.len(), 1);
    assert_eq!(offers[0].1, "Mountain");
    game.step(offers[0].0).unwrap();
    assert_eq!(game.state.players[0].life, 21);
}

/// The receipt on a spell while it is on the stack,
/// then on the permanent it became.
fn spell_receipt(game: &Game, name: &str) -> PaymentReceipt {
    let cast = game
        .state
        .stack_objects
        .iter()
        .find_map(|object| match object {
            StackObject::Spell(spell) => Some(spell.spent.clone()),
            _ => None,
        });
    cast.unwrap_or_else(|| {
        game.state
            .permanents
            .iter()
            .flatten()
            .find(|permanent| game.state[permanent.object].name == name)
            .expect("the spell was cast")
            .spell_entry
            .as_deref()
            .unwrap()
            .payment
            .clone()
            .unwrap()
    })
}

/// The lands a receipt names, sorted, with the color each one's unit had.
fn paid_by(game: &Game, receipt: &PaymentReceipt) -> Vec<(String, u8)> {
    let mut paid: Vec<(String, u8)> = receipt
        .mana()
        .iter()
        .map(|unit| {
            let origin = unit.source.expect("a land produced this unit");
            // The origin is the land's own battlefield appearance.
            let permanent = game.state.permanent_of(origin.object).expect("the land");
            assert!(game.state.permanents[permanent].as_ref().unwrap().tapped);
            assert_eq!(game.state[origin.object].definition_id, origin.definition);
            (name(game, permanent), unit.color)
        })
        .collect();
    paid.sort();
    paid
}

const RED: u8 = Color::Red as u8;
const GREEN: u8 = Color::Green as u8;

/// CR 601.2h: the receipt names the object that produced each unit spent,
/// however the payment was made: taps chosen by hand, the automatic payer,
/// and mana tapped at priority before the cast.
#[test]
fn receipt_names_the_source_of_each_unit_under_manual_auto_and_floated_payment() {
    let lands = ["Mountain", "Mountain", "Mountain", "Forest"];
    let expected = [
        ("Forest".to_string(), GREEN),
        ("Mountain".to_string(), RED),
        ("Mountain".to_string(), RED),
    ];

    // Manual: the Forest and two Mountains, tapped one at a time.
    let mut by_hand = game(&lands);
    cast(&mut by_hand);
    let forest = taps(&by_hand)
        .into_iter()
        .find(|(_, land)| land == "Forest")
        .unwrap()
        .0;
    by_hand.step(forest).unwrap();
    while by_hand.pending_choice.is_some() {
        let tap = taps(&by_hand)[0].0;
        by_hand.step(tap).unwrap();
    }
    assert_eq!(
        paid_by(&by_hand, &spell_receipt(&by_hand, "Gray Ogre")),
        expected
    );

    // Auto: the payer taps Mountains first; the receipt names exactly those.
    let mut automatic = game(&lands);
    automatic.mana_payment[0] = ManaPayment::Auto;
    cast(&mut automatic);
    assert!(automatic.pending_choice.is_none());
    assert_eq!(
        paid_by(&automatic, &spell_receipt(&automatic, "Gray Ogre")),
        [RED; 3].map(|color| ("Mountain".to_string(), color))
    );
    assert_eq!(untapped(&automatic), ["Forest"]);

    // Floated: the Forest is tapped at priority. Its unit keeps its origin
    // in the pool and the later receipt names it.
    let mut floated = game(&lands);
    floated.mana_floats = [ManaFloats::All, ManaFloats::Off];
    floated.scenario_refresh_priority().unwrap();
    let forest = taps(&floated)
        .into_iter()
        .find(|(_, land)| land == "Forest")
        .unwrap()
        .0;
    floated.step(forest).unwrap();
    let pool = &floated.state.players[0].mana_pool;
    let origins: Vec<_> = pool.origins(None, Color::Green).collect();
    assert_eq!(origins.len(), 1);
    let origin = origins[0].expect("the Forest");
    assert_eq!(floated.state[origin.object].name, "Forest");
    cast(&mut floated);
    // Floats stay offered during payment, so the payer stops by hand: two
    // Mountains, then PayCost.
    for _ in 0..2 {
        let tap = taps(&floated)[0].0;
        floated.step(tap).unwrap();
    }
    if floated.pending_choice.is_some() {
        pay_cost(&mut floated);
    }
    assert_eq!(
        paid_by(&floated, &spell_receipt(&floated, "Gray Ogre")),
        expected
    );
    assert_eq!(untapped(&floated), ["Mountain"]);
    assert_eq!(floated.state.players[0].mana_pool, Mana::default());

    // Whole object ids, so a later appearance of a land is never mistaken
    // for the one that paid.
    let json = serde_json::to_value(spell_receipt(&floated, "Gray Ogre")).unwrap();
    let source = &json["mana"][0]["source"];
    assert!(source["object"]["entity"].is_u64() && source["object"]["incarnation"].is_u64());
    assert!(source["definition"].is_u64());
}

/// Mana no object produced has no origin; mixing it with a land's mana
/// keeps each unit's own fact, and only the unit spent reaches the receipt.
#[test]
fn receipt_separates_sourceless_mana_from_a_lands() {
    let mut game = game(&["Mountain", "Forest"]);
    game.state.players[0].mana_pool = Mana::parse("2").into();
    game.mana_payment[0] = ManaPayment::Auto;
    game.scenario_refresh_priority().unwrap();
    cast(&mut game);
    assert!(game.pending_choice.is_none());
    let receipt = spell_receipt(&game, "Gray Ogre");
    assert_eq!(receipt.total(), 3);
    assert_eq!(receipt.amounts(), [0, 0, 0, 1, 0, 2]);
    let sources = receipt.mana_sources();
    assert_eq!(sources.len(), 1);
    assert_eq!(game.state[sources[0].object].name, "Mountain");
    assert_eq!(untapped(&game), ["Forest"]);
}
