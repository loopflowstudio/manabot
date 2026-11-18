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
    let deck: BTreeMap<String, usize> = [("Mountain", 20), ("Forest", 12), ("Gray Ogre", 8)]
        .into_iter()
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
    let player = PlayerId(0);
    game.scenario_clear_hand(player);
    game.scenario_force_card_in_hand(player, "Gray Ogre")
        .unwrap();
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
        .expect("Gray Ogre is castable");
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
    assert_eq!(game.state.players[0].mana_pool, Mana::parse("2RG"));
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

/// The receipt of the Ogre's payment: on the spell while it is on the stack,
/// then on the permanent it became.
fn ogre_receipt(game: &Game) -> PaymentReceipt {
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
            .find(|permanent| game.state[permanent.object].name == "Gray Ogre")
            .expect("the Ogre was cast")
            .cast_spent
            .clone()
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
    assert_eq!(paid_by(&by_hand, &ogre_receipt(&by_hand)), expected);

    // Auto: the payer taps Mountains first; the receipt names exactly those.
    let mut automatic = game(&lands);
    automatic.mana_payment[0] = ManaPayment::Auto;
    cast(&mut automatic);
    assert!(automatic.pending_choice.is_none());
    assert_eq!(
        paid_by(&automatic, &ogre_receipt(&automatic)),
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
    assert_eq!(paid_by(&floated, &ogre_receipt(&floated)), expected);
    assert_eq!(untapped(&floated), ["Mountain"]);
    assert_eq!(floated.state.players[0].mana_pool, Mana::default());

    // Whole object ids, so a later appearance of a land is never mistaken
    // for the one that paid.
    let json = serde_json::to_value(ogre_receipt(&floated)).unwrap();
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
    let receipt = ogre_receipt(&game);
    assert_eq!(receipt.total(), 3);
    assert_eq!(receipt.amounts(), [0, 0, 0, 1, 0, 2]);
    let sources = receipt.mana_sources();
    assert_eq!(sources.len(), 1);
    assert_eq!(game.state[sources[0].object].name, "Mountain");
    assert_eq!(untapped(&game), ["Forest"]);
}
