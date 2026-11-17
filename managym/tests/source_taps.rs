//! Source choices and automatic completion share the pending payment.

use std::collections::BTreeMap;

use managym::{
    agent::action::{Action, ActionType},
    state::{
        game_object::{PermanentId, PlayerId},
        mana::Mana,
        player::PlayerConfig,
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
    funded.state.players[0].mana_pool = Mana::parse("2R");
    cast(&mut funded);
    assert!(funded.pending_choice.is_none());
    assert_eq!(
        untapped(&funded),
        ["Forest", "Mountain", "Mountain", "Mountain"]
    );
}
