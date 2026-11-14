// source_taps.rs
// MTG-347's per-source payment arm (`MANAGYM_EXPERIMENT_SOURCE_TAPS=1`): a
// cast's mana payment offers each useful fixed source as a tap step beside
// `PayCost`. This file is its own test binary because the switch is read
// from the process environment when a game is constructed.

use std::collections::BTreeMap;

use managym::{
    agent::action::{Action, ActionType},
    state::{
        game_object::{PermanentId, PlayerId},
        player::PlayerConfig,
    },
    Game,
};

fn game() -> Game {
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
    for land in ["Mountain", "Mountain", "Mountain", "Forest"] {
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
    game.state.objects[object].name.clone()
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
        .map(|permanent| game.state.objects[permanent.object].name.clone())
        .filter(|land| land != "Gray Ogre")
        .collect();
    names.sort();
    names
}

#[test]
fn per_source_taps_choose_what_stays_untapped_and_pay_cost_finishes() {
    std::env::set_var("MANAGYM_EXPERIMENT_SOURCE_TAPS", "1");

    // {2}{R} from three Mountains and a Forest: every land can pay a pip and
    // one will be left, so all four are offered beside PayCost.
    let mut by_hand = game();
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
    let mut automatic = game();
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

    // Without the switch the payment commits at once, as it does today.
    std::env::remove_var("MANAGYM_EXPERIMENT_SOURCE_TAPS");
    let mut off = game();
    cast(&mut off);
    assert!(off.pending_choice.is_none(), "switch off pays at once");
    assert_eq!(untapped(&off), ["Forest"]);
}
