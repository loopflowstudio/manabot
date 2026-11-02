//! Deterministic interactive-mirror mid-game positions.
//!
//! Each fixture plays a fixed seed and action tape from a fresh match and checks
//! its shape, so clone-cost and witness tests measure the same exact state.

use std::{collections::BTreeMap, hint::black_box};

use rand::Rng;

use crate::{agent::observation::Observation, state::player::PlayerConfig, Game};

const SETUP_SEED: u64 = 377;
const HEAVY_ACTION_SEED: u64 = 0xc10e;

fn interactive_deck() -> BTreeMap<String, usize> {
    BTreeMap::from([
        ("Island".to_string(), 12),
        ("Mountain".to_string(), 12),
        ("Gray Ogre".to_string(), 6),
        ("Wind Drake".to_string(), 6),
        ("Man-o'-War".to_string(), 4),
        ("Raging Goblin".to_string(), 4),
        ("Lightning Bolt".to_string(), 6),
        ("Counterspell".to_string(), 4),
        ("Ancestral Recall".to_string(), 3),
        ("Pyroclasm".to_string(), 3),
    ])
}

fn player_configs() -> Vec<PlayerConfig> {
    vec![
        PlayerConfig::new("hero", interactive_deck()),
        PlayerConfig::new("villain", interactive_deck()),
    ]
}

fn new_match(seed: u64) -> Game {
    Game::new(player_configs(), seed, true)
}

/// Plays fixture `id` from its seed and checks the position has not drifted.
pub fn build_fixture(id: &str) -> Result<Game, String> {
    let mut game = new_match(SETUP_SEED);
    let (count, drain_observations) = match id {
        "interactive-midgame-48-w4-v1" => (48, true),
        "interactive-heavy-80-w4-v1" => {
            game.reseed(HEAVY_ACTION_SEED);
            (80, false)
        }
        _ => return Err(format!("unknown fixture {id}")),
    };
    for decision in 0..count {
        if game.is_game_over() {
            return Err(format!("fixture {id} terminated at decision {decision}"));
        }
        let action_count = game.action_space().map_or(0, |space| space.actions.len());
        if action_count == 0 {
            return Err(format!(
                "fixture {id} has no legal action at decision {decision}"
            ));
        }
        let action = game.state.rng.gen_range(0..action_count);
        game.step(action)
            .map_err(|error| format!("fixture {id} failed at decision {decision}: {error}"))?;
        if drain_observations {
            let events = game.take_observation_events();
            black_box(Observation::new(&game, &events));
        }
    }
    check_shape(id, &game)?;
    Ok(game)
}

fn check_shape(id: &str, game: &Game) -> Result<(), String> {
    if game.is_game_over() {
        return Err(format!("fixture {id} unexpectedly reached game over"));
    }
    let action_count = game
        .action_space()
        .ok_or_else(|| format!("fixture {id} has no action space"))?
        .actions
        .len();
    let cards = game.state.cards.len();
    let slots = game.state.permanents.len();
    let events = game.state.events.len();
    match id {
        "interactive-midgame-48-w4-v1" if action_count != 2 => Err(format!(
            "midgame fixture drift: expected 2 root actions, got {action_count}"
        )),
        "interactive-heavy-80-w4-v1" if cards != 120 || slots != 24 || events != 512 => {
            Err(format!(
                "heavy fixture drift: expected cards/slots/events 120/24/512, got {cards}/{slots}/{events}"
            ))
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search_state::witness;

    #[test]
    fn fixtures_are_exact_and_deterministic() {
        for id in ["interactive-midgame-48-w4-v1", "interactive-heavy-80-w4-v1"] {
            let first = build_fixture(id).expect("first fixture build");
            let second = build_fixture(id).expect("second fixture build");
            assert_eq!(
                witness(&first).authority.hash,
                witness(&second).authority.hash,
                "fixture {id}"
            );
        }
    }

    #[test]
    fn historical_fixtures_are_not_reinterpreted_under_current_rules() {
        for id in ["interactive-midgame-48-v1", "interactive-heavy-80-v1"] {
            assert!(build_fixture(id).is_err(), "historical fixture {id}");
        }
    }
}
