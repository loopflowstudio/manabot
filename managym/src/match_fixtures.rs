//! Deterministic interactive-mirror mid-game positions.
//!
//! Each fixture plays a fixed seed and action tape from a fresh match and checks
//! its shape, so clone-cost and witness tests measure the same exact state.

use std::{collections::BTreeMap, hint::black_box};

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    agent::{action::ActionSpaceKind, observation::Observation},
    search_state::witness,
    state::player::PlayerConfig,
    Game,
};

pub const FIXTURE_IDS: [&str; 2] = ["interactive-midgame-48-w4-v1", "interactive-heavy-80-w4-v1"];
pub const SETUP_SEED: u64 = 377;
pub const HEAVY_ACTION_SEED: u64 = 0xc10e;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FixtureSummary {
    pub id: String,
    pub deck: String,
    pub setup_seed: u64,
    pub action_seed: Option<u64>,
    pub action_tape: Vec<usize>,
    pub semantic_hash: String,
    pub action_hash: String,
    pub observation_hash: String,
    pub action_kind: String,
    pub action_count: usize,
    pub card_count: usize,
    pub allocated_permanent_slots: usize,
    pub live_permanent_count: usize,
    pub token_count: usize,
    pub plus1_counter_count: i32,
    pub stack_count: usize,
    pub committed_event_count: usize,
    pub pending_event_count: usize,
    pub observation_event_count: usize,
    pub snapshot_bytes: usize,
}

pub fn interactive_deck() -> BTreeMap<String, usize> {
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

pub fn build_fixture(id: &str) -> Result<(Game, FixtureSummary), String> {
    let mut game = new_match(SETUP_SEED);
    let (count, action_seed, drain_observations) = match id {
        "interactive-midgame-48-w4-v1" => (48, None, true),
        "interactive-heavy-80-w4-v1" => {
            game.reseed(HEAVY_ACTION_SEED);
            (80, Some(HEAVY_ACTION_SEED), false)
        }
        _ => return Err(format!("unknown fixture {id}")),
    };
    let mut tape = Vec::with_capacity(count);
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
        tape.push(action);
        game.step(action)
            .map_err(|error| format!("fixture {id} failed at decision {decision}: {error}"))?;
        if drain_observations {
            let events = game.take_observation_events();
            black_box(Observation::new(&game, &events));
        }
    }
    fixture_with_summary(id, action_seed, tape, game)
}

fn fixture_with_summary(
    id: &str,
    action_seed: Option<u64>,
    action_tape: Vec<usize>,
    game: Game,
) -> Result<(Game, FixtureSummary), String> {
    if game.is_game_over() {
        return Err(format!("fixture {id} unexpectedly reached game over"));
    }
    let state_witness = witness(&game);
    let action_space = game
        .action_space()
        .ok_or_else(|| format!("fixture {id} has no action space"))?;
    let summary = FixtureSummary {
        id: id.to_string(),
        deck: "interactive-mirror-60-v1".to_string(),
        setup_seed: SETUP_SEED,
        action_seed,
        action_tape,
        semantic_hash: state_witness.authority.hash.clone(),
        action_hash: state_witness.legal_surface.hash.clone(),
        observation_hash: state_witness.acting_projection.hash.clone(),
        action_kind: action_kind_name(action_space.kind).to_string(),
        action_count: action_space.actions.len(),
        card_count: game.state.cards.len(),
        allocated_permanent_slots: game.state.permanents.len(),
        live_permanent_count: game.state.permanents.iter().flatten().count(),
        token_count: game.state.cards.iter().filter(|card| card.is_token).count(),
        plus1_counter_count: game
            .state
            .permanents
            .iter()
            .flatten()
            .map(|permanent| permanent.plus1_counters)
            .sum(),
        stack_count: game.state.stack_objects.len(),
        committed_event_count: game.state.events.len(),
        pending_event_count: game.state.pending_events.len(),
        observation_event_count: game.state.observation_events.len(),
        snapshot_bytes: state_witness.authority.bytes.len(),
    };
    match id {
        "interactive-midgame-48-w4-v1" if summary.action_count != 2 => {
            return Err(format!(
                "midgame fixture drift: expected 2 root actions, got {}",
                summary.action_count
            ));
        }
        "interactive-heavy-80-w4-v1"
            if summary.card_count != 120
                || summary.allocated_permanent_slots != 24
                || summary.committed_event_count != 505 =>
        {
            return Err(format!(
                "heavy fixture drift: expected cards/slots/events 120/24/505, got {}/{}/{}",
                summary.card_count,
                summary.allocated_permanent_slots,
                summary.committed_event_count
            ));
        }
        _ => {}
    }
    Ok((game, summary))
}

fn action_kind_name(kind: ActionSpaceKind) -> &'static str {
    match kind {
        ActionSpaceKind::GameOver => "game_over",
        ActionSpaceKind::Priority => "priority",
        ActionSpaceKind::DeclareAttacker => "declare_attacker",
        ActionSpaceKind::DeclareBlocker => "declare_blocker",
        ActionSpaceKind::ChooseTarget => "choose_target",
        ActionSpaceKind::Scry => "scry",
        ActionSpaceKind::LookAndSelect => "look_and_select",
        ActionSpaceKind::PayOrNot => "pay_or_not",
        ActionSpaceKind::Modal => "modal",
        ActionSpaceKind::Learn => "learn",
        ActionSpaceKind::Waterbend => "waterbend",
        ActionSpaceKind::Discard => "discard",
        ActionSpaceKind::LegendRule => "legend_rule",
        ActionSpaceKind::OrderTriggers => "order_triggers",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixtures_are_exact_and_deterministic() {
        for id in FIXTURE_IDS {
            let (_, first) = build_fixture(id).expect("first fixture build");
            let (_, second) = build_fixture(id).expect("second fixture build");
            assert_eq!(first.semantic_hash, second.semantic_hash, "fixture {id}");
            assert_eq!(first.action_tape, second.action_tape, "fixture {id}");
        }
    }

    #[test]
    fn historical_fixtures_are_not_reinterpreted_under_current_rules() {
        for id in ["interactive-midgame-48-v1", "interactive-heavy-80-v1"] {
            assert!(build_fixture(id).is_err(), "historical fixture {id}");
        }
    }
}
