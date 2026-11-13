//! Bound payment search on the reported slow game and unpayable large boards.
//! Debug CI has the same counters as bench_payment; release wheels omit them.
#![cfg(any(debug_assertions, feature = "benchmark"))]

use std::cell::Cell;
use std::collections::BTreeMap;

use managym::agent::action::ActionType;
use managym::agent::env::Env;
use managym::agent::opponent::OpponentPolicy;
use managym::agent::vector_env::VectorEnv;
use managym::benchmark_payments::{reset, snapshot, with_call_limit, Metrics};
use managym::state::player::{DecklistDisclosure, PlayerConfig};
use sha2::{Digest, Sha256};

const REPLAY: &str = include_str!("fixtures/payment_search/manabot_slow_step.py");
const CALL_LIMIT: u64 = 1_000;

// Read only the reporter's data literals, as bench_payment does. Never execute
// or rewrite the historical script to adapt it to the current Python API.
fn literal<'a>(source: &'a str, name: &str, close: &str) -> &'a str {
    let (_, value) = source.split_once(&format!("\n{name} = ")).unwrap();
    value[..value.find(close).unwrap() + close.len()].trim()
}

fn assert_decision_matches(scalar: &Env, vector: &VectorEnv, boundary: usize) {
    assert_eq!(
        vector.current_agent_indices(),
        vec![scalar.current_agent_index().unwrap() as i64],
        "chooser after {boundary} actions"
    );
    assert_eq!(
        vector.semantic_decision_frames_json(None).unwrap(),
        vec![Some(
            serde_json::to_string(&scalar.semantic_decision_frame().unwrap()).unwrap()
        )],
        "complete decision after {boundary} actions"
    );
}

#[test]
fn recorded_game_keeps_payment_search_bounded() {
    assert_eq!(
        format!("{:x}", Sha256::digest(REPLAY.as_bytes())),
        "42c6bec5f6a94fbb189ef40a92c7f197489e75756d3e813eaff063bd996596fa",
        "keep the reporter's fixture unchanged"
    );
    let seed: u64 = literal(REPLAY, "SEED", "\n").parse().unwrap();
    let decks: Vec<BTreeMap<String, usize>> =
        serde_json::from_str(literal(REPLAY, "DECKS", "\n]\n")).unwrap();
    let actions: Vec<i64> = serde_json::from_str(literal(REPLAY, "ACTIONS", "]\n")).unwrap();
    assert_eq!(seed, 5064105440874770157);
    assert_eq!(actions.len(), 426);
    assert_eq!(decks.len(), 2);
    let players: Vec<PlayerConfig> = decks
        .into_iter()
        .enumerate()
        .map(|(seat, deck)| {
            assert_eq!(deck.values().sum::<usize>(), 40);
            let mut player = PlayerConfig::new(format!("seat {seat}"), deck).submit_to_world();
            player.disclosure = DecklistDisclosure::Closed;
            player
        })
        .collect();
    let mut scalar = Env::new(seed, true, false, false);
    let mut vector = VectorEnv::new(1, seed, true, OpponentPolicy::None);
    with_call_limit(CALL_LIMIT, "replay reset", || {
        let (obs, _) = scalar.reset(players.clone()).unwrap();
        let batch = vector.reset_all(players).unwrap();
        assert_eq!(obs, batch[0].0);
    });
    assert_decision_matches(&scalar, &vector, 0);
    reset();
    for (index, &action) in actions.iter().enumerate() {
        let boundary = index + 1;
        let context = format!(
            "action {boundary} of {} (action index {action})",
            actions.len()
        );
        assert!(
            (0..scalar.action_count().unwrap() as i64).contains(&action),
            "{context}"
        );
        let (obs, reward, terminated, truncated, _) =
            with_call_limit(CALL_LIMIT, &format!("scalar {context}"), || {
                scalar.step(action)
            })
            .unwrap();
        // The recording ends at the formerly slow step, not at game over.
        // Reject early termination rather than comparing an autoreset root.
        assert!(
            !terminated && !truncated,
            "recording ended early: {context}"
        );
        let written = Cell::new(false);
        let infos = with_call_limit(CALL_LIMIT, &format!("vector {context}"), || {
            vector.step_isolated_into(&[action], &[true], false, |slot, actual, r, done, cut| {
                assert_eq!(slot, 0);
                assert_eq!((r, done, cut), (reward, terminated, truncated), "{context}");
                assert_eq!(actual, &obs, "observation after {context}");
                assert!(!written.replace(true), "duplicate write: {context}");
                Ok(())
            })
        })
        .unwrap();
        assert!(
            !infos[0].contains_key("game_error"),
            "{context}: {:?}",
            infos[0]
        );
        assert!(written.get(), "missing observation: {context}");
        assert_decision_matches(&scalar, &vector, boundary);
    }
    let work = snapshot();
    assert!(work.plans > 0 && work.calls > 0 && work.states > 0);
    assert!(
        work.max_sources >= 13,
        "recording must reach the large board"
    );
    assert!(work.max_plan_calls <= CALL_LIMIT);
    eprintln!(
        "scalar + vector replay: {}",
        serde_json::to_string(&work).unwrap()
    );
}

fn unpayable_board(source: &str, lands: usize) -> Metrics {
    let context = format!("{source} plus {lands} Forests, unpayable {{U}}{{R}}");
    with_call_limit(CALL_LIMIT, &context, || {
        let deck = BTreeMap::from([
            ("Forest".into(), lands),
            (source.into(), 1),
            ("Sokka, Bold Boomeranger".into(), 1),
            ("Plains".into(), 20),
        ]);
        let mut env = Env::new(7, true, false, false);
        env.reset(vec![
            PlayerConfig::new("subject", deck),
            PlayerConfig::new("idle", BTreeMap::from([("Plains".into(), 40)])),
        ])
        .unwrap();
        env.scenario_clear_hand(0).unwrap();
        for _ in 0..lands {
            env.scenario_force_battlefield(0, "Forest", true).unwrap();
        }
        env.scenario_force_battlefield(0, source, true).unwrap();
        env.scenario_force_card_in_hand(0, "Sokka, Bold Boomeranger")
            .unwrap();
        reset();
        let obs = env.scenario_refresh().unwrap();
        assert!(
            obs.action_space
                .actions
                .iter()
                .all(|action| action.action_type != ActionType::PriorityCastSpell),
            "{context} must remain unpayable"
        );
        let work = snapshot();
        assert_eq!(work.max_sources, (lands + 1) as u64, "{context}");
        assert!(work.plans > 0 && work.states > 0, "{context}");
        eprintln!("{context}: {}", serde_json::to_string(&work).unwrap());
        work
    })
}

#[test]
fn unpayable_boards_do_not_grow_with_lands() {
    for source in [
        "Hermitic Herbalist",
        "Airship Engine Room",
        "Prophetic Prism",
    ] {
        let small = unpayable_board(source, 6);
        let large = unpayable_board(source, 12);
        assert_eq!(
            small.states, large.states,
            "{source}: extra Forests must not expand search"
        );
    }
}
