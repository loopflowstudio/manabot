//! Seeded encode-only attribution. Run in release; JSON lines retain corpus
//! identity, buffer size, allocation traffic, bytes written and repeated timing.
use managym::{
    agent::{
        action::ActionType,
        env::Env,
        observation::Observation,
        observation_encoder::{encode, encode_into, EncodedObservation, ObservationEncoderConfig},
    },
    state::player::PlayerConfig,
};
use sha2::{Digest, Sha256};
use std::{hint::black_box, time::Instant};

fn root(large: bool) -> (Env, Observation) {
    let mut env = Env::new(23, false, false, false);
    let deck = [
        ("Swamp".into(), 20),
        ("Deadly Precision".into(), 40),
        ("Gray Ogre".into(), 4),
    ]
    .into();
    let (mut obs, _) = env
        .reset(vec![
            PlayerConfig::new("a", deck),
            PlayerConfig::new(
                "b",
                [
                    ("Swamp".into(), 20),
                    ("Deadly Precision".into(), 40),
                    ("Gray Ogre".into(), 4),
                ]
                .into(),
            ),
        ])
        .unwrap();
    while obs.turn.phase as i32 != 1 {
        let pass = obs
            .action_space
            .actions
            .iter()
            .position(|a| a.action_type == ActionType::PriorityPassPriority)
            .unwrap();
        obs = env.step(pass as i64).unwrap().0;
    }
    let actor = env.current_agent_index().unwrap();
    env.scenario_clear_hand(actor).unwrap();
    for _ in 0..if large { 32 } else { 1 } {
        env.scenario_force_card_in_hand(actor, "Deadly Precision")
            .unwrap();
    }
    env.scenario_force_battlefield(actor, "Swamp", true)
        .unwrap();
    env.scenario_force_battlefield(actor, "Gray Ogre", true)
        .unwrap();
    env.scenario_force_battlefield(1 - actor, "Gray Ogre", true)
        .unwrap();
    obs = env.scenario_refresh().unwrap();
    (env, obs)
}
fn corpus() -> Vec<(&'static str, Observation)> {
    let (mut env, mut obs) = root(false);
    let mut result = vec![("priority", obs.clone())];
    let cast = obs
        .action_space
        .actions
        .iter()
        .position(|a| a.action_type == ActionType::PriorityCastSpell)
        .unwrap();
    obs = env.step(cast as i64).unwrap().0;
    result.push(("target", obs));
    obs = env.step(0).unwrap().0;
    result.push(("additional_cost", obs));
    result.push(("large_legal", root(true).1));
    result
}

fn bytes(encoded: &EncodedObservation) -> usize {
    4 * (encoded.relations.len()
        + encoded.semantic_cards.len()
        + encoded.known_hand.len()
        + encoded.agent_player.len()
        + encoded.opponent_player.len()
        + encoded.agent_cards.len()
        + encoded.opponent_cards.len()
        + encoded.agent_permanents.len()
        + encoded.opponent_permanents.len()
        + encoded.actions.len()
        + encoded.events.len()
        + encoded.action_focus.len()
        + encoded.agent_player_valid.len()
        + encoded.opponent_player_valid.len()
        + encoded.agent_cards_valid.len()
        + encoded.opponent_cards_valid.len()
        + encoded.agent_permanents_valid.len()
        + encoded.opponent_permanents_valid.len()
        + encoded.actions_valid.len()
        + encoded.events_valid.len())
}
fn input_digest(observation: &Observation) -> String {
    let mut hash = Sha256::new();
    hash.update(observation.to_json());
    hash.update(serde_json::to_vec(&observation.continuation).unwrap());
    format!("{hash:x}", hash = hash.finalize())
}

fn encoded_digest(encoded: &EncodedObservation) -> String {
    let mut hash = Sha256::new();
    for value in &encoded.semantic_cards {
        hash.update(value.to_le_bytes());
    }
    for value in &encoded.known_hand {
        hash.update(value.to_le_bytes());
    }
    for value in &encoded.agent_player {
        hash.update(value.to_le_bytes());
    }
    for value in &encoded.opponent_player {
        hash.update(value.to_le_bytes());
    }
    for value in &encoded.agent_cards {
        hash.update(value.to_le_bytes());
    }
    for value in &encoded.opponent_cards {
        hash.update(value.to_le_bytes());
    }
    for value in &encoded.agent_permanents {
        hash.update(value.to_le_bytes());
    }
    for value in &encoded.opponent_permanents {
        hash.update(value.to_le_bytes());
    }
    for value in &encoded.actions {
        hash.update(value.to_le_bytes());
    }
    for value in &encoded.events {
        hash.update(value.to_le_bytes());
    }
    for value in &encoded.action_focus {
        hash.update(value.to_le_bytes());
    }
    for value in &encoded.agent_player_valid {
        hash.update(value.to_le_bytes());
    }
    for value in &encoded.opponent_player_valid {
        hash.update(value.to_le_bytes());
    }
    for value in &encoded.agent_cards_valid {
        hash.update(value.to_le_bytes());
    }
    for value in &encoded.opponent_cards_valid {
        hash.update(value.to_le_bytes());
    }
    for value in &encoded.agent_permanents_valid {
        hash.update(value.to_le_bytes());
    }
    for value in &encoded.opponent_permanents_valid {
        hash.update(value.to_le_bytes());
    }
    for value in &encoded.actions_valid {
        hash.update(value.to_le_bytes());
    }
    for value in &encoded.events_valid {
        hash.update(value.to_le_bytes());
    }
    for value in &encoded.relations {
        hash.update(value.to_le_bytes());
    }
    format!("{hash:x}", hash = hash.finalize())
}

fn main() {
    let config = ObservationEncoderConfig {
        relations: !std::env::args().any(|arg| arg == "--legacy"),
        ..Default::default()
    };
    let corpus = corpus();
    let digest = format!(
        "{:x}",
        Sha256::digest(
            corpus
                .iter()
                .map(|(_, obs)| obs.to_json())
                .collect::<Vec<_>>()
                .join("\n")
        )
    );
    for batch in [1, 64] {
        for (name, observation) in &corpus {
            let mut buffers: Vec<_> = (0..batch)
                .map(|_| encode(observation, &config).unwrap())
                .collect();
            for _ in 0..100 {
                for buffer in &mut buffers {
                    encode_into(observation, &config, buffer.buffers_mut()).unwrap();
                }
            }
            let repeats = if batch == 1 { 10000 } else { 200 };
            for repetition in 0..5 {
                let start = Instant::now();
                for _ in 0..repeats {
                    for buffer in &mut buffers {
                        encode_into(black_box(observation), &config, buffer.buffers_mut()).unwrap();
                        black_box(&*buffer);
                    }
                }
                let seconds = start.elapsed().as_secs_f64();
                managym::benchmark_allocations::start();
                encode_into(observation, &config, buffers[0].buffers_mut()).unwrap();
                let (allocations, allocated_bytes) = managym::benchmark_allocations::stop();
                let count = repeats * batch;
                println!(
                    "{}",
                    serde_json::json!({"corpus": digest, "case": name, "actions": observation.action_space.actions.len(), "profile": "release", "batch": batch, "repetition": repetition, "encodes": count, "seconds": seconds, "encodes_per_second": count as f64 / seconds, "ns_per_encode": seconds * 1e9 / count as f64, "allocations_per_encode": allocations, "allocated_bytes_per_encode": allocated_bytes, "output_bytes_per_encode": bytes(&buffers[0]), "input_digest": input_digest(observation), "encoded_digest": encoded_digest(&buffers[0]), "counting_during_timing": false})
                );
            }
        }
    }
}
