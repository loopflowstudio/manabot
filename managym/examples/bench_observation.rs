//! Seeded encode-only attribution of the policy transport: each timed encode
//! appends one observation to a `PackedRows` collection batch. Run in release;
//! JSON lines retain corpus identity, transported bytes, allocation traffic of
//! one append and repeated timing.
use managym::{
    agent::{
        action::ActionType,
        env::Env,
        observation::Observation,
        observation_encoder::ObservationEncoderConfig,
        observation_layout::{configured_arrays, ArrayLayout},
        observation_rows::PackedRows,
    },
    flow::turn::PhaseKind,
    state::player::PlayerConfig,
};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, hint::black_box, time::Instant};

fn root(large: bool) -> (Env, Observation) {
    let mut env = Env::new(23, false, false);
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
    while obs.turn.phase != PhaseKind::PrecombatMain {
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

/// Bytes one batch transports: every table's row offsets and packed values.
fn bytes(rows: &PackedRows) -> usize {
    rows.tables
        .values()
        .map(|table| 4 * table.offsets.len() + table.values.len())
        .sum()
}
fn input_digest(observation: &Observation) -> String {
    let mut hash = Sha256::new();
    hash.update(observation.to_json());
    hash.update(serde_json::to_vec(&observation.continuation).unwrap());
    format!("{hash:x}", hash = hash.finalize())
}

fn encoded_digest(rows: &PackedRows) -> String {
    let mut hash = Sha256::new();
    for (name, table) in &rows.tables {
        hash.update(name.as_bytes());
        for offset in &table.offsets {
            hash.update(offset.to_le_bytes());
        }
        hash.update(&table.values);
    }
    format!("{hash:x}", hash = hash.finalize())
}

/// One collection batch of `batch` copies of `observation`, as rollout
/// collection packs it for a policy.
fn pack(
    observation: &Observation,
    config: &ObservationEncoderConfig,
    arrays: &BTreeMap<String, ArrayLayout>,
    batch: usize,
) -> PackedRows {
    let mut rows = PackedRows::new(arrays);
    for _ in 0..batch {
        rows.append(observation, config).unwrap();
    }
    rows
}

fn main() {
    let config = ObservationEncoderConfig::default();
    let arrays = configured_arrays(&config).unwrap();
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
            for _ in 0..100 {
                black_box(pack(observation, &config, &arrays, batch));
            }
            let single = pack(observation, &config, &arrays, 1);
            let repeats = if batch == 1 { 10000 } else { 200 };
            for repetition in 0..5 {
                let start = Instant::now();
                for _ in 0..repeats {
                    black_box(pack(black_box(observation), &config, &arrays, batch));
                }
                let seconds = start.elapsed().as_secs_f64();
                let mut rows = PackedRows::new(&arrays);
                managym::allocator::start();
                rows.append(observation, &config).unwrap();
                let (allocations, allocated_bytes) = managym::allocator::stop();
                black_box(rows);
                let count = repeats * batch;
                println!(
                    "{}",
                    serde_json::json!({"corpus": digest, "case": name, "actions": observation.action_space.actions.len(), "profile": "release", "batch": batch, "repetition": repetition, "encodes": count, "seconds": seconds, "encodes_per_second": count as f64 / seconds, "ns_per_encode": seconds * 1e9 / count as f64, "allocations_per_encode": allocations, "allocated_bytes_per_encode": allocated_bytes, "output_bytes_per_encode": bytes(&single), "input_digest": input_digest(observation), "encoded_digest": encoded_digest(&single), "counting_during_timing": false})
                );
            }
        }
    }
}
