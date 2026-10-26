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
use std::{
    alloc::{GlobalAlloc, Layout, System},
    hint::black_box,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    time::Instant,
};

struct CountingAllocator;
static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(size as u64, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

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
fn main() {
    let mut config = ObservationEncoderConfig::default();
    config.relations = !std::env::args().any(|arg| arg == "--legacy");
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
                ALLOCATIONS.store(0, Ordering::Relaxed);
                ALLOCATED_BYTES.store(0, Ordering::Relaxed);
                COUNTING.store(true, Ordering::Relaxed);
                let start = Instant::now();
                for _ in 0..repeats {
                    for buffer in &mut buffers {
                        encode_into(black_box(observation), &config, buffer.buffers_mut()).unwrap();
                        black_box(&*buffer);
                    }
                }
                let seconds = start.elapsed().as_secs_f64();
                COUNTING.store(false, Ordering::Relaxed);
                let count = repeats * batch;
                println!(
                    "{}",
                    serde_json::json!({"corpus": digest, "case": name, "actions": observation.action_space.actions.len(), "profile": "release", "batch": batch, "repetition": repetition, "encodes": count, "seconds": seconds, "encodes_per_second": count as f64 / seconds, "ns_per_encode": seconds * 1e9 / count as f64, "allocations_per_encode": ALLOCATIONS.load(Ordering::Relaxed) as f64 / count as f64, "allocated_bytes_per_encode": ALLOCATED_BYTES.load(Ordering::Relaxed) as f64 / count as f64, "bytes_written_per_encode": bytes(&buffers[0])})
                );
            }
        }
    }
}
