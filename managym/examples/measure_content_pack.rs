use std::{
    alloc::{GlobalAlloc, Layout, System},
    collections::BTreeMap,
    hint::black_box,
    mem::size_of,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

use managym::{
    agent::env::Env,
    cardsets::alpha::CardRegistry,
    flow::game::GameState,
    state::{card::Card, player::PlayerConfig},
    Game,
};

const STEP_COUNT: u64 = 20_000;
const CLONE_COUNT: u64 = 20_000;
const ROLLOUT_REPEATS: u64 = 8;
const ROLLOUT_WORLDS: usize = 4;
const ROLLOUTS_PER_WORLD: usize = 4;
const ROLLOUT_STEP_CAP: usize = 2_000;

struct CountingAllocator;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        System.alloc_zeroed(layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

#[derive(Clone, Copy)]
struct AllocationSnapshot {
    allocations: u64,
    bytes: u64,
}

impl AllocationSnapshot {
    fn now() -> Self {
        Self {
            allocations: ALLOCATIONS.load(Ordering::Relaxed),
            bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
        }
    }

    fn since(self) -> Self {
        let now = Self::now();
        Self {
            allocations: now.allocations - self.allocations,
            bytes: now.bytes - self.bytes,
        }
    }
}

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

fn configs() -> Vec<PlayerConfig> {
    vec![
        PlayerConfig::new("hero", interactive_deck()),
        PlayerConfig::new("villain", interactive_deck()),
    ]
}

fn make_game(seed: u64) -> Game {
    Game::new(configs(), seed, true)
}

fn make_env(seed: u64) -> Env {
    let mut env = Env::new(seed, true, false, false);
    env.reset(configs())
        .expect("benchmark reset should succeed");
    env
}

fn advance_game(game: &mut Game, steps: usize, seed: u64) {
    use rand::Rng;

    game.reseed(seed);
    for _ in 0..steps {
        if game.is_game_over() {
            break;
        }
        let action_count = game.action_space().map_or(0, |space| space.actions.len());
        if action_count == 0 {
            break;
        }
        let action = game.state.rng.gen_range(0..action_count);
        game.step(action).expect("benchmark step should succeed");
        let _ = game.take_observation_events();
    }
}

fn advance_env(env: &mut Env, steps: usize) {
    for _ in 0..steps {
        if env.is_game_over() {
            break;
        }
        let action = env
            .random_action_index()
            .expect("benchmark action should exist");
        env.step(action as i64)
            .expect("benchmark step should succeed");
    }
}

fn measure_steps() {
    let mut env = make_env(0x179);
    advance_env(&mut env, 200);
    if env.is_game_over() {
        env.set_seed(0x179);
        env.reset(configs()).expect("warmup reset should succeed");
    }

    let allocations = AllocationSnapshot::now();
    let started = Instant::now();
    let mut completed = 0_u64;
    let mut games = 0_u64;
    while completed < STEP_COUNT {
        if env.is_game_over() {
            games += 1;
            env.set_seed(0x179 + games);
            env.reset(configs())
                .expect("benchmark reset should succeed");
        }
        let action = env
            .random_action_index()
            .expect("benchmark action should exist");
        env.step(action as i64)
            .expect("benchmark step should succeed");
        completed += 1;
    }
    let elapsed = started.elapsed();
    let allocated = allocations.since();

    println!("step.count={completed}");
    println!("step.seconds={:.9}", elapsed.as_secs_f64());
    println!(
        "step.per_second={:.3}",
        completed as f64 / elapsed.as_secs_f64()
    );
    println!(
        "step.allocations_per_step={:.3}",
        allocated.allocations as f64 / completed as f64
    );
    println!(
        "step.allocated_bytes_per_step={:.3}",
        allocated.bytes as f64 / completed as f64
    );
}

fn measure_clones() {
    let mut game = make_game(0x179);
    advance_game(&mut game, 80, 0xc10e);
    assert!(
        !game.is_game_over(),
        "clone fixture ended before measurement"
    );

    for _ in 0..200 {
        black_box(black_box(&game.state).clone());
    }

    let allocations = AllocationSnapshot::now();
    let started = Instant::now();
    for _ in 0..CLONE_COUNT {
        black_box(black_box(&game.state).clone());
    }
    let elapsed = started.elapsed();
    let allocated = allocations.since();

    println!("clone.count={CLONE_COUNT}");
    println!("clone.seconds={:.9}", elapsed.as_secs_f64());
    println!(
        "clone.nanoseconds_per_clone={:.3}",
        elapsed.as_nanos() as f64 / CLONE_COUNT as f64
    );
    println!(
        "clone.per_second={:.3}",
        CLONE_COUNT as f64 / elapsed.as_secs_f64()
    );
    println!(
        "clone.allocations_per_clone={:.3}",
        allocated.allocations as f64 / CLONE_COUNT as f64
    );
    println!(
        "clone.allocated_bytes_per_clone={:.3}",
        allocated.bytes as f64 / CLONE_COUNT as f64
    );
    println!("fixture.cards={}", game.state.cards.len());
    println!("fixture.permanent_slots={}", game.state.permanents.len());
    println!("fixture.events={}", game.state.events.len());
}

fn rollout_fixture() -> Env {
    for seed in 0x179..0x189 {
        let mut env = make_env(seed);
        advance_env(&mut env, 48);
        if !env.is_game_over() && env.action_count().is_ok_and(|count| count > 1) {
            return env;
        }
    }
    panic!("could not build a live rollout fixture");
}

fn measure_rollouts() {
    let env = rollout_fixture();
    let action_count = env.action_count().expect("rollout action count");

    let _ = env
        .flat_mc_scores(1, 1, 0xbeef, ROLLOUT_STEP_CAP)
        .expect("rollout warmup should succeed");

    let allocations = AllocationSnapshot::now();
    let started = Instant::now();
    let mut simulations = 0_u64;
    let mut cap_hits = 0_u64;
    let mut score_checksum = 0.0_f64;
    for repeat in 0..ROLLOUT_REPEATS {
        let result = env
            .flat_mc_scores(
                ROLLOUT_WORLDS,
                ROLLOUTS_PER_WORLD,
                0xbeef + repeat,
                ROLLOUT_STEP_CAP,
            )
            .expect("rollout measurement should succeed");
        simulations += result.simulations;
        cap_hits += result.cap_hits;
        score_checksum += result.scores.iter().sum::<f64>();
    }
    let elapsed = started.elapsed();
    let allocated = allocations.since();

    println!("rollout.root_actions={action_count}");
    println!("rollout.repeats={ROLLOUT_REPEATS}");
    println!("rollout.worlds={ROLLOUT_WORLDS}");
    println!("rollout.rollouts_per_world={ROLLOUTS_PER_WORLD}");
    println!("rollout.simulations={simulations}");
    println!("rollout.cap_hits={cap_hits}");
    println!("rollout.seconds={:.9}", elapsed.as_secs_f64());
    println!(
        "rollout.simulations_per_second={:.3}",
        simulations as f64 / elapsed.as_secs_f64()
    );
    println!(
        "rollout.allocations_per_simulation={:.3}",
        allocated.allocations as f64 / simulations as f64
    );
    println!(
        "rollout.allocated_bytes_per_simulation={:.3}",
        allocated.bytes as f64 / simulations as f64
    );
    println!("rollout.score_checksum={score_checksum:.9}");
}

fn measure_sizes() {
    let registry = CardRegistry::default();
    let definition_count = registry.definitions().count();

    println!("size.card_bytes={}", size_of::<Card>());
    println!(
        "size.card_definition_bytes={}",
        size_of::<managym::state::card::CardDefinition>()
    );
    println!("size.game_state_bytes={}", size_of::<GameState>());
    println!("size.game_bytes={}", size_of::<Game>());
    println!("size.definition_store_bytes={}", size_of::<CardRegistry>());
    println!("size.definition_count={definition_count}");
}

fn main() {
    println!("benchmark=w2-179-content-pack-local-diagnostic");
    println!("profile=release");
    println!("seed=377");
    println!("deck=interactive-mirror-60x2");
    measure_sizes();
    measure_steps();
    measure_clones();
    measure_rollouts();
}
