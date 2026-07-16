//! Contract-v1 search-state benchmark support.
//!
//! The workload code is generic over [`BranchDriver`], so every candidate runs
//! byte-identical logical work and the cells are matched by construction. Two
//! implementations exist: the full-clone reference baseline, and compact clone
//! plus undo. Dense page-COW storage intentionally remains future work.

use std::{collections::BTreeMap, hint::black_box, time::Instant};

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    agent::{action::ActionSpaceKind, env::Env, observation::Observation},
    flow::search::mix_seed,
    state::player::PlayerConfig,
    Game,
};

pub use crate::search_state::{
    witness, ApplyReceipt, AuthorityFingerprint, BenchCommand, BranchDriver, ClonePlusUndoDriver,
    ContractDiagnostics, DriverCounters, FullCloneDriver, LegalSurfaceFingerprint,
    SearchStateWitness, ViewerProjectionWitness, SEARCH_WITNESS_SCHEMA_VERSION,
};

pub const CONTRACT_ID: &str = "manabot.search-branching.v1";
pub const DRIVER_ID: &str = "full_clone/current_game_v1";
pub const CLONE_PLUS_UNDO_DRIVER_ID: &str = "compact_clone_undo/current_game_v1";

/// The branching candidates under measurement. Selecting a candidate changes
/// only how a branch is taken and released; it never changes the workload's
/// logical work, so `result_checksum` must agree across every candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DriverKind {
    FullClone,
    ClonePlusUndo,
}

impl DriverKind {
    pub fn parse(id: &str) -> Result<Self, String> {
        match id {
            DRIVER_ID => Ok(Self::FullClone),
            CLONE_PLUS_UNDO_DRIVER_ID => Ok(Self::ClonePlusUndo),
            other => Err(format!("unknown driver {other}")),
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            Self::FullClone => DRIVER_ID,
            Self::ClonePlusUndo => CLONE_PLUS_UNDO_DRIVER_ID,
        }
    }
}
pub const MANIFEST_SCHEMA: u32 = 1;
pub const RESULT_SCHEMA: u32 = 1;
pub const SETUP_SEED: u64 = 377;
pub const HEAVY_ACTION_SEED: u64 = 0xc10e;
pub const WARMUP_SEED: u64 = 0xbeee;
pub const EQUIVALENCE_SEEDS: [u64; 4] = [0x5eed, 0x5eee, 0x5eef, 0x5ef0];
pub const MEASURED_SEEDS: [u64; 8] = [
    0xbeef, 0xbef0, 0xbef1, 0xbef2, 0xbef3, 0xbef4, 0xbef5, 0xbef6,
];
pub const MAX_STEPS: usize = 2_000;
pub const STEP_WARMUP: usize = 2_000;
pub const STEP_SAMPLES: usize = 20_000;
pub const CLONE_WARMUP: usize = 200;
pub const CLONE_SAMPLES: usize = 20_000;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkManifest {
    pub schema_version: u32,
    pub contract_id: String,
    pub driver: String,
    pub snapshot_schema: u32,
    pub seed_derivation: String,
    pub max_steps: usize,
    pub equivalence_seeds: Vec<u64>,
    pub warmup_seed: u64,
    pub measured_seeds: Vec<u64>,
    pub fixtures: Vec<String>,
    pub workloads: Vec<WorkloadSpec>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkloadSpec {
    pub id: String,
    pub fixture: String,
    pub shape: WorkloadShape,
    pub workers: usize,
    pub actors_per_worker: usize,
    pub worlds: usize,
    pub rollouts_per_world: usize,
    pub policy_plies: usize,
    pub warmup_count: usize,
    pub measured_count: usize,
    pub primary_evidence: bool,
}

impl WorkloadSpec {
    pub fn simulations_per_action(&self) -> usize {
        self.worlds * self.rollouts_per_world
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WorkloadShape {
    Step,
    CloneLatency,
    Sequential,
    Retained,
}

pub fn manifest() -> BenchmarkManifest {
    manifest_for(DriverKind::FullClone)
}

/// The manifest is identical for every candidate except its driver name: the
/// fixtures, seeds, and workload dimensions are the matched contract.
pub fn manifest_for(driver: DriverKind) -> BenchmarkManifest {
    BenchmarkManifest {
        schema_version: MANIFEST_SCHEMA,
        contract_id: CONTRACT_ID.to_string(),
        driver: driver.id().to_string(),
        // Preserve the v1 manifest field name for artifact compatibility.
        snapshot_schema: SEARCH_WITNESS_SCHEMA_VERSION,
        seed_derivation: "worker=mix(root,worker); actor=mix(worker,actor); world=mix(actor,world); rollout=mix(world,action*R+rollout+1); policy=mix(rollout,ply)".to_string(),
        max_steps: MAX_STEPS,
        equivalence_seeds: EQUIVALENCE_SEEDS.to_vec(),
        warmup_seed: WARMUP_SEED,
        measured_seeds: MEASURED_SEEDS.to_vec(),
        fixtures: vec![
            "interactive-midgame-48-v1".to_string(),
            "interactive-heavy-80-v1".to_string(),
        ],
        workloads: vec![
            WorkloadSpec {
                id: "step-v1".to_string(),
                fixture: "interactive-midgame-48-v1".to_string(),
                shape: WorkloadShape::Step,
                workers: 1,
                actors_per_worker: 1,
                worlds: 1,
                rollouts_per_world: 1,
                policy_plies: 0,
                warmup_count: STEP_WARMUP,
                measured_count: STEP_SAMPLES,
                primary_evidence: false,
            },
            WorkloadSpec {
                id: "clone-v1".to_string(),
                fixture: "interactive-heavy-80-v1".to_string(),
                shape: WorkloadShape::CloneLatency,
                workers: 1,
                actors_per_worker: 1,
                worlds: 1,
                rollouts_per_world: 1,
                policy_plies: 0,
                warmup_count: CLONE_WARMUP,
                measured_count: CLONE_SAMPLES,
                primary_evidence: false,
            },
            WorkloadSpec {
                id: "flat-single-64-v1".to_string(),
                fixture: "interactive-midgame-48-v1".to_string(),
                shape: WorkloadShape::Sequential,
                workers: 1,
                actors_per_worker: 1,
                worlds: 16,
                rollouts_per_world: 4,
                policy_plies: 0,
                warmup_count: 1,
                measured_count: MEASURED_SEEDS.len(),
                primary_evidence: true,
            },
            WorkloadSpec {
                id: "flat-saturated-64-v1".to_string(),
                fixture: "interactive-midgame-48-v1".to_string(),
                shape: WorkloadShape::Sequential,
                workers: 8,
                actors_per_worker: 1,
                worlds: 16,
                rollouts_per_world: 4,
                policy_plies: 0,
                warmup_count: 1,
                measured_count: MEASURED_SEEDS.len(),
                primary_evidence: true,
            },
            WorkloadSpec {
                id: "retained-single-8-v1".to_string(),
                fixture: "interactive-heavy-80-v1".to_string(),
                shape: WorkloadShape::Retained,
                workers: 1,
                actors_per_worker: 1,
                worlds: 8,
                rollouts_per_world: 1,
                policy_plies: 8,
                warmup_count: 1,
                measured_count: MEASURED_SEEDS.len(),
                primary_evidence: true,
            },
            WorkloadSpec {
                id: "retained-saturated-16-v1".to_string(),
                fixture: "interactive-heavy-80-v1".to_string(),
                shape: WorkloadShape::Retained,
                workers: 1,
                actors_per_worker: 8,
                worlds: 16,
                rollouts_per_world: 1,
                policy_plies: 8,
                warmup_count: 1,
                measured_count: MEASURED_SEEDS.len(),
                primary_evidence: true,
            },
        ],
    }
}

pub fn workload(id: &str) -> Result<WorkloadSpec, String> {
    manifest()
        .workloads
        .into_iter()
        .find(|spec| spec.id == id)
        .ok_or_else(|| format!("unknown workload {id}"))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkerRequest {
    pub schema_version: u32,
    pub driver: String,
    pub workload: WorkloadSpec,
    pub root_seed: u64,
    pub worker: usize,
    pub max_steps: usize,
}

impl WorkerRequest {
    pub fn driver_kind(&self) -> Result<DriverKind, String> {
        DriverKind::parse(&self.driver)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != RESULT_SCHEMA {
            return Err(format!(
                "request schema {} != {}",
                self.schema_version, RESULT_SCHEMA
            ));
        }
        self.driver_kind()?;
        if self.workload.actors_per_worker == 0
            || self.workload.worlds == 0
            || self.workload.rollouts_per_world == 0
            || self.max_steps == 0
        {
            return Err("actors, worlds, rollouts, and max_steps must be positive".to_string());
        }
        if self.worker >= self.workload.workers {
            return Err(format!(
                "worker {} outside workload worker count {}",
                self.worker, self.workload.workers
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkerResult {
    pub schema_version: u32,
    pub driver: String,
    pub fixture: FixtureSummary,
    pub workload_id: String,
    pub shape: WorkloadShape,
    pub root_seed: u64,
    pub seed_path: SeedPath,
    pub metrics: RunMetrics,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RunMetrics {
    pub elapsed_seconds: f64,
    pub reset_seconds: f64,
    pub hash_seconds: f64,
    pub fork_seconds: f64,
    pub mark_seconds: f64,
    pub rollback_seconds: f64,
    pub determinize_seconds: f64,
    pub apply_seconds: f64,
    pub policy_seconds: f64,
    pub tail_seconds: f64,
    pub simulations: u64,
    pub transitions: u64,
    pub root_decisions: u64,
    pub resets: u64,
    pub cap_hits: u64,
    pub hero_wins: u64,
    pub villain_wins: u64,
    pub draws: u64,
    pub eager_forks: u64,
    pub checkpoint_copies: u64,
    pub max_live_states: usize,
    /// Hashes only logical inputs/outcomes; never timings, RSS, or addresses.
    pub result_checksum: String,
    pub sampled_final_hashes: Vec<String>,
    pub step_latency_ns: Vec<u64>,
    pub clone_latency_ns: Vec<u64>,
    pub allocation_count: Option<u64>,
    pub allocation_bytes: Option<u64>,
    pub journal_bytes: Option<u64>,
    pub journal_entries: Option<u64>,
    pub journal_peak_entries: Option<u64>,
    pub journal_marks: Option<u64>,
    pub journal_commits: Option<u64>,
    pub journal_rollbacks: Option<u64>,
    pub cow_bytes: Option<u64>,
    pub unsupported_counters_reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SeedPath {
    pub root_seed: u64,
    pub worker_index: usize,
    pub worker_seed: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EquivalenceReceipt {
    pub fixture_id: String,
    pub trace_seed: u64,
    pub root_hash: String,
    pub action_hash: String,
    pub observation_hash: String,
    pub fork_hash: String,
    pub rollback_hash: String,
    pub replay_final_hash: String,
    pub replay_checksum: String,
    pub compared_steps: usize,
    pub root_isolated: bool,
    pub sibling_isolated: bool,
    pub structural_comparisons: usize,
    pub passed: bool,
}

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
        "interactive-midgame-48-v1" => (48, None, true),
        "interactive-heavy-80-v1" => {
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
        "interactive-midgame-48-v1" if summary.action_count != 6 => {
            return Err(format!(
                "midgame fixture drift: expected 6 root actions, got {}",
                summary.action_count
            ));
        }
        "interactive-heavy-80-v1"
            if summary.card_count != 120
                || summary.allocated_permanent_slots != 28
                || summary.committed_event_count != 498 =>
        {
            return Err(format!(
                "heavy fixture drift: expected cards/slots/events 120/28/498, got {}/{}/{}",
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
        ActionSpaceKind::DiscardThenDraw => "discard_then_draw",
        ActionSpaceKind::Waterbend => "waterbend",
    }
}

pub fn equivalence_check(
    fixture_id: &str,
    trace_seed: u64,
    max_steps: usize,
) -> Result<EquivalenceReceipt, String> {
    equivalence_check_with(DriverKind::FullClone, fixture_id, trace_seed, max_steps)
}

/// Every candidate must produce a byte-identical receipt here. The receipt
/// carries logical hashes only, so equality across candidates is the evidence
/// that a timing comparison between them is measuring the same work.
pub fn equivalence_check_with(
    driver: DriverKind,
    fixture_id: &str,
    trace_seed: u64,
    max_steps: usize,
) -> Result<EquivalenceReceipt, String> {
    match driver {
        DriverKind::FullClone => {
            equivalence_for(&FullCloneDriver, fixture_id, trace_seed, max_steps)
        }
        DriverKind::ClonePlusUndo => equivalence_for(
            &ClonePlusUndoDriver::default(),
            fixture_id,
            trace_seed,
            max_steps,
        ),
    }
}

fn equivalence_for<D: BranchDriver<State = Game>>(
    driver: &D,
    fixture_id: &str,
    trace_seed: u64,
    max_steps: usize,
) -> Result<EquivalenceReceipt, String> {
    let (game, _) = build_fixture(fixture_id)?;
    let root = driver.witness(&game);
    let mut left = driver.fork_exact(&game);
    let mut right = driver.fork_exact(&game);
    let fork = driver.witness(&left);
    if root != fork || root != driver.witness(&right) {
        return Err("exact fork differs from root".to_string());
    }
    let mark = driver.mark(&mut left);
    driver.apply(
        &mut left,
        BenchCommand {
            action_index: 0,
            expected_state_hash: Some(root.authority.hash.clone()),
            expected_action_hash: Some(root.legal_surface.hash.clone()),
        },
    )?;
    let root_isolated = driver.witness(&game) == root;
    let sibling_isolated = driver.witness(&right) == root;
    driver.rollback(&mut left, mark);
    let rollback = driver.witness(&left);
    if rollback != root {
        return Err("rollback failed to restore root".to_string());
    }

    let viewer = game
        .action_space()
        .and_then(|space| space.player)
        .ok_or_else(|| "equivalence fixture has no deciding player".to_string())?;
    let world_seed = mix_seed(trace_seed, 0xdecafbad);
    driver.determinize(&mut left, viewer, world_seed);
    driver.determinize(&mut right, viewer, world_seed);
    let rollout_seed = mix_seed(trace_seed, 0x51a7e);
    driver.reseed_rollout(&mut left, rollout_seed);
    driver.reseed_rollout(&mut right, rollout_seed);

    let mut compared_steps = 0;
    let mut structural_comparisons = 0;
    let mut replay_checksum = blake3::Hasher::new();
    replay_checksum.update(fixture_id.as_bytes());
    replay_checksum.update(&trace_seed.to_le_bytes());
    for step in 0..max_steps {
        let before_left = driver.witness(&left);
        let before_right = driver.witness(&right);
        structural_comparisons += 1;
        if before_left != before_right {
            return Err(format!("deterministic replay diverged before step {step}"));
        }
        replay_checksum.update(before_left.authority.hash.as_bytes());
        if left.is_game_over() {
            break;
        }
        let action_count = left.action_space().map_or(0, |space| space.actions.len());
        if action_count == 0 {
            return Err(format!("empty action space at replay step {step}"));
        }
        let action = (mix_seed(trace_seed, step as u64) as usize) % action_count;
        replay_checksum.update(&(action as u64).to_le_bytes());
        let command = BenchCommand {
            action_index: action,
            expected_state_hash: Some(before_left.authority.hash),
            expected_action_hash: Some(before_left.legal_surface.hash),
        };
        let left_done = driver.apply(&mut left, command.clone())?.done;
        let right_done = driver.apply(&mut right, command)?.done;
        if left_done != right_done {
            return Err(format!("terminal result diverged at replay step {step}"));
        }
        compared_steps += 1;
        if left_done {
            break;
        }
    }
    let replay_final = driver.witness(&left);
    if replay_final != driver.witness(&right) {
        return Err("deterministic replay final states differ".to_string());
    }
    replay_checksum.update(replay_final.authority.hash.as_bytes());
    Ok(EquivalenceReceipt {
        fixture_id: fixture_id.to_string(),
        trace_seed,
        root_hash: root.authority.hash,
        action_hash: root.legal_surface.hash,
        observation_hash: root.acting_projection.hash,
        fork_hash: fork.authority.hash,
        rollback_hash: rollback.authority.hash,
        replay_final_hash: replay_final.authority.hash,
        replay_checksum: replay_checksum.finalize().to_hex().to_string(),
        compared_steps,
        root_isolated,
        sibling_isolated,
        structural_comparisons,
        passed: root_isolated && sibling_isolated,
    })
}

fn deterministic_playout(
    game: &mut Game,
    max_steps: usize,
) -> Result<(Option<usize>, usize, bool), String> {
    let mut steps = 0;
    while !game.is_game_over() {
        if steps >= max_steps {
            return Ok((None, steps, true));
        }
        let action_count = game.action_space().map_or(0, |space| space.actions.len());
        if action_count == 0 {
            return Err(format!("empty action space at playout step {steps}"));
        }
        let action = game.state.rng.gen_range(0..action_count);
        game.step(action).map_err(|error| error.to_string())?;
        steps += 1;
    }
    Ok((game.winner_index(), steps, false))
}

fn record_outcome(metrics: &mut RunMetrics, winner: Option<usize>, cap_hit: bool) {
    if cap_hit {
        metrics.cap_hits += 1;
        metrics.draws += 1;
    } else {
        match winner {
            Some(0) => metrics.hero_wins += 1,
            Some(1) => metrics.villain_wins += 1,
            Some(_) | None => metrics.draws += 1,
        }
    }
}

fn checksum_state(
    hasher: &mut blake3::Hasher,
    metrics: &mut RunMetrics,
    game: &Game,
    winner: Option<usize>,
    steps: usize,
    cap_hit: bool,
) {
    let started = Instant::now();
    let final_witness = witness(game);
    metrics.hash_seconds += started.elapsed().as_secs_f64();
    hasher.update(final_witness.authority.hash.as_bytes());
    hasher.update(&final_witness.diagnostics.event_boundary[0].to_le_bytes());
    for probe in final_witness.diagnostics.rng_probe {
        hasher.update(&probe.to_le_bytes());
    }
    hasher.update(&(winner.unwrap_or(usize::MAX) as u64).to_le_bytes());
    hasher.update(&(steps as u64).to_le_bytes());
    hasher.update(&[u8::from(cap_hit)]);
    if metrics.sampled_final_hashes.len() < 8 {
        metrics
            .sampled_final_hashes
            .push(final_witness.authority.hash);
    }
}

/// Counters are read after the measured run so journal peaks reflect it.
/// Unsupported counters stay `null` with a reason; they are never zeroed.
fn install_counters<D: BranchDriver>(metrics: &mut RunMetrics, driver: &D) {
    let counters = driver.counters();
    metrics.allocation_count = counters.allocation_count;
    metrics.allocation_bytes = counters.allocation_bytes;
    metrics.journal_bytes = counters.journal_bytes;
    metrics.journal_entries = counters.journal_entries;
    metrics.journal_peak_entries = counters.journal_peak_entries;
    metrics.journal_marks = counters.journal_marks;
    metrics.journal_commits = counters.journal_commits;
    metrics.journal_rollbacks = counters.journal_rollbacks;
    metrics.cow_bytes = counters.cow_bytes;
    metrics.unsupported_counters_reason = counters.unsupported_reason;
}

fn warmup_clone<D: BranchDriver<State = Game>>(driver: &D, root: &Game, count: usize) {
    for _ in 0..count {
        let clone = driver.fork_exact(root);
        black_box(clone.action_space().map_or(0, |space| space.actions.len()));
        drop(clone);
    }
}

/// Prices one exact fork. Every candidate must still fork exactly, so this
/// diagnostic isolates that cost from the branch lifecycle around it.
fn measure_clone<D: BranchDriver<State = Game>>(
    driver: &D,
    root: &Game,
    count: usize,
    metrics: &mut RunMetrics,
) {
    let count = count.max(1);
    metrics.clone_latency_ns.reserve(count);
    let elapsed = Instant::now();
    for _ in 0..count {
        let sample = Instant::now();
        let clone = driver.fork_exact(root);
        black_box(clone.action_space().map_or(0, |space| space.actions.len()));
        drop(clone);
        metrics
            .clone_latency_ns
            .push(sample.elapsed().as_nanos() as u64);
        metrics.eager_forks += 1;
    }
    metrics.elapsed_seconds = elapsed.elapsed().as_secs_f64();
    metrics.root_decisions = 1;
    metrics.max_live_states = 2;
    let root_witness = witness(root);
    let mut checksum = blake3::Hasher::new();
    checksum.update(b"clone-v1");
    checksum.update(root_witness.authority.hash.as_bytes());
    checksum.update(&(count as u64).to_le_bytes());
    metrics.result_checksum = checksum.finalize().to_hex().to_string();
}

fn drive_env_steps(count: usize, timed: bool, metrics: &mut RunMetrics) -> Result<(), String> {
    let mut env = Env::new(SETUP_SEED, true, false, false);
    env.reset(player_configs())
        .map_err(|error| error.to_string())?;
    let mut completed_games = 0_u64;
    let mut checksum = blake3::Hasher::new();
    checksum.update(b"step-v1");
    for _ in 0..count {
        let action = env
            .random_action_index()
            .map_err(|error| error.to_string())?;
        let sample = Instant::now();
        let (_, reward, done, _, _) = env.step(action as i64).map_err(|error| error.to_string())?;
        if timed {
            metrics
                .step_latency_ns
                .push(sample.elapsed().as_nanos() as u64);
            metrics.transitions += 1;
            checksum.update(&(action as u64).to_le_bytes());
            checksum.update(&reward.to_bits().to_le_bytes());
            checksum.update(&[u8::from(done)]);
        }
        if done {
            completed_games += 1;
            let reset = Instant::now();
            env.set_seed(SETUP_SEED + completed_games);
            env.reset(player_configs())
                .map_err(|error| error.to_string())?;
            if timed {
                metrics.reset_seconds += reset.elapsed().as_secs_f64();
                metrics.resets += 1;
            }
        }
    }
    if timed {
        metrics.elapsed_seconds = metrics
            .step_latency_ns
            .iter()
            .map(|value| *value as f64 / 1_000_000_000.0)
            .sum();
        checksum.update(&(metrics.transitions).to_le_bytes());
        checksum.update(&metrics.resets.to_le_bytes());
        metrics.result_checksum = checksum.finalize().to_hex().to_string();
        metrics.root_decisions = metrics.transitions;
        metrics.max_live_states = 1;
    }
    Ok(())
}

/// Flat Monte-Carlo over one root: every simulation is a nested sequential
/// sibling of its world.
///
/// Siblings are taken with `mark` and released with `rollback` rather than a
/// fork per simulation. This is the branch lifecycle both candidates implement,
/// so the logical work is identical and only the branch mechanism differs:
/// full clone's `mark` copies the whole `Game` and its `rollback` restores that
/// copy, while clone plus undo marks a journal cursor and inverts it. Sequential
/// siblings are exactly the case an undo journal can serve without a copy, so
/// this is where a win, if there is one, must show up.
fn run_sequential<D: BranchDriver<State = Game>>(
    driver: &D,
    root: &Game,
    request: &WorkerRequest,
    worker_seed: u64,
    metrics: &mut RunMetrics,
) -> Result<(), String> {
    let viewer = root
        .action_space()
        .and_then(|space| space.player)
        .ok_or_else(|| "sequential root has no deciding player".to_string())?;
    let action_count = root.action_space().map_or(0, |space| space.actions.len());
    if action_count == 0 {
        return Err("sequential root has no legal actions".to_string());
    }
    let mut checksum = blake3::Hasher::new();
    checksum.update(request.workload.id.as_bytes());
    checksum.update(&request.root_seed.to_le_bytes());
    checksum.update(&(request.worker as u64).to_le_bytes());
    let started = Instant::now();
    for actor in 0..request.workload.actors_per_worker {
        let actor_seed = mix_seed(worker_seed, actor as u64);
        for world_index in 0..request.workload.worlds {
            let world_seed = mix_seed(actor_seed, world_index as u64);
            let fork_start = Instant::now();
            let mut world = driver.fork_exact(root);
            metrics.fork_seconds += fork_start.elapsed().as_secs_f64();
            metrics.eager_forks += 1;
            let determinize_start = Instant::now();
            driver.determinize(&mut world, viewer, world_seed);
            metrics.determinize_seconds += determinize_start.elapsed().as_secs_f64();
            for action in 0..action_count {
                for rollout in 0..request.workload.rollouts_per_world {
                    let mark_start = Instant::now();
                    let mark = driver.mark(&mut world);
                    metrics.mark_seconds += mark_start.elapsed().as_secs_f64();
                    metrics.checkpoint_copies += 1;
                    let rollout_seed = mix_seed(
                        world_seed,
                        (action * request.workload.rollouts_per_world + rollout + 1) as u64,
                    );
                    driver.reseed_rollout(&mut world, rollout_seed);
                    let apply_start = Instant::now();
                    let done = driver
                        .apply(
                            &mut world,
                            BenchCommand {
                                action_index: action,
                                expected_state_hash: None,
                                expected_action_hash: None,
                            },
                        )?
                        .done;
                    metrics.apply_seconds += apply_start.elapsed().as_secs_f64();
                    let (winner, tail_steps, cap_hit) = if done {
                        (world.winner_index(), 0, false)
                    } else {
                        let tail_start = Instant::now();
                        let outcome = deterministic_playout(
                            &mut world,
                            request.max_steps.saturating_sub(1).max(1),
                        )?;
                        metrics.tail_seconds += tail_start.elapsed().as_secs_f64();
                        outcome
                    };
                    let transitions = tail_steps + 1;
                    metrics.transitions += transitions as u64;
                    metrics.simulations += 1;
                    record_outcome(metrics, winner, cap_hit);
                    checksum_state(&mut checksum, metrics, &world, winner, transitions, cap_hit);
                    let rollback_start = Instant::now();
                    driver.rollback(&mut world, mark);
                    metrics.rollback_seconds += rollback_start.elapsed().as_secs_f64();
                }
            }
        }
        metrics.root_decisions += 1;
    }
    metrics.elapsed_seconds = (started.elapsed().as_secs_f64() - metrics.hash_seconds).max(0.0);
    // root, world, and the world's single live sibling.
    metrics.max_live_states = 3;
    metrics.result_checksum = checksum.finalize().to_hex().to_string();
    Ok(())
}

#[derive(Clone)]
struct RetainedSlot {
    game: Game,
    seed: u64,
    steps: usize,
    active: bool,
}

fn project_observation(game: &mut Game) {
    let events = game.take_observation_events();
    black_box(Observation::new(game, &events));
}

/// Every simulation stays live at once, so each slot needs its own exact fork
/// regardless of candidate. An undo journal cannot serve concurrently retained
/// cells: it only reverses one sequential branch at a time. Retained cells are
/// therefore expected to be a wash, and exist to price journal overhead.
fn run_retained<D: BranchDriver<State = Game>>(
    driver: &D,
    root: &Game,
    request: &WorkerRequest,
    worker_seed: u64,
    metrics: &mut RunMetrics,
) -> Result<(), String> {
    let viewer = root
        .action_space()
        .and_then(|space| space.player)
        .ok_or_else(|| "retained root has no deciding player".to_string())?;
    let action_count = root.action_space().map_or(0, |space| space.actions.len());
    if action_count == 0 {
        return Err("retained root has no legal actions".to_string());
    }
    let capacity = request.workload.actors_per_worker
        * request.workload.worlds
        * action_count
        * request.workload.rollouts_per_world;
    let mut checksum = blake3::Hasher::new();
    checksum.update(request.workload.id.as_bytes());
    checksum.update(&request.root_seed.to_le_bytes());
    checksum.update(&(request.worker as u64).to_le_bytes());
    let started = Instant::now();
    let mut slots = Vec::with_capacity(capacity);
    for actor in 0..request.workload.actors_per_worker {
        let actor_seed = mix_seed(worker_seed, actor as u64);
        for world_index in 0..request.workload.worlds {
            let world_seed = mix_seed(actor_seed, world_index as u64);
            let fork_start = Instant::now();
            let mut world = driver.fork_exact(root);
            metrics.fork_seconds += fork_start.elapsed().as_secs_f64();
            metrics.eager_forks += 1;
            let determinize_start = Instant::now();
            driver.determinize(&mut world, viewer, world_seed);
            metrics.determinize_seconds += determinize_start.elapsed().as_secs_f64();
            for action in 0..action_count {
                for rollout in 0..request.workload.rollouts_per_world {
                    let fork_start = Instant::now();
                    let mut sim = driver.fork_exact(&world);
                    metrics.fork_seconds += fork_start.elapsed().as_secs_f64();
                    metrics.eager_forks += 1;
                    let rollout_seed = mix_seed(
                        world_seed,
                        (action * request.workload.rollouts_per_world + rollout + 1) as u64,
                    );
                    driver.reseed_rollout(&mut sim, rollout_seed);
                    let apply_start = Instant::now();
                    let done = driver
                        .apply(
                            &mut sim,
                            BenchCommand {
                                action_index: action,
                                expected_state_hash: None,
                                expected_action_hash: None,
                            },
                        )?
                        .done;
                    metrics.apply_seconds += apply_start.elapsed().as_secs_f64();
                    project_observation(&mut sim);
                    slots.push(RetainedSlot {
                        game: sim,
                        seed: rollout_seed,
                        steps: 1,
                        active: !done,
                    });
                    metrics.transitions += 1;
                }
            }
        }
        metrics.root_decisions += 1;
    }
    metrics.max_live_states = slots.len() + request.workload.actors_per_worker;

    let policy_start = Instant::now();
    for ply in 0..request.workload.policy_plies {
        for (slot_index, slot) in slots.iter_mut().enumerate() {
            if !slot.active {
                continue;
            }
            if slot.steps >= request.max_steps {
                slot.active = false;
                continue;
            }
            let action_count = slot
                .game
                .action_space()
                .map_or(0, |space| space.actions.len());
            if action_count == 0 {
                return Err(format!("retained slot {slot_index} has empty action space"));
            }
            let action = (mix_seed(slot.seed, ply as u64) as usize) % action_count;
            let done = slot.game.step(action).map_err(|error| error.to_string())?;
            project_observation(&mut slot.game);
            slot.steps += 1;
            metrics.transitions += 1;
            slot.active = !done;
        }
    }
    metrics.policy_seconds = policy_start.elapsed().as_secs_f64();

    let tail_start = Instant::now();
    for slot in &mut slots {
        let (winner, tail_steps, cap_hit) = if slot.active {
            deterministic_playout(
                &mut slot.game,
                request.max_steps.saturating_sub(slot.steps).max(1),
            )?
        } else {
            (slot.game.winner_index(), 0, slot.steps >= request.max_steps)
        };
        slot.steps += tail_steps;
        metrics.transitions += tail_steps as u64;
        metrics.simulations += 1;
        record_outcome(metrics, winner, cap_hit);
        checksum_state(
            &mut checksum,
            metrics,
            &slot.game,
            winner,
            slot.steps,
            cap_hit,
        );
    }
    metrics.tail_seconds = (tail_start.elapsed().as_secs_f64() - metrics.hash_seconds).max(0.0);
    metrics.elapsed_seconds = (started.elapsed().as_secs_f64() - metrics.hash_seconds).max(0.0);
    metrics.result_checksum = checksum.finalize().to_hex().to_string();
    Ok(())
}

pub struct PreparedWorker {
    request: WorkerRequest,
    root: Game,
    fixture: FixtureSummary,
}

/// Constructs the exact fixture and performs contract warmup. Callers emit
/// their ready barrier only after this returns.
pub fn prepare_worker(request: WorkerRequest) -> Result<PreparedWorker, String> {
    request.validate()?;
    let (root, fixture) = build_fixture(&request.workload.fixture)?;
    // Warm up through the candidate actually under measurement, so its own
    // caches and allocation behavior are warm rather than the baseline's.
    match request.driver_kind()? {
        DriverKind::FullClone => warmup_worker(&FullCloneDriver, &root, &request)?,
        DriverKind::ClonePlusUndo => {
            warmup_worker(&ClonePlusUndoDriver::default(), &root, &request)?
        }
    }
    Ok(PreparedWorker {
        request,
        root,
        fixture,
    })
}

fn warmup_worker<D: BranchDriver<State = Game>>(
    driver: &D,
    root: &Game,
    request: &WorkerRequest,
) -> Result<(), String> {
    match request.workload.shape {
        WorkloadShape::Step => {
            let mut discarded = RunMetrics::default();
            drive_env_steps(request.workload.warmup_count, false, &mut discarded)?;
        }
        WorkloadShape::CloneLatency => warmup_clone(driver, root, request.workload.warmup_count),
        WorkloadShape::Sequential | WorkloadShape::Retained => {
            let warmup_request = WorkerRequest {
                root_seed: WARMUP_SEED,
                ..request.clone()
            };
            let mut discarded = RunMetrics::default();
            let worker_seed = mix_seed(WARMUP_SEED, request.worker as u64);
            if request.workload.shape == WorkloadShape::Sequential {
                run_sequential(driver, root, &warmup_request, worker_seed, &mut discarded)?;
            } else {
                run_retained(driver, root, &warmup_request, worker_seed, &mut discarded)?;
            }
        }
    }
    Ok(())
}

impl PreparedWorker {
    pub fn measure(self) -> Result<WorkerResult, String> {
        let worker_seed = mix_seed(self.request.root_seed, self.request.worker as u64);
        let driver_kind = self.request.driver_kind()?;
        let mut metrics = RunMetrics::default();
        match driver_kind {
            DriverKind::FullClone => {
                let driver = FullCloneDriver;
                measure_with(
                    &driver,
                    &self.root,
                    &self.request,
                    worker_seed,
                    &mut metrics,
                )?;
                install_counters(&mut metrics, &driver);
            }
            DriverKind::ClonePlusUndo => {
                let driver = ClonePlusUndoDriver::default();
                measure_with(
                    &driver,
                    &self.root,
                    &self.request,
                    worker_seed,
                    &mut metrics,
                )?;
                install_counters(&mut metrics, &driver);
            }
        }
        Ok(WorkerResult {
            schema_version: RESULT_SCHEMA,
            driver: driver_kind.id().to_string(),
            fixture: self.fixture,
            workload_id: self.request.workload.id.clone(),
            shape: self.request.workload.shape,
            root_seed: self.request.root_seed,
            seed_path: SeedPath {
                root_seed: self.request.root_seed,
                worker_index: self.request.worker,
                worker_seed,
            },
            metrics,
        })
    }
}

fn measure_with<D: BranchDriver<State = Game>>(
    driver: &D,
    root: &Game,
    request: &WorkerRequest,
    worker_seed: u64,
    metrics: &mut RunMetrics,
) -> Result<(), String> {
    match request.workload.shape {
        WorkloadShape::Step => drive_env_steps(request.workload.measured_count, true, metrics)?,
        WorkloadShape::CloneLatency => {
            measure_clone(driver, root, request.workload.measured_count, metrics)
        }
        WorkloadShape::Sequential => run_sequential(driver, root, request, worker_seed, metrics)?,
        WorkloadShape::Retained => run_retained(driver, root, request, worker_seed, metrics)?,
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_is_the_canonical_contract() {
        let manifest = manifest();
        assert_eq!(manifest.contract_id, CONTRACT_ID);
        assert_eq!(manifest.fixtures.len(), 2);
        assert_eq!(manifest.measured_seeds, MEASURED_SEEDS);
        assert!(manifest.workloads.iter().any(|spec| {
            spec.id == "flat-saturated-64-v1"
                && spec.workers == 8
                && spec.simulations_per_action() == 64
        }));
        assert!(manifest.workloads.iter().any(|spec| {
            spec.id == "retained-saturated-16-v1"
                && spec.actors_per_worker == 8
                && spec.worlds == 16
                && spec.policy_plies == 8
        }));
    }

    #[test]
    fn fixtures_are_exact_and_deterministic() {
        for id in manifest().fixtures {
            let (_, first) = build_fixture(&id).expect("first fixture build");
            let (_, second) = build_fixture(&id).expect("second fixture build");
            assert_eq!(first.semantic_hash, second.semantic_hash, "fixture {id}");
            assert_eq!(first.action_tape, second.action_tape, "fixture {id}");
        }
    }

    #[test]
    fn full_clone_is_exact_isolated_and_rollback_safe() {
        let receipt =
            equivalence_check("interactive-midgame-48-v1", 0x5eed, 16).expect("equivalence");
        assert!(receipt.passed);
        assert!(receipt.root_isolated);
        assert!(receipt.sibling_isolated);
        assert_eq!(receipt.root_hash, receipt.fork_hash);
        assert_eq!(receipt.root_hash, receipt.rollback_hash);
    }

    #[test]
    fn clone_checksum_excludes_latency_samples() {
        let (root, _) = build_fixture("interactive-heavy-80-v1").expect("heavy fixture");
        let mut first = RunMetrics::default();
        let mut second = RunMetrics::default();
        measure_clone(&FullCloneDriver, &root, 2, &mut first);
        measure_clone(&FullCloneDriver, &root, 2, &mut second);
        assert_eq!(first.result_checksum, second.result_checksum);
    }
}
