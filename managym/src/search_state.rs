//! Representation-neutral search-state contract and full-clone reference.
//!
//! Canonical snapshots contain logical rules facts only. They deliberately
//! exclude pointers, allocator layout, timings, RSS, driver counters, and
//! behavior analytics. `ContentPack` definitions are named once by digest;
//! every mutable match fact needed by fork and rollback is serialized.
//!
//! Marks are used as well-formed nested LIFO scopes by this contract. A future
//! transactional driver may add branch/depth/revision diagnostics to its mark,
//! but it must pass the same logical snapshot tests without changing fixtures,
//! actions, or seeds.

use rand::RngCore;
use serde::Serialize;

use crate::{
    agent::{action::ActionSpace, observation::Observation},
    flow::{game::PendingChoice, search::mix_seed},
    state::game_object::PlayerId,
    Game,
};

/// Schema 2 closes the identity and fixed-viewer omissions in the benchmark's
/// original shallow schema 1 snapshot.
pub const SNAPSHOT_SCHEMA: u32 = 2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalSnapshotV2 {
    pub canonical: Vec<u8>,
    pub hash: String,
    pub action_bytes: Vec<u8>,
    pub action_hash: String,
    pub action_count: usize,
    /// Compatibility hash for the current acting-player observation.
    pub observation_hash: String,
    pub viewer_projections: [Vec<u8>; 2],
    pub viewer_hashes: [String; 2],
    pub event_boundary: [usize; 3],
    pub rng_probe: [u64; 8],
    pub terminal: bool,
}

/// Source-compatible name used by the W2-182 benchmark code.
pub type EquivalenceSnapshot = CanonicalSnapshotV2;

#[derive(Serialize)]
struct SemanticSnapshot<'a> {
    schema_version: u32,
    match_state: serde_json::Value,
    current_action_space: &'a Option<ActionSpace>,
    decision_epoch: u64,
    pending_choice: &'a Option<PendingChoice>,
    skip_trivial: bool,
    skip_trivial_count: usize,
    rng_probe: [u64; 8],
}

/// Serialize one exact logical rules state with fixed ordering and derive its
/// authoritative, action, fixed-viewer, event-boundary, and RNG evidence.
pub fn snapshot(game: &Game) -> CanonicalSnapshotV2 {
    let mut rng = game.state.rng.clone();
    let mut rng_probe = [0_u64; 8];
    for value in &mut rng_probe {
        *value = rng.next_u64();
    }
    let semantic = SemanticSnapshot {
        schema_version: SNAPSHOT_SCHEMA,
        match_state: game.state.deterministic_hash_value(),
        current_action_space: &game.current_action_space,
        decision_epoch: game.decision_epoch,
        pending_choice: &game.pending_choice,
        skip_trivial: game.skip_trivial,
        skip_trivial_count: game.skip_trivial_count,
        rng_probe,
    };
    let canonical = serde_json::to_vec(&semantic).expect("semantic snapshot serializes");
    let action_bytes = serde_json::to_vec(&game.current_action_space)
        .expect("action space serializes deterministically");
    let viewer_projections = [
        Observation::for_player(game, PlayerId(0))
            .to_json()
            .into_bytes(),
        Observation::for_player(game, PlayerId(1))
            .to_json()
            .into_bytes(),
    ];
    let viewer_hashes = [
        blake3::hash(&viewer_projections[0]).to_hex().to_string(),
        blake3::hash(&viewer_projections[1]).to_hex().to_string(),
    ];
    let acting_observation = Observation::new(game, &game.state.observation_events).to_json();
    CanonicalSnapshotV2 {
        hash: blake3::hash(&canonical).to_hex().to_string(),
        action_hash: blake3::hash(&action_bytes).to_hex().to_string(),
        action_count: game
            .current_action_space
            .as_ref()
            .map_or(0, |space| space.actions.len()),
        observation_hash: blake3::hash(acting_observation.as_bytes())
            .to_hex()
            .to_string(),
        viewer_projections,
        viewer_hashes,
        event_boundary: [
            game.state.events.len(),
            game.state.pending_events.len(),
            game.state.observation_events.len(),
        ],
        canonical,
        action_bytes,
        rng_probe,
        terminal: game.is_game_over(),
    }
}

#[derive(Clone, Debug)]
pub struct BenchCommand {
    pub action_index: usize,
    pub expected_state_hash: Option<String>,
    pub expected_action_hash: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApplyReceipt {
    pub done: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct DriverCounters {
    pub allocation_count: Option<u64>,
    pub allocation_bytes: Option<u64>,
    pub journal_bytes: Option<u64>,
    pub cow_bytes: Option<u64>,
    pub unsupported_reason: String,
}

/// Logical branch lifecycle implemented by every benchmark candidate.
pub trait BranchDriver {
    type State;
    type Mark;

    fn fork_exact(&self, source: &Self::State) -> Self::State;
    fn determinize(&self, state: &mut Self::State, viewer: PlayerId, seed: u64);
    fn reseed_rollout(&self, state: &mut Self::State, seed: u64);
    fn mark(&self, state: &mut Self::State) -> Self::Mark;
    fn apply(&self, state: &mut Self::State, command: BenchCommand)
        -> Result<ApplyReceipt, String>;
    fn rollback(&self, state: &mut Self::State, mark: Self::Mark);
    fn snapshot(&self, state: &Self::State) -> CanonicalSnapshotV2;
    fn counters(&self) -> DriverCounters;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FullCloneDriver;

impl BranchDriver for FullCloneDriver {
    type State = Game;
    type Mark = Game;

    fn fork_exact(&self, source: &Game) -> Game {
        source.clone()
    }

    fn determinize(&self, state: &mut Game, viewer: PlayerId, seed: u64) {
        state.determinize(viewer, seed);
    }

    fn reseed_rollout(&self, state: &mut Game, seed: u64) {
        state.reseed(seed);
    }

    fn mark(&self, state: &mut Game) -> Game {
        state.clone()
    }

    fn apply(&self, state: &mut Game, command: BenchCommand) -> Result<ApplyReceipt, String> {
        if command.expected_state_hash.is_some() || command.expected_action_hash.is_some() {
            let pre = snapshot(state);
            if command
                .expected_state_hash
                .as_ref()
                .is_some_and(|expected| expected != &pre.hash)
            {
                return Err("state precondition mismatch".to_string());
            }
            if command
                .expected_action_hash
                .as_ref()
                .is_some_and(|expected| expected != &pre.action_hash)
            {
                return Err("action precondition mismatch".to_string());
            }
        }
        let action_count = state.action_space().map_or(0, |space| space.actions.len());
        if command.action_index >= action_count {
            return Err(format!(
                "action {} out of bounds for {action_count} legal actions",
                command.action_index
            ));
        }
        let done = state
            .step(command.action_index)
            .map_err(|error| error.to_string())?;
        Ok(ApplyReceipt { done })
    }

    fn rollback(&self, state: &mut Game, mark: Game) {
        *state = mark;
    }

    fn snapshot(&self, state: &Game) -> CanonicalSnapshotV2 {
        snapshot(state)
    }

    fn counters(&self) -> DriverCounters {
        DriverCounters {
            unsupported_reason: "system allocator has no counting hook; full-clone baseline has no undo journal or page-COW counters".to_string(),
            ..DriverCounters::default()
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeededTraceReceipt {
    pub compared_steps: usize,
    pub final_hash: String,
}

/// Drive two independently supplied states through the same explicit seed
/// path and external legal-action sequence, comparing complete snapshots at
/// every boundary.
pub fn verify_seeded_trace_equivalence<D: BranchDriver>(
    driver: &D,
    first: &mut D::State,
    second: &mut D::State,
    viewer: PlayerId,
    trace_seed: u64,
    max_steps: usize,
) -> Result<SeededTraceReceipt, String> {
    let world_seed = mix_seed(trace_seed, 0xdecafbad);
    driver.determinize(first, viewer, world_seed);
    driver.determinize(second, viewer, world_seed);
    let rollout_seed = mix_seed(trace_seed, 0x51a7e);
    driver.reseed_rollout(first, rollout_seed);
    driver.reseed_rollout(second, rollout_seed);

    for step in 0..max_steps {
        let first_snapshot = driver.snapshot(first);
        let second_snapshot = driver.snapshot(second);
        if first_snapshot != second_snapshot {
            return Err(format!("seeded trace diverged before step {step}"));
        }
        if first_snapshot.terminal {
            return Ok(SeededTraceReceipt {
                compared_steps: step,
                final_hash: first_snapshot.hash,
            });
        }
        if first_snapshot.action_count == 0 {
            return Err(format!("nonterminal state has no actions at step {step}"));
        }
        let action_index =
            (mix_seed(trace_seed, step as u64) as usize) % first_snapshot.action_count;
        let command = BenchCommand {
            action_index,
            expected_state_hash: Some(first_snapshot.hash),
            expected_action_hash: Some(first_snapshot.action_hash),
        };
        let first_result = driver.apply(first, command.clone())?;
        let second_result = driver.apply(second, command)?;
        if first_result != second_result {
            return Err(format!("terminal result diverged at step {step}"));
        }
    }

    Err(format!(
        "seeded trace exceeded the {max_steps}-decision contract cap"
    ))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BranchContractReceipt {
    pub root_hash: String,
    pub nested_hash: String,
    pub replay_final_hash: String,
    pub replay_steps: usize,
}

fn checked_command(snapshot: &CanonicalSnapshotV2, action_index: usize) -> BenchCommand {
    BenchCommand {
        action_index,
        expected_state_hash: Some(snapshot.hash.clone()),
        expected_action_hash: Some(snapshot.action_hash.clone()),
    }
}

/// Exercise the representation-neutral fork, nested rollback, atomic failure,
/// reverse-isolation, and deterministic seeded-trace contract.
pub fn verify_branch_contract<D: BranchDriver>(
    driver: &D,
    root: &D::State,
    viewer: PlayerId,
    trace_seed: u64,
    max_steps: usize,
) -> Result<BranchContractReceipt, String> {
    let root_snapshot = driver.snapshot(root);
    if root_snapshot.terminal || root_snapshot.action_count == 0 {
        return Err("contract root must be nonterminal with legal actions".to_string());
    }

    let mut left = driver.fork_exact(root);
    let right = driver.fork_exact(root);
    if driver.snapshot(&left) != root_snapshot || driver.snapshot(&right) != root_snapshot {
        return Err("exact fork differs from root".to_string());
    }

    driver.determinize(&mut left, viewer, mix_seed(trace_seed, 0xd37e));
    if driver.snapshot(root) != root_snapshot || driver.snapshot(&right) != root_snapshot {
        return Err("fork determinization changed root or sibling".to_string());
    }
    left = driver.fork_exact(root);

    let outer = driver.mark(&mut left);
    let first_action = (mix_seed(trace_seed, 0x0a11) as usize) % root_snapshot.action_count;
    driver.apply(&mut left, checked_command(&root_snapshot, first_action))?;
    let after_outer = driver.snapshot(&left);
    if after_outer.terminal || after_outer.action_count == 0 {
        return Err("outer transaction did not leave a nested decision".to_string());
    }
    if driver.snapshot(root) != root_snapshot || driver.snapshot(&right) != root_snapshot {
        return Err("fork mutation changed root or sibling".to_string());
    }

    let inner = driver.mark(&mut left);
    let second_action = (mix_seed(trace_seed, 0x1aa2) as usize) % after_outer.action_count;
    let second_receipt = driver.apply(&mut left, checked_command(&after_outer, second_action))?;
    let after_inner = driver.snapshot(&left);
    driver.rollback(&mut left, inner);
    if driver.snapshot(&left) != after_outer {
        return Err("inner rollback did not restore outer state".to_string());
    }
    let replay_receipt = driver.apply(&mut left, checked_command(&after_outer, second_action))?;
    if replay_receipt != second_receipt || driver.snapshot(&left) != after_inner {
        return Err("replayed inner command did not reproduce its result".to_string());
    }
    driver.rollback(&mut left, outer);
    if driver.snapshot(&left) != root_snapshot {
        return Err("outer rollback did not restore root".to_string());
    }

    let stale_precondition = BenchCommand {
        action_index: 0,
        expected_state_hash: Some(format!("{}-stale", root_snapshot.hash)),
        expected_action_hash: Some(root_snapshot.action_hash.clone()),
    };
    if driver.apply(&mut left, stale_precondition).is_ok()
        || driver.snapshot(&left) != root_snapshot
    {
        return Err("stale state precondition was not an exact no-op".to_string());
    }
    let invalid_index = BenchCommand {
        action_index: root_snapshot.action_count,
        expected_state_hash: Some(root_snapshot.hash.clone()),
        expected_action_hash: Some(root_snapshot.action_hash.clone()),
    };
    if driver.apply(&mut left, invalid_index).is_ok() || driver.snapshot(&left) != root_snapshot {
        return Err("invalid action was not an exact no-op".to_string());
    }

    let mut mutable_root = driver.fork_exact(root);
    let root_child = driver.fork_exact(&mutable_root);
    let root_sibling = driver.fork_exact(&mutable_root);
    driver.apply(&mut mutable_root, checked_command(&root_snapshot, 0))?;
    if driver.snapshot(&root_child) != root_snapshot
        || driver.snapshot(&root_sibling) != root_snapshot
    {
        return Err("root mutation changed an existing fork".to_string());
    }

    let mut replay_left = driver.fork_exact(root);
    let mut replay_right = driver.fork_exact(root);
    let replay = verify_seeded_trace_equivalence(
        driver,
        &mut replay_left,
        &mut replay_right,
        viewer,
        trace_seed,
        max_steps,
    )?;

    Ok(BranchContractReceipt {
        root_hash: root_snapshot.hash,
        nested_hash: after_inner.hash,
        replay_final_hash: replay.final_hash,
        replay_steps: replay.compared_steps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        flow::event::GameEvent,
        state::{
            game_object::{CardId, PermanentId},
            zone::ZoneType,
        },
    };

    fn assert_semantic_change(baseline: &CanonicalSnapshotV2, changed: &Game, field: &str) {
        let changed = snapshot(changed);
        assert_ne!(baseline.canonical, changed.canonical, "{field} bytes");
        assert_ne!(baseline.hash, changed.hash, "{field} hash");
    }

    #[test]
    fn snapshot_hash_covers_identity_rng_events_allocation_zones_and_decisions() {
        let (game, _) =
            crate::benchmark::build_fixture("interactive-midgame-48-v1").expect("contract fixture");
        let baseline = snapshot(&game);

        let mut incarnation = game.clone();
        incarnation.state.object_incarnations[CardId(0)] = incarnation.state.object_incarnations
            [CardId(0)]
        .checked_next()
        .expect("next incarnation");
        assert_semantic_change(&baseline, &incarnation, "incarnation");

        let permanent = PermanentId(
            game.state
                .permanents
                .iter()
                .position(Option::is_some)
                .expect("fixture has a live permanent"),
        );
        let card = game.state.permanents[permanent]
            .as_ref()
            .expect("live permanent")
            .card;
        let lki = game
            .snapshot_current_permanent(card)
            .expect("live permanent has snapshot facts");
        let mut object_lki = game.clone();
        object_lki.state.object_lki.insert(lki.object_ref, lki);
        assert_semantic_change(&baseline, &object_lki, "object LKI");

        let mut rng = game.clone();
        rng.reseed(0x197_5eed);
        assert_semantic_change(&baseline, &rng, "RNG continuation");

        let mut events = game.clone();
        events.state.events.push(GameEvent::TurnStarted {
            player: PlayerId(0),
            turn_number: events.state.turn.turn_number,
        });
        assert_semantic_change(&baseline, &events, "event ledger");

        let mut allocation = game.clone();
        allocation.state.id_gen.next_id();
        assert_semantic_change(&baseline, &allocation, "allocation watermark");

        let mut zone_order = game.clone();
        let library = zone_order
            .state
            .zones
            .zone_cards_mut(ZoneType::Library, PlayerId(0));
        let last = library.len() - 1;
        library.swap(0, last);
        assert_semantic_change(&baseline, &zone_order, "zone order");

        let mut decision = game.clone();
        decision.decision_epoch += 1;
        assert_semantic_change(&baseline, &decision, "decision epoch");
    }
}
