//! Representation-neutral search-state contract and full-clone reference.
//!
//! Search-state witnesses contain logical rules facts only. They deliberately
//! exclude pointers, allocator layout, timings, RSS, driver counters, and
//! behavior analytics. `ContentPack` definitions are named once by digest;
//! every mutable match fact needed by fork and rollback is serialized.
//!
//! Marks are used as well-formed nested LIFO scopes by this contract. A future
//! transactional driver may add branch/depth/revision diagnostics to its mark,
//! but it must pass the same logical witness tests without changing fixtures,
//! actions, or seeds.

use std::sync::Arc;

use rand::RngCore;
use serde::Serialize;

use crate::{
    agent::{action::ActionSpace, observation::Observation},
    flow::{
        event_log::{CowStats, EVENT_PAGE_BYTES},
        game::PendingChoice,
        search::mix_seed,
        undo::{ClonePlusUndoMark, JournalStats},
    },
    state::game_object::PlayerId,
    Game,
};

/// Schema 2 closes the identity and fixed-viewer omissions in the benchmark's
/// original shallow schema 1 witness.
pub const SEARCH_WITNESS_SCHEMA_VERSION: u32 = 2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthorityFingerprint {
    /// Canonical bytes for the complete mutable rules authority.
    pub bytes: Vec<u8>,
    pub hash: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LegalSurfaceFingerprint {
    pub bytes: Vec<u8>,
    pub hash: String,
    pub action_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ViewerProjectionWitness {
    pub bytes: Vec<u8>,
    pub hash: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContractDiagnostics {
    pub event_boundary: [usize; 3],
    pub rng_probe: [u64; 8],
    pub terminal: bool,
}

/// Evidence used to compare two search states without implying that the
/// evidence can itself restore either state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SearchStateWitness {
    pub authority: AuthorityFingerprint,
    pub legal_surface: LegalSurfaceFingerprint,
    /// Compatibility projection for the current acting-player observation.
    pub acting_projection: ViewerProjectionWitness,
    pub viewers: [ViewerProjectionWitness; 2],
    pub diagnostics: ContractDiagnostics,
}

#[derive(Serialize)]
struct CanonicalAuthorityState<'a> {
    schema_version: u32,
    match_state: serde_json::Value,
    current_action_space: &'a Option<ActionSpace>,
    decision_epoch: u64,
    pending_choice: &'a Option<PendingChoice>,
    skip_trivial: bool,
    skip_trivial_count: usize,
    rng_probe: [u64; 8],
}

/// Derive independent authority, legal-surface, viewer, and diagnostic evidence
/// for one exact logical rules state using fixed serialization order.
pub fn witness(game: &Game) -> SearchStateWitness {
    let mut rng = game.state.rng.clone();
    let mut rng_probe = [0_u64; 8];
    for value in &mut rng_probe {
        *value = rng.next_u64();
    }
    let authority_state = CanonicalAuthorityState {
        schema_version: SEARCH_WITNESS_SCHEMA_VERSION,
        match_state: game.state.deterministic_hash_value(),
        current_action_space: &game.current_action_space,
        decision_epoch: game.decision_epoch,
        pending_choice: &game.pending_choice,
        skip_trivial: game.skip_trivial,
        skip_trivial_count: game.skip_trivial_count,
        rng_probe,
    };
    let authority_bytes = serde_json::to_vec(&authority_state).expect("authority state serializes");
    let action_bytes = serde_json::to_vec(&game.current_action_space)
        .expect("action space serializes deterministically");
    let viewer_bytes = [
        Observation::for_player(game, PlayerId(0))
            .to_json()
            .into_bytes(),
        Observation::for_player(game, PlayerId(1))
            .to_json()
            .into_bytes(),
    ];
    let [player_zero_view, player_one_view] = viewer_bytes;
    let acting_projection_bytes = Observation::new(game, &game.state.observation_events)
        .to_json()
        .into_bytes();
    SearchStateWitness {
        authority: AuthorityFingerprint {
            hash: blake3::hash(&authority_bytes).to_hex().to_string(),
            bytes: authority_bytes,
        },
        legal_surface: LegalSurfaceFingerprint {
            hash: blake3::hash(&action_bytes).to_hex().to_string(),
            action_count: game
                .current_action_space
                .as_ref()
                .map_or(0, |space| space.actions.len()),
            bytes: action_bytes,
        },
        acting_projection: ViewerProjectionWitness {
            hash: blake3::hash(&acting_projection_bytes).to_hex().to_string(),
            bytes: acting_projection_bytes,
        },
        viewers: [
            ViewerProjectionWitness {
                hash: blake3::hash(&player_zero_view).to_hex().to_string(),
                bytes: player_zero_view,
            },
            ViewerProjectionWitness {
                hash: blake3::hash(&player_one_view).to_hex().to_string(),
                bytes: player_one_view,
            },
        ],
        diagnostics: ContractDiagnostics {
            event_boundary: [
                game.state.events.len(),
                game.state.pending_events.len(),
                game.state.observation_events.len(),
            ],
            rng_probe,
            terminal: game.is_game_over(),
        },
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
    pub journal_entries: Option<u64>,
    pub journal_peak_entries: Option<u64>,
    pub journal_marks: Option<u64>,
    pub journal_commits: Option<u64>,
    pub journal_rollbacks: Option<u64>,
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
    fn witness(&self, state: &Self::State) -> SearchStateWitness;
    fn counters(&self) -> DriverCounters;
    fn fork_copies_full_state(&self) -> bool;
    fn mark_copies_full_state(&self) -> bool;
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
            let pre = witness(state);
            if command
                .expected_state_hash
                .as_ref()
                .is_some_and(|expected| expected != &pre.authority.hash)
            {
                return Err("state precondition mismatch".to_string());
            }
            if command
                .expected_action_hash
                .as_ref()
                .is_some_and(|expected| expected != &pre.legal_surface.hash)
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

    fn witness(&self, state: &Game) -> SearchStateWitness {
        witness(state)
    }

    fn counters(&self) -> DriverCounters {
        DriverCounters {
            unsupported_reason: "system allocator has no counting hook; full-clone baseline has no undo journal or page-COW counters".to_string(),
            ..DriverCounters::default()
        }
    }

    fn fork_copies_full_state(&self) -> bool {
        true
    }

    fn mark_copies_full_state(&self) -> bool {
        true
    }
}

/// Exact outer clones with a dense inverse journal for nested sequential work.
#[derive(Clone, Debug, Default)]
pub struct ClonePlusUndoDriver {
    stats: Arc<JournalStats>,
}

impl ClonePlusUndoDriver {
    fn ensure_enabled(&self, state: &mut Game) {
        if state.undo.is_none() {
            state.enable_undo(self.stats.clone(), self.stats.next_branch_id());
        }
    }
}

impl BranchDriver for ClonePlusUndoDriver {
    type State = Game;
    type Mark = ClonePlusUndoMark;

    fn fork_exact(&self, source: &Game) -> Game {
        let mut fork = source.clone();
        self.ensure_enabled(&mut fork);
        fork
    }

    fn determinize(&self, state: &mut Game, viewer: PlayerId, seed: u64) {
        self.ensure_enabled(state);
        state.determinize(viewer, seed);
    }

    fn reseed_rollout(&self, state: &mut Game, seed: u64) {
        self.ensure_enabled(state);
        state.reseed(seed);
    }

    fn mark(&self, state: &mut Game) -> ClonePlusUndoMark {
        self.ensure_enabled(state);
        state.undo_mark()
    }

    fn apply(&self, state: &mut Game, command: BenchCommand) -> Result<ApplyReceipt, String> {
        self.ensure_enabled(state);
        if command.expected_state_hash.is_some() || command.expected_action_hash.is_some() {
            let pre = witness(state);
            if command
                .expected_state_hash
                .as_ref()
                .is_some_and(|expected| expected != &pre.authority.hash)
            {
                return Err("state precondition mismatch".to_string());
            }
            if command
                .expected_action_hash
                .as_ref()
                .is_some_and(|expected| expected != &pre.legal_surface.hash)
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

        let atomic = state.undo_mark();
        match state.step(command.action_index) {
            Ok(done) => {
                state.undo_commit(atomic);
                Ok(ApplyReceipt { done })
            }
            Err(error) => {
                state.undo_rollback(atomic);
                Err(error.to_string())
            }
        }
    }

    fn rollback(&self, state: &mut Game, mark: ClonePlusUndoMark) {
        state.undo_rollback(mark);
    }

    fn witness(&self, state: &Game) -> SearchStateWitness {
        witness(state)
    }

    fn counters(&self) -> DriverCounters {
        let stats = self.stats.snapshot();
        DriverCounters {
            journal_bytes: Some(stats.peak_bytes),
            journal_entries: Some(stats.entries),
            journal_peak_entries: Some(stats.peak_entries),
            journal_marks: Some(stats.marks),
            journal_commits: Some(stats.commits),
            journal_rollbacks: Some(stats.rollbacks),
            unsupported_reason:
                "system allocator has no counting hook; clone-plus-undo does not implement page COW"
                    .to_string(),
            ..DriverCounters::default()
        }
    }

    fn fork_copies_full_state(&self) -> bool {
        true
    }

    fn mark_copies_full_state(&self) -> bool {
        false
    }
}

/// Fixed 4 KiB event-page COW outer forks with the dense inverse journal for
/// nested sequential work.
#[derive(Clone, Debug, Default)]
pub struct DensePageCowUndoDriver {
    journal_stats: Arc<JournalStats>,
    cow_stats: Arc<CowStats>,
}

impl DensePageCowUndoDriver {
    pub fn admit_root(&self, state: &mut Game) {
        state.admit_page_cow_root();
    }

    fn ensure_enabled(&self, state: &mut Game) {
        if state.undo.is_none() {
            state.enable_undo(
                self.journal_stats.clone(),
                self.journal_stats.next_branch_id(),
            );
        }
    }
}

impl BranchDriver for DensePageCowUndoDriver {
    type State = Game;
    type Mark = ClonePlusUndoMark;

    fn fork_exact(&self, source: &Game) -> Game {
        let mut fork = source.page_cow_fork(self.cow_stats.clone());
        self.ensure_enabled(&mut fork);
        fork
    }

    fn determinize(&self, state: &mut Game, viewer: PlayerId, seed: u64) {
        self.ensure_enabled(state);
        state.determinize(viewer, seed);
    }

    fn reseed_rollout(&self, state: &mut Game, seed: u64) {
        self.ensure_enabled(state);
        state.reseed(seed);
    }

    fn mark(&self, state: &mut Game) -> ClonePlusUndoMark {
        self.ensure_enabled(state);
        state.undo_mark()
    }

    fn apply(&self, state: &mut Game, command: BenchCommand) -> Result<ApplyReceipt, String> {
        self.ensure_enabled(state);
        if command.expected_state_hash.is_some() || command.expected_action_hash.is_some() {
            let pre = witness(state);
            if command
                .expected_state_hash
                .as_ref()
                .is_some_and(|expected| expected != &pre.authority.hash)
            {
                return Err("state precondition mismatch".to_string());
            }
            if command
                .expected_action_hash
                .as_ref()
                .is_some_and(|expected| expected != &pre.legal_surface.hash)
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

        let atomic = state.undo_mark();
        match state.step(command.action_index) {
            Ok(done) => {
                state.undo_commit(atomic);
                Ok(ApplyReceipt { done })
            }
            Err(error) => {
                state.undo_rollback(atomic);
                Err(error.to_string())
            }
        }
    }

    fn rollback(&self, state: &mut Game, mark: ClonePlusUndoMark) {
        state.undo_rollback(mark);
    }

    fn witness(&self, state: &Game) -> SearchStateWitness {
        witness(state)
    }

    fn counters(&self) -> DriverCounters {
        let stats = self.journal_stats.snapshot();
        DriverCounters {
            journal_bytes: Some(stats.peak_bytes),
            journal_entries: Some(stats.entries),
            journal_peak_entries: Some(stats.peak_entries),
            journal_marks: Some(stats.marks),
            journal_commits: Some(stats.commits),
            journal_rollbacks: Some(stats.rollbacks),
            cow_bytes: Some(self.cow_stats.peak_bytes()),
            unsupported_reason: format!(
                "system allocator has no counting hook; cow_bytes measures branch-private copied {EVENT_PAGE_BYTES}-byte event pages"
            ),
            ..DriverCounters::default()
        }
    }

    fn fork_copies_full_state(&self) -> bool {
        false
    }

    fn mark_copies_full_state(&self) -> bool {
        false
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeededTraceReceipt {
    pub compared_steps: usize,
    pub final_hash: String,
}

/// Drive two independently supplied states through the same explicit seed
/// path and external legal-action sequence, comparing complete witnesses at
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
        let first_witness = driver.witness(first);
        let second_witness = driver.witness(second);
        if first_witness != second_witness {
            return Err(format!("seeded trace diverged before step {step}"));
        }
        if first_witness.diagnostics.terminal {
            return Ok(SeededTraceReceipt {
                compared_steps: step,
                final_hash: first_witness.authority.hash,
            });
        }
        if first_witness.legal_surface.action_count == 0 {
            return Err(format!("nonterminal state has no actions at step {step}"));
        }
        let action_index =
            (mix_seed(trace_seed, step as u64) as usize) % first_witness.legal_surface.action_count;
        let command = BenchCommand {
            action_index,
            expected_state_hash: Some(first_witness.authority.hash),
            expected_action_hash: Some(first_witness.legal_surface.hash),
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

fn checked_command(witness: &SearchStateWitness, action_index: usize) -> BenchCommand {
    BenchCommand {
        action_index,
        expected_state_hash: Some(witness.authority.hash.clone()),
        expected_action_hash: Some(witness.legal_surface.hash.clone()),
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
    let root_witness = driver.witness(root);
    if root_witness.diagnostics.terminal || root_witness.legal_surface.action_count == 0 {
        return Err("contract root must be nonterminal with legal actions".to_string());
    }

    let mut left = driver.fork_exact(root);
    let right = driver.fork_exact(root);
    if driver.witness(&left) != root_witness || driver.witness(&right) != root_witness {
        return Err("exact fork differs from root".to_string());
    }

    driver.determinize(&mut left, viewer, mix_seed(trace_seed, 0xd37e));
    if driver.witness(root) != root_witness || driver.witness(&right) != root_witness {
        return Err("fork determinization changed root or sibling".to_string());
    }
    left = driver.fork_exact(root);

    let outer = driver.mark(&mut left);
    let first_action =
        (mix_seed(trace_seed, 0x0a11) as usize) % root_witness.legal_surface.action_count;
    driver.apply(&mut left, checked_command(&root_witness, first_action))?;
    let after_outer = driver.witness(&left);
    if after_outer.diagnostics.terminal || after_outer.legal_surface.action_count == 0 {
        return Err("outer transaction did not leave a nested decision".to_string());
    }
    if driver.witness(root) != root_witness || driver.witness(&right) != root_witness {
        return Err("fork mutation changed root or sibling".to_string());
    }

    let inner = driver.mark(&mut left);
    let second_action =
        (mix_seed(trace_seed, 0x1aa2) as usize) % after_outer.legal_surface.action_count;
    let second_receipt = driver.apply(&mut left, checked_command(&after_outer, second_action))?;
    let after_inner = driver.witness(&left);
    driver.rollback(&mut left, inner);
    if driver.witness(&left) != after_outer {
        return Err("inner rollback did not restore outer state".to_string());
    }
    let replay_receipt = driver.apply(&mut left, checked_command(&after_outer, second_action))?;
    if replay_receipt != second_receipt || driver.witness(&left) != after_inner {
        return Err("replayed inner command did not reproduce its result".to_string());
    }
    driver.rollback(&mut left, outer);
    if driver.witness(&left) != root_witness {
        return Err("outer rollback did not restore root".to_string());
    }

    let stale_precondition = BenchCommand {
        action_index: 0,
        expected_state_hash: Some(format!("{}-stale", root_witness.authority.hash)),
        expected_action_hash: Some(root_witness.legal_surface.hash.clone()),
    };
    if driver.apply(&mut left, stale_precondition).is_ok() || driver.witness(&left) != root_witness
    {
        return Err("stale state precondition was not an exact no-op".to_string());
    }
    let invalid_index = BenchCommand {
        action_index: root_witness.legal_surface.action_count,
        expected_state_hash: Some(root_witness.authority.hash.clone()),
        expected_action_hash: Some(root_witness.legal_surface.hash.clone()),
    };
    if driver.apply(&mut left, invalid_index).is_ok() || driver.witness(&left) != root_witness {
        return Err("invalid action was not an exact no-op".to_string());
    }

    let mut mutable_root = driver.fork_exact(root);
    let root_child = driver.fork_exact(&mutable_root);
    let root_sibling = driver.fork_exact(&mutable_root);
    driver.apply(&mut mutable_root, checked_command(&root_witness, 0))?;
    if driver.witness(&root_child) != root_witness || driver.witness(&root_sibling) != root_witness
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
        root_hash: root_witness.authority.hash,
        nested_hash: after_inner.authority.hash,
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

    fn assert_authority_change(baseline: &SearchStateWitness, changed: &Game, field: &str) {
        let changed = witness(changed);
        assert_ne!(
            baseline.authority.bytes, changed.authority.bytes,
            "{field} bytes"
        );
        assert_ne!(
            baseline.authority.hash, changed.authority.hash,
            "{field} hash"
        );
    }

    #[test]
    fn authority_fingerprint_covers_identity_rng_events_allocation_zones_and_decisions() {
        let (game, _) =
            crate::benchmark::build_fixture("interactive-midgame-48-v1").expect("contract fixture");
        let baseline = witness(&game);

        let mut incarnation = game.clone();
        incarnation.state.object_incarnations[CardId(0)] = incarnation.state.object_incarnations
            [CardId(0)]
        .checked_next()
        .expect("next incarnation");
        assert_authority_change(&baseline, &incarnation, "incarnation");

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
        assert_authority_change(&baseline, &object_lki, "object LKI");

        let mut rng = game.clone();
        rng.reseed(0x197_5eed);
        assert_authority_change(&baseline, &rng, "RNG continuation");

        let mut events = game.clone();
        events.state.events.push(GameEvent::TurnStarted {
            player: PlayerId(0),
            turn_number: events.state.turn.turn_number,
        });
        assert_authority_change(&baseline, &events, "event ledger");

        let mut allocation = game.clone();
        allocation.state.id_gen.next_id();
        assert_authority_change(&baseline, &allocation, "allocation watermark");

        let mut zone_order = game.clone();
        let library = zone_order
            .state
            .zones
            .zone_cards_mut(ZoneType::Library, PlayerId(0));
        let last = library.len() - 1;
        library.swap(0, last);
        assert_authority_change(&baseline, &zone_order, "zone order");

        let mut decision = game.clone();
        decision.decision_epoch += 1;
        assert_authority_change(&baseline, &decision, "decision epoch");
    }
}
