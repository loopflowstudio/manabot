//! An immutable decision boundary. Only live branches have projection caches,
//! producer sinks or a generation used to admit append-only memory transport.
use super::*;
use crate::{
    agent::{
        action::{ActionSpace, ActionSpaceKind},
        decision_memory::DecisionLog,
        policy_history::HistoryWindow,
    },
    decision::ObjectCandidateAddress,
    flow::{
        game::{PendingChoice, TrackedState},
        turn::PhaseKind,
    },
    state::{card::CardDefId, game_object::ObjectId},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// A detached live position shareable by any number of restoring threads.
pub struct GameSnapshot {
    state: TrackedState,
    decision_memory: [DecisionLog; 2],
    memory_enabled: bool,
    include_own: bool,
    history: HistoryWindow,
    action_space: Option<ActionSpace>,
    pending_choice: Option<PendingChoice>,
    epoch: u64,
    candidates: Arc<BTreeMap<ObjectCandidateAddress, ObjectId>>,
    skip_trivial: bool,
    skipped: usize,
    priority_passes: usize,
    mana_floats: [ManaFloats; 2],
    mana_payment: [ManaPayment; 2],
    heuristics: [Option<Box<Heuristic>>; 2],
    heuristic_events: Vec<EventData>,
    policies: [PlayerPolicy; 2],
    reset_seed: u64,
    reset_setup: Option<Arc<GameSetup>>,
    provenance: Option<GameProvenance>,
}

const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<GameSnapshot>();
};

#[derive(Clone, Copy)]
pub enum BranchMode {
    Exact { reseed: Option<u64> },
    ShuffleLibraries { seed: u64 },
    Determinize { viewer: usize, seed: u64 },
}

#[derive(Clone, Copy)]
pub struct RestoreOptions {
    pub mode: BranchMode,
    pub enable_decision_memory: Option<bool>,
    pub forced_first_action: Option<usize>,
}

impl RestoreOptions {
    pub fn exact(reseed: Option<u64>) -> Self {
        Self {
            mode: BranchMode::Exact { reseed },
            enable_decision_memory: None,
            forced_first_action: None,
        }
    }
}

impl Default for RestoreOptions {
    fn default() -> Self {
        Self::exact(None)
    }
}

/// Public board and decision summary at the saved boundary.
pub struct SnapshotMetadata {
    pub turn: u32,
    pub phase: PhaseKind,
    pub life: [i32; 2],
    pub board_counts: [usize; 2],
    pub stack_len: usize,
    pub action_count: usize,
    pub action_kind: Option<ActionSpaceKind>,
    pub acting_player: Option<usize>,
    pub terminal: bool,
    pub active_card_defs: Vec<CardDefId>,
}

impl Env {
    /// Replay the operation payload of a retained boundary intervention.
    #[cfg(feature = "python")]
    pub(crate) fn apply_intervention_json(&mut self, raw: &str) -> Result<(), AgentError> {
        let op = serde_json::from_str(raw)
            .map_err(|e| AgentError(format!("invalid snapshot intervention: {e}")))?;
        self.apply_intervention(op)
    }

    pub fn snapshot(&self) -> Result<GameSnapshot, AgentError> {
        let game = self
            .game
            .as_ref()
            .ok_or_else(|| AgentError("snapshot before reset".into()))?;
        Ok(GameSnapshot {
            state: game.state.clone(),
            decision_memory: game.decision_memory.clone(),
            memory_enabled: game.decision_memory_enabled,
            include_own: game.decision_memory_include_own,
            history: game.policy_history.clone(),
            action_space: game.current_action_space.clone(),
            pending_choice: game.pending_choice.clone(),
            epoch: game.decision_epoch,
            candidates: Arc::new(game.semantic_object_candidates.borrow().clone()),
            skip_trivial: game.skip_trivial,
            skipped: game.skip_trivial_count,
            priority_passes: game.skipped_priority_passes,
            mana_floats: game.mana_floats,
            mana_payment: game.mana_payment,
            heuristics: self.heuristics.clone(),
            heuristic_events: self.heuristic_events.clone(),
            policies: self.player_policies,
            reset_seed: self.reset_seed,
            reset_setup: self.reset_setup.clone(),
            provenance: self.provenance.clone(),
        })
    }
}

impl GameSnapshot {
    /// Materialize a new live game generation, continuing both viewer logs.
    pub fn restore(&self, options: RestoreOptions) -> Result<Env, AgentError> {
        if let BranchMode::Determinize { viewer, .. } = options.mode {
            self.state.sampling_viewer(viewer)?;
        }
        let forced = options
            .forced_first_action
            .map(|index| {
                let space = self
                    .action_space
                    .as_ref()
                    .filter(|_| self.state.players.iter().all(|p| p.alive))
                    .ok_or_else(|| {
                        AgentError("forced_first_action requires a non-terminal decision".into())
                    })?;
                let action = space.actions.get(index).ok_or_else(|| {
                    AgentError(format!("forced_first_action {index} out of bounds"))
                })?;
                Ok::<_, AgentError>((index, space.player, action))
            })
            .transpose()?;
        let game = Game {
            state: self.state.clone(),
            decision_memory: self.decision_memory.clone(),
            decision_memory_enabled: self.memory_enabled,
            decision_memory_include_own: self.include_own,
            policy_history: self.history.clone(),
            current_action_space: self.action_space.clone(),
            pending_choice: self.pending_choice.clone(),
            decision_epoch: self.epoch,
            semantic_object_candidates: (*self.candidates).clone().into(),
            skip_trivial: self.skip_trivial,
            skip_trivial_count: self.skipped,
            skipped_priority_passes: self.priority_passes,
            mana_floats: self.mana_floats,
            mana_payment: self.mana_payment,
            derived: Default::default(),
            payment_cache: Default::default(),
            payment_sources: Default::default(),
            statistics: None,
            card_capture: None,
            capture_originals: Default::default(),
            capture_drawing: None,
        };
        let mut env = Env {
            game: Some(game),
            heuristics: self.heuristics.clone(),
            heuristic_events: self.heuristic_events.clone(),
            player_policies: self.policies,
            provenance: self.provenance.clone(),
            reset_setup: self.reset_setup.clone(),
            decision_memory_enabled: self.memory_enabled,
            decision_memory_include_own: self.include_own,
            capture_generation: GAME_GENERATION.fetch_add(1, Ordering::Relaxed),
            ..Env::new(self.reset_seed, self.skip_trivial, false, false)
        };
        let intervention = match options.mode {
            BranchMode::Exact { reseed } => reseed.map(|seed| Intervention::Reseed { seed }),
            BranchMode::ShuffleLibraries { seed } => Some(Intervention::ShuffleLibraries { seed }),
            BranchMode::Determinize { viewer, seed } => {
                Some(Intervention::Determinize { viewer, seed })
            }
        };
        if let Some(op) = intervention {
            env.apply_intervention(op)?;
        }
        if let Some(enabled) = options
            .enable_decision_memory
            .filter(|&enabled| enabled != self.memory_enabled)
        {
            env.apply_intervention(Intervention::SetDecisionMemory { enabled })?;
        }
        if let Some((index, chooser, action)) = forced {
            let current = env.game.as_ref().and_then(|g| g.action_space());
            let seat = env
                .current_agent_index()
                .ok_or_else(|| AgentError("no active chooser".into()))?;
            if Some(PlayerId(seat)) != chooser
                || current.and_then(|s| s.actions.get(index)) != Some(action)
            {
                return Err(AgentError(format!(
                    "forced_first_action {index} changed after resampling"
                )));
            }
            if env.player_policies[seat] == PlayerPolicy::Agent {
                env.step(index as i64)?;
            } else {
                env.apply_intervention(Intervention::ForceAction { action: index })?;
            }
        }
        Ok(env)
    }

    pub fn metadata(&self) -> SnapshotMetadata {
        let mut definitions = BTreeSet::new();
        let mut board_counts = [0; 2];
        for seat in [PlayerId(0), PlayerId(1)] {
            for &object in self.state.zones.battlefield_objects(seat) {
                if let Some(id) = self.state.permanent_of(object) {
                    let permanent = self.state.permanents[id]
                        .as_ref()
                        .expect("live battlefield permanent");
                    board_counts[permanent.controller.0] += 1;
                    definitions.insert(self.state[object].definition_id);
                }
            }
        }
        definitions.extend(
            self.state
                .stack_objects
                .iter()
                .map(|obj| obj.source_definition_id()),
        );
        let terminal = self.state.players.iter().any(|p| !p.alive);
        let space = self.action_space.as_ref().filter(|_| !terminal);
        SnapshotMetadata {
            turn: self.state.turn.turn_number,
            phase: self.state.turn.current_phase_kind(),
            life: self.state.players.each_ref().map(|p| p.life),
            board_counts,
            stack_len: self.state.stack_objects.len(),
            action_count: space.map_or(0, |s| s.actions.len()),
            action_kind: space.map(|s| s.kind),
            acting_player: space.and_then(|s| s.player).map(|p| p.0),
            terminal,
            active_card_defs: definitions.into_iter().collect(),
        }
    }

    /// Export private replay inputs, or fail if a mutation broke the prefix.
    pub fn to_recipe_json(&self) -> Result<String, AgentError> {
        let provenance = self.provenance.as_ref().ok_or_else(|| {
            AgentError("snapshot has no replayable prefix after an unjournaled mutation".into())
        })?;
        let mut recipe = Recipe {
            recipe_digest: String::new(),
            schema: Schema::Snapshot,
            object_candidate_frames: self
                .candidates
                .keys()
                .map(|addr| addr.decision_fingerprint.clone())
                .collect(),
            terminal: self.state.players.iter().any(|p| !p.alive),
            managym_version: env!("CARGO_PKG_VERSION").into(),
            card_digest: default_content_pack().content_digest(),
            seed: self.reset_seed,
            state_digest: state_digest(
                &self.state,
                &self.action_space,
                &self.pending_choice,
                self.skip_trivial,
                self.skipped,
            ),
            provenance: provenance.clone(),
        };
        recipe.recipe_digest = recipe.digest()?;
        serde_json::to_string(&recipe).map_err(|e| AgentError(e.to_string()))
    }

    /// Rebuild the position from private inputs and verify its rules digest.
    pub fn from_recipe_json(raw: &str) -> Result<Self, AgentError> {
        Self::replay_recipe(raw, |_| {})?.snapshot()
    }

    /// A boundary ordinal counts actions and interventions already applied.
    /// Recording readers use these locations to reissue historical object addresses.
    pub(crate) fn replay_recipe(
        raw: &str,
        mut issued_frame: impl FnMut(usize),
    ) -> Result<Env, AgentError> {
        let mut recipe: Recipe = serde_json::from_str(raw)
            .map_err(|e| AgentError(format!("invalid snapshot recipe: {e}")))?;
        if recipe.card_digest != default_content_pack().content_digest() {
            return Err(AgentError("snapshot card_digest mismatch".into()));
        }
        let expected = std::mem::take(&mut recipe.recipe_digest);
        if recipe.digest()? != expected {
            return Err(AgentError("snapshot recipe_digest mismatch".into()));
        }
        let provenance = recipe.provenance;
        if provenance.decision_memory_include_own && !provenance.decision_memory {
            return Err(AgentError(
                "invalid initial decision memory settings".into(),
            ));
        }
        if provenance
            .interventions
            .last()
            .is_some_and(|item| item.at_action > provenance.actions.len())
            || provenance
                .interventions
                .windows(2)
                .any(|pair| pair[0].at_action > pair[1].at_action)
        {
            return Err(AgentError("invalid snapshot intervention offset".into()));
        }
        let mut remaining_frames = recipe.object_candidate_frames;

        let mut env = Env::new(recipe.seed, provenance.skip_trivial, false, false)
            .with_player_policies(provenance.player_policies)
            .with_decision_memory(provenance.decision_memory)
            .with_decision_memory_include_own(provenance.decision_memory_include_own);
        env.reset((*provenance.setup).clone())?;
        let mut interventions = provenance.interventions.iter().peekable();
        let mut applied_interventions = 0;
        for k in 0..=provenance.actions.len() {
            loop {
                let before = remaining_frames.len();
                env.game
                    .as_ref()
                    .expect("reset game")
                    .restore_frame_candidates(&mut remaining_frames)
                    .map_err(|error| {
                        AgentError(format!("snapshot frame at action {k}: {error}"))
                    })?;
                if remaining_frames.len() != before {
                    issued_frame(k + applied_interventions);
                }
                let Some(item) = interventions.next_if(|item| item.at_action == k) else {
                    break;
                };
                env.apply_intervention(item.op).map_err(|error| {
                    AgentError(format!("snapshot intervention at action {k}: {error}"))
                })?;
                applied_interventions += 1;
            }
            if let Some(action) = provenance.actions.get(k) {
                let result = match action {
                    RecordedAction::Index(i) => env.step(*i).map(|_| ()),
                    RecordedAction::Command(raw) => serde_json::from_str(raw)
                        .map_err(|e| AgentError(e.to_string()))
                        .and_then(|c| env.execute_semantic_command(&c).map(|_| ())),
                };
                result.map_err(|e| AgentError(format!("snapshot action {k}: {e}")))?;
            }
        }
        if !remaining_frames.is_empty() {
            return Err(AgentError("snapshot object candidate frame missing".into()));
        }
        if env.is_game_over() != recipe.terminal {
            return Err(AgentError("snapshot terminal status mismatch".into()));
        }
        if env.state_digest()? != recipe.state_digest {
            return Err(AgentError("snapshot state_digest mismatch".into()));
        }
        Ok(env)
    }
}

#[derive(Serialize, Deserialize)]
enum Schema {
    #[serde(rename = "managym-snapshot-v1")]
    Snapshot,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Recipe {
    #[serde(skip_serializing_if = "String::is_empty")]
    recipe_digest: String,
    schema: Schema,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    object_candidate_frames: BTreeSet<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    terminal: bool,
    managym_version: String,
    seed: u64,
    card_digest: String,
    state_digest: String,
    #[serde(flatten)]
    provenance: GameProvenance,
}

/// The rules digest intentionally excludes chooser policies and memory capture.
/// Bind all recipe fields as well so editing those cannot silently change the
/// next step. Canonical object key order makes this independent of formatting.
impl Recipe {
    fn digest(&self) -> Result<String, AgentError> {
        let canonical = serde_json::to_value(self).map_err(|e| AgentError(e.to_string()))?;
        let bytes = serde_json::to_vec(&canonical).map_err(|e| AgentError(e.to_string()))?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }
}

pub(super) fn state_digest(
    state: &TrackedState,
    actions: &Option<ActionSpace>,
    pending: &Option<PendingChoice>,
    skip_trivial: bool,
    skipped: usize,
) -> String {
    let surface = (
        state.deterministic_hash_value(),
        actions,
        pending,
        skip_trivial,
        skipped,
    );
    let canonical = serde_json::to_vec(&surface).expect("semantic differential surface serializes");
    blake3::hash(&canonical).to_hex().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn checked_recipe_reports_the_first_divergent_boundary() {
        let mut source = Env::new(42, true, false, false);
        source
            .reset(
                ["a", "b"]
                    .map(|name| PlayerConfig::new(name, BTreeMap::from([("Forest".into(), 20)])))
                    .to_vec(),
            )
            .unwrap();
        let raw = source.snapshot().unwrap().to_recipe_json().unwrap();
        let fields: serde_json::Value = serde_json::from_str(&raw).unwrap();
        for (key, bad, reason) in [
            ("actions", serde_json::json!([0, 999]), "action 1"),
            (
                "interventions",
                serde_json::json!([{"at_action": 1, "op":"reseed", "seed":9}]),
                "offset",
            ),
            ("state_digest", serde_json::json!("bad"), "state_digest"),
        ] {
            let mut invalid = fields.clone();
            invalid[key] = bad;
            let mut recipe: Recipe = serde_json::from_value(invalid).unwrap();
            recipe.recipe_digest.clear();
            recipe.recipe_digest = recipe.digest().unwrap();
            let raw = serde_json::to_string(&recipe).unwrap();
            let error = GameSnapshot::from_recipe_json(&raw)
                .err()
                .expect("invalid replay");
            assert!(error.to_string().contains(reason), "{error}");
        }
    }

    #[test]
    fn concurrent_exact_snapshots_preserve_the_witness_and_live_decision_memory() {
        let mut source = Env::new(42, true, false, false).with_decision_memory_include_own(true);
        let players = ["a", "b"].map(|name| {
            PlayerConfig::new(
                name,
                BTreeMap::from([("Mountain".into(), 8), ("Gray Ogre".into(), 4)]),
            )
        });
        source.reset(players.to_vec()).unwrap();
        source.step(0).unwrap();
        source.set_seed(99);
        let snapshot = Arc::new(source.snapshot().unwrap());
        let prefix = source.selected_witness().unwrap();
        let logs = source.game.as_ref().unwrap().decision_memory.clone();
        assert!(logs.iter().all(|log| !log.is_empty()));
        let generation = source.capture_generation;
        source.step(0).unwrap();
        let after = source.selected_witness().unwrap();
        let next_logs = source.game.as_ref().unwrap().decision_memory.clone();
        assert_ne!(logs, next_logs);
        drop(source);

        let cold =
            Arc::new(GameSnapshot::from_recipe_json(&snapshot.to_recipe_json().unwrap()).unwrap());
        let mut generations = BTreeSet::from([generation]);
        thread::scope(|scope| {
            let handles: Vec<_> = [snapshot.clone(), snapshot.clone(), cold]
                .into_iter()
                .map(|snapshot| {
                    let (prefix, logs, after, next_logs) = (&prefix, &logs, &after, &next_logs);
                    scope.spawn(move || {
                        let mut branch = snapshot.restore(RestoreOptions::default()).unwrap();
                        assert_eq!(branch.seed(), 42);
                        assert_eq!(&branch.selected_witness().unwrap(), prefix);
                        assert_eq!(&branch.game.as_ref().unwrap().decision_memory, logs);
                        branch.step(0).unwrap();
                        assert_eq!(&branch.selected_witness().unwrap(), after);
                        assert_eq!(&branch.game.as_ref().unwrap().decision_memory, next_logs);
                        branch.capture_generation
                    })
                })
                .collect();
            for handle in handles {
                assert!(generations.insert(handle.join().unwrap()));
            }
        });
        assert_eq!(
            snapshot
                .restore(Default::default())
                .unwrap()
                .selected_witness()
                .unwrap(),
            prefix
        );
    }
}
