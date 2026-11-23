#[cfg(feature = "python")]
use crate::search_state::{identity, SearchStateIdentity};
use crate::state::player::{ManaFloats, ManaPayment};
use crate::{
    agent::{
        action::AgentError,
        heuristic::Heuristic,
        observation::{EventData, Observation},
        observation_encoder::{
            encode, encode_into, EncodedObservation, EncodedObservationMut, ObservationEncodeError,
            ObservationEncoderConfig,
        },
        player_policy::PlayerPolicy,
        structured_offer::{OfferSubmission, StructuredOfferSet},
    },
    cards::{
        content_pack::{default_content_pack, ContentPackManifest},
        names::NameVocabulary,
    },
    decision::{
        Command as SemanticCommand, DecisionFrame, Observation as SemanticObservation,
        SemanticTransition,
    },
    flow::{card_capture::CaptureMode, game::Game, search::mix_seed},
    infra::profiler::{empty_info_dict, insert_info, InfoDict, InfoValue, Profiler},
    possible_worlds::{MaterializeMode, PossibleWorldSpace, PossibleWorldSpaceProjection},
    search_state::{witness, SearchStateWitness},
    state::{
        game_object::PlayerId,
        player::{DecklistDisclosure, GameSetup, PlayerConfig},
        zone::ZoneType,
    },
};
use rand::Rng;
use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

/// Detached authoritative hand for one native seat at the read boundary.
/// Never attach this to viewer observations, decision captures or policy inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "python", pyo3::pyclass(get_all, frozen))]
pub struct HandSnapshot {
    pub player: usize,
    pub game_id: u64,
    pub seed: u64,
    pub revision: u64,
    pub cards: BTreeMap<String, usize>,
}

// A live game's lifetime token never enters replay or world hashes.
// Process-wide allocation distinguishes resets, vector replacements and branches.
static GAME_GENERATION: AtomicU64 = AtomicU64::new(1);

fn current_agent(game: &Game) -> Result<PlayerId, AgentError> {
    game.action_space()
        .ok_or_else(|| AgentError("no active action space".to_string()))?
        .player
        .ok_or_else(|| AgentError("no agent player in current action space".to_string()))
}

fn finish_game_step(
    game: &mut Game,
    agent: PlayerId,
    done: bool,
) -> (Observation, f64, bool, bool, InfoDict) {
    #[cfg(feature = "benchmark")]
    let _timer = crate::benchmark_search::Timer::new("returned_observation_build");
    let events = game.take_observation_events();
    let observation = Observation::new(game, &events);

    let mut reward = 0.0;
    let mut info = empty_info_dict();
    if done {
        if let Some(winner) = game.winner_index() {
            reward = if winner == agent.0 { 1.0 } else { -1.0 };
            insert_info(&mut info, "winner_index", InfoValue::Int(winner as i64));
            insert_info(
                &mut info,
                "winner_name",
                InfoValue::String(game.state.players[winner].name.clone()),
            );
        } else {
            insert_info(
                &mut info,
                "winner_name",
                InfoValue::String("draw".to_string()),
            );
        }
        for i in 0..2 {
            if let Some(reason) = game.state.loss(i).and_then(|loss| loss.reason()) {
                insert_info(
                    &mut info,
                    format!("p{i}_loss_reason"),
                    InfoValue::String(reason.to_string()),
                );
            }
        }
    }

    (observation, reward, done, false, info)
}

/// Result of a flat Monte Carlo evaluation of the current action space.
#[derive(Clone, Debug)]
pub struct FlatMcResult {
    /// Mean playout score per legal action (win = 1.0, loss = 0.0,
    /// draw/step-cap = 0.5), for the player holding the current decision.
    pub scores: Vec<f64>,
    /// Total playouts performed (actions x worlds x rollouts).
    pub simulations: u64,
    /// Playouts that hit the step cap without terminating.
    pub cap_hits: u64,
}

mod provenance;
mod snapshot;
use provenance::{GameProvenance, Intervention, RecordedAction};
pub use snapshot::{BranchMode, GameSnapshot, RestoreOptions, SnapshotMetadata};

#[derive(Debug)]
pub struct Env {
    game: Option<Game>,
    pub(crate) heuristics: [Option<Box<Heuristic>>; 2],
    player_policies: [PlayerPolicy; 2],
    // Observation::for_player deliberately omits events. Built-in index stepping
    // must reuse the live projection's events, especially combat declarations.
    heuristic_events: Vec<EventData>,
    provenance: Option<GameProvenance>,
    reset_setup: Option<Arc<GameSetup>>,
    collect_statistics: bool,
    retained_statistics: Vec<crate::flow::statistics::GameStatistics>,
    drained_generation: Option<u64>,
    pub(crate) record_terminal: bool,
    capture: Option<CaptureMode>,
    retained_exposure: Vec<String>,
    skip_trivial: bool,
    decision_memory_enabled: bool,
    decision_memory_include_own: bool,
    /// Configured seed for the next reset.
    seed: u64,
    /// Originating reset seed, independent of pending resets and branch RNG seeds.
    reset_seed: u64,
    pub profiler: Profiler,
    possible_world_space_constructions: AtomicU64,
    pub(crate) capture_generation: u64,
}

/// One canonical space prepared for bounded materialization against the same
/// live `Env` root. It retains no source snapshot and never re-enumerates.
#[derive(Debug)]
pub struct PreparedPossibleWorldMaterializer {
    space: PossibleWorldSpace,
    space_identity: String,
    expected_space_identity: String,
    max_batch_size: usize,
    construction_count: u64,
}

impl PreparedPossibleWorldMaterializer {
    pub fn viewer(&self) -> usize {
        self.space.viewer().0
    }

    pub fn space_identity(&self) -> &str {
        &self.space_identity
    }

    pub fn support_size(&self) -> usize {
        self.space.worlds().len()
    }

    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    pub fn construction_count(&self) -> u64 {
        self.construction_count
    }

    pub fn materialize_indexes(
        &self,
        source: &Env,
        world_indexes: &[usize],
        seeds: &[u64],
        refresh_opponent_commitment: bool,
    ) -> Result<Vec<Env>, AgentError> {
        if self.space_identity != self.expected_space_identity {
            return Err(AgentError(
                "prepared possible-world space identity changed".to_string(),
            ));
        }
        let game = source.game.as_ref().ok_or_else(|| {
            AgentError("prepared materializer source called before reset".to_string())
        })?;
        let mode = if refresh_opponent_commitment {
            MaterializeMode::RefreshOpponentCommitment
        } else {
            MaterializeMode::PreserveViewerRoot
        };
        let games = self
            .space
            .materialize_indexes(game, world_indexes, seeds, mode, self.max_batch_size)
            .map_err(|error| {
                AgentError(format!("prepared possible-world materializer: {error}"))
            })?;
        Ok(games
            .into_iter()
            .map(|game| source.branch_from_game(game))
            .collect())
    }
}

impl Env {
    pub fn new(
        seed: u64,
        skip_trivial: bool,
        enable_profiler: bool,
        _enable_behavior_tracking: bool,
    ) -> Self {
        Self {
            game: None,
            heuristics: [None, None],
            heuristic_events: Vec::new(),
            player_policies: [PlayerPolicy::Agent; 2],
            provenance: None,
            reset_setup: None,
            collect_statistics: false,
            retained_statistics: Vec::new(),
            drained_generation: None,
            record_terminal: false,
            capture: None,
            retained_exposure: Vec::new(),
            skip_trivial,
            decision_memory_enabled: false,
            decision_memory_include_own: false,
            seed,
            reset_seed: seed,
            profiler: Profiler::new(enable_profiler, 64),
            possible_world_space_constructions: AtomicU64::new(0),
            capture_generation: 0,
        }
    }

    fn construct_possible_world_space(
        &self,
        game: &Game,
        viewer: PlayerId,
    ) -> Result<PossibleWorldSpace, AgentError> {
        game.require_open_decklist(viewer)?;
        self.possible_world_space_constructions
            .fetch_add(1, Ordering::Relaxed);
        Ok(PossibleWorldSpace::for_viewer(game, viewer))
    }

    fn branch_from_game(&self, game: Game) -> Env {
        // Materialized worlds have no replay prefix. Exact forks restore theirs.
        Env {
            game: Some(game),
            heuristics: self.heuristics.clone(),
            heuristic_events: self.heuristic_events.clone(),
            player_policies: self.player_policies,
            reset_seed: self.reset_seed,
            reset_setup: self.reset_setup.clone(),
            decision_memory_enabled: self.decision_memory_enabled,
            decision_memory_include_own: self.decision_memory_include_own,
            capture_generation: GAME_GENERATION.fetch_add(1, Ordering::Relaxed),
            ..Self::new(self.seed, self.skip_trivial, false, false)
        }
    }

    pub fn possible_world_space_construction_count(&self) -> u64 {
        self.possible_world_space_constructions
            .load(Ordering::Relaxed)
    }

    pub fn reset(
        &mut self,
        player_configs: impl Into<GameSetup>,
    ) -> Result<(Observation, InfoDict), AgentError> {
        let mut player_configs = player_configs.into();
        if player_configs.len() != 2 {
            return Err(AgentError(format!(
                "expected exactly 2 player configs, got {}",
                player_configs.len()
            )));
        }
        player_configs.decklists = player_configs.decklists.resolve(self.seed).into();
        // Game creation panics on these; reject them as errors first.
        crate::cards::deck_admission::admit_submitted(&player_configs)
            .map_err(|error| AgentError(error.to_string()))?;
        let content = default_content_pack();
        for player in player_configs.iter() {
            for name in player.deck.main.keys() {
                if content.definition_id(name).is_none() {
                    return Err(AgentError(format!("unknown card in decklist: {name}")));
                }
            }
            for (name, count) in &player.deck.sideboard {
                let valid = content
                    .definition_id(name)
                    .and_then(|id| content.definition(id))
                    .is_some_and(|definition| !definition.is_token);
                if *count == 0 || !valid {
                    return Err(AgentError(format!("invalid sideboard card: {name}")));
                }
            }
        }
        let _scope = self.profiler.track("env_reset");
        if self.capture == Some(CaptureMode::Exposure) && self.game.is_some() {
            self.retained_exposure.push(self.exposure_json()?);
        }
        if self.drained_generation != Some(self.capture_generation) {
            if let Some(summary) = self.statistics() {
                self.retained_statistics.push(summary);
            }
        }
        let mut game = Game::construct(
            &player_configs,
            self.seed,
            self.skip_trivial,
            self.capture,
            self.collect_statistics,
        );
        game.decision_memory_enabled =
            self.decision_memory_enabled || self.decision_memory_include_own;
        game.decision_memory_include_own = self.decision_memory_include_own;
        let setup = Arc::new(player_configs);
        self.reset_setup = Some(setup.clone());
        self.provenance = Some(GameProvenance {
            setup,
            skip_trivial: self.skip_trivial,
            player_policies: self.player_policies,
            decision_memory: game.decision_memory_enabled,
            decision_memory_include_own: self.decision_memory_include_own,
            actions: Arc::default(),
            interventions: Arc::default(),
        });
        let events = game.take_observation_events();
        let observation = Observation::new(&game, &events);
        self.capture_generation = GAME_GENERATION.fetch_add(1, Ordering::Relaxed);
        self.reset_seed = self.seed;
        self.game = Some(game);
        self.heuristics = [None, None];
        drop(_scope);
        self.remember_heuristic_events(&observation);
        let (observation, _, _, _, info) =
            self.settle_players(observation, 0.0, false, false, empty_info_dict())?;
        Ok((observation, info))
    }

    /// Prepare an atomic root replacement without dropping optional recording sinks.
    /// Unlike a search branch, this continues the producer's retained evidence.
    pub(crate) fn replacement_root(&self) -> Result<Self, AgentError> {
        let mut replacement = Self::new(self.seed(), self.skip_trivial, false, false)
            .with_player_policies(self.player_policies)
            .with_decision_memory(self.decision_memory_enabled)
            .with_decision_memory_include_own(self.decision_memory_include_own);
        replacement.collect_statistics = self.collect_statistics;
        replacement.capture = self.capture;
        replacement.record_terminal = self.record_terminal;
        replacement.retained_statistics = self.retained_statistics.clone();
        if self.drained_generation != Some(self.capture_generation) {
            if let Some(summary) = self.statistics() {
                replacement.retained_statistics.push(summary);
            }
        }
        replacement.retained_exposure = self.retained_exposure.clone();
        if self.capture == Some(CaptureMode::Exposure) && self.game.is_some() {
            replacement.retained_exposure.push(self.exposure_json()?);
        }
        if self.profiler.is_enabled() {
            replacement.profiler = Profiler::new(true, 64);
        }
        Ok(replacement)
    }

    pub fn enable_statistics(&mut self) -> Result<(), AgentError> {
        if self.game.is_some() {
            return Err(AgentError("statistics must be enabled before reset".into()));
        }
        self.collect_statistics = true;
        Ok(())
    }

    pub fn statistics(&self) -> Option<crate::flow::statistics::GameStatistics> {
        let game = self.game.as_ref()?;
        let stats = game.statistics.as_ref()?;
        Some(crate::flow::statistics::GameStatistics {
            seed: self.reset_seed,
            generation: self.capture_generation,
            stream: 0,
            terminal: game.is_game_over(),
            winner: game.winner_index(),
            cards: stats.rows(),
        })
    }

    pub fn take_statistics(&mut self) -> Vec<crate::flow::statistics::GameStatistics> {
        if self.is_game_over() && self.drained_generation != Some(self.capture_generation) {
            if let Some(summary) = self.statistics() {
                self.retained_statistics.push(summary);
                self.drained_generation = Some(self.capture_generation);
            }
        }
        std::mem::take(&mut self.retained_statistics)
    }

    /// Must be enabled before reset so setup facts cannot be missed.
    pub fn enable_card_capture(&mut self) -> Result<(), AgentError> {
        if self.game.is_some() {
            return Err(AgentError(
                "card capture must be enabled before reset".into(),
            ));
        }
        self.capture.get_or_insert(CaptureMode::Cards);
        Ok(())
    }

    /// Training uses the same card capture, adding consumed choices and completed frames.
    pub fn enable_exposure_capture(&mut self) -> Result<(), AgentError> {
        self.enable_card_capture()?;
        self.capture = Some(CaptureMode::Exposure);
        Ok(())
    }

    pub fn exposure_json(&self) -> Result<String, AgentError> {
        let game = self
            .game
            .as_ref()
            .ok_or_else(|| AgentError("capture before reset".into()))?;
        serde_json::to_string(&serde_json::json!({
            "seed": self.reset_seed,
            "terminal": game.is_game_over(),
            "trace": self.provenance,
            "cards": self.card_capture_value()?,
        }))
        .map_err(|e| AgentError(e.to_string()))
    }

    pub fn take_exposure(&mut self) -> Vec<String> {
        std::mem::take(&mut self.retained_exposure)
    }

    pub fn card_capture_json(&self) -> Result<String, AgentError> {
        serde_json::to_string(&self.card_capture_value()?).map_err(|e| AgentError(e.to_string()))
    }

    fn card_capture_value(&self) -> Result<serde_json::Value, AgentError> {
        let game = self
            .game
            .as_ref()
            .ok_or_else(|| AgentError("capture before reset".into()))?;
        let capture = game
            .card_capture
            .as_ref()
            .ok_or_else(|| AgentError("card capture unavailable".into()))?;
        Ok(serde_json::json!({
            "capture": capture,
            "final_cards": crate::flow::card_capture::snapshots(game),
            "turn": game.state.turn.turn_number,
        }))
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
    }

    /// Configured seed for the next reset.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Seed the live game was reset with; a pending `set_seed` does not change it.
    pub(crate) fn reset_seed(&self) -> u64 {
        self.reset_seed
    }

    /// The current game's resolved disclosure; open before the first reset.
    pub fn decklists(&self) -> DecklistDisclosure {
        self.game
            .as_ref()
            .map_or(Default::default(), |game| game.state.disclosure)
    }

    pub fn mana_payment(&self) -> [ManaPayment; 2] {
        self.game
            .as_ref()
            .map_or([ManaPayment::Manual; 2], |game| game.mana_payment)
    }

    pub fn mana_floats(&self) -> [ManaFloats; 2] {
        self.game
            .as_ref()
            .map_or([ManaFloats::Off; 2], |game| game.mana_floats)
    }

    /// Setups this game was reset with; empty before the first reset.
    pub fn player_configs(&self) -> &[PlayerConfig] {
        self.reset_setup
            .as_deref()
            .map_or(&[], |s| s.players.as_slice())
    }

    /// Number of trivial decision points auto-collapsed by `skip_trivial`
    /// since the current game began. Resets to zero on `reset`.
    pub fn skip_trivial_count(&self) -> usize {
        self.game.as_ref().map_or(0, |game| {
            game.skip_trivial_count + game.skipped_priority_passes
        })
    }

    /// Training-only error context. Public scalar/API errors do not expose
    /// hidden cards. The snapshot may be partially mutated by the failed step;
    /// the reset input and successful prefix reconstruct its pre-action root.
    pub(crate) fn vector_failure_context(&self, action: i64) -> String {
        serde_json::json!({
            "schema": "managym-native-failure-v1",
            "match_state_hash_version": crate::MATCH_STATE_HASH_VERSION,
            "seed": self.reset_seed,
            "skip_trivial": self.skip_trivial,
            "trace": self.provenance,
            "failed_action_index": action,
            "post_error_state": self.game.as_ref().map(|game| game.state.deterministic_hash_value()),
            "post_error_pending_choice": self.game.as_ref().map(|game| &game.pending_choice),
            "post_error_offers": self.game.as_ref().and_then(|game| game.action_space()),
        }).to_string()
    }

    /// Opt in to viewer-safe decision capture before constructing a game.
    pub fn with_decision_memory(mut self, enabled: bool) -> Self {
        assert!(
            self.game.is_none(),
            "decision history must be configured before reset"
        );
        self.decision_memory_enabled = enabled;
        self
    }

    pub fn with_decision_memory_include_own(mut self, enabled: bool) -> Self {
        assert!(
            self.game.is_none(),
            "decision history must be configured before reset"
        );
        self.decision_memory_include_own = enabled;
        self
    }

    /// Set chooser policies without changing game configuration or world identity.
    pub fn with_player_policies(mut self, policies: [PlayerPolicy; 2]) -> Self {
        if self.player_policies != policies && self.game.is_some() {
            self.provenance = None;
        }
        self.player_policies = policies;
        self
    }

    /// A negative action requests one built-in decision. Mixed games then settle
    /// built-ins until an external chooser or terminal; two built-ins step once.
    pub fn step(
        &mut self,
        action: i64,
    ) -> Result<(Observation, f64, bool, bool, InfoDict), AgentError> {
        let chooser = self
            .current_agent_index()
            .ok_or_else(|| AgentError("no active chooser".into()))?;
        let external_action = action;
        let policy = self.player_policies[chooser];
        let action = if policy == PlayerPolicy::Agent {
            if action == -1 {
                return Err(AgentError(format!(
                    "external chooser (seat {chooser}) requires an action index"
                )));
            }
            action
        } else {
            if action != -1 {
                return Err(AgentError(
                    "built-in chooser requires action=None (vector: -1)".into(),
                ));
            }
            let mut obs = self.observation_for_player(chooser)?;
            obs.recent_events.clone_from(&self.heuristic_events);
            policy.select_for_step(self, &obs)?
        };
        let result = self.step_and_settle(chooser, action)?;
        if let Some(trace) = &mut self.provenance {
            trace.push(RecordedAction::Index(external_action));
        }
        Ok(result)
    }

    fn step_and_settle(
        &mut self,
        chooser: usize,
        action: i64,
    ) -> Result<(Observation, f64, bool, bool, InfoDict), AgentError> {
        let (obs, mut reward, terminated, truncated, info) = self.step_once(action)?;
        // A mixed game's rewards belong to its external seat.
        if self.external_seat().is_some_and(|seat| seat != chooser) {
            reward = -reward;
        }
        self.settle_players(obs, reward, terminated, truncated, info)
    }

    /// The agent seat of a mixed game; None when both or neither seat is external.
    fn external_seat(&self) -> Option<usize> {
        match self
            .player_policies
            .map(|policy| policy == PlayerPolicy::Agent)
        {
            [true, false] => Some(0),
            [false, true] => Some(1),
            _ => None,
        }
    }

    /// Play a mixed game's built-in decisions until its external seat chooses
    /// or the game ends, crediting each reward to the external seat.
    fn settle_players(
        &mut self,
        mut obs: Observation,
        mut reward: f64,
        mut terminated: bool,
        mut truncated: bool,
        mut info: InfoDict,
    ) -> Result<(Observation, f64, bool, bool, InfoDict), AgentError> {
        let Some(external) = self.external_seat() else {
            return Ok((obs, reward, terminated, truncated, info));
        };
        for _ in 0..10_000 {
            let chooser = obs.agent.player_index as usize;
            if terminated || truncated || chooser == external {
                return Ok((obs, reward, terminated, truncated, info));
            }
            let action = self.player_policies[chooser].select_for_step(self, &obs)?;
            (obs, reward, terminated, truncated, info) = self.step_once(action)?;
            reward = -reward;
        }
        Err(AgentError(
            "exceeded 10000 built-in decisions without an external chooser".into(),
        ))
    }

    fn step_once(
        &mut self,
        action: i64,
    ) -> Result<(Observation, f64, bool, bool, InfoDict), AgentError> {
        let _scope = self.profiler.track("env_step");
        let game = self
            .game
            .as_mut()
            .ok_or_else(|| AgentError("env.step called before reset".to_string()))?;

        if game.is_game_over() {
            return Err(AgentError("env.step called after game over".to_string()));
        }

        let action_space = game
            .action_space()
            .ok_or_else(|| AgentError("no active action space".to_string()))?;
        let agent = action_space
            .player
            .ok_or_else(|| AgentError("no agent player in current action space".to_string()))?;
        let action_count = action_space.actions.len();
        let out_of_bounds = || {
            AgentError(format!(
                "Action index {action} out of bounds: {action_count}"
            ))
        };
        let action = match usize::try_from(action) {
            Ok(index) if index < action_count => index,
            _ => return Err(out_of_bounds()),
        };

        let done = game.step(action)?;
        let mut result = finish_game_step(game, agent, done);
        // The profiler scope borrows self, so this cannot call remember_heuristic_events.
        if self.player_policies.contains(&PlayerPolicy::Heuristic) {
            self.heuristic_events.clone_from(&result.0.recent_events);
        }
        if done {
            self.add_profiler_info(&mut result.4);
        }
        Ok(result)
    }

    fn remember_heuristic_events(&mut self, observation: &Observation) {
        if self.player_policies.contains(&PlayerPolicy::Heuristic) {
            self.heuristic_events.clone_from(&observation.recent_events);
        }
    }

    fn require_external_chooser(&self) -> Result<(), AgentError> {
        if self
            .current_agent_index()
            .is_some_and(|seat| self.player_policies[seat] != PlayerPolicy::Agent)
        {
            return Err(AgentError(
                "built-in chooser requires native stepping".into(),
            ));
        }
        Ok(())
    }

    /// Project the exact currently supported structured decision.
    pub fn structured_offers(&self) -> Result<StructuredOfferSet, AgentError> {
        self.game
            .as_ref()
            .ok_or_else(|| AgentError("env.structured_offers called before reset".to_string()))?
            .structured_offers()
            .map_err(|error| AgentError(error.to_string()))
    }

    /// Project the complete action-aligned structured surface used by search.
    pub fn compound_offers(&self) -> Result<StructuredOfferSet, AgentError> {
        self.game
            .as_ref()
            .ok_or_else(|| AgentError("env not reset".into()))?
            .compound_offers()
            .map_err(|error| AgentError(error.to_string()))
    }

    pub fn compound_commands(
        &self,
        offers: &StructuredOfferSet,
        submission: &OfferSubmission,
    ) -> Result<Vec<SemanticCommand>, AgentError> {
        self.game
            .as_ref()
            .ok_or_else(|| AgentError("env not reset".into()))?
            .compound_commands(offers, submission)
            .map_err(|error| AgentError(error.to_string()))
    }

    pub fn structured_search_offers(&self) -> Result<StructuredOfferSet, AgentError> {
        self.game
            .as_ref()
            .ok_or_else(|| {
                AgentError("env.structured_search_offers called before reset".to_string())
            })?
            .structured_search_offers()
            .map_err(|error| AgentError(error.to_string()))
    }

    /// Apply one prompt-bound structured submission through the atomic path.
    pub fn step_structured(
        &mut self,
        offers: &StructuredOfferSet,
        submission: &OfferSubmission,
    ) -> Result<(Observation, f64, bool, bool, InfoDict, usize), AgentError> {
        self.require_external_chooser()?;
        let mut result = {
            let game = self
                .game
                .as_mut()
                .ok_or_else(|| AgentError("env.step_structured called before reset".to_string()))?;
            let agent = current_agent(game)?;
            let done = game
                .apply_offer_submission(offers, submission)
                .map_err(|error| AgentError(error.to_string()))?;
            finish_game_step(game, agent, done)
        };
        if result.2 {
            self.add_profiler_info(&mut result.4);
        }
        self.provenance = None;
        self.remember_heuristic_events(&result.0);
        let (observation, reward, terminated, truncated, info) = result;
        Ok((observation, reward, terminated, truncated, info, 1))
    }

    /// Apply one structured submission through the independent positional ABI
    /// adapter. The final count is the number of legacy actions consumed.
    pub fn step_legacy_submission(
        &mut self,
        offers: &StructuredOfferSet,
        submission: &OfferSubmission,
    ) -> Result<(Observation, f64, bool, bool, InfoDict, usize), AgentError> {
        self.require_external_chooser()?;
        let (mut result, legacy_actions) = {
            let game = self.game.as_mut().ok_or_else(|| {
                AgentError("env.step_legacy_submission called before reset".to_string())
            })?;
            let agent = current_agent(game)?;
            let (done, legacy_actions) = game
                .apply_legacy_offer_submission(offers, submission)
                .map_err(|error| AgentError(error.to_string()))?;
            (finish_game_step(game, agent, done), legacy_actions)
        };
        if result.2 {
            self.add_profiler_info(&mut result.4);
        }
        self.provenance = None;
        self.remember_heuristic_events(&result.0);
        let (observation, reward, terminated, truncated, info) = result;
        Ok((
            observation,
            reward,
            terminated,
            truncated,
            info,
            legacy_actions,
        ))
    }

    /// Project the current shared semantic decision frame. See
    /// [`Game::semantic_decision_frame`].
    pub fn semantic_decision_frame(&self) -> Result<DecisionFrame, AgentError> {
        self.game
            .as_ref()
            .ok_or_else(|| {
                AgentError("env.semantic_decision_frame called before reset".to_string())
            })?
            .semantic_decision_frame()
            .map_err(|error| AgentError(error.to_string()))
    }

    pub fn payment_preview(
        &self,
        viewer: usize,
        revision: u64,
        fingerprint: &str,
        action_index: usize,
    ) -> Result<crate::flow::mana::PaymentPreview, AgentError> {
        self.game
            .as_ref()
            .ok_or_else(|| AgentError("no active game".into()))?
            .payment_preview(PlayerId(viewer), revision, fingerprint, action_index)
    }

    /// Read-only whole-declaration preview; see [`Game::combat_preview`].
    pub fn combat_preview(
        &self,
        viewer: usize,
        revision: u64,
        fingerprint: &str,
        plan: &crate::flow::combat_preview::CombatPlan,
    ) -> Result<crate::flow::combat_preview::CombatPreview, AgentError> {
        self.game
            .as_ref()
            .ok_or_else(|| AgentError("no active game".into()))?
            .combat_preview(PlayerId(viewer), revision, fingerprint, plan)
    }

    /// Public event identities used to explain a committed transition.
    pub fn presentation_subjects(
        &self,
    ) -> Result<std::collections::BTreeMap<String, crate::experience::SubjectRef>, AgentError> {
        Ok(self
            .game
            .as_ref()
            .ok_or_else(|| AgentError("no active game".into()))?
            .presentation_subjects())
    }

    pub fn decision_offer_subjects(
        &self,
        viewer: usize,
    ) -> Result<Vec<Vec<crate::experience::OfferSubjectView>>, AgentError> {
        if viewer >= 2 {
            return Err(AgentError("viewer out of bounds".into()));
        }
        Ok(self
            .game
            .as_ref()
            .ok_or_else(|| AgentError("no active game".into()))?
            .decision_offer_subjects(PlayerId(viewer)))
    }

    pub fn player_controls(&self) -> Result<Vec<crate::experience::PlayerControlView>, AgentError> {
        let game = self
            .game
            .as_ref()
            .ok_or_else(|| AgentError("env.player_controls called before reset".into()))?;
        Ok(game.player_control_views())
    }

    pub fn decision_context(
        &self,
        viewer: usize,
    ) -> Result<Option<crate::experience::DecisionContextView>, AgentError> {
        if viewer >= 2 {
            return Err(AgentError("viewer out of bounds".into()));
        }
        Ok(self
            .game
            .as_ref()
            .ok_or_else(|| AgentError("no active game".into()))?
            .decision_context(PlayerId(viewer)))
    }

    /// Identical for both viewers: only committed public turn/phase state.
    pub fn public_schedule(&self) -> Result<crate::experience::ScheduleView, AgentError> {
        Ok(self
            .game
            .as_ref()
            .ok_or_else(|| AgentError("no active game".into()))?
            .state
            .turn
            .public_schedule())
    }

    pub fn capture_decision(
        &self,
        viewer: usize,
        history_capacity: usize,
    ) -> Result<crate::decision_capture::DecisionCapture, AgentError> {
        self.game
            .as_ref()
            .ok_or_else(|| AgentError("no active game".into()))?
            .capture_decision(PlayerId(viewer), history_capacity)
            .map_err(|error| AgentError(error.to_string()))
    }

    /// Project the composite viewer-safe semantic observation for `viewer`.
    /// See [`Game::semantic_observation`].
    pub fn semantic_observation(&self, viewer: usize) -> Result<SemanticObservation, AgentError> {
        self.game
            .as_ref()
            .ok_or_else(|| AgentError("env.semantic_observation called before reset".to_string()))?
            .semantic_observation(PlayerId(viewer))
            .map_err(|error| AgentError(error.to_string()))
    }

    /// The exact semantic observation identity, without its composite envelope.
    pub fn semantic_viewer_state_hash(&self, viewer: usize) -> Result<String, AgentError> {
        self.game
            .as_ref()
            .ok_or_else(|| AgentError("env.semantic_viewer_state_hash called before reset".into()))?
            .semantic_viewer_state_hash(PlayerId(viewer))
            .map_err(|error| AgentError(error.to_string()))
    }

    /// Project hidden-hand constraints without enumerating compatible hands.
    pub fn hidden_hand_constraints(
        &self,
        viewer: usize,
    ) -> Result<crate::possible_worlds::HiddenHandConstraints, AgentError> {
        let game = self.game.as_ref().ok_or_else(|| {
            AgentError("env.hidden_hand_constraints called before reset".to_string())
        })?;
        if viewer >= game.state.players.len() {
            return Err(AgentError(
                "hidden_hand_constraints: invalid viewer".to_string(),
            ));
        }
        game.require_open_decklist(PlayerId(viewer))?;
        Ok(crate::possible_worlds::HiddenHandConstraints::for_viewer(
            game,
            PlayerId(viewer),
        ))
    }

    /// Canonical viewer-relative possible-world space. Enumeration, exact
    /// physical-deal weights, ordering, and identity all remain managym-owned.
    pub fn possible_world_space(
        &self,
        viewer: usize,
    ) -> Result<PossibleWorldSpaceProjection, AgentError> {
        let game = self.game.as_ref().ok_or_else(|| {
            AgentError("env.possible_world_space called before reset".to_string())
        })?;
        if viewer >= game.state.players.len() {
            return Err(AgentError(format!(
                "possible_world_space: viewer {viewer} out of range"
            )));
        }
        Ok(self
            .construct_possible_world_space(game, PlayerId(viewer))?
            .projection())
    }

    pub fn possible_world_support(
        &self,
        viewer: usize,
        space_identity: &str,
        query: crate::possible_worlds::WorldQueryWire,
    ) -> Result<crate::possible_worlds::SupportReceiptProjection, AgentError> {
        let game = self.game.as_ref().ok_or_else(|| {
            AgentError("env.possible_world_support called before reset".to_string())
        })?;
        if viewer >= game.state.players.len() {
            return Err(AgentError(format!(
                "possible_world_support: viewer {viewer} out of range"
            )));
        }
        let space = self.construct_possible_world_space(game, PlayerId(viewer))?;
        if space.identity() != space_identity {
            return Err(AgentError(
                "possible_world_support: space identity mismatch".to_string(),
            ));
        }
        Ok(space.support_projection(query))
    }

    pub fn possible_world_condition(
        &self,
        viewer: usize,
        space_identity: &str,
        query: crate::possible_worlds::WorldQueryWire,
    ) -> Result<crate::possible_worlds::ConditioningReceiptProjection, AgentError> {
        let game = self.game.as_ref().ok_or_else(|| {
            AgentError("env.possible_world_condition called before reset".to_string())
        })?;
        if viewer >= game.state.players.len() {
            return Err(AgentError(format!(
                "possible_world_condition: viewer {viewer} out of range"
            )));
        }
        let space = self.construct_possible_world_space(game, PlayerId(viewer))?;
        if space.identity() != space_identity {
            return Err(AgentError(
                "possible_world_condition: space identity mismatch".to_string(),
            ));
        }
        space
            .condition_projection(query)
            .map_err(|error| AgentError(format!("possible_world_condition: {error:?}")))
    }

    /// Direct sampled-hand materialization; never enumerates possible worlds.
    pub fn materialize_sampled_hand(
        &self,
        viewer: usize,
        source_json: &str,
        hand: &std::collections::BTreeMap<String, u32>,
        seed: u64,
    ) -> Result<Env, AgentError> {
        let constraints = self.hidden_hand_constraints(viewer)?;
        let expected: serde_json::Value = serde_json::from_str(source_json)
            .map_err(|error| AgentError(format!("invalid hand constraints: {error}")))?;
        if serde_json::to_value(&constraints).map_err(|error| AgentError(error.to_string()))?
            != expected
        {
            return Err(AgentError(
                "sampled hand constraints are stale or inconsistent".to_string(),
            ));
        }
        let game = self
            .game
            .as_ref()
            .ok_or_else(|| AgentError("sampled hand before reset".to_string()))?;
        let branch = constraints
            .materialize_hand(game, hand, seed)
            .map_err(|error| AgentError(format!("sampled hand: {error}")))?;
        Ok(self.branch_from_game(branch))
    }

    /// Materialize one canonical world index into an isolated branch. The
    /// supplied identity must match the current source exactly.
    pub fn materialize_possible_world(
        &self,
        viewer: usize,
        space_identity: &str,
        world_index: usize,
        seed: u64,
        refresh_opponent_commitment: bool,
    ) -> Result<Env, AgentError> {
        let game = self.game.as_ref().ok_or_else(|| {
            AgentError("env.materialize_possible_world called before reset".to_string())
        })?;
        if viewer >= game.state.players.len() {
            return Err(AgentError(format!(
                "materialize_possible_world: viewer {viewer} out of range"
            )));
        }
        let space = self.construct_possible_world_space(game, PlayerId(viewer))?;
        if space.identity() != space_identity {
            return Err(AgentError(
                "materialize_possible_world: space identity mismatch".to_string(),
            ));
        }
        let mode = if refresh_opponent_commitment {
            MaterializeMode::RefreshOpponentCommitment
        } else {
            MaterializeMode::PreserveViewerRoot
        };
        let branch_game = space
            .materialize_index(game, world_index, seed, mode)
            .map_err(|error| AgentError(format!("materialize_possible_world: {error}")))?;
        Ok(self.branch_from_game(branch_game))
    }

    /// Enumerate and bind one canonical space to this live source. Later batch
    /// calls validate the same root and index the retained rows directly.
    pub fn prepare_possible_world_materializer(
        &self,
        viewer: usize,
        expected_space_identity: &str,
        max_batch_size: usize,
    ) -> Result<PreparedPossibleWorldMaterializer, AgentError> {
        let game = self.game.as_ref().ok_or_else(|| {
            AgentError("prepare_possible_world_materializer called before reset".to_string())
        })?;
        if viewer >= game.state.players.len() {
            return Err(AgentError(format!(
                "prepare_possible_world_materializer: viewer {viewer} out of range"
            )));
        }
        if expected_space_identity.is_empty() {
            return Err(AgentError(
                "prepare_possible_world_materializer: expected identity must not be empty"
                    .to_string(),
            ));
        }
        if max_batch_size == 0 {
            return Err(AgentError(
                "prepare_possible_world_materializer: max batch size must be positive".to_string(),
            ));
        }
        let before = self.possible_world_space_construction_count();
        let space = self.construct_possible_world_space(game, PlayerId(viewer))?;
        let space_identity = space.identity();
        if space_identity != expected_space_identity {
            return Err(AgentError(
                "prepare_possible_world_materializer: space identity mismatch".to_string(),
            ));
        }
        let construction_count = self.possible_world_space_construction_count() - before;
        debug_assert_eq!(construction_count, 1);
        Ok(PreparedPossibleWorldMaterializer {
            space,
            expected_space_identity: expected_space_identity.to_string(),
            space_identity,
            max_batch_size,
            construction_count,
        })
    }

    /// Validate and apply one revision-bound semantic Command atomically,
    /// returning the fail-closed receipt and next observation. See
    /// [`Game::execute_semantic_command`].
    pub fn execute_semantic_command(
        &mut self,
        command: &SemanticCommand,
    ) -> Result<SemanticTransition, AgentError> {
        self.commit_command(command)
            .map(|(transition, _, _)| transition)
    }

    fn commit_command(
        &mut self,
        command: &SemanticCommand,
    ) -> Result<(SemanticTransition, Observation, bool), AgentError> {
        self.require_external_chooser()?;
        let (transition, observation, done) = self
            .game
            .as_mut()
            .ok_or_else(|| AgentError("env.execute_semantic_command called before reset".into()))?
            .execute_semantic_command_with_observation(command)
            .map_err(|error| AgentError(error.to_string()))?;
        self.remember_heuristic_events(&observation);
        if let Some(trace) = &mut self.provenance {
            trace.push(RecordedAction::Command(
                serde_json::to_string(command).expect("Command serializes"),
            ));
        }
        Ok((transition, observation, done))
    }

    pub fn step_semantic_command(
        &mut self,
        command: &SemanticCommand,
    ) -> Result<(SemanticTransition, Observation, f64, bool, bool, InfoDict), AgentError> {
        let game = self
            .game
            .as_ref()
            .ok_or_else(|| AgentError("env.step_semantic_command called before reset".into()))?;
        let actor = current_agent(game)?;
        let (transition, observation, done) = self.commit_command(command)?;
        let reward = if done {
            self.winner_index()
                .map(|winner| if winner == actor.0 { 1.0 } else { -1.0 })
                .unwrap_or(0.0)
        } else {
            0.0
        };
        Ok((
            transition,
            observation,
            reward,
            done,
            false,
            empty_info_dict(),
        ))
    }

    /// Transport-only counts for both viewer logs. Pair with capture_generation;
    /// neither seeds nor policy observations authorize append-only reuse.
    pub fn decision_memory_cursors(&self) -> Result<[[u64; 5]; 2], AgentError> {
        let game = self
            .game
            .as_ref()
            .ok_or_else(|| AgentError("env.decision_memory_cursors called before reset".into()))?;
        Ok(std::array::from_fn(|viewer| {
            let log = &game.decision_memory[viewer];
            [
                log.len() as u64,
                log.detail_count() as u64,
                log.reference_count() as u64,
                log.member_count() as u64,
                log.tail_version(),
            ]
        }))
    }

    pub fn decision_memory_counts(&self) -> Result<[[usize; 4]; 2], AgentError> {
        let game = self
            .game
            .as_ref()
            .ok_or_else(|| AgentError("env.decision_memory_counts called before reset".into()))?;
        Ok(std::array::from_fn(|viewer| {
            let log = &game.decision_memory[viewer];
            [
                log.len(),
                log.detail_count(),
                log.reference_count(),
                log.member_count(),
            ]
        }))
    }

    pub fn semantic_event_cursor(&self) -> Result<u64, AgentError> {
        self.game
            .as_ref()
            .map(Game::semantic_event_cursor)
            .ok_or_else(|| AgentError("env.semantic_event_cursor called before reset".to_string()))
    }

    /// Canonical semantic digest used by differential ABI assertions.
    ///
    /// The structured ABI can commit one logical choice where the legacy ABI
    /// publishes several intermediate prompts. Those paths intentionally have
    /// different decision epochs even when they reach the same rules state,
    /// legal action, and pending choice. Exact search-fork admission uses
    /// `search_state::snapshot`, which includes that authority epoch.
    pub fn state_digest(&self) -> Result<String, AgentError> {
        let game = self
            .game
            .as_ref()
            .ok_or_else(|| AgentError("env.state_digest called before reset".to_string()))?;
        Ok(snapshot::state_digest(
            &game.state,
            &game.current_action_space,
            &game.pending_choice,
            game.skip_trivial,
            game.skip_trivial_count,
        ))
    }

    pub fn info(&self) -> InfoDict {
        let _scope = self.profiler.track("env_info");
        let mut info = empty_info_dict();
        self.add_profiler_info(&mut info);

        info
    }

    /// Manifest of the catalog this match runs on.
    pub fn content_pack_manifest(&self) -> Result<ContentPackManifest, AgentError> {
        self.game.as_ref().ok_or_else(|| {
            AgentError("env.content_pack_manifest called before reset".to_string())
        })?;
        Ok(default_content_pack().manifest())
    }

    pub fn encode_observation(
        &self,
        observation: &Observation,
    ) -> Result<EncodedObservation, ObservationEncodeError> {
        encode(observation, &ObservationEncoderConfig::default())
    }

    pub fn encode_observation_into(
        &self,
        observation: &Observation,
        out: EncodedObservationMut<'_>,
    ) -> Result<(), ObservationEncodeError> {
        encode_into(observation, &ObservationEncoderConfig::default(), out)
    }

    pub fn export_profile_baseline(&self) -> String {
        if self.profiler.is_enabled() {
            self.profiler.export_baseline()
        } else {
            String::new()
        }
    }

    pub fn compare_profile(&self, baseline: &str) -> String {
        if self.profiler.is_enabled() {
            self.profiler.compare_to_baseline(baseline)
        } else {
            "Profiler not enabled".to_string()
        }
    }

    /// Independent copy of this env's game for search rollouts.
    ///
    /// Profiling is disabled on the fork; stepping the
    /// fork never mutates the original.
    pub fn fork(&self) -> Result<Env, AgentError> {
        let game = self
            .game
            .as_ref()
            .ok_or_else(|| AgentError("env.fork called before reset".to_string()))?;
        let mut branch = self.branch_from_game(game.clone());
        branch.provenance = self.provenance.clone();
        if game.decision_memory_enabled {
            branch.apply_intervention(Intervention::SetDecisionMemory { enabled: false })?;
        }
        Ok(branch)
    }

    /// Production search fork: an exact full clone of the match.
    pub fn selected_fork(&self) -> Result<Env, AgentError> {
        self.fork()
    }

    pub fn selected_determinize(
        &mut self,
        perspective: usize,
        seed: u64,
    ) -> Result<(), AgentError> {
        self.determinize(perspective, seed)
    }

    pub fn selected_reseed_rollout(&mut self, seed: u64) -> Result<(), AgentError> {
        self.apply_intervention(Intervention::Reseed { seed })
    }

    pub(crate) fn restore_frame_candidates(
        &self,
        take: impl FnOnce(&str) -> Result<bool, AgentError>,
    ) -> Result<(), AgentError> {
        self.game
            .as_ref()
            .ok_or_else(|| AgentError("cannot restore frame candidates before reset".into()))?
            .restore_frame_candidates(take)
    }

    /// Replay the operation payload of a retained boundary intervention.
    #[cfg(feature = "python")]
    pub(crate) fn apply_intervention_json(&mut self, raw: &str) -> Result<(), AgentError> {
        let op = serde_json::from_str(raw)
            .map_err(|e| AgentError(format!("invalid snapshot intervention: {e}")))?;
        self.apply_intervention(op)
    }

    fn apply_intervention(&mut self, op: Intervention) -> Result<(), AgentError> {
        let game = self
            .game
            .as_mut()
            .ok_or_else(|| AgentError("intervention before reset".into()))?;
        match op {
            Intervention::Reseed { seed } => game.reseed(seed),
            Intervention::ShuffleLibraries { seed } => game.shuffle_unseen_libraries(seed),
            Intervention::Determinize { viewer, seed } => {
                let viewer = game.state.sampling_viewer(viewer)?;
                game.determinize(viewer, seed)?;
            }
            Intervention::SetDecisionMemory { enabled } => {
                game.decision_memory_enabled = enabled;
                self.decision_memory_enabled = enabled;
            }
            Intervention::ForceAction { action } => {
                let chooser = self
                    .current_agent_index()
                    .ok_or_else(|| AgentError("no active chooser".into()))?;
                if self.player_policies[chooser] == PlayerPolicy::Agent {
                    return Err(AgentError(
                        "force_action requires a built-in chooser".into(),
                    ));
                }
                self.step_and_settle(chooser, action as i64)?;
            }
        }
        if let Some(trace) = &mut self.provenance {
            trace.record_intervention(op);
        }
        Ok(())
    }

    /// Recompute pre-Command identity from the current authority, without views.
    #[cfg(feature = "python")]
    pub(crate) fn selected_identity(&self) -> Result<SearchStateIdentity, AgentError> {
        let game = self
            .game
            .as_ref()
            .ok_or_else(|| AgentError("env.selected_identity called before reset".to_string()))?;
        Ok(identity(game))
    }

    pub fn selected_witness(&self) -> Result<SearchStateWitness, AgentError> {
        let game = self
            .game
            .as_ref()
            .ok_or_else(|| AgentError("env.selected_witness called before reset".to_string()))?;
        Ok(witness(game))
    }

    /// Complete native support, independent of any policy tensor cutoff.
    pub fn current_action_summary(&self) -> (usize, String) {
        self.game
            .as_ref()
            .and_then(|game| game.action_space())
            .map_or((0, "None".into()), |space| {
                (space.actions.len(), format!("{:?}", space.kind))
            })
    }

    pub fn selected_revision(&self) -> Result<u64, AgentError> {
        self.game
            .as_ref()
            .map(|game| game.decision_epoch)
            .ok_or_else(|| AgentError("env.selected_revision called before reset".to_string()))
    }

    /// Detached authoritative hand contents for a native seat at this read.
    /// Hidden information, separate from observations, captures and policy inputs.
    pub fn hand_snapshot(&self, player: usize) -> Result<HandSnapshot, AgentError> {
        let cards = self.hand_counts(player)?;
        Ok(HandSnapshot {
            player,
            game_id: self.capture_generation,
            seed: self.reset_seed,
            revision: self.selected_revision()?,
            cards,
        })
    }

    /// Privileged read of the native seat's complete hand zone,
    /// without constructing a viewer observation or applying disclosure rules.
    fn hand_counts(&self, player: usize) -> Result<BTreeMap<String, usize>, AgentError> {
        let game = self
            .game
            .as_ref()
            .ok_or_else(|| AgentError("env must be reset before reading hand snapshots".into()))?;
        if player >= game.state.players.len() {
            return Err(AgentError("hand snapshot player must be 0 or 1".into()));
        }
        let mut counts = BTreeMap::new();
        for object in game
            .state
            .zones
            .zone_objects(ZoneType::Hand, PlayerId(player))
            .iter()
        {
            *counts.entry(game.state[*object].name.clone()).or_insert(0) += 1;
        }
        Ok(counts)
    }

    #[cfg(test)]
    pub(super) fn scenario_control_player(&mut self, player: usize, controller: usize) {
        use crate::{flow::turn::PlayerControl, state::ability::PlayerControlDuration};

        let game = self.game.as_mut().expect("reset fixture");
        game.state.turn.player_controls.push(PlayerControl {
            player: PlayerId(player),
            controller: PlayerId(controller),
            duration: PlayerControlDuration::NextTurn,
            active: true,
        });
        game.scenario_refresh_priority().unwrap();
    }

    /// Kind of the current action space, if any.
    pub fn action_space_kind(&self) -> Option<crate::agent::action::ActionSpaceKind> {
        self.game
            .as_ref()
            .and_then(|game| game.action_space())
            .map(|space| space.kind)
    }

    /// Number of legal actions in the current action space.
    pub fn action_count(&self) -> Result<usize, AgentError> {
        let game = self
            .game
            .as_ref()
            .ok_or_else(|| AgentError("env.action_count called before reset".to_string()))?;
        let action_space = game
            .action_space()
            .ok_or_else(|| AgentError("no active action space".to_string()))?;
        Ok(action_space.actions.len())
    }

    /// Player index holding the current decision, if any.
    pub fn current_agent_index(&self) -> Option<usize> {
        self.game
            .as_ref()
            .and_then(|game| game.action_space())
            .and_then(|space| space.player)
            .map(|player| player.0)
    }

    /// Project the current state for one fixed player without following the
    /// acting seat. Study branches use this after a counterfactual command so
    /// an opponent decision cannot turn into an opponent-private observation.
    pub fn observation_for_player(&self, player_index: usize) -> Result<Observation, AgentError> {
        let game = self.game.as_ref().ok_or_else(|| {
            AgentError("env.observation_for_player called before reset".to_string())
        })?;
        if player_index >= game.state.players.len() {
            return Err(AgentError(format!(
                "player index {player_index} is out of bounds"
            )));
        }
        Ok(Observation::for_player(game, PlayerId(player_index)))
    }

    pub fn is_game_over(&self) -> bool {
        self.game.as_ref().is_some_and(|game| game.is_game_over())
    }

    pub fn winner_index(&self) -> Option<usize> {
        self.game.as_ref().and_then(|game| game.winner_index())
    }

    // ------------------------------------------------------------------
    // Scenario / state-injection surface (flow/scenario.rs).
    //
    // FOR TEST AND MEASUREMENT HARNESSES ONLY: bypasses the rules engine.
    // Typical use: reset, inject at the first priority decision, then call
    // scenario_refresh to recompute the action space and observation. The
    // naming fixture requires and republishes an existing naming decision.
    // ------------------------------------------------------------------

    fn scenario_game_mut(&mut self, method: &str) -> Result<&mut Game, AgentError> {
        if self.game.as_ref().is_some_and(|game| game.is_game_over()) {
            return Err(AgentError(format!("env.{method} called after game over")));
        }
        self.provenance = None;
        self.game
            .as_mut()
            .ok_or_else(|| AgentError(format!("env.{method} called before reset")))
    }

    /// Fixture access for opt-in native measurement, absent from Python/wheels.
    /// The harness must refresh priority after changing its constructed board.
    #[cfg(feature = "benchmark")]
    pub fn benchmark_game_mut(&mut self) -> &mut Game {
        self.scenario_game_mut("benchmark_game_mut").unwrap()
    }

    fn scenario_player(player: usize) -> Result<PlayerId, AgentError> {
        if player > 1 {
            return Err(AgentError(format!(
                "scenario: player {player} out of range"
            )));
        }
        Ok(PlayerId(player))
    }

    pub fn scenario_naming_vocabulary(&mut self) -> Result<Arc<NameVocabulary>, AgentError> {
        self.scenario_game_mut("scenario_naming_vocabulary")?
            .scenario_naming_vocabulary()
    }

    pub fn scenario_set_life(&mut self, player: usize, life: i32) -> Result<(), AgentError> {
        let player = Self::scenario_player(player)?;
        self.scenario_game_mut("scenario_set_life")?
            .scenario_set_life(player, life);
        Ok(())
    }

    pub fn scenario_clear_hand(&mut self, player: usize) -> Result<(), AgentError> {
        let player = Self::scenario_player(player)?;
        self.scenario_game_mut("scenario_clear_hand")?
            .scenario_clear_hand(player);
        Ok(())
    }

    pub fn scenario_force_card_in_hand(
        &mut self,
        player: usize,
        name: &str,
    ) -> Result<(), AgentError> {
        let player = Self::scenario_player(player)?;
        self.scenario_game_mut("scenario_force_card_in_hand")?
            .scenario_force_card_in_hand(player, name)
    }

    pub fn scenario_force_battlefield(
        &mut self,
        player: usize,
        name: &str,
        ready: bool,
    ) -> Result<usize, AgentError> {
        let player = Self::scenario_player(player)?;
        self.scenario_game_mut("scenario_force_battlefield")?
            .scenario_force_battlefield(player, name, ready)
            .map(|permanent| permanent.0)
    }

    /// Recompute the current priority action space after injections and
    /// return a fresh observation of the repaired state.
    pub fn scenario_refresh(&mut self) -> Result<Observation, AgentError> {
        let game = self.scenario_game_mut("scenario_refresh")?;
        game.scenario_refresh_priority()?;
        Ok(Observation::new(game, &[]))
    }

    /// Resample hidden information from `perspective`'s point of view.
    /// See [`Game::determinize`].
    pub fn determinize(&mut self, perspective: usize, seed: u64) -> Result<(), AgentError> {
        self.apply_intervention(Intervention::Determinize {
            viewer: perspective,
            seed,
        })
    }

    /// Reseed the game RNG and play both sides uniformly-random-legal to
    /// terminal. Returns the winner index, or None on draw / step cap.
    pub fn random_playout(
        &mut self,
        seed: u64,
        max_steps: usize,
    ) -> Result<Option<usize>, AgentError> {
        let game = self
            .game
            .as_mut()
            .ok_or_else(|| AgentError("env.random_playout called before reset".to_string()))?;
        self.provenance = None;
        game.reseed(seed);
        game.random_playout(max_steps, None)
    }

    /// Flat determinized Monte Carlo evaluation of the current action space.
    ///
    /// For each of `worlds` determinizations (sampled from the perspective of
    /// the player holding the decision), every legal action is applied to a
    /// clone of the world and scored by `rollouts` uniformly-random playouts
    /// (win 1.0 / loss 0.0 / draw-or-cap 0.5). Worlds are shared across
    /// actions (common random numbers) to reduce comparison variance.
    pub fn flat_mc_scores(
        &self,
        worlds: usize,
        rollouts: usize,
        seed: u64,
        max_steps: usize,
    ) -> Result<FlatMcResult, AgentError> {
        self.flat_mc_scores_prepared(
            worlds,
            rollouts,
            max_steps,
            false,
            |game, hero, world_index| {
                let world_seed = mix_seed(seed, world_index as u64);
                let mut world = game.clone();
                world.determinize(hero, world_seed)?;
                Ok((world, world_seed))
            },
        )
    }

    /// Flat MC over canonical possible-world indexes sampled by manabot.
    /// Hand rows and installation remain private to managym.
    pub fn flat_mc_scores_for_worlds(
        &self,
        viewer: usize,
        space_identity: &str,
        world_indexes: &[usize],
        world_seeds: &[u64],
        rollouts: usize,
        max_steps: usize,
    ) -> Result<FlatMcResult, AgentError> {
        if world_indexes.is_empty() {
            return Err(AgentError(
                "flat_mc_scores_for_worlds requires at least one world".to_string(),
            ));
        }
        if world_indexes.len() != world_seeds.len() {
            return Err(AgentError(format!(
                "flat_mc_scores_for_worlds received {} worlds and {} seeds",
                world_indexes.len(),
                world_seeds.len()
            )));
        }
        let game = self.game.as_ref().ok_or_else(|| {
            AgentError("env.flat_mc_scores_for_worlds called before reset".to_string())
        })?;
        let actor = current_agent(game)?;
        if actor.0 != viewer {
            return Err(AgentError(format!(
                "flat_mc_scores_for_worlds viewer {viewer} is not acting player {}",
                actor.0
            )));
        }
        game.require_open_decklist(actor)?;
        let space = PossibleWorldSpace::for_viewer(game, actor);
        if space.identity() != space_identity {
            return Err(AgentError(
                "flat_mc_scores_for_worlds: space identity mismatch".to_string(),
            ));
        }
        self.flat_mc_scores_prepared(
            world_indexes.len(),
            rollouts,
            max_steps,
            true,
            |source, _hero, sample_index| {
                let world_seed = world_seeds[sample_index];
                let world = space
                    .materialize_index(
                        source,
                        world_indexes[sample_index],
                        world_seed,
                        MaterializeMode::PreserveViewerRoot,
                    )
                    .map_err(|error| AgentError(format!("flat_mc_scores_for_worlds: {error}")))?;
                Ok((world, world_seed))
            },
        )
    }

    fn flat_mc_scores_prepared<F>(
        &self,
        worlds: usize,
        rollouts: usize,
        max_steps: usize,
        share_rollout_streams: bool,
        mut prepare: F,
    ) -> Result<FlatMcResult, AgentError>
    where
        F: FnMut(&Game, PlayerId, usize) -> Result<(Game, u64), AgentError>,
    {
        let game = self
            .game
            .as_ref()
            .ok_or_else(|| AgentError("env.flat_mc_scores called before reset".to_string()))?;
        if game.is_game_over() {
            return Err(AgentError(
                "env.flat_mc_scores called after game over".to_string(),
            ));
        }
        let action_space = game
            .action_space()
            .ok_or_else(|| AgentError("no active action space".to_string()))?;
        let hero = action_space
            .player
            .ok_or_else(|| AgentError("no agent player in current action space".to_string()))?;
        game.require_open_decklist(hero)?;
        let num_actions = action_space.actions.len();
        if num_actions == 0 {
            return Err(AgentError("no valid actions available".to_string()));
        }

        let mut totals = vec![0.0f64; num_actions];
        let mut simulations = 0u64;
        let mut cap_hits = 0u64;

        for world_index in 0..worlds {
            let (world, world_seed) = prepare(game, hero, world_index)?;
            #[allow(clippy::needless_range_loop)] // action is a semantic index
            for action in 0..num_actions {
                for rollout in 0..rollouts {
                    let mut sim = world.clone();
                    let stream = if share_rollout_streams {
                        rollout + 1
                    } else {
                        action * rollouts + rollout + 1
                    };
                    sim.reseed(mix_seed(world_seed, stream as u64));
                    let done = sim.step(action)?;
                    let outcome = if done {
                        sim.winner_index()
                    } else {
                        let mut hit_cap = false;
                        let result = sim.random_playout(max_steps, Some(&mut hit_cap))?;
                        if hit_cap {
                            cap_hits += 1;
                        }
                        result
                    };
                    simulations += 1;
                    totals[action] += match outcome {
                        Some(winner) if winner == hero.0 => 1.0,
                        Some(_) => 0.0,
                        None => 0.5,
                    };
                }
            }
        }

        let denominator = (worlds * rollouts).max(1) as f64;
        Ok(FlatMcResult {
            scores: totals.into_iter().map(|t| t / denominator).collect(),
            simulations,
            cap_hits,
        })
    }

    /// Build a batched rollout pool from the current decision point.
    /// See [`crate::agent::rollout_pool::RolloutPool`].
    pub fn rollout_pool(
        &self,
        worlds: usize,
        rollouts: usize,
        seed: u64,
        max_steps: usize,
    ) -> Result<crate::agent::rollout_pool::RolloutPool, AgentError> {
        let game = self
            .game
            .as_ref()
            .ok_or_else(|| AgentError("env.rollout_pool called before reset".to_string()))?;
        crate::agent::rollout_pool::RolloutPool::from_game(game, worlds, rollouts, seed, max_steps)
    }

    pub fn random_action_index(&mut self) -> Result<usize, AgentError> {
        let action = self.sample_action_index()?;
        self.provenance = None;
        Ok(action)
    }

    pub(super) fn forget_provenance(&mut self) {
        self.provenance = None;
    }

    pub(super) fn sample_action_index(&mut self) -> Result<usize, AgentError> {
        let game = self
            .game
            .as_mut()
            .ok_or_else(|| AgentError("env.random_action_index called before reset".to_string()))?;
        let action_space = game
            .action_space()
            .ok_or_else(|| AgentError("no active action space".to_string()))?;
        let count = action_space.actions.len();
        if count == 0 {
            return Err(AgentError("no valid actions available".to_string()));
        }
        Ok(game.state.rng_mut().gen_range(0..count))
    }

    fn add_profiler_info(&self, info: &mut InfoDict) {
        let mut out = empty_info_dict();
        if self.profiler.is_enabled() {
            for (name, stats) in self.profiler.get_stats() {
                let mut scoped = empty_info_dict();
                insert_info(
                    &mut scoped,
                    "total_time",
                    InfoValue::Float(stats.total_time),
                );
                insert_info(&mut scoped, "count", InfoValue::Int(stats.count as i64));
                insert_info(&mut out, name, InfoValue::Map(scoped));
            }
        }
        insert_info(info, "profiler", InfoValue::Map(out));
    }
}

#[cfg(test)]
mod terminal_tests {
    use super::*;

    #[test]
    fn instructed_outcomes_pay_both_seats() {
        use crate::flow::game::WinCause;
        let root = crate::match_fixtures::build_fixture("interactive-midgame-48-w4-v1").unwrap();
        for winner in [0, 1] {
            for explicit_win in [false, true] {
                let mut game = root.clone();
                let loser = 1 - winner;
                if explicit_win {
                    game.win_game(PlayerId(winner), WinCause::MazesEnd);
                } else {
                    game.lose_game(PlayerId(loser));
                }
                for seat in 0..2 {
                    let done = game.is_game_over();
                    let (_, reward, done, _, info) =
                        finish_game_step(&mut game, PlayerId(seat), done);
                    assert!(done);
                    assert_eq!(reward, if seat == winner { 1.0 } else { -1.0 });
                    assert_eq!(info["winner_index"], InfoValue::Int(winner as i64));
                    if !explicit_win {
                        assert_eq!(
                            info[&format!("p{loser}_loss_reason")],
                            InfoValue::String("instructed_loss".into())
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn terminal_rewards_and_reasons_read_the_committed_batch() {
        let root = crate::match_fixtures::build_fixture("interactive-midgame-48-w4-v1").unwrap();
        for simultaneous in [false, true] {
            let mut game = root.clone();
            game.state.players[1]
                .counters
                .change(crate::state::permanent::CounterKind::Poison, 10);
            if simultaneous {
                game.state.players[0].life = 0;
                game.state.players[0].drew_when_empty = true;
            }
            let mut fork = game.clone();
            for position in [&mut game, &mut fork] {
                position.perform_state_based_actions();
                for seat in 0..2 {
                    let (_, reward, done, _, info) =
                        finish_game_step(position, PlayerId(seat), position.is_game_over());
                    assert!(done);
                    assert_eq!(
                        reward,
                        if simultaneous {
                            0.0
                        } else if seat == 0 {
                            1.0
                        } else {
                            -1.0
                        }
                    );
                    assert_eq!(info["p1_loss_reason"], InfoValue::String("poison".into()));
                    if simultaneous {
                        let causes = position.state.loss(0).unwrap();
                        assert!(causes.life_total && causes.deck_empty);
                        assert_eq!(
                            info["p0_loss_reason"],
                            InfoValue::String("deck_empty".into())
                        );
                    }
                }
            }
            assert_eq!(
                game.state.deterministic_hash(),
                fork.state.deterministic_hash()
            );
        }
    }
}

#[cfg(test)]
mod hand_snapshot_tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::Env;
    use crate::state::player::PlayerConfig;

    fn configs() -> Vec<PlayerConfig> {
        ["hero", "villain"]
            .into_iter()
            .map(|name| {
                PlayerConfig::new(
                    name,
                    BTreeMap::from([("Mountain".into(), 8), ("Gray Ogre".into(), 4)]),
                )
            })
            .collect()
    }

    #[test]
    fn hand_snapshot_retains_live_seed_until_successful_reset() {
        let mut env = Env::new(42, true, false, false);
        env.reset(configs()).unwrap();
        let original = env.hand_snapshot(1).unwrap();
        let witness = env.selected_witness().unwrap();
        env.set_seed(99);
        assert_eq!(env.hand_snapshot(1).unwrap(), original);
        assert_eq!(env.selected_witness().unwrap(), witness);
        let mut invalid = configs();
        invalid[0] = PlayerConfig::new("invalid", BTreeMap::from([("Not a card".into(), 8)]))
            .submit_to_world();
        assert!(env.reset(invalid).is_err());
        assert_eq!(env.hand_snapshot(1).unwrap(), original);

        env.reset(configs()).unwrap();
        let reset = env.hand_snapshot(1).unwrap();
        assert_eq!(reset.seed, 99);
        assert_ne!(reset.game_id, original.game_id);
        let mut reference = Env::new(99, true, false, false);
        reference.reset(configs()).unwrap();
        assert_eq!(reset.cards, reference.hand_snapshot(1).unwrap().cards);
        assert_eq!(
            env.selected_witness().unwrap(),
            reference.selected_witness().unwrap()
        );
    }

    #[test]
    fn hand_snapshot_forks_have_distinct_lifetimes_without_changing_replay_state() {
        let mut env = Env::new(42, true, false, false);
        env.reset(configs()).unwrap();
        let original = env.hand_snapshot(1).unwrap();
        let witness = env.selected_witness().unwrap();
        env.set_seed(99);
        let mut ids = BTreeSet::from([original.game_id]);
        for mut branch in [
            env.fork().unwrap(),
            env.fork().unwrap(),
            env.selected_fork().unwrap(),
            env.selected_fork().unwrap(),
        ] {
            let snapshot = branch.hand_snapshot(1).unwrap();
            assert!(ids.insert(snapshot.game_id));
            assert_eq!(snapshot.seed, 42);
            assert_eq!(snapshot.cards, original.cards);
            assert_eq!(snapshot.revision, original.revision);
            assert_eq!(branch.selected_witness().unwrap(), witness);
            assert_eq!(branch.reset_seed(), 42);
            branch.selected_reseed_rollout(123).unwrap();
            assert_eq!(branch.hand_snapshot(1).unwrap(), snapshot);
            branch.step(0).unwrap();
            assert_eq!(branch.hand_snapshot(1).unwrap().game_id, snapshot.game_id);
            assert_eq!(env.hand_snapshot(1).unwrap(), original);
            branch.reset(configs()).unwrap();
            let reset = branch.hand_snapshot(1).unwrap();
            assert!(ids.insert(reset.game_id));
            assert_eq!(reset.seed, 99);
        }
        assert_eq!(env.selected_witness().unwrap(), witness);
    }

    #[test]
    fn hand_snapshot_materializations_keep_reset_seed_and_distinct_lifetimes() {
        let mut env = Env::new(42, true, false, false);
        env.reset(configs()).unwrap();
        let original = env.hand_snapshot(1).unwrap();
        let witness = env.selected_witness().unwrap();
        let constraints = serde_json::to_string(&env.hidden_hand_constraints(0).unwrap()).unwrap();
        let hand = original
            .cards
            .iter()
            .map(|(name, &count)| (name.clone(), count as u32))
            .collect();
        let space = env.possible_world_space(0).unwrap();
        env.set_seed(99);
        let prepared = env
            .prepare_possible_world_materializer(0, &space.identity, 2)
            .unwrap();
        let mut branches = prepared
            .materialize_indexes(&env, &[0, 0], &[123, 123], false)
            .unwrap();
        for _ in 0..2 {
            branches.push(
                env.materialize_possible_world(0, &space.identity, 0, 123, false)
                    .unwrap(),
            );
            branches.push(
                env.materialize_sampled_hand(0, &constraints, &hand, 456)
                    .unwrap(),
            );
        }
        assert_eq!(
            branches[0].selected_witness().unwrap(),
            branches[1].selected_witness().unwrap()
        );
        assert_eq!(
            branches[0].selected_witness().unwrap(),
            branches[2].selected_witness().unwrap()
        );
        let mut ids = BTreeSet::from([original.game_id]);
        for mut branch in branches {
            let snapshot = branch.hand_snapshot(1).unwrap();
            assert!(ids.insert(snapshot.game_id));
            assert_eq!(snapshot.seed, 42);
            assert_eq!(snapshot.cards.values().sum::<usize>(), 7);
            branch.step(0).unwrap();
            assert_eq!(branch.hand_snapshot(1).unwrap().game_id, snapshot.game_id);
            assert_eq!(env.hand_snapshot(1).unwrap(), original);
        }
        assert_eq!(env.selected_witness().unwrap(), witness);
        assert_eq!(
            env.possible_world_space(0).unwrap().identity,
            space.identity
        );
    }
}

#[cfg(test)]
mod content_pack_contract_tests {
    use std::collections::BTreeMap;

    use super::{default_content_pack, Env};
    use crate::{cards::content_pack::CONTENT_PACK_SCHEMA_VERSION, state::player::PlayerConfig};

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

    #[test]
    fn content_pack_contract_covers_env_roots_siblings_and_rollout_slots() {
        let absent = Env::new(1, true, false, false);
        let error = absent
            .fork()
            .expect_err("an environment without a match has no content pack");
        assert_eq!(error.0, "env.fork called before reset");
        let error = absent
            .content_pack_manifest()
            .expect_err("an environment without a match has no manifest");
        assert_eq!(error.0, "env.content_pack_manifest called before reset");

        let mut first = Env::new(11, true, false, false);
        let mut second = Env::new(12, true, false, false);
        first.reset(configs()).expect("first reset");
        second.reset(configs()).expect("second reset");

        let admitted = default_content_pack();
        let expected_digest = admitted.content_digest();
        let first_game = first.game.as_ref().expect("first game");
        let second_game = second.game.as_ref().expect("second game");
        assert_eq!(admitted.schema_version, CONTENT_PACK_SCHEMA_VERSION);
        for (first_card, second_card) in first_game
            .state
            .objects
            .iter()
            .zip(second_game.state.objects.iter())
        {
            assert!(first_card.shares_definition_with(second_card));
        }

        let manifest = first.content_pack_manifest().expect("content manifest");
        assert_eq!(manifest.schema_version, CONTENT_PACK_SCHEMA_VERSION);
        assert_eq!(manifest.content_digest, expected_digest);
        assert_eq!(manifest.definitions.len(), admitted.len());
        for entry in &manifest.definitions {
            let definition = admitted
                .definition(entry.card_def_id)
                .expect("manifest definition");
            assert_eq!(definition.name, entry.registry_name);
            assert_eq!(definition.key, entry.key.as_deref().unwrap_or(""));
        }

        let root_hash = first_game.state.deterministic_hash();
        let root_life = first_game.state.players[0].life;
        let mut changed = first.fork().expect("changed sibling");
        let untouched = first.fork().expect("untouched sibling");

        let action = changed.action_count().expect("changed action count");
        assert!(action > 0);
        changed.step(0).expect("legal branch action");
        changed.game.as_mut().expect("changed game").state.players[0].life -= 3;

        let root = first.game.as_ref().expect("root game");
        let changed_game = changed.game.as_ref().expect("changed game");
        let untouched_game = untouched.game.as_ref().expect("untouched game");
        assert_eq!(root.state.deterministic_hash(), root_hash);
        assert_eq!(untouched_game.state.deterministic_hash(), root_hash);
        assert_ne!(changed_game.state.deterministic_hash(), root_hash);
        assert_eq!(root.state.players[0].life, root_life);
        assert_eq!(untouched_game.state.players[0].life, root_life);
        assert_eq!(changed_game.state.players[0].life, root_life - 3);

        for branch in [changed_game, untouched_game] {
            for (root_card, branch_card) in
                root.state.objects.iter().zip(branch.state.objects.iter())
            {
                assert_eq!(root_card.definition_id, branch_card.definition_id);
                assert!(root_card.shares_definition_with(branch_card));
            }
        }
    }

    #[test]
    fn fixed_player_observation_does_not_follow_the_acting_seat() {
        let absent = Env::new(1, true, false, false);
        assert_eq!(
            absent
                .observation_for_player(0)
                .expect_err("an environment without a match has no projection")
                .0,
            "env.observation_for_player called before reset"
        );

        let mut env = Env::new(11, true, false, false);
        env.reset(configs()).expect("reset");
        let player_zero = env.observation_for_player(0).expect("player zero view");
        let player_one = env.observation_for_player(1).expect("player one view");

        assert_eq!(player_zero.agent.player_index, 0);
        assert_eq!(player_zero.opponent.player_index, 1);
        assert_eq!(player_one.agent.player_index, 1);
        assert_eq!(player_one.opponent.player_index, 0);
        assert!(player_zero
            .opponent_cards
            .iter()
            .all(|card| card.zone != crate::state::zone::ZoneType::Hand));
        assert!(player_one
            .opponent_cards
            .iter()
            .all(|card| card.zone != crate::state::zone::ZoneType::Hand));
        assert_eq!(
            env.observation_for_player(2)
                .expect_err("only two seats exist")
                .0,
            "player index 2 is out of bounds"
        );
    }
}
