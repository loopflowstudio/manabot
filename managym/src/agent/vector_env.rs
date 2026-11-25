//! Batched games with native reset-seed ownership and admitted per-game roster sampling.

use crate::state::player::{ManaFloats, ManaPayment};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
mod slots;
#[cfg(feature = "python")]
pub(crate) use slots::SlotReplacement;
pub use slots::{SlotLoad, SlotLoadResult};

use crate::{
    agent::{
        action::AgentError,
        env::{Env, HandSnapshot},
        observation::Observation,
        observation_encoder::ObservationEncodeError,
        player_policy::PlayerPolicy,
    },
    infra::profiler::{InfoDict, InfoValue},
    state::player::{GameSetup, PlayerConfig},
};

/// Preserve encoder capacity errors across the buffer boundary. Pointer/shape
/// faults and malformed observation payloads are never disposable game outcomes.
#[derive(Debug)]
pub enum ObservationWriteError {
    Encoding(ObservationEncodeError),
    Buffer(AgentError),
}

impl From<AgentError> for ObservationWriteError {
    fn from(error: AgentError) -> Self {
        Self::Buffer(error)
    }
}

impl From<ObservationEncodeError> for ObservationWriteError {
    fn from(error: ObservationEncodeError) -> Self {
        Self::Encoding(error)
    }
}

impl From<ObservationWriteError> for AgentError {
    fn from(error: ObservationWriteError) -> Self {
        match error {
            ObservationWriteError::Encoding(error) => Self(error.to_string()),
            ObservationWriteError::Buffer(error) => error,
        }
    }
}

#[derive(Debug)]
pub struct StepResult {
    pub obs: Observation,
    pub reward: f64,
    pub terminated: bool,
    pub truncated: bool,
    pub info: InfoDict,
}

/// Identity of one viewer-safe encoded row, paired with HandSnapshot at this boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "python", pyo3::pyclass(get_all, frozen))]
pub struct ObservationRead {
    pub slot: usize,
    pub player: usize,
    pub game_id: u64,
    pub seed: u64,
    pub revision: u64,
}

/// One root sampler with explicit pair semantics and roster ownership.
#[derive(Debug, Default)]
struct RootRoster {
    players: Vec<PlayerConfig>,
    distinct: bool,
    elemental: bool,
}

impl RootRoster {
    fn is_empty(&self) -> bool {
        self.players.is_empty()
    }

    fn pair(&self, seed: u64) -> (usize, usize) {
        if !self.distinct {
            return uniform_pair(seed, self.players.len());
        }
        // uniform-distinct-v1: independent of shuffle/policy RNG. Drawing the
        // second index from N-1 makes every ordered non-mirror pair equiprobable.
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x6465636b70616972);
        let first = rng.gen_range(0..self.players.len());
        let second = rng.gen_range(0..self.players.len() - 1);
        (first, second + usize::from(second >= first))
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum SlotState {
    Unset,
    /// Replacing an untouched opening with a snapshot returns its seed.
    FreshOpening,
    Playing,
    Terminal,
    Poisoned,
}

impl SlotState {
    fn gameplay_failure(&mut self, class: &str, message: String, frame: String) -> InfoDict {
        *self = Self::Poisoned;
        InfoDict::from([
            ("game_error".into(), InfoValue::String(message)),
            ("game_error_class".into(), InfoValue::String(class.into())),
            ("game_error_frame".into(), InfoValue::String(frame)),
            (
                "game_error_trace".into(),
                InfoValue::String(std::backtrace::Backtrace::force_capture().to_string()),
            ),
        ])
    }

    fn readable(self) -> Result<(), AgentError> {
        match self {
            Self::Unset => Err(AgentError(
                "vector env must be reset before reading observations".into(),
            )),
            Self::Poisoned => Err(AgentError("cannot read a poisoned slot".into())),
            _ => Ok(()),
        }
    }
}

#[derive(Debug)]
pub struct VectorEnv {
    envs: Vec<Env>,
    player_configs: Vec<GameSetup>,
    player_policies: [PlayerPolicy; 2],
    next_seeds: Vec<u64>,
    seed_stride: u64,
    uniform_roster: RootRoster,
    skip_trivial: bool,
    decision_memory_enabled: bool,
    decision_memory_include_own: bool,
    slots: Vec<SlotState>,
}

impl VectorEnv {
    /// Untimed fixture restoration for matched, one-command benchmark samples.
    /// This API is absent from normal builds and never substitutes for stepping.
    #[cfg(feature = "benchmark")]
    pub fn benchmark_load_root(&mut self, root: &Env) -> Result<(), AgentError> {
        self.validate_ready_for_step()?;
        self.envs = (0..self.len())
            .map(|_| root.fork())
            .collect::<Result<_, _>>()?;
        self.player_policies = [PlayerPolicy::Agent; 2];
        Ok(())
    }

    /// Scenario-only large naming roots, written through the normal vector encoder.
    pub fn scenario_naming_vocabulary_into(
        &mut self,
        mut write: impl FnMut(usize, &Observation) -> Result<(), AgentError>,
    ) -> Result<(), AgentError> {
        self.validate_ready_for_step()?;
        for (index, env) in self.envs.iter_mut().enumerate() {
            env.scenario_naming_vocabulary()?;
            let obs = env.observation_for_player(
                env.current_agent_index()
                    .ok_or_else(|| AgentError("naming fixture has no chooser".into()))?,
            )?;
            write(index, &obs)?;
        }
        Ok(())
    }

    pub fn new(
        num_envs: usize,
        seed: u64,
        skip_trivial: bool,
        player_policies: [PlayerPolicy; 2],
    ) -> Self {
        let envs = (0..num_envs)
            .map(|idx| {
                Env::new(seed.saturating_add(idx as u64), skip_trivial, false, false)
                    .with_player_policies(player_policies)
            })
            .collect();
        let next_seeds = (0..num_envs)
            .map(|idx| seed.saturating_add(idx as u64))
            .collect();

        Self {
            envs,
            player_configs: Vec::new(),
            player_policies,
            next_seeds,
            seed_stride: num_envs.max(1) as u64,
            uniform_roster: RootRoster::default(),
            skip_trivial,
            decision_memory_enabled: false,
            decision_memory_include_own: false,
            slots: vec![SlotState::Unset; num_envs],
        }
    }

    #[cfg(feature = "python")]
    pub(crate) fn parked_slots(&self) -> impl Iterator<Item = usize> + '_ {
        self.slots.iter().enumerate().filter_map(|(slot, state)| {
            matches!(state, SlotState::Terminal | SlotState::Poisoned).then_some(slot)
        })
    }

    pub fn next_fresh_seeds(&self) -> &[u64] {
        &self.next_seeds
    }

    pub fn decision_memory_enabled(&self) -> bool {
        self.decision_memory_enabled
    }

    pub fn with_decision_memory(mut self, enabled: bool) -> Self {
        self.decision_memory_enabled = enabled;
        self.envs = self
            .envs
            .into_iter()
            .map(|env| env.with_decision_memory(enabled))
            .collect();
        self
    }

    pub fn with_decision_memory_include_own(mut self, enabled: bool) -> Self {
        self.decision_memory_include_own = enabled;
        self.envs = self
            .envs
            .into_iter()
            .map(|env| env.with_decision_memory_include_own(enabled))
            .collect();
        self
    }

    pub fn enable_statistics(&mut self) -> Result<(), AgentError> {
        if !self.player_configs.is_empty() {
            return Err(AgentError("statistics must be enabled before reset".into()));
        }
        for env in &mut self.envs {
            env.enable_statistics()?;
        }
        Ok(())
    }

    pub fn statistics(&self) -> Vec<Option<crate::flow::statistics::GameStatistics>> {
        self.envs
            .iter()
            .enumerate()
            .map(|(stream, env)| {
                env.statistics().map(|mut s| {
                    s.stream = stream;
                    s
                })
            })
            .collect()
    }

    pub fn take_statistics(&mut self) -> Vec<crate::flow::statistics::GameStatistics> {
        self.envs
            .iter_mut()
            .enumerate()
            .flat_map(|(stream, env)| {
                env.take_statistics().into_iter().map(move |mut s| {
                    s.stream = stream;
                    s
                })
            })
            .collect()
    }

    pub fn enable_exposure_capture(&mut self) -> Result<(), AgentError> {
        if !self.player_configs.is_empty() {
            return Err(AgentError(
                "exposure capture must be enabled before reset".into(),
            ));
        }
        for env in &mut self.envs {
            env.enable_exposure_capture()?;
        }
        Ok(())
    }

    pub fn take_exposure(&mut self) -> Vec<Vec<String>> {
        self.envs.iter_mut().map(Env::take_exposure).collect()
    }

    /// Detached chooser observations only for requested active streams.
    /// Optional Python analyses call this; the buffered policy path does not.
    pub fn decision_observations(
        &self,
        active: &[bool],
    ) -> Result<Vec<Option<Observation>>, AgentError> {
        self.validate_actions_len(active.len())?;
        self.validate_ready_for_step()?;
        self.envs
            .iter()
            .zip(active)
            .map(|(env, on)| {
                if *on {
                    env.observation_for_player(env.current_agent_index().unwrap_or(0))
                        .map(Some)
                } else {
                    Ok(None)
                }
            })
            .collect()
    }

    pub fn current_exposure(&self) -> Result<Vec<String>, AgentError> {
        self.selected_exposure(None)
    }

    pub(crate) fn selected_exposure(
        &self,
        active: Option<&[bool]>,
    ) -> Result<Vec<String>, AgentError> {
        if let Some(mask) = active {
            self.validate_actions_len(mask.len())?;
        }
        self.envs
            .iter()
            .enumerate()
            .filter(|(slot, _)| active.is_none_or(|mask| mask[*slot]))
            .map(|(_, env)| env.exposure_json())
            .collect()
    }

    /// Configure only before allocation of roots; native authored setups are the authority.
    pub fn set_uniform_roster(&mut self, roster: Vec<PlayerConfig>) -> Result<(), AgentError> {
        if !self.player_configs.is_empty() {
            return Err(AgentError("uniform roster must be set before reset".into()));
        }
        let setup = crate::cards::setups::authored_setup("optional_payments")
            .map_err(|e| AgentError(e.to_string()))?;
        let expected: Vec<PlayerConfig> = setup
            .decks
            .iter()
            .map(|deck| setup.player_config(&deck.key, &deck.key).unwrap())
            .collect();
        if roster.len() != expected.len()
            || expected
                .iter()
                .any(|e| roster.iter().filter(|r| r.deck == e.deck).count() != 1)
        {
            return Err(AgentError(
                "uniform roster requires every exact Elemental setup once".into(),
            ));
        }
        self.uniform_roster = RootRoster {
            players: roster,
            distinct: false,
            elemental: true,
        };
        Ok(())
    }

    /// Replace the training decks sampled at subsequent roots.
    /// Independent draws include mirrors and preserve repeated population entries.
    ///
    /// Every deck is submitted to the world ([`crate::world::WORLD_POOL`]) and
    /// passes its native admission before the replacement takes effect; on any
    /// error the previous decks remain. Games already underway keep their
    /// decks. Membership never grants permission: game creation admits each
    /// submitted deck again.
    pub fn set_training_decks(
        &mut self,
        decks: Vec<PlayerConfig>,
        distinct_pairs: bool,
    ) -> Result<(), AgentError> {
        if self.uniform_roster.elemental {
            return Err(AgentError(
                "the Elemental roster cannot be replaced by training decks".into(),
            ));
        }
        if decks.is_empty() || (distinct_pairs && decks.len() < 2) {
            return Err(AgentError(
                "uniform-distinct-v1 requires at least two training decks".into(),
            ));
        }
        let mut distinct = std::collections::BTreeSet::new();
        for player in &decks {
            if !player.submitted {
                return Err(AgentError(
                    "training decks must be submitted to the world".into(),
                ));
            }
            if let Some(message) = crate::cards::deck_admission::development_rejection(player) {
                return Err(AgentError(format!(
                    "training deck {}: {message}",
                    player.name
                )));
            }
            if distinct_pairs && !distinct.insert(&player.deck) {
                return Err(AgentError(
                    "training decks must be distinct normalized lists".into(),
                ));
            }
        }
        self.uniform_roster = RootRoster {
            players: decks,
            distinct: distinct_pairs,
            elemental: false,
        };
        Ok(())
    }

    /// Each current game's resolved disclosure, never mixed.
    pub fn current_decklists(&self) -> Vec<crate::state::player::DecklistDisclosure> {
        self.envs.iter().map(Env::decklists).collect()
    }

    pub fn current_mana_payment(&self) -> Vec<[ManaPayment; 2]> {
        self.envs.iter().map(Env::mana_payment).collect()
    }

    pub fn current_mana_floats(&self) -> Vec<[ManaFloats; 2]> {
        self.envs.iter().map(Env::mana_floats).collect()
    }

    /// The setups each game was actually reset with, in seat order.
    pub fn current_player_configs(&self) -> Vec<Vec<PlayerConfig>> {
        self.envs
            .iter()
            .map(|env| env.player_configs().to_vec())
            .collect()
    }

    /// Current ordered indices when all games use decks from the fixed Elemental
    /// roster. A detached branch carries its own resolved seats and seed.
    pub fn current_matchup_indices(&self) -> Vec<(usize, usize)> {
        if !self.uniform_roster.elemental {
            return Vec::new();
        }
        let index = |player: &PlayerConfig| {
            self.uniform_roster
                .players
                .iter()
                .position(|p| p.deck == player.deck)
        };
        self.envs
            .iter()
            .map(|env| match env.player_configs() {
                [a, b] => Some((index(a)?, index(b)?)),
                _ => Some(self.uniform_roster.pair(env.seed())),
            })
            .collect::<Option<Vec<_>>>()
            .unwrap_or_default()
    }

    pub fn len(&self) -> usize {
        self.envs.len()
    }

    /// Preserve global stream reset identities when this vector owns a shard.
    /// Configure before resetting; ordinary vectors retain their local stride.
    pub fn with_seed_stride(mut self, stride: u64) -> Result<Self, AgentError> {
        if stride < self.envs.len().max(1) as u64 {
            return Err(AgentError(
                "seed stride must cover all local streams".into(),
            ));
        }
        if !self.player_configs.is_empty() {
            return Err(AgentError("seed stride must be set before reset".into()));
        }
        self.seed_stride = stride;
        Ok(self)
    }

    pub fn is_empty(&self) -> bool {
        self.envs.is_empty()
    }

    /// Seeds of the current games, including parked terminals.
    pub fn current_game_seeds(&self) -> Vec<u64> {
        self.envs.iter().map(Env::seed).collect()
    }

    /// Opaque process-local game generations, changed by every reset or slot
    /// replacement (including a repeated seed). Never a replay or world identity.
    pub fn current_game_generations(&self) -> Vec<u64> {
        self.envs.iter().map(|env| env.capture_generation).collect()
    }

    pub fn decision_memory_cursors(&self) -> Result<Vec<[[u64; 5]; 2]>, AgentError> {
        self.envs.iter().map(Env::decision_memory_cursors).collect()
    }

    pub fn decision_memory_counts(&self) -> Result<Vec<[[usize; 4]; 2]>, AgentError> {
        self.envs.iter().map(Env::decision_memory_counts).collect()
    }

    pub fn enable_profiler(&mut self) {
        for env in &mut self.envs {
            if !env.profiler.is_enabled() {
                env.profiler = crate::infra::profiler::Profiler::new(true, 64);
            }
        }
    }

    /// Select terminal digest retention for the next step.
    pub fn record_games(&mut self, active: &[bool]) -> Result<(), AgentError> {
        self.validate_actions_len(active.len())?;
        for (env, selected) in self.envs.iter_mut().zip(active) {
            env.record_terminal = *selected;
        }
        Ok(())
    }

    /// Hash only selected games, preserving slot order without advancing RNG.
    pub fn state_digests(
        &self,
        active: Option<&[bool]>,
    ) -> Result<Vec<Option<String>>, AgentError> {
        if let Some(mask) = active {
            self.validate_actions_len(mask.len())?;
        }
        self.envs
            .iter()
            .enumerate()
            .map(|(index, env)| {
                if active.is_none_or(|mask| mask[index]) {
                    env.state_digest().map(Some)
                } else {
                    Ok(None)
                }
            })
            .collect()
    }

    /// Read all requested seats in slot order, including parked terminal hands.
    pub fn hand_snapshots(&self, players: &[usize]) -> Result<Vec<HandSnapshot>, AgentError> {
        if players.len() != self.len() {
            return Err(AgentError(format!(
                "expected {} hand snapshot players, got {}",
                self.len(),
                players.len()
            )));
        }
        self.validate_ready_for_step().map_err(|_| {
            AgentError("vector env must be reset before reading hand snapshots".to_string())
        })?;
        self.envs
            .iter()
            .zip(players)
            .map(|(env, &player)| env.hand_snapshot(player))
            .collect()
    }

    /// Project each requested native seat without advancing games or chooser state.
    /// The whole request is validated before the first write; an encoding failure
    /// can leave a written prefix, so callers must discard all rows on failure.
    pub fn write_observations_for_players(
        &self,
        players: &[Option<usize>],
        write: impl FnMut(usize, &Observation) -> Result<(), AgentError>,
    ) -> Result<Vec<ObservationRead>, AgentError> {
        self.read_observations_for_players(players, write)
    }

    pub(crate) fn read_observations_for_players<E: From<AgentError>>(
        &self,
        players: &[Option<usize>],
        mut write: impl FnMut(usize, &Observation) -> Result<(), E>,
    ) -> Result<Vec<ObservationRead>, E> {
        if players.len() != self.len() {
            return Err(AgentError(format!(
                "expected {} observation players, got {}",
                self.len(),
                players.len()
            ))
            .into());
        }
        if players.iter().flatten().any(|&player| player > 1) {
            return Err(AgentError("observation player must be 0 or 1".into()).into());
        }
        let reads = players
            .iter()
            .enumerate()
            .filter_map(|(slot, player)| player.map(|player| (slot, player)))
            .map(|(slot, player)| {
                self.slots[slot].readable()?;
                let env = &self.envs[slot];
                Ok(ObservationRead {
                    slot,
                    player,
                    game_id: env.capture_generation,
                    seed: env.reset_seed(),
                    revision: env.selected_revision()?,
                })
            })
            .collect::<Result<Vec<_>, AgentError>>()?;
        for read in &reads {
            write(
                read.slot,
                &self.envs[read.slot].observation_for_player(read.player)?,
            )?;
        }
        Ok(reads)
    }

    /// Retain the exact surfaced offers before a strict batch can mutate any row.
    /// Paused streams need no frame and must not be queried for a live decision.
    pub fn semantic_decision_frames_json(
        &self,
        active: Option<&[bool]>,
    ) -> Result<Vec<Option<String>>, AgentError> {
        if let Some(mask) = active {
            self.validate_actions_len(mask.len())?;
        }
        self.envs
            .iter()
            .enumerate()
            .map(|(index, env)| {
                if active.is_some_and(|mask| !mask[index]) {
                    return Ok(None);
                }
                serde_json::to_string(&env.semantic_decision_frame()?)
                    .map(Some)
                    .map_err(|error| AgentError(error.to_string()))
            })
            .collect()
    }

    /// Restore current roots before replaying their bounded legal action prefixes.
    /// Reset advances each native next-seed counter exactly as ordinary collection.
    pub fn reset_seeds_into<F>(
        &mut self,
        seeds: Vec<u64>,
        write: F,
    ) -> Result<Vec<InfoDict>, AgentError>
    where
        F: Fn(usize, &Observation, f64, bool, bool) -> Result<(), AgentError>,
    {
        self.reset_seeds_with_writer(seeds, |index, obs, reward, done, cut| {
            write(index, obs, reward, done, cut).map_err(Into::into)
        })
        .map_err(Into::into)
    }

    pub(crate) fn reset_seeds_with_writer(
        &mut self,
        seeds: Vec<u64>,
        write: impl Fn(usize, &Observation, f64, bool, bool) -> Result<(), ObservationWriteError>,
    ) -> Result<Vec<InfoDict>, ObservationWriteError> {
        self.validate_ready_for_step()?;
        self.reset_with_writer(self.player_configs.clone(), Some(seeds), write)
    }

    /// Per-env `skip_trivial` collapse counters for the current games.
    /// Each counter resets when its env's game resets.
    pub fn skip_trivial_counts(&self) -> Vec<usize> {
        self.envs.iter().map(Env::skip_trivial_count).collect()
    }

    /// Per-env index of the player holding the current decision (-1 if no
    /// decision is pending). With two `agent` policies both seats
    /// surface, so callers routing decisions to per-seat policies need this.
    pub fn current_agent_indices(&self) -> Vec<i64> {
        self.envs
            .iter()
            .zip(&self.slots)
            .map(|(env, state)| match state {
                SlotState::FreshOpening | SlotState::Playing => {
                    env.current_agent_index().map_or(-1, |i| i as i64)
                }
                _ => -1,
            })
            .collect()
    }

    pub fn current_action_summaries(&self) -> Vec<(usize, String)> {
        self.envs.iter().map(Env::current_action_summary).collect()
    }

    pub fn reset_all(
        &mut self,
        player_configs: impl Into<GameSetup>,
    ) -> Result<Vec<(Observation, InfoDict)>, AgentError> {
        let player_configs = player_configs.into();
        Self::validate_player_configs(player_configs.len())?;

        self.player_configs = vec![player_configs; self.envs.len()];
        (0..self.len()).map(|slot| self.reset_slot(slot)).collect()
    }

    pub fn step(&mut self, actions: &[i64]) -> Result<Vec<StepResult>, AgentError> {
        self.validate_step(actions, None)?;
        actions
            .iter()
            .enumerate()
            .map(|(slot, &action)| self.step_slot(slot, action))
            .collect()
    }

    pub fn reset_all_into<F>(
        &mut self,
        player_configs: impl Into<GameSetup>,
        write: F,
    ) -> Result<Vec<InfoDict>, AgentError>
    where
        F: Fn(usize, &Observation, f64, bool, bool) -> Result<(), AgentError>,
    {
        self.reset_each_into(vec![player_configs.into(); self.envs.len()], write)
    }

    /// Retain one exact setup per stream, for subsequent fresh loads.
    pub fn reset_each_into<F, S: Into<GameSetup>>(
        &mut self,
        player_configs: Vec<S>,
        write: F,
    ) -> Result<Vec<InfoDict>, AgentError>
    where
        F: Fn(usize, &Observation, f64, bool, bool) -> Result<(), AgentError>,
    {
        self.reset_each_into_at(player_configs, None, write)
    }

    /// Install the supplied roots directly. Each next game continues by the vector stride.
    pub fn reset_each_into_at<F, S: Into<GameSetup>>(
        &mut self,
        player_configs: Vec<S>,
        seeds: Option<Vec<u64>>,
        write: F,
    ) -> Result<Vec<InfoDict>, AgentError>
    where
        F: Fn(usize, &Observation, f64, bool, bool) -> Result<(), AgentError>,
    {
        self.reset_with_writer(player_configs, seeds, |i, obs, r, d, t| {
            write(i, obs, r, d, t).map_err(Into::into)
        })
        .map_err(Into::into)
    }

    pub(crate) fn reset_with_writer<F, S: Into<GameSetup>>(
        &mut self,
        player_configs: Vec<S>,
        seeds: Option<Vec<u64>>,
        write: F,
    ) -> Result<Vec<InfoDict>, ObservationWriteError>
    where
        F: Fn(usize, &Observation, f64, bool, bool) -> Result<(), ObservationWriteError>,
    {
        let player_configs: Vec<GameSetup> = player_configs.into_iter().map(Into::into).collect();
        self.validate_actions_len(player_configs.len())?;
        for configs in &player_configs {
            Self::validate_player_configs(configs.len())?;
        }
        if let Some(seeds) = seeds {
            self.validate_actions_len(seeds.len())?;
            self.next_seeds = seeds;
        }
        self.player_configs = player_configs;

        (0..self.len())
            .map(|slot| {
                let (obs, info) = self.reset_slot(slot)?;
                write(slot, &obs, 0.0, false, false)?;
                Ok(info)
            })
            .collect()
    }

    pub fn step_into<F>(
        &mut self,
        actions: &[i64],
        active: Option<&[bool]>,
        write: F,
    ) -> Result<Vec<InfoDict>, AgentError>
    where
        F: Fn(usize, &Observation, f64, bool, bool) -> Result<(), AgentError>,
    {
        self.step_with_writer(actions, active, write)
    }

    pub(crate) fn step_with_writer<E: From<AgentError>>(
        &mut self,
        actions: &[i64],
        active: Option<&[bool]>,
        mut write: impl FnMut(usize, &Observation, f64, bool, bool) -> Result<(), E>,
    ) -> Result<Vec<InfoDict>, E> {
        self.validate_step(actions, active)?;
        actions
            .iter()
            .enumerate()
            .map(|(slot, &action)| {
                if active.is_some_and(|mask| !mask[slot]) {
                    return Ok(InfoDict::new());
                }
                let result = self.step_slot(slot, action)?;
                write(
                    slot,
                    &result.obs,
                    result.reward,
                    result.terminated,
                    result.truncated,
                )?;
                Ok(result.info)
            })
            .collect()
    }

    /// Settle each active gameplay call independently. Failed games are poisoned:
    /// callers must retain their inputs and replace them before stepping again.
    /// Only explicitly selected observation capacity failures may join gameplay
    /// failures. Reset, buffer integrity and malformed payload failures stay fatal.
    pub fn step_isolated_into<F>(
        &mut self,
        actions: &[i64],
        active: &[bool],
        recover_capacity: bool,
        write: F,
    ) -> Result<Vec<InfoDict>, AgentError>
    where
        F: Fn(usize, &Observation, f64, bool, bool) -> Result<(), ObservationWriteError>,
    {
        self.step_isolated_with_writer(actions, active, recover_capacity, write)
            .map_err(Into::into)
    }

    pub(crate) fn step_isolated_with_writer<F>(
        &mut self,
        actions: &[i64],
        active: &[bool],
        recover_capacity: bool,
        write: F,
    ) -> Result<Vec<InfoDict>, ObservationWriteError>
    where
        F: Fn(usize, &Observation, f64, bool, bool) -> Result<(), ObservationWriteError>,
    {
        self.validate_step(actions, Some(active))?;
        actions
            .iter()
            .enumerate()
            .map(|(index, &action)| {
                if !active[index] {
                    return Ok(InfoDict::new());
                }
                let frame = serde_json::to_string(&self.envs[index].semantic_decision_frame()?)
                    .map_err(|error| AgentError(error.to_string()))?;
                // A Rust unwind is contained only around gameplay of a disposable
                // stream. Buffer errors and invalid setup must never be tolerated.
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    self.step_slot(index, action)
                }));
                let out = match result {
                    Ok(Ok(out)) => out,
                    Ok(Err(error)) => {
                        return Ok(self.slots[index].gameplay_failure(
                            "AgentError",
                            error.to_string(),
                            frame,
                        ))
                    }
                    Err(payload) => {
                        let message = payload
                            .downcast_ref::<String>()
                            .cloned()
                            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                            .unwrap_or_else(|| "native gameplay panic".into());
                        return Ok(self.slots[index].gameplay_failure(
                            "NativePanic",
                            message,
                            frame,
                        ));
                    }
                };
                match write(index, &out.obs, out.reward, out.terminated, out.truncated) {
                    Ok(()) => Ok(out.info),
                    Err(ObservationWriteError::Encoding(
                        error @ ObservationEncodeError::Capacity { .. },
                    )) if recover_capacity && !out.terminated && !out.truncated => {
                        self.slots[index] = SlotState::Poisoned;
                        Ok(InfoDict::from([
                            ("game_error".into(), InfoValue::String(error.to_string())),
                            (
                                "game_error_class".into(),
                                InfoValue::String("ObservationCapacity".into()),
                            ),
                            (
                                "game_error_kind".into(),
                                InfoValue::String("resource_limit".into()),
                            ),
                            ("game_error_frame".into(), InfoValue::String(frame)),
                        ]))
                    }
                    Err(error) => Err(error),
                }
            })
            .collect()
    }

    fn validate_player_configs(config_len: usize) -> Result<(), AgentError> {
        if config_len != 2 {
            return Err(AgentError(format!(
                "expected exactly 2 player configs, got {config_len}"
            )));
        }
        Ok(())
    }

    fn validate_actions_len(&self, actions_len: usize) -> Result<(), AgentError> {
        if actions_len != self.envs.len() {
            return Err(AgentError(format!(
                "expected {} actions, got {actions_len}",
                self.envs.len()
            )));
        }
        Ok(())
    }

    fn validate_ready_for_step(&self) -> Result<(), AgentError> {
        if self.player_configs.len() != self.envs.len() || self.player_configs.is_empty() {
            return Err(AgentError(
                "vector env must be reset before step".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_step(&self, actions: &[i64], active: Option<&[bool]>) -> Result<(), AgentError> {
        self.validate_actions_len(actions.len())?;
        self.validate_ready_for_step()?;
        if let Some(mask) = active {
            self.validate_actions_len(mask.len())?;
        }
        for (slot, state) in self.slots.iter().enumerate() {
            if active.is_none_or(|mask| mask[slot])
                && matches!(
                    state,
                    SlotState::Terminal | SlotState::Poisoned | SlotState::Unset
                )
            {
                return Err(AgentError(format!(
                    "cannot step {state:?} slot {slot} before reset or replacement"
                )));
            }
        }
        Ok(())
    }

    fn step_slot(&mut self, slot: usize, action: i64) -> Result<StepResult, AgentError> {
        self.slots[slot] = SlotState::Playing;
        let env = &mut self.envs[slot];
        let (obs, reward, terminated, truncated, mut info) = env.step(action).map_err(|error| {
            AgentError(format!(
                "{}; native_failure={}",
                error.0,
                env.vector_failure_context(action)
            ))
        })?;
        if terminated || truncated {
            self.slots[slot] = SlotState::Terminal;
            info.insert(
                "skipped_decisions".into(),
                InfoValue::Int(env.skip_trivial_count() as i64),
            );
            if env.record_terminal {
                info.insert(
                    "record_terminal_digest".into(),
                    InfoValue::String(env.state_digest()?),
                );
            }
        }
        Ok(StepResult {
            obs,
            reward,
            terminated,
            truncated,
            info,
        })
    }

    fn reset_slot(&mut self, slot: usize) -> Result<(Observation, InfoDict), AgentError> {
        let env = &mut self.envs[slot];
        let next_seed = &mut self.next_seeds[slot];
        if self.slots[slot] == SlotState::Poisoned {
            env.replace_slot(env.replacement_root(*next_seed), Default::default());
        }
        let opening = Self::deal_fresh_game(
            &self.uniform_roster,
            env,
            next_seed,
            self.seed_stride,
            &self.player_configs[slot],
        )?;
        self.slots[slot] = if env.is_game_over() {
            SlotState::Terminal
        } else {
            SlotState::FreshOpening
        };
        Ok(opening)
    }

    fn deal_fresh_game(
        roster: &RootRoster,
        env: &mut Env,
        next_seed: &mut u64,
        seed_stride: u64,
        player_configs: &GameSetup,
    ) -> Result<(Observation, InfoDict), AgentError> {
        let seed = *next_seed;
        *next_seed = seed.saturating_add(seed_stride);
        env.set_seed(seed);
        let configs = if roster.is_empty() {
            player_configs.clone()
        } else {
            let (a, b) = roster.pair(seed);
            GameSetup {
                players: vec![roster.players[a].clone(), roster.players[b].clone()],
                decklists: player_configs.decklists,
                mana_floats: player_configs.mana_floats,
                mana_payment: player_configs.mana_payment,
            }
        };
        env.reset(configs)
    }
}

/// Separate domain from shuffling and policy RNG; rejection sampling avoids modulo bias.
fn uniform_pair(seed: u64, size: usize) -> (usize, usize) {
    assert!(size > 0);
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x656c656d70616972);
    (rng.gen_range(0..size), rng.gen_range(0..size))
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, collections::BTreeMap};

    use crate::{
        agent::{
            action::AgentError,
            env::Env,
            observation_encoder::ObservationEncodeError,
            player_policy::PlayerPolicy,
            vector_env::{SlotLoad, VectorEnv},
        },
        infra::profiler::InfoValue,
        state::player::{GameSetup, PlayerConfig},
    };

    fn load_fresh(vector: &mut VectorEnv, slot: usize, setup: Option<GameSetup>) {
        vector
            .load_slots_into(
                &[SlotLoad::Fresh(slot, setup)],
                |_, _| Ok(()),
                |_, _, _| Ok(()),
            )
            .unwrap();
    }

    #[test]
    fn deferred_steps_construct_and_encode_only_requested_games() {
        use crate::flow::setup::CONSTRUCTIONS;
        let load = |vector: &mut VectorEnv, request: SlotLoad| {
            vector
                .load_slots_into(
                    &[request],
                    |_, _| Ok(()),
                    |_, _, terminal| {
                        assert!(!terminal);
                        Ok(())
                    },
                )
                .unwrap()
                .pop()
                .unwrap()
                .0
        };
        for isolated in [false, true] {
            let mut vector = VectorEnv::new(1, u64::MAX, true, [PlayerPolicy::Agent; 2]);
            let before = CONSTRUCTIONS.get();
            let players = ["a", "b"]
                .map(|name| PlayerConfig::new(name, BTreeMap::from([("Mountain".into(), 12)])));
            vector.reset_all(players.to_vec()).unwrap();
            let root = vector
                .snapshot_slots(&[true])
                .unwrap()
                .pop()
                .flatten()
                .unwrap();
            let branch = || SlotLoad::Snapshot(0, root.clone(), Default::default());
            load(&mut vector, branch());
            assert_eq!(vector.next_fresh_seeds(), [u64::MAX]);
            let writes = std::cell::Cell::new(0);
            let observe = |obs: &crate::agent::observation::Observation, done| {
                writes.set(writes.get() + 1);
                assert_eq!(obs.action_space.actions.is_empty(), done);
            };
            for steps in 1..=64 {
                if isolated {
                    vector
                        .step_isolated_into(&[0], &[true], false, |_, obs, _, done, _| {
                            observe(obs, done);
                            Ok(())
                        })
                        .unwrap();
                } else {
                    vector
                        .step_into(&[0], None, |_, obs, _, done, _| {
                            observe(obs, done);
                            Ok(())
                        })
                        .unwrap();
                }
                assert_eq!(writes.get(), steps);
                if vector.current_agent_indices() == [-1] {
                    break;
                }
            }
            assert_eq!(vector.current_agent_indices(), [-1]);
            load(&mut vector, branch());
            vector
                .step_isolated_into(&[999], &[true], false, |_, _, _, _, _| {
                    panic!("a poisoned slot must not encode a replacement opening")
                })
                .unwrap();
            load(&mut vector, branch());
            assert_eq!(CONSTRUCTIONS.get() - before, 1);
            assert_eq!(load(&mut vector, SlotLoad::Fresh(0, None)).seed, u64::MAX);
            assert_eq!(CONSTRUCTIONS.get() - before, 2);
        }
    }

    #[test]
    fn decision_observations_keep_chooser_and_active_mask() {
        let mut vector = VectorEnv::new(2, 42, true, [PlayerPolicy::Agent; 2]);
        assert!(vector.decision_observations(&[true, false]).is_err());
        vector.reset_all(sample_player_configs()).unwrap();
        let before = vector.state_digests(None).unwrap();
        let observations = vector.decision_observations(&[true, false]).unwrap();
        assert_eq!(
            observations[0].as_ref().unwrap().agent.player_index,
            i32::try_from(vector.envs[0].current_agent_index().unwrap()).unwrap()
        );
        assert!(observations[1].is_none());
        assert_eq!(vector.state_digests(None).unwrap(), before);
        assert!(vector.decision_observations(&[true]).is_err());
    }

    #[test]
    fn state_digests_match_scalar_and_respect_mask() {
        let players = sample_player_configs();
        let mut vector = VectorEnv::new(2, 42, true, [PlayerPolicy::Agent; 2]);
        vector.reset_all(players.clone()).unwrap();
        let mut scalar = Env::new(42, true, false, false);
        scalar.reset(players).unwrap();
        let before = scalar.state_digest().unwrap();
        assert_eq!(
            vector.state_digests(Some(&[true, false])).unwrap(),
            vec![Some(before.clone()), None]
        );
        assert_eq!(vector.state_digests(None).unwrap()[0], Some(before));
        assert!(vector.state_digests(Some(&[true])).is_err());
    }

    #[test]
    fn fixed_seat_observations_validate_before_writing_and_preserve_control() {
        let mut vector = VectorEnv::new(2, 42, true, [PlayerPolicy::Agent; 2]);
        let mut writes = 0;
        assert!(vector
            .write_observations_for_players(&[Some(0), Some(1)], |_, _| {
                writes += 1;
                Ok(())
            })
            .is_err());
        vector
            .reset_all(vec![
                PlayerConfig::new("a", BTreeMap::from([("Mountain".into(), 20)])),
                PlayerConfig::new("b", BTreeMap::from([("Island".into(), 20)])),
            ])
            .unwrap();
        for _ in 0..2 {
            vector.envs[0].step(0).unwrap(); // Both seats keep.
        }
        for players in [vec![], vec![Some(0)], vec![Some(0), Some(2)]] {
            assert!(vector
                .write_observations_for_players(&players, |_, _| {
                    writes += 1;
                    Ok(())
                })
                .is_err());
        }
        assert_eq!(writes, 0);
        vector.envs[0].scenario_control_player(0, 1);
        assert_eq!(vector.envs[0].current_agent_index(), Some(1));
        // A pending seed belongs to the next reset, not to the game being read.
        vector.envs[1].set_seed(99);
        let before = vector.hand_snapshots(&[0, 1]).unwrap();
        let frames = vector.semantic_decision_frames_json(None).unwrap();
        for players in [[0, 1], [1, 0]] {
            let reads = vector
                .write_observations_for_players(&players.map(Some), |slot, obs| {
                    assert_eq!(
                        obs.to_json(),
                        vector.envs[slot]
                            .observation_for_player(players[slot])
                            .unwrap()
                            .to_json()
                    );
                    assert_eq!(
                        obs.action_space.actions.is_empty(),
                        vector.envs[slot].current_agent_index() != Some(players[slot])
                    );
                    writes += 1;
                    Ok(())
                })
                .unwrap();
            for (read, hand) in reads.iter().zip(vector.hand_snapshots(&players).unwrap()) {
                assert_eq!(read.player, players[read.slot]);
                assert_eq!(
                    (read.game_id, read.seed, read.revision),
                    (hand.game_id, hand.seed, hand.revision)
                );
            }
        }
        assert_eq!(writes, 4);
        assert_eq!(before, vector.hand_snapshots(&[0, 1]).unwrap());
        assert_eq!(frames, vector.semantic_decision_frames_json(None).unwrap());
    }

    #[test]
    fn hand_snapshots_keep_native_seats_when_another_player_chooses() {
        let mut vector = VectorEnv::new(1, 42, true, [PlayerPolicy::Agent; 2]);
        vector
            .reset_all(vec![
                PlayerConfig::new("a", BTreeMap::from([("Mountain".into(), 20)])),
                PlayerConfig::new("b", BTreeMap::from([("Island".into(), 20)])),
            ])
            .unwrap();
        let env = &mut vector.envs[0];
        for _ in 0..2 {
            env.step(0).unwrap(); // Both seats keep.
        }
        let before = [env.hand_snapshot(0).unwrap(), env.hand_snapshot(1).unwrap()];
        env.scenario_control_player(0, 1);
        assert_eq!(env.current_agent_index(), Some(1));
        for (player, previous) in before.iter().enumerate() {
            let scalar = vector.envs[0].hand_snapshot(player).unwrap();
            let batch = vector.hand_snapshots(&[player]).unwrap();
            assert_eq!(batch[0], scalar);
            assert_eq!(scalar.cards, previous.cards);
        }
        assert_eq!(before[0].cards, BTreeMap::from([("Mountain".into(), 7)]));
        assert_eq!(before[1].cards, BTreeMap::from([("Island".into(), 7)]));
    }

    #[test]
    fn hand_snapshots_are_complete_detached_native_seat_snapshots() {
        let players: Vec<_> = ["a", "b"]
            .into_iter()
            .map(|name| PlayerConfig::new(name, BTreeMap::from([("Mountain".into(), 100)])))
            .collect();
        let mut vector = VectorEnv::new(2, 42, true, [PlayerPolicy::Agent; 2]);
        assert!(vector.hand_snapshots(&[0, 1]).is_err());
        vector.reset_all(players).unwrap();
        assert!(vector.hand_snapshots(&[0]).is_err());
        assert!(vector.hand_snapshots(&[0, 2]).is_err());
        vector.envs[0].scenario_clear_hand(1).unwrap();
        vector.envs[1].scenario_clear_hand(0).unwrap();
        for _ in 0..70 {
            vector.envs[1]
                .scenario_force_card_in_hand(0, "Mountain")
                .unwrap();
        }
        let snapshots = vector.hand_snapshots(&[1, 0]).unwrap();
        for (slot, player) in [1, 0].into_iter().enumerate() {
            assert_eq!(
                snapshots[slot],
                vector.envs[slot].hand_snapshot(player).unwrap()
            );
        }
        assert!(snapshots[0].cards.is_empty());
        assert_eq!(
            snapshots[1].cards,
            BTreeMap::from([("Mountain".into(), 70)])
        );
        assert_eq!(snapshots[1].player, 0);
        vector.envs[1].scenario_clear_hand(0).unwrap();
        assert!(vector.hand_snapshots(&[1, 0]).unwrap()[1].cards.is_empty());
        assert_eq!(
            snapshots[1].cards,
            BTreeMap::from([("Mountain".into(), 70)])
        );
    }

    #[test]
    fn hand_snapshots_distinguish_same_seed_resets_reassignment_and_replacement() {
        let mut vector = VectorEnv::new(2, 42, true, [PlayerPolicy::Agent; 2]);
        vector.reset_all(sample_player_configs()).unwrap();
        let original = vector.hand_snapshots(&[0, 1]).unwrap();
        vector
            .reset_seeds_into(vec![42, 43], |_, _, _, _, _| Ok(()))
            .unwrap();
        let reset = vector.hand_snapshots(&[0, 1]).unwrap();
        for (before, after) in original.iter().zip(&reset) {
            assert_ne!(before.game_id, after.game_id);
            assert_eq!(before.seed, after.seed);
            assert_eq!(before.cards, after.cards);
        }
        load_fresh(&mut vector, 0, Some(sample_player_configs().into()));
        let reassigned = vector.hand_snapshots(&[0, 1]).unwrap();
        assert_ne!(reset[0].game_id, reassigned[0].game_id);
        assert_eq!(reassigned[0].seed, reset[0].seed + 2);
        assert_eq!(reset[1], reassigned[1]);
        load_fresh(&mut vector, 0, None);
        let replaced = vector.hand_snapshots(&[0, 1]).unwrap();
        assert_ne!(reassigned[0].game_id, replaced[0].game_id);
        assert_eq!(reassigned[1], replaced[1]);
    }

    #[test]
    fn exposure_capture_retains_old_seed_before_reset() {
        let players: Vec<_> = ["a", "b"]
            .into_iter()
            .map(|name| PlayerConfig::new(name, BTreeMap::from([("Mountain".into(), 20)])))
            .collect();
        let mut vector = VectorEnv::new(1, 245, true, [PlayerPolicy::Agent; 2]);
        vector.enable_exposure_capture().unwrap();
        vector.reset_all(players.clone()).unwrap();
        vector.step(&[0]).unwrap();
        vector.reset_all(players).unwrap();
        let retained = vector.take_exposure();
        assert_eq!(retained[0].len(), 1);
        let receipt: serde_json::Value = serde_json::from_str(&retained[0][0]).unwrap();
        assert_eq!(receipt["seed"], 245);
        assert_eq!(receipt["trace"]["actions"], serde_json::json!([0]));
        assert_eq!(
            receipt["cards"]["capture"]["exposure"]["choices"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        assert_ne!(vector.current_game_seeds()[0], 245);
        assert!(vector.take_exposure()[0].is_empty());
    }

    #[test]
    fn first_legal_policy_prefix_keeps_vector_execution_identity() {
        use crate::agent::observation_encoder::{encode, ActionOverflow, ObservationEncoderConfig};
        let players: Vec<_> = ["a", "b"]
            .into_iter()
            .map(|name| PlayerConfig::new(name, BTreeMap::from([("Mountain".into(), 100)])))
            .collect();
        let mut vector = VectorEnv::new(1, 213, true, [PlayerPolicy::Agent; 2]);
        vector.reset_all(players).unwrap();
        for _ in 0..2 {
            vector.envs[0].step(0).unwrap(); // Both seats keep.
        }
        vector.envs[0].scenario_clear_hand(0).unwrap();
        for _ in 0..70 {
            vector.envs[0]
                .scenario_force_card_in_hand(0, "Mountain")
                .unwrap();
        }
        let observation = vector.envs[0].scenario_refresh().unwrap();
        assert_eq!(observation.action_space.actions.len(), 71);
        let config = ObservationEncoderConfig {
            max_actions: 64,
            max_cards_per_player: 128,
            action_overflow: ActionOverflow::FirstLegal,
            ..ObservationEncoderConfig::default()
        };
        let encoded = encode(&observation, &config).unwrap();
        assert_eq!(encoded.actions_valid, vec![1.0; 64]);
        assert_eq!(
            vector.current_action_summaries(),
            vec![(71, "Priority".into())]
        );
        let mut scalar = vector.envs[0].fork().unwrap();
        let expected = scalar.step(63).unwrap().0;
        vector
            .step_into(&[63], None, |_, obs, _, _, _| {
                assert_eq!(obs.to_json(), expected.to_json());
                assert_eq!(
                    encode(obs, &config).unwrap().actions,
                    encode(&expected, &config).unwrap().actions
                );
                Ok(())
            })
            .unwrap();
        assert_eq!(
            vector.envs[0].selected_witness().unwrap(),
            scalar.selected_witness().unwrap()
        );
    }

    #[test]
    fn training_decks_are_admitted_distinct_replayable_and_replaceable() {
        let deck = |name: &str| {
            PlayerConfig::new(name, BTreeMap::from([(name.into(), 12)])).submit_to_world()
        };
        let decks: Vec<_> = ["Plains", "Mountain", "Forest"].map(deck).into();
        let mut vector = VectorEnv::new(2, 17402, true, [PlayerPolicy::Agent; 2]);
        assert!(vector
            .set_training_decks(vec![decks[0].clone(); 2], true)
            .is_err());
        let mut bad = decks.clone();
        bad[0].deck.main = BTreeMap::from([("Black Lotus".into(), 12)]);
        assert!(vector.set_training_decks(bad, true).is_err());
        vector.set_training_decks(decks.clone(), true).unwrap();
        vector.reset_all(decks[..2].to_vec()).unwrap();
        let seeds = vector.current_game_seeds();
        let actual = vector.current_player_configs();
        for (index, players) in actual.iter().enumerate() {
            assert_ne!(players[0].deck.main, players[1].deck.main);
            let mut scalar = Env::new(seeds[index], true, false, false);
            scalar.reset(players.clone()).unwrap();
            assert_eq!(
                scalar.state_digest().unwrap(),
                vector.envs[index].state_digest().unwrap()
            );
        }
        vector.reset_all(decks[..2].to_vec()).unwrap();
        assert_ne!(seeds, vector.current_game_seeds());
        vector
            .reset_seeds_into(seeds.clone(), |_, _, _, _, _| Ok(()))
            .unwrap();
        assert_eq!(actual, vector.current_player_configs());

        // A later root may use a deck nobody registered before training began.
        // An unadmitted replacement fails before any game and keeps prior decks.
        let mut generated = deck("Island");
        generated.deck.main.insert("Cat-Gator".into(), 1);
        let mut invalid = generated.clone();
        invalid.deck.main.insert("Ally".into(), 1);
        assert!(vector
            .set_training_decks(vec![decks[0].clone(), invalid], true)
            .is_err());
        vector
            .reset_seeds_into(seeds.clone(), |_, _, _, _, _| Ok(()))
            .unwrap();
        assert_eq!(actual, vector.current_player_configs());
        let replacement = vec![generated, deck("Swamp")];
        vector
            .set_training_decks(replacement.clone(), true)
            .unwrap();
        assert_eq!(actual, vector.current_player_configs());
        vector
            .reset_seeds_into(seeds, |_, _, _, _, _| Ok(()))
            .unwrap();
        for players in vector.current_player_configs() {
            assert!(players.iter().all(|player| replacement.contains(player)));
        }
        assert!(vector.current_matchup_indices().is_empty());
    }

    #[test]
    fn independent_training_decks_preserve_mirrors_entries_and_seed_recovery() {
        let mut first = PlayerConfig::new("event-a", BTreeMap::from([("Plains".into(), 12)]));
        first.submitted = true;
        let mut second = first.clone();
        second.name = "event-b".into();
        let decks = vec![first.clone(), second];
        let mut vector = VectorEnv::new(2, 21301, true, [PlayerPolicy::Agent; 2]);
        vector.set_training_decks(decks.clone(), false).unwrap();
        vector.reset_all(vec![first.clone(); 2]).unwrap();
        let seeds = vector.current_game_seeds();
        let actual = vector.current_player_configs();
        for (index, players) in actual.iter().enumerate() {
            let (a, b) = super::uniform_pair(seeds[index], decks.len());
            assert_eq!(players, &vec![decks[a].clone(), decks[b].clone()]);
            // Identical content in different events remains in the population.
            assert_eq!(players[0].deck, players[1].deck);
        }
        vector.reset_all(vec![first.clone(); 2]).unwrap();
        vector
            .reset_seeds_into(seeds, |_, _, _, _, _| Ok(()))
            .unwrap();
        assert_eq!(actual, vector.current_player_configs());
        assert!(vector.current_matchup_indices().is_empty());
        vector
            .set_training_decks(vec![first.clone()], false)
            .unwrap();
        vector.reset_all(vec![first.clone(); 2]).unwrap();
        assert!(vector
            .current_player_configs()
            .iter()
            .all(|pair| pair == &vec![first.clone(); 2]));
    }

    fn sample_player_configs() -> Vec<PlayerConfig> {
        vec![
            PlayerConfig::new(
                "Hero",
                BTreeMap::from([
                    ("Mountain".to_string(), 12usize),
                    ("Forest".to_string(), 12usize),
                    ("Llanowar Elves".to_string(), 18usize),
                    ("Gray Ogre".to_string(), 18usize),
                ]),
            ),
            PlayerConfig::new(
                "Villain",
                BTreeMap::from([
                    ("Mountain".to_string(), 12usize),
                    ("Forest".to_string(), 12usize),
                    ("Llanowar Elves".to_string(), 18usize),
                    ("Gray Ogre".to_string(), 18usize),
                ]),
            ),
        ]
    }

    fn elemental_roster() -> Vec<PlayerConfig> {
        let setup = crate::cards::setups::authored_setup("optional_payments").unwrap();
        setup
            .decks
            .iter()
            .map(|d| setup.player_config(&d.key, &d.key).unwrap())
            .collect()
    }

    #[test]
    fn uniform_roots_cover_all_pairs_and_seats_with_exact_setups() {
        use std::collections::BTreeSet;
        let roster = elemental_roster();
        let mut reached = [BTreeSet::new(), BTreeSet::new()];
        let mut vector = VectorEnv::new(2, 174, true, [PlayerPolicy::Agent; 2]);
        let mut bad = roster.clone();
        bad[0].deck.sideboard.clear();
        assert!(vector.set_uniform_roster(bad).is_err());
        assert!(vector.set_uniform_roster(roster[..9].to_vec()).is_err());
        vector.set_uniform_roster(roster.clone()).unwrap();
        vector.reset_all(roster[..2].to_vec()).unwrap();
        for seed in 0..2000_u64 {
            let seeds = vec![seed * 2, seed * 2 + 1];
            let observed = RefCell::new(vec![String::new(); 2]);
            vector
                .reset_seeds_into(seeds.clone(), |index, obs, _, _, _| {
                    observed.borrow_mut()[index] = format!("{obs:?}");
                    Ok(())
                })
                .unwrap();
            for (seat, pair) in vector.current_matchup_indices().into_iter().enumerate() {
                reached[seat].insert(pair);
                if seed < 100 {
                    let mut scalar = Env::new(seeds[seat], true, false, false);
                    let obs = scalar
                        .reset(vec![roster[pair.0].clone(), roster[pair.1].clone()])
                        .unwrap()
                        .0;
                    assert_eq!(format!("{obs:?}"), observed.borrow()[seat]);
                }
            }
        }
        assert_eq!(reached[0].len(), 100);
        assert_eq!(reached[1].len(), 100);
        assert!(vector.set_uniform_roster(roster.clone()).is_err());
        // Real terminal resets must select the next setup, not just report its index.
        let mut scalar: Vec<_> = vector
            .current_game_seeds()
            .iter()
            .zip(vector.current_matchup_indices())
            .map(|(seed, (a, b))| {
                let mut env = Env::new(*seed, true, false, false);
                env.reset(vec![roster[a].clone(), roster[b].clone()])
                    .unwrap();
                env
            })
            .collect();
        let mut terminals = 0;
        let mut changed = 0;
        for _ in 0..1200 {
            let previous = vector.current_matchup_indices();
            let results = vector.step(&[0, 0]).unwrap();
            for (slot, result) in results.iter().enumerate() {
                if result.terminated || result.truncated {
                    load_fresh(&mut vector, slot, None);
                }
            }
            let pairs = vector.current_matchup_indices();
            for index in 0..2 {
                let (obs, _, done, trunc, _) = scalar[index].step(0).unwrap();
                assert_eq!(format!("{:?}", results[index].obs), format!("{obs:?}"));
                if done || trunc {
                    terminals += 1;
                    changed += usize::from(previous[index] != pairs[index]);
                    scalar[index].set_seed(vector.current_game_seeds()[index]);
                    let (a, b) = pairs[index];
                    scalar[index]
                        .reset(vec![roster[a].clone(), roster[b].clone()])
                        .unwrap();
                }
            }
        }
        assert!(terminals > 0 && changed > 0);
    }

    #[test]
    fn fresh_load_preserves_recording_sinks_after_rejected_setup() {
        let mut vector = VectorEnv::new(1, 23, true, [PlayerPolicy::Agent; 2]);
        vector.enable_statistics().unwrap();
        vector.enable_exposure_capture().unwrap();
        vector.reset_all(sample_player_configs()).unwrap();
        let generation = vector.current_game_generations();
        assert!(vector
            .load_slots_into(
                &[SlotLoad::Fresh(0, Some(sample_player_configs().into()))],
                |_, _| Err(AgentError("injected admission".into()).into()),
                |_, _, _| Ok(())
            )
            .is_err());
        assert_eq!(vector.current_game_generations(), generation);
        assert!(vector.take_exposure()[0].is_empty());
        assert!(vector.take_statistics().is_empty());
        load_fresh(&mut vector, 0, Some(sample_player_configs().into()));
        assert_eq!(vector.take_exposure()[0].len(), 1);
        let summaries = vector.take_statistics();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].generation, generation[0]);
        assert!(!summaries[0].terminal);
        assert!(vector.statistics()[0].is_some());
        vector.step(&[0]).unwrap();
        vector.reset_all(sample_player_configs()).unwrap();
        let receipt: serde_json::Value =
            serde_json::from_str(&vector.take_exposure()[0][0]).unwrap();
        assert_eq!(receipt["trace"]["actions"], serde_json::json!([0]));
    }

    #[test]
    fn fresh_setups_commit_only_after_admission_and_persist_on_the_slot() {
        let mut vector = VectorEnv::new(3, 23, true, [PlayerPolicy::Agent; 2]);
        vector.reset_all(sample_player_configs()).unwrap();
        vector.step(&[0; 3]).unwrap();
        let original = vector.player_configs.clone();
        let seeds = vector.current_game_seeds();
        let next = vector.next_seeds.clone();
        let mut swapped = sample_player_configs();
        swapped.reverse();
        let replacement = SlotLoad::Fresh(0, Some(swapped.clone().into()));
        let mut invalid = sample_player_configs();
        invalid[0].deck.main.insert("Not A Real Card".into(), 1);
        for bad in [invalid, vec![]] {
            assert!(vector
                .load_slots_into(
                    &[replacement.clone(), SlotLoad::Fresh(2, Some(bad.into()))],
                    |_, _| Ok(()),
                    |_, _, _| panic!("failed admission wrote a row")
                )
                .is_err());
        }
        assert_eq!(vector.player_configs, original);
        assert_eq!(vector.current_game_seeds(), seeds);
        assert_eq!(vector.next_seeds, next);
        load_fresh(&mut vector, 0, Some(swapped.clone().into()));
        assert_eq!(vector.current_game_seeds(), [26, 24, 25]);
        load_fresh(&mut vector, 0, None);
        assert_eq!(vector.current_game_seeds(), [29, 24, 25]);
        assert_eq!(vector.current_player_configs()[0], swapped);
    }

    #[test]
    fn reset_all_and_step_batch() {
        let mut env = VectorEnv::new(3, 7, true, [PlayerPolicy::Agent; 2]);
        let reset = env
            .reset_all(sample_player_configs())
            .expect("reset_all should succeed");
        assert_eq!(reset.len(), 3);

        let stepped = env.step(&[0, 0, 0]).expect("step should succeed");
        assert_eq!(stepped.len(), 3);
    }

    #[test]
    fn isolated_encoding_recovers_only_opted_in_capacity_errors() {
        for recover in [false, true] {
            for kind in 0..3 {
                let mut vector = VectorEnv::new(3, 7, true, [PlayerPolicy::Agent; 2]);
                vector.reset_all(sample_player_configs()).unwrap();
                let mut control = VectorEnv::new(3, 7, true, [PlayerPolicy::Agent; 2]);
                control.reset_all(sample_player_configs()).unwrap();
                let result =
                    vector.step_isolated_into(&[0; 3], &[true; 3], recover, |index, _, _, _, _| {
                        if index != 1 {
                            return Ok(());
                        }
                        Err(match kind {
                            0 => ObservationEncodeError::Capacity {
                                field: "agent_permanents",
                                capacity: 40,
                                actual: 41,
                            }
                            .into(),
                            1 => ObservationEncodeError::InvalidShared("missing source").into(),
                            _ => AgentError("invalid buffer pointer".into()).into(),
                        })
                    });
                if !recover || kind != 0 {
                    assert!(result.is_err());
                    continue;
                }
                let info = result.unwrap();
                assert!(
                    matches!(info[1].get("game_error_kind"), Some(InfoValue::String(kind)) if kind == "resource_limit")
                );
                assert!(info[1].contains_key("game_error_frame"));
                assert!(!info[0].contains_key("game_error"));
                assert!(!info[2].contains_key("game_error"));
                control.step(&[0; 3]).unwrap();
                load_fresh(&mut vector, 1, None);
                assert_eq!(vector.current_game_seeds(), vec![7, 11, 9]);
                for index in [0, 2] {
                    assert_eq!(
                        vector.envs[index].state_digest().unwrap(),
                        control.envs[index].state_digest().unwrap()
                    );
                }
            }
        }
    }

    #[test]
    fn isolated_failure_preserves_successful_streams_and_seed_stride() {
        let mut vector = VectorEnv::new(3, 7, true, [PlayerPolicy::Agent; 2]);
        vector.reset_all(sample_player_configs()).unwrap();
        let mut control = VectorEnv::new(3, 7, true, [PlayerPolicy::Agent; 2]);
        control.reset_all(sample_player_configs()).unwrap();
        let rows = RefCell::new(Vec::new());
        let outcomes = vector
            .step_isolated_into(&[0, -1, 0], &[true; 3], false, |index, _, _, _, _| {
                rows.borrow_mut().push(index);
                Ok(())
            })
            .unwrap();
        assert_eq!(*rows.borrow(), vec![0, 2]);
        assert!(!outcomes[0].contains_key("game_error"));
        assert!(outcomes[1].contains_key("game_error_frame"));
        assert!(!outcomes[2].contains_key("game_error"));
        control
            .step_into(&[0, 0, 0], Some(&[true, false, true]), |_, _, _, _, _| {
                Ok(())
            })
            .unwrap();
        for index in [0, 2] {
            assert_eq!(
                vector.envs[index].state_digest().unwrap(),
                control.envs[index].state_digest().unwrap()
            );
        }
        load_fresh(&mut vector, 1, None);
        assert_eq!(vector.current_game_seeds(), vec![7, 11, 9]);
        for index in [0, 2] {
            assert_eq!(
                vector.envs[index].state_digest().unwrap(),
                control.envs[index].state_digest().unwrap()
            );
        }
    }

    #[test]
    fn reset_skips_to_hero_turn_with_random_policy() {
        let mut env = VectorEnv::new(1, 11, true, [PlayerPolicy::Agent, PlayerPolicy::Random]);
        let reset = env
            .reset_all(sample_player_configs())
            .expect("reset_all should succeed");
        assert_eq!(reset[0].0.agent.player_index, 0);
    }

    #[test]
    fn terminal_step_parks_the_finished_game() {
        let mut env = VectorEnv::new(1, 23, true, [PlayerPolicy::Agent, PlayerPolicy::Random]);
        env.reset_all(sample_player_configs())
            .expect("reset_all should succeed");

        let mut found_terminal = false;
        for _ in 0..4000 {
            let results = env.step(&[0]).expect("step should succeed");
            let result = &results[0];
            assert_eq!(result.obs.agent.player_index, 0);
            if result.terminated {
                found_terminal = true;
                assert!(result.obs.game_over);
                assert_eq!(env.current_agent_indices(), [-1]);
                assert_eq!(env.next_fresh_seeds(), [24]);
                assert!(env.step(&[0]).unwrap_err().0.contains("Terminal slot 0"));
                break;
            }
        }

        assert!(found_terminal, "expected at least one terminal transition");
    }

    #[test]
    fn step_into_and_reset_all_into() {
        let mut env = VectorEnv::new(3, 31, true, [PlayerPolicy::Agent, PlayerPolicy::Random]);

        let reset_infos = env
            .reset_all_into(
                sample_player_configs(),
                |_, obs, reward, terminated, truncated| {
                    assert_eq!(obs.agent.player_index, 0);
                    assert_eq!(reward, 0.0);
                    assert!(!terminated);
                    assert!(!truncated);
                    Ok(())
                },
            )
            .expect("reset_all_into should succeed");
        assert_eq!(reset_infos.len(), 3);

        let step_infos = env
            .step_into(&[0, 0, 0], None, |_, _, _, _, _| Ok(()))
            .expect("step_into should succeed");
        assert_eq!(step_infos.len(), 3);
    }

    #[test]
    fn masked_step_preserves_paused_stream_and_ignores_its_action() {
        let mut paused = VectorEnv::new(2, 99, true, [PlayerPolicy::Agent; 2]);
        let mut control = VectorEnv::new(2, 99, true, [PlayerPolicy::Agent; 2]);
        paused.reset_all(sample_player_configs()).unwrap();
        control.reset_all(sample_player_configs()).unwrap();
        paused
            .step_into(&[-999, 0], Some(&[false, true]), |index, _, _, _, _| {
                assert_eq!(index, 1);
                Ok(())
            })
            .unwrap();
        let actual = paused.step(&[0, 0]).unwrap();
        let expected = control.step(&[0, 0]).unwrap();
        assert_eq!(
            format!("{:?}", actual[0].obs),
            format!("{:?}", expected[0].obs)
        );
        assert!(paused
            .step_into(&[0, 0], Some(&[true]), |_, _, _, _, _| Ok(()))
            .is_err());
    }

    #[test]
    fn restored_game_seeds_preserve_fresh_seed_stride() {
        let mut vector = VectorEnv::new(2, 1, true, [PlayerPolicy::Agent; 2]);
        vector.reset_all(sample_player_configs()).unwrap();
        let mut seeds = [33_u64, 71_u64];
        vector
            .reset_seeds_into(seeds.to_vec(), |_, _, _, _, _| Ok(()))
            .unwrap();
        let mut scalar = seeds.map(|seed| Env::new(seed, true, false, false));
        for env in &mut scalar {
            env.reset(sample_player_configs()).unwrap();
        }
        let mut terminals = 0;
        for _ in 0..1000 {
            let actual = vector.step(&[0, 0]).unwrap();
            for index in 0..2 {
                let (obs, _, done, truncated, _) = scalar[index].step(0).unwrap();
                assert_eq!(format!("{:?}", actual[index].obs), format!("{obs:?}"));
                if done || truncated {
                    terminals += 1;
                    seeds[index] += 2;
                    scalar[index].set_seed(seeds[index]);
                    scalar[index].reset(sample_player_configs()).unwrap();
                    load_fresh(&mut vector, index, None);
                }
            }
            assert_eq!(vector.current_game_seeds(), seeds);
        }
        assert!(terminals > 0);
    }

    #[test]
    fn vector_error_retains_exact_scalar_replay_inputs() {
        let configs = sample_player_configs();
        let mut vector = VectorEnv::new(1, 174, true, [PlayerPolicy::Agent; 2]);
        vector.reset_all(configs.clone()).unwrap();
        vector.step(&[0]).unwrap();
        let error = vector.step(&[999_999]).unwrap_err();
        let (_, encoded) = error
            .0
            .split_once("; native_failure=")
            .expect("private failure context");
        let context: serde_json::Value = serde_json::from_str(encoded).unwrap();
        assert_eq!(
            context["trace"]["player_configs"],
            serde_json::to_value(&configs).unwrap()
        );
        assert_eq!(context["failed_action_index"], 999_999);
        let mut scalar = Env::new(context["seed"].as_u64().unwrap(), true, false, false);
        scalar.reset(configs).unwrap();
        for action in context["trace"]["actions"].as_array().unwrap() {
            scalar.step(action.as_i64().unwrap()).unwrap();
        }
        let restored: serde_json::Value =
            serde_json::from_str(&scalar.vector_failure_context(999_999)).unwrap();
        assert_eq!(restored["post_error_state"], context["post_error_state"]);
        assert_eq!(restored["post_error_offers"], context["post_error_offers"]);
    }

    #[test]
    fn step_into_errors_are_returned_in_index_order() {
        let mut env = VectorEnv::new(3, 99, true, [PlayerPolicy::Agent; 2]);
        env.reset_all_into(sample_player_configs(), |_, _, _, _, _| Ok(()))
            .expect("reset_all_into should succeed");

        let err = env
            .step_into(&[0, 0, 0], None, |env_index, _, _, _, _| {
                if env_index >= 1 {
                    return Err(AgentError(format!("env error {env_index}")));
                }
                Ok(())
            })
            .expect_err("step_into should fail");
        assert_eq!(err.0, "env error 1");
    }

    #[test]
    fn masked_mixed_setups_match_scalar_games() {
        let configs: Vec<_> = (0..3)
            .map(|index| {
                let mut players = sample_player_configs();
                players[0].name = format!("Hero {index}");
                players[0].deck.main.insert("Mountain".into(), 12 + index);
                players[1]
                    .deck
                    .sideboard
                    .insert("Gray Ogre".into(), index + 1);
                players
            })
            .collect();
        let mut seeds = [33_u64, 34, 35];
        let mut scalar = seeds.map(|seed| Env::new(seed, true, false, false));
        for (env, setup) in scalar.iter_mut().zip(&configs) {
            env.reset(setup.clone()).unwrap();
        }
        let mut vector = VectorEnv::new(3, 33, true, [PlayerPolicy::Agent; 2]);
        vector
            .reset_each_into(configs.clone(), |_, _, _, _, _| Ok(()))
            .unwrap();
        let mut terminals = [0; 3];
        for step in 0..1000 {
            let active = [step % 3 != 0, true, step % 2 == 0];
            let actions = active.map(|enabled| if enabled { 0 } else { -999 });
            let observed = RefCell::new(Vec::new());
            vector
                .step_into(
                    &actions,
                    Some(&active),
                    |index, obs, reward, done, trunc| {
                        observed.borrow_mut().push((
                            index,
                            format!("{obs:?}"),
                            reward,
                            done,
                            trunc,
                        ));
                        Ok(())
                    },
                )
                .unwrap();
            assert_eq!(
                observed.borrow().len(),
                active.iter().filter(|x| **x).count()
            );
            for (index, actual, reward, done, trunc) in observed.into_inner() {
                let (obs, expected_reward, expected_done, expected_trunc, _) =
                    scalar[index].step(0).unwrap();
                assert_eq!(
                    (reward, done, trunc),
                    (expected_reward, expected_done, expected_trunc)
                );
                assert_eq!(actual, format!("{obs:?}"));
                if done || trunc {
                    terminals[index] += 1;
                    seeds[index] += 3;
                    scalar[index].set_seed(seeds[index]);
                    scalar[index].reset(configs[index].clone()).unwrap();
                    load_fresh(&mut vector, index, None);
                }
            }
            assert_eq!(vector.current_game_seeds(), seeds);
        }
        assert!(terminals.iter().all(|count| *count > 0));
        assert_eq!(
            vector.player_configs,
            configs.into_iter().map(GameSetup::from).collect::<Vec<_>>()
        );
    }

    #[test]
    fn shards_preserve_global_seeds_through_fresh_load_and_restore() {
        let mut whole = VectorEnv::new(3, 19, true, [PlayerPolicy::Agent; 2]);
        let mut left = VectorEnv::new(2, 19, true, [PlayerPolicy::Agent; 2])
            .with_seed_stride(3)
            .unwrap();
        let mut right = VectorEnv::new(1, 21, true, [PlayerPolicy::Agent; 2])
            .with_seed_stride(3)
            .unwrap();
        for vector in [&mut whole, &mut left, &mut right] {
            vector.reset_all(sample_player_configs()).unwrap();
        }
        let mut terminals = 0;
        for step in 0..1000 {
            if step == 333 {
                for (vector, seeds) in [
                    (&mut whole, vec![45, 109, 1011]),
                    (&mut left, vec![45, 109]),
                    (&mut right, vec![1011]),
                ] {
                    vector
                        .reset_seeds_into(seeds, |_, _, _, _, _| Ok(()))
                        .unwrap();
                }
            }
            let expected = whole.step(&[0, 0, 0]).unwrap();
            let mut actual = left.step(&[0, 0]).unwrap();
            actual.extend(right.step(&[0]).unwrap());
            for (expected, actual) in expected.iter().zip(actual.iter()) {
                assert_eq!(format!("{:?}", expected.obs), format!("{:?}", actual.obs));
                assert_eq!(expected.reward, actual.reward);
                assert_eq!(expected.terminated, actual.terminated);
                assert_eq!(expected.truncated, actual.truncated);
                terminals += usize::from(actual.terminated);
            }
            for (slot, result) in expected.iter().enumerate() {
                if result.terminated || result.truncated {
                    load_fresh(&mut whole, slot, None);
                    if slot < 2 {
                        load_fresh(&mut left, slot, None);
                    } else {
                        load_fresh(&mut right, 0, None);
                    }
                }
            }
            let mut seeds = left.current_game_seeds();
            seeds.extend(right.current_game_seeds());
            assert_eq!(whole.current_game_seeds(), seeds);
        }
        assert!(terminals > 0);
        assert!(VectorEnv::new(2, 0, true, [PlayerPolicy::Agent; 2])
            .with_seed_stride(1)
            .is_err());
        assert!(left.with_seed_stride(3).is_err());
    }
}
