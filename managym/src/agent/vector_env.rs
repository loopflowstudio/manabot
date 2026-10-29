//! Batched games with native reset-seed ownership and admitted per-game roster sampling.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{
    agent::{action::AgentError, env::Env, observation::Observation, opponent::OpponentPolicy},
    infra::profiler::{InfoDict, InfoValue},
    state::player::PlayerConfig,
};

const HERO_PLAYER_INDEX: i32 = 0;

#[derive(Debug)]
pub struct StepResult {
    pub obs: Observation,
    pub reward: f64,
    pub terminated: bool,
    pub truncated: bool,
    pub info: InfoDict,
}

/// One root sampler: the frozen Elemental roster, or replaceable distinct training decks.
#[derive(Debug, Default)]
struct RootRoster {
    players: Vec<PlayerConfig>,
    distinct: bool,
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

#[derive(Debug)]
pub struct VectorEnv {
    envs: Vec<Env>,
    player_configs: Vec<Vec<PlayerConfig>>,
    opponent_policy: OpponentPolicy,
    next_seeds: Vec<u64>,
    seed_stride: u64,
    uniform_roster: RootRoster,
    skip_trivial: bool,
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
        self.opponent_policy = OpponentPolicy::None;
        Ok(())
    }

    pub fn new(
        num_envs: usize,
        seed: u64,
        skip_trivial: bool,
        opponent_policy: OpponentPolicy,
    ) -> Self {
        let envs = (0..num_envs)
            .map(|idx| Env::new(seed.saturating_add(idx as u64), skip_trivial, false, false))
            .collect();
        let next_seeds = (0..num_envs)
            .map(|idx| seed.saturating_add(idx as u64))
            .collect();

        Self {
            envs,
            player_configs: Vec::new(),
            opponent_policy,
            next_seeds,
            seed_stride: num_envs.max(1) as u64,
            uniform_roster: RootRoster::default(),
            skip_trivial,
        }
    }

    /// Configure only before allocation of roots; native authored setups are the authority.
    pub fn set_uniform_roster(&mut self, roster: Vec<PlayerConfig>) -> Result<(), AgentError> {
        if !self.player_configs.is_empty() {
            return Err(AgentError("uniform roster must be set before reset".into()));
        }
        let pack = crate::semantic::SemanticPack::current("optional_payments")
            .map_err(|e| AgentError(e.to_string()))?;
        let expected: Vec<PlayerConfig> = pack
            .decks
            .iter()
            .map(|deck| pack.player_config(&deck.key, &deck.key).unwrap())
            .collect();
        if roster.len() != expected.len()
            || expected.iter().any(|e| {
                roster
                    .iter()
                    .filter(|r| {
                        r.decklist == e.decklist
                            && r.sideboard == e.sideboard
                            && r.content_pack.as_deref() == Some("elemental-world-v2")
                    })
                    .count()
                    != 1
            })
        {
            return Err(AgentError(
                "uniform roster requires every exact Elemental setup once".into(),
            ));
        }
        self.uniform_roster = RootRoster {
            players: roster,
            distinct: false,
        };
        Ok(())
    }

    /// Replace the training decks sampled at subsequent roots.
    ///
    /// Every deck passes the content pool's native admission before the
    /// replacement takes effect; on any error the previous decks remain. Games
    /// already underway keep their decks. Membership never grants permission:
    /// game creation admits each submitted deck again.
    pub fn set_training_decks(&mut self, decks: Vec<PlayerConfig>) -> Result<(), AgentError> {
        if !self.uniform_roster.is_empty() && !self.uniform_roster.distinct {
            return Err(AgentError(
                "the Elemental roster cannot be replaced by training decks".into(),
            ));
        }
        if decks.len() < 2 {
            return Err(AgentError(
                "uniform-distinct-v1 requires at least two training decks".into(),
            ));
        }
        let pack = decks[0]
            .content_pack
            .as_deref()
            .ok_or_else(|| AgentError("training decks require an explicit content pool".into()))?;
        let mut distinct = std::collections::BTreeSet::new();
        for player in &decks {
            if player.content_pack.as_deref() != Some(pack) {
                return Err(AgentError(
                    "training decks must share one content pool".into(),
                ));
            }
            let issues = crate::semantic::deck_admission::admit_deck(
                pack,
                &player.decklist,
                &player.sideboard,
                crate::semantic::deck_admission::DEVELOPMENT_RULESET,
            )
            .map_err(|error| AgentError(error.to_string()))?;
            if !issues.is_empty() {
                return Err(AgentError(format!(
                    "training deck {}: {}",
                    player.name,
                    issues
                        .iter()
                        .map(|issue| issue.message.as_str())
                        .collect::<Vec<_>>()
                        .join("; ")
                )));
            }
            if !distinct.insert((&player.decklist, &player.sideboard)) {
                return Err(AgentError(
                    "training decks must be distinct normalized lists".into(),
                ));
            }
        }
        self.uniform_roster = RootRoster {
            players: decks,
            distinct: true,
        };
        Ok(())
    }

    /// The setups each game was actually reset with, in seat order.
    pub fn current_player_configs(&self) -> Vec<Vec<PlayerConfig>> {
        self.envs
            .iter()
            .map(|env| env.player_configs().to_vec())
            .collect()
    }

    /// Current ordered indices into the fixed Elemental roster; empty otherwise.
    /// Replaceable training decks report actual setups via `current_player_configs`.
    pub fn current_matchup_indices(&self) -> Vec<(usize, usize)> {
        if self.uniform_roster.is_empty() || self.uniform_roster.distinct {
            return Vec::new();
        }
        self.envs
            .iter()
            .map(|env| self.uniform_roster.pair(env.seed()))
            .collect()
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

    /// Exact current-game seeds; auto-reset succession remains native authority.
    pub fn current_game_seeds(&self) -> Vec<u64> {
        self.envs.iter().map(Env::seed).collect()
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
        self.validate_actions_len(seeds.len())?;
        self.validate_ready_for_step()?;
        self.next_seeds = seeds;
        self.reset_each_into(self.player_configs.clone(), write)
    }

    /// Per-env `skip_trivial` collapse counters for the current games.
    /// Each counter resets when its env's game resets.
    pub fn skip_trivial_counts(&self) -> Vec<usize> {
        self.envs.iter().map(Env::skip_trivial_count).collect()
    }

    /// Per-env index of the player holding the current decision (-1 if no
    /// decision is pending). With `opponent_policy = none` both seats
    /// surface, so callers routing decisions to per-seat policies need this.
    pub fn current_agent_indices(&self) -> Vec<i64> {
        self.envs
            .iter()
            .map(|env| env.current_agent_index().map_or(-1, |index| index as i64))
            .collect()
    }

    pub fn reset_all(
        &mut self,
        player_configs: Vec<PlayerConfig>,
    ) -> Result<Vec<(Observation, InfoDict)>, AgentError> {
        Self::validate_player_configs(player_configs.len())?;
        self.player_configs = vec![player_configs; self.envs.len()];
        let player_configs = &self.player_configs;
        let opponent_policy = self.opponent_policy;
        let seed_stride = self.seed_stride;
        let roster = &self.uniform_roster;

        let mut results = Vec::with_capacity(self.envs.len());
        for (index, (env, next_seed)) in self
            .envs
            .iter_mut()
            .zip(self.next_seeds.iter_mut())
            .enumerate()
        {
            let (obs, info) = Self::reset_to_hero_turn_on_env(
                roster,
                env,
                next_seed,
                &player_configs[index],
                opponent_policy,
                seed_stride,
            )?;
            results.push((obs, info));
        }
        Ok(results)
    }

    pub fn step(&mut self, actions: &[i64]) -> Result<Vec<StepResult>, AgentError> {
        self.validate_actions_len(actions.len())?;
        self.validate_ready_for_step()?;

        let player_configs = &self.player_configs;
        let opponent_policy = self.opponent_policy;
        let seed_stride = self.seed_stride;
        let roster = &self.uniform_roster;

        let mut results = Vec::with_capacity(self.envs.len());
        for (index, ((env, next_seed), action)) in self
            .envs
            .iter_mut()
            .zip(self.next_seeds.iter_mut())
            .zip(actions.iter())
            .enumerate()
        {
            let out = Self::step_with_autoreset_on_env(
                roster,
                env,
                next_seed,
                *action,
                &player_configs[index],
                opponent_policy,
                seed_stride,
            )?;
            results.push(out);
        }

        Ok(results)
    }

    pub fn reset_all_into<F>(
        &mut self,
        player_configs: Vec<PlayerConfig>,
        write: F,
    ) -> Result<Vec<InfoDict>, AgentError>
    where
        F: Fn(usize, &Observation, f64, bool, bool) -> Result<(), AgentError>,
    {
        self.reset_each_into(vec![player_configs; self.envs.len()], write)
    }

    /// Retain one exact setup per stream, including through terminal auto-reset.
    pub fn reset_each_into<F>(
        &mut self,
        player_configs: Vec<Vec<PlayerConfig>>,
        write: F,
    ) -> Result<Vec<InfoDict>, AgentError>
    where
        F: Fn(usize, &Observation, f64, bool, bool) -> Result<(), AgentError>,
    {
        self.validate_actions_len(player_configs.len())?;
        for configs in &player_configs {
            Self::validate_player_configs(configs.len())?;
        }
        self.player_configs = player_configs;

        let opponent_policy = self.opponent_policy;
        let seed_stride = self.seed_stride;

        self.for_each_env(|env_index, env, next_seed, configs, roster| {
            let (obs, info) = Self::reset_to_hero_turn_on_env(
                roster,
                env,
                next_seed,
                configs,
                opponent_policy,
                seed_stride,
            )?;
            write(env_index, &obs, 0.0, false, false)?;
            Ok(info)
        })
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
        self.validate_actions_len(actions.len())?;
        self.validate_ready_for_step()?;

        let opponent_policy = self.opponent_policy;
        let seed_stride = self.seed_stride;

        if let Some(mask) = active {
            self.validate_actions_len(mask.len())?;
        }
        self.for_each_env(|env_index, env, next_seed, configs, roster| {
            if active.is_some_and(|mask| !mask[env_index]) {
                return Ok(InfoDict::new());
            }
            let action = actions[env_index];
            let result = Self::step_with_autoreset_on_env(
                roster,
                env,
                next_seed,
                action,
                configs,
                opponent_policy,
                seed_stride,
            )?;
            write(
                env_index,
                &result.obs,
                result.reward,
                result.terminated,
                result.truncated,
            )?;
            Ok(result.info)
        })
    }

    /// Settle each active gameplay call independently. Failed games are poisoned:
    /// callers must retain their inputs and replace them before stepping again.
    /// Encoding and reset failures remain fatal, rather than looking like games.
    pub fn step_isolated_into<F>(
        &mut self,
        actions: &[i64],
        active: &[bool],
        write: F,
    ) -> Result<Vec<InfoDict>, AgentError>
    where
        F: Fn(usize, &Observation, f64, bool, bool) -> Result<(), AgentError>,
    {
        self.validate_actions_len(actions.len())?;
        self.validate_actions_len(active.len())?;
        self.validate_ready_for_step()?;
        let policy = self.opponent_policy;
        let stride = self.seed_stride;
        self.for_each_env(|index, env, next_seed, configs, roster| {
            if !active[index] {
                return Ok(InfoDict::new());
            }
            let frame = serde_json::to_string(&env.semantic_decision_frame()?)
                .map_err(|error| AgentError(error.to_string()))?;
            // A Rust unwind is contained only around gameplay of a disposable
            // stream. Buffer errors and invalid setup must never be tolerated.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                Self::step_one_on_env(env, actions[index], policy)
            }));
            let mut out = match result {
                Ok(Ok(out)) => out,
                failure => {
                    let (class, message) = match failure {
                        Ok(Err(error)) => ("AgentError", error.to_string()),
                        Err(payload) => (
                            "NativePanic",
                            payload
                                .downcast_ref::<String>()
                                .cloned()
                                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                                .unwrap_or_else(|| "native gameplay panic".into()),
                        ),
                        Ok(Ok(_)) => unreachable!(),
                    };
                    return Ok(InfoDict::from([
                        ("game_error".into(), InfoValue::String(message)),
                        ("game_error_class".into(), InfoValue::String(class.into())),
                        ("game_error_frame".into(), InfoValue::String(frame)),
                        (
                            "game_error_trace".into(),
                            InfoValue::String(
                                std::backtrace::Backtrace::force_capture().to_string(),
                            ),
                        ),
                    ]));
                }
            };
            if out.terminated || out.truncated {
                let (obs, _) = Self::reset_to_hero_turn_on_env(
                    roster, env, next_seed, configs, policy, stride,
                )?;
                out.obs = obs;
            }
            write(index, &out.obs, out.reward, out.terminated, out.truncated)?;
            Ok(out.info)
        })
    }

    /// Replace quarantined games under the unchanged native seed succession.
    pub fn replace_failed_into<F>(
        &mut self,
        failed: &[bool],
        write: F,
    ) -> Result<Vec<InfoDict>, AgentError>
    where
        F: Fn(usize, &Observation, f64, bool, bool) -> Result<(), AgentError>,
    {
        self.validate_actions_len(failed.len())?;
        self.validate_ready_for_step()?;
        let policy = self.opponent_policy;
        let stride = self.seed_stride;
        let skip_trivial = self.skip_trivial;
        self.for_each_env(|index, env, next_seed, configs, roster| {
            if !failed[index] {
                return Ok(InfoDict::new());
            }
            *env = Env::new(*next_seed, skip_trivial, false, false);
            let (obs, info) =
                Self::reset_to_hero_turn_on_env(roster, env, next_seed, configs, policy, stride)?;
            write(index, &obs, 0.0, false, false)?;
            Ok(info)
        })
    }

    fn for_each_env<R, F>(&mut self, work: F) -> Result<Vec<R>, AgentError>
    where
        F: Fn(usize, &mut Env, &mut u64, &[PlayerConfig], &RootRoster) -> Result<R, AgentError>,
    {
        // Setups are immutable between resets. Borrow each stream's setup while
        // mutating its environment/seed, including when most streams are paused.
        // Cloning the complete setup list here would repeat deck allocations on
        // every micro-step even for streams excluded by the active mask.
        let configs = &self.player_configs;
        self.envs
            .iter_mut()
            .zip(self.next_seeds.iter_mut())
            .enumerate()
            .map(|(env_index, (env, next_seed))| {
                work(
                    env_index,
                    env,
                    next_seed,
                    &configs[env_index],
                    &self.uniform_roster,
                )
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

    fn step_with_autoreset_on_env(
        roster: &RootRoster,
        env: &mut Env,
        next_seed: &mut u64,
        action: i64,
        player_configs: &[PlayerConfig],
        opponent_policy: OpponentPolicy,
        seed_stride: u64,
    ) -> Result<StepResult, AgentError> {
        let mut out = Self::step_one_on_env(env, action, opponent_policy)?;

        if out.terminated || out.truncated {
            let terminal_reward = out.reward;
            let terminal_terminated = out.terminated;
            let terminal_truncated = out.truncated;
            let terminal_info = out.info;

            let (reset_obs, _) = Self::reset_to_hero_turn_on_env(
                roster,
                env,
                next_seed,
                player_configs,
                opponent_policy,
                seed_stride,
            )?;

            out = StepResult {
                obs: reset_obs,
                reward: terminal_reward,
                terminated: terminal_terminated,
                truncated: terminal_truncated,
                info: terminal_info,
            };
        }

        Ok(out)
    }

    fn step_one_on_env(
        env: &mut Env,
        action: i64,
        opponent_policy: OpponentPolicy,
    ) -> Result<StepResult, AgentError> {
        let (mut obs, mut reward, mut terminated, mut truncated, mut info) =
            env.step(action).map_err(|error| {
                AgentError(format!(
                    "{}; native_failure={}",
                    error.0,
                    env.vector_failure_context(action)
                ))
            })?;

        if terminated || truncated || opponent_policy == OpponentPolicy::None {
            return Ok(StepResult {
                obs,
                reward,
                terminated,
                truncated,
                info,
            });
        }

        while Self::is_opponent_turn(&obs) {
            let opponent_action = opponent_policy.select_action(env)?;
            let (next_obs, opponent_reward, opp_terminated, opp_truncated, opp_info) =
                env.step(opponent_action).map_err(|error| {
                    AgentError(format!(
                        "{}; native_failure={}",
                        error.0,
                        env.vector_failure_context(opponent_action)
                    ))
                })?;
            obs = next_obs;
            info = opp_info;
            terminated = opp_terminated;
            truncated = opp_truncated;

            if terminated || truncated {
                // Zero-sum: hero reward = negated opponent terminal reward.
                reward = -opponent_reward;
                break;
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

    fn skip_opponent_turns_until_hero_on_env(
        roster: &RootRoster,
        env: &mut Env,
        mut obs: Observation,
        next_seed: &mut u64,
        player_configs: &[PlayerConfig],
        opponent_policy: OpponentPolicy,
        seed_stride: u64,
    ) -> Result<Observation, AgentError> {
        const MAX_OPPONENT_STEPS: usize = 10_000;
        let mut steps = 0;
        while Self::is_opponent_turn(&obs) {
            steps += 1;
            if steps > MAX_OPPONENT_STEPS {
                return Err(AgentError(
                    "exceeded max opponent steps without reaching hero turn".to_string(),
                ));
            }
            let opponent_action = opponent_policy.select_action(env)?;
            let (next_obs, _, terminated, truncated, _) =
                env.step(opponent_action).map_err(|error| {
                    AgentError(format!(
                        "{}; native_failure={}",
                        error.0,
                        env.vector_failure_context(opponent_action)
                    ))
                })?;
            obs = next_obs;
            if terminated || truncated {
                let (reset_obs, _) =
                    Self::reset_env_on_env(roster, env, next_seed, seed_stride, player_configs)?;
                obs = reset_obs;
                steps = 0;
            }
        }
        Ok(obs)
    }

    fn reset_env_on_env(
        roster: &RootRoster,
        env: &mut Env,
        next_seed: &mut u64,
        seed_stride: u64,
        player_configs: &[PlayerConfig],
    ) -> Result<(Observation, InfoDict), AgentError> {
        let seed = *next_seed;
        *next_seed = seed.saturating_add(seed_stride);
        env.set_seed(seed);
        let configs = if roster.is_empty() {
            player_configs.to_vec()
        } else {
            let (a, b) = roster.pair(seed);
            vec![roster.players[a].clone(), roster.players[b].clone()]
        };
        env.reset(configs)
    }

    fn reset_to_hero_turn_on_env(
        roster: &RootRoster,
        env: &mut Env,
        next_seed: &mut u64,
        player_configs: &[PlayerConfig],
        opponent_policy: OpponentPolicy,
        seed_stride: u64,
    ) -> Result<(Observation, InfoDict), AgentError> {
        let (mut obs, info) =
            Self::reset_env_on_env(roster, env, next_seed, seed_stride, player_configs)?;
        if opponent_policy != OpponentPolicy::None {
            obs = Self::skip_opponent_turns_until_hero_on_env(
                roster,
                env,
                obs,
                next_seed,
                player_configs,
                opponent_policy,
                seed_stride,
            )?;
        }
        Ok((obs, info))
    }

    fn is_opponent_turn(obs: &Observation) -> bool {
        obs.agent.player_index != HERO_PLAYER_INDEX
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
        agent::{action::AgentError, env::Env, opponent::OpponentPolicy, vector_env::VectorEnv},
        state::player::PlayerConfig,
    };

    #[test]
    fn training_decks_are_admitted_distinct_replayable_and_replaceable() {
        let deck = |name: &str| {
            let mut player = PlayerConfig::new(name, BTreeMap::from([(name.into(), 12)]));
            player.content_pack = Some("tla-pool-v1".into());
            player
        };
        let decks: Vec<_> = ["Plains", "Mountain", "Forest"].map(deck).into();
        let mut vector = VectorEnv::new(2, 17402, true, OpponentPolicy::None);
        assert!(vector
            .set_training_decks(vec![decks[0].clone(); 2])
            .is_err());
        let mut bad = decks.clone();
        bad[0].decklist = BTreeMap::from([("Black Lotus".into(), 12)]);
        assert!(vector.set_training_decks(bad).is_err());
        vector.set_training_decks(decks.clone()).unwrap();
        vector.reset_all(decks[..2].to_vec()).unwrap();
        let seeds = vector.current_game_seeds();
        let actual = vector.current_player_configs();
        for (index, players) in actual.iter().enumerate() {
            assert_ne!(players[0].decklist, players[1].decklist);
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
        generated.decklist.insert("Rocky Rebuke".into(), 1);
        assert!(vector
            .set_training_decks(vec![decks[0].clone(), generated])
            .is_err());
        vector
            .reset_seeds_into(seeds.clone(), |_, _, _, _, _| Ok(()))
            .unwrap();
        assert_eq!(actual, vector.current_player_configs());
        let replacement = vec![deck("Island"), deck("Swamp")];
        vector.set_training_decks(replacement.clone()).unwrap();
        assert_eq!(actual, vector.current_player_configs());
        vector
            .reset_seeds_into(seeds, |_, _, _, _, _| Ok(()))
            .unwrap();
        for players in vector.current_player_configs() {
            assert!(players.iter().all(|player| replacement.contains(player)));
        }
        assert!(vector.current_matchup_indices().is_empty());
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
        let pack = crate::semantic::SemanticPack::current("optional_payments").unwrap();
        pack.decks
            .iter()
            .map(|d| {
                let mut p = pack.player_config(&d.key, &d.key).unwrap();
                p.content_pack = Some(pack.pack_key.clone());
                p
            })
            .collect()
    }

    #[test]
    fn uniform_roots_cover_all_pairs_and_seats_with_exact_setups() {
        use std::collections::BTreeSet;
        let roster = elemental_roster();
        let mut reached = [BTreeSet::new(), BTreeSet::new()];
        let mut vector = VectorEnv::new(2, 174, true, OpponentPolicy::None);
        let mut bad = roster.clone();
        bad[0].sideboard.clear();
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
            let pairs = vector.current_matchup_indices();
            for index in 0..2 {
                let (mut obs, _, done, trunc, _) = scalar[index].step(0).unwrap();
                if done || trunc {
                    terminals += 1;
                    changed += usize::from(previous[index] != pairs[index]);
                    scalar[index].set_seed(vector.current_game_seeds()[index]);
                    let (a, b) = pairs[index];
                    obs = scalar[index]
                        .reset(vec![roster[a].clone(), roster[b].clone()])
                        .unwrap()
                        .0;
                }
                assert_eq!(format!("{:?}", results[index].obs), format!("{obs:?}"));
            }
        }
        assert!(terminals > 0 && changed > 0);
    }

    #[test]
    fn reset_all_and_step_batch() {
        let mut env = VectorEnv::new(3, 7, true, OpponentPolicy::None);
        let reset = env
            .reset_all(sample_player_configs())
            .expect("reset_all should succeed");
        assert_eq!(reset.len(), 3);

        let stepped = env.step(&[0, 0, 0]).expect("step should succeed");
        assert_eq!(stepped.len(), 3);
    }

    #[test]
    fn isolated_failure_preserves_successful_streams_and_seed_stride() {
        let mut vector = VectorEnv::new(3, 7, true, OpponentPolicy::None);
        vector.reset_all(sample_player_configs()).unwrap();
        let mut control = VectorEnv::new(3, 7, true, OpponentPolicy::None);
        control.reset_all(sample_player_configs()).unwrap();
        let rows = RefCell::new(Vec::new());
        let outcomes = vector
            .step_isolated_into(&[0, -1, 0], &[true; 3], |index, _, _, _, _| {
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
        vector
            .replace_failed_into(&[false, true, false], |_, _, _, _, _| Ok(()))
            .unwrap();
        assert_eq!(vector.current_game_seeds(), vec![7, 11, 9]);
        for index in [0, 2] {
            assert_eq!(
                vector.envs[index].state_digest().unwrap(),
                control.envs[index].state_digest().unwrap()
            );
        }
    }

    #[test]
    fn reset_skips_to_hero_turn_with_passive_policy() {
        let mut env = VectorEnv::new(1, 11, true, OpponentPolicy::Passive);
        let reset = env
            .reset_all(sample_player_configs())
            .expect("reset_all should succeed");
        assert_eq!(reset[0].0.agent.player_index, 0);
    }

    #[test]
    fn terminal_step_returns_post_reset_observation() {
        let mut env = VectorEnv::new(1, 23, true, OpponentPolicy::Passive);
        env.reset_all(sample_player_configs())
            .expect("reset_all should succeed");

        let mut found_terminal = false;
        for _ in 0..4000 {
            let results = env.step(&[0]).expect("step should succeed");
            let result = &results[0];
            assert_eq!(result.obs.agent.player_index, 0);
            if result.terminated {
                found_terminal = true;
                assert!(!result.obs.game_over);
                break;
            }
        }

        assert!(found_terminal, "expected at least one terminal transition");
    }

    #[test]
    fn step_into_and_reset_all_into() {
        let mut env = VectorEnv::new(3, 31, true, OpponentPolicy::Passive);

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
        let mut paused = VectorEnv::new(2, 99, true, OpponentPolicy::Passive);
        let mut control = VectorEnv::new(2, 99, true, OpponentPolicy::Passive);
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
    fn restored_game_seeds_preserve_future_autoreset_stride() {
        let mut vector = VectorEnv::new(2, 1, true, OpponentPolicy::None);
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
                let (mut obs, _, done, truncated, _) = scalar[index].step(0).unwrap();
                if done || truncated {
                    terminals += 1;
                    seeds[index] += 2;
                    scalar[index].set_seed(seeds[index]);
                    obs = scalar[index].reset(sample_player_configs()).unwrap().0;
                }
                assert_eq!(format!("{:?}", actual[index].obs), format!("{obs:?}"));
            }
            assert_eq!(vector.current_game_seeds(), seeds);
        }
        assert!(terminals > 0);
    }

    #[test]
    fn vector_error_retains_exact_scalar_replay_inputs() {
        let configs = sample_player_configs();
        let mut vector = VectorEnv::new(1, 174, true, OpponentPolicy::None);
        vector.reset_all(configs.clone()).unwrap();
        vector.step(&[0]).unwrap();
        let error = vector.step(&[999_999]).unwrap_err();
        let (_, encoded) = error
            .0
            .split_once("; native_failure=")
            .expect("private failure context");
        let context: serde_json::Value = serde_json::from_str(encoded).unwrap();
        assert_eq!(context["world"], crate::WORLD_VERSION);
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
        let mut env = VectorEnv::new(3, 99, true, OpponentPolicy::Passive);
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
    fn masked_mixed_setups_preserve_scalar_autoreset() {
        let configs: Vec<_> = (0..3)
            .map(|index| {
                let mut players = sample_player_configs();
                players[0].name = format!("Hero {index}");
                players[0].decklist.insert("Mountain".into(), 12 + index);
                players[1].sideboard.insert("Gray Ogre".into(), index + 1);
                players
            })
            .collect();
        let mut seeds = [33_u64, 34, 35];
        let mut scalar = seeds.map(|seed| Env::new(seed, true, false, false));
        for (env, setup) in scalar.iter_mut().zip(&configs) {
            env.reset(setup.clone()).unwrap();
        }
        let mut vector = VectorEnv::new(3, 33, true, OpponentPolicy::None);
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
                let (mut obs, expected_reward, expected_done, expected_trunc, _) =
                    scalar[index].step(0).unwrap();
                assert_eq!(
                    (reward, done, trunc),
                    (expected_reward, expected_done, expected_trunc)
                );
                if done || trunc {
                    terminals[index] += 1;
                    seeds[index] += 3;
                    scalar[index].set_seed(seeds[index]);
                    obs = scalar[index].reset(configs[index].clone()).unwrap().0;
                }
                assert_eq!(actual, format!("{obs:?}"));
            }
            assert_eq!(vector.current_game_seeds(), seeds);
        }
        assert!(terminals.iter().all(|count| *count > 0));
        assert_eq!(vector.player_configs, configs);
    }

    #[test]
    fn shards_preserve_global_seeds_through_autoreset_and_restore() {
        let mut whole = VectorEnv::new(3, 19, true, OpponentPolicy::None);
        let mut left = VectorEnv::new(2, 19, true, OpponentPolicy::None)
            .with_seed_stride(3)
            .unwrap();
        let mut right = VectorEnv::new(1, 21, true, OpponentPolicy::None)
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
            let mut seeds = left.current_game_seeds();
            seeds.extend(right.current_game_seeds());
            assert_eq!(whole.current_game_seeds(), seeds);
        }
        assert!(terminals > 0);
        assert!(VectorEnv::new(2, 0, true, OpponentPolicy::None)
            .with_seed_stride(1)
            .is_err());
        assert!(left.with_seed_stride(3).is_err());
    }
}
