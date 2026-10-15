use crate::{
    agent::{action::AgentError, env::Env, observation::Observation, opponent::OpponentPolicy},
    infra::profiler::InfoDict,
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

#[derive(Debug)]
pub struct VectorEnv {
    envs: Vec<Env>,
    player_configs: Vec<Vec<PlayerConfig>>,
    opponent_policy: OpponentPolicy,
    next_seeds: Vec<u64>,
    seed_stride: u64,
}

impl VectorEnv {
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
        }
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

        let mut results = Vec::with_capacity(self.envs.len());
        for (index, (env, next_seed)) in self
            .envs
            .iter_mut()
            .zip(self.next_seeds.iter_mut())
            .enumerate()
        {
            let (obs, info) = Self::reset_to_hero_turn_on_env(
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

        let mut results = Vec::with_capacity(self.envs.len());
        for (index, ((env, next_seed), action)) in self
            .envs
            .iter_mut()
            .zip(self.next_seeds.iter_mut())
            .zip(actions.iter())
            .enumerate()
        {
            let out = Self::step_with_autoreset_on_env(
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

        self.for_each_env(|env_index, env, next_seed, configs| {
            let (obs, info) = Self::reset_to_hero_turn_on_env(
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
        self.for_each_env(|env_index, env, next_seed, configs| {
            if active.is_some_and(|mask| !mask[env_index]) {
                return Ok(InfoDict::new());
            }
            let action = actions[env_index];
            let result = Self::step_with_autoreset_on_env(
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

    fn for_each_env<R, F>(&mut self, work: F) -> Result<Vec<R>, AgentError>
    where
        F: Fn(usize, &mut Env, &mut u64, &[PlayerConfig]) -> Result<R, AgentError>,
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
                work(env_index, env, next_seed, &configs[env_index])
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
        let (mut obs, mut reward, mut terminated, mut truncated, mut info) = env.step(action)?;

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
                env.step(opponent_action)?;
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
            let (next_obs, _, terminated, truncated, _) = env.step(opponent_action)?;
            obs = next_obs;
            if terminated || truncated {
                let (reset_obs, _) =
                    Self::reset_env_on_env(env, next_seed, seed_stride, player_configs)?;
                obs = reset_obs;
                steps = 0;
            }
        }
        Ok(obs)
    }

    fn reset_env_on_env(
        env: &mut Env,
        next_seed: &mut u64,
        seed_stride: u64,
        player_configs: &[PlayerConfig],
    ) -> Result<(Observation, InfoDict), AgentError> {
        let seed = *next_seed;
        *next_seed = seed.saturating_add(seed_stride);
        env.set_seed(seed);
        env.reset(player_configs.to_vec())
    }

    fn reset_to_hero_turn_on_env(
        env: &mut Env,
        next_seed: &mut u64,
        player_configs: &[PlayerConfig],
        opponent_policy: OpponentPolicy,
        seed_stride: u64,
    ) -> Result<(Observation, InfoDict), AgentError> {
        let (mut obs, info) = Self::reset_env_on_env(env, next_seed, seed_stride, player_configs)?;
        if opponent_policy != OpponentPolicy::None {
            obs = Self::skip_opponent_turns_until_hero_on_env(
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

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, collections::BTreeMap};

    use crate::{
        agent::{action::AgentError, env::Env, opponent::OpponentPolicy, vector_env::VectorEnv},
        state::player::PlayerConfig,
    };

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
