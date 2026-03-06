use crate::{
    agent::{action::AgentError, env::Env, observation::Observation, opponent::OpponentPolicy},
    infra::profiler::InfoDict,
    state::player::PlayerConfig,
};
use std::thread;

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
    player_configs: Vec<PlayerConfig>,
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

    pub fn is_empty(&self) -> bool {
        self.envs.is_empty()
    }

    pub fn reset_all(
        &mut self,
        player_configs: Vec<PlayerConfig>,
    ) -> Result<Vec<(Observation, InfoDict)>, AgentError> {
        Self::validate_player_configs(player_configs.len())?;
        self.player_configs = player_configs;
        let player_configs = self.player_configs.clone();
        let opponent_policy = self.opponent_policy;
        let seed_stride = self.seed_stride;

        let mut results = Vec::with_capacity(self.envs.len());
        for (env, next_seed) in self.envs.iter_mut().zip(self.next_seeds.iter_mut()) {
            let (obs, info) = Self::reset_to_hero_turn_on_env(
                env,
                next_seed,
                &player_configs,
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

        let player_configs = self.player_configs.clone();
        let opponent_policy = self.opponent_policy;
        let seed_stride = self.seed_stride;

        let mut results = Vec::with_capacity(self.envs.len());
        for ((env, next_seed), action) in self
            .envs
            .iter_mut()
            .zip(self.next_seeds.iter_mut())
            .zip(actions.iter())
        {
            let out = Self::step_with_autoreset_on_env(
                env,
                next_seed,
                *action,
                &player_configs,
                opponent_policy,
                seed_stride,
            )?;
            results.push(out);
        }

        Ok(results)
    }

    pub fn par_reset_all_into<F>(
        &mut self,
        num_threads: usize,
        player_configs: Vec<PlayerConfig>,
        write: F,
    ) -> Result<Vec<InfoDict>, AgentError>
    where
        F: Fn(usize, &Observation, f64, bool, bool) -> Result<(), AgentError> + Sync,
    {
        Self::validate_player_configs(player_configs.len())?;
        self.player_configs = player_configs;

        let player_configs = self.player_configs.clone();
        let opponent_policy = self.opponent_policy;
        let seed_stride = self.seed_stride;

        self.parallelize_envs(num_threads, |env_index, env, next_seed| {
            let (obs, info) = Self::reset_to_hero_turn_on_env(
                env,
                next_seed,
                &player_configs,
                opponent_policy,
                seed_stride,
            )?;
            write(env_index, &obs, 0.0, false, false)?;
            Ok(info)
        })
    }

    pub fn par_step_into<F>(
        &mut self,
        num_threads: usize,
        actions: &[i64],
        write: F,
    ) -> Result<Vec<InfoDict>, AgentError>
    where
        F: Fn(usize, &Observation, f64, bool, bool) -> Result<(), AgentError> + Sync,
    {
        self.validate_actions_len(actions.len())?;
        self.validate_ready_for_step()?;

        let player_configs = self.player_configs.clone();
        let opponent_policy = self.opponent_policy;
        let seed_stride = self.seed_stride;

        self.parallelize_envs(num_threads, |env_index, env, next_seed| {
            let action = actions[env_index];
            let result = Self::step_with_autoreset_on_env(
                env,
                next_seed,
                action,
                &player_configs,
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

    fn parallelize_envs<R, F>(&mut self, num_threads: usize, work: F) -> Result<Vec<R>, AgentError>
    where
        R: Send,
        F: Fn(usize, &mut Env, &mut u64) -> Result<R, AgentError> + Sync,
    {
        let len = self.envs.len();
        if len == 0 {
            return Ok(Vec::new());
        }

        let threads = num_threads.max(1).min(len);
        let chunk_size = len.div_ceil(threads);
        let mut ordered_results: Vec<Result<R, AgentError>> = Vec::with_capacity(len);

        thread::scope(|scope| -> Result<(), AgentError> {
            let mut handles = Vec::new();

            for (chunk_index, (env_chunk, seed_chunk)) in self
                .envs
                .chunks_mut(chunk_size)
                .zip(self.next_seeds.chunks_mut(chunk_size))
                .enumerate()
            {
                let start = chunk_index * chunk_size;
                let work_ref = &work;

                handles.push(scope.spawn(move || {
                    let mut local = Vec::with_capacity(env_chunk.len());
                    for (offset, env) in env_chunk.iter_mut().enumerate() {
                        let env_index = start + offset;
                        let next_seed = &mut seed_chunk[offset];
                        local.push(work_ref(env_index, env, next_seed));
                    }
                    local
                }));
            }

            for handle in handles {
                let local = handle.join().map_err(|_| {
                    AgentError("parallel worker panicked while stepping envs".to_string())
                })?;
                ordered_results.extend(local);
            }

            Ok(())
        })?;

        ordered_results.into_iter().collect()
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
        if self.player_configs.len() != 2 {
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
    use std::collections::BTreeMap;

    use crate::{
        agent::{action::AgentError, opponent::OpponentPolicy, vector_env::VectorEnv},
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
                    ("Grey Ogre".to_string(), 18usize),
                ]),
            ),
            PlayerConfig::new(
                "Villain",
                BTreeMap::from([
                    ("Mountain".to_string(), 12usize),
                    ("Forest".to_string(), 12usize),
                    ("Llanowar Elves".to_string(), 18usize),
                    ("Grey Ogre".to_string(), 18usize),
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
    fn parallel_entrypoints_step_and_reset() {
        let mut env = VectorEnv::new(3, 31, true, OpponentPolicy::Passive);

        let reset_infos = env
            .par_reset_all_into(
                2,
                sample_player_configs(),
                |_, obs, reward, terminated, truncated| {
                    assert_eq!(obs.agent.player_index, 0);
                    assert_eq!(reward, 0.0);
                    assert!(!terminated);
                    assert!(!truncated);
                    Ok(())
                },
            )
            .expect("parallel reset should succeed");
        assert_eq!(reset_infos.len(), 3);

        let step_infos = env
            .par_step_into(2, &[0, 0, 0], |_, _, _, _, _| Ok(()))
            .expect("parallel step should succeed");
        assert_eq!(step_infos.len(), 3);
    }

    #[test]
    fn parallel_errors_are_returned_in_index_order() {
        let mut env = VectorEnv::new(3, 99, true, OpponentPolicy::Passive);
        env.par_reset_all_into(2, sample_player_configs(), |_, _, _, _, _| Ok(()))
            .expect("parallel reset should succeed");

        let err = env
            .par_step_into(2, &[0, 0, 0], |env_index, _, _, _, _| {
                if env_index >= 1 {
                    return Err(AgentError(format!("env error {env_index}")));
                }
                Ok(())
            })
            .expect_err("parallel step should fail");
        assert_eq!(err.0, "env error 1");
    }
}
