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
        if player_configs.len() != 2 {
            return Err(AgentError(format!(
                "expected exactly 2 player configs, got {}",
                player_configs.len()
            )));
        }

        self.player_configs = player_configs;
        let mut results = Vec::with_capacity(self.envs.len());
        for env_index in 0..self.envs.len() {
            let (obs, info) = self.reset_to_hero_turn(env_index)?;
            results.push((obs, info));
        }
        Ok(results)
    }

    pub fn step(&mut self, actions: &[i64]) -> Result<Vec<StepResult>, AgentError> {
        if actions.len() != self.envs.len() {
            return Err(AgentError(format!(
                "expected {} actions, got {}",
                self.envs.len(),
                actions.len()
            )));
        }
        if self.player_configs.len() != 2 {
            return Err(AgentError(
                "vector env must be reset before step".to_string(),
            ));
        }

        let mut results = Vec::with_capacity(self.envs.len());
        for (env_index, action) in actions.iter().enumerate() {
            let mut out = self.step_one(env_index, *action)?;

            if out.terminated || out.truncated {
                let terminal_reward = out.reward;
                let terminal_terminated = out.terminated;
                let terminal_truncated = out.truncated;
                let terminal_info = out.info;

                let (reset_obs, _) = self.reset_to_hero_turn(env_index)?;

                out = StepResult {
                    obs: reset_obs,
                    reward: terminal_reward,
                    terminated: terminal_terminated,
                    truncated: terminal_truncated,
                    info: terminal_info,
                };
            }

            results.push(out);
        }

        Ok(results)
    }

    fn step_one(&mut self, env_index: usize, action: i64) -> Result<StepResult, AgentError> {
        let (mut obs, mut reward, mut terminated, mut truncated, mut info) =
            self.envs[env_index].step(action)?;

        if terminated || truncated || self.opponent_policy == OpponentPolicy::None {
            return Ok(StepResult {
                obs,
                reward,
                terminated,
                truncated,
                info,
            });
        }

        while self.is_opponent_turn(&obs) {
            let opponent_action = self
                .opponent_policy
                .select_action(&mut self.envs[env_index])?;
            let (next_obs, opponent_reward, opp_terminated, opp_truncated, opp_info) =
                self.envs[env_index].step(opponent_action)?;
            obs = next_obs;
            info = opp_info;
            terminated = opp_terminated;
            truncated = opp_truncated;

            if terminated || truncated {
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

    fn skip_opponent_turns_until_hero(
        &mut self,
        env_index: usize,
        mut obs: Observation,
    ) -> Result<Observation, AgentError> {
        while self.is_opponent_turn(&obs) {
            let opponent_action = self
                .opponent_policy
                .select_action(&mut self.envs[env_index])?;
            let (next_obs, _, terminated, truncated, _) =
                self.envs[env_index].step(opponent_action)?;
            obs = next_obs;
            if terminated || truncated {
                let (reset_obs, _) = self.reset_env(env_index)?;
                obs = reset_obs;
            }
        }
        Ok(obs)
    }

    fn reset_env(&mut self, env_index: usize) -> Result<(Observation, InfoDict), AgentError> {
        let seed = self
            .next_seeds
            .get(env_index)
            .copied()
            .ok_or_else(|| AgentError(format!("invalid env index: {env_index}")))?;
        self.next_seeds[env_index] = seed.saturating_add(self.seed_stride);
        let env = &mut self.envs[env_index];
        env.set_seed(seed);
        env.reset(self.player_configs.clone())
    }

    fn reset_to_hero_turn(
        &mut self,
        env_index: usize,
    ) -> Result<(Observation, InfoDict), AgentError> {
        let (mut obs, info) = self.reset_env(env_index)?;
        if self.opponent_policy != OpponentPolicy::None {
            obs = self.skip_opponent_turns_until_hero(env_index, obs)?;
        }
        Ok((obs, info))
    }

    fn is_opponent_turn(&self, obs: &Observation) -> bool {
        obs.agent.player_index != HERO_PLAYER_INDEX
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::{
        agent::{opponent::OpponentPolicy, vector_env::VectorEnv},
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
}
